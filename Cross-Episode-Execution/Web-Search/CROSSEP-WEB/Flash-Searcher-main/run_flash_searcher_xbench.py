#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import json
import logging
import csv
import base64
import sys
import re
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid
from FlashOAgents import build_model, ArkBatchServerModel
from base_agent import SearchAgent
from utils import read_jsonl, write_jsonl
from EvolveLab.memory_types import MemoryType, TrajectoryData, PROVIDER_MAPPING
from EvolveLab.config import get_memory_config
from eval_utils import (
    TaskTimer, TokenCounter, save_task_result,
    generate_unified_report, enrich_result_with_metrics, create_run_directory
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# # Add xbench-evals path for imports
# xbench_path = os.path.join(os.path.dirname(__file__), 'xbench-evals-main')
# print()
# if os.path.exists(xbench_path):
#     sys.path.insert(0, xbench_path)
# else:
#     logger.error(f"xbench-evals-main directory not found at {xbench_path}")
#     raise FileNotFoundError(f"xbench-evals-main directory not found at {xbench_path}")

load_dotenv(override=True)


def xor_decrypt(data, key):
    """XOR decrypt data with a key"""
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def parse_task_indices(indices_str):
    """Parse index string like "5", "1-10" or "1,3,5-8,10" into a 1-based index set."""
    if not indices_str:
        return None
    
    indices = set()
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                indices.update(range(start, end + 1))
            except ValueError as e:
                logger.error(f"Invalid range format: {part}. Error: {e}")
                raise
        else:
            try:
                indices.add(int(part))
            except ValueError:
                logger.error(f"Invalid number format: {part}")
                raise
    
    return indices


LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


def parse_match_result(match):
    if match is None:
        return match

    match = match.group(0)

    try:
        target = match.split(':')[1].strip()
        return target
    except Exception:
        return match


def grade_question(question_text, correct_answer, llm_response, judge_model):
    if llm_response is None:
        return 0, "", ""

    # If there's direct match, do not need LLM judge
    simple_match = re.search(r'最终答案:*(.*)', llm_response)
    simple_match = parse_match_result(simple_match)
    if simple_match == correct_answer:
        return 1, simple_match, "答案完全正确, 无需调用LLM Judge"

    # Otherwise, use LLM Judge
    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    try:
        judge_message = judge_model([
            {"role": "user", "content": judge_prompt}
        ])
        judge_response = judge_message.content if judge_message else None
    except Exception as e:
        logger.warning(f"Judge model call failed: {e}")
        return 0, "", "Judge Response error"

    if not isinstance(judge_response, str):
        logger.warning("Judge response is not a string; returning error")
        return 0, "", "Judge Response error"

    # Extract grader conclusions
    extract_match = re.search(r'最终答案:*(.*)', judge_response)
    extract_match = parse_match_result(extract_match)

    correct_match = re.search(r"结论:*.(正确|错误)", judge_response)
    correct_match = parse_match_result(correct_match)

    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match = parse_match_result(explain_match)

    score = 1 if (correct_match == "正确") else 0

    return score, extract_match, explain_match


def load_memory_provider(memory_type_str, model=None, config_overrides=None):
    """Load and initialize memory provider from type string"""
    if not memory_type_str:
        return None

    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None

    if memory_type not in PROVIDER_MAPPING:
        logger.error(f"Memory type {memory_type_str} not found in PROVIDER_MAPPING")
        return None

    try:
        class_name, module_name = PROVIDER_MAPPING[memory_type]
        module = __import__(f"EvolveLab.providers.{module_name}", fromlist=[class_name])
        provider_class = getattr(module, class_name)
        config = get_memory_config(memory_type)
        if config_overrides:
            config.update(config_overrides)
        if model is not None:
            try:
                # many providers expect a 'model' in config
                config["model"] = model
            except Exception:
                pass
        provider = provider_class(config=config)
        
        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None
        
        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item(item, model_config, summary_interval, prompts_type, max_steps,
                 memory_type_str=None, enable_memory_evolution=True, judge_model=None,
                 shared_memory_provider=None, memory_lock=None):
    """Process a single XBench task with timing and metrics tracking"""
    if memory_type_str:
        task_model = build_model(memory_type_str, **model_config)
    else:
        task_model = ArkBatchServerModel(**model_config)
    task_model.reset_total_counts()

    if shared_memory_provider is not None:
        memory_provider = shared_memory_provider
        _memory_lock = memory_lock
    elif memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)
        _memory_lock = None
    else:
        memory_provider = None
        _memory_lock = None
    
    timer = TaskTimer()
    timer.start()
    
    search_agent = SearchAgent(
        task_model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps,
        memory_provider=memory_provider
    )

    question = item["prompt"]
    golden_answer = item["answer"]
    task_id = item.get("id")
    
    try:
        result = search_agent(question)
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        is_correct = False
        score = 0
        extracted_answer = ""
        grader_explanation = ""
        if judge_model:
            try:
                agent_result = result.get("agent_result", "")
                if agent_result is None:
                    agent_response = ""
                elif isinstance(agent_result, dict):
                    agent_response = agent_result.get("answer", "")
                    if not agent_response:
                        agent_response = json.dumps(agent_result, ensure_ascii=False)
                else:
                    agent_response = str(agent_result) if agent_result else ""
                
                if not isinstance(agent_response, str):
                    agent_response = str(agent_response) if agent_response else ""
                
                score, extracted_answer, grader_explanation = grade_question(
                    question,
                    golden_answer,
                    agent_response,
                    judge_model
                )
                is_correct = (score == 1)
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
                score = 0
                extracted_answer = ""
                grader_explanation = f"Error: {str(e)}"
        
        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "full_query": question,
                    }
                )
                if _memory_lock:
                    with _memory_lock:
                        success, msg = memory_provider.take_in_memory(trajectory_data)
                else:
                    success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.debug(f"Memory ingested: {msg}")
                else:
                    logger.warning(f"Memory ingestion failed: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed: {e}")
        
        token_counter = TokenCounter.from_model(task_model)
        
        task_result = {
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "success",
            "agent_messages": agent_messages,
            "score": score,
            "extracted_answer": extracted_answer,
            "grader_explanation": grader_explanation,
            **result,
        }
        
        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception occurred while processing question: {question[:100]}...\nError: {error_msg}")
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=None,
                    metadata={
                        "task_id": task_id,
                        "status": "error",
                        "is_correct": False,
                        "full_query": question,
                    }
                )
                if _memory_lock:
                    with _memory_lock:
                        success, msg = memory_provider.take_in_memory(trajectory_data)
                else:
                    success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.debug(f"Memory ingested (error case): {msg}")
                else:
                    logger.warning(f"Memory ingestion failed (error case): {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed (error case): {e}")
        
        task_result = {
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "agent_result": None,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "score": 0,
            "extracted_answer": "",
            "grader_explanation": f"Error: {str(e)}",
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)


def run_xbench_tasks(data_to_run, model_config, judge_model, args, run_dir,
                     shared_memory_provider=None, memory_lock=None):
    """Run XBench tasks and return list of results. Importable by run_pipeline.py."""
    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            result_to_save = {k: v for k, v in result.items()
                              if k not in ["agent_messages", "grader_explanation"]}
            ordered_result = {}
            if "score" in result_to_save:
                ordered_result["score"] = result_to_save["score"]
            if "extracted_answer" in result_to_save:
                ordered_result["extracted_answer"] = result_to_save["extracted_answer"]
            for k, v in result_to_save.items():
                if k not in ["score", "extracted_answer"]:
                    ordered_result[k] = v

            skip_summary = getattr(args, "skip_summary", False)
            outfile = getattr(args, "outfile", None)
            if not skip_summary and outfile:
                write_jsonl(outfile, [ordered_result], "a")

            tid = ordered_result.get("task_id") or uuid.uuid4().hex
            filename = f"{tid}.json"
            save_task_result(ordered_result, run_dir, filename)

    effective_concurrency = getattr(args, "concurrency", 1)
    if shared_memory_provider is not None:
        logger.info(f"Using shared memory provider across {len(data_to_run)} tasks "
                    f"({effective_concurrency} workers)")
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        summary_interval = random.randint(
            getattr(args, "summary_interval", 8) - 1,
            getattr(args, "summary_interval", 8) + 1,
        )
        futures = [
            executor.submit(
                process_item,
                item,
                model_config,
                summary_interval,
                getattr(args, "prompts_type", "default"),
                getattr(args, "max_steps", 40),
                getattr(args, "memory_provider", None),  # needed for ArkModel selection
                getattr(args, "enable_memory_evolution", True),
                judge_model,
                shared_memory_provider,
                memory_lock,
            ) for item in data_to_run
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing XBench"):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    safe_write(result)
                    metrics = result.get("metrics", {})
                    if result.get("status") == "success":
                        logger.info(
                            f"Task done [{len(results)}/{len(futures)}]: "
                            f"{result['question'][:80]}... "
                            f"| Time: {metrics.get('elapsed_time', 0):.1f}s "
                            f"| Tokens: {metrics.get('total_tokens', 0)} "
                            f"| Score: {result.get('score', 'N/A')}"
                        )
                    elif result.get("status") == "error":
                        logger.warning(
                            f"Task error [{len(results)}/{len(futures)}]: "
                            f"{result['question'][:80]}... "
                            f"| Error: {result.get('error', 'Unknown')}"
                        )
            except Exception:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")

    return results


def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }
    
    judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_api_base = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if args.memory_provider:
        judge_model = build_model(
            args.memory_provider,
            model_id=args.judge_model,
            custom_role_conversions=custom_role_conversions,
            api_key=judge_api_key,
            api_base=judge_api_base,
            max_completion_tokens=4096,
        )
    else:
        judge_model = ArkBatchServerModel(
            model_id=args.judge_model,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=4096,
        )

    data = []
    with open(args.infile, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for question in reader:
            key = question["canary"]
            question["prompt"] = xor_decrypt(base64.b64decode(question["prompt"]), key).decode('utf-8')
            question["answer"] = xor_decrypt(base64.b64decode(question["answer"]), key).decode('utf-8')
            data.append(question)

    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            data = [data[i-1] for i in sorted(selected_indices) if 0 < i <= len(data)]
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            return
    elif args.sample_num is not None:
        data = data[:args.sample_num]
    
    data_to_run = data
    logger.info(f"Total data to process: {len(data_to_run)}")

    shared_memory_provider = None
    memory_lock = None
    memory_name = ""
    if args.memory_provider:
        memory_model = build_model(args.memory_provider, **model_config)
        shared_memory_provider = load_memory_provider(args.memory_provider, memory_model)
        if shared_memory_provider:
            memory_name = shared_memory_provider.get_memory_type().value + "_"
            memory_lock = threading.Lock()
            logger.info(f"Shared memory provider created: {args.memory_provider}")
        else:
            logger.warning("Failed to create shared memory provider; tasks will run without memory")
            try:
                memory_name = MemoryType(args.memory_provider).value + "_"
            except Exception:
                pass
    
    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Using direct output directory: {run_dir}")
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, memory_name)
        logger.info(f"Run directory created: {run_dir}")

    results = run_xbench_tasks(
        data_to_run, model_config, judge_model, args, run_dir,
        shared_memory_provider=shared_memory_provider,
        memory_lock=memory_lock,
    )

    logger.info(f"Processing completed. Completed this run: {len(results)}")

    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        results,
        report_path,
        dataset_name="XBench",
        has_levels=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XBench text search (SearchAgent: web + crawl)')

    parser.add_argument('--infile', type=str, default="./data/xbench/DeepSearch.csv", help='input path')
    parser.add_argument('--outfile', type=str, default="./xbench_output/lightweight_results.jsonl", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run, supports: single number (e.g., "5"), range (e.g., "23-165"), or mixed (e.g., "1,3,5-10,20")')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrency (default=1 to avoid memory provider concurrency issues)')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--memory_provider', type=str, default=None, help='Memory provider type (e.g., "agent_kb", "skillweaver") or None')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    parser.add_argument('--skip_summary', action='store_true', help='Only save per-task json files, skip appending to summary outfile')
    parser.add_argument('--judge_model', type=str, default=os.getenv('DEFAULT_JUDGE_MODEL', 'gemini-2.5-flash'), help='Judge model id')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')

    args = parser.parse_args()
    
    main(args)
    