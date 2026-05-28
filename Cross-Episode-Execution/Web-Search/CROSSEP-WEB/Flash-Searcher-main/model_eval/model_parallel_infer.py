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

import json
import re
from .infer_tools import search_tool, crawl_tool
import time
from tqdm import tqdm
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import logging
import os
from utils import read_jsonl, write_jsonl, openai_service


FINAL_PROMPT = '''
An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:
'''
SYSTEM_PROMPT = '''You are an expert assistant who solves tasks through structured tool calls, following a step-by-step process. Each step (action) involves analyzing needs, selecting tools, and executing calls to achieve the task goal.
Each action you take should include a reasoning process and tool calls. After executing the tools, you will receive the results of tool calls, which can be used as input for subsequent actions. This Action/Observation cycle may repeat as needed.

# Task Instructions:
### 1. Parse the plan or summary:
To address the problem of understanding parallel execution requirements, follow these steps centered on parsing <plan></plan> or <summary></summary>:
**CRITICAL: All goals MUST be advanced simultaneously in parallel. Each goal's paths MUST be executed sequentially (one path at a time per goal).**
### 2. Execute parallel tool calls:  
For each goal in the plan, execute the specified tools in parallel according to the paths defined.  
**MANDATORY: Advance ALL goals concurrently. Within each goal, execute paths sequentially (never parallelize paths within a single goal).**
### 3. Handle path diversity:  
For each goal, if multiple paths are provided, execute them sequentially as fallback options if the primary path fails.  
**ABSOLUTE REQUIREMENT: NEVER prematurely assume a goal is achieved. Continue advancing ALL other goals in parallel while handling fallback paths for any individual goal.**
### 4. Process results:  
Synthesize information from all tool outputs to generate comprehensive responses that address all goals.  
**ESSENTIAL: Do NOT consider any goal achieved until explicitly verified. Maintain parallel advancement of ALL goals throughout synthesis.**
### 5. Final answer:  
Once all goals are addressed, consolidate their results, and ensure that the consolidated outcome can accurately and correctly answer the original task, then call the 'final_answer' tool with such consolidated results.
**FINAL CONDITION: Only proceed when ALL goals are resolved. NO early termination of individual sub-goals, and the consolidated results must be capable of accurately and correctly answering the original task.**

# Available Tools
You have access to these tools:
- web_search: Perform a web search query and return the search results.
    Takes inputs: {'query': {'type': 'string', 'description': 'The web search query to perform.'}}
    Returns an output of type: string
- crawl_page: Access webpage using the provided URL and extract relevant content.  Please make full use of this tool to verify the accuracy of the searched content.
    Takes inputs: {'url': {'type': 'string', 'description': 'The URL of the webpage to visit.'}, 'query': {'type': 'string', 'description': 'The specific information to extract from the webpage.'}}
    Returns an output of type: string
- final_answer: Gives a clear, accurate final answer to the given task.
    Takes inputs: {'answer': {'type': 'string', 'description': 'The clear, accurate final answer to the task'}}
    Returns an output of type: string

# Rules
Here are the rules you should always follow to solve your task:
1. Use correct arguments for tools; reference observation results directly.
2. Call tools to solve the task. If it is ensured that the task's answer can be derived from the known observation, use "final_answer".
3. Do not repeat tool calls with identical parameters.
4. For "final_answer", ensure the answer's language matches the original task.
5. You can invoke up to 5 tools.

# Answer Format
Each answer has one of 3 functions, with "tools" embedded in reasoning and execution:
- think: Reason about which tools to use, tool call order, and execution paths to reach the goal. Start with <think>, end with </think>.
- plan: Break down the question into detailed, tool-executable sub-questions. Start with <plan>, end with </plan>.
- summary: Analyze if the plan’s sub-goals/paths are completed. Start with <summary>, end with </summary>.

# Examples
1. <think>...</think><plan>..</plan>
2. <think>...</think><tools>..</tools>
3. <think>...</think><summary>..</summary>

Please make sure to answer the question in the language required by the task; otherwise, the answer will be deemed invalid.
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
'''

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
    ]
)
logger = logging.getLogger('my_logger')
load_dotenv(override=True)

def get_search_results_with_format(tool, tool_args):
    if tool == 'web_search':
        web_results = search_tool(query=tool_args["query"])
        return web_results

    elif tool == "crawl_page":
        crawl_results = crawl_tool(
            url=tool_args["url"],
            query=tool_args["query"]
        )
        return crawl_results
    else:
        return "Unsupported tool"
    
def process_single_data(item, args):
    
    query = item.get("question")

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is: {query}.\nNow Begin! Solve the task!"}
    ]
    
    try:
        for attempt in range(args.max_steps):
            tmp_answer = None
            system_msg = [msg for msg in conversation_history if msg["role"] == "system"]
            other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]
            truncated_msgs = other_msgs[-30:]
            truncated_history = system_msg + truncated_msgs

            try:
                final_content = openai_service(
                    messages=truncated_history,
                    api_key=args.vllm_api_key, 
                    base_url=args.vllm_url, 
                    model=args.model_name,
                )
                
                conversation_history.append({
                    "role": "assistant", 
                    "content": final_content
                })
                
                match = re.search(r'<tools>(.*?)</tools>', final_content, re.DOTALL) 
                tool_results = []
                if match:
                    tools_content = match.group(1).strip()
                    try:
                        tools_list = json.loads(tools_content)
                        if isinstance(tools_list, list):
                            final_answer_tool = next(
                                (tool for tool in tools_list 
                                    if isinstance(tool, dict) and tool.get('name') == "final_answer"),
                                None
                            )
                            
                            if final_answer_tool:
                                tmp_answer = final_answer_tool['arguments']
                                print(f"Final answer: {tmp_answer}")
                                break

                            with ThreadPoolExecutor(max_workers=5) as executor:
                                futures = []
                                for idx, tool in enumerate(tools_list):
                                    if isinstance(tool, dict) and "name" in tool and "arguments" in tool:
                                        future = executor.submit(
                                            get_search_results_with_format, 
                                            tool["name"], 
                                            tool["arguments"]
                                        )
                                        futures.append((idx, future, tool))
                                
                                futures.sort(key=lambda x: x[0])
                                
                                for idx, future, tool in futures:
                                    result = future.result()
                                    tool_results.append(
                                        f'''Results for tool call {tool["name"]} with arguments {tool["arguments"]}: {result}'''
                                    )
                            
                        if tool_results:
                            tools_result_str = "\n\n".join(tool_results)
                            if attempt % 8 == 0 and attempt != 0:
                                tools_result_str += "\n\n# Note: Now, you should analyze the task completion status and provide recommendations for next steps."
                            conversation_history.append({
                                "role": "user",
                                "content": tools_result_str
                            })
                            
                    except Exception as parse_err:
                        logger.warning(f"Failed to parse tool call content: {str(parse_err)}")
                        logger.warning(f"Unparseable content: {tools_content}")
                else:
                    conversation_history.append({
                            "role": "user",
                            "content": "Based on the plan/summary and previous conversations, continue solving the task!"
                        })

            except Exception as e:
                logger.warning(f"Error occurred in round {attempt + 1}: {str(e)}")
                if attempt < args.max_steps - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e
        if tmp_answer:
            return {
                "question": item["question"],
                "golden_answer": item["answer"],
                "agent_result": tmp_answer,
                "agent_trajectory": conversation_history
            }
        else:
            conversation=conversation_history.copy()[1:]
            final_task = {
                "role": "user",
                "content": f'''Based on the above agent memory, please provide a brief answer to the following user task. 
                Here is your task:
                {query}
                '''
            }
            all_conversation = conversation + [final_task]
            final_conversation = all_conversation[-30:]

            final_content = openai_service(
                    messages=final_conversation,
                    api_key=os.getenv("OPENAI_BASE_URL"), 
                    key=os.getenv("OPENAI_API_KEY"), 
                    model=os.getenv("SUMMARY_MODEL"),
                )
            return {
                "question": item["question"],
                "golden_answer": item["answer"],
                "agent_result": final_content,
                "agent_trajectory": conversation_history
            }
        
    except Exception as e:
        logger.error(f"Error occurred while processing data: {str(e)}")


def main(args):
    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    if args.sample_num is not None:
        data = data[:args.sample_num]
    try:
        out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []
    done_questions = set([item.get("question") for item in out_data])
    data_to_run = [item for item in data if item.get("question") not in done_questions]
    logger.info(f"Total data: {len(data)}, Completed: {len(done_questions)}, Remaining to run: {len(data_to_run)}")
    
    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            write_jsonl(args.outfile, [result], "a")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:

        futures = [
            executor.submit(process_single_data, item, args) for item in data_to_run
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
            safe_write(result)

    logger.info(f"Processing complete. Newly added: {len(results)}, Total completed: {len(done_questions) + len(results)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data generation')

    parser.add_argument('--infile', type=str, default="./data/<example.json>", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='sample num')
    parser.add_argument('--concurrency', type=int, default=15, help='num of concurrency')
    parser.add_argument('--model_name', type=str, required=True, help='vllm model name')
    parser.add_argument('--max_steps', type=int, default=40, help='max steps')
    parser.add_argument('--vllm_url', type=str, required=True, help='URL for vllm service')
    parser.add_argument('--vllm_api_key', type=str, default="EMPTY", help='service api key')
    args = parser.parse_args()
    
    main(args)
