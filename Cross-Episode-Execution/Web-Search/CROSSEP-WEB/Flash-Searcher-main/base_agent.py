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

from dotenv import load_dotenv
from utils import safe_json_loads

from FlashOAgents import ToolCallingAgent
from FlashOAgents import ActionStep, PlanningStep, TaskStep, SummaryStep
from FlashOAgents import WebSearchTool, CrawlPageTool, VisualInspectorTool, AudioInspectorTool, TextInspectorTool

load_dotenv(override=True)

class BaseAgent:
    def __init__(self, model):
        self.model = model
        self.agent_fn = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def capture_trajectory(self, ):
        if not hasattr(self, 'agent_fn'):
            raise ValueError("[capture_trajectory] agent_fn is not defined.")
        if not isinstance(self.agent_fn, ToolCallingAgent):
            raise ValueError("[capture_trajectory] agent_fn must be an instance of ToolCallingAgent.")
        trajectory = []
        for step_num, step in enumerate(self.agent_fn.memory.steps):
            if isinstance(step, TaskStep):
                continue
            elif isinstance(step, PlanningStep):
                traj = {
                    "name": "plan", 
                    "value": step.plan, 
                    "think": step.plan_think, 
                    "cot_think": step.plan_reasoning,
                    "memory_guidance": step.memory_guidance
                }
                trajectory.append(traj)
            elif isinstance(step, SummaryStep):
                traj = {"name": "summary", "value": step.summary, "cot_think": step.summary_reasoning}
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                safe_tool_calls = step.tool_calls if step.tool_calls is not None else []
                traj = {
                    "name": "action", 
                    "tool_calls": [st.dict() for st in safe_tool_calls], 
                    "obs": step.observations,
                    "think": step.action_think, 
                    "cot_think": step.action_reasoning,
                    "memory_guidance": step.memory_guidance
                }
                trajectory.append(traj)
            else:
                raise ValueError("[capture_trajectory] Unknown Step:", step)

        return {
            "agent_trajectory": trajectory,
        }

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                if answer is not None:
                    result = self.agent_fn.run(task, answer=answer)
                else:
                    result = self.agent_fn.run(task)
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)
                return {
                    "agent_result": result, **self.capture_trajectory()
                }
            except Exception as e:
                if "TERMINATE_QUOTA" in str(e):
                    return {"error": "quota_exceeded"}
                last_error = e
                print(f"[BaseAgent] error: {e}")
                continue
        return {"error": str(last_error)}


class SearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, memory_provider=None, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        tools = [web_tool, crawl_tool]
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )

class MMSearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, memory_provider=None, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        visual_tool = VisualInspectorTool(model, 100000)
        text_tool = TextInspectorTool(model, 100000) 
        audio_tool = AudioInspectorTool(model, 100000) 
        tools = [web_tool, crawl_tool, visual_tool,text_tool,audio_tool] 

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )


class AnalysisAgent(BaseAgent):
    """Agent for analyzing task execution trajectories"""
    
    def __init__(self, model, task_logs_dir, prompts_type="default", max_steps=15, summary_model_id="gpt-4o-mini", **kwargs):
        super().__init__(model)
        
        from MemEvolve.utils.trajectory_tools import (
            TrajectoryViewerTool,
            StepViewerTool,
            MemoryDatabaseViewerTool
        )
        
        tools = [
            TrajectoryViewerTool(task_logs_dir, max_tasks=3, model_id=summary_model_id),
            StepViewerTool(task_logs_dir),
            MemoryDatabaseViewerTool()
        ]
        
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=5,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=None
        )


class ReviewAgent(BaseAgent):
    """Agent for code review and validation"""
    
    def __init__(self, model, prompts_type="default", max_steps=10, **kwargs):
        super().__init__(model)
        
        # No tools needed - pure reasoning agent
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=[],
            summary_interval=5,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=None
        )