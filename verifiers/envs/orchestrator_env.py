import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Sequence, Union

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from ..imports import LLM, SamplingParams

from verifiers.envs.environment import Environment
from verifiers.rubrics.orchestrator_rubric import OrchestratorRubric
from verifiers.prompts.orchestrator_prompts import ORCHESTRATOR_PROMPT, ORCHESTRATOR_FINAL_PROMPT, WORKER_PROMPT
from verifiers.utils import preprocess_dataset

class OrchestratorWorkerEnv(Environment):
    def __init__(self,
                 dataset: str = "jhn9803/hendrycks-math-with-answers",
                 num_agents: int = 2,
                 max_steps: int = 6, # Safety limit
                 sleep_time: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        self.num_agents = num_agents
        self.max_steps = max_steps # Not strictly used as logic is driven by num_agents
        self.sleep_time = sleep_time
        self.rubric = OrchestratorRubric()
        
        # Preprocess dataset (standard logic)
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt="", # We handle system prompts dynamically
        )
        self.eval_dataset = None
        
        self.orchestrator_system_prompt = "You are a helpful assistant that decompose the problem into multiple sub-problems and solve them one by one."
        self.worker_system_prompt="You are a helpful assistant that solves complex problems."

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="validation",
                system_prompt="",
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return [self.rubric.format_reward_func, self.rubric.accuracy_reward_func]

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        if not live_indices:
            return states

        # Prepare inputs for LLM based on current role
        messages_batch = []
        for i in live_indices:
            state = states[i]
            if state["current_role"] == "orchestrator":
                # Construct Orchestrator Prompt
                # Prompt: System(Orch) + User(Problem + History + Remaining)
                
                # Format history
                history_text = ""
                if state["worker_outputs"]:
                    history_text = "\n\nPrevious Outputs:\n"
                    # Ensure we zip correctly. orchestrator_outputs should be same length as worker_outputs at this point
                    # because we just finished a worker step to get here (or it's start and both are empty)
                    # Wait, if it's start, both empty.
                    # If we are at step 2 (Orch turn 2), we have 1 orch out and 1 worker out.
                    for idx, (orch_out, work_out) in enumerate(zip(state["orchestrator_outputs"], state["worker_outputs"])):
                        history_text += f"Orchestrator {idx+1}: {orch_out}\n"
                        history_text += f"Worker {idx+1}: {work_out}\n"
                
                content = (f"Original Problem: {state['original_problem']}\n"
                           f"{history_text}\n"
                           f"Remaining Agents: {state['remaining_agents']}\n\n"
                           "Provide the next subtask.")
                
                # Select Prompt based on remaining agents
                if state['remaining_agents'] == 1:
                    prompt_template = ORCHESTRATOR_FINAL_PROMPT
                else:
                    prompt_template = ORCHESTRATOR_PROMPT

                msgs = [
                    {"role": "system", "content": self.orchestrator_system_prompt},
                    {"role": "user", "content": prompt_template.format(original_problem=state["original_problem"], previous_outputs=history_text, num_agents=state["remaining_agents"])}
                ]
                messages_batch.append(msgs)
                
            else: # Worker
                # Construct Worker Prompt
                # Prompt: System(Worker) + User(Subtask)
                msgs = [
                    {"role": "system", "content": self.worker_system_prompt},
                    {"role": "user", "content": WORKER_PROMPT.format(original_problem=state["original_problem"], orchestrator_instruction=state["current_subtask"])}
                ]
                messages_batch.append(msgs)

        # Call LLM
        # Note: use_tqdm=False to avoid noise
        llm_responses = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=False)

        # Update states
        def update_state(j, llm_response):
            # Random sleep to avoid potential race conditions if any (standard in this repo)
            time.sleep(self.sleep_time * random.random())
            
            state = states[j].copy()
            output_text = llm_response.outputs[0].text
            token_ids = list(llm_response.outputs[0].token_ids)
            
            # Append to full history (for Rubric and Logging)
            # Note: We append the *output* generated by the model. 
            # The prompt used for this generation is NOT appended to 'messages' 
            # because 'messages' tracks the logical flow of the conversation from the perspective 
            # of the Orchestrator-Worker chain. 
            # Ideally, we should log what happened.
            # Let's append the generated text as 'assistant' role.
            state["messages"].append({"role": "assistant", "content": output_text})
            
            # Append IDs and Masks
            state["completion_ids"].extend(token_ids)
            state["completion_mask"].extend([1] * len(token_ids)) # Train on everything

            # State Transition
            if state["current_role"] == "orchestrator":
                # Parse instruction from <answer> tags
                instruction = output_text
                match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
                if match:
                    instruction = match.group(1).strip()
                
                state["orchestrator_outputs"].append(output_text)
                state["current_subtask"] = instruction
                state["current_role"] = "worker"
            else: # Worker finished
                state["worker_outputs"].append(output_text)
                state["remaining_agents"] -= 1
                state["current_role"] = "orchestrator"
                
                if state["remaining_agents"] <= 0:
                    state["completed"] = True

            return j, state

        # Parallel update (though in Python GIL makes this mostly serial for CPU work, useful for IO or heavy logic)
        with ThreadPoolExecutor(max_workers=len(live_indices)) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state
            
        return states

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, Any]:
        
        # Initialize states
        states = []
        for prompt in prompts:
            # prompt is [{"role": "user", "content": "Problem..."}]
            original_problem = prompt[-1]["content"]
            
            states.append({
                "messages": prompt, # Starts with the user problem
                "original_problem": original_problem,
                "orchestrator_outputs": [],
                "worker_outputs": [],
                "remaining_agents": self.num_agents,
                "current_role": "orchestrator",
                "current_subtask": None,
                "prompt_ids": [], # Filled if needed, but GRPO trainer handles prompt encoding usually
                "completion_ids": [],
                "completion_mask": [],
                "completed": False
            })
            
        all_completed = False
        while not all_completed:
            states = self.step(states, llm, sampling_params)
            all_completed = all(s["completed"] for s in states)
            
        # Format output for Trainer
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        completion_messages = [s["messages"] for s in states] # This includes the initial prompt?
        # Usually trainer expects completion_messages to be just the new messages?
        # MultiTurnEnv returns: s["messages"][s["prompt_messages"]:]
        # Let's follow that.
        
        # Re-calculate prompt length to slice messages
        final_completion_messages = []
        for i, s in enumerate(states):
            input_prompt_len = len(prompts[i])
            final_completion_messages.append(s["messages"][input_prompt_len:])

        return {
            "ids": completion_ids,
            "mask": completion_mask,
            "messages": final_completion_messages
        }

