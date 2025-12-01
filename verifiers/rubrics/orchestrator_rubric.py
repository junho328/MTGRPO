from typing import List, Dict, Any
import re
from verifiers.rubrics import Rubric
from verifiers.parsers import XMLParser
from math_verify import parse, verify

class OrchestratorRubric(Rubric):
    def __init__(self):
        self.parser = XMLParser(fields=["think", "answer"])
        
    def extract_answer_content(self, content: str) -> str:
        """
        Extract content between <answer> and </answer> tags.
        If tags are not found, return the original content.
        """
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

        
    def format_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Checks if all Worker agents followed the <think>...</think><answer>...</answer> format.
        Orchestrator output is not checked for this format.
        Assumption: Turns alternate Orchestrator -> Worker -> Orchestrator -> Worker ...
        The completion messages start after the initial prompt.
        """
        rewards = []
        for completion in completions:
            all_workers_correct = True
            # Filter for assistant messages only
            assistant_msgs = [msg for msg in completion if msg['role'] == 'assistant']
            
            # Worker turns are at indices 1, 3, 5... (0-indexed)
            # Index 0 is Orchestrator, Index 1 is Worker, etc.
            for i, msg in enumerate(assistant_msgs):
                # Check all agents (Orchestrator and Worker)
                content = msg['content']
                # Check for <think> and <answer> tags
                if not (("<think>" in content and "</think>" in content) and 
                        ("<answer>" in content and "</answer>" in content)):
                    all_workers_correct = False
                    break
            
            rewards.append(0.5 if all_workers_correct else 0.0)
        return rewards

    def accuracy_reward_func(self, completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
        """
        Checks if the final Worker agent's answer matches the ground truth.
        """
        rewards = []
        for completion, ground_truth in zip(completions, answer):
            # Get the last assistant message (should be from the last Worker)
            assistant_msgs = [msg for msg in completion if msg['role'] == 'assistant']
            if not assistant_msgs:
                rewards.append(0.0)
                continue
                
            last_msg = assistant_msgs[-1]
            content = last_msg['content']
            
            # Parse the answer from <answer> tags
            try:
                parsed = self.parser.parse(content)
                if parsed.answer:
                    pred_answer = parsed.answer
                else:
                    # Fallback: try to extract answer if parsing failed but tags exist or generic extraction
                    pred_answer = self.extract_answer_content(content) or ""
            except:
                pred_answer = self.extract_answer_content(content) or ""

            # Compare with ground truth (using simple string inclusion or exact match after normalization)
            # GSM8K typically uses exact match of the number
            if self._check_answer(pred_answer, ground_truth):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def _check_answer(self, pred: str, gold: str) -> bool:
        """
        Check if the predicted answer matches the gold standard using math_verify.
        """
        try:
            return verify(gold, pred)
        except Exception:
            return False

