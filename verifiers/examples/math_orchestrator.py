import argparse
import verifiers as vf
from verifiers.envs.orchestrator_env import OrchestratorWorkerEnv
from peft import LoraConfig

parser = argparse.ArgumentParser(description='Run GSM8K Orchestrator-Worker example with PEFT')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Coder-Instruct", help='Model name or path')
parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--num_generations', type=int, default=8, help='Rollouts per prompt')
parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Per device train batch size')
parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--num_iterations', type=int, default=1, help='Number of iterations')
parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of training steps')
parser.add_argument('--beta', type=float, default=0.01, help='Beta parameter for KL divergence')
parser.add_argument('--num_agents', type=int, default=2, help='Number of worker agents (default: 2)')
parser.add_argument('--trainer', type=str, default="grpo", choices=["mt_grpo", "grpo"], help='Trainer type')

args = parser.parse_args()

model_name = args.model_name
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Initialize Environment
# Note: num_agents controls the number of Orchestrator-Worker cycles.
num_agents = args.num_agents
num_gpus = args.num_gpus

vf_env = OrchestratorWorkerEnv(
    dataset="math",
    num_agents=num_agents,
    max_steps=num_agents * 2 + 2 # Margin for safety
)

train_dataset = vf_env.get_dataset()
reward_funcs = vf_env.get_rubric()

turn_reward_funcs = [reward_funcs[0]]
print(f"Turn reward functions: {turn_reward_funcs}")
outcome_reward_funcs = [reward_funcs[1]]
print(f"Outcome reward functions: {outcome_reward_funcs}")

run_name = "orchestrator-math-peft_"+model_name.split("/")[-1].lower()

training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=num_gpus
)
training_args.learning_rate = args.learning_rate
training_args.num_generations = args.num_generations
training_args.per_device_train_batch_size = args.per_device_train_batch_size
training_args.gradient_accumulation_steps = args.grad_accum_steps
training_args.num_iterations = args.num_iterations
training_args.max_steps = args.max_steps
training_args.beta = args.beta

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules="all-linear"
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    turn_reward_funcs=turn_reward_funcs,
    outcome_reward_funcs=outcome_reward_funcs,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config
)
trainer.train()

