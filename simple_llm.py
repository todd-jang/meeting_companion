from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import random
my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}
params = {
        GenParams.MAX_NEW_TOKENS: 700, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }
LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )
llm = WatsonxLLM(LLAMA2_model)  
#print(llm("How to read a book effectively?"))


# --- Hyperparameters ---
M = 10  # Number of steps
G = 5  # Number of outputs per prompt
N = 20  # Buffer size
µ = 3  # Number of inner loop iterations
εlow = 0.2
εhigh = 0.8

# --- Task Prompts (D) ---
D = [
    "What is the capital of France?",
    "How do I bake a cake?",
    "Explain the theory of relativity.",
    "What are the benefits of exercise?",
    "Write a short story about a cat."
]

# --- Reward Model (R) - Simple Example ---
def reward_model(output, prompt):
    """A very basic reward model.
    In a real scenario, this would be much more complex."""
    if "Paris" in output and "capital of France" in prompt:
        return 1.0
    elif "cake" in output and "bake a cake" in prompt:
        return 0.8
    else:
        return 0.1

# --- Dynamic Sampling Buffer ---
buffer = []

# --- Main Loop ---
for step in range(M):
    # 1. Sample a batch
    Db = random.sample(D, k=2)  # Sample 2 prompts

    # 2. Update old policy
    # In this example, we don't have a separate "old" policy.
    # We're just using the same LLM for now.
    # In a more complex implementation, you'd save a copy of the model here.

    # 3. Sample outputs
    outputs_with_rewards = []
    for q in Db:
        for _ in range(G):
            oi = llm(q)  # Generate output
            ri = reward_model(oi, q)  # Compute reward
            outputs_with_rewards.append((oi, ri, q))

    # 4. Compute rewards (done in the previous step)

    # 5. Filter and add to buffer
    for oi, ri, q in outputs_with_rewards:
        if εlow < ri < εhigh:
            buffer.append((oi, ri, q))
    
    # 6. Buffer size check
    if len(buffer) < N:
        continue

    # 7. Compute advantage estimates (placeholder)
    # This is where you'd implement Equation (9)
    # For now, we'll just use the reward as a simple advantage estimate.
    advantages = [ri for _, ri, _ in buffer]

    # 8. Inner loop
    for iteration in range(µ):
        # Update the policy model (placeholder)
        # This is where you'd implement Equation (8)
        # For now, we're just printing a message.
        print(f"Step {step+1}, Iteration {iteration+1}: Updating policy...")

# Output πθ (the trained model)
print("Training complete.")

# Post-training inference:
new_prompt = "What is the best way to learn a new language?"
generated_text = llm(new_prompt)
print(f"\nGenerated text for prompt '{new_prompt}':\n{generated_text}")

new_prompt = "Write a short story about a dog."
generated_text = llm(new_prompt)
print(f"\nGenerated text for prompt '{new_prompt}':\n{generated_text}")