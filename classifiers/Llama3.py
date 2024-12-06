# Libraries
import transformers
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Llama Model
model_id = "meta-llama/Llama-3.1-8B"
# model_id = "meta-llama/Meta-Llama-3-70B"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    use_auth_token=True
)


# Dataset
df = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")

# Sample a small fraction of the dataset
df = df.sample(frac=0.0001)
print(df.shape)

# Llama3 for probability
def llama_ai_probability(text):
    # Create the prompt
    prompt = (
        "You are an AI content detector. Estimate the probability that the following text was generated by an AI model. Respond only with a single probability value between 0 and 1, formatted to 7 decimal points.\n\n"
        "Text:\n"
        f"{text}\n"
    )

    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate a response from Llama
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10*3, temperature=0.7, do_sample=True)


        # Decode and parse the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # # Extract the probability from the response (assuming Llama generates the probability at the end)
        # probability_text = response.split("\n")[-1].strip()
        # probability = float(probability_text)
        
        print(response)

    except Exception as e:
        print(f"Error: {e}")
        return None

# Apply the function to each row in the dataframe and create a new column
df['Llama3.1-8B_probability'] = df['essay'].apply(lambda x: llama_ai_probability(x))

# Print the first few rows to verify results
print(df[['essay', 'Llama3.1-8B_probability']])
