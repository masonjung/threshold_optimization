import transformers
import torch
import numpy as np
import pandas as pd

from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

login("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")


# from transformers import pipeline

# # Load Llama 3 model from Hugging Face
# llama3_model = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

# # Generate text using the Llama 3 model
# prompt = "Once upon a time"
# generated_text = llama3_model(prompt, max_length=50, do_sample=True)

# # Print the generated text
# print(generated_text[0]['generated_text'])


model_id = "meta-llama/Meta-Llama-3-8B"
 #"meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    use_auth_token=True  # This will use your Hugging Face token
)




# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=True
)


df = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")
df_sample = df.sample(frac=0.0001)
df_sample.shape


# Function to classify AI-written text using a pre-prompt strategy
def classify_ai_written(model, tokenizer, essay, device="cuda"):
    # Define a pre-prompt to guide the model
    pre_prompt = (
        "You are an AI content detector. For the given text, determine whether it is likely "
        "written by an AI or a human. Provide a classification (AI-written or Human-written) "
        "and a probability score between 0 and 1 representing the likelihood of being AI-written.\n\n"
        "Text: "
    )
    text = essay
    # Combine pre-prompt with the input text
    full_prompt = pre_prompt + text

    # Tokenize the combined prompt
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate the model's output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, temperature=0.7, do_sample=False)

    # Decode the model's output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract classification and probability from the response (parsing logic may vary)
    if "AI-written" in response:
        classification = "AI-written"
    elif "Human-written" in response:
        classification = "Human-written"
    else:
        classification = "Unknown"

    # Extract probability score from the response
    try:
        probability = float(response.split("Probability:")[1].strip().split()[0])
    except (IndexError, ValueError):
        probability = None  # If parsing fails, return None

    return {"classification": classification, "probability": probability, "response": response}

# Apply the function to classify essays
df_sample['meta-llama/Meta-Llama-3-8B_ai_probability'] = df_sample['essay'].apply(
    lambda x: classify_ai_written(model, tokenizer, x)['probability']
)

# Display the results
df_sample[['AI_written', 'meta-llama/Meta-Llama-3-8B_ai_probability']]

