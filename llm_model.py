from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llm():
    model_id = "gpt2"  # Use a simple model first

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return tokenizer, model

def generate_answer(context, question, tokenizer, model):
    inputs = tokenizer.encode(context + question, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=256, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer



