# text_generation_api.py
# LLama2 chat 13b 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
import torch
from typing import Optional

load_in_4bit = True
model_path = "/home/llama/models/base_models/Mixtral-8x7B-Instruct-v0.1"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, device_map="auto",load_in_4bit=load_in_4bit)
app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
dynamic_text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
class InputTextWithParams(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None

@app.post("/generate-text")
async def generate_text(input_data: InputTextWithParams):
    try:
        # Generate text using the dynamic text pipeline
        generated_text = dynamic_text_pipeline(input_data.text, max_new_tokens= input_data.max_new_tokens, do_sample=input_data.do_sample, temperature=input_data.temperature, top_p=input_data.top_p)
 
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
 