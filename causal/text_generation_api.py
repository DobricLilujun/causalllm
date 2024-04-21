# text_generation_api.py
# LLama2 chat 13b 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from typing import Optional
 
api_params = {
    "host": "127.0.1.1",
    "port": 8889,
}


model_path = "/home/llama/models/base_models/Mixtral-8x7B-Instruct-v0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(3)  # Set the gpu output

app = FastAPI()
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map=device
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, load_in_8bit = True, trust_remote_code=True, device_map=device
# )
 
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
        dynamic_generation_config = GenerationConfig(
            max_length=input_data.max_length,
            max_new_tokens = input_data.max_new_tokens,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            do_sample=input_data.do_sample,
            repetition_penalty=input_data.repetition_penalty
        )

        # Create the text pipeline with the dynamic generation config
        dynamic_text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            config=dynamic_generation_config
        )
 
        # Generate text using the dynamic text pipeline
        generated_text = dynamic_text_pipeline(input_data.text)
 
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
 