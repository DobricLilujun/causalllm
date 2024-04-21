# text_generation_api.py
# LLama2 chat 13b 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
 
model_path = "/home/llama/Personal_Directories/srb/causalllm-main/model/Llama-2-7b-chat-hf"
load_in_4bit = True

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, device_map="auto",load_in_4bit=load_in_4bit)

dynamic_text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer = tokenizer,
            device_map = "auto"
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
        generated_text =  dynamic_text_pipeline(input_data.text, max_new_tokens= input_data.max_new_tokens, do_sample=input_data.do_sample, temperature=input_data.temperature, top_p=input_data.top_p)

        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
 