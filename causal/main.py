# main.py
from text_generation_api import create_text_generation_app
import os

api_params = {
    "host": "127.0.1.1",
    "port": 8889,
}

model_name = "meta-llama/Llama-2-7b-hf"

model_params = {
    "max_new_tokens": 1024,
    "temperature": 0.0001,
    "top_p": 0.95,
    "do_sample": True,
    "repetition_penalty": 1.15,
}

app = create_text_generation_app(api_params, model_name, model_params)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=api_params["host"], port=api_params["port"], reload=True)
    print(
        f"FastAPI is ongoing, please use the following address: http://{api_params['host']}:{api_params['port']}"
    )


# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
