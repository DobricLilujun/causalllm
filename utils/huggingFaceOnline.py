import os
import requests

def huggingface_api_query(model_name, input_text):
    """
    Query Hugging Face API for model inference.

    Parameters:
    - model_name (str): The name or ID of the Hugging Face model.
    - input_text (str): The input text for model inference.

    Returns:
    - response (dict): The JSON response from the API.
    """
    hugging_face_api_token = os.environ.get("HUGGING_FACE_API_TOKEN")
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hugging_face_api_token}"}

    payload = {"inputs": input_text}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

model_name = "gpt2"
input_text = "Can you please let us know more details about your"

result = huggingface_api_query(model_name, input_text)
print(result)
