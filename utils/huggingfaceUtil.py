import requests

import pandas as pd
from langchain import PromptTemplate


# Make sure that you have opened the required API endpoint.
# This function generates a response using the specified API URL and data from a dataframe row.
def generate_response_hf(df_row, api_url="http://127.0.0.1:8899/generate-text"):
    """
    Generate response from one row of the dataframe.

    Args:
    - df_row (pandas.Series): The row of the dataframe containing necessary data.
    - api_url (str): The API endpoint URL for generating the response.

    Returns:
    - str: The generated response from the API.
    """
    prompt = str(df_row["prompt"])
    temperature = df_row["temperature"]
    print(temperature)
    payload = {
        "text": prompt[:4096],
        "temperature": temperature,
        "max_new_tokens": 256,
        "max_length": 1024,
    }
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()["result"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        result = ""

    return result


def generate_story_by_loop(
    plot,
    initial_prompt,
    recurrent_check_prompt,
    word_count_requirement,
    temperature,
    max_new_tokens,
    max_length,
    api_url="http://127.0.0.1:8899/generate-text",
):
    initial_prompt_template = PromptTemplate.from_template(initial_prompt)
    recurrent_check_prompt_template = PromptTemplate.from_template(
        recurrent_check_prompt
    )
    prompt = initial_prompt_template.format(
        plot=plot, word_count=word_count_requirement
    )
    payload = {
        "text": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "max_length": max_length,
    }
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        story = response.json()["result"]
    current_word_count = len(story.split())
    while abs(current_word_count - word_count_requirement) > 200:
        prompt = recurrent_check_prompt_template.format(
            story=story,
            word_count=word_count_requirement,
            current_word_count=current_word_count,
        )
        payload["text"] = prompt
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            story = response.json()["result"]

    return story


# This function generates a response with specified parameters.
def generate_ollama(
    prompt="",
    api_url="http://localhost:11434/api/generate",
    tag="llama2:7b",
    **extra_params,
):
    """
    Generate response with the input of a dataframe row and additional parameters.

    Args:
    - prompt (str): The prompt for generating the response.
    - api_url (str): The API endpoint URL for generating the response.
    - tag (str): The model tag for the response.
    - **extra_params: Additional parameters to be included in the options.

    Returns:
    - str: The generated response from the API.
    """
    payload = {
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "options": extra_params,
    }
    api_response = requests.post(api_url, json=payload)
    try:
        api_data = api_response.json()
        response = api_data["response"]
        return response
    except Exception as e:
        print(f"Error processing API response: {e}")
        return ""


# Example for the test and how to use
def main():

    # Example to use generate response functions
    data = {
        "model_tag": ["llama2", "llama2"],
        "prompt": ["Hello", "How are you?"],
        "temperature": [0.7, 0.8],
    }
    df = pd.DataFrame(data)
    for index, row in df.iterrows():
        response = generate_response(row)
        print(
            f"Model: {row['model_tag']}, Prompt: {row['prompt']}, Response: {response}"
        )
    print("\n")

    # Example to use generate_ollama function
    for index, row in df.iterrows():
        extra_params = {"temperature": row["temperature"]}
        response = generate_ollama(
            prompt=row["prompt"], tag=row["model_tag"], **extra_params
        )
        print(
            f"Model: {row['model_tag']}, Prompt: {row['prompt']}, Response: {response}"
        )


if __name__ == "__main__":
    main()
