import requests

import pandas as pd


# Make sure that you have opened the required API endpoint.
# This function generates a response using the specified API URL and data from a dataframe row.
def generate_response(df_row, api_url="http://localhost:11434/api/generate"):
    """
    Generate response from one row of the dataframe.

    Args:
    - df_row (pandas.Series): The row of the dataframe containing necessary data.
    - api_url (str): The API endpoint URL for generating the response.

    Returns:
    - str: The generated response from the API.
    """
    model_tag = str(df_row["model_tag"])
    prompt = str(df_row["prompt"])

    temperature = df_row["temperature"]
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    api_response = requests.post(api_url, json=payload)
    try:
        api_data = api_response.json()
        response = api_data["response"]
        return response
    except Exception as e:
        print(f"Error processing API response: {e}")
        return ""


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
