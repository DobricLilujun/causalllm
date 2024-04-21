# https://github.com/openai/openai-python

from openai import OpenAI


def generate_text(prompt, model="gpt-3.5-turbo"):
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an smart assistant!",
            },
            {"role": "user", "content": prompt},
        ],
    )
    generated_text = completion.choices[0].message.content
    return generated_text


prompt = "Compose a poem that explains the concept of recursion in programming."
result = generate_text(prompt)
print(result)
