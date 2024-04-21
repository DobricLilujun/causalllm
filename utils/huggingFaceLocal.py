from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TextGenerator:
    cached_model = None
    cached_tokenizer = None

    def __init__(self, model_name="microsoft/phi-2", max_length=200):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        if TextGenerator.cached_model is None or TextGenerator.cached_tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            TextGenerator.cached_model, TextGenerator.cached_tokenizer = (
                self.model,
                self.tokenizer,
            )
        else:
            self.model, self.tokenizer = (
                TextGenerator.cached_model,
                TextGenerator.cached_tokenizer,
            )

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda")

    def generate_text(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        )
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=self.max_length)
        generated_text = self.tokenizer.batch_decode(outputs)[0]
        return generated_text


# Example usage:
prompt = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """
'''
text_generator = TextGenerator()
generated_text = text_generator.generate_text(prompt)
print(generated_text)
