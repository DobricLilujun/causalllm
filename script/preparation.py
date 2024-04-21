# This file is used to download and udpate the model and let the script runnble

import pandas as pd
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM


# Dataset

dataset_links = []


# Model

TOKEN = "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"
device = "cpu"  # CPU only for downloading the files
model_id = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    use_auth_token=TOKEN,
)
tokenizer.save_pretrained("model/Llama-2-70b-chat-hf")
model.save_pretrained("model/Llama-2-70b-chat-hf")


# Check environment to do
