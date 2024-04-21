from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
from difflib import ndiff
from transformers import AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU (Bilingual Evaluation Understudy) score between a reference and a candidate sentence.

    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - float: The BLEU score between the reference and candidate sentences.
    """
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)


def calculate_bleu_scores_smooth(ground_truth_responses, generated_responses):
    smooth = SmoothingFunction()
    bleu_scores = [
        round(
            sentence_bleu(
                [ref.split()], gen.split(), smoothing_function=smooth.method1
            ),
            2,
        )
        for ref, gen in zip(ground_truth_responses, generated_responses)
    ]

    return bleu_scores


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence) score
    between a reference and a candidate sentence.

    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - float: The ROUGE-L score between the reference and candidate sentences.
    """
    lcs_length = edit_distance(
        reference, candidate, substitution_cost=2, transpositions=True
    )

    recall = lcs_length / len(reference)
    precision = lcs_length / len(candidate)

    if precision + recall == 0:
        rouge_l_score = 0
    else:
        rouge_l_score = 2 * (precision * recall) / (precision + recall)

    return rouge_l_score


def calculate_cosine_similarity(text, groundtruth, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    input_ids_text = tokenizer.encode(text, truncation = True, return_tensors="pt", max_length = 1024)
    input_ids_groundtruth = tokenizer.encode(groundtruth, return_tensors="pt")

    with torch.no_grad():
        embeddings_text = model(input_ids_text)[0].squeeze().numpy()
        embeddings_groundtruth = model(input_ids_groundtruth)[0].squeeze().numpy()

    similarity_text = cosine_similarity(embeddings_text, embeddings_groundtruth)[0, 0]
    return similarity_text
