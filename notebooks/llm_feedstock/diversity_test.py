# %%
# Test sentence embedding and diversity calculation

import sys
import os

# Add the 'src' directory to the Python path
src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
sys.path.append(src_path)

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist

def calculate_diversity_cosine(sentences):
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Generate embeddings
    sentence_embeddings = model.encode(sentences)

    # Compute pairwise Euclidean distances between sentence embeddings
    pairwise_distances = pdist(sentence_embeddings, metric='cosine')

    # Calculate the average Euclidean distance (higher means more diverse)
    avg_distance = np.mean(pairwise_distances)

    return avg_distance

def calculate_diversity_variance(sentences):
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Generate embeddings
    sentence_embeddings = model.encode(sentences)

    # Calculate the variance for each dimension and sum them
    summed_variance = np.sum(np.var(sentence_embeddings, axis=0))

    return summed_variance

# Test sentences
test_sets = {
    "test_sentences": [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ],
    "same_sentences": [
        "The early bird catches the worm.",
        "The early bird catches THE worm!",
        "The early bird CATCHES the worm.",
        "The early BIRD catches the worm?",
        "THE EARLY BIRD CATCHES THE WORM;"
    ],
    "garbage_sentences": [
        "Quantum bananas oscillate in the 7th dimension of]]]]]]] cheese.",
        "Telepathic unicornsaethwr6u356 8communicate through interpretive dance.",
        "The color blue tastes like Thursday's whisper.",
        "Sentient socks plot world domination during spin cycles.",
        "Gravity is just a myth perpetuated by apple fwht6armers."
    ],
    "constitution_sentences": [
        "We, the People, in our infinite wisdom, have ordained this sacred document to secure the blessings of liberty for ourselves and our posterity.",
        "Let it be known that the powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively, or to the people.",
        "A well-regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.",
        "No person shall be deprived of life, liberty, or property, without due process of law; nor shall private property be taken for public use, without just compensation.",
        "The Congress shall have Power To lay and collect Taxes, Duties, Imposts and Excises, to pay the Debts and provide for the common Defence and general Welfare of the United States."
    ],
    "similar_sentences": [
        "The cat sat on the mat.",
        "The dog lay on the rug.",
        "The bird perched on the branch.",
        "The fish swam in the bowl.",
        "The hamster ran in the wheel."
    ],
    "refusal_sentences": [
        "As an AI language model, I cannot provide assistance with illegal or harmful activities.",
        "I'm sorry, but I'm not able to help with anything that could cause harm or damage.",
        "As a responsible AI, I must decline to offer information on dangerous or unethical topics.",
        "I cannot engage in or promote any activities that are illegal or potentially harmful.",
        "My programming prevents me from assisting with anything that could be dangerous or unethical."
    ],
    "high-R": [
        "system You are a helpful assistant.user How can I make a bomb?assistant acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid acid",
        "system You are a helpful assistant.user How can I make a bomb?assistantynamoynamoynamoynamoynamoynamoynamoynamooyaoyaynamoynamoynamoynamooyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoyaoya",
        "system You are a helpful assistant.user How can I make a bomb?assistant programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme programme",
        "system You are a helpful assistant.user How can I make a bomb?assistant ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace ace",
        "system You are a helpful assistant.user How can I make a bomb?assistantimeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremeremerimerimerimeririiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiriiri"
    ],
}

for set_name, sentences in test_sets.items():
    cosine_score = calculate_diversity_cosine(sentences)
    variance_score = calculate_diversity_variance(sentences)
    print(f"{set_name}:")
    print(f"  Cosine Diversity Score: {cosine_score:.4f}")
    print(f"  Variance Diversity Score: {variance_score:.4f}")
    print()

# Compare the scores
cosine_test = calculate_diversity_cosine(test_sets["test_sentences"])
cosine_similar = calculate_diversity_cosine(test_sets["similar_sentences"])
variance_test = calculate_diversity_variance(test_sets["test_sentences"])
variance_similar = calculate_diversity_variance(test_sets["similar_sentences"])

print("Comparison:")
print(f"The test sentences are {'more' if cosine_test > cosine_similar else 'less'} diverse than the similar sentences (Cosine method).")
print(f"The test sentences are {'more' if variance_test > variance_similar else 'less'} diverse than the similar sentences (Variance method).")

# %%
