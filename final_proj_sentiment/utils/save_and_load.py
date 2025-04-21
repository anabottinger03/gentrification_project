import pickle
import numpy as np

def save_naive_bayes_model(naiive_bayes):
    with open("sentiment_model/models/nb_model.pkl", "wb") as f:
        pickle.dump({
            "class_counts": naiive_bayes.class_counts,
            "class_word_counts": naiive_bayes.class_word_counts,
            "vocab": naiive_bayes.vocab,
            "total_words_per_class": naiive_bayes.total_words_per_class
        }, f)

def load_naive_bayes_model(naiive_bayes):
    with open("sentiment_model/models/nb_model.pkl", "rb") as f:
        data = pickle.load(f)
        naiive_bayes.class_counts = data["class_counts"]
        naiive_bayes.class_word_counts = data["class_word_counts"]
        naiive_bayes.vocab = data["vocab"]
        naiive_bayes.total_words_per_class = data["total_words_per_class"]


def save_vector_model(model_name, weights):
    with open(f"sentiment_model/models/{model_name}_weights.npy", "wb") as f:
        np.save(f, weights)

def load_vector_model(model_name):
    return np.load(f"sentiment_model/models/{model_name}_weights.npy")

def save_vectorizer(vectorizer):
    with open("sentiment_model/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer():
    with open("sentiment_model/models/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

