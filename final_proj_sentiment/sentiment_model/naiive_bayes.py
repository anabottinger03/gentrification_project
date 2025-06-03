import math
import re
from collections import defaultdict, Counter

class_word_counts = defaultdict(Counter)
class_counts = Counter()
total_words_per_class = defaultdict(int)
vocab = set()

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def train_naive_bayes(texts, labels):
    for text, label in zip(texts, labels):
        words = tokenize(text)
        class_counts[label] += 1
        class_word_counts[label].update(words)
        total_words_per_class[label] += len(words)
        vocab.update(words)

# Prediction function
def predict_naive_bayes(text):
    words = tokenize(text)
    scores = {}

    for label in class_counts:
        # Start with log prior
        log_prob = math.log(class_counts[label] / sum(class_counts.values()))
        for word in words:
            word_count = class_word_counts[label][word] + 1  # Laplace smoothing
            total = total_words_per_class[label] + len(vocab)
            log_prob += math.log(word_count / total)
        scores[label] = log_prob

    return max(scores, key=scores.get)

def compute_loss(text, true_label):
    words = tokenize(text)
    log_probs = {}
    
    for label in class_counts:
        # Log prior
        log_prob = math.log(class_counts[label] / sum(class_counts.values()))
        for word in words:
            word_count = class_word_counts[label][word] + 1  # Laplace smoothing
            total = total_words_per_class[label] + len(vocab)
            log_prob += math.log(word_count / total)
        log_probs[label] = log_prob

    # Convert log_probs to normalized probs using log-sum-exp trick
    max_log = max(log_probs.values())
    exp_scores = {label: math.exp(log_probs[label] - max_log) for label in log_probs}
    total = sum(exp_scores.values())
    probs = {label: score / total for label, score in exp_scores.items()}

    # Return the negative log-likelihood of the true label
    epsilon = 1e-12  # Avoid log(0)
    return -math.log(probs.get(true_label, epsilon))

