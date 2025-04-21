"""
for training all of the models 
"""
import numpy as np
import pandas as pd 
import argparse
import os
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sentiment_model.logistic_regression import logistic_fit, log_predict
from sentiment_model.svm import train_svm, predict_svm
from sentiment_model import naiive_bayes
from utils.save_and_load import save_naive_bayes_model, save_vector_model, save_vectorizer
from sentiment_model.naiive_bayes import train_naive_bayes, predict_naive_bayes, compute_loss
from final_proj_sentiment.config import LogConfig


def split_and_vectorize(train_sentiment_data):
    """
    top level train test split to use accross all models. 
    """
 
    if "tweet" not in train_sentiment_data.columns:
        raise ValueError("Input data must contain a 'tweet' column.")

    df = train_sentiment_data.dropna(subset=["tweet", "sentiment_label"]).copy()
    df["tweet"] = df["tweet"].astype(str)

    df = df[df["tweet"].str.lower() != "nan"]
    assert df["tweet"].isnull().sum() == 0

    X = df["tweet"]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    

    return X_train, y_train, X_test, y_test, X_val, y_val 

def run_log_model(log_config, X_train_vec, y_train, X_test_vec, y_test, X_val_vec, y_val):
    
    
    prev_loss, theta = logistic_fit(X_train_vec, y_train, learning_rate=log_config.learning_rate, strength=log_config.strength, num_iterations=log_config.num_iterations, regularization=log_config.regularization)
    predict_train = log_predict(X_train_vec, theta)
    predict_val = log_predict(X_val_vec, theta)
    predict_test = log_predict(X_test_vec, theta)

    accuracy_logistic_train = accuracy_score(y_train, predict_train)
    accuracy_logistic_test = accuracy_score(y_test, predict_test)
    accuracy_logistic_val = accuracy_score(y_val, predict_val)

    print("Logistic Regression Accuracy on train Set:", accuracy_logistic_train)
    print("Logistic Regression Accuracy on val Set:", accuracy_logistic_val)
    print("Logistic Regression Accuracy on test Set:", accuracy_logistic_test)

    return prev_loss, theta, accuracy_logistic_train, accuracy_logistic_test, accuracy_logistic_val

def run_svm_model(svm_config, X_train_vec, y_train, X_test_vec, y_test, X_val_vec, y_val): 
    w_svm_train, losses = train_svm(
        X_train_vec,
        y_train,
        learning_rate=svm_config.learning_rate,
        strength=svm_config.strength,
        iters=svm_config.num_iterations,
        regularization=svm_config.regularization
    )

    y_pred_train = predict_svm(X_train_vec, w_svm_train)
    y_pred_test = predict_svm(X_test_vec, w_svm_train)
    y_pred_val = predict_svm(X_val_vec, w_svm_train)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_val = accuracy_score(y_val, y_pred_val)
    
    print("Train accuracy (SVM):", acc_train)
    print("Test accuracy (SVM):", acc_test)
    print("Val accuracy (SVM):", acc_val)

    return losses, w_svm_train, acc_train, acc_test, acc_val


def run_nb_model(X_train_raw, y_train, X_test_raw, y_test, X_val_raw, y_val, config=None):
    losses = []
    n = len(X_train_raw)

    # Simulate increasing training data to visualize loss reduction
    for i in range(100, n + 1, max(n // 20, 100)):
        train_naive_bayes(X_train_raw[:i], y_train[:i])
        avg_loss = np.mean([
            compute_loss(text, label)
            for text, label in zip(X_val_raw[:i], y_val[:i])
        ])
        losses.append(avg_loss)

    # Final training on full set
    train_naive_bayes(X_train_raw, y_train)

    y_pred_train = [predict_naive_bayes(text) for text in X_train_raw]
    y_pred_test = [predict_naive_bayes(text) for text in X_test_raw]
    y_pred_val = [predict_naive_bayes(text) for text in X_val_raw]

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_val = accuracy_score(y_val, y_pred_val)

    print("Train accuracy (NB):", acc_train)
    print("Val accuracy (NB):", acc_val)
    print("Test accuracy (NB):", acc_test)

    # Plot loss
    x_vals = list(range(100, n + 1, max(n // 20, 100)))
    plt.plot(x_vals, losses)
    plt.title("Naive Bayes NLL over Increasing Training Samples")
    plt.xlabel("Training Samples")
    plt.ylabel("Average Negative Log-Likelihood")
    plt.grid(True)

    os.makedirs("results/figs", exist_ok=True)
    fname = "results/figs/nb_loss.png"
    if config:
        fname = f"results/figs/nb_loss_{config.learning_rate}_{config.strength}_{config.num_iterations}_{config.regularization}.png"
    plt.savefig(fname)
    plt.clf()

    return losses[-1], acc_train, acc_test, acc_val



def log_experiment(config, prev_loss, train_acc, val_acc, test_acc, filename):
    """
    Logs the config and accuracy results to a CSV file.
    Appends a new row if file already exists.
    """
    os.makedirs("results", exist_ok=True)

    headers = ["learning_rate", "strength", "num_iterations", "regularization",
               "train_accuracy", "val_accuracy", "test_accuracy", "loss", 'training_set_size']

    row = {
        "learning_rate": config.learning_rate,
        "strength": config.strength,
        "num_iterations": config.num_iterations,
        "regularization": config.regularization,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "loss": prev_loss,
        'training_set_size': 200000
    }

    write_header = not os.path.exists(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)




def main():
    train_sentiment_data = pd.read_csv("data/training_tweets_100k.csv")
    X_train, y_train, X_test, y_test, X_val, y_val = split_and_vectorize(train_sentiment_data)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),         
        norm='l2',                  
        min_df=5,                   
        max_df=0.9                  
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    X_val_vec = vectorizer.transform(X_val)


    save_vectorizer(vectorizer)
    model_params = get_cli_args()
    # logistic regression 
    if model_params.model == "log": 
        prev_loss, theta, accuracy_train, accuracy_test, accuracy_val = run_log_model(model_params, X_train_vec, y_train, X_test_vec, y_test, X_val_vec, y_val)
        if model_params.save_model:
            save_vector_model("log", theta)
    elif model_params.model == "svm":
        prev_loss, weights, accuracy_train, accuracy_test, accuracy_val = run_svm_model(model_params, X_train_vec, y_train, X_test_vec, y_test, X_val_vec, y_val)
        if model_params.save_model:
            save_vector_model("svm", weights)
    elif model_params.model == "nb": 
        prev_loss, accuracy_train, accuracy_test, accuracy_val = run_nb_model(
            X_train, y_train, X_test, y_test, X_val, y_val
        )
        if model_params.save_model:
            save_naive_bayes_model(naiive_bayes)

       

    log_experiment(model_params, prev_loss, accuracy_train, accuracy_val, accuracy_test, f"results/{model_params.model}_results_test.csv")


def get_cli_args(): 
    parser = argparse.ArgumentParser(description="pipeline execution for sentiment model training")
    parser.add_argument("model", type=str, help="sentiment model (svm, nb, log)", default = "nb")
    parser.add_argument("--learning_rate", type=float, help="learning rate for model", default = 0.0)
    parser.add_argument("--strength", type=float, help="strength", default = 0.0)
    parser.add_argument("--num_iterations", type=int, help="number of training iterations", default = 1)
    parser.add_argument("--regularization", type=str, help="regularization method for model (L1, L2, None, Elastic Net)", default = "None")
    parser.add_argument("--save_model", action="store_true", help="Save trained model and vectorizer after training")

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
