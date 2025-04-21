# gentrification_project
AI for understanding the social, economic, and environmental factors that influence gentrification of a neighborhood.

# Part 1: Gentrification Prediction Analysis 

# Part 2: Sentiment Analysis 
### How to run 
1. navigate inside the sentiment_analysis directory from your terminal. 

2. install the environment: 
```
pip install -r requirements.txt
```

### About the Code
```
final_proj_sentiment/
│
├── main.py                      # Entry point for training and evaluation
├── analysis.ipynb              # Jupyter notebook for visualizing results
├── config.py                   # Configuration for paths and constants
├── requirements.txt            # Project dependencies
│
├── data/                       # Cleaned and raw tweet data (CSV format)
│
├── data_collection/            # Twitter scraping code using Twikit
│   └── ...                     # Collection scripts and helper functions for training data
│
├── sentiment_model/            # Trained sentiment models and checkpoint files
│   └── ...                     # Model weights, vectorizers, training metadata
│
├── utils/                      # Utility functions (e.g., preprocessing, plotting)
│   └── ...                     # Shared helper scripts
│
├── results/                    # Saved plots, logs, evaluation outputs
│   └── ...                     # Graphs, figures, confusion matrices, etc.
```


#### Important 
- pretrained models are contained within senitment_model/models. 
- figures and test analysis for paper can be found in analysis.ipynb. 
- model training entrypoint is from main.py. Model training can be run as follows: 
```
python main.py <model_name> --learning_rate --strength --num_iterations --regularization

```
