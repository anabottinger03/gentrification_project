# AI for Understanding Gentrification
AI for understanding the social, economic, and environmental factors that influence gentrification of a neighborhood.

## Part 1: Gentrification Prediction Analysis 
### About the Code 

```
final_proj_prediction_analysis/
|
├── data_processing/          # data processing notebook and zip file containing all prediction data
├── fnn/                     # FNN training code
├── figs/                    # figs used in paper
├── random_forests/         # random forest training code
```
## Part 2: Sentiment Analysis 

### Important Notes

- **Pretrained Models**:  
  Pretrained sentiment models are stored in `sentiment_model/models/`.

- **Figures & Test Analysis**:  
  All visualizations and evaluation results used in the paper are available in `analysis.ipynb`.

- **Model Training Entry Point**:  
  Training begins from `main.py`.

  To train the model, run:
  ```
  python main.py <model_name> --learning_rate --strength --num_iterations --regularization
  ```
### Installing Dependencies 
1. navigate inside the `final_proj_sentiment/` directory from your terminal. 

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