# Cyberbullying-Detection-ML

## Overview

This repository hosts a machine learning project aimed at detecting cyberbullying in tweets. Built with Python, it leverages libraries like scikit-learn, XGBoost, and pandas to preprocess tweet data, extract features using TF-IDF, and train models including Logistic Regression, Decision Tree, Random Forest, XGBoost, and LinearSVC. The project classifies tweets as "cyberbullying" or "not cyberbullying".

## Requirements

- Python 3.x
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`
- Installation: Run `pip install pandas numpy scikit-learn xgboost imbalanced-learn`

## Dataset

- **Files**:
  - `cyberbullying_tweets.csv`
  - `kaggle_cyberbullying_tweets.csv`
- **Note**: Place these files in the project directory before running the script.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TharunKorivi/Cyberbullying-Detection-ML.git
   cd Cyberbullying-Detection-ML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with the listed libraries if needed.)

## Usage

- Run the main script to train models and evaluate results:
  ```bash
  python cyberbullying_detection.py
  ```
- **Output**: Displays accuracy, precision, recall, F1-score, confusion matrix, and identifies the best model based on F1-score.
