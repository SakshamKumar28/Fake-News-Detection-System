# Fake News Detection System

A machine learning model that detects fake news based on article titles using TF-IDF vectorization and Linear Support Vector Classification.

## Overview

This project implements a fake news detection system that:
- Uses both word-level and character-level TF-IDF features
- Employs Linear SVC with balanced class weights
- Achieves classification of news articles as real (1) or fake (0)

## Dataset

The model uses the `FakeNewsNet.csv` dataset which contains:
- News article titles
- Source URLs
- Source domains
- Engagement metrics
- Binary classification labels (real/fake)

## Model Architecture

The pipeline consists of:
1. Text preprocessing and cleaning
2. Feature extraction using:
   - Word-level TF-IDF (1-3 grams)
   - Character-level TF-IDF (3-5 grams)
3. Linear SVC classifier with balanced class weights

## Project Structure

```
├── fake_news_model.py      # Main model implementation
├── FakeNewsNet.csv        # Dataset
└── fake_news_model.pkl    # Trained model (generated after running)
```

## Requirements

- pandas
- scikit-learn
- joblib

## Usage

1. Clone the repository:
```bash
git clone https://github.com/SakshamKumar28/Fake-News-Detection-System.git
```

2. Install dependencies:
```bash
pip install pandas scikit-learn joblib
```

3. Run the model:
```bash
python fake_news_model.py
```

The script will:
- Load and preprocess the data
- Train the model
- Save the trained model as `fake_news_model.pkl`

## Model Details

- Features: 10,000 word-level + 5,000 character-level TF-IDF features
- Training/Test split: 80%/20% (stratified)
- Model: LinearSVC with balanced class weights
- Handles missing values and duplicate entries
