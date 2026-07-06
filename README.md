# Transport Complaints Classifier

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A machine learning system for automatic classification of public transport feedback as **positive (1)** or **negative (0)** using Natural Language Processing (NLP) techniques.

## 📋 Overview

This project implements a complete NLP pipeline to classify customer reviews about public transport services. It processes Russian-language text, extracts features using TF-IDF vectorization, and trains a Logistic Regression model for binary sentiment classification.

### Key Features

-  **Text Preprocessing**: Cleaning, tokenization, stopword removal, and lemmatization
-  **Russian Language Support**: Specialized handling for Russian text
-  **TF-IDF Vectorization**: Feature extraction from text data
-  **Logistic Regression**: Machine learning classification
-  **Visualizations**: Label distribution, word frequency, and confusion matrix
-  **Imbalance Handling**: Automatic handling of imbalanced datasets

## 🎯 Project Goal

Automatically detect if a transport review expresses:
- **Positive sentiment (1)**: Good service, thanks, comfortable, fast
- **Negative sentiment (0)**: Complaints, delays, bad service, waiting

## 📁 Project Structure

transport-complaints-classifier/
├── data/
│   └── AI_dataset.xlsx          # Dataset file
├── results/                     # Output directory (created automatically)
│   ├── confusion_matrix.png     # Model performance visualization
│   └── label_distribution.png   # Sentiment distribution
├── transport_classifier.py      # Main script
├── requirements.txt             # Dependencies
└── README.md                    # This file


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/akassymbekova/transport-complaints-classifier.git
cd transport-complaints-classifier

2. **Install dependencies**
pip install -r requirements.txt
```

### Dataset

File: AI_dataset.xlsx
Sheet: Лист1
Language: Russian
Content: Public transport feedback and complaints
Data Format

- The script automatically labels texts using keyword matching:

Positive keywords: хорошо, спасибо, отлично, удобно, быстро
Negative keywords: нет, плохо, задержка, ждать, долго, опоздание

### 🧠 Methodology

1. Text Preprocessing Pipeline

Raw Text → Lowercase → Remove Special Characters → Tokenization → Remove Stopwords → Lemmatization → Clean Text

2. Feature Extraction

Method: TF-IDF (Term Frequency-Inverse Document Frequency)
Purpose: Convert text data into numerical feature vectors

3. Model Training

Algorithm: Logistic Regression
Split: 80% training, 20% testing (stratified)
Random State: 42 (for reproducibility)

4. Evaluation Metrics

Accuracy: Overall correctness
Precision: Quality of positive predictions
Recall: Ability to find positive samples
F1-Score: Harmonic mean of precision and recall
Confusion Matrix: Detailed prediction breakdown

5. Visualizations

The script generates three visualizations:

Label Distribution (label_distribution.png)

Shows balance between positive and negative samples
Word Frequency (word_frequency.png)

Top 20 most frequent words in the dataset
Confusion Matrix (confusion_matrix.png)

Visual representation of model predictions vs actual labels

6. 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

7. Author: @akassymbekova