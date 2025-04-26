# Transport Complaints Classifier

## Project Goal and  Project Overview

- Goal: Automatically detect if a transport review is positive (1) or negative (0).
- Dataset: AI_dataset.xlsx (Sheet: Лист1).
- Technologies Used:
Python pandas, scikit-learn, nltk, matplotlib, seaborn
- ML Model: Logistic Regression
- Evaluation Metrics: Classification Report, Confusion Matrix

---

## 🛠️ Tasks Completed
- Text preprocessing (cleaning, lemmatization, stopword removal)
- Automatic labeling of complaints (positive = 1, negative = 0)
- Feature extraction using TF-IDF vectorization
- Model training using Logistic Regression
- Model evaluation (classification report, confusion matrix visualization)

---

##  Dataset
- File used: `AI_dataset.xlsx`
- Source: Provided as part of the project
- Sheet Name: `Лист1`
- Contains text descriptions of public transport feedback in Russian.

---

## 📚 Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`

---

##  Visualizations
- Review distribution by sentiment (positive vs negative)
- Word frequency analysis
- Confusion matrix of model predictions

---

## Preprocessing Steps
- Lowercasing text
- Removing special characters and numbers
- Tokenization
- Stopword removal (using Russian stopwords)
- Lemmatization

---

## ⚙️ Model Training
- Text features extracted using **TF-IDF Vectorizer**
- Model: **Logistic Regression**
- Stratified train/test split to maintain label balance
- Handling imbalance by forcing a minimum number of positive samples if necessary

---

## Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🚀 How to Run

1. Install the required libraries by running this command in your terminal:

pip install pandas numpy matplotlib seaborn nltk scikit-learn openpyxl

2. Make sure you have the dataset file AI_dataset.xlsx placed in the same folder as your script task.py. Run the Python script by executing the following command:

python task.py

or, if you are on a Mac and using Python 3:

python3 task.py

3. What the script will do:
- Preprocess and clean the complaint review texts.
- Automatically label them as positive (1) or negative (0) based on keywords.
- Handle imbalance if there are too few positive samples.
- Split the data into training and testing sets using a stratified method.
- Train a Logistic Regression model on the data.
- Print the classification report showing accuracy, precision, recall, and F1-score.
- Visualize the distribution of labels (positive/negative) and the model's confusion matrix.

✅ Note:
If this is your first time running the script, nltk will automatically download some necessary resources (stopwords and wordnet). Please ensure you are connected to the Internet for that.

✅ If you face a problem with python, try running the script using python3.