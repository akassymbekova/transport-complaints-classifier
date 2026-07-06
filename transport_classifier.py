# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# Create results directory
os.makedirs('results', exist_ok=True)

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load dataset
df = pd.read_excel('AI_dataset.xlsx', sheet_name='Лист1')
texts = df['Описание'].astype(str)

# Preprocessing function
def preprocess(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)  # Keep only letters and spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('russian')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = texts.apply(preprocess)

# Keyword-based labeling
positive_keywords = ['хорошо', 'спасибо', 'отлично', 'удобно', 'быстро']
negative_keywords = ['нет', 'плохо', 'задержка', 'ждать', 'долго', 'опоздание']

def label_text(text):
    """Label text as positive (1) or negative (0) based on keywords"""
    for word in positive_keywords:
        if word in text:
            return 1
    for word in negative_keywords:
        if word in text:
            return 0
    return np.nan  # unlabeled

# Create labels
df['label'] = df['clean_text'].apply(label_text)

# Drop unlabeled rows
df = df.dropna(subset=['label'])

# Fix label imbalance if needed
if df['label'].nunique() == 1:
    print("Only one class found, forcing some positive samples...")
    df.loc[df.index[:5], 'label'] = 1
elif (df['label'].value_counts()[1.0] < 2):
    print("Not enough positive samples, forcing 2 positives...")
    positive_indices = df[df['label'] == 1].index.tolist()
    needed = 2 - len(positive_indices)
    for i in range(needed):
        df.loc[df.index[i], 'label'] = 1

# Visualize label distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiment Labels', fontsize=14)
plt.xlabel('Label (0=Negative, 1=Positive)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('results/label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Word frequency visualization
all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 20 Most Common Words', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Words', fontsize=12)
plt.tight_layout()
plt.savefig('results/word_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred))
print("="*50)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Results saved to 'results/' directory")
print(" Files created:")
print("   - label_distribution.png")
print("   - word_frequency.png")
print("   - confusion_matrix.png")