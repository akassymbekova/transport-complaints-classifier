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

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_excel('AI_dataset.xlsx', sheet_name='Лист1')

# Only keep the 'Описание' column
texts = df['Описание'].astype(str)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)  # Keep only letters and spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('russian')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = texts.apply(preprocess)

# Simple keyword-based labeling (simulate sentiment)
positive_keywords = ['хорошо', 'спасибо', 'отлично', 'удобно', 'быстро']
negative_keywords = ['нет', 'плохо', 'задержка', 'ждать', 'долго', 'опоздание']

def label_text(text):
    for word in positive_keywords:
        if word in text:
            return 1
    for word in negative_keywords:
        if word in text:
            return 0
    return np.nan  # unlabeled

# Create labels
df['label'] = df['clean_text'].apply(label_text)

# Drop rows where label is NaN (unlabeled)
df = df.dropna(subset=['label'])

# Check label distribution
print("Label distribution before fixing:")
print(df['label'].value_counts())

# Fix if needed
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
sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiment Labels')
plt.show()

# >>>>>>>>>>>> ADD WORD FREQUENCY VISUALIZATION <<<<<<<<<<<
all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(10,6))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 20 Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split dataset WITH stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train model
model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
