import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Data Collection
def scrap_customer_feedback(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    feedbacks = soup.find_all('div', class_='feedback')
    return [feedback.text for feedback in feedbacks]

# Preprocessing and Text Cleaning
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_and_tokenize(text):
    text = preprocess_text(text)
    tokens = tokenize_and_stem(text)
    return tokens

# Neural Network Model Development
def train_sentiment_analysis_model(X, y):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    max_sequence_length = max([len(sequence) for sequence in sequences])
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 128, input_length=max_sequence_length))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=10,
                        batch_size=32,
                        callbacks=[early_stopping])

    return model, tokenizer

def evaluate_model(model, X_test, y_test, tokenizer):
    sequences = tokenizer.texts_to_sequences(X_test)
    max_sequence_length = model.input.shape[1]
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length)
    y_pred = model.predict_classes(X_padded)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    return cm, cr

# Data Visualization and Reporting
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.show()

def report_classification_results(cr):
    print('Classification Report:')
    print(cr)

# Main function
def main():
    # Data Collection
    feedbacks = scrap_customer_feedback('https://example.com/feedback')

    # Preprocessing
    processed_feedbacks = [preprocess_and_tokenize(feedback) for feedback in feedbacks]
    stop_words = set(stopwords.words('english'))
    filtered_feedbacks = [[word for word in feedback if word not in stop_words] for feedback in processed_feedbacks]
    X = [' '.join(feedback) for feedback in filtered_feedbacks]
    y = [1 if 'positive' in feedback.lower() else 0 for feedback in feedbacks]

    # Neural Network Model Training
    model, tokenizer = train_sentiment_analysis_model(X, y)

    # Model Evaluation
    cm, cr = evaluate_model(model, X, y, tokenizer)
    classes = ['Negative', 'Positive']
    plot_confusion_matrix(cm, classes)
    report_classification_results(cr)

if __name__ == '__main__':
    main()