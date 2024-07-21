# <Reviews>
#     <Review rid="1004293">
#         <sentences>
#             <sentence id="1004293:3">
#                 <text>The food was lousy - too sweet or too salty and the portions tiny.</text>
#                 <Opinions>
#                     <Opinion target="food" category="FOOD#QUALITY" polarity="negative" from="4" to="8"/>
#                     <Opinion target="portions" category="FOOD#STYLE_OPTIONS" polarity="negative" from="52" to="60"/>
#                 </Opinions>
#             </sentence>
#             # More <sentence> ..... </sentence>
#         <sentences>
#     </Review>
#     # More <Review> ..... </Review>
# </Reviews>

import os
import re
import string
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
import argparse
from itertools import cycle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, roc_curve, auc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
import nltk
from scipy.sparse import hstack
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sentiment_analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text: str, lemmatize=True) -> str:
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    tokens = word_tokenize(text)
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)


def extract_sentiment(text: str) -> float:
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    return sentiment_score['compound']


def train_test_parts_to_df(parts: List[str]) -> pd.DataFrame:
    data = []

    for part in parts:
        tree = etree.parse(part)
        root = tree.getroot()

        part_number = part.split("part")[-1].split(".xml")[0]

        for review in root.findall('.//Review'):
            review_id = review.get('rid')
            for sentence in review.findall('.//sentence'):
                sentence_id = sentence.get('id')
                text = sentence.find('text').text
                processed_text = preprocess_text(text, lemmatize=False)
                sentiment_score = extract_sentiment(text)
                opinions = sentence.find('Opinions')
                if opinions is not None:
                    for opinion in opinions.findall('Opinion'):
                        target = preprocess_text(opinion.get('target'), lemmatize=False)
                        category = preprocess_text(opinion.get('category').replace('#', ' '), lemmatize=False)
                        polarity = opinion.get('polarity')

                        data.append({
                            'reviewID': review_id,
                            'sentenceID': sentence_id,
                            'sentenceText': text,
                            'processedText': processed_text,
                            'sentimentScore': sentiment_score,
                            'target': target,
                            'category': category,
                            'part': part_number,
                            'polarity': polarity,
                        })
                else:
                    data.append({
                        'reviewID': review_id,
                        'sentenceID': sentence_id,
                        'sentenceText': text,
                        'processedText': processed_text,
                        'sentimentScore': sentiment_score,
                        'target': "",
                        'category': "",
                        'part': part_number,
                        'polarity': "",
                    })

    df = pd.DataFrame(data)

    return df

def construct_all_dataset(parts: List[int]) -> pd.DataFrame:
    all_parts = set(range(1, 11))
    train_parts = parts
    test_parts = list(all_parts - set(train_parts))

    train_files = [f"xml_parts/part{part}.xml" for part in train_parts]
    test_files = [f"xml_parts/part{part}.xml" for part in test_parts]

    train_df = train_test_parts_to_df(parts=train_files)
    test_df = train_test_parts_to_df(parts=test_files)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    valid_polarities = ['negative', 'positive', 'neutral']
    # Create a boolean mask for rows with valid polarity values
    mask = combined_df['polarity'].isin(valid_polarities)
    combined_df = combined_df[mask]
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv("0_starting_data.csv", index=False)

    return combined_df


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    # Train Word2Vec model for 'target' and 'category'
    word2vec_model = train_word2vec(df, ['target', 'category'])
    # Get Sentence Transformer embeddings for 'processedText'
    sentence_embeddings, sentence_transformer_model = get_sentence_embeddings(df, 'processedText')
    # Combine all features
    df, final_features = combine_features(df, word2vec_model, sentence_embeddings)
    df.to_csv("1_combined_features_data.csv", index=False)

    return df, final_features, word2vec_model, sentence_transformer_model


def train_word2vec(df, columns):
    sentences = []
    for col in columns:
        sentences += [word_tokenize(text) for text in df[col].dropna().values]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model


def get_average_word2vec(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def get_sentence_embeddings(df, text_column='processedText'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[text_column].tolist(), show_progress_bar=True)
    return embeddings, model


def combine_features(df, word2vec_model, sentence_embeddings):
    # Fill NaN values in 'target' and 'category' with an empty string
    df['target'] = df['target'].fillna('')
    df['category'] = df['category'].fillna('')

    target_features = np.vstack(df['target'].apply(lambda x: get_average_word2vec(x, word2vec_model)).values)
    category_features = np.vstack(df['category'].apply(lambda x: get_average_word2vec(x, word2vec_model)).values)
    sentence_features = np.vstack(sentence_embeddings)

    # Create DataFrames for the features
    sentiment_df = df[['sentimentScore']].reset_index(drop=True)
    target_df = pd.DataFrame(target_features, columns=[f'target_word2vec_{i}' for i in range(target_features.shape[1])])
    category_df = pd.DataFrame(category_features,
                               columns=[f'category_word2vec_{i}' for i in range(category_features.shape[1])])
    sentence_df = pd.DataFrame(sentence_features,
                               columns=[f'sentence_embedding_{i}' for i in range(sentence_features.shape[1])])

    # Concatenate all feature DataFrames
    feature_df = pd.concat([sentiment_df, target_df, category_df, sentence_df], axis=1)
    # Remove the original 'sentimentScore' column from the DataFrame to avoid duplication
    df = df.drop(columns=['sentimentScore'])
    # Combine with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    # Combine all features into a single array
    final_features = feature_df.values

    return df, final_features


def plot_roc_curve(y_test, y_score, n_classes, class_names, algorithm):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {algorithm}')
    plt.legend(loc="lower right")
    plt.show()


def train_model(df, features, target, algorithm, model_save_path, train_parts):
    X = df[features]
    y = df[target]

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data into training and testing sets based on 'part' column
    train_df = df[df['part'].isin(train_parts)]
    test_df = df[~df['part'].isin(train_parts)]

    X_train = train_df[features]
    y_train = le.transform(train_df[target])
    X_test = test_df[features]
    y_test = le.transform(test_df[target])

    # Class Weighting: The models are trained with class_weight='balanced' to handle class imbalance.
    if algorithm == 'SVM':
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight='balanced'))
    elif algorithm == 'LR':
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    elif algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    else:
        raise ValueError("Unsupported algorithm. Choose from 'SVM', 'LR', 'RF'.")

    # Fit & Evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0)}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    plot_roc_curve(y_test, y_score, n_classes=len(le.classes_), class_names=le.classes_, algorithm=algorithm)

    # Save model and feature names to disk
    with open(model_save_path, 'wb') as f:
        pickle.dump((model, features), f)

    return model


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Process training and testing parts for classification.")
        parser.add_argument('--parts', type=str, required=False, default="1,2,3,4,5,6,7,8,9,10",
                            help="Comma separated values with parts from 1 to 10 for training, e.g., 1,4,5,6,7,8,9")
        parser.add_argument('--construct_all_dataset_features', action='store_true',
                            help="Boolean flag, that when set it runs all parts dataset construction and feature extraction in 2 .csvs")
        parser.add_argument('--algorithm', type=str, required=True, choices=['SVM', 'LR', 'RF'],
                            help="Algorithm to use for training: 'SVM' (Support Vector Machines), 'LR' (Logistic Regression), or 'RF' (Random Forest)")

        args = parser.parse_args()
        parts = [int(part) for part in args.parts.split(",")]

        if args.construct_all_dataset_features:
            # Construct the dataset
            combined_df = construct_all_dataset(parts)
            print(f"Dataset constructed with shape: {combined_df.shape}")
            # Extract features
            combined_df, final_features, word2vec_model, sentence_transformer_model = extract_all_features(combined_df)
            print(f"Features extracted with shape: {combined_df.shape}")
        else:
            # Load the already constructed and feature-extracted dataset
            combined_df = pd.read_csv("1_combined_features_data.csv", dtype={'polarity': str, 'part': int}, low_memory=False)

        # Define the feature columns for training
        feature_columns = [col for col in combined_df.columns if 'word2vec' in col or 'embedding' in col or col == 'sentimentScore']

        # Train the model and save it to disk
        os.makedirs('models', exist_ok=True)
        model_save_path = f"models/{args.algorithm}_parts_{'_'.join(map(str, parts))}_allFeatures.pkl"
        trained_model = train_model(combined_df, feature_columns, 'polarity', args.algorithm, model_save_path, parts)
        print(f"Model trained and saved to {model_save_path}")