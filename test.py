import pandas as pd
import argparse
import pickle
from lxml import etree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from train import (
    preprocess_text,
    extract_sentiment,
    train_word2vec,
    get_sentence_embeddings,
    combine_features
)


def load_single_part_to_df(part: str) -> pd.DataFrame:
    data = []
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


def predict_and_evaluate(part: str, model_path: str):
    # Load the model and feature names
    with open(model_path, 'rb') as f:
        model, feature_columns = pickle.load(f)

    df = load_single_part_to_df(part)
    # Retain the necessary columns for later use
    df_original = df[['sentenceText', 'polarity']].copy()

    # Extract features
    word2vec_model = train_word2vec(df, ['target', 'category'])
    sentence_embeddings, _ = get_sentence_embeddings(df, 'processedText')
    df, _ = combine_features(df, word2vec_model, sentence_embeddings)

    # Filter out rows where 'polarity' is not in ['negative', 'neutral', 'positive']
    valid_polarities = ['negative', 'neutral', 'positive']
    df = df[df['polarity'].isin(valid_polarities)]

    # Ensure the test data has the same feature columns in the same order
    df = df[feature_columns + ['polarity']]  # Ensure polarity is included
    X = df[feature_columns]
    y_true = df['polarity']

    predictions = model.predict(X)

    # Decode the predictions
    le = LabelEncoder()
    le.fit(['negative', 'neutral', 'positive'])
    decoded_predictions = le.inverse_transform(predictions)

    # Add predictions to the original DataFrame
    df_original = df_original[
        df_original['polarity'].isin(valid_polarities)]  # Ensure the original DataFrame is also filtered
    df_original['predicted_polarity'] = decoded_predictions

    # Evaluate the predictions
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, decoded_predictions)}")
    print(f"Recall: {recall_score(y_true, decoded_predictions, average='weighted', zero_division=0)}")
    print(f"Precision: {precision_score(y_true, decoded_predictions, average='weighted', zero_division=0)}")
    print(f"F1 Score: {f1_score(y_true, decoded_predictions, average='weighted', zero_division=0)}")
    print(classification_report(y_true, decoded_predictions, zero_division=0))

    return df_original[['sentenceText', 'polarity', 'predicted_polarity']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict polarities for a specific part using a saved model.")
    parser.add_argument('--part', type=str, required=True, help="Path to the XML part file, e.g., xml_parts/part1.xml")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the saved model file, e.g., models/RF_parts_1_2_3_4_allFeatures.pkl")

    args = parser.parse_args()

    predictions_df = predict_and_evaluate(args.part, args.model)
    # print(predictions_df)
