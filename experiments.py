import numpy as np
import argparse
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import construct_all_dataset, extract_all_features
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def train_model_for_experiment(df, features, target, algorithm, model_save_path, train_parts, k_best=None):
    # Make sure 'part' column is int
    df['part'] = df['part'].astype(int)
    # Encode target labels
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    # Split the data into training and testing sets based on 'part' column
    train_df = df[df['part'].isin(train_parts)]
    test_df = df[~df['part'].isin(train_parts)]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Feature selection
    selected_features = None
    if k_best:
        selector = SelectKBest(mutual_info_classif, k=k_best)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        selected_features = [features[i] for i in selector.get_support(indices=True)]

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

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    # print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    # plot_roc_curve(y_test, y_score, n_classes=len(le.classes_), class_names=le.classes_, algorithm=algorithm)

    # Save model and feature names to disk
    with open(model_save_path, 'wb') as f:
        pickle.dump((model, selected_features if k_best else features), f)

    return model, accuracy, precision, recall, f1, selected_features


def cross_validation_experiment(algorithm: str, k_best: int = None):
    parts = list(range(1, 11))
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(1, 11):
        print(f"===== Fold {i} =====")
        # Split parts into training and testing sets
        train_parts = [p for p in parts if p != i]
        test_parts = [i]

        print(f"Training parts: {train_parts}")
        print(f"Testing parts: {test_parts}")

        # Construct the dataset
        combined_df = construct_all_dataset(train_parts + test_parts)
        # Extract features
        combined_df, final_features, word2vec_model, sentence_transformer_model = extract_all_features(combined_df)
        # Define the feature columns for training
        feature_columns = [col for col in combined_df.columns if 'word2vec' in col or 'embedding' in col or col == 'sentimentScore']
        # Train the model and save it to disk
        if k_best:
            model_save_path = f"models/CV/{algorithm}_fold_{i}_kbest_{k_best}.pkl"
        else:
            model_save_path = f"models/CV/{algorithm}_fold_{i}_allFeatures.pkl"
        trained_model, accuracy, precision, recall, f1, selected_features = train_model_for_experiment(combined_df, feature_columns, 'polarity', algorithm, model_save_path, train_parts, k_best)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    print("================================ FINAL RESULTS ================================")
    print(f"Avg Accuracy: {avg_accuracy}")
    print(f"Avg Precision: {avg_precision}")
    print(f"Avg Recall: {avg_recall}")
    print(f"Avg F1: {avg_f1}")

    return accuracies, avg_accuracy, avg_precision, avg_recall, avg_f1, selected_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation experiments for classification.")
    parser.add_argument('--algorithm', type=str, required=True, choices=['SVM', 'LR', 'RF'],
                        help="Algorithm to use for training: 'SVM', 'LR' (Logistic Regression), or 'RF' (Random Forest)")
    parser.add_argument('--k_best', type=int, required=False, help="Number of top features to select. If not provided, all features are used.")

    args = parser.parse_args()
    algorithm = args.algorithm
    k_best = args.k_best

    accuracies, avg_accuracy, avg_precision, avg_recall, avg_f1, selected_features = cross_validation_experiment(algorithm, k_best)
    if selected_features:
        print("Selected Features:")
        print(selected_features)