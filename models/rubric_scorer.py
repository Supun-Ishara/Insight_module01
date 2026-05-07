# rubric_scorer.py
# Stage 2 - Trains and runs the Rubric Scoring Model

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_essays import essays
from features.feature_extractor import extract_features


def build_dataset():
    rows = []
    for essay in essays:
        features = extract_features(essay["text"])
        row = {
            "ttr"                    : features["ttr"],
            "academic_vocab_count"   : features["academic_vocab_count"],
            "academic_vocab_density" : features["academic_vocab_density"],
            "named_entity_count"     : features["named_entity_count"],
            "discourse_marker_count" : features["discourse_marker_count"],
            "sentence_count"         : features["sentence_count"],
            "avg_sentence_length"    : features["avg_sentence_length"],
            "sentence_length_variety": features["sentence_length_variety"],
            "total_words"            : features["total_words"],
            "score_accuracy"         : essay["scores"]["historical_accuracy"],
            "score_coherence"        : essay["scores"]["coherence"],
            "score_vocabulary"       : essay["scores"]["vocabulary"],
            "score_structure"        : essay["scores"]["structure"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def train_models(df):
    feature_cols = [
        "ttr",
        "academic_vocab_count",
        "academic_vocab_density",
        "named_entity_count",
        "discourse_marker_count",
        "sentence_count",
        "avg_sentence_length",
        "sentence_length_variety",
        "total_words"
    ]

    target_cols = {
        "Historical Accuracy"  : "score_accuracy",
        "Coherence & Idea Flow": "score_coherence",
        "Vocabulary Richness"  : "score_vocabulary",
        "Structural Adherence" : "score_structure"
    }

    X = df[feature_cols]
    trained_models = {}

    print("\n" + "="*55)
    print("   RUBRIC SCORING MODEL - TRAINING RESULTS")
    print("="*55)

    for dimension, target in target_cols.items():
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        trained_models[dimension] = model

        print(f"\n  Dimension : {dimension}")
        print(f"  MAE Score : {mae:.3f}  (lower = better, max = 4.0)")

    print("\n" + "="*55)
    return trained_models, feature_cols


def predict_scores(essay_text, trained_models, feature_cols):
    features = extract_features(essay_text)

    feature_values = [[
        features["ttr"],
        features["academic_vocab_count"],
        features["academic_vocab_density"],
        features["named_entity_count"],
        features["discourse_marker_count"],
        features["sentence_count"],
        features["avg_sentence_length"],
        features["sentence_length_variety"],
        features["total_words"]
    ]]

    scores = {}
    for dimension, model in trained_models.items():
        raw_score = model.predict(feature_values)[0]
        clamped = max(1.0, min(5.0, raw_score))
        scores[dimension] = round(clamped, 1)

    return scores


def print_scores(scores):
    print("\n" + "="*55)
    print("   RUBRIC SCORES FOR YOUR ESSAY")
    print("="*55)

    total = 0
    for dimension, score in scores.items():
        bar = "█" * int(score) + "░" * (5 - int(score))
        status = (
            "✓ Good" if score >= 4 else
            "⚠ Average" if score >= 3 else
            "✗ Needs Work"
        )
        print(f"  {dimension:<25} {bar} {score}/5  {status}")
        total += score

    print(f"\n  Overall Score: {total:.1f} / 20")
    print("="*55)

    weakest = min(scores, key=scores.get)
    print(f"\n  Weakest Area : {weakest} ({scores[weakest]}/5)")
    print(f"  → Focus on improving this dimension first.\n")


def save_models(trained_models, feature_cols):
    os.makedirs("models", exist_ok=True)
    with open("models/rubric_models.pkl", "wb") as f:
        pickle.dump({
            "models": trained_models,
            "feature_cols": feature_cols
        }, f)
    print("  ✓ Models saved to models/rubric_models.pkl")


def load_models():
    with open("models/rubric_models.pkl", "rb") as f:
        data = pickle.load(f)
    return data["models"], data["feature_cols"]


if __name__ == "__main__":

    print("\n Building dataset from synthetic essays...")
    df = build_dataset()
    print(f" ✓ Dataset built: {len(df)} essays, {len(df.columns)} columns")
    print("\n Dataset Preview:")
    print(df.to_string())

    trained_models, feature_cols = train_models(df)

    save_models(trained_models, feature_cols)

    print("\n" + "="*55)
    print("   TESTING WITH A NEW SAMPLE ESSAY")
    print("="*55)

    test_essay = """
    දේවානම්පියතිස්ස රජතුමා ක්‍රි.පූ. 247 සිට
    ක්‍රි.පූ. 207 දක්වා රාජ්‍ය කළේය.
    මහින්ද හිමිගේ ලංකා ගමනය ශ්‍රේෂ්ඨ
    ඓතිහාසික සිදුවීමකි.
    එබැවින් රජතුමා මහාවිහාරය ස්ථාපිත කළේය.
    නමුත් ඔහුගේ රාජ්‍ය සමය පිළිබඳ
    වැඩිදුර පර්යේෂණ අවශ්‍ය වේ.
    """

    scores = predict_scores(test_essay, trained_models, feature_cols)
    print_scores(scores)