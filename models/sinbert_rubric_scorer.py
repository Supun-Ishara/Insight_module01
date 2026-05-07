# sinbert_rubric_scorer.py
# Stage 2 - Rubric Scoring using SinBERT embeddings

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_essays import essays
from utils.sinbert_embedder import SinBERTEmbedder
from features.feature_extractor import extract_features

class SinBERTRubricScorer:
    """
    Rubric scorer that combines:
    1. SinBERT embeddings (deep Sinhala understanding)
    2. Hand-crafted features (TTR, discourse markers etc.)
    This hybrid approach gives better results
    than either alone - especially with small data.
    """
    DIMENSIONS = [
        "Historical Accuracy",
        "Coherence & Idea Flow",
        "Vocabulary Richness",
        "Structural Adherence"
    ]
    SCORE_KEYS = [
        "historical_accuracy",
        "coherence",
        "vocabulary",
        "structure"
    ]

    def __init__(self):
        self.embedder = SinBERTEmbedder()
        self.models   = {}
        self.scaler   = StandardScaler()
        self.is_trained = False

    def _get_combined_features(self, text):
        """
        Combine SinBERT embeddings + hand-crafted features
        into one feature vector.
        """
        # SinBERT embedding (768 dimensions)
        sinbert_features = self.embedder.get_embedding(text)

        # Hand-crafted features (9 dimensions)
        hc = extract_features(text)
        handcrafted = np.array([
            hc["ttr"],
            hc["academic_vocab_count"],
            hc["academic_vocab_density"],
            hc["named_entity_count"],
            hc["discourse_marker_count"],
            hc["sentence_count"],
            hc["avg_sentence_length"],
            hc["sentence_length_variety"],
            hc["total_words"]
        ])

        # Combine both into one vector
        combined = np.concatenate([sinbert_features, handcrafted])
        return combined

    def build_dataset(self):
        """
        Build feature matrix and target scores
        from synthetic essays.
        """
        print("\n Extracting features from essays...")
        print(" (SinBERT processes each essay)\n")

        X_list = []
        y_dict = {dim: [] for dim in self.DIMENSIONS}

        for i, essay in enumerate(essays):
            print(f" Processing essay {i+1}/{len(essays)}: "
                  f"{essay['topic'][:30]}")

            features = self._get_combined_features(essay["text"])
            X_list.append(features)

            for dim, key in zip(self.DIMENSIONS, self.SCORE_KEYS):
                y_dict[dim].append(essay["scores"][key])

        X = np.array(X_list)
        print(f"\n ✓ Feature matrix shape: {X.shape}")
        print(f"   ({X.shape[0]} essays x "
              f"{X.shape[1]} features per essay)")

        return X, y_dict

    def train(self, X, y_dict):
        """
        Train one model per rubric dimension.
        Uses Ridge regression (works well with
        high-dimensional embeddings + small data).
        """
        print("\n" + "="*55)
        print("   SINBERT RUBRIC SCORER — TRAINING")
        print("="*55)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        for dimension in self.DIMENSIONS:
            y = np.array(y_dict[dimension])

            # Ridge regression for high-dim embeddings
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)

            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=min(3, len(essays)),
                scoring="neg_mean_absolute_error"
            )
            mae = -cv_scores.mean()

            self.models[dimension] = model

            print(f"\n  Dimension : {dimension}")
            print(f"  CV MAE    : {mae:.3f}  "
                  f"(lower = better)")

        self.is_trained = True
        print("\n" + "="*55)
        print(" ✓ All 4 dimension models trained!")

    def predict(self, essay_text):
        """
        Predict rubric scores for a new essay.
        Returns dict of dimension → score.
        """
        if not self.is_trained:
            raise Exception("Model not trained yet!")

        features = self._get_combined_features(essay_text)
        features_scaled = self.scaler.transform(
            features.reshape(1, -1)
        )

        scores = {}
        for dimension in self.DIMENSIONS:
            raw = self.models[dimension].predict(
                features_scaled
            )[0]
            # Clamp between 1 and 5
            scores[dimension] = round(
                max(1.0, min(5.0, raw)), 1
            )

        return scores

    def save(self, path="models/sinbert_rubric_models.pkl"):
        """Save trained models to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "models" : self.models,
                "scaler" : self.scaler,
                "is_trained": self.is_trained
            }, f)
        print(f"\n  Models saved to {path}")

    def load(self, path="models/sinbert_rubric_models.pkl"):
        """Load trained models from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models     = data["models"]
        self.scaler     = data["scaler"]
        self.is_trained = data["is_trained"]
        print(f" Models loaded from {path}")


def print_scores(scores):
    """Print rubric scores nicely."""
    print("\n" + "="*55)
    print("   RUBRIC SCORES  (SinBERT Powered)")
    print("="*55)

    total = 0
    for dimension, score in scores.items():
        bar    = "█" * int(score) + "░" * (5 - int(score))
        status = (
            " Good"       if score >= 4 else
            " Average"    if score >= 3 else
            " Needs Work"
        )
        print(f"  {dimension:<25} {bar} {score}/5  {status}")
        total += score

    print(f"\n  Overall Score : {total:.1f} / 20")
    print("="*55)

    weakest = min(scores, key=scores.get)
    print(f"\n   Weakest Area : {weakest} "
          f"({scores[weakest]}/5)")
    print(f"  Focus on improving this first.\n")


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":

    scorer = SinBERTRubricScorer()

    MODEL_PATH = "models/sinbert_rubric_models.pkl"

    # Load if already trained, else train fresh
    if os.path.exists(MODEL_PATH):
        scorer.load(MODEL_PATH)
        print(" ✓ Using saved SinBERT models")
    else:
        X, y_dict = scorer.build_dataset()
        scorer.train(X, y_dict)
        scorer.save(MODEL_PATH)

    # Test with a sample essay
    print("\n" + "="*55)
    print("   TESTING WITH A SAMPLE ESSAY")
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

    print(f"\n Essay: {test_essay[:80]}...")
    scores = scorer.predict(test_essay)
    print_scores(scores)