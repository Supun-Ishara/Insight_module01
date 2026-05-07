# app.py — No training data needed

import os
import sys
from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.rule_based_scorer import score_essay

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/score", methods=["POST"])
def score():
    try:
        essay_text = ""

        # Check uploaded file
        if "essay_file" in request.files:
            file = request.files["essay_file"]
            if file.filename != "":
                essay_text = file.read().decode("utf-8")

        # Check typed text
        if not essay_text:
            essay_text = request.form.get(
                "essay_text", ""
            ).strip()

        # Validate
        if not essay_text:
            return jsonify({
                "error": "රචනයක් ඇතුළත් කරන්න හෝ ගොනුවක් උඩුගත කරන්න."
            }), 400

        if len(essay_text.split()) < 10:
            return jsonify({
                "error": "රචනය ඉතා කෙටිය. අවම වශයෙන් වචන 10ක් ලියන්න."
            }), 400

        # Score the essay
        scores, features, hints = score_essay(essay_text)

        weakest = min(scores, key=scores.get)
        total   = round(sum(scores.values()), 1)

        return jsonify({
            "scores"    : scores,
            "hints"     : hints,
            "weakest"   : weakest,
            "total"     : total,
            "word_count": features["total_words"],
            "features"  : {
                "named_entities"   : features["named_entity_count"],
                "discourse_markers": features["discourse_marker_count"],
                "ttr"              : features["ttr"],
                "sentences"        : features["sentence_count"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)