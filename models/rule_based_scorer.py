import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import extract_features

# Scoring Rules
def score_historical_accuracy(features):
    """
    Measures: named entity count + academic vocab
    Logic: More specific historical names/terms
           = more historically grounded essay
    """
    score = 1

    named = features["named_entity_count"]
    vocab = features["academic_vocab_count"]

    # Score based on named entities found
    if named >= 5:
        score = 5
    elif named >= 4:
        score = 4
    elif named >= 3:
        score = 3
    elif named >= 2:
        score = 2
    else:
        score = 1

    # Boost by academic vocabulary
    if vocab >= 5:
        score = min(5, score + 1)
    elif vocab >= 3:
        score = min(5, score + 0.5)

    return round(max(1, min(5, score)), 1)


def score_coherence(features):
    """
    Measures: discourse marker count + sentence count
    Logic: More logical connectors = better idea flow
    """
    score = 1

    markers  = features["discourse_marker_count"]
    sentences = features["sentence_count"]

    # Base score on discourse markers
    if markers >= 5:
        score = 5
    elif markers >= 4:
        score = 4
    elif markers >= 3:
        score = 3
    elif markers >= 2:
        score = 2
    elif markers >= 1:
        score = 1.5
    else:
        score = 1

    # Penalty for very few sentences
    if sentences < 3:
        score = max(1, score - 1)

    return round(max(1, min(5, score)), 1)


def score_vocabulary(features):
    """
    Measures: TTR + academic vocab density + total words
    Logic: Higher variety + academic terms = richer vocabulary
    """
    score = 1

    ttr     = features["ttr"]
    density = features["academic_vocab_density"]
    words   = features["total_words"]

    # Base score on Type-Token Ratio
    if ttr >= 0.80:
        score = 5
    elif ttr >= 0.70:
        score = 4
    elif ttr >= 0.55:
        score = 3
    elif ttr >= 0.40:
        score = 2
    else:
        score = 1

    # Boost by academic vocabulary density
    if density >= 0.08:
        score = min(5, score + 1)
    elif density >= 0.05:
        score = min(5, score + 0.5)

    # Penalty for very short essays
    if words < 30:
        score = max(1, score - 1)

    return round(max(1, min(5, score)), 1)


def score_structure(features):
    """
    Measures: sentence count + sentence length variety
              + avg sentence length
    Logic: More sentences + varied lengths = better structure
    """
    score = 1

    sentences = features["sentence_count"]
    variety   = features["sentence_length_variety"]
    avg_len   = features["avg_sentence_length"]

    # Base score on sentence count
    if sentences >= 8:
        score = 5
    elif sentences >= 6:
        score = 4
    elif sentences >= 4:
        score = 3
    elif sentences >= 2:
        score = 2
    else:
        score = 1

    # Boost by sentence length variety
    if variety >= 15:
        score = min(5, score + 1)
    elif variety >= 8:
        score = min(5, score + 0.5)

    # Penalty for very short average sentences
    if avg_len < 4:
        score = max(1, score - 1)

    return round(max(1, min(5, score)), 1)


def score_essay(essay_text):
    """
    Main function — scores an essay across all 4 dimensions.
    Returns scores dict + extracted features + feedback hints.
    """
    # Extract all features
    features = extract_features(essay_text)

    # Score each dimension
    scores = {
        "Historical Accuracy"  : score_historical_accuracy(features),
        "Coherence & Idea Flow": score_coherence(features),
        "Vocabulary Richness"  : score_vocabulary(features),
        "Structural Adherence" : score_structure(features)
    }

    # Generate basic hints per dimension
    hints = generate_hints(features, scores)

    return scores, features, hints


def generate_hints(features, scores):
    """
    Generate specific improvement hints
    based on what features are weak.
    """
    hints = {}

    # Historical Accuracy hints
    if scores["Historical Accuracy"] <= 2:
        hints["Historical Accuracy"] = (
            "ඔබගේ රචනයේ නිශ්චිත ඓතිහාසික නාම, "
            "දිනයන් සහ ස්ථාන නාම ඇතුළත් කරන්න. "
            "උදා: දේවානම්පියතිස්ස, අනුරාධපුරය, "
            "ක්‍රි.පූ. 247 ආදිය."
        )
    elif scores["Historical Accuracy"] <= 3:
        hints["Historical Accuracy"] = (
            "තවත් ඓතිහාසික විස්තර එකතු කරන්න. "
            "රාජවංශ නාම, ශිලා ලිපි, රාජකීය ගොඩනැගිලි "
            "පිළිබඳ සඳහන් කරන්න."
        )
    else:
        hints["Historical Accuracy"] = (
            "ඉතා හොඳයි! ඔබ ඓතිහාසික තොරතුරු "
            "නිවැරදිව ඇතුළත් කර ඇත."
        )

    # Coherence hints
    if scores["Coherence & Idea Flow"] <= 2:
        hints["Coherence & Idea Flow"] = (
            "ඔබගේ අදහස් අතර සම්බන්ධකාරක වචන "
            "භාවිතා කරන්න. "
            "උදා: එබැවින්, එහෙත්, එමෙන්ම, "
            "ඉන්පසු, එම නිසා ආදිය."
        )
    elif scores["Coherence & Idea Flow"] <= 3:
        hints["Coherence & Idea Flow"] = (
            "සිදුවීම් අතර හේතු-ඵල සම්බන්ධය "
            "පැහැදිලි කරන්න. "
            "කරුණු ලැයිස්තු කිරීම වෙනුවට "
            "ඒවා සම්බන්ධ කරන්න."
        )
    else:
        hints["Coherence & Idea Flow"] = (
            "ඉතා හොඳයි! ඔබගේ අදහස් "
            "තාර්කිකව සම්බන්ධ වී ඇත."
        )

    # Vocabulary hints
    if scores["Vocabulary Richness"] <= 2:
        hints["Vocabulary Richness"] = (
            "එකම වචන නැවත නැවත භාවිතා කිරීම "
            "අඩු කරන්න. "
            "ශාස්ත්‍රීය වචන භාවිතා කරන්න: "
            "ස්ථාපිත, නිර්මාණය, අනුග්‍රහය, "
            "ශිෂ්ටාචාරය ආදිය."
        )
    elif scores["Vocabulary Richness"] <= 3:
        hints["Vocabulary Richness"] = (
            "ශාස්ත්‍රීය ඉතිහාස පදාවලිය "
            "වැඩිපුර භාවිතා කරන්න. "
            "විවිධ ක්‍රියාපද යොදාගන්න."
        )
    else:
        hints["Vocabulary Richness"] = (
            "ඉතා හොඳයි! ඔබ පොහොසත් "
            "භාෂාවක් භාවිතා කර ඇත."
        )

    # Structure hints
    if scores["Structural Adherence"] <= 2:
        hints["Structural Adherence"] = (
            "ඔබගේ රචනය හඳුන්වාදීමක්, "
            "ප්‍රධාන කොටස් (අවම 3), සහ "
            "නිගමනයක් ඇතිව සකස් කරන්න."
        )
    elif scores["Structural Adherence"] <= 3:
        hints["Structural Adherence"] = (
            "ශරීර ඡේදවල එක් එක් ඡේදයේ "
            "එක් ප්‍රධාන කරුණක් පමණක් "
            "ඇතුළත් කරන්න."
        )
    else:
        hints["Structural Adherence"] = (
            "ඉතා හොඳයි! ඔබගේ රචනය "
            "හොඳ ව්‍යූහයක් ඇතිව ලියා ඇත."
        )

    return hints


# Test
if __name__ == "__main__":

    test_essays = [
        {
            "label": "Weak Essay",
            "text": """රජු ආවේය. හාමුදුරුවෝ ආවෝය.
            විහාරය හැදුවේය. ජනතාව සතුටු වූහ."""
        },
        {
            "label": "Medium Essay",
            "text": """දේවානම්පියතිස්ස රජතුමා ක්‍රි.පූ. 247 සිට
            රාජ්‍ය කළේය. මහින්ද හිමිගේ ධර්ම දේශනාවෙන්
            ආභාෂය ලද රජතුමා බෞද්ධ ධර්මය පිළිගත්තේය.
            එබැවින් මහාවිහාරය ස්ථාපිත කළේය."""
        },
        {
            "label": "Strong Essay",
            "text": """දේවානම්පියතිස්ස රජතුමා ක්‍රි.පූ. 247 සිට
            ක්‍රි.පූ. 207 දක්වා අනුරාධපුර රාජධානිය
            පාලනය කළේය. ඔහු තෙවන අශෝක රජුගේ
            සමකාලීනයෙකි. පළමුව, මහින්ද හිමිගේ
            ලංකා ගමනය ශ්‍රේෂ්ඨ ඓතිහාසික සිදුවීමකි.
            එම ධර්ම දේශනාවෙන් ආභාෂය ලද රජතුමා
            බෞද්ධ ධර්මය රාජ්‍ය ආගම ලෙස ස්ථාපිත කළේය.
            එමෙන්ම මහාවිහාරය, ථූපාරාමය නිර්මාණය
            කිරීමෙන් බෞද්ධ ශිෂ්ටාචාරය ශ්‍රේෂ්ඨ
            මට්ටමකට ළඟා කළේය. එබැවින් ඔහු
            ශ්‍රී ලංකා ඉතිහාසයේ ශ්‍රේෂ්ඨ රජෙකු
            ලෙස සැලකේ."""
        }
    ]

    for essay in test_essays:
        scores, features, hints = score_essay(essay["text"])

        print(f"\n{'='*55}")
        print(f"  {essay['label']}")
        print(f"{'='*55}")
        print(f"  Words     : {features['total_words']}")
        print(f"  Entities  : {features['named_entity_count']}")
        print(f"  Markers   : {features['discourse_marker_count']}")
        print(f"  TTR       : {features['ttr']}")
        print(f"{'─'*55}")

        total = 0
        for dim, score in scores.items():
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"  {dim:<25} {bar} {score}/5")
            total += score

        print(f"\n  Total: {total:.1f}/20")
        print(f"{'='*55}")