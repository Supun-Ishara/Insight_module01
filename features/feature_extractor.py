# feature_extractor.py
# Stage 1 - Extracts features from Sinhala essays

import re

# Discourse markers in Sinhala
DISCOURSE_MARKERS = [
    "එබැවින්", "එම නිසා", "එහෙත්", "නමුත්",
    "එමෙන්ම", "එසේම", "ඊට අමතරව", "පළමුව",
    "දෙවනුව", "තෙවනුව", "අවසානයේදී", "එම හේතුවෙන්",
    "ඒ අනුව", "ඒ නිසා", "එම නිසාම", "ඉන්පසු",
    "ඊට අනුව", "ඒ සමඟ", "එලෙසම", "කෙසේ වෙතත්"
]

# Academic/historical vocabulary
ACADEMIC_VOCAB = [
    "ස්ථාපිත", "නිර්මාණය", "අනුග්‍රහය", "රාජකීය",
    "සංස්කෘතික", "ශිෂ්ටාචාරය", "රාජධානිය", "ඓතිහාසික",
    "ශ්‍රේෂ්ඨ", "සංවර්ධනය", "ආක්‍රමණය", "රාජ්‍ය",
    "පාලනය", "ග්‍රන්ථය", "මූලාශ්‍රය", "රාජවංශ",
    "ශිෂ්ටාචාරය", "ධාර්මික", "සවිස්තරාත්මක",
    "කෘෂිකාර්මික", "සංඝරත්නය", "රචනා", "ආරාධනා"
]

# Named entities (historical) 
NAMED_ENTITIES = [
    "දේවානම්පියතිස්ස", "මහින්ද", "මහාවිහාරය",
    "අනුරාධපුරය", "පොළොන්නරුව", "මහාවංශය",
    "පරාක්‍රමබාහු", "විජය", "අශෝක", "මහානාම",
    "ථූපාරාමය", "දීපවංශය", "සීගිරිය", "මිහින්තලය",
    "ජේතවනාරාමය", "රුවන්වැලිසෑය", "අභයගිරිය"
]


def extract_features(essay_text):
    """
    Extract all features from a Sinhala essay text.
    Returns a dictionary of feature values.
    """

    features = {}

    # 1. Type Token Ratio
    words = essay_text.split()
    total_words = len(words)
    unique_words = len(set(words))

    if total_words > 0:
        features["ttr"] = round(unique_words / total_words, 3)
    else:
        features["ttr"] = 0.0

    features["total_words"] = total_words
    features["unique_words"] = unique_words

    # 2. Academic Vocabulary Density
    academic_count = sum(
        1 for word in ACADEMIC_VOCAB if word in essay_text
    )
    features["academic_vocab_count"] = academic_count
    features["academic_vocab_density"] = round(
        academic_count / max(total_words, 1), 3
    )

    # 3. Named Entity Count
    entity_count = sum(
        1 for entity in NAMED_ENTITIES if entity in essay_text
    )
    features["named_entity_count"] = entity_count

    # 4. Discourse Marker Count
    marker_count = sum(
        1 for marker in DISCOURSE_MARKERS if marker in essay_text
    )
    features["discourse_marker_count"] = marker_count

    # 5. Sentence Length Distribution 
    sentences = [
        s.strip() for s in re.split(r'[።\.!\?।]', essay_text)
        if s.strip()
    ]
    sentence_lengths = [len(s.split()) for s in sentences]

    features["sentence_count"] = len(sentences)

    if sentence_lengths:
        features["avg_sentence_length"] = round(
            sum(sentence_lengths) / len(sentence_lengths), 2
        )
        features["min_sentence_length"] = min(sentence_lengths)
        features["max_sentence_length"] = max(sentence_lengths)
        length_range = (
            features["max_sentence_length"] -
            features["min_sentence_length"]
        )
        features["sentence_length_variety"] = length_range
    else:
        features["avg_sentence_length"] = 0
        features["min_sentence_length"] = 0
        features["max_sentence_length"] = 0
        features["sentence_length_variety"] = 0

    return features


def print_features(essay_id, features):
    """Nicely print extracted features."""
    print(f"\n{'='*50}")
    print(f"  Essay {essay_id} - Extracted Features")
    print(f"{'='*50}")
    print(f"  Total Words          : {features['total_words']}")
    print(f"  Unique Words         : {features['unique_words']}")
    print(f"  Type-Token Ratio     : {features['ttr']}")
    print(f"  Academic Vocab Count : {features['academic_vocab_count']}")
    print(f"  Academic Vocab Density:{features['academic_vocab_density']}")
    print(f"  Named Entities Found : {features['named_entity_count']}")
    print(f"  Discourse Markers    : {features['discourse_marker_count']}")
    print(f"  Sentence Count       : {features['sentence_count']}")
    print(f"  Avg Sentence Length  : {features['avg_sentence_length']}")
    print(f"  Sentence Variety     : {features['sentence_length_variety']}")
    print(f"{'='*50}")


# Test it
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.synthetic_essays import essays

    for essay in essays[:3]:
        features = extract_features(essay["text"])
        print_features(essay["id"], features)