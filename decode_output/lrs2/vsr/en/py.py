import json
import csv
from collections import defaultdict
from difflib import SequenceMatcher

# Load JSON
with open("hypo-685605.json", "r") as file:
    data = json.load(file)

# Dictionary to count mispredictions
error_counts = defaultdict(lambda: defaultdict(int))

# Process each ref-hyp pair
for ref_sentence, hyp_sentence in zip(data["ref"], data["hypo"]):
    ref_words = ref_sentence.split()
    hyp_words = hyp_sentence.split()

    # Align words using SequenceMatcher
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":  # Word substitution error
            for ref_word, hyp_word in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                error_counts[ref_word][hyp_word] += 1
        elif tag == "insert":  # Extra words in prediction
            for hyp_word in hyp_words[j1:j2]:
                error_counts["<MISSING>"][hyp_word] += 1
        elif tag == "delete":  # Missing words in prediction
            for ref_word in ref_words[i1:i2]:
                error_counts[ref_word]["<MISSING>"] += 1

# Sort data by total errors
sorted_errors = sorted(
    [(ref_word, sum(predictions.values()), predictions) for ref_word, predictions in error_counts.items()],
    key=lambda x: -x[1]  # Sort by total errors in descending order
)

# Write to CSV
with open("word_error_counts_sorted.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Reference Word", "Total Errors", "Most Common Mistakes (Predicted -> Count)"])

    for ref_word, total_errors, predictions in sorted_errors:
        common_mistakes = ", ".join(f"{hyp}->{count}" for hyp, count in sorted(predictions.items(), key=lambda x: -x[1])[:3])  # Top 3 mistakes
        writer.writerow([ref_word, total_errors, common_mistakes])

print("CSV file 'word_error_counts_sorted.csv' has been saved with sorted errors.")
