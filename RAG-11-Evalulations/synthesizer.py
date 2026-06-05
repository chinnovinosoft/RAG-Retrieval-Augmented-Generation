from deepeval.synthesizer import Synthesizer
import json

# Create synthesizer
synthesizer = Synthesizer()

# Generate goldens
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["labour.pdf"],
    include_expected_output=True
)

print(f"Generated {len(goldens)} goldens")

# Convert to JSON serializable format
dataset = []

for golden in goldens:
    dataset.append({
        "question": golden.input,
        "expected_answer": golden.expected_output
    })

# Save dataset
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("Dataset saved to dataset.json")