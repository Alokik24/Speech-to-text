from jiwer import wer, cer

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def evaluate(reference_path, predicted_path):
    reference = load_text(reference_path)
    predicted = load_text(predicted_path)

    print("=== Evaluation Results ===")
    print(f"WER (Word Error Rate): {wer(reference, predicted):.3f}")
    print(f"CER (Character Error Rate): {cer(reference, predicted):.3f}")

if __name__ == "__main__":
    # Replace with your file paths
    reference_file = "reference.txt"
    predicted_file = "transcript.txt"
    evaluate(reference_file, predicted_file)
