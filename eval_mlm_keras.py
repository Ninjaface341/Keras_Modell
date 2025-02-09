import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import os
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pfad zum MLM-Modell
MLM_PATH = "./bookcorpus_mlm_model_keras"

# Überprüfung, ob das Modell existiert
if not os.path.exists(MLM_PATH):
    raise FileNotFoundError("Das gespeicherte MLM-Modell wurde nicht gefunden.")

# Modell und Tokenizer laden
logger.info("Lade gespeichertes MLM-Modell und Tokenizer...")
try:
    mlm_model = TFAutoModelForMaskedLM.from_pretrained(MLM_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MLM_PATH)
    logger.info("Modell und Tokenizer erfolgreich geladen.")
except Exception as e:
    logger.error(f"Fehler beim Laden des Modells: {e}")
    exit(1)

# Funktion für Masked Language Modeling (MLM)
def evaluate_mlm(input_text: str, top_k=5):
    inputs = tokenizer(input_text, return_tensors="tf")
    mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)

    if tf.size(mask_token_index) == 0:
        raise ValueError("Kein [MASK] Token im Eingabetext gefunden.")

    outputs = mlm_model(**inputs)
    logits = outputs.logits
    softmax_logits = tf.nn.softmax(logits, axis=-1)

    predictions = []
    for idx in mask_token_index[:, 1].numpy():
        mask_token_logits = softmax_logits[0, idx, :]
        top_tokens = tf.math.top_k(mask_token_logits, k=top_k).indices.numpy().tolist()
        predictions.append([
            (tokenizer.decode([token]), float(mask_token_logits[token].numpy()))
            for token in top_tokens
        ])

    return predictions

# Funktion zur Berechnung der MLM-Genauigkeit (Top-5 Treffer zählen)
def evaluate_mlm_accuracy(examples_with_answers, top_k=5):
    correct_predictions = 0

    for example, correct_word in examples_with_answers:
        try:
            predictions = evaluate_mlm(example, top_k=top_k)[0]
            predicted_words = [word.strip().lower() for word, _ in predictions]

            if correct_word.lower() in predicted_words:
                correct_predictions += 1
        except ValueError as e:
            logger.error(f"Fehler für Eingabe '{example}': {e}")

    accuracy = correct_predictions / len(examples_with_answers)
    return accuracy

# Beispieltexte für MLM mit richtigen Antworten
mlm_examples_with_answers = [
    ("To be, or not to be, that is the [MASK]:", "question"),
    ("All the world's a [MASK], and all the men and women merely [MASK].", "stage"),
    ("Shall I compare thee to a [MASK]'s day?", "summer"),
    ("If [MASK] be the food of [MASK], play on.", "music"),
    ("O Romeo, Romeo! Wherefore art thou [MASK] Romeo?", "Romeo"),
    ("The lady doth protest too [MASK], methinks.", "much"),
    ("A horse! A horse! My [MASK] for a horse!", "kingdom"),
    ("Brevity is the soul of [MASK].", "wit"),
    ("She opened the door and saw the [MASK] shining brightly.", "sun"),
    ("It was a long journey, but finally they reached the [MASK].", "destination"),
    ("The cat jumped onto the [MASK] and knocked over the vase.", "table"),
    ("He couldn't believe his [MASK] when he saw the results.", "eyes"),
    ("They sat around the [MASK], sharing stories from their past.", "fire"),
    ("She always dreamed of visiting the [MASK] during summer.", "beach"),
    ("He picked up the [MASK] and started reading quietly.", "book"),
    ("After the storm, the sky turned a brilliant shade of [MASK].", "blue"),
    ("He whispered the secret into her [MASK], hoping no one else would hear.", "ear"),
    ("She found the hidden [MASK] under the old wooden floor.", "treasure"),
    ("The sound of the waves crashing against the [MASK] was calming.", "shore"),
    ("They packed their bags and left for the [MASK] early in the morning.", "airport"),
]

# MLM-Ergebnisse
logger.info("\n=== Masked Language Modeling (MLM) Ergebnisse ===")
for example, correct_word in mlm_examples_with_answers:
    try:
        predictions = evaluate_mlm(example)[0]
        logger.info(f"\nInput: {example}")
        for word, prob in predictions:
            logger.info(f" - {word}: {prob:.4f}")
        logger.info(f" - Richtige Antwort: {correct_word}")
    except ValueError as e:
        logger.error(f"Fehler für Eingabe '{example}': {e}")

# MLM-Genauigkeit testen
accuracy = evaluate_mlm_accuracy(mlm_examples_with_answers)
logger.info(f"\nMLM Genauigkeit: {accuracy:.2%}")
