import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import os
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pfad zum NSP-Modell
NSP_PATH = "./bookcorpus_nsp_model"

# Überprüfung, ob das Modell existiert
if not os.path.exists(NSP_PATH):
    raise FileNotFoundError("Das gespeicherte NSP-Modell wurde nicht gefunden.")

# Modell und Tokenizer laden
logger.info("Lade gespeichertes NSP-Modell und Tokenizer...")
try:
    nsp_model = TFAutoModelForSequenceClassification.from_pretrained(NSP_PATH)
    tokenizer = AutoTokenizer.from_pretrained(NSP_PATH)
    logger.info("Modell und Tokenizer erfolgreich geladen.")
except Exception as e:
    logger.error(f"Fehler beim Laden des Modells: {e}")
    exit(1)

# Funktion für Next Sentence Prediction (NSP)
def evaluate_nsp(sentence1: str, sentence2: str):
    inputs = tokenizer(sentence1, sentence2, return_tensors="tf")
    outputs = nsp_model(**inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)

    next_sentence_prob = probabilities[0][1].numpy()
    not_next_sentence_prob = probabilities[0][0].numpy()

    nsp_prediction = "Next sentence" if next_sentence_prob > not_next_sentence_prob else "Not next sentence"
    return nsp_prediction, next_sentence_prob, not_next_sentence_prob

# Beispieltexte für Next Sentence Prediction (NSP) mit Labels
nsp_examples = [
    ("To be, or not to be,", "that is the question:", 1),
    ("The sun rises in the east,", "and sets in the west.", 1),
    ("Friends, Romans, countrymen, lend me your ears!", "I come to bury Caesar, not to praise him.", 1),
    ("O Romeo, Romeo!", "Wherefore art thou Romeo?", 1),
    ("In the beginning,", "God created the heavens and the earth.", 1),
    ("The quick brown fox", "jumps over the lazy dog.", 1),
    ("Once upon a time,", "there was a little girl named Red Riding Hood.", 1),
    ("He opened the old book,", "and dust flew into the air.", 1),
    ("The storm was fierce,", "but the sailors held their course.", 1),
    ("She knocked on the door,", "and waited for a response.", 1),

    ("To be, or not to be,", "The cat ran across the street.", 0),
    ("The sun rises in the east,", "Bananas are rich in potassium.", 0),
    ("Friends, Romans, countrymen, lend me your ears!", "It’s going to rain tomorrow.", 0),
    ("O Romeo, Romeo!", "The price of oil has dropped significantly.", 0),
    ("In the beginning,", "The concert was sold out in minutes.", 0),
    ("The quick brown fox", "She loves to paint in her free time.", 0),
    ("Once upon a time,", "The temperature in Antarctica is freezing.", 0),
    ("He opened the old book,", "They decided to buy a new car.", 0),
    ("The storm was fierce,", "Mathematics is a fundamental subject in school.", 0),
    ("She knocked on the door,", "The stock market closed higher today.", 0),
]

# Funktion zur Berechnung der NSP-Genauigkeit mit Labels
def evaluate_nsp_accuracy(examples, model, tokenizer):
    correct = 0
    total = len(examples)

    for s1, s2, label in examples:
        inputs = tokenizer(s1, s2, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = tf.argmax(logits, axis=-1).numpy()[0]

        if prediction == label:
            correct += 1

    accuracy = correct / total
    return accuracy

# NSP-Ergebnisse
logger.info("\n=== Next Sentence Prediction (NSP) Ergebnisse ===")
for s1, s2, label in nsp_examples:
    try:
        prediction, next_prob, not_next_prob = evaluate_nsp(s1, s2)
        logger.info(f"\nInput:\nSatz 1: {s1}\nSatz 2: {s2}")
        logger.info(f" - Vorhersage: {prediction}")
        logger.info(f" - Wahrscheinlichkeiten: Next: {next_prob:.4f}, Not Next: {not_next_prob:.4f}")
    except Exception as e:
        logger.error(f"Fehler für Eingabe ('{s1}', '{s2}'): {e}")

# NSP-Genauigkeit testen
accuracy = evaluate_nsp_accuracy(nsp_examples, nsp_model, tokenizer)
logger.info(f"\nNSP Genauigkeit: {accuracy:.2%}")
