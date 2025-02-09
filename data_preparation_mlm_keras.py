import logging

from transformers import AutoTokenizer, TFAutoModelForMaskedLM

from utils_keras import prepare_tf_dataset

logging.basicConfig(level=logging.INFO)

# Tokenizer und Modell laden
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Funktion zur Vorbereitung der MLM-Daten im TensorFlow-Format
def prepare_data_mlm():
    dataset = prepare_tf_dataset(tokenizer_name=model_checkpoint)

    # Ausgabe zur Kontrolle
    logging.info(f"Anzahl der vorbereiteten Batches für MLM: {len(list(dataset))}")

    # Aufteilen in Trainings- und Validierungsdaten
    total_batches = len(list(dataset))
    train_size = int(0.8 * total_batches)

    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    logging.info(f"Trainingsdaten: {train_size} Batches")
    logging.info(f"Validierungsdaten: {total_batches - train_size} Batches")

    return {"train": train_dataset, "validation": validation_dataset}

# Test: MLM-Daten vorbereiten
if __name__ == "__main__":
    datasets = prepare_data_mlm()
    for batch in datasets["train"].take(1):
        print(batch)
