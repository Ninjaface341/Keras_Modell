import logging
import random

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForNextSentencePrediction

from utils_keras import combine_datasets

logging.basicConfig(level=logging.INFO)

# Tokenizer und Modell für NSP laden
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForNextSentencePrediction.from_pretrained(model_checkpoint)

# Funktion zur Erstellung des NSP-Datensatzes
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []
    for i in range(len(sentences) - 1):
        # Positives Beispiel (tatsächliche Folge)
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "label": 0})

        # Negatives Beispiel (zufällige Sätze)
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "label": 1})

    random.shuffle(data)
    logging.info(f"NSP-Daten erstellt mit {len(data)} Beispielen.")
    return data

# Funktion zur Vorbereitung der NSP-Daten im TensorFlow-Format
def prepare_data_nsp(batch_size=32):
    combined_texts = combine_datasets()  # Kombiniertes Dataset aus BookCorpus & Shakespeare
    nsp_data = create_nsp_dataset(combined_texts)

    # Tokenisierung der NSP-Daten
    def tokenize_nsp(example):
        return tokenizer(
            example['sentence1'], example['sentence2'],
            truncation=True, padding='max_length', max_length=128, return_tensors='tf'
        )

    # Konvertieren in TensorFlow Dataset
    sentence1 = [example['sentence1'] for example in nsp_data]
    sentence2 = [example['sentence2'] for example in nsp_data]
    labels = [example['label'] for example in nsp_data]

    tokenized_inputs = tokenizer(sentence1, sentence2, truncation=True, padding='max_length', max_length=128, return_tensors='tf')
    dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels))

    # Shuffle und Batching
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Aufteilen in Trainings- und Validierungsdaten
    total_batches = len(list(dataset))
    train_size = int(0.8 * total_batches)

    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    logging.info(f"Trainingsdaten: {train_size} Batches")
    logging.info(f"Validierungsdaten: {total_batches - train_size} Batches")

    return {"train": train_dataset, "validation": validation_dataset}

# Test: NSP-Daten vorbereiten
if __name__ == "__main__":
    datasets = prepare_data_nsp()
    for batch in datasets["train"].take(1):
        print(batch)
