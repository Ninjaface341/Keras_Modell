import logging
import os

import tensorflow as tf
from data_preparation_mlm_keras import prepare_data_mlm
from transformers import (
    TFAutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Daten vorbereiten
        logger.info("Bereite MLM-Datasets vor...")
        mlm_datasets = prepare_data_mlm()

        if mlm_datasets is None:
            raise ValueError("Fehler beim Vorbereiten des MLM-Datensatzes.")

        # Modell und Tokenizer laden
        checkpoint = "./distilbert-base-cased"  # Erstes Training mit 'distilbert-base-cased', danach mit dem erstellten Modell
        logger.info(f"Lade Modell und Tokenizer von {checkpoint}...")
        model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

        # Modell kompilieren
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss)

        # Training starten
        logger.info("Starte Training des MLM-Modells...")
        model.fit(
            mlm_datasets["train"],
            validation_data=mlm_datasets["validation"],
            epochs=5,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(filepath="./bookcorpus_mlm_model", save_best_only=True)
            ]
        )

        # Modell speichern
        logger.info("Speichere Modell...")
        model.save_pretrained("./bookcorpus_mlm_model_keras")

        # Überprüfung des Speicherns
        if os.path.exists("./bookcorpus_mlm_model_keras"):
            print("Modell erfolgreich gespeichert.")
        else:
            raise FileNotFoundError("WARNUNG: Modell wurde nicht korrekt gespeichert!")

        # Modell-Validierung
        logger.info("Lade gespeichertes Modell und Tokenizer zur Validierung...")
        loaded_model = TFAutoModelForMaskedLM.from_pretrained("./bookcorpus_mlm_model_keras")
        loaded_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

        # Beispiel-Validierung: Test-Tokenisierung und Vorhersage
        test_text = "To be, or not to be, that is the [MASK]."
        inputs = loaded_tokenizer(test_text, return_tensors="tf")
        outputs = loaded_model(**inputs)
        predicted_index = tf.argmax(outputs.logits, axis=-1)
        predicted_token = loaded_tokenizer.decode(predicted_index[0].numpy())
        logger.info(f"Vorhersage abgeschlossen. Vorhergesagtes Wort: {predicted_token}")

    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")

if __name__ == "__main__":
    main()
