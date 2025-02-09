import tensorflow as tf
import logging
from transformers import TFAutoModelForNextSentencePrediction, AutoTokenizer
from data_preparation_nsp_keras import prepare_data_nsp

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Daten vorbereiten
        logger.info("Bereite NSP-Datasets vor...")
        nsp_datasets = prepare_data_nsp()

        if nsp_datasets is None:
            raise ValueError("Fehler beim Vorbereiten des NSP-Datensatzes.")

        # Modell und Tokenizer laden
        checkpoint = "bert-base-uncased"
        logger.info(f"Lade Modell und Tokenizer von {checkpoint}...")
        model = TFAutoModelForNextSentencePrediction.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

        # Modell kompilieren
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Training starten
        logger.info("Starte Training des NSP-Modells...")
        model.fit(
            nsp_datasets["train"],
            validation_data=nsp_datasets["validation"],
            epochs=5,
            batch_size=64,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(filepath="./bookcorpus_nsp_model", save_best_only=True)
            ]
        )

        # Modell speichern
        logger.info("Speichere Modell...")
        model.save_pretrained("./bookcorpus_nsp_model_keras")

        # Überprüfung des Speicherns
        if tf.io.gfile.exists("./bookcorpus_nsp_model_keras"):
            print("Modell erfolgreich gespeichert.")
        else:
            raise FileNotFoundError("WARNUNG: Modell wurde nicht korrekt gespeichert!")

    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")

if __name__ == "__main__":
    main()
