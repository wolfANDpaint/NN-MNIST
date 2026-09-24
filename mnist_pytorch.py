# MNIST - PyTorch + scikit-learn

# Bibliotheken
import matplotlib.pyplot as plt
import pandas as pd
import torch as tp
import numpy as np
import platform
import joblib
import random
import uuid
import json
import time
import sys
import os
from copy import deepcopy
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Einstellungen
SEED = 42
DATA_FILE = "mnist_data_60k.csv"
OUTPUT_DIR = "mnist_experiments"

# Modell
INPUT_SIZE = 784
HIDDEN_SIZE = 300
OUTPUT_SIZE = 10

# Training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EPOCHS = 10
BATCH_SIZE = 64

# Cross-Validation
CV_FOLDS = 10

# Train/Test-Split
TEST_SIZE = 0.20

# Verzeichnisse
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dateien
EXPERIMENT_ID = (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
LOG_JSONL = os.path.join(OUTPUT_DIR, "experiment_runs.jsonl")
CV_RESULTS_CSV = os.path.join(OUTPUT_DIR, "cross_validation_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "experiment_summary.json")
CONFIG_JSON = os.path.join(OUTPUT_DIR, "model_config.json")
CONFUSION_CSV = os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv")
CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "classification_report_test.csv")
LOSS_CSV = os.path.join(OUTPUT_DIR, "best_model_loss_history.csv")
CONFUSION_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
LOSS_PNG = os.path.join(OUTPUT_DIR, "best_model_loss.png")

# Neue Dateien für Modell und Scaler
MODEL_FILE = os.path.join(OUTPUT_DIR, "mnist_model_weights.pth")
SCALER_FILE = os.path.join(OUTPUT_DIR, "mnist_scaler.joblib")
TEST_IMAGE_PNG = os.path.join(OUTPUT_DIR, "random_test_image.png")

# Zufallszahlen / Device
np.random.seed(SEED)
random.seed(SEED)
tp.manual_seed(SEED)
if tp.cuda.is_available():
    tp.cuda.manual_seed_all(SEED)
device = tp.device("cuda" if tp.cuda.is_available() else "cpu")
START_TIME = time.time()

# Hilfsfunktionen
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=str)

def get_system_info():
    return {"timestamp": now_iso(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pytorch_version": tp.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": str(device),
            "cuda_available": tp.cuda.is_available(),
            "cuda_version": tp.version.cuda,
            "gpu_name": (tp.cuda.get_device_name(0)
                         if tp.cuda.is_available()
                         else None),
            "seed": SEED}

def calculate_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}

# Experiment Logger
class ExperimentLogger:
    def __init__(self):
        self.records = {}
        self.counter = 0
    def new_run(self, run_type, params):
        self.counter += 1
        run_id = (f"run_{self.counter:04d}")
        self.records[run_id] = {"experiment_id": EXPERIMENT_ID,
                                "run_id": run_id,
                                "run_type": run_type,
                                "start_time": now_iso(),
                                "status": "running",
                                "parameters": deepcopy(params),
                                "system": get_system_info()}
        return run_id
    def update(self, run_id, **kwargs):
        if run_id in self.records:
            self.records[run_id].update(kwargs)
    def finish(self, run_id, status="completed"):
        if run_id in self.records:
            self.records[run_id]["status"] = status
            self.records[run_id]["end_time"] = now_iso()
    def save(self, path):
        with open(path, "w", encoding="utf-8",) as file:
            for record in (self.records.values()):
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
logger = ExperimentLogger()

# Ausgabe Experimentinformationen
print()
print("MNIST EXPERIMENT")
print(f"CSV-Datei    : {DATA_FILE}")
print(f"Experiment-ID: {EXPERIMENT_ID}")
print(f"Device       : {device}")
print("Optimizer    : Adam")

# Prüfen, ob gespeichertes Modell vorhanden ist
model_exists = os.path.exists(MODEL_FILE)
scaler_exists = os.path.exists(SCALER_FILE)

# Abfrage Training / Laden
print()
print("TRAININGSMODUS")
if model_exists and scaler_exists:
    while True:
        answer = input("Soll ein neues Training durchgeführt werden? [j/n]: ").strip().lower()
        if answer in ("j", "ja", "y", "yes"):
            NEW_TRAINING = True
            break
        elif answer in ("n", "nein", "no"):
            NEW_TRAINING = False
            break
        else:
            print("Bitte 'j' oder 'n' eingeben.")
else:
    print("Kein vollständiges gespeichertes Modell gefunden.")
    print("Ein neues Training wird durchgeführt.")
    NEW_TRAINING = True

# Daten einlesen
df = pd.read_csv(DATA_FILE, header=None)
data = df.values

# Labels
y = data[:, 0].astype(np.int64)

# Pixel
X = data[:, 1:].astype(np.float32)
if X.shape[1] != INPUT_SIZE:
    raise ValueError(f"Erwartet wurden {INPUT_SIZE} Pixel, "
                     f"gefunden wurden {X.shape[1]}.")

# Datensatzinformationen
print()
print("DATENSATZ")
print(f"Datensätze : {len(X)}")
print(f"Pixel      : {X.shape[1]}")
print(f"Klassen    : {len(np.unique(y))}")
print(f"Klassen    : {np.unique(y)}")

# Train/Test-Split
X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y))
print()
print("DATENAUFTEILUNG")
print(f"Training: {len(X_train):6d} "
      f"({len(X_train) / len(X):.0%})")
print(f"Test:     {len(X_test):6d} "
      f"({len(X_test) / len(X):.0%})")

# Neuronales Netz
class NN(tp.nn.Module):
    def __init__(self, layer_input=INPUT_SIZE, layer_hidden=HIDDEN_SIZE, layer_output=OUTPUT_SIZE):
        super().__init__()
        self.flatten = tp.nn.Flatten()
        self.network = tp.nn.Sequential(
            tp.nn.Linear(layer_input, layer_hidden),
            tp.nn.ReLU(),
            tp.nn.Linear(layer_hidden, layer_hidden),
            tp.nn.ReLU(),
            tp.nn.Linear(layer_hidden, layer_hidden),
            tp.nn.ReLU(),
            tp.nn.Linear(layer_hidden, layer_output))
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# sklearn-kompatibler PyTorch-Wrapper
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 layer_input=INPUT_SIZE,
                 layer_hidden=HIDDEN_SIZE,
                 layer_output=OUTPUT_SIZE,
                 learning_rate=LEARNING_RATE,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 weight_decay=WEIGHT_DECAY,
                 random_state=SEED,):
        self.layer_input = layer_input
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
    def fit(self, X, y):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tp.manual_seed(self.random_state)
        if tp.cuda.is_available():
            tp.cuda.manual_seed_all(self.random_state)
        params = {"layer_input": self.layer_input,
                  "layer_hidden": self.layer_hidden,
                  "layer_output": self.layer_output,
                  "learning_rate": self.learning_rate,
                  "epochs": self.epochs,
                  "batch_size": self.batch_size,
                  "weight_decay": self.weight_decay,
                  "optimizer": "Adam",
                  "random_state": self.random_state}
        self.run_id_ = logger.new_run(run_type="cv_fit", params=params)
        start_time = time.time()
        X_tensor = tp.tensor(X, dtype=tp.float32)
        y_tensor = tp.tensor(y, dtype=tp.long)
        dataset = tp.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = tp.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_ = NN(layer_input=self.layer_input,
                         layer_hidden=self.layer_hidden,
                         layer_output=self.layer_output).to(device)
        self.loss_function_ = (tp.nn.CrossEntropyLoss())
        self.optimizer_ = tp.optim.Adam(self.model_.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
        self.classes_ = np.unique(y)
        self.loss_history_ = []
        self.train_accuracy_history_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_samples = 0
            correct = 0
            for (batch_data, batch_label) in loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                self.optimizer_.zero_grad()
                output = self.model_(batch_data)
                loss = self.loss_function_(output, batch_label)
                loss.backward()
                self.optimizer_.step()
                number_samples = (batch_data.size(0))
                total_loss += (loss.item() * number_samples)
                total_samples += (number_samples)
                correct += (output.argmax(dim=1) == batch_label).sum().item()
            average_loss = (total_loss / total_samples)
            train_accuracy = (correct / total_samples)
            self.loss_history_.append(float(average_loss))
            self.train_accuracy_history_.append(float(train_accuracy))
        train_predictions = self.predict(X)
        train_metrics = calculate_metrics(y, train_predictions)
        runtime = (time.time() - start_time)
        logger.update(self.run_id_,
                      training_samples=len(X),
                      training_metrics=train_metrics,
                      loss_history=self.loss_history_,
                      train_accuracy_history=(self.train_accuracy_history_),
                      runtime_seconds=runtime)
        return self
    def predict(self, X):
        X_tensor = tp.tensor(X, dtype=tp.float32)
        self.model_.eval()
        predictions = []
        with tp.no_grad():
            for start_index in range(0, len(X_tensor), self.batch_size):
                end_index = (start_index + self.batch_size)
                batch = X_tensor[start_index:end_index].to(device)
                output = self.model_(batch)
                prediction = (output.argmax(dim=1).cpu().numpy())
                predictions.extend(prediction)
        return np.asarray(predictions)

# Wrapper für geladenes Modell
class LoadedPyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, batch_size=BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        self.classes_ = np.arange(OUTPUT_SIZE)
    def fit(self, X, y):
        raise RuntimeError("Das geladene Modell darf nicht trainiert werden.")
    def predict(self, X):
        X_tensor = tp.tensor(X, dtype=tp.float32)
        self.model.eval()
        predictions = []
        with tp.no_grad():
            for start_index in range(0, len(X_tensor), self.batch_size):
                end_index = (start_index + self.batch_size)
                batch = X_tensor[start_index:end_index].to(device)
                output = self.model(batch)
                prediction = (output.argmax(dim=1).cpu().numpy())
                predictions.extend(prediction)
        return np.asarray(predictions)

# Variablen initialisieren
cv_results_df = None
cv_mean = np.nan
cv_std = np.nan
cv_mean_f1 = np.nan
cv_std_f1 = np.nan
cv_runtime = 0.0
final_runtime = 0.0
loss_history = []
train_accuracy_history = []

# NEUES TRAINING
if NEW_TRAINING:
    print()
    print("10-FOLD CROSS-VALIDATION")
    print(f"Folds         : {CV_FOLDS}")
    print("Optimizer     : Adam")
    print(f"Learning Rate : {LEARNING_RATE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch Size    : {BATCH_SIZE}")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_results = []
    cv_start = time.time()
    for fold, (train_indices, validation_indices) in enumerate(cv.split(X_train, y_train), start=1):
        print()
        print(f"Fold {fold}/{CV_FOLDS}")
        fold_start = time.time()
        X_fold_train = (X_train[train_indices])
        y_fold_train = (y_train[train_indices])
        X_fold_validation = (X_train[validation_indices])
        y_fold_validation = (y_train[validation_indices])
        fold_pipeline = Pipeline([("scaler", MinMaxScaler(feature_range=(0, 1)),),
                                  ("neural_network", PyTorchClassifier(
                                      layer_input=INPUT_SIZE,
                                      layer_hidden=HIDDEN_SIZE,
                                      layer_output=OUTPUT_SIZE,
                                      learning_rate=LEARNING_RATE,
                                      epochs=EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      weight_decay=WEIGHT_DECAY,
                                      random_state=(SEED + fold)))])
        fold_pipeline.fit(X_fold_train, y_fold_train)
        validation_predictions = (fold_pipeline.predict(X_fold_validation))
        validation_metrics = (calculate_metrics(y_fold_validation, validation_predictions))
        fold_runtime = (time.time() - fold_start)
        fold_result = {"fold": fold,
                       "training_samples": len(X_fold_train),
                       "validation_samples": len(X_fold_validation),
                       "accuracy": validation_metrics["accuracy"],
                       "precision_macro": validation_metrics["precision_macro"],
                       "recall_macro": validation_metrics["recall_macro"],
                       "f1_macro": validation_metrics["f1_macro"],
                       "precision_weighted": validation_metrics["precision_weighted"],
                       "recall_weighted": validation_metrics["recall_weighted"],
                       "f1_weighted": validation_metrics["f1_weighted"],
                       "runtime_seconds": fold_runtime}
        cv_results.append(fold_result)
        fold_classifier = (fold_pipeline.named_steps["neural_network"])
        if getattr(fold_classifier, "run_id_", None):
            logger.update(fold_classifier.run_id_,
                          fold=fold,
                          validation_samples=(len(X_fold_validation)),
                          validation_metrics=(validation_metrics),
                          validation_confusion_matrix=(confusion_matrix(y_fold_validation, validation_predictions).tolist()))
            logger.finish(fold_classifier.run_id_, status="completed")
        print(f"Validation Accuracy: "
              f"{validation_metrics['accuracy']:.4%}")
        print(f"Validation F1 Macro: "
              f"{validation_metrics['f1_macro']:.4%}")
        print(f"Laufzeit: "
            f"{fold_runtime:.2f} s")
    cv_runtime = (time.time() - cv_start)
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(CV_RESULTS_CSV, index=False)
    cv_mean = (cv_results_df["accuracy"].mean())
    cv_std = (cv_results_df["accuracy"].std())
    cv_mean_f1 = (cv_results_df["f1_macro"].mean())
    cv_std_f1 = (cv_results_df["f1_macro"].std())
    print()
    print("CROSS-VALIDATION ERGEBNIS")
    print(f"Accuracy Mittelwert : "
          f"{cv_mean:.4%}")
    print(f"Accuracy Stdabw.    : "
          f"{cv_std:.4%}")
    print(f"F1 Macro Mittelwert : "
          f"{cv_mean_f1:.4%}")
    print(f"F1 Macro Stdabw.    : "
          f"{cv_std_f1:.4%}")
    print(f"CV-Laufzeit         : "
          f"{cv_runtime:.2f} s")
    print()
    print(cv_results_df[["fold",
                         "accuracy",
                         "f1_macro",
                         "runtime_seconds"]].to_string(index=False))

    # Finales Modell
    print()
    print("FINALES MODELL")
    final_start = time.time()
    final_pipeline = Pipeline([("scaler", MinMaxScaler(feature_range=(0, 1)),),
                               ("neural_network", PyTorchClassifier(layer_input=INPUT_SIZE,
                                                                    layer_hidden=HIDDEN_SIZE,
                                                                    layer_output=OUTPUT_SIZE,
                                                                    learning_rate=LEARNING_RATE,
                                                                    epochs=EPOCHS,
                                                                    batch_size=BATCH_SIZE,
                                                                    weight_decay=WEIGHT_DECAY,
                                                                    random_state=SEED))])
    final_pipeline.fit(X_train, y_train)
    final_runtime = (time.time() - final_start)
    final_classifier = (final_pipeline.named_steps["neural_network"])
    checkpoint = {"model_state_dict": final_classifier.model_.state_dict(),
                  "input_size": INPUT_SIZE,
                  "hidden_size": HIDDEN_SIZE,
                  "output_size": OUTPUT_SIZE,
                  "optimizer": "Adam",
                  "learning_rate": LEARNING_RATE,
                  "weight_decay": WEIGHT_DECAY,
                  "epochs": EPOCHS,
                  "batch_size": BATCH_SIZE,
                  "random_state": SEED,
                  "saved_at": now_iso()}
    tp.save(checkpoint, MODEL_FILE)
    scaler = (final_pipeline.named_steps["scaler"])
    joblib.dump(scaler, SCALER_FILE)
    print()
    print("MODELL GESPEICHERT")
    print(f"Gewichte : {MODEL_FILE}")
    print(f"Scaler   : {SCALER_FILE}")
    loss_history = (final_classifier.loss_history_)
    train_accuracy_history = (final_classifier.train_accuracy_history_)
    if getattr(final_classifier, "run_id_", None):
        logger.update(final_classifier.run_id_,
                      run_type="final_model",
                      final_training_samples=(len(X_train)),
                      final_training_runtime_seconds=(final_runtime))
else:
    print()
    print("GESPEICHERTES MODELL LADEN")
    print(f"Gewichte : {MODEL_FILE}")
    print(f"Scaler   : {SCALER_FILE}")
    scaler = joblib.load(SCALER_FILE)
    checkpoint = tp.load(MODEL_FILE, map_location=device)
    loaded_model = NN(layer_input=checkpoint["input_size"],
                      layer_hidden=checkpoint["hidden_size"],
                      layer_output=checkpoint["output_size"],).to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    final_pipeline = Pipeline([("scaler", scaler), ("neural_network", LoadedPyTorchClassifier(model=loaded_model, batch_size=BATCH_SIZE))])
    final_classifier = (final_pipeline.named_steps["neural_network"])
    print()
    print("Modell erfolgreich geladen.")
    print(f"Gespeichert am: "
          f"{checkpoint.get('saved_at', 'unbekannt')}")
    print(f"Hidden Size: "
          f"{checkpoint.get('hidden_size', 'unbekannt')}")
    print(f"Epochs: "
          f"{checkpoint.get('epochs', 'unbekannt')}")
    if os.path.exists(CV_RESULTS_CSV):
        try:
            cv_results_df = pd.read_csv(CV_RESULTS_CSV)
            cv_mean = (cv_results_df["accuracy"].mean())
            cv_std = (cv_results_df["accuracy"].std())
            cv_mean_f1 = (cv_results_df["f1_macro"].mean())
            cv_std_f1 = (cv_results_df["f1_macro"].std())
        except Exception:
            cv_results_df = None

# Training-Performance
training_predictions = (final_pipeline.predict(X_train))
training_metrics = calculate_metrics(y_train, training_predictions)

# Test-Performance
test_start = time.time()
test_predictions = (final_pipeline.predict(X_test))
test_runtime = (time.time() - test_start)
test_metrics = calculate_metrics(y_test, test_predictions)

# Konfusionsmatrix
cm = confusion_matrix(y_test, test_predictions)
pd.DataFrame(cm, index=[f"true_{i}" for i in range(10)], columns=[f"pred_{i}" for i in range(10)]).to_csv(CONFUSION_CSV)

# Classification Report
report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(CLASSIFICATION_CSV)

# Loss-Historie speichern
if len(loss_history) > 0:
    loss_df = pd.DataFrame({"epoch": np.arange(1, len(loss_history) + 1), "loss": loss_history, "training_accuracy": train_accuracy_history})
    loss_df.to_csv(LOSS_CSV, index=False)
else:
    loss_df = None

# Konfusionsmatrix visualisieren
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Konfusionsmatrix - Testdaten")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(CONFUSION_PNG, dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Loss visualisieren
if loss_df is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_df["epoch"], loss_df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss - finales Modell")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# Zufälliges Testbild
index = np.random.randint(0, len(X_test))
random_image = X_test[index]
random_label = y_test[index]
random_prediction = (final_pipeline.predict(random_image.reshape(1, -1))[0])
print()
print("ZUFÄLLIGES TESTBILD")
print(f"Index:      {index}")
print(f"Vorhersage: {random_prediction}")
print(f"Label:      {random_label}")
plt.figure(figsize=(4, 4))
plt.imshow(random_image.reshape(28, 28), cmap="gray")
plt.title(f"Vorhersage: {random_prediction}\n"
          f"Label: {random_label}")
plt.axis("off")
plt.tight_layout()
plt.savefig(TEST_IMAGE_PNG, dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Konfiguration speichern
config = {"experiment_id": EXPERIMENT_ID,
          "model": {"input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "output_size": OUTPUT_SIZE},
          "training": {"optimizer": "Adam",
                       "learning_rate": LEARNING_RATE,
                       "weight_decay": WEIGHT_DECAY,
                       "epochs": EPOCHS,
                       "batch_size": BATCH_SIZE,
                       "random_state": SEED},
          "cross_validation": {"method": "StratifiedKFold",
                               "folds": CV_FOLDS,
                               "shuffle": True,
                               "random_state": SEED},
          "files": {"model": MODEL_FILE,
                    "scaler": SCALER_FILE,
                    "test_image": TEST_IMAGE_PNG}}
save_json(CONFIG_JSON, config)

# Logger für finales Modell aktualisieren
if getattr(final_classifier, "run_id_", None):
    logger.update(final_classifier.run_id_,
                  final_training_metrics=(training_metrics),
                  test_samples=len(X_test),
                  test_metrics=(test_metrics),
                  test_confusion_matrix=(cm.tolist()),
                  test_classification_report=(report),
                  test_inference_runtime_seconds=(test_runtime),
                  final_loss_history=(loss_history),
                  final_training_accuracy_history=(train_accuracy_history))
    logger.finish(final_classifier.run_id_, status="completed")

# Gesamtzusammenfassung
total_runtime = (time.time() - START_TIME)
summary = {"experiment_id": EXPERIMENT_ID,
           "timestamp": now_iso(),
           "data_file": DATA_FILE,
           "output_directory": OUTPUT_DIR,
           "new_training": NEW_TRAINING,
           "system": get_system_info(),
           "dataset": {"samples_total": len(X),
                       "samples_training": len(X_train),
                       "samples_test": len(X_test),
                       "features": X.shape[1],
                       "classes": int(len(np.unique(y))),
                       "class_values": np.unique(y).tolist(),
                       "train_test_ratio": "80/20"},
           "model": {"input_size": INPUT_SIZE,
                     "hidden_size": HIDDEN_SIZE,
                     "output_size": OUTPUT_SIZE,
                     "optimizer": "Adam",
                     "learning_rate": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY,
                     "epochs": EPOCHS,
                     "batch_size": BATCH_SIZE},
           "cross_validation": {"method": "StratifiedKFold",
                                "folds": CV_FOLDS,
                                "shuffle": True,
                                "random_state": SEED,
                                "mean_accuracy":
                                                None
                                                if np.isnan(cv_mean)
                                                else float(cv_mean),
                                "std_accuracy":
                                                None
                                                if np.isnan(cv_std)
                                                else float(cv_std),
                                "mean_f1_macro":
                                                None
                                                if np.isnan(cv_mean_f1)
                                                else float(cv_mean_f1),
                                "std_f1_macro":
                                                None
                                                if np.isnan(cv_std_f1)
                                                else float(cv_std_f1),
                                "runtime_seconds": cv_runtime},
           "final_model": {"training_metrics": training_metrics,
                           "test_metrics": test_metrics,
                           "training_runtime_seconds": final_runtime,
                           "test_inference_runtime_seconds": test_runtime,
                           "random_test_image": {"index":
                                                     int(index),
                                                 "label":
                                                     int(random_label),
                                                 "prediction":
                                                     int(random_prediction),
                                                 "correct":
                                                     bool(random_label == random_prediction)}},
           "total_runtime_seconds": total_runtime,
           "files": {"jsonl_log": LOG_JSONL,
                     "model": MODEL_FILE,
                     "scaler": SCALER_FILE,
                     "cross_validation_results": CV_RESULTS_CSV,
                     "summary": SUMMARY_JSON,
                     "config": CONFIG_JSON,
                     "confusion_matrix_csv": CONFUSION_CSV,
                     "classification_report_csv": CLASSIFICATION_CSV,
                     "loss_history_csv": LOSS_CSV,
                     "confusion_matrix_png": CONFUSION_PNG,
                     "loss_png": LOSS_PNG,
                     "random_test_image": TEST_IMAGE_PNG}}
save_json(SUMMARY_JSON, summary)

# Logbuch speichern
logger.save(LOG_JSONL)

# Ausgabe
print()
print("ERGEBNISSE")
print()
print("MODELL")
print("  Optimizer            : Adam")
print(f"  Learning Rate        : "
      f"{LEARNING_RATE}")
print(f"  Hidden Size          : "
      f"{HIDDEN_SIZE}")
print(f"  Epochs               : "
      f"{EPOCHS}")
print(f"  Batch Size           : "
      f"{BATCH_SIZE}")
print()
print(f"  Neues Training       : "
      f"{'Ja' if NEW_TRAINING else 'Nein'}")
print()
print("10-FOLD CROSS-VALIDATION")
if not np.isnan(cv_mean):
    print(f"  Accuracy Mittelwert  : {cv_mean:.2%}")
    print(f"  Accuracy Stdabw.     : {cv_std:.2%}")
    print(f"  F1 Macro Mittelwert  : {cv_mean_f1:.2%}")
    print(f"  F1 Macro Stdabw.     : {cv_std_f1:.2%}")
else:
    print("  Keine neue Cross-Validation durchgeführt.")
print()
print("TRAINING")
for metric, value in (training_metrics.items()):
    print(f"  {metric:22s}: {value:.2%}")
print()
print("TEST")
for metric, value in (test_metrics.items()):
    print(f"  {metric:22s}: {value:.2%}")
print()
print(f"CV-Laufzeit:       {cv_runtime:.2f} s")
print(f"Training-Laufzeit: {final_runtime:.2f} s")
print(f"Test-Inferenz:     {test_runtime:.4f} s")
print(f"Gesamtlaufzeit:    {total_runtime:.2f} s")
print()
print("ZUFÄLLIGES TESTBILD")
print(f"  Index              : {index}")
print(f"  Label              : {random_label}")
print(f"  Vorhersage         : {random_prediction}")
print(f"  Ergebnis           : {'RICHTIG' if random_label == random_prediction else 'FALSCH'}")
print()
print("LOGBUCH / AUSGABEDATEIEN")
print(f"Experiment-Log:       {LOG_JSONL}")
print(f"Modellgewichte:       {MODEL_FILE}")
print(f"Scaler:               {SCALER_FILE}")
print(f"CV-Ergebnisse:        {CV_RESULTS_CSV}")
print(f"Zusammenfassung:      {SUMMARY_JSON}")
print(f"Konfiguration:        {CONFIG_JSON}")
print(f"Konfusionsmatrix:     {CONFUSION_CSV}")
print(f"Classification:       {CLASSIFICATION_CSV}")
print(f"Loss-Historie:        {LOSS_CSV}")
print(f"Konfusionsmatrix PNG: {CONFUSION_PNG}")
print(f"Loss-Plot PNG:        {LOSS_PNG}")
print(f"Zufälliges Testbild:  {TEST_IMAGE_PNG}")
print()
print("Programm beendet.")