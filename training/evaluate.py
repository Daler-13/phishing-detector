import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)


DATASET_PATH = "dataset/processed/features.csv"
MODEL_PATH = "models/gradient_boosting.joblib"

RESULTS_DIR = "models/evaluation"


def main():

    print("=" * 70)
    print(" PHISHING DETECTOR — ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Загружаем данные
    # ---------------------------------------------------------

    print("\n[1/5] Загрузка датасета...")

    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Используем ТОЧНО такое же разбиение,
    # как при обучении моделей.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Тестовых объектов: {len(X_test)}")

    # ---------------------------------------------------------
    # 2. Загружаем Gradient Boosting
    # ---------------------------------------------------------

    print("\n[2/5] Загрузка лучшей модели...")

    model = joblib.load(MODEL_PATH)

    print("Модель:", MODEL_PATH)

    # ---------------------------------------------------------
    # 3. Предсказания
    # ---------------------------------------------------------

    print("\n[3/5] Выполнение предсказаний...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 4. Classification Report
    # ---------------------------------------------------------

    print("\n[4/5] Classification Report")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing",
            "Safe"
        ],
        digits=4
    )

    print(report)

    with open(
        f"{RESULTS_DIR}/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # ---------------------------------------------------------
    # 5. Confusion Matrix
    # ---------------------------------------------------------

    print("[5/5] Создание Confusion Matrix...")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Phishing",
            "Safe"
        ]
    )

    display.plot()

    plt.title(
        "Confusion Matrix — Gradient Boosting"
    )

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # ROC Curve
    # ---------------------------------------------------------

    print("\nСоздание ROC Curve...")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_proba
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"Gradient Boosting (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve — Gradient Boosting"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/roc_curve.png",
        dpi=300
    )

    plt.close()

    print("\n" + "=" * 70)
    print(" ГОТОВО!")
    print("=" * 70)

    print("\nФайлы сохранены в:")
    print(RESULTS_DIR)

    print("\nСозданы:")
    print("1. classification_report.txt")
    print("2. confusion_matrix.png")
    print("3. roc_curve.png")


if __name__ == "__main__":
    main()