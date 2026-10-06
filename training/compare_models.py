import os
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


DATASET_PATH = "dataset/processed/features.csv"
MODEL_DIR = "models"


def main():

    print("=" * 70)
    print(" PHISHING DETECTOR — СРАВНЕНИЕ ML-МОДЕЛЕЙ")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Загрузка данных
    # ---------------------------------------------------------

    print("\n[1/6] Загрузка обработанного датасета...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Количество строк: {len(df)}")
    print(f"Количество столбцов: {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Разделение признаков и целевой переменной
    # ---------------------------------------------------------

    print("\n[2/6] Разделение признаков и целевого класса...")

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Признаков: {X.shape[1]}")
    print(f"Объектов: {X.shape[0]}")

    # ---------------------------------------------------------
    # 3. Train / Test
    # ---------------------------------------------------------

    print("\n[3/6] Разделение данных на train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Train: {len(X_train)}")
    print(f"Test:  {len(X_test)}")

    # ---------------------------------------------------------
    # 4. Создание моделей
    # ---------------------------------------------------------

    print("\n[4/6] Создание моделей...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=20
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100
        )
    }

    # ---------------------------------------------------------
    # 5. Обучение и оценка
    # ---------------------------------------------------------

    print("\n[5/6] Обучение моделей...")
    print()

    results = {}

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():

        print("-" * 70)
        print(f"Модель: {name}")

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Предсказания
        y_pred = model.predict(X_test)

        # Вероятности для ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_proba
        )

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "Training Time": training_time
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Время:     {training_time:.2f} сек.")

        # Сохраняем модель
        filename = name.lower().replace(" ", "_") + ".joblib"

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(model, model_path)

        print(f"Модель сохранена: {model_path}")

    # ---------------------------------------------------------
    # 6. Итоговая таблица
    # ---------------------------------------------------------

    print("\n[6/6] Формирование итоговой таблицы...")

    results_df = pd.DataFrame(results).T

    print()
    print("=" * 90)
    print(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 90)

    print(
        results_df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Сохраняем результаты
    results_df.to_csv(
        "models/model_comparison.csv"
    )

    # Определяем лучшую модель по F1
    best_model_name = results_df["F1"].idxmax()
    best_f1 = results_df.loc[
        best_model_name,
        "F1"
    ]

    print()
    print("=" * 70)
    print(" ЛУЧШАЯ МОДЕЛЬ")
    print("=" * 70)

    print(f"Модель: {best_model_name}")
    print(f"F1-score: {best_f1:.4f}")

    print()
    print("Результаты сохранены:")
    print("models/model_comparison.csv")

    print()
    print("Все модели сохранены в папке:")
    print("models/")


if __name__ == "__main__":
    main()