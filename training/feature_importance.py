import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "dataset/processed/features.csv"
MODEL_PATH = "models/gradient_boosting.joblib"
RESULTS_DIR = "models/evaluation"


def main():

    print("=" * 70)
    print(" АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Загружаем данные
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["label"])

    # Загружаем модель
    model = joblib.load(MODEL_PATH)

    # Получаем важность признаков
    importance = model.feature_importances_

    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nТОП-15 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
    print("=" * 70)

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    # Сохраняем таблицу
    importance_df.to_csv(
        f"{RESULTS_DIR}/feature_importance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # График
    # ---------------------------------------------------------

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Наиболее важные признаки для обнаружения фишинга"
    )

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\n" + "=" * 70)
    print(" ГОТОВО!")
    print("=" * 70)

    print("\nСохранено:")
    print(
        f"{RESULTS_DIR}/feature_importance.csv"
    )

    print(
        f"{RESULTS_DIR}/feature_importance.png"
    )


if __name__ == "__main__":
    main()