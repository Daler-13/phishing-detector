import os
import pandas as pd

from features import extract_features


RAW_DATASET = "dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED_DATASET = "dataset/processed/features.csv"


def prepare_dataset():
    print("=" * 70)
    print(" PHISHING DETECTOR — ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 70)

    print("\n[1/4] Загрузка исходного датасета...")

    df = pd.read_csv(RAW_DATASET)

    print(f"Загружено URL: {len(df)}")
    print(f"Исходных столбцов: {len(df.columns)}")

    # Проверяем необходимые столбцы
    if "URL" not in df.columns:
        raise ValueError("В датасете отсутствует столбец URL")

    if "label" not in df.columns:
        raise ValueError("В датасете отсутствует столбец label")

    print("\n[2/4] Извлечение признаков из URL...")

    features = []

    total = len(df)

    for index, url in enumerate(df["URL"]):
        features.append(extract_features(url))

        # Показываем прогресс каждые 10 000 URL
        if (index + 1) % 10000 == 0:
            percent = (index + 1) / total * 100
            print(
                f"Обработано: {index + 1}/{total} "
                f"({percent:.1f}%)"
            )

    features_df = pd.DataFrame(features)

    print("\nПризнаки успешно извлечены.")
    print(f"Количество признаков: {len(features_df.columns)}")

    print("\n[3/4] Добавление целевого признака...")

    features_df["label"] = df["label"].values

    print("Распределение классов:")
    print(features_df["label"].value_counts())

    print("\n[4/4] Сохранение обработанного датасета...")

    os.makedirs("dataset/processed", exist_ok=True)

    features_df.to_csv(
        PROCESSED_DATASET,
        index=False
    )

    print("\n" + "=" * 70)
    print(" ГОТОВО!")
    print("=" * 70)

    print(f"\nФайл сохранён:")
    print(PROCESSED_DATASET)

    print(f"\nРазмер:")
    print(features_df.shape)

    print("\nПервые 5 строк:")
    print(features_df.head())


if __name__ == "__main__":
    prepare_dataset()