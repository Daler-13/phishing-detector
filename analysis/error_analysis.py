import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


# Пути
DATASET_PATH = "dataset/processed/features.csv"
OUTPUT_PATH = "analysis/misclassified_urls.csv"


print("================================")
print("PHISHING DETECTOR ERROR ANALYSIS")
print("================================")


# 1. Загружаем данные
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["label"])
y = df["label"]


# 2. Делим данные так же, как при обучении
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Создаём и обучаем модель
model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=100
)

print("\nОбучение модели...")
model.fit(X_train, y_train)


# 4. Получаем прогнозы
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)


# 5. Создаём таблицу результатов
results = X_test.copy()

results["actual_label"] = y_test.values
results["predicted_label"] = y_pred

results["probability_safe"] = y_probability[:, 1]
results["probability_phishing"] = y_probability[:, 0]


# 6. Определяем ошибки
false_positive = results[
    (results["actual_label"] == 1) &
    (results["predicted_label"] == 0)
]

false_negative = results[
    (results["actual_label"] == 0) &
    (results["predicted_label"] == 1)
]


# 7. Выводим статистику
print("\n================================")
print("РЕЗУЛЬТАТЫ")
print("================================")

print(f"Всего тестовых URL: {len(results)}")

print(f"False Positive: {len(false_positive)}")
print(f"False Negative: {len(false_negative)}")

print(
    f"Всего ошибок: "
    f"{len(false_positive) + len(false_negative)}"
)


# 8. Сохраняем ошибки
errors = results[
    (results["actual_label"] != results["predicted_label"])
].copy()

errors.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nФайл с ошибками сохранён:")
print(OUTPUT_PATH)


# 9. Показываем первые ошибки
print("\n================================")
print("ПРИМЕРЫ ОШИБОК")
print("================================")

if len(errors) == 0:
    print("Ошибок не найдено.")

else:
    print(
        errors[
            [
                "actual_label",
                "predicted_label",
                "probability_safe",
                "probability_phishing"
            ]
        ].head(10)
    )


print("\n================================")
print("АНАЛИЗ ЗАВЕРШЁН")
print("================================")