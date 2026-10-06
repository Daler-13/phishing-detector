import sqlite3
from datetime import datetime


DATABASE_PATH = "database/phishing_detector.db"


def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            probability REAL NOT NULL,
            risk TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    url,
    result,
    probability,
    risk
):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO predictions (
            url,
            result,
            probability,
            risk,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        url,
        result,
        probability,
        risk,
        created_at
    ))

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


def get_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            url,
            result,
            probability,
            risk,
            created_at
        FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    connection.close()

    return predictions


def get_statistics():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """)

    total = cursor.fetchone()["total"]


    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE result = 'Safe'
    """)

    safe = cursor.fetchone()["count"]


    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE result = 'Phishing'
    """)

    phishing = cursor.fetchone()["count"]


    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk = 'LOW'
    """)

    low = cursor.fetchone()["count"]


    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk = 'MEDIUM'
    """)

    medium = cursor.fetchone()["count"]


    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk = 'HIGH'
    """)

    high = cursor.fetchone()["count"]


    connection.close()


    phishing_percent = (
        phishing / total * 100
        if total > 0
        else 0
    )


    return {
        "total": total,
        "safe": safe,
        "phishing": phishing,
        "phishing_percent": phishing_percent,
        "low": low,
        "medium": medium,
        "high": high
    }