import math
import re
import ipaddress
from urllib.parse import urlparse


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "confirm",
    "confirmation",
    "password",
    "banking",
    "wallet",
    "payment",
    "bonus",
    "free",
    "gift",
    "crypto",
]


def calculate_entropy(text: str) -> float:
    """Вычисляет энтропию строки."""
    if not text:
        return 0.0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def is_ip_address(hostname: str) -> int:
    """Проверяет, является ли hostname IP-адресом."""
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_features(url: str) -> dict:
    """
    Извлекает признаки непосредственно из URL.

    Сайт не открывается и сетевые запросы не выполняются.
    """

    url = str(url).strip()

    # Для корректного разбора добавляем схему, если её нет.
    normalized_url = url

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized_url):
        normalized_url = "http://" + normalized_url

    try:
        parsed = urlparse(normalized_url)
        hostname = parsed.hostname or ""
    except ValueError:
        parsed = urlparse("")
        hostname = ""

    url_lower = url.lower()
    hostname_lower = hostname.lower()

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # Количество поддоменов
    domain_parts = [
        part for part in hostname_lower.split(".")
        if part
    ]

    if is_ip_address(hostname_lower):
        subdomain_count = 0
    else:
        # Например:
        # www.google.com -> 1
        # login.mail.google.com -> 2
        subdomain_count = max(len(domain_parts) - 2, 0)

    # Отношение цифр и букв к длине URL
    digits = sum(char.isdigit() for char in url)
    letters = sum(char.isalpha() for char in url)

    url_length = len(url)

    digit_ratio = digits / url_length if url_length else 0
    letter_ratio = letters / url_length if url_length else 0

    # Поиск подозрительных слов
    suspicious_word_count = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in url_lower
    )

    features = {
        # Основные размеры
        "url_length": url_length,
        "domain_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),

        # Символы
        "num_digits": digits,
        "num_letters": letters,
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_percent_symbols": url.count("%"),
        "num_exclamation_marks": url.count("!"),
        "num_tildes": url.count("~"),
        "num_colons": url.count(":"),
        "num_semicolons": url.count(";"),

        # URL / домен
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_www": int(hostname_lower.startswith("www.")),
        "has_ip_address": is_ip_address(hostname_lower),
        "num_subdomains": subdomain_count,
        "has_punycode": int("xn--" in hostname_lower),

        # Соотношения
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,

        # Подозрительные признаки
        "suspicious_word_count": suspicious_word_count,

        # Энтропия
        "url_entropy": calculate_entropy(url_lower),
    }

    return features


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login.php",
        "https://secure-login-account-verification.com/update",
    ]

    for test_url in test_urls:
        print("=" * 70)
        print("URL:", test_url)
        print()

        features = extract_features(test_url)

        for name, value in features.items():
            print(f"{name:25} : {value}")