from src.features import extract_features


def extract_features_v3(url: str) -> dict:
    """
    Экспериментальная версия извлечения признаков.
    Удаляет начальный www. перед расчётом признаков,
    чтобы проверить устойчивость модели к этому префиксу.
    """
    import re
    from urllib.parse import urlparse

    original_url = str(url).strip()

    normalized_url = original_url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized_url):
        normalized_url = "http://" + normalized_url

    parsed = urlparse(normalized_url)
    hostname = parsed.hostname or ""

    if hostname.lower().startswith("www."):
        normalized_hostname = hostname[4:]
        # Меняем hostname в URL, сохраняя остальную часть адреса
        normalized_url = normalized_url.replace(
            hostname, normalized_hostname, 1
        )

    features = extract_features(normalized_url)

    # URL length и прочие признаки структуры URL должны
    # вычисляться по нормализованному URL в этом эксперименте.
    return features

