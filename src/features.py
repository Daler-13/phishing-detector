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
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_features(url: str) -> dict:
    url = str(url).strip()

    normalized_url = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        normalized_url
    ):
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

    # ---------------------------------------------------------
    # DOMAIN
    # ---------------------------------------------------------

    domain_parts = [
        part
        for part in hostname_lower.split(".")
        if part
    ]

    if is_ip_address(hostname_lower):
        subdomain_count = 0
    else:
        subdomain_count = max(
            len(domain_parts) - 2,
            0
        )

    if len(domain_parts) >= 2:
        registered_domain = ".".join(
            domain_parts[-2:]
        )
    else:
        registered_domain = hostname_lower

    if subdomain_count > 0:
        subdomain = ".".join(
            domain_parts[:-2]
        )
    else:
        subdomain = ""

    tld = domain_parts[-1] if domain_parts else ""

    # ---------------------------------------------------------
    # BASIC COUNTS
    # ---------------------------------------------------------

    digits = sum(
        char.isdigit()
        for char in url
    )

    letters = sum(
        char.isalpha()
        for char in url
    )

    url_length = len(url)

    digit_ratio = (
        digits / url_length
        if url_length
        else 0
    )

    letter_ratio = (
        letters / url_length
        if url_length
        else 0
    )

    # ---------------------------------------------------------
    # PATH / QUERY
    # ---------------------------------------------------------

    path_segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    query_params = [
        parameter
        for parameter in query.split("&")
        if parameter
    ]

    # ---------------------------------------------------------
    # HOSTNAME-SPECIFIC COUNTS
    # ---------------------------------------------------------

    hostname_digits = sum(
        char.isdigit()
        for char in hostname
    )

    hostname_hyphens = hostname.count("-")

    # ---------------------------------------------------------
    # SUSPICIOUS WORDS
    # ---------------------------------------------------------

    suspicious_word_count = sum(
        1
        for word in SUSPICIOUS_WORDS
        if word in url_lower
    )

    # ---------------------------------------------------------
    # ADDITIONAL FEATURES
    # ---------------------------------------------------------

    has_encoded_chars = int(
        bool(re.search(
            r"%[0-9a-fA-F]{2}",
            url
        ))
    )

    has_port = int(
        parsed.port is not None
        if parsed.hostname
        else False
    )

    has_double_slash_path = int(
        "//" in path
    )

    has_long_subdomain = int(
        len(subdomain) >= 20
    )

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    features = {

        # Basic URL features
        "url_length": url_length,
        "domain_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),

        # Character counts
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

        # Protocol / domain
        "has_https": int(
            parsed.scheme.lower() == "https"
        ),
        "has_www": int(
            hostname_lower.startswith("www.")
        ),
        "has_ip_address": is_ip_address(
            hostname_lower
        ),
        "num_subdomains": subdomain_count,
        "has_punycode": int(
            "xn--" in hostname_lower
        ),

        # Ratios
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,

        # Suspicious content
        "suspicious_word_count":
            suspicious_word_count,

        # Entropy
        "url_entropy":
            calculate_entropy(url_lower),

        # -----------------------------------------------------
        # NEW V2 FEATURES
        # -----------------------------------------------------

        "num_path_slashes":
            path.count("/"),

        "num_query_params":
            len(query_params),

        "path_segment_count":
            len(path_segments),

        "domain_entropy":
            calculate_entropy(hostname_lower),

        "path_entropy":
            calculate_entropy(path.lower()),

        "subdomain_length":
            len(subdomain),

        "registered_domain_length":
            len(registered_domain),

        "tld_length":
            len(tld),

        "has_port":
            has_port,

        "has_double_slash_path":
            has_double_slash_path,

        "has_encoded_chars":
            has_encoded_chars,

        "has_long_subdomain":
            has_long_subdomain,

        "hostname_digit_count":
            hostname_digits,

        "hostname_hyphen_count":
            hostname_hyphens,
    }

    return features