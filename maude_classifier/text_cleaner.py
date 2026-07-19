# Vendored from mukundisb/maude-nlp-classifier (src/preprocessing/text_cleaner.py),
# trimmed to the inference-time cleaning path only.

import re

ABBREVIATION_MAP = {
    r"\bpt\b": "patient",
    r"\bpts\b": "patients",
    r"\bmd\b": "physician",
    r"\bdr\b": "doctor",
    r"\bhosp\b": "hospital",
    r"\badm\b": "admitted",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\brx\b": "prescription",
    r"\bs/p\b": "status post",
    r"\bw/\b": "with",
    r"\bh/o\b": "history of",
    r"\bc/o\b": "complaint of",
    r"\bn/v\b": "nausea vomiting",
    r"\bSOB\b": "shortness of breath",
    r"\bUNK\b": "unknown",
}


def expand_abbreviations(text: str) -> str:
    """Replace common medical abbreviations with full forms."""
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common MAUDE boilerplate phrases that carry no signal."""
    boilerplate_patterns = [
        r"it was reported that",
        r"the reporter stated",
        r"according to the report",
        r"this is a report",
        r"per the report",
        r"the following information was received",
        r"information has been received",
        r"no further information (is|was) available",
        r"follow.?up (is|will be) requested",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, lowercase: bool = True, preserve_digits: bool = False) -> str:
    """Full cleaning pipeline for a single narrative string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()
    text = remove_boilerplate(text)
    text = expand_abbreviations(text)

    if preserve_digits:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text
