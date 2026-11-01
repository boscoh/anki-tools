"""Generate jyutping romanization for Cantonese text."""

import re

try:
    import ToJyutping

    HAS_TOJYUTPING = True
except ImportError:
    HAS_TOJYUTPING = False


def get_jyutping(text: str) -> str:
    """Generate jyutping romanization for Cantonese text.

    :param text: Cantonese text (traditional Chinese).
    :returns: Space-separated jyutping with tone numbers.
    """
    if not HAS_TOJYUTPING:
        return ""
    try:
        clean_text = re.sub(r'[！？。，、：；"""（）!?.,;:]', "", text)
        if not clean_text:
            return ""
        result = ToJyutping.get_jyutping_text(clean_text)
        return result
    except Exception:
        return ""
