"""Generate jyutping romanization for Cantonese text."""

import re

try:
    import ToJyutping
except ImportError:
    ToJyutping = None


def get_jyutping(text: str) -> str:
    """Generate jyutping romanization for Cantonese text.

    :param text: Cantonese text (traditional Chinese).
    :returns: Space-separated jyutping with tone numbers.
    """
    if ToJyutping is None:
        return ""
    try:
        clean_text = re.sub(r'[！？。，、：；"""（）!?.,;:]', "", text)
        if not clean_text:
            return ""
        result = ToJyutping.get_jyutping_text(clean_text)
        return result
    except Exception:
        return ""
