import re


def fix_orthography(answer: str) -> str:
    """fix_orthography adds spaces between punctuation and
    reduces whitespace to single spaces to align an answer
    with the punctuation in CoNLL texts."""
    space_punctuation = re.sub('\s*([,.])\s+', r' \1 ', answer).rstrip()
    single_spaces = re.sub('\s+', ' ', space_punctuation)
    return single_spaces