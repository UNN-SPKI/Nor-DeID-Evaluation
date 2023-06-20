import re
from typing import List, Tuple

# _ENCLOSED_IN_TAGS matches on expressions with XML-style tags (e.g. '<Age>23</Age>')
# putting the tag name in the first capturing group, and the contents in the second
# capturing group.
# NOTE: This will fail if you have nested annotations.
_ENCLOSED_IN_TAGS = re.compile(r'<([\w_]*)>([^<]*)<\/\1>')

def remove_tags(task: str) -> str:
    """remove_tags removes simple XML tags from a text,
    replacing them with their contents."""
    return _ENCLOSED_IN_TAGS.sub(r'\2', task)

def list_annotations(annotated: str, expected_tags: List[str] = None) -> List[Tuple[int, int, str]]:
    annotations = []
    matches = _ENCLOSED_IN_TAGS.finditer(annotated)

    # We want to find the character spans as they will be
    # in the unannotated text. To achieve this, we keep a running
    # count of how many markup characters have found so far
    # in markup_offset:
    markup_offset = 0
    for match in matches:
        tag_name, contents = match.groups()
        # The annotations consist of an opening tag and a closing tag
        # (e.g. <Age></Age>) - the tags themselves add 5 characters:
        total_markup_chars = (2*len(tag_name) + 5)
        tag_start = match.span()[0] - markup_offset
        tag_end = match.span()[1] - markup_offset - total_markup_chars
        markup_offset += total_markup_chars
        if expected_tags is not None and tag_name not in expected_tags:
            continue
        annotations.append((tag_start, tag_end, tag_name))
    return annotations