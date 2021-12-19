import re
first_cap_re = re.compile("(.)([A-Z][a-z]+)")
all_cap_re = re.compile("([a-z0-9])([A-Z])")


def to_snake_case(string: str) -> str:
    s1 = first_cap_re.sub(r"\1_\2", string)
    return all_cap_re.sub(r"\1_\2", s1).lower()


def to_camel_case(string: str) -> str:
    first, *rest = string.split("_")
    return first + "".join(word.capitalize() for word in rest)
