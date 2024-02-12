# Could be more complicated, e.g. use UI to ask user
def ask_user(prompt: str) -> str:
    return input(prompt)


def enum_to_keys(enum):
    return [e.name for e in enum]
