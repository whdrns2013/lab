from typing import List


def save_nouns(file_path:str, nouns:List[str]):
    text = ''
    for noun in nouns:
        text = text + ' ' + noun
    with open(file_path, 'w') as f:
        f.write(text)