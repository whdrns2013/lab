from typing import List
import os
from datetime import datetime
import hashlib

def make_file_name(dir_path:str, text:str):
    date = datetime.now().strftime('%Y-%m-%d')
    hash_object = hashlib.sha256()
    hash_object.update(text[0:20].encode('utf-8'))
    file_name = f'{date}-{hash_object.hexdigest()}'
    
    file_list = os.listdir(dir_path)
    count = 0
    for file in file_list:
        if file_name == file:
            count += 1
    if count > 0:
        file_name = file_name + '-' + str(count)
    return file_name

def save_nouns(dir_path:str, nouns:List[str]):
    text = ''
    for noun in nouns:
        text = text + ' ' + noun
    file_name = make_file_name(dir_path, text)
    with open(dir_path + '/' + file_name, 'w') as f:
        f.write(text)