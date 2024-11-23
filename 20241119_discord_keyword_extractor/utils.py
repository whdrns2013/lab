from typing import List
import os
from datetime import datetime
import hashlib
import configparser

# config parser
config = configparser.ConfigParser()
config.read('./config.ini')

# open existed files
def open_existed_files(dir_path:str):
    file_list = os.listdir(dir_path)
    return file_list

# make file name
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

# save nouns at text file
def save_nouns_txt(dir_path:str, nouns:List[str]):
    text = ''
    for noun in nouns:
        text = text + ' ' + noun
    file_name = make_file_name(dir_path, text)
    with open(dir_path + '/' + file_name, 'w') as f:
        f.write(text)
        
# replace special characters
def remove_specail_characters(text:str):
    specail_character_list = ['|', '#', '\n', '[', ']']
    for sc in specail_character_list:
        text.replace(sc, ' ')
    return text

# replace stop words
def remove_stop_words(text:str):
    stop_word_dic_file = config['Files']['stopWordsDict']
    with open(stop_word_dic_file, 'r') as f:
        stop_words = f.readlines()
    for sw in stop_words:
        text.replace(sw, '')
    return text