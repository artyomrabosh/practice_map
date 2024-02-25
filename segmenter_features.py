import regex as re

reg_code = "(.*)([[:alnum:]]+\.)+([[:alnum:]]|<[[:alnum:]]+>)+\((.*)"
reg_attribute = '[A-Za-z]+\.[A-Za-z]+'
reg_word = '[^\d\W]+'
reg_link = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
reg_lower_camel = '(\b([a-z_][a-z_0-9]*)+([A-Z_0-9][a-z_0-9]*)+\b)'
reg_index_num = '\[[0-9]\]'
reg_index_letter = '\[[a-z]\]'
reg_word_ru = '[А-Яа-я]+'
reg_word_en = '[A-Za-z]+'
# code_words = list(dct.keys())


def end_of_line(line):
    if not line.strip():
        return 0.0
    if line.strip()[-1] in [';', '}', '{'] and not re.findall('[а-яА-ЯёЁ]', line):
        return 1.0
    else:
        return 0.0
    
def code_reg(line):
    if re.match(reg_code, line):
        return 1.0
    else:
        return 0.0
    
def is_link(line):
    if re.match(reg_link, line):
        return 1.0
    else:
        return 0.0
    
def is_camelcase(line):
    if re.match(reg_lower_camel, line):
        return 1.0
    else:
        return 0.0

def is_comment(line):
    if not is_link(line):
        if "// " in line:
            return 1.0
    return 0.0

def is_attribute(line):
    if not is_link(line):
        return len(re.findall(reg_attribute, line.lower()))
    else:
        return 0
# def count_code_reg(line):
#     cnt = 0
#     for word in code_words:
#         if word in line:
#             cnt += 1
#     return float(cnt)

def num_of_words(line):
    return len(re.findall(reg_word, line.lower()))

def num_of_words_ru(line):
    return len(re.findall(reg_word_ru, line.lower()))

def num_of_words_en(line):
    return len(re.findall(reg_word_en, line.lower()))
    
def num_of_index_num(line):
    return len(re.findall(reg_index_num, line.lower()))

def num_of_index_letter(line):
    return len(re.findall(reg_index_letter, line.lower()))

def ru_percantage(line):
    return num_of_words_ru(line) / num_of_words(line)

def en_percantage(line):
    return num_of_words_en(line) / num_of_words(line)

def brackets(line):
    if '()' in line:
        return 1.0
    else:
        return 0.0