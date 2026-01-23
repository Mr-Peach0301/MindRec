import json
from collections import defaultdict
dataset="Yelp" # Arts Instruments

with open(f'{dataset}/{dataset}.item.json', 'r') as f:
    data = json.load(f)

import json

def process_field(value):
    if not value or value.strip() == '.' or value.strip() == '':
        return 'na'
    return value.strip()

import re
from collections import Counter
import matplotlib.pyplot as plt
from collections import OrderedDict

def extract_categories(json_data):
    item_categories = {}
    for item_id, item_data in json_data.items():
        if dataset=="Yelp":
            categories = re.split(r'(?<!\s), (?!\s)', item_data['categories'])
        else:
            categories = re.split(r'(?<!\s),(?!\s)', item_data['categories'])
        categories = [cat.strip() for cat in categories]
        categories = list(OrderedDict.fromkeys(categories))
        item_categories[item_id] = categories
    return item_categories
    
item_categories = extract_categories(data)

with open(f'{dataset}/item_categories.json', 'w') as f:
    json.dump(item_categories, f, indent=2)

import numpy as np
import re
from collections import defaultdict
def get_hash(x):
    x = [str(_) for _ in x]
    return '_'.join(x)

if dataset!="Yelp":
    pre_dict = {0:'first', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i', 10:'j', 11:'k', 12:'l', 13:'m', 14:'n', 15:'o', 16:'p', 17:'q', 18:'r', 19:'s', 20:'t', 21:'u', 22:'v', 23:'w', 24:'x', 25:'y', 26:'z'}
else:
    pre_dict = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}
category_codes = {}
level2num = defaultdict(int)
for d,v in item_categories.items():
    for i in range(len(v)):
        if i==0:
            if v[i] not in category_codes:
                category_codes[v[i]] = [f"<{pre_dict[i]}:{level2num['first']}>"]
                level2num['first'] += 1
        else:
            if v[i] not in category_codes:
                pre_code = []
                pattern = r":(\d+)>"
                for s in category_codes[v[i-1]]:
                    match = re.search(pattern, s)
                    if match:
                        pre_code.append(int(match.group(1)))
                    else:
                        print("!!!")
                category_codes[v[i]] = category_codes[v[i-1]] + [f'<{pre_dict[i]}:{level2num[get_hash(pre_code)]}>']
                level2num[get_hash(pre_code)] += 1

if dataset!="Yelp":
    category_codes = {k:v[1:] for k,v in category_codes.items()}
with open(f'{dataset}/category_codes.json', 'w') as f:
    json.dump(category_codes, f, indent=2)

def convert_to_codes(item_categories, category_codes):
    item_category_codes = {}
    for item_id, categories in item_categories.items():
        codes = []
        for i in range(1,len(categories)):
            codes.append(category_codes[categories[i]])
        item_category_codes[item_id] = codes
    return item_category_codes

item_category_codes = convert_to_codes(item_categories, category_codes)
with open(f'{dataset}/item_category_codes.json', 'w') as f:
    json.dump(item_category_codes, f, indent=2)

with open(f'{dataset}/item_category_codes.json', 'r') as f:
    data = json.load(f)

with open(f'{dataset}/level2num.json', 'w') as f:
    json.dump(level2num, f, indent=2)

with open(f'{dataset}/item_category_codes.json', 'r') as f:
    data = json.load(f)

def find_sublist(lst, level):
    for sublist in lst:
        if len(sublist) == level:
            return sublist
    
    max_length = max(len(sublist) for sublist in lst) if lst else 0
    
    for sublist in lst:
        if len(sublist) == max_length:
            return sublist
    return None

import json

def get_target_list(data_list, level):
    longest_sublist = []
    
    for row in data_list:
        valid_prefix = []
        for i, item in enumerate(row):
            expected_char = chr(ord('a') + i)
            try:
                current_char = item.split(':')[0].lstrip('<')
            except Exception:
                break

            if current_char == expected_char:
                valid_prefix.append(item)
            else:
                break

        if not valid_prefix:
            continue

        current_len = len(valid_prefix)

        if current_len == level:
            return valid_prefix

        if current_len > len(longest_sublist):
            longest_sublist = valid_prefix
    return longest_sublist

level=3
item2code = {}
for item, code in data.items():
    if len(code)==0:
        c = [f"<a:{level2num['0']}>"]
    else:
        if dataset!="Yelp":
            c = find_sublist(code, level)
        else:
            c = get_target_list(code, level)
    if len(c) == level:
        item2code[item] = c
    elif len(c)>level:
        item2code[item] = c[:level]
    else:
        if dataset!="Yelp":
            for i in range(len(c)+1, leve+1):
                pre_code = [0]
                pattern = r":(\d+)>"
                for s in c:
                    match = re.search(pattern, s)
                    if match:
                        pre_code.append(int(match.group(1)))
                    else:
                        print("!!!")
                c = c + [f'<{pre_dict[i]}:{level2num[get_hash(pre_code)]}>']
        else:
            for i in range(len(c), level):
                pre_code = [0]
                pattern = r":(\d+)>"
                for s in c:
                    match = re.search(pattern, s)
                    if match:
                        pre_code.append(int(match.group(1)))
                    else:
                        print("!!!")
                c = c + [f'<{pre_dict[i]}:{level2num[get_hash(pre_code)]}>']
        item2code[item] = c
with open(f'{dataset}/{dataset}.category.json', 'w') as f:
    json.dump(item2code, f, indent=2)