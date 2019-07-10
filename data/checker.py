import csv
import os
from tqdm.auto import tqdm


def get_categories(metadata_path):
    all_categories = set() 
    excluded = [
            "Burping_or_eructation",
            "Bus",
            "Chime",
            "Drawer_open_or_close",
            "Fireworks",
            "Fart",
            "Hi-hat",
            "Gong",
            "Glockenspiel",
            "Harmonica", 
            "Microwave_oven",
            "Scissors",
            "Squeak",
            "Telephone",
            "label"]
    with open(metadata_path, newline='') as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            if row[1] not in excluded:
                all_categories.add(row[1])
        assert(len(all_categories) + len(excluded) - 1 == 41)
    return list(all_categories)


def get_filename(metadata_path, category, file_id):
    all_files = []
    with open(metadata_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1].strip() == category.strip() and row[2] == "1":
                all_files.append(row[0])
    
    return all_files[file_id], len(all_files) 


all_categories = get_categories("./metadata.csv") 
# print(len(all_categories))
minlen = 100000
for i in tqdm(range(len(all_categories))):
    category = all_categories[i]
    # print("%s: %d" %(category, get_filename("./metadata.csv", category, -1)))
    # minlen = min(minlen, get_filename("./metadata.csv", category, -1))
    for file_id in range(60): 
        file_name, length = get_filename("./metadata.csv", category, file_id)
        minlen = min(length, minlen)
        # print(file_name)
        a = os.path.join("aa", file_name)

print(minlen)
