import io
import os
import unicodedata
import string
import glob

import torch
import random

# alphabet small + capital letters + " .,;'"
# print(string.ascii_letters)
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Turn a Unicode string to plain ASCII,
# thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def load_data():
    """ Build the category_lines dictionary,
    a list of names per language
    """
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    def read_lines(filename):
        # read a file and split into lines
        return [unicode_to_ascii(line.strip()) for line in open(filename)]

    for filename in find_files("data/names/*.txt"):
        language_name = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(language_name)
        lines = read_lines(filename)
        category_lines[language_name] = lines

    return category_lines, all_categories


"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""


# Find letter index from all_letters, e.g., 'a'=0
def letter_to_index(letter):
    # if not found, returns -1
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories):
    language = random.choice(all_categories)
    line = random.choice(category_lines[language])
    category_tensor = torch.tensor([all_categories.index(language)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return language, line, category_tensor, line_tensor


if __name__ == "__main__":
    print(ALL_LETTERS)  # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'

    print(unicode_to_ascii("Abelló"))

    # print(glob.glob("./data/names/*"))

    category_lines, all_categories = load_data()
    print(category_lines["Italian"][:5])

    print(letter_to_tensor("J"))  # [1, 57]
    print(line_to_tensor("Jones").size())  # [5, 1, 57]

    print(random_training_example(category_lines, all_categories))
