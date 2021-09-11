import os
import re
import pickle
import warnings
from tqdm import tqdm


DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data.pkl')


if os.path.isfile(DATA):
    with open(DATA, 'rb') as file:
        word_dict, word_pos_dict, synonym_dict, gloss_dict, definition_dict, example_dict = pickle.load(file)
else:
    warnings.warn('No data found, run preprocess()')


def dict_append(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]

def filter(definition):
    definition = definition.strip()
    definition = re.sub(' +', ' ', definition)
    definition = re.sub('^(,|\.|;|:)', '', definition).strip()
    definition = re.sub('(,|;|:)$', '', definition).strip()
    definition = re.sub(',$', '.', definition).strip()
    return definition

def example_in_parenthesis(s):
    examples = []
    state = 0
    parenthesis = []
    for i, c in enumerate(s):
        if state == 0:
            if c == '(':
                parenthesis.append(i)
                state = 1
        elif state == 1:
            if c == '(':
                parenthesis.append(i)
            elif c == ')':
                parenthesis.pop()
                if len(parenthesis) == 0:
                    state = 0
            elif c == 'e' and s[i:i + 4] == 'e.g.':
                parenthesis = parenthesis[-1:]
                state = 2
        elif state == 2:
            if c == '(':
                parenthesis.append(i)
            elif c == ')':
                e = parenthesis.pop()
                if len(parenthesis) == 0:
                    examples.append((e, i + 1))
                    state = 0
    return examples

def process_gloss(gloss):
    examples = []
    offset = 0
    for b, e in example_in_parenthesis(gloss):
        b -= offset; e -= offset
        examples.append(gloss[b + 1:e - 1].replace('"e.g."', '').replace('e.g.', ''))
        gloss = gloss[:b] + gloss[e:]
        offset += e - b
    gloss = gloss.split('e.g.')
    examples += gloss[1:]
    examples = [filter(example) for example in examples]
    return filter(gloss[0]), [example for example in examples if len(example) > 0]

def preprocess(sense_file):
    word_dict = {}
    word_pos_dict = {}
    synonym_dict = {}
    gloss_dict = {}
    definition_dict = {}
    example_dict = {}
    sense_id = None; word = None; word_pos = None
    with open(sense_file, 'r') as file:
        for line in tqdm(file.readlines()):
            line = line.strip()
            if len(line) == 0:
                sense_id = None; word = None; word_pos = None
                continue
            line = line.split('\t')
            if line[0] == 'sense_id:':
                sense_id = line[1]
                word_pos = sense_id.rsplit('.', 1)[0]
                word = word_pos.rsplit('.', 1)[0]
                dict_append(word_dict, word, sense_id)
                dict_append(word_pos_dict, word_pos, sense_id)
            elif line[0] == 'synonyms:':
                synonym_dict[sense_id] = []
                if len(line) > 1:
                    synonym_dict[sense_id] += line[1].strip().split(', ')
            elif line[0] == 'gloss:':
                gloss_dict[sense_id] = line[1]
                definition_dict[sense_id], example_dict[sense_id] = process_gloss(line[1])
    with open(DATA, 'wb') as file:
        pickle.dump((word_dict, word_pos_dict, synonym_dict, gloss_dict, definition_dict, example_dict), file, pickle.HIGHEST_PROTOCOL)

def get_lemmas(word, pos=None):
    return word_dict[word] if pos is None else word_pos_dict[f'{word}.{pos}']

def get_synonyms(sense_id):
    return synonym_dict[sense_id]

def get_gloss(sense_id):
    return gloss_dict[sense_id]

def get_definition(sense_id):
    return definition_dict[sense_id]

def get_examples(sense_id):
    return example_dict[sense_id]
