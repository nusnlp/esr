import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Optional
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


TAG = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV, 'PRT': wn.ADV}
P2W = {
    'CC': 'CONJ', 'CD': 'NUM', 'DT': 'DET', 'EX': 'DET', 'FW': 'X',
    'IN': 'ADP', 'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ', 'LS': 'NUM',
    'MD': 'VERB', 'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
    'PDT': 'DET', 'POS': 'PRT', 'PRP': 'PRON', 'PRP$': 'PRON', 'RB': 'ADV',
    'RBR': 'ADV', 'RBS': 'ADV', 'RP': 'PRT', 'SYM': '.', 'TO': 'PRT',
    'UH': 'X', 'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
    'VBP': 'VERB', 'VBZ': 'VERB', 'WDT': 'DET', 'WP': 'PRON', 'WP$': 'PRON',
    'WRB': 'ADV', '$': '.', '#': '.', "``": '.', "''": '.',
    '(': '.', ')': '.', ',': '.', '.': '.', ':': '.'
}
STOPWORDS = list(set(stopwords.words('english')))


@dataclass(frozen=True)
class InputFeatures:

    input_ids: List[int]
    input_len: int
    context_len: int
    instance: Tuple[int, int, int]
    label: Optional[int]


class WsdDataset(Dataset):

    def __init__(self, tokenizer, limit, wsd_xml, wsd_label, annotators, is_dev, is_main_process, extra_xml=None):
        self.analyze_tokenizer(tokenizer)
        begin_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token if self.tokenize_bert_stype else tokenizer.bos_token)
        end_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token if self.tokenize_bert_stype else tokenizer.eos_token)
        col_id = self.tokenize(':')
        col_len = len(col_id)

        root = ET.parse(wsd_xml).getroot()
        sentence_dict = {}
        instance_dict = {}
        sentences = root.findall('.//sentence')
        for sentence in tqdm(sentences, disable=not is_main_process):
            word_token = []
            word_token_len = []
            index = 0
            for word in sentence.getchildren():
                tokens = self.tokenize(word.text)
                word_token.append(tokens)
                word_token_len.append(len(tokens))
                if word.tag == 'instance':
                    instance_dict[word.get('id')] = index
                index += 1
            sentence_dict[sentence.get('id')] = (word_token, word_token_len)
        if wsd_label is not None:
            label_dict = {}
            with open(wsd_label, 'r') as file:
                for line in file.readlines():
                    line = line.split()
                    label_dict[line[0]] = line[1:]
            self.segment_list = [0]
        self.example_list = []
        gloss_dict = {}
        instances = root.findall('.//instance')
        for instance in tqdm(instances, disable=not is_main_process):
            instance_id = instance.get('id')
            lemmas = wn.lemmas(instance.get('lemma'), TAG[instance.get('pos')])
            if wsd_label is None:
                for lemma in lemmas:
                    lemma_key = lemma.key()
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma.synset()))
                    self.example_list.append((instance_id, lemma_key, None))
            else:
                gold_lemma_keys = label_dict[instance_id]
                if len(gold_lemma_keys) > 1:
                    gold_lemma_keys = [gold_lemma_keys[annotator] for annotator in annotators]
                for gold_lemma_key in gold_lemma_keys:
                    self.example_list.append((instance_id, gold_lemma_key, 1))
                for lemma in lemmas:
                    lemma_key = lemma.key()
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma.synset()))
                    if lemma_key in gold_lemma_keys: continue
                    self.example_list.append((instance_id, lemma_key, 0))
                self.segment_list.append(len(self.example_list))
        self.features = []
        for instance_id, lemma_key, label in tqdm(self.example_list, disable=not is_main_process):
            input_ids = [0] * limit
            if self.tokenize_bert_stype:
                total = col_len + 3
            else:
                total = col_len + 4
            prev_sentence_id = self.get_sentence_id(instance_id, -1)
            if prev_sentence_id in sentence_dict:
                prev_word_token, prev_word_token_len = sentence_dict[prev_sentence_id]
                prev_word_token_len_sum = sum(prev_word_token_len)
                total += prev_word_token_len_sum
            next_sentence_id = self.get_sentence_id(instance_id, 1)
            if next_sentence_id in sentence_dict:
                next_word_token, next_word_token_len = sentence_dict[next_sentence_id]
                next_word_token_len_sum = sum(next_word_token_len)
                total += next_word_token_len_sum
            cur_sentence_id = self.get_sentence_id(instance_id, 0)
            cur_word_token, cur_word_token_len = sentence_dict[cur_sentence_id]
            cur_word_token_len_sum = sum(cur_word_token_len)
            total += cur_word_token_len_sum
            index = instance_dict[instance_id]
            instance_len = cur_word_token_len[index]
            total += instance_len
            gloss_token = gloss_dict[lemma_key]
            gloss_token_len = len(gloss_token)
            total += gloss_token_len
            # [CLS] / <s>
            offset = 0
            input_ids[offset] = begin_id
            offset += 1
            # Previous sentence
            if total <= limit and prev_sentence_id in sentence_dict:
                offset += prev_word_token_len_sum
                input_ids[1:offset] = [token for tokens in prev_word_token for token in tokens]
            # Current sentence
            instance_begin = offset + sum(cur_word_token_len[:index])
            instance_end = instance_begin + instance_len
            input_ids[offset:offset + cur_word_token_len_sum] = [token for tokens in cur_word_token for token in tokens]
            offset += cur_word_token_len_sum
            # Next sentence
            if total <= limit and next_sentence_id in sentence_dict:
                input_ids[offset:offset + next_word_token_len_sum] = [token for tokens in next_word_token for token in tokens]
                offset += next_word_token_len_sum
            # [SEP] / </s> + <s>
            input_ids[offset] = end_id
            offset += 1
            context_len = offset
            if not self.tokenize_bert_stype:
                input_ids[offset] = begin_id
                offset += 1
            # Ambiguous word
            input_ids[offset:offset + instance_len] = cur_word_token[index]
            offset += instance_len
            # Colon
            input_ids[offset:offset + col_len] = col_id
            offset += col_len
            # Gloss
            if offset + gloss_token_len > limit - 1:
                gloss_token_len = limit - 1 - offset
            input_ids[offset:offset + gloss_token_len] = gloss_token[:gloss_token_len]
            offset += gloss_token_len
            # [SEP] / </s>
            input_ids[offset] = end_id
            offset += 1
            assert offset <= limit
            self.features.append(InputFeatures(
                input_ids=input_ids[:offset],
                input_len=offset,
                context_len=context_len,
                instance=(instance_begin, instance_end, instance_len),
                label=label
            ))

        if wsd_label is None or extra_xml is None: return
        root = ET.parse(extra_xml).getroot()
        sentence_list = []
        instance_list = []
        sentences = root.findall('.//sentence')
        for sentence in tqdm(sentences, disable=not is_main_process):
            word_token = []
            word_token_len = []
            index = 0
            head = ' , '.join([lemma_key.split('%')[0].replace('_', ' ') for lemma_key in sentence.get('wn30_key').split(';')])
            head = f'{head} :'
            tokens = self.tokenize(head)
            word_token.append(tokens)
            word_token_len.append(len(tokens))
            index += 1
            for word in sentence.getchildren():
                tokens = self.tokenize(word.get('surface_form').replace('_', ' '))
                word_token.append(tokens)
                word_token_len.append(len(tokens))
                wn30_key = word.get('wn30_key')
                if wn30_key is not None:
                    lemma = word.get('lemma')
                    pos = TAG[P2W[word.get('pos')]]
                    instance_list.append((len(sentence_list), index, lemma, pos, wn30_key))
                index += 1
            sentence_list.append((word_token, word_token_len))
        example_list = []
        for sentence_id, index, lemma, pos, gold_lemma_key in tqdm(instance_list, disable=not is_main_process):
            if not gold_lemma_key in gloss_dict:
                gold_lemma_key = self.correct_lemma_key(gold_lemma_key)
            example_list.append((sentence_id, index, gold_lemma_key, 1))
            for lemma in wn.lemmas(lemma, pos):
                lemma_key = lemma.key()
                if not lemma_key in gloss_dict:
                    gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma.synset()))
                if lemma_key == gold_lemma_key: continue
                example_list.append((sentence_id, index, lemma_key, 0))
        features = []
        for sentence_id, index, lemma_key, label in tqdm(example_list, disable=not is_main_process):
            input_ids = [0] * limit
            offset = 0
            if self.tokenize_bert_stype:
                total = col_len + 3
            else:
                total = col_len + 4
            word_token, word_token_len = sentence_list[sentence_id]
            word_token_len_sum = sum(word_token_len)
            total += word_token_len_sum
            instance_len = word_token_len[index]
            total += instance_len
            gloss_token = gloss_dict[lemma_key]
            gloss_token_len = len(gloss_token)
            total += gloss_token_len
            # [CLS] / <s>
            input_ids[offset] = begin_id
            offset += 1
            # Current sentence
            instance_begin = offset + sum(word_token_len[:index])
            instance_end = instance_begin + instance_len
            input_ids[offset:offset + word_token_len_sum] = [token for tokens in word_token for token in tokens]
            offset += word_token_len_sum
            # [SEP] / </s> + <s>
            input_ids[offset] = end_id
            offset += 1
            context_len = offset
            if not self.tokenize_bert_stype:
                input_ids[offset] = begin_id
                offset += 1
            # Ambiguous word
            input_ids[offset:offset + instance_len] = word_token[index]
            offset += instance_len
            # Colon
            input_ids[offset:offset + col_len] = col_id
            offset += col_len
            # Gloss
            if offset + gloss_token_len > limit - 1:
                gloss_token_len = limit - 1 - offset
            input_ids[offset:offset + gloss_token_len] = gloss_token[:gloss_token_len]
            offset += gloss_token_len
            # [SEP] / </s>
            input_ids[offset] = end_id
            offset += 1
            assert offset <= limit
            features.append(InputFeatures(
                input_ids=input_ids[:offset],
                input_len=offset,
                context_len=context_len,
                instance=(instance_begin, instance_end, instance_len),
                label=label
            ))
        self.features = features + self.features

    def analyze_tokenizer(self, tokenizer):
        tokenize_stype_dict = {
            'BertTokenizer': True,
            'AlbertTokenizer': True,
            'RobertaTokenizerFast': False,
            'XLMRobertaTokenizer': False,
            'BartTokenizer': False
        }
        self.tokenize_bert_stype = tokenize_stype_dict[type(tokenizer).__name__]
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer.encode(' ' + text.strip(), add_special_tokens=False)

    def correct_lemma_key(self, key):
        try:
            wn.lemma_from_key(key)
            return key
        except:
            key = key.split('%', 1)
            assert key[1][0] == '3'
            return f'{key[0]}%5{key[1][1:]}'

    def get_synonym_tokens(self, synset):
        synonym_tokens = []
        for lemma in synset.lemmas():
            synonym_tokens += lemma.name().replace('_', ' ').split()
        return synonym_tokens

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in STOPWORDS and token.isalnum()]

    def get_example_tokens(self, synset):
        example_tokens = []
        for example in synset.examples():
            example_tokens += self.remove_stopwords(word_tokenize(example.lower()))
        return example_tokens

    def get_hypernym_tokens(self, synset):
        hypernym_tokens = []
        for hypernym in synset.hypernyms():
            hypernym_tokens += self.remove_stopwords(word_tokenize(hypernym.definition().lower()))
        return hypernym_tokens

    def get_related_words(self, synset):
        synonym_tokens = self.get_synonym_tokens(synset)
        example_tokens = self.get_example_tokens(synset)
        hypernym_tokens = self.get_hypernym_tokens(synset)
        all_tokens = synonym_tokens + example_tokens + hypernym_tokens
        sorting_counts = Counter(all_tokens)
        all_tokens = sorted(list(OrderedDict.fromkeys(all_tokens)), key=lambda x: sorting_counts[x], reverse=True)
        return ' '.join(all_tokens)

    def get_definition(self, synset):
        return synset.definition() + ' ' + self.get_related_words(synset)

    def get_sentence_id(self, instance_id, offset):
        sentence_id = instance_id.rsplit('.', 1)[0]
        if offset == 0:
            return sentence_id
        else:
            t_id, s_id = sentence_id.rsplit('.', 1)
            length = len(s_id) - 1
            index = int(s_id[1:]) + offset
            if index < 0: return None
            if len(str(index)) > length: return None
            s_id = 's{0:0={1}d}'.format(index, length)
            return '.'.join([t_id, s_id])

    def get_example_list(self):
        return self.example_list

    def get_segment_list(self):
        return self.segment_list

    def probe(self, feature):
        instance_begin, instance_end, _ = feature.instance
        logger.info(
            f'{self.tokenizer.convert_ids_to_tokens(feature.input_ids[instance_begin:instance_end])}\n\n'
            + f'{self.tokenizer.convert_ids_to_tokens(feature.input_ids)}\n\n'
            + f'{feature.input_ids}\n\n'
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # self.probe(self.features[idx])
        return self.features[idx]


@dataclass
class DataCollatorForWsd:

    def __call__(self, features):
        batch = {}
        batch_size = len(features)
        max_input_len = max([feature.input_len for feature in features])
        input_ids = [[0] * max_input_len for _ in range(batch_size)]
        attention_mask = [[0] * max_input_len for _ in range(batch_size)]
        token_type_ids = [[0] * max_input_len for _ in range(batch_size)]
        instance_mask = [[0] * max_input_len for _ in range(batch_size)]
        instance_lens = [0] * batch_size
        for i, feature in enumerate(features):
            input_len = feature.input_len
            input_ids[i][:input_len] = feature.input_ids
            attention_mask[i][:input_len] = [1] * input_len
            context_len = feature.context_len
            token_type_ids[i][context_len:input_len] = [1] * (input_len - context_len)
            instance_begin, instance_end, instance_len = feature.instance
            instance_mask[i][instance_begin:instance_end] = [1] * instance_len
            instance_lens[i] = instance_len
        batch['input_ids'] = torch.as_tensor(input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.as_tensor(attention_mask, dtype=torch.long)
        batch['token_type_ids'] = torch.as_tensor(token_type_ids, dtype=torch.long)
        batch['instance_mask'] = torch.as_tensor(instance_mask, dtype=torch.long)
        batch['instance_lens'] = torch.as_tensor(instance_lens, dtype=torch.long)
        if features[0].label is not None:
            batch['labels'] = torch.as_tensor([feature.label for feature in features], dtype=torch.long)
        return batch
