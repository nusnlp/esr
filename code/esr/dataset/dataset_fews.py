import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from tqdm import tqdm
import logging

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../fews'))
import fews


logger = logging.getLogger(__name__)


STOPWORDS = list(set(stopwords.words('english')))

LIMIT = 432


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
            lemma_keys = fews.get_lemmas(instance.get('lemma'), instance.get('pos'))
            if wsd_label is None:
                for lemma_key in lemma_keys:
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma_key))
                    self.example_list.append((instance_id, lemma_key, None))
            else:
                gold_lemma_keys = label_dict[instance_id]
                if len(gold_lemma_keys) > 1:
                    gold_lemma_keys = [gold_lemma_keys[annotator] for annotator in annotators]
                for gold_lemma_key in gold_lemma_keys:
                    self.example_list.append((instance_id, gold_lemma_key, 1))
                for lemma_key in lemma_keys:
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma_key))
                    if lemma_key in gold_lemma_keys: continue
                    self.example_list.append((instance_id, lemma_key, 0))
                self.segment_list.append(len(self.example_list))
        self.features = []
        ignored_fews = 0
        for instance_id, lemma_key, label in tqdm(self.example_list, disable=not is_main_process):
            input_ids = [0] * limit
            if self.tokenize_bert_stype:
                total = col_len + 3
            else:
                total = col_len + 4
            cur_sentence_id = self.get_sentence_id(instance_id, 0)
            cur_word_token, cur_word_token_len = sentence_dict[cur_sentence_id]
            cur_word_token_len_sum = sum(cur_word_token_len)
            total += cur_word_token_len_sum
            index = instance_dict[instance_id]
            instance_len = cur_word_token_len[index]
            total += instance_len
            if total > limit:
                ignored_fews += 1
                continue
            gloss_token = gloss_dict[lemma_key]
            gloss_token_len = len(gloss_token)
            total += gloss_token_len
            # [CLS] / <s>
            offset = 0
            input_ids[offset] = begin_id
            offset += 1
            # Current sentence
            instance_begin = offset + sum(cur_word_token_len[:index])
            instance_end = instance_begin + instance_len
            input_ids[offset:offset + cur_word_token_len_sum] = [token for tokens in cur_word_token for token in tokens]
            offset += cur_word_token_len_sum
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
        logger.info(f'Input limit: {limit}; Ignored FEWS: {ignored_fews}')

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

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in STOPWORDS and token.isalnum()]

    def get_definition(self, lemma_key):
        all_tokens = fews.get_synonyms(lemma_key)
        for example in fews.get_examples(lemma_key):
            all_tokens += self.remove_stopwords(word_tokenize(example.lower()))
        sorting_counts = Counter(all_tokens)
        all_tokens = sorted(list(OrderedDict.fromkeys(all_tokens)), key=lambda x: sorting_counts[x], reverse=True)
        return fews.get_definition(lemma_key) + ' ' + ' '.join(all_tokens)

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
