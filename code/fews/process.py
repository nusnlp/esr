import argparse
import os
import fews
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from nltk import word_tokenize
from tqdm import tqdm


def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--do_fews_senses', action='store_true')
    argparser.add_argument('--fews_senses')
    argparser.add_argument('--do_fews_data', action='store_true')
    argparser.add_argument('--fews_txt')
    argparser.add_argument('--document_id', type=int)
    argparser.add_argument('--fews_xml')
    argparser.add_argument('--fews_label')
    return argparser.parse_args()

def make_dir(file):
    path = os.path.dirname(os.path.realpath(file))
    if not os.path.exists(path):
        os.makedirs(path)

def check_num_instance(fews_txt):
    with open(fews_txt, 'r') as file:
        for line in file.readlines():
            line = line.strip().split('\t')
            assert len(line) == 2

def to_xml(fews_txt, document_id, fews_xml, fews_label):
    root = ET.Element('corpus')
    root.attrib['lang'] = 'en'
    root.attrib['source'] = 'fews'
    document_id = 'd{0:0={1}d}'.format(document_id, 5)
    text = ET.SubElement(root, 'text')
    text.attrib['id'] = document_id
    sentence_index = 0
    with open(fews_txt, 'r') as txt_file, open(fews_label, 'w') as label_file:
        for line in tqdm(txt_file.readlines()):
            line = line.strip().split('\t')
            lemma, pos, _ = line[1].rsplit('.', 2)
            sentence_id = 's{0:0={1}d}'.format(sentence_index, 5)
            sentence_id = f'{document_id}.{sentence_id}'
            sentence = ET.SubElement(text, 'sentence')
            sentence.attrib['id'] = sentence_id
            instance_index = 0
            for context, ambiguous in zip(re.split('<WSD>.*?</WSD>', line[0]), re.findall('<WSD>.*?</WSD>', line[0]) + [None]):
                for wf in word_tokenize(context):
                    ET.SubElement(sentence, 'wf').text = wf
                if ambiguous is None: continue
                instance_id = 't{0:0={1}d}'.format(instance_index, 5)
                instance_id = f'{sentence_id}.{instance_id}'
                label_file.write(f'{instance_id} {line[1]}\n')
                instance = ET.SubElement(sentence, 'instance')
                instance.attrib['id'] = instance_id
                instance.attrib['lemma'] = lemma
                instance.attrib['pos'] = pos
                instance.text = ambiguous.replace('<WSD>', '').replace('</WSD>', '').strip()
                instance_index += 1
            sentence_index += 1
    with open(fews_xml, 'wb') as file:
        file.write(minidom.parseString(ET.tostring(root)).toprettyxml(encoding="utf-8", indent=''))


if __name__ == '__main__':
    args = set_args()
    if args.do_fews_senses:
        fews.preprocess(args.fews_senses)
    if args.do_fews_data:
        check_num_instance(args.fews_txt)
        make_dir(args.fews_xml)
        to_xml(args.fews_txt, args.document_id, args.fews_xml, args.fews_label)
