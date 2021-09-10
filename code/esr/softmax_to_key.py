import argparse


def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--softmax_file')
    argparser.add_argument('--key_file')
    return argparser.parse_args()

def convert(softmax_file, key_file):
    with open(softmax_file, 'r') as softmax_file, open(key_file, 'w') as key_file:
        prev_index = ''
        max_score = 0
        for line in softmax_file.readlines():
            line = line.strip().split(',')
            if prev_index == line[0] or prev_index == '':
                if float(line[2]) > max_score:
                    lemma = line[1]
                    max_score = float(line[2])
            else:
                key_file.write(f'{prev_index} {lemma}\n')
                lemma = line[1]
                max_score = float(line[2])
            prev_index = line[0]
        key_file.write(f'{prev_index} {lemma}\n')


if __name__ == '__main__':
    args = set_args()
    convert(args.softmax_file, args.key_file)
