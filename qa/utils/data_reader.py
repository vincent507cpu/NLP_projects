from collections import defaultdict
import os
BASE_DIR = '/home/nlp/Documents/dataset/qa'


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    build vocabulary
    :param items: list  [item1, item2, ... ]
    :param sort: order by freq if True, otherwise order by name
    :param min_count: freq in dict
    :param lower: lower case or not
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort

        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(result[i], i) for i in range(len(result))]
    reverse_vocab = [(pair[1], pair[0]) for pair in vocab]
    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('{}/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/test_set.seg_x.txt'.format(BASE_DIR))
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/vocab.txt'.format(BASE_DIR))