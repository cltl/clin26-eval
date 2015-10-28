#!/usr/bin/python
'''
Created on Oct 13, 2015

@author: Minh Ngoc Le
'''
from StringIO import StringIO
from collections import defaultdict
import os
from subprocess import call
import sys

import numpy as np


def next_line(f):
    try:
        return next(f)
    except StopIteration:
        return ''

def read_spans_conll(f, path=''):
    spans = set()
    sent = 1
    line = next_line(f)
    while line:
        if line == '\n':
            sent += 1
            line = next_line(f)
            continue
        fields = line.strip().split('\t')
        token = int(fields[0])
        url = fields[3]
        if url != '_':
            start = end = token
            while True:
                fields = line.strip().split('\t')
                if (not line) or line == '\n' or fields[3] != url:
                    break
                end = int(fields[0])
                line = next_line(f)
            spans.add((sent, start, end, url))
        else:
            line = next_line(f)
    return spans

def compute_performance(data):
    data = np.asarray(data)
    assert 'int' in data.dtype.name
    # micro
    true_pos, true, pos = np.sum(data, axis=0)
    micro_p = true_pos / float(pos) if pos > 0 else 0
    micro_r = true_pos / float(true) if true > 0 else 0
    micro_f1 = 2 / (1/micro_p + 1/micro_r) if micro_p > 0 and micro_r > 0 else 0
    total = pos
    missed = true - true_pos
    invented = pos - true_pos
    # macro
    true_pos, true, pos = data[:,0], data[:,1], data[:,2]
    with np.errstate(divide='ignore', invalid='ignore'):
        true_pos = np.asarray(true_pos, dtype='float')
        macro_p = np.where(pos == 0, 0, true_pos / pos)
        macro_r = np.where(true == 0, 0, true_pos / true)
        macro_f1 = np.where(np.logical_or(macro_p == 0, macro_r == 0), 
                            0, 2 / (1/macro_p + 1/macro_r))
    macro_p = np.mean(macro_p)
    macro_r = np.mean(macro_r)
    macro_f1 = np.mean(macro_f1)
    return total, missed, invented, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

def compare_spans_exact(key, res):
    key_set = set(key)
    res_set = set(res)
    return len(res_set.intersection(key_set)), len(key_set), len(res_set)

def compare_spans_partial(key, res):
    tp = 0
    for sent, start, end, url in key:
        matched = False
        for sent2, start2, end2, url2 in res:
            if sent2 == sent and start2 <= end and end2 >= start and url2 == url:
                matched = True
                break
        if matched: tp += 1
    return tp, len(key), len(res)

def first_five_sentences(f):
    sent = 1
    for line in f:
        if line == '\n':
            sent += 1
            if sent > 5:
                return
        yield line

def test_read_spans_conll():
    s = '1\tA\t_\t_\n'
    assert(len(read_spans_conll(StringIO(s))) == 0)
    s = '1\tB\t(PRO\thttp://\n2\tC\tPRO)\t_'
    spans = read_spans_conll(StringIO(s))
    assert(len(spans) == 1)
    assert(list(spans)[0] == (1, 1, 1, 'http://'))
    s = '1\tB\t(PRO\thttp://\n2\tC\tPRO\thttp://\n3\tD\tPRO)\t_'
    spans = read_spans_conll(StringIO(s))
    assert(len(spans) == 1)
    assert(list(spans)[0] == (1, 1, 2, 'http://'))
    s = '1\tB\t(PRO\thttp://\n2\tC\tPRO|(PER\t_\n3\tD\tPER)|PRO)\thttp://'
    spans = read_spans_conll(StringIO(s))
    assert(len(spans) == 2)
    spans = sorted(spans)
    assert(spans[0] == (1, 1, 1, 'http://'))
    assert(spans[1] == (1, 3, 3, 'http://'))

def test_compute_performance():
    p = compute_performance([[0, 10, 0]])
    assert np.all(np.array(p[3:]) == 0)
    p = compute_performance([[0, 10, 5]])
    assert np.all(np.array(p[3:]) == 0)
    p = compute_performance([[1, 1, 1]])
    assert np.all(np.array(p[3:]) == 1)
    p = compute_performance([[1, 5, 2], [3, 10, 9]])
    assert np.allclose(p[3:], [0.364, 0.267, 0.308, 0.417, 0.25, 0.301], 0.01)
    
def test_all():
    test_compute_performance()
    test_read_spans_conll()
    sys.stderr.write('Passed all tests.\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Score the response of a system at Named-Entity Disambiguation.')
    parser.add_argument('key', help='path to a directory containing all key files')
    parser.add_argument('response', help='path to a directory containing all response files')
    args = parser.parse_args()

    call('date')
    test_all() # never run evaluation script without thorough testing
    data = defaultdict(list)
    if len(os.listdir(args.key)) < len(os.listdir(args.response)):
        sys.stderr.write('WARN: response folder holds more files than key folder. Some files will be ignored.\n')
    for fname in os.listdir(args.key):
        path = os.path.join(args.key, fname)
        with open(path) as f:
            key = read_spans_conll(first_five_sentences(f), path)
        path = os.path.join(args.response, fname)
        if os.path.exists(path):
            with open(path) as f:
                res = read_spans_conll(first_five_sentences(f), path)
        else:
            res = set()
        data['exact'].append(compare_spans_exact(key, res))
        data['partial'].append(compare_spans_partial(key, res))
    for name in data:
        print('\n\nPerformance (%s spans):\n' %name)
        p = compute_performance(data[name])
        print('# response total: %d\n'
              '# missed: %d\n'
              '# invented: %d\n\n'
              'Micro average:\n'
              'precision\t%.3f\n'
              'recall\t%.3f\n'
              'f1\t%.3f\n\n'
              'Macro average:\n'
              'precision\t%.3f\n'
              'recall\t%.3f\n'
              'f1\t%.3f\n' %p)
