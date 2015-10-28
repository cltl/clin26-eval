#!/usr/bin/python
'''
Created on Oct 13, 2015

@author: Minh Ngoc Le
'''
from StringIO import StringIO
from collections import defaultdict
from itertools import ifilter
import os
import re
from subprocess import call
import sys

import pytest

import numpy as np


def find_last_item_by_tag(stack, tag):
    for i in range(len(stack)-1, -1, -1):
        if stack[i][0] == tag:
            return i
    return -1

def read_spans_conll(f, path=''):
    spans = set()
    stack = []
    sent = 1
    for line in f:
        if line == '\n':
            sent += 1
            assert len(stack) == 0, 'Unclosed tag(s) in sentence %d' %sent
            continue
        fields = line.split('\t')
        token = int(fields[0]) 
        tags = fields[2].split('|')
        if len(tags) < len(stack):
            sys.stderr.write('WARN: Missing element in file %s, sentence %d, token %d.\n' %(path, sent, token))
        for tag in tags:
            new_span = None
            if re.match('^\([A-Z]+$', tag): # opening
                stack.append((tag[1:], token))
            elif re.match('^[A-Z]+$', tag): # continue
                assert tag in [item[0] for item in stack]
            elif re.match('^[A-Z]+\)$', tag): # closing
                tag = tag[:-1]
                index = find_last_item_by_tag(stack, tag)
                assert index >= 0
                new_span = (sent, stack[index][1], token, tag)
                del stack[index]
            elif re.match('^\([A-Z]+\)$', tag): # singleton
                new_span = (sent, token, token, tag[1:-1])
            else:
                assert tag == '_', 'Unrecognized format for tag: %s' %tag
            if new_span:
                if new_span in spans:
                    sys.stderr.write('WARN: Duplicate span in file %s, sentence %d, token %d.\n' %(path, sent, token))
                spans.add(new_span) 
    assert len(stack) == 0, 'Unclosed tag(s) in sentence %d' %sent
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

def compare_spans(key, res, filter_tag='__ALL__'):
    criteria = lambda item: filter_tag == '__ALL__' or item[3] == filter_tag 
    key_set = set(ifilter(criteria, key))
    res_set = set(ifilter(criteria, res))
    return len(res_set.intersection(key_set)), len(key_set), len(res_set)

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
    assert(list(spans)[0] == (1, 1, 2, 'PRO'))
    s = '1\tB\t(PRO\thttp://\n2\tC\tPRO\t_\n3\tD\tPRO)\t_'
    spans = read_spans_conll(StringIO(s))
    assert(len(spans) == 1)
    assert(list(spans)[0] == (1, 1, 3, 'PRO'))
    s = '1\tB\t(PRO\thttp://\n2\tC\tPRO|(PER\t_\n3\tD\tPER)|PRO)\t_'
    spans = read_spans_conll(StringIO(s))
    assert(len(spans) == 2)
    spans = sorted(spans)
    assert(spans[0] == (1, 1, 3, 'PRO'))
    assert(spans[1] == (1, 2, 3, 'PER'))
    with pytest.raises(AssertionError):
        s = '1\tB\t(PRO\thttp://\n2\tC\tPRO|(PER\t_\n3\tD\tPRO)\t_'
        read_spans_conll(StringIO(s))
    with pytest.raises(AssertionError):
        s = '1\tB\t(1342\thttp://\n2\tC\tPRO|(PER\t_\n3\tD\tPRO)\t_'
        read_spans_conll(StringIO(s))

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
    parser = argparse.ArgumentParser(description='Score the response of a system at Named-Entity Recognition and Classification.')
    parser.add_argument('key', help='path to a directory containing all key files')
    parser.add_argument('response', help='path to a directory containing all response files')
    args = parser.parse_args()

    call('date')
    test_all() # never run evaluation script without thorough testing
    data = defaultdict(list)
    filter_tags = ('PER', 'LOC', 'PRO', 'ORG', 'FIN', '__ALL__')
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
        for filter_tag in filter_tags:
            data[filter_tag].append(compare_spans(key, res, filter_tag))
    for filter_tag in filter_tags:
        if filter_tag == '__ALL__':
            print('\n\nOverall performance:\n')
        else:
            print('\n\nPerformance for entity type %s:\n' %filter_tag)
        p = compute_performance(data[filter_tag])
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
