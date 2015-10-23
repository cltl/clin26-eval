#!/usr/bin/python
'''
Created on Oct 13, 2015

@author: Minh Ngoc Le
'''
import os
import numpy as np
from StringIO import StringIO
import sys
from collections import defaultdict
from subprocess import call
import pytest

def next_line(f):
    try:
        return next(f)
    except StopIteration:
        return ''

def read_event_spans_conll(f, path=''):
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
        label = fields[2]
        if label != '_':
            if label != 'B-E':
                print("Format error in file %s, sentence %d, token %d: "
                      "an event begins with strange marker: %s"
                      %(path, sent, token, label))
            start = end = token
            while True:
                line = next_line(f)
                fields = line.strip().split('\t')
                if (not line) or line == '\n' or fields[2] != 'I-E':
                    break
                end = int(fields[0])
            spans.add((sent, start, end))
        else:
            line = next_line(f)
    return spans

def read_polarity_spans_conll(f, path=''):
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
        label = fields[4]
        if label != '_':
            if not label.startswith('B-'):
                print("Format error in file %s, sentence %d, token %d: "
                      "an polarity span begins with strange marker: %s"
                      %(path, sent, token, label))
            label = label[2:]
            start = end = token
            while True:
                line = next_line(f)
                fields = line.strip().split('\t')
                if (not line) or line == '\n' or fields[4] != ('I-' + label):
                    break
                end = int(fields[0])
            spans.add((sent, start, end, label))
        else:
            line = next_line(f)
    return spans

def read_certainty_spans_conll(f, path=''):
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
        label = fields[3]
        if label != '_':
            if not label.startswith('B-'):
                print("Format error in file %s, sentence %d, token %d: "
                      "an certainty span begins with strange marker: %s"
                      %(path, sent, token, label))
            label = label[2:]
            start = end = token
            while True:
                line = next_line(f)
                fields = line.strip().split('\t')
                if (not line) or line == '\n' or fields[3] != ('I-' + label):
                    break
                end = int(fields[0])
            spans.add((sent, start, end, label))
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

def compare_spans(key, res):
    key_set = set(key)
    res_set = set(res)
    return len(res_set.intersection(key_set)), len(key_set), len(res_set)

def compare_dependent_spans(key, res, key_event, res_event):
    key_set = set(key)
    res_set = set(res)
    for s in key: assert s[:3] in key_event
    correct = sum(1 for s in res
                  if s in key_set and s[:3] in res_event and s[:3] in key_event)
    return correct, len(key_set), len(res_set)

def compare_dependent_spans2(key1, res1, key2, res2, key_event, res_event):
    key = {}
    for s in key1: 
        assert s[:3] in key_event
        key[s[:3]] = s
    for s in key2: 
        assert s[:3] in key_event
        assert s[:3] in key
        key[s[:3]] = (key[s[:3]], s) 
    res = {}
    for s in res1: res[s[:3]] = s
    for s in res2: res[s[:3]] = (res[s[:3]], s) if s[:3] in res else s 
    correct = sum(1 for s in res
                  if s in key and res[s] == key[s] and s in res_event and s in key_event)
    return correct, len(key), len(res)

def first_five_sentences(f):
    sent = 1
    for line in f:
        if line == '\n':
            sent += 1
            if sent > 5:
                return
        yield line

def test_read_spans_conll():
    s = '1\tA\t_\t_\t_\n'
    assert len(read_event_spans_conll(StringIO(s))) == 0
    assert len(read_polarity_spans_conll(StringIO(s))) == 0
    assert len(read_certainty_spans_conll(StringIO(s))) == 0
    s = '1\tB\tB-E\tB-CERTAIN\tB-POS'
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 1
    assert list(event_spans)[0] == (1, 1, 1)
    polarity_spans = read_polarity_spans_conll(StringIO(s))
    assert len(polarity_spans) == 1
    assert list(polarity_spans)[0] == (1, 1, 1, 'POS')
    certainty_spans = read_certainty_spans_conll(StringIO(s))
    assert len(certainty_spans) == 1
    assert list(certainty_spans)[0] == (1, 1, 1, 'CERTAIN')
    s = '1\tA\tB-E\tB-CERTAIN\tB-POS\n2\tB\tI-E\tI-CERTAIN\tI-POS'
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 1
    assert list(event_spans)[0] == (1, 1, 2)
    polarity_spans = read_polarity_spans_conll(StringIO(s))
    assert len(polarity_spans) == 1
    assert list(polarity_spans)[0] == (1, 1, 2, 'POS')
    certainty_spans = read_certainty_spans_conll(StringIO(s))
    assert len(certainty_spans) == 1
    assert list(certainty_spans)[0] == (1, 1, 2, 'CERTAIN')
    s = '1\tA\tB-E\tB-CERTAIN\tB-POS\n2\tB\tB-E\tB-CERTAIN\tB-POS'
    assert len(read_event_spans_conll(StringIO(s))) == 2
    assert len(read_polarity_spans_conll(StringIO(s))) == 2
    assert len(read_certainty_spans_conll(StringIO(s))) == 2
#     with pytest.raises(AssertionError):
#         s = '1\tA\tB-E\tB-CERTAIN\tB-POS\n2\tB\tB-X\tB-CERTAIN\tI-POS'
#         read_event_spans_conll(StringIO(s))
#     with pytest.raises(AssertionError):
#         s = '1\tA\tB-E\tB-CERTAIN\tB-POS\n2\tB\tB-E\tB-CERTAIN\tI-NEG'
#         read_polarity_spans_conll(StringIO(s))
#     with pytest.raises(AssertionError):
#         s = '1\tA\tB-E\tB-CERTAIN\tB-POS\n2\tB\tB-E\tI-XYZ\tB-POS'
#         read_certainty_spans_conll(StringIO(s))
    
def test_compare_spans():
    assert compare_spans(((1,1,2), (1,3,3)), 
                         ((1,1,2), (1,3,3))) == (2,2,2)
    assert compare_spans(((1,1,2), (1,3,3)), 
                         ((1,1,2), )) == (1,2,1)
    assert compare_spans(((1,1,2), ), 
                         ((1,1,2), (1,3,3))) == (1,1,2)
    assert compare_spans(((1,1,2), (1,4,5)), 
                         ((1,1,2), (1,3,3))) == (1,2,2)
    assert compare_dependent_spans(((1,1,2,'POS'), (1,3,3,'NEG')), # key
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (2,2,2), 'should have been all correct'
    assert compare_dependent_spans(((1,1,2,'POS'), (1,3,3,'NEG')), # key
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,5)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong event span'
    assert compare_dependent_spans(((1,1,2,'POS'), (1,3,3,'NEG')), # key
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), ) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of missing event span'
    assert compare_dependent_spans(((1,1,2,'POS'), (1,4,5,'NEG')), # key
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res
                                   ((1,1,2), (1,4,5)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity span'
    assert compare_dependent_spans(((1,1,2,'POS'), (1,3,3,'XXX')), # key
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity label'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (2,2,2), 'should have been all correct'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'XXX')), # key (polarity)
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity label'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'CERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong certainty label'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), (1,5,6,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,6,'UNCERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,3), "polarity and certainty spans should be counted as two because they aren't exactly the same"
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), (1,3,6,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,6,'UNCERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,6)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong event span'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), (1,3,3,'NEG')), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), ), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of missing certainty span'
    assert compare_dependent_spans2(((1,1,2,'POS'), (1,3,3,'NEG')), # key (polarity)
                                   ((1,1,2,'POS'), ), # res (polarity)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # key (certainty)
                                   ((1,1,2,'CERTAIN'), (1,3,3,'UNCERTAIN')), # res (certainty)
                                   ((1,1,2), (1,3,3)), # key_event
                                   ((1,1,2), (1,3,3)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of missing polarity span'

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
    test_compare_spans()
    sys.stderr.write('Passed all tests.\n')

if __name__ == '__main__':
    call('date')
    test_all() # never run evaluation script without thorough testing
    data = defaultdict(list)
    if len(os.listdir('key')) < len(os.listdir('response')):
        sys.stderr.write('WARN: response folder holds more files than key folder. Some files will be ignored.\n')
    for fname in os.listdir('key/factuality'):
        path = 'key/factuality/%s' %fname
        with open(path) as f: key_event = read_event_spans_conll(first_five_sentences(f), path)
        with open(path) as f: key_polarity = read_polarity_spans_conll(first_five_sentences(f), path)
        with open(path) as f: key_certainty = read_certainty_spans_conll(first_five_sentences(f), path)
        path = 'response/factuality/%s' %fname
        with open(path) as f: res_event = read_event_spans_conll(first_five_sentences(f), path)
        with open(path) as f: res_polarity = read_polarity_spans_conll(first_five_sentences(f), path)
        with open(path) as f: res_certainty = read_certainty_spans_conll(first_five_sentences(f), path)
        data['event'].append(compare_spans(key_event, key_event))
        data['polarity'].append(compare_dependent_spans(key_polarity, res_polarity, key_event, key_event))
        data['certainty'].append(compare_dependent_spans(key_certainty, res_certainty, key_event, key_event))
        data['polarity+certainty'].append(compare_dependent_spans2(key_polarity, res_polarity, key_certainty, res_certainty, key_event, key_event))
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
