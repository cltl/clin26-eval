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

import pytest

import numpy as np
from util import first_n_sentences, next_line
import re

def read_event_spans_conll(f, path=''):
    spans = set()
    sent = 1
    line = next_line(f)
    id2tokens = defaultdict(list)
    while line:
        if line == '\n':
            sent += 1
        else:
            fields = line.strip().split('\t')
            if fields[2] != '_':
                type_, label, event_id = fields[2].split('-')
                assert type_ in ['I', 'B']
                assert label == 'E'
                assert (type_ == 'B' and len(id2tokens[event_id]) == 0) or \
                        (type_ == 'I' and len(id2tokens[event_id]) > 0), \
                        "Format error in file %s, sentence %d, token %s: %s" \
                        %(path, sent, fields[0], fields[2])
                id2tokens[event_id].append(int(fields[0])) 
        line = next_line(f)
    for _, tokens in id2tokens.iteritems():
        spans.add((sent, tuple(tokens)))
    return spans


def read_tokens_conll(cols, f, path=''):
    tokens = []
    for _ in cols: tokens.append(set())
    tokens = tuple(tokens)
    sent = 1
    line = next_line(f)
    while line:
        if line == '\n':
            sent += 1
            line = next_line(f)
            continue
        fields = line.strip().split('\t')
        token = int(fields[0])
        for i, col in enumerate(cols):
            label = re.sub('-\w+$', '', fields[col])
            if label != '_':
                tokens[i].add((sent, token, label))
        line = next_line(f)
    return tokens


def read_generic_spans_conll(col, f, path):
    spans = set()
    id2tokens = defaultdict(list)
    sent = 1
    line = next_line(f)
    while line:
        if line == '\n':
            sent += 1
        else:
            fields = line.strip().split('\t')
            token = int(fields[0])
            label = fields[col]
            if label != '_':
                type_, label, id_ = label.split('-')
                assert type_ in ['I', 'B']
                assert (type_ == 'B' and len(id2tokens[(label, id_)]) == 0) or \
                        (type_ == 'I' and len(id2tokens[(label, id_)]) > 0), \
                        "Format error in file %s, sentence %d, token %s: %s" \
                        %(path, sent, fields[0], fields[col])
                id2tokens[(label, id_)].append(token)
        line = next_line(f)
    for (label, _), tokens in id2tokens.iteritems():
        spans.add((sent, tuple(tokens), label))
    return spans

def group_by_type(spans):
    spans_by_type = defaultdict(list)
    for span in spans:
        type_ = re.sub('^[BI]-', '', span[2])
        spans_by_type[type_].append(span[:2])
    return spans_by_type

def read_polarity_spans_conll(f, path=''):
    return read_generic_spans_conll(4, f, path)

def read_certainty_spans_conll(f, path=''):
    return read_generic_spans_conll(3, f, path)

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

def compare_tokens(key, res):
    key_set = set(key)
    res_set = set(res)
    return len(res_set.intersection(key_set)), len(key_set), len(res_set)

def compare_dependent_spans(key, res, key_event, res_event):
    key_set = set(key)
    res_set = set(res)
    for s in key: 
        if s[:2] not in key_event:
            sys.stderr.write("WARN: Error in gold data: Span does not match any event: "
                             "sentence %d, tokens %s.\n" %(s[0], str(s[1])))
    correct = sum(1 for s in res
                  if s in key_set and s[:2] in res_event and s[:2] in key_event)
    return correct, len(key_set), len(res_set)

def intersecting(tuple1, tuple2):
    return len(set(tuple1).intersection(tuple2)) > 0

def compare_dependent_spans2(key1, res1, key2, res2, key_event, res_event):
    '''
    Assume span format: (<sentence id>, (<tuple ids>) [, <label>])
    '''
    key = {}
    assert len(key1) == len(key2) <= len(key_event)
    for s in key1:
        if s[:2] not in key_event:
            sys.stderr.write("WARN: Error in gold data: Span does not match any event: "
                             "sentence %d, tokens %s.\n" %(s[0], str(s[1])))
        key[s[:2]] = (s, None)
    for s in key2:
        if s[:2] not in key_event:
            sys.stderr.write("WARN: Error in gold data: Span does not match any event: "
                             "sentence %d, tokens %s.\n" %(s[0], str(s[1])))
        if s[:2] not in key:
            sys.stderr.write("WARN: Error in gold data: Span does not match any span of other event type: "
                             "sentence %d, tokens %s.\n" %(s[0], str(s[1])))
            key[s[:2]] = (None, s)
        else:
            key[s[:2]] = (key[s[:2]][0], s)
    res = [] # because they may not completely match, we can't index them by span
    res1 = sorted(res1)
    res2 = sorted(res2)
    it1 = iter(res1)
    it2 = iter(res2)
    s1 = s2 = None
    try: # zip the two lists of spans
        s1 = next(it1)
        s2 = next(it2)
        while True:
            if (s1[:2] == s2[:2] or # fully match, phew...
                intersecting(s1[1], s2[1])): # partially match
                res.append((s1, s2))
                s1 = None 
                s2 = None 
                s1 = next(it1)
                s2 = next(it2)
            else: 
                if s1[:2] < s2[:2]:
                    res.append((s1, None))
                    s1 = None
                    s1 = next(it1)
                else:
                    res.append((None, s2))
                    s2 = None
                    s2 = next(it2)
    except StopIteration:
        # add the rest of any list to res
        if s1 is not None: res.append((s1, None))
        if s2 is not None: res.append((None, s2))
        for s1 in it1: res.append((s1, None))
        for s2 in it2: res.append((None, s2))
    correct = sum(1 for s1, s2 in res
                  if s1 is not None and s2 is not None and s1[:2]==s2[:2] and s1[:2] in res_event 
                  and s1[:2] in key and (s1, s2) == key[s1[:2]])
    return correct, len(key), len(res)

def test_read_tokens_conll():
    s = '1\tA\t_\t_\t_\n'
    events, certainty, polarity = read_tokens_conll((2, 3, 4), StringIO(s))
    assert len(events) == 0
    assert len(certainty) == 0
    assert len(polarity) == 0
    s = '1\tB\tB-E-1\tB-CERTAIN-1\tB-POS-1'
    events, certainty, polarity = read_tokens_conll((2, 3, 4), StringIO(s))
    assert len(events) == 1
    assert (1, 1, 'B-E') in events 
    assert len(polarity) == 1
    assert (1, 1, 'B-POS') in polarity
    assert len(certainty) == 1
    assert (1, 1, 'B-CERTAIN') in certainty
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\tI-E-1\tI-CERTAIN-1\tI-POS-1')
    events, certainty, polarity = read_tokens_conll((2, 3, 4), StringIO(s))
    assert len(events) == 2
    assert (1, 1, 'B-E') in events
    assert (1, 2, 'I-E') in events
    assert len(polarity) == 2
    assert (1, 1, 'B-POS') in polarity 
    assert (1, 2, 'I-POS') in polarity 
    assert len(certainty) == 2
    assert (1, 1, 'B-CERTAIN') in certainty
    assert (1, 2, 'I-CERTAIN') in certainty

def test_read_spans_conll():
    s = '1\tA\t_\t_\t_\n'
    assert len(read_event_spans_conll(StringIO(s))) == 0
    assert len(read_polarity_spans_conll(StringIO(s))) == 0
    assert len(read_certainty_spans_conll(StringIO(s))) == 0
    s = '1\tB\tB-E-1\tB-CERTAIN-1\tB-POS-1'
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 1
    assert list(event_spans)[0] == (1, (1,))
    polarity_spans = read_polarity_spans_conll(StringIO(s))
    assert len(polarity_spans) == 1
    assert list(polarity_spans)[0] == (1, (1,), 'POS')
    certainty_spans = read_certainty_spans_conll(StringIO(s))
    assert len(certainty_spans) == 1
    assert list(certainty_spans)[0] == (1, (1,), 'CERTAIN')
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\tI-E-1\tI-CERTAIN-1\tI-POS-1')
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 1
    assert list(event_spans)[0] == (1, (1, 2))
    polarity_spans = read_polarity_spans_conll(StringIO(s))
    assert len(polarity_spans) == 1
    assert list(polarity_spans)[0] == (1, (1, 2), 'POS')
    certainty_spans = read_certainty_spans_conll(StringIO(s))
    assert len(certainty_spans) == 1
    assert list(certainty_spans)[0] == (1, (1, 2), 'CERTAIN')
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\tB-E-2\tB-CERTAIN-2\tB-POS-2')
    assert len(read_event_spans_conll(StringIO(s))) == 2
    assert len(read_polarity_spans_conll(StringIO(s))) == 2
    assert len(read_certainty_spans_conll(StringIO(s))) == 2
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\t_\t_\t_\n'
         '3\tC\tI-E-1\tI-CERTAIN-1\tI-POS-1')
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 1, 'Discontinuous event span'
    assert list(event_spans)[0] == (1, (1, 3))
    polarity_spans = read_polarity_spans_conll(StringIO(s))
    assert len(polarity_spans) == 1, 'Discontinuous polarity span'
    assert list(polarity_spans)[0] == (1, (1, 3), 'POS')
    certainty_spans = read_certainty_spans_conll(StringIO(s))
    assert len(certainty_spans) == 1, 'Discontinuous certainty span'
    assert list(certainty_spans)[0] == (1, (1, 3), 'CERTAIN')
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\t_\t_\t_\n'
         '3\tC\tI-E-1\tI-CERTAIN-1\tI-POS-1\n'
         '4\tD\tB-E-2\t_\t_')
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(event_spans) == 2, 'Discontinuous span followed by singleton span'
    assert list(event_spans)[0] == (1, (1, 3))
    assert list(event_spans)[1] == (1, (4,))
    assert len(read_polarity_spans_conll(StringIO(s))) == 1
    assert len(read_certainty_spans_conll(StringIO(s))) == 1
    s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
         '2\tB\tB-E-2\t_\tB-NEG-2\n'
         '3\tC\tI-E-1\tI-CERTAIN-1\tI-POS-1\n'
         '4\tD\tI-E-2\t_\tI-NEG-2')
    event_spans = read_event_spans_conll(StringIO(s))
    assert len(read_polarity_spans_conll(StringIO(s))) == 2, 'Discontinuous, intertwined spans'
    assert len(read_certainty_spans_conll(StringIO(s))) == 1
    with pytest.raises(AssertionError):
        s = ('1\tA\tB-E-1\tB-CERTAIN-1\tB-POS-1\n'
             '2\tB\tB-X-2\tB-CERTAIN-1\tI-POS-1')
        read_event_spans_conll(StringIO(s))
        
        
def test_compare_spans():
    # span format: (<sentence id>, (<tuple ids>) [, <label>])
    # answer format: (<correct>, <total gold>, <total predicted>)
    assert compare_spans((), ()) == (0,0,0)
    assert compare_spans(((1,(1,2)), (1,(3,))), 
                         ((1,(1,2)), (1,(3,)))) == (2,2,2)
    assert compare_spans(((1,(1,2)), (1,(3,))), 
                         ((1,(1,2)), )) == (1,2,1)
    assert compare_spans(((1,(1,2)), ), 
                         ((1,(1,2)), (1,(3,)))) == (1,1,2)
    assert compare_spans(((1,(1,2)), (1,(4,5))), 
                         ((1,(1,2)), (1,(3,)))) == (1,2,2)
    assert compare_dependent_spans((), # key
                                   (), # res
                                   (), # key_event
                                   () # res_event
                                   ) == (0,0,0), 'handle empty tuples'
    assert compare_dependent_spans(((1,(1,2),'POS'), (1,(3,),'NEG')), # key
                                   ((1,(1,2),'POS'), (1,(3,),'NEG')), # res
                                   ((1,(1,2)), (1,(3,))), # key_event
                                   ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (2,2,2), 'should have been all correct'
    assert compare_dependent_spans(((1,(1,2),'POS'), (1,(3,),'NEG')), # key
                                   ((1,(1,2),'POS'), (1,(3,),'NEG')), # res
                                   ((1,(1,2)), (1,(3,))), # key_event
                                   ((1,(1,2)), (1,3,5)) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong event span'
    assert compare_dependent_spans(((1,(1,2),'POS'), (1,(3,),'NEG')), # key
                                   ((1,(1,2),'POS'), (1,(3,),'NEG')), # res
                                   ((1,(1,2)), (1,(3,))), # key_event
                                   ((1,(1,2)), ) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of missing event span'
    assert compare_dependent_spans(((1,(1,2),'POS'), (1,(4,5),'NEG')), # key
                                   ((1,(1,2),'POS'), (1,(3,),'NEG')), # res
                                   ((1,(1,2)), (1,(4,5))), # key_event
                                   ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity span'
    assert compare_dependent_spans(((1,(1,2),'POS'), (1,(3,),'XXX')), # key
                                   ((1,(1,2),'POS'), (1,(3,),'NEG')), # res
                                   ((1,(1,2)), (1,(3,))), # key_event
                                   ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity label'
    assert compare_dependent_spans2((), # key (polarity)
                                    (), # res (polarity)
                                    (), # key (certainty)
                                    (), # res (certainty)
                                    (), # key_event
                                    () # res_event
                                   ) == (0,0,0), 'handle empty tuples'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(3,),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (2,2,2), 'should have been all correct'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'XXX')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(3,),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong polarity label'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(3,),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'CERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong certainty label'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(5,6),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,4,5,6),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), "polarity and certainty spans should be counted as one because they are partially overlapping"
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(5,6),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,4),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,3), "polarity and certainty spans should be counted as two because they are disjoint"
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(3,4,5,6),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,4,5,6),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,4,5,6))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of wrong event span'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), (1,(3,),'NEG')), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), ), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
                                   ) == (1,2,2), 'one should have been incorrect because of missing certainty span'
    assert compare_dependent_spans2(((1,(1,2),'POS'), (1,(3,),'NEG')), # key (polarity)
                                    ((1,(1,2),'POS'), ), # res (polarity)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # key (certainty)
                                    ((1,(1,2),'CERTAIN'), (1,(3,),'UNCERTAIN')), # res (certainty)
                                    ((1,(1,2)), (1,(3,))), # key_event
                                    ((1,(1,2)), (1,(3,))) # res_event
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
    test_read_tokens_conll()
    sys.stderr.write('Passed all tests.\n')

test_all() # never run evaluation script without thorough testing

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Score the response of a system at factuality.')
    parser.add_argument('key', help='path to a directory containing all key files')
    parser.add_argument('response', help='path to a directory containing all response files')
    parser.add_argument('measurement', help='measure performance on tokens or spans. possible values: tokens, ')
    parser.add_argument('-n', type=int, default=5, help='number of sentences to consider, 0 for all')
    args = parser.parse_args()

    call('date')
    data = defaultdict(list)
    if len(os.listdir(args.key)) < len(os.listdir(args.response)):
        sys.stderr.write('WARN: response folder holds more files than key folder. Some files will be ignored.\n')
    for fname in os.listdir(args.key):
        if args.measurement == 'spans':
            sys.stderr.write("Current file name: %s\n" %fname)
            path = os.path.join(args.key, fname)
            with open(path) as f: key_event = read_event_spans_conll(first_n_sentences(f, args.n), path)
            with open(path) as f: key_polarity = read_polarity_spans_conll(first_n_sentences(f, args.n), path)
            with open(path) as f: key_certainty = read_certainty_spans_conll(first_n_sentences(f, args.n), path)
            path = os.path.join(args.response, fname)
            if os.path.exists(path):
                with open(path) as f: res_event = read_event_spans_conll(first_n_sentences(f, args.n), path)
                with open(path) as f: res_polarity = read_polarity_spans_conll(first_n_sentences(f, args.n), path)
                with open(path) as f: res_certainty = read_certainty_spans_conll(first_n_sentences(f, args.n), path)
            else:
                res_event = res_polarity = res_certainty = set() 
            data['event'].append(compare_spans(key_event, res_event))
            data['polarity'].append(compare_dependent_spans(key_polarity, res_polarity, key_event, res_event))
            key_by_type, res_by_type = group_by_type(key_polarity), group_by_type(res_polarity)
            for type_ in set(key_by_type).union(set(res_by_type)):
                data['polarity:' + type_].append(compare_dependent_spans(key_by_type[type_], res_by_type[type_], key_event, res_event))
            data['certainty'].append(compare_dependent_spans(key_certainty, res_certainty, key_event, res_event))
            key_by_type, res_by_type = group_by_type(key_certainty), group_by_type(res_certainty)
            for type_ in set(key_by_type).union(set(res_by_type)):
                data['certainty:' + type_].append(compare_dependent_spans(key_by_type[type_], res_by_type[type_], key_event, res_event))
            data['polarity+certainty'].append(compare_dependent_spans2(key_polarity, res_polarity, key_certainty, res_certainty, key_event, res_event))
        elif args.measurement == 'tokens':
            path = os.path.join(args.key, fname)
            with open(path) as f: 
                key_event, key_polarity, key_certainty = read_tokens_conll((2, 3, 4), first_n_sentences(f, args.n), path)
            path = os.path.join(args.response, fname)
            if os.path.exists(path):
                with open(path) as f:
                    res_event, res_polarity, res_certainty = read_tokens_conll((2, 3, 4), first_n_sentences(f, args.n), path)
            else:
                res_event = res_polarity = res_certainty = set() 
            data['event'].append(compare_tokens(key_event, res_event))
            data['polarity'].append(compare_tokens(key_polarity, res_polarity))
            key_by_type, res_by_type = group_by_type(key_polarity), group_by_type(res_polarity)
            for type_ in set(key_by_type).union(set(res_by_type)):
                data['polarity:' + type_].append(compare_tokens(key_by_type[type_], res_by_type[type_]))
            data['certainty'].append(compare_tokens(key_certainty, res_certainty))
            key_by_type, res_by_type = group_by_type(key_certainty), group_by_type(res_certainty)
            for type_ in set(key_by_type).union(set(res_by_type)):
                data['certainty:' + type_].append(compare_tokens(key_by_type[type_], res_by_type[type_]))
        else:
            raise ValueError('Unsupported measurement: %s' %args.measurement)
    for name in sorted(data.keys()):
        print('\n\nPerformance (%s %s):\n' %(name, args.measurement))
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
