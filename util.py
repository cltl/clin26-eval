'''
Created on Dec 2, 2015

@author: Minh Ngoc Le
'''

def first_n_sentences(f, n):
    if n <= 0: return f
    sent = 1
    for line in f:
        yield line
        if line == '\n':
            sent += 1
            if sent > n:
                return


def next_line(f):
    try:
        return next(f)
    except StopIteration:
        return ''


def test_first_n_sentences():
    lines = ['abc', '\n']*100
    assert sum(1 for line in first_n_sentences(lines, 0) if line == '\n') == 100
    assert sum(1 for line in first_n_sentences(lines, 1) if line == '\n') == 1
    assert sum(1 for line in first_n_sentences(lines, 2) if line == '\n') == 2

test_first_n_sentences()