import heapq
from collections import defaultdict
from itertools import combinations
from operator import itemgetter

import pandas as pd
from parse import *
import statistics
import matplotlib.pyplot as plt
import numpy as np


def parse_line(line):
    # if last == '#':
    #     if line == '#':
    #         return '##', '##'
    # elif last == '#':
    #     pass
    # parsed = parse("<{},{};{}>{}\t{}", line)
    parsed = parse("<{},{}>{}", line)
    line_type = 0
    if not parsed:
        parsed = parse("<{}>{}<! $I={} $Q={} $P={} $L={} $H={} $X={}>{}", line)
        line_type = 2
    if not parsed:
        parsed = parse("<{}>{}<! $I={} $Q={} $P={} $L={} $H={}>{}", line)
        line_type = 3
    if not parsed:
        parsed = parse("<{}>{}<! $I={} $Q={} $P={} $L={}>{}", line)
        line_type = 4
    if not parsed:
        parsed = parse("<{}>{}<! $I={} $Q={} $P={}>{}", line)
        line_type = 5
    if not parsed:
        parsed = parse('#', line)
        line_type = 1
    if not parsed:
        parsed = parse('', line)
        line_type = 1
    if not parsed:
        print(line)
        raise AssertionError("Parsing failed")
    if line_type in [2, 3, 4, 5]:
        parsed = list(parsed) + ([''] * (line_type - 2))
    return parsed, line_type

def convert_to_strings(big_text, line_count=5000):
    with open(big_text) as corpus:
        paragraphs = []
        line = corpus.readline().rstrip('\n').rstrip('.')
        cur_paragraph = []
        i=0
        while line and i<line_count:
            cur_paragraph = cur_paragraph + line.split(' ')
            line = corpus.readline()
            if not line or line == '\n':
                paragraphs.append(cur_paragraph)
                cur_paragraph = []
            if line != '\n':
                line = line.rstrip('\n').rstrip('.')
        i += 1
        return paragraphs



def convert_to_strings_voynich(df):
    i = 0
    paragraph = ''
    output = []
    while i < len(df):
        line = df.iloc[i, :]
        if line['Ending'] == '@P0':
            if paragraph:
                output.append(paragraph)
            paragraph = line['Line']
        elif line['Ending'] == '+P0':
            paragraph += ('.' + line['Line'])
        elif line['Ending'] == '=Pt':
            paragraph += ('.' + line['Line'])
        else:
            output.append(paragraph)
            paragraph = ''
            # pass
            # paragraph += line['Line']
            # TODO these are all different
        i += 1
    for k, paragraph in enumerate(output):

        output[k] = paragraph.replace('<->', '.')
        output[k] = paragraph.replace('<\\$>', '')
        output[k] = paragraph.split('.')
 
    return output

def get_levenshtein(pair):
    if not (pair[0].isalpha() and pair[1].isalpha()):
        return -1
    seq1, seq2 = pair
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def create_df(file, hand):
    with open(file) as corpus:
        row = 0
        parsed_lines = []
        line = corpus.readline()
        line_def, line_type = parse_line(line)
        # row += 1
        while line:
            line = corpus.readline()
            parsed, line_type = parse_line(line)
            if line_type == 0:
                # print(list(parsed))
                # print(line_def)
                parsed_lines.append(line_def + list(parsed))
            elif line_type in [2, 3, 4, 5]:
                line_def = parsed
            # elif line_type == 2:
            #     line_def = parsed
            elif line_type == 1:
                pass
            else:
                raise NotImplementedError("This Line Type Not implemented")
            # row += 1
    column_names = ['Folio', 'Empty1', 'I', 'Q', 'P', 'L', 'H', 'X', 'Empty2', 'Folio', 'Ending', 'Line']
    df = pd.DataFrame(parsed_lines, columns=column_names)
    df.drop(['Empty1', 'Empty2'], axis=1, inplace=True)
    df['Line'] = df['Line'].apply(lambda s: s.lstrip())
    df['Line'] = df['Line'].apply(lambda s: s.rstrip())
    A_df = df[df['H'] == '1']
    B_df = df[df['H'] == '2']
    if hand=='A':
        str_list_output = convert_to_strings_voynich(A_df)
    if hand=='B':
        str_list_output = convert_to_strings_voynich(B_df)
    else:
        str_list_output = convert_to_strings_voynich(df)
    return str_list_output

def gen_comps(str_list_output, neg_dist=1):
    word_comp = defaultdict(list)
    comp_count = defaultdict(lambda: 0)
    for paragraph in str_list_output:
        i = 0
        n = 10
        while i < len(paragraph) - n:
            window = paragraph[i:i + n]
            word1 = window[0]
            for k, word_2 in enumerate(window[1:]):
                if comp_count[word1, word_2] == 0:
                    if comp_count[word_2, word1] == 0:
                        word_comp[word1, word_2].append(k)
                        comp_count[word1,word_2] += 1
                    else:
                        word_comp[word_2, word1].append(neg_dist * k)
                        comp_count[word_2, word1] += 1
                else:
                    word_comp[word1, word_2].append(k)
                    comp_count[word1,word_2] += 1
            i += 1
        for k, word1 in enumerate(paragraph[i:]):
            for m, word_2 in enumerate(paragraph[(i+k+1):]):
                if comp_count[word1, word_2] == 0:
                    if comp_count[word_2, word1] == 0:
                        word_comp[word1, word_2].append(m)
                        comp_count[word1,word_2] += 1
                    else:
                        word_comp[word_2, word1].append(neg_dist * m)
                        comp_count[word_2, word1] += 1
                else:
                    word_comp[word1, word_2].append(m)
                    comp_count[word1,word_2] += 1

            # for k, word in enumerate(window):
            #     for m, word_2 in enumerate(window[(k+1):]):
            #         if comp_count[word, word_2] == 0:
            #             if comp_count[word_2, word] == 0:
            #                 word_comp[word, word_2].append(m)
            #                 comp_count[word,word_2] += 1
            #             else:
            #                 word_comp[word_2, word].append(neg_dist * m)
            #                 comp_count[word_2, word] += 1
            #         else:
            #             word_comp[word, word_2].append(m)
            #             comp_count[word,word_2] += 1
            # for comb in combinations(window, 3):
            #
            #     word_comp[comb] += 2
            #     word_comb[comb].append(dist)
    return word_comp, comp_count

def analysis(word_comp, comp_count, n=10000):

    topitems = heapq.nlargest(n, comp_count.items(), key=itemgetter(1))

    topitemsdict = dict(topitems)
    topitemsdict = {item: topitemsdict[item] for item in topitemsdict.keys() if (item[0].isalpha() and item[1].isalpha())}
    top_comps = {item: word_comp[item] for item in topitemsdict.keys()}


    pair = list(top_comps.keys())[0]
    # vis = top_comps[pair]

    stdevs = {a:statistics.stdev(top_comps[a]) for a in top_comps.keys()}
    median_dist = {a:statistics.median(top_comps[a]) for a in top_comps.keys()}
    # mode_dist = {a:statistics.mode(top_comps[a]) for a in top_comps.keys()}

    # lowest_stdevs = heapq.nsmallest(10, stdevs)
    # highest_stdevs = heapq.nlargest(10, stdevs)
    

    # m = min(stdevs, key=stdevs.get)

    # levenshteins = {item:get_levenshtein(item) for item in word_comp.keys()}
    levenshteins_top = {item:get_levenshtein(item) for item in topitemsdict.keys()}

    # x = [levenshteins[item] for item in stdevs.keys()]
    # y = [stdevs[item] for item in stdevs.keys()]
    # y1 = [comp_count[item] for item in stdevs.keys()]
    # y2 = [median_dist[item] for item in stdevs.keys()]

    return topitemsdict, stdevs, levenshteins_top, comp_count


def evaluate_corpus(file, num_lines=6000, hand='Both', voynich=False):
    if voynich:
        paragraphs = create_df(file, hand)
    else:
        paragraphs = convert_to_strings(file, num_lines)
    word_comp, comp_count = gen_comps(paragraphs)
    return analysis(word_comp, comp_count)


def main():
    corpora = ['war_peace.txt', 'don_quixote.txt', 'great_expectations.txt']
    voynich_file = 'voynich_data.txt'

    corpora_output = {}

    for corpus in corpora:
        corpora_output[corpus] = evaluate_corpus(corpus)

    topitems_v, stdevs_v, lev_top_v, comp_count_v = evaluate_corpus(voynich_file, voynich=True)

if __name__ == "__main__":
    main()