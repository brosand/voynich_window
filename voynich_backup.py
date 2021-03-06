import heapq
from collections import defaultdict, Counter
from itertools import combinations, chain
from operator import itemgetter
import math
import os
import datetime
import time
from tqdm import tqdm

import pandas as pd
from parse import *
import statistics
import matplotlib
matplotlib.use('Agg')
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


def convert_to_strings(big_text, lines=True, line_count=6000):
    failed=False
    if not lines:
        with open(big_text) as corpus:
            paragraphs = []
            try:
                line = corpus.readline().rstrip('\n')
            except:
                print('Analysis failed 62')
                failed=True
                return None
    if not failed:
        line = line.split(" ")
        for i in range(line_count):
            section = line[i*10:(i*10 + 10)]
            tmp = (' ').join(section)
            paragraphs.append(tmp)
            # paragraphs.append[' '.join(line[i*10:(i*10+10)])]
        with open(big_text) as corpus:
            paragraphs = []
            line = corpus.readline().rstrip('\n').rstrip('.')
            cur_paragraph = []
            # i = 0
            # while line and i < line_count:
            for i in tqdm(range(line_count)):
                if not line:
                    break
                cur_paragraph = cur_paragraph + line.split(' ')
                # try:
                #     line = corpus.readline()
                # except:
                #     print('Analysis failed 81')
                #     break
                if not line or line == '\n':
                    paragraphs.append(cur_paragraph)
                    cur_paragraph = []
                if line != '\n':
                    line = line.rstrip('\n').rstrip('.')
            # i += 1
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
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def create_df(file, hand):
    with open(file) as corpus:
        # row = 0
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
    column_names = ['Folio', 'Empty1', 'I', 'Q', 'P', 'L', 'H', 'X', 'Empty2',
                    'Folio', 'Ending', 'Line']
    df = pd.DataFrame(parsed_lines, columns=column_names)
    df.drop(['Empty1', 'Empty2'], axis=1, inplace=True)
    df['Line'] = df['Line'].apply(lambda s: s.lstrip())
    df['Line'] = df['Line'].apply(lambda s: s.rstrip())
    A_df = df[df['H'] == '1']
    B_df = df[df['H'] == '2']
    if hand == 'A':
        str_list_output = convert_to_strings_voynich(A_df)
    if hand == 'B':
        str_list_output = convert_to_strings_voynich(B_df)
    else:
        str_list_output = convert_to_strings_voynich(df)
    return str_list_output

def words_weight(word_pair, size_of_corpus, cnt):
    
    frac = math.log(min(size_of_corpus / cnt[word_pair[0]], size_of_corpus / cnt[word_pair[1]]))
    return frac

def gen_comps(str_list_output, neg_dist=1, weighted=False):
    word_comp = defaultdict(list)
    comp_count = defaultdict(lambda: 0)
    if not str_list_output:
        return 'fail'
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
                        comp_count[word1, word_2] += 1
                    else:
                        word_comp[word_2, word1].append(neg_dist * k)
                        comp_count[word_2, word1] += 1
                else:
                    word_comp[word1, word_2].append(k)
                    comp_count[word1, word_2] += 1
            i += 1
        for k, word1 in enumerate(paragraph[i:]):
            for m, word_2 in enumerate(paragraph[(i+k+1):]):
                if comp_count[word1, word_2] == 0:
                    if comp_count[word_2, word1] == 0:
                        word_comp[word1, word_2].append(m)
                        comp_count[word1, word_2] += 1
                    else:
                        word_comp[word_2, word1].append(neg_dist * m)
                        comp_count[word_2, word1] += 1
                else:
                    word_comp[word1, word_2].append(m)
                    comp_count[word1, word_2] += 1

    if weighted==True:
        raise NotImplementedError
        # l = sum([len(line) for line in str_list_output])
        # cnt = Counter(list(chain(*str_list_output)))
        # for comp in comp_count.keys():
        #     comp_count[comp] = comp_count[comp] * words_weight(comp, l, cnt)
    return word_comp, comp_count


def analysis(word_comp, comp_count, n=5000):

    topitems = heapq.nlargest(n, comp_count.items(), key=itemgetter(1))

    topitemsdict = dict(topitems)
    topitemsdict = {item: word_comp[item] for item in topitemsdict.keys() if (item[0].isalpha() and item[1].isalpha())}
    top_comps = {item: word_comp[item] for item in topitemsdict.keys()}

    # pair = list(top_comps.keys())[0]
    # vis = top_comps[pair]

    stdevs = {a: statistics.stdev(top_comps[a]) for a in top_comps.keys()}
    # median_dist = {a: statistics.median(top_comps[a]) for a in top_comps.keys()}
    # mode_dist = {a:statistics.mode(top_comps[a]) for a in top_comps.keys()}

    # lowest_stdevs = heapq.nsmallest(10, stdevs)
    # highest_stdevs = heapq.nlargest(10, stdevs)

    # m = min(stdevs, key=stdevs.get)

    # levenshteins = {item:get_levenshtein(item) for item in word_comp.keys()}
    levenshteins_top = {item: get_levenshtein(
        item) for item in topitemsdict.keys()}

    # x = [levenshteins_top[item] for item in stdevs.keys()]
    # y = [stdevs[item] for item in stdevs.keys()]
    # y1 = [comp_count[item] for item in stdevs.keys()]
    # # y2 = [median_dist[item] for item in stdevs.keys()]
    # plt.scatter(x,y1)
    # plt.scatter(x,y)
    # plt.scatter(y,y1)

    return topitemsdict, stdevs, levenshteins_top, comp_count, topitems


def evaluate_corpus(file, lines=True, num_lines=6000, hand='Both', voynich=False, n=5000):
    if voynich:
        paragraphs = create_df(file, hand)
    # elif lines:
    #     paragraphs = convert_to_strings(file, num_lines)
    else:
        #try:
            paragraphs = convert_to_strings(file, lines, num_lines)
            print('paragraph')
            if not paragraphs:
                return None
            # print(paragraphs)
            word_comp, comp_count = gen_comps(paragraphs, weighted=False)
        #except: 
        #   print("Analysis failed")
        #   return('Fail')
    return analysis(word_comp, comp_count, n=n)


def focus_corpus(corpora_output, fin, tag=''):
    output = corpora_output[fin]
    # output = corpora_output[voynich_file]
    topitemsdict = output[0]
    le = output[2]
    plt.hist(le.values(), bins=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    plt.title(fin + tag)
    plt.savefig('output/'+ fin + tag + '.png')
    plt.clf()

def multiple_corpora(corpora_output, fins, title, tag=''):
    plt.figure(num=None, figsize=(20,20))
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.hist(corpora_output[fins[0]][2].values(), bins=list(range(13)))
    ax2.hist(corpora_output[fins[1]][2].values(), bins=list(range(13)))
    ax3.hist(corpora_output[fins[2]][2].values(), bins=list(range(13)))
    ax4.hist(corpora_output[fins[3]][2].values(), bins=list(range(13)))
    fig.suptitle(title)
    plt.savefig('output/' + title + '.png')
    plt.clf()
    
def run_on_folder(folder='txts'):
    corpora_output = {}
    global_n = 5000
    last_time = time.time()
    for corpus in os.listdir(folder):
#        print(os.path.exists('output/' + corpus.rstrip('.txt') + '.png'))
        if os.path.exists('output/' + corpus.rstrip('.txt') + '.png'):
            print(corpus + " already evaluated")
            continue
        t = time.time()
        time_dif = str(int(-last_time + t))
        last_time = t
        print(corpus + " @ " + datetime.datetime.now().strftime("%H:%M:%S with ") + time_dif + "s for last corpus")
        if corpus=='Chinese':
            corpora_output[corpus.rstrip('.txt')] = evaluate_corpus(folder + '/' + corpus, lines=False, n=1000)
        else:
            corpora_output[corpus.rstrip('.txt')] = evaluate_corpus(folder + '/' + corpus, lines=False, n=global_n)
        if None not in corpora_output.values():
            focus_corpus(corpora_output, corpus.rstrip('.txt'))
        else:
            print(corpora_output.values())
            print("Analysis failed")
        

def main():
    corpora = ['war_peace.txt', 'don_quixote.txt',
               'great_expectations.txt', '60878-0.txt']
    wiki_corpora = ['russian_wiki.txt', 'arabic_wiki.txt', 'Basque',
                'Chinese', 'Latvian', 'Latin', 'Macedonian', 'Norweigen_Nyornsk']
    gibberish = ['Non-Specialist/DA_01.txt', 
    'Non-Specialist/DC_10.txt',
    'Non-Specialist/DC_08.txt',
    'Non-Specialist/DC_11.txt',]
    # wiki_corpora = ['data/xaa.txt', 'data/arabic_wiki.txt']
    # wiki_corpora = ['data/xaa']
    voynich_file = 'voynich_data.txt'

    corpora_output = {}
    global_n = 5000
    for corpus in corpora:
        corpora_output[corpus.rstrip('.txt')] = evaluate_corpus('data/' + corpus, n=global_n)
        focus_corpus(corpora_output, corpus.rstrip('.txt'))

    for corpus in wiki_corpora:
        if corpus=='Chinese':
            corpora_output[corpus.rstrip('.txt')] = evaluate_corpus('data/' + corpus, lines=False, n=1000)
        else:
            corpora_output[corpus.rstrip('.txt')] = evaluate_corpus('data/' + corpus, lines=False, n=global_n)
        focus_corpus(corpora_output, corpus.rstrip('.txt'))
    
    for corpus in gibberish:
        corpora_output[corpus.rstrip('.txt')] = evaluate_corpus('data/' + 'gibberish_voynich/' + corpus, lines=False, n=50)
        focus_corpus(corpora_output, corpus.rstrip('.txt'))

    multiple_corpora(corpora_output, [a.rstrip('.txt') for a in gibberish], 'gibberish_corpora (non-specialist)')

    for hand in ['A', 'B', 'Both']:
        corpora_output[voynich_file.rstrip('.txt')] = evaluate_corpus('data/' + voynich_file, hand=hand, voynich=True, n=global_n)
        focus_corpus(corpora_output, voynich_file.rstrip('.txt'), hand)
    
    # for corpora in corpora_output.values():
        # plt.savefig()
    return corpora_output
# # a = [int(x) for x in (corpora_output[corpora[0].rstrip('.txt')][3]).values()]
# # plt.bar(range(len(a)), a)
# a = corpora_output[wiki_corpora[0].rstrip('.txt')][4][:5][0][0]
# corpora_output[wiki_corpora[0].rstrip('.txt')][0][a]




if __name__ == "__main__":
    #data = main()
    run_on_folder()
