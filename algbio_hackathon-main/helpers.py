import argparse
import csv
import os
from tqdm import tqdm

import numpy as np

OUTPUT_FILE = 'alignments_result'
NOC = 'NT'
AMINO = 'AA'
SEQ_START = '>'
LETTER_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '_': 4}
CODONS = dict()
STOP_CODON = 'STOP'
STOP_CODONS = {'TAA', 'TAG', 'TGA'}
AMINO_DICT = {'TGG': 'W',
              'TAC': 'Y', 'TAT': 'Y',
              'TGC': 'C', 'TGT': 'C',
              'GAA': 'E', 'GAG': 'E',
              'AAA': 'K', 'AAG': 'K',
              'CAA': 'Q', 'CAG': 'Q',
              'AGC': 'S', 'AGT': 'S', 'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
              'TTA': 'L', 'TTG': 'L', 'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
              'AGA': 'R', 'AGG': 'R', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
              'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
              'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
              'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
              'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
              'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
              'ATA': 'I', 'ATC': 'I', 'ATT': 'I',
              'TTC': 'F', 'TTT': 'F',
              'GAC': 'D', 'GAT': 'D',
              'CAC': 'H', 'CAT': 'H',
              'AAC': 'N', 'AAT': 'N',
              'ATG': 'M'
              }

AMINO_DICT2NUM = dict()
AMINO_LETTER_DICT = dict()
REV_AMINO_DICT = dict()

STOP_CODON_SCORE = -100
GAP_SCORE = -7
FRAME_SHIFT_SCORE = -20


def arg_parse():
    """
    parse the arguments from the command line
    the arguments are: first sequence location, second sequence location, matrix_score csv location
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('score')
    command_args = parser.parse_args()
    return command_args


def translate_sequence(fasta_file):
    """
    translates a sequence into a numeric representation
    :param fasta_file: fasta file location which contains the desired sequence
    :return: the translated sequence
    """
    file = open(fasta_file, "r").readlines()
    sequence = ''
    for line_index in range(len(file)):
        line = file[line_index].strip()
        if len(line) == 0 or line[0] == SEQ_START:
            continue

        sequence += line
    return sequence



def translate_matrix(matrix_file):
    """
    reads a sigma score csv file into a numpy matrix
    :param matrix_file: matrix file location
    :return: score matrix (numpy)
    """
    tsv_file = open(matrix_file)
    read_tsv = list(csv.reader(tsv_file, delimiter="\t"))

    length = len(read_tsv)
    new_table = np.empty((length, length))

    for i in range(length):
        AMINO_LETTER_DICT[read_tsv[i][0]] = i
        REV_AMINO_DICT[i] = read_tsv[i][0]

    AMINO_DICT2NUM.update({k: AMINO_LETTER_DICT[v] for k, v in AMINO_DICT.items()})
    AMINO_DICT2NUM.update({'TAA': STOP_CODON, 'TAG': STOP_CODON, 'TGA': STOP_CODON})

    for i in range(length):
        for j in range(1, i + 2):
            new_table[i][j - 1] = int(read_tsv[i][j])
            new_table[j - 1][i] = int(read_tsv[i][j])

    return new_table


def write_result_to_txt(alignments, alignment_type):
    """
    writes an alignment to a txt file
    :param alignments: alignment to write to txt
    :param alignment_type: amino acid or noc
    :return: NONE
    """
    curr_dir = os.path.curdir
    with open(os.path.join(curr_dir, f'{OUTPUT_FILE}_{alignment_type}.txt'), mode='w', encoding='utf-8') as file:
        file.write(alignments)


def nt_to_aa(alignment):
    """
    translates noc to amino acid
    :param alignment: alignment to translate
    :return: translated alignment (AA)
    """
    result_aa = ''
    i = 0
    while i + 3 < len(alignment):
        curr = alignment[i:i + 3]
        if curr in AMINO_DICT.keys():
            result_aa += AMINO_DICT[curr]
        elif curr in STOP_CODONS:
            result_aa += '*'
        elif '-' in curr:
            result_aa += '-'
        else:
            result_aa += '!'
        i += 3

    return result_aa


def align_sequences(alignment_score, alignment_a, alignment_b, score_type):
    """
    aligns to sequences in a string representation
    :param alignment_score: the alignment's score
    :param alignment_a: alignment for the first sequence
    :param alignment_b: alignment for the second sequence
    :param score_type: matrix score type. e.g BLOSUM62
    :return: the string of the two alignments
    """
    result = ""
    i = 0
    while i + 50 < len(alignment_a):
        result += f"{alignment_a[i:i + 50]}\n{alignment_b[i:i + 50]}\n\n"
        i += 50
    if i < len(alignment_a):
        result += f"{alignment_a[i:]}\n{alignment_b[i:]}\n\n"
    result += f'Score Matrix Name: {score_type}\n'
    result += f'Alignment Score: {alignment_score}'
    return result


def output_alignments(alignment_score, alignment_a, alignment_b, score_type):
    """
    :param alignment_score: the alignment's score
    :param alignment_a: alignment for the first sequence
    :param alignment_b: alignment for the second sequence
    :param score_type: matrix score type. e.g BLOSUM62
    :return: None
    """
    result_noc = align_sequences(alignment_score, alignment_a, alignment_b, score_type)
    alignment_a_aa = nt_to_aa(alignment_a)
    alignment_b_aa = nt_to_aa(alignment_b)
    result_aa = align_sequences(alignment_score, alignment_a_aa, alignment_b_aa, score_type)

    write_result_to_txt(result_noc, NOC)
    write_result_to_txt(result_aa, AMINO)