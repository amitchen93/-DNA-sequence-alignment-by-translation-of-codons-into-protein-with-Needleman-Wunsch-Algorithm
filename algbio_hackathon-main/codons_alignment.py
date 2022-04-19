from helpers import *


def recursive_formula(alignment_scores, i, j, stop_a, stop_b, substitute_amino_acid, trace_matrix):
    """
    recursive formula for the dynamic programing
    :param alignment_scores: dynamic table (numpy)
    :param i: seq_a index
    :param j: seq_b index
    :param stop_a: STOP_CODON penalty if the seq_a codon is a stop codon, otherwise 0
    :param stop_b: STOP_CODON penalty if the seq_b codon is a stop codon, otherwise 0
    :param substitute_amino_acid: value of the AA alignment
    :param trace_matrix: trace matrix for recovering the optimal alignment
    :return: the best option for the i, j indices
    """
    results = np.full(15, np.NINF)
    indices = []

    if i - 3 >= 0 and j - 3 >= 0:
        results[0] = alignment_scores[i - 3, j - 3] + substitute_amino_acid
    indices.append(tuple((i - 3, j - 3)))

    if i - 3 >= 0:
        results[1] = alignment_scores[i - 3, j] + substitute_amino_acid + GAP_SCORE
    indices.append(tuple((i - 3, j)))

    if j - 3 >= 0:
        results[2] = alignment_scores[i, j - 3] + GAP_SCORE + substitute_amino_acid
    indices.append(tuple((i, j - 3)))

    if i - 3 >= 0 and j - 2 >= 0:
        results[3] = alignment_scores[i - 3, j - 2] + stop_a + FRAME_SHIFT_SCORE
    indices.append(tuple((i - 3, j - 2)))

    if i - 3 >= 0 and j - 1 >= 0:
        results[4] = alignment_scores[i - 3, j - 1] + stop_a + FRAME_SHIFT_SCORE
    indices.append(tuple((i - 3, j - 1)))

    if i - 2 >= 0 and j - 3 >= 0:
        results[5] = alignment_scores[i - 2, j - 3] + stop_a + FRAME_SHIFT_SCORE
    indices.append(tuple((i - 2, j - 3)))

    if i - 1 >= 0 and j - 3 >= 0:
        results[6] = alignment_scores[i - 1, j - 3] + FRAME_SHIFT_SCORE + stop_b
    indices.append(tuple((i - 1, j - 3)))

    if j - 1 >= 0:
        results[7] = alignment_scores[i, j - 1] + GAP_SCORE + FRAME_SHIFT_SCORE
    indices.append(tuple((i, j - 1)))

    if j - 2 >= 0:
        results[8] = alignment_scores[i, j - 2] + GAP_SCORE + FRAME_SHIFT_SCORE
    indices.append(tuple((i, j - 2)))

    if i - 1 >= 0:
        results[9] = alignment_scores[i - 1, j] + FRAME_SHIFT_SCORE + GAP_SCORE
    indices.append(tuple((i - 1, j)))

    if i - 2 >= 0:
        results[10] = alignment_scores[i - 2, j] + FRAME_SHIFT_SCORE + GAP_SCORE
    indices.append(tuple((i - 2, j)))

    if i - 1 >= 0 and j - 1 >= 0:
        results[11] = alignment_scores[i - 1, j - 1] + (2 * FRAME_SHIFT_SCORE)
    indices.append(tuple((i - 1, j - 1)))

    if i - 1 >= 0 and j - 2 >= 0:
        results[12] = alignment_scores[i - 1, j - 2] + (2 * FRAME_SHIFT_SCORE)
    indices.append(tuple((i - 1, j - 2)))

    if i - 2 >= 0 and j - 1 >= 0:
        results[13] = alignment_scores[i - 2, j - 1] + (2 * FRAME_SHIFT_SCORE)
    indices.append(tuple((i - 2, j - 1)))

    if i - 2 >= 0 and j - 2 >= 0:
        results[14] = alignment_scores[i - 2, j - 2] + (2 * FRAME_SHIFT_SCORE)
    indices.append(tuple((i - 2, j - 2)))

    best_index = np.argmax(results)

    trace_matrix[i][j] = indices[best_index]

    return results[best_index]


def initialize_dynamic_table(a_length, b_length):
    """
    :param a_length: the length of the first sequence
    :param b_length: the length of the second sequence
    :return: dynamic table (numpy) and trace matrix (tuples)
    """
    alignment_scores = np.zeros((a_length + 1, b_length + 1))
    alignment_scores[0, 1:] = np.cumsum(np.full(b_length, GAP_SCORE))
    alignment_scores[1:, 0] = np.cumsum(np.full(a_length, GAP_SCORE))
    trace_matrix = [[tuple((i, j)) for j in range(b_length + 1)] for i in range(a_length + 1)]
    trace_matrix[0] = [tuple((0, 0))] + [tuple((0, j - 1)) for j in range(1, b_length + 1)]
    for i in range(1, a_length + 1):
        trace_matrix[i][0] = tuple((i - 1, 0))

    return alignment_scores, trace_matrix


def get_alignment_scores(seq_a, seq_b, sigma):
    """
    the main algorithm
    :param seq_a: first sequence (NT)
    :param seq_b: second sequence (NT)
    :param sigma: score matrix, e.g - BLOSUM62
    :return: trace matrix for recovering the optimal alignment and the optimal score
    """
    a_length, b_length = len(seq_a), len(seq_b)
    alignment_scores, trace_matrix = initialize_dynamic_table(a_length, b_length)
    for i in tqdm(range(1, a_length + 1)):
        for j in range(1, b_length + 1):

            AA1 = '' if i - 3 < 0 else AMINO_DICT2NUM[seq_a[i - 3: i]]
            AA2 = '' if j - 3 < 0 else AMINO_DICT2NUM[seq_b[j - 3: j]]

            stop_a = STOP_CODON_SCORE if AA1 == STOP_CODON else 0
            stop_b = STOP_CODON_SCORE if AA2 == STOP_CODON else 0

            if AA1 == STOP_CODON or AA2 == STOP_CODON:
                substitute_amino_acid = stop_a + stop_b
            else:
                if i - 3 >= 0 and j - 3 >= 0:
                    substitute_amino_acid = sigma[AA1, AA2]
                else:
                    substitute_amino_acid = np.NINF
            alignment_scores[i, j] = recursive_formula(alignment_scores, i, j, stop_a, stop_b, substitute_amino_acid,
                                                       trace_matrix)

    return trace_matrix, alignment_scores[-1, -1]


def get_trace(seq_a, seq_b, trace_matrix):
    """
    extracts the optimal alignments
    :param seq_a: first sequence (NT)
    :param seq_b: second sequence (NT)
    :param trace_matrix: trace matrix for recovering the oprimal alignment
    :return: optimal alignments for the two given sequences
    """
    alignment_a, alignment_b = '', ''
    i, j = len(seq_a), len(seq_b)

    while i > 0 or j > 0:
        prev_i, prev_j = trace_matrix[i][j]
        ################### GAPS and FULL CODON #######################
        if (i - prev_i) == 3 and (j - prev_j) == 3:
            alignment_a = seq_a[prev_i: i] + alignment_a
            alignment_b = seq_b[prev_j: j] + alignment_b
        elif (i - prev_i) == 3 and j == prev_j:
            alignment_a = seq_a[prev_i: i] + alignment_a
            alignment_b = '-' * 3 + alignment_b
        elif i == prev_i and (j - prev_j) == 3:
            alignment_a = '-' * 3 + alignment_a
            alignment_b = seq_b[prev_j: j] + alignment_b

        ################### SHIFTS and FULL CODON #######################
        elif (i - prev_i) == 3 and (j - prev_j) == 2:
            alignment_a = seq_a[prev_i: i] + alignment_a
            alignment_b = seq_b[prev_j: j] + '!' + alignment_b
        elif (i - prev_i) == 3 and (j - prev_j) == 1:
            alignment_a = seq_a[prev_i: i] + alignment_a
            alignment_b = seq_b[prev_j: j] + ('!' * 2) + alignment_b
        elif (i - prev_i) == 2 and (j - prev_j) == 3:
            alignment_a = seq_a[prev_i: i] + '!' + alignment_a
            alignment_b = seq_b[prev_j: j] + alignment_b
        elif (i - prev_i) == 1 and (j - prev_j) == 3:
            alignment_a = seq_a[prev_i: i] + ('!' * 2) + alignment_a
            alignment_b = seq_b[prev_j: j] + alignment_b

        ################### GAPS and SHIFTS #######################
        elif prev_i == i and (j - prev_j) == 1:
            alignment_a = '-' * 3 + alignment_a
            alignment_b = seq_b[prev_j: j] + ('!' * 2) + alignment_b
        elif prev_i == i and (j - prev_j) == 2:
            alignment_a = '-' * 3 + alignment_a
            alignment_b = seq_b[prev_j: j] + '!' + alignment_b
        elif (i - prev_i) == 1 and j == prev_j:
            alignment_a = seq_a[prev_i: i] + ('!' * 2) + alignment_a
            alignment_b = '-' * 3 + alignment_b
        elif (i - prev_i) == 2 and j == prev_j:
            alignment_a = seq_a[prev_i: i] + '!' + alignment_a
            alignment_b = '-' * 3 + alignment_b

        ################### ONLY SHIFTS #######################
        elif (i - prev_i) == 1 and (j - prev_j) == 1:
            alignment_a = seq_a[prev_i: i] + ('!' * 2) + alignment_a
            alignment_b = seq_b[prev_j: j] + ('!' * 2) + alignment_b
        elif (i - prev_i) == 1 and (j - prev_j) == 2:
            alignment_a = seq_a[prev_i: i] + ('!' * 2) + alignment_a
            alignment_b = seq_b[prev_j: j] + '!' + alignment_b
        elif (i - prev_i) == 2 and (j - prev_j) == 1:
            alignment_a = seq_a[prev_i: i] + '!' + alignment_a
            alignment_b = seq_b[prev_j: j] + ('!' * 2) + alignment_b
        elif (i - prev_i) == 2 and (j - prev_j) == 2:
            alignment_a = seq_a[prev_i: i] + '!' + alignment_a
            alignment_b = seq_b[prev_j: j] + '!' + alignment_b
        else:
            break

        if prev_i == 0 and prev_j == 0:
            break
        i, j = prev_i, prev_j

    return alignment_a, alignment_b


if __name__ == '__main__':
    parser = arg_parse()
    seq_a, seq_b = translate_sequence(parser.seq_a), translate_sequence(parser.seq_b)
    sigma_score = translate_matrix(parser.score)
    trace_matrix, alignments_score = get_alignment_scores(seq_a, seq_b, sigma_score)
    alignment_a, alignment_b = get_trace(seq_a, seq_b, trace_matrix)
    score_type = os.path.basename(parser.score).split('.')[0]
    output_alignments(alignments_score, alignment_a, alignment_b, score_type)


