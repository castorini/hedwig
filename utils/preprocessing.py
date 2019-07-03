import numpy as np


def pad_input_matrix(unpadded_matrix, max_doc_length):
    """
    Returns a zero-padded matrix for a given jagged list
    :param unpadded_matrix: jagged list to be padded
    :return: zero-padded matrix
    """
    max_doc_length = min(max_doc_length, max(len(x) for x in unpadded_matrix))
    zero_padding_array = [0 for i0 in range(len(unpadded_matrix[0][0]))]

    for i0 in range(len(unpadded_matrix)):
        if len(unpadded_matrix[i0]) < max_doc_length:
            unpadded_matrix[i0] += [zero_padding_array for i1 in range(max_doc_length - len(unpadded_matrix[i0]))]
        elif len(unpadded_matrix[i0]) > max_doc_length:
            unpadded_matrix[i0] = unpadded_matrix[i0][:max_doc_length]
