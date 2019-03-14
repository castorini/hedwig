import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def auto_rnn_bilstm(lstm: nn.LSTM, seqs, lengths):
    batch_size = seqs.size(1)
    state_shape = lstm.num_layers * 2, batch_size, lstm.hidden_size
    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
    
    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)
    output, (hn, cn) = lstm(packed_pinputs, (h0, c0))
    output = unpack_from_rnn_seq(output, r_index)
    
    return output

def pack_for_rnn_seq(inputs, lengths):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: 
    """
    _, sorted_indices = lengths.sort()
    '''
        Reverse to decreasing order
    '''
    r_index = reversed(list(sorted_indices))
    s_inputs_list = []
    lengths_list = []
    reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

    for j, i in enumerate(r_index):
        s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
        lengths_list.append(lengths[i])
        reverse_indices[i] = j

    reverse_indices = list(reverse_indices)

    s_inputs = torch.cat(s_inputs_list, 1)
    packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

    return packed_seq, reverse_indices

def unpack_from_rnn_seq(packed_seq, reverse_indices):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq)
    s_inputs_list = []

    for i in reverse_indices:
        s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
    return torch.cat(s_inputs_list, 1)

def max_along_time(inputs, lengths):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    ls = list(lengths)

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = inputs[:l, i, :]
        seq_i_max, _ = seq_i.max(dim=0)
        seq_i_max = seq_i_max.squeeze()
        b_seq_max_list.append(seq_i_max)

    return torch.stack(b_seq_max_list)

