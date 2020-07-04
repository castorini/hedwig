import torch
import torch.nn as nn

import torch.nn.functional as F


class CharCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.is_cuda_enabled = config.cuda

        num_conv_filters = config.num_conv_filters
        output_channel = config.output_channel
        num_affine_neurons = config.num_affine_neurons
        target_class = config.target_class
        # we can add these parameters in the config
        input_channel = 68 #number of characters
        first_kernel_size = 7
        second_kernel_size = 3
        pool_size = 3
        max_sentence_length = 1014 #maximum number of characters per sentence

        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=first_kernel_size)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=first_kernel_size)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=second_kernel_size)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=second_kernel_size)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=second_kernel_size)
        self.conv6 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=second_kernel_size)

        # due to reduction based on the convolutional  neural network
        temp = first_kernel_size - 1 + pool_size * (first_kernel_size - 1) + (
                pool_size ** 2 * 4 * (second_kernel_size - 1))
        linear_size_temp = int((max_sentence_length - temp) / (pool_size ** 3)) * num_conv_filters

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(linear_size_temp, num_affine_neurons)
        self.fc2 = nn.Linear(num_affine_neurons, num_affine_neurons)
        self.fc3 = nn.Linear(num_affine_neurons, target_class)

    def forward(self, x, **kwargs):
        if torch.cuda.is_available() and self.is_cuda_enabled:
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)

        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool1d(F.relu(self.conv6(x)), 3)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
