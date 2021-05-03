import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepGRU(nn.Module):
    """A modular PyTorch implementation of the DeepGRU (Deep Gesture Recognition Utility) recurrent neural network architecture
    designed by Maghoumi & LaViola Jr. [#maghoumi]_, originally for gesture recognition, but applicable to general sequence classification tasks.

    Parameters
    ----------
    n_features: int
        The number of features that each observation within a sequence has.

    n_classes: int
        The number of different sequence classes.

    dims: dict
        A dictionary consisting of dimension configuration for the GRUs and fully-connected layers.

        .. note::
            Values for the keys ``'gru1'``, ``'gru2'``, ``'gru3'`` and ``'fc'`` must be set.

    device: str, optional
        The device to send the model parameters to for computation.

        If no device is specified, a check is made for any available CUDA device, otherwise the CPU is used.

    Notes
    -----
    .. [#maghoumi] **Mehran Maghoumi & Joseph J. LaViola Jr.** `"DeepGRU: Deep Gesture Recognition Utility" <https://arxiv.org/abs/1810.12514>`_
      *Advances in Visual Computing, 14th International Symposium on Visual Computing*, ISVC 2019,
      Lake Tahoe, NV, USA, October 7–9, 2019, Proceedings, Part I (pp.16-31)
    """
    def __init__(self, n_features, n_classes, dims={'gru1': 512, 'gru2': 256, 'gru3': 128, 'fc': 256}, device=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Specify sub-modules
        self.model = nn.ModuleDict({
            'enc': _EncoderNetwork(dims={'in': n_features, **{k:v for k, v in dims.items() if k.startswith('gru')}}, device=device),
            'attn': _AttentionModule(dims={'in': dims['gru3']}, device=device),
            'clf': _Classifier(dims={'in': dims['gru3']*2, 'fc': dims['fc'], 'out': n_classes}, device=device),
        })

        # Send model to device
        self.to(device)

    def forward(self, x, x_lengths):
        """Passes the batched input sequences through the encoder network, attention module and classifier to generate log-softmax scores.

        .. note::
            Since log-softmax scores are returned, it is advised to use the negative log-likelihood loss :class:`torch:torch.nn.NLLLoss`.

        Parameters
        ----------
        x: torch.PackedSequence
            A packed representation of a batch of input observation sequences.

        x_lengths: torch.Tensor (long/int)
            A tensor of the sequence lengths of the batch in descending order.

        Returns
        -------
        log_softmax: :class:`torch:torch.Tensor` (float)
            :math:`B\\times C` tensor of :math:`C` log-softmax scores (class predictions) for each observation sequence in the batch.
        """
        h, h_last = self.model['enc'](x, x_lengths)
        o_attn = self.model['attn'](h, h_last)
        return self.model['clf'](o_attn)

class _EncoderNetwork(nn.Module):
    def __init__(self, dims, device):
        super().__init__()
        self.dims = dims
        self.device = device

        # Specify sub-modules
        self.model = nn.ModuleDict({
           'gru1': nn.GRU(self.dims['in'], self.dims['gru1'], num_layers=2, batch_first=True).to(device),
           'gru2': nn.GRU(self.dims['gru1'], self.dims['gru2'], num_layers=2, batch_first=True).to(device),
           'gru3': nn.GRU(self.dims['gru2'], self.dims['gru3'], num_layers=1, batch_first=True).to(device)
        })

        # Send model to device
        self.to(device)

    def forward(self, x, x_lengths):
        x = x.to(self.device)

        # Pack the padded Tensor into a PackedSequence
        x_packed = pack_padded_sequence(x, x_lengths.cpu(), batch_first=True)

        # Pass the PackedSequence through the GRUs
        h_packed, _ = self.model['gru1'](x_packed)
        h_packed, _ = self.model['gru2'](h_packed)
        h_packed, h_last = self.model['gru3'](h_packed)

        # Unpack the hidden state PackedSequence into a padded Tensor
        h_padded = pad_packed_sequence(h_packed, batch_first=True, padding_value=0.0, total_length=max(x_lengths))
        return h_padded[0], h_last
        # Shape: B x T_max x D_out, 1 x B x D_out

class _AttentionModule(nn.Module):
    def __init__(self, dims, device):
        super().__init__()
        self.device = device
        self.dims = dims

        # Specify sub-modules
        self.model = nn.ModuleDict({
            # Attentional context vector weights
            'attn_ctx': nn.Linear(self.dims['in'], self.dims['in'], bias=False).to(device),
            # Auxilliary context
            'aux_ctx': nn.GRU(input_size=self.dims['in'], hidden_size=self.dims['in']).to(device)
        })

        # Send model to device
        self.to(device)

    def forward(self, h, h_last):
        h_last.transpose_(1, 0)
        # Shape: B x 1 x D_out

        # Calculate attentional context
        h.transpose_(1, 2)
        c = F.softmax(self.model['attn_ctx'](h_last) @ h, dim=0)
        c = (c @ h.transpose(2, 1)).transpose(1, 0)
        # Shape: 1 x B x D_out

        # Calculate auxilliary context
        c_aux, _ = self.model['aux_ctx'](c, h_last.transpose(1, 0))
        # Shape: 1 x B x D_out

        # Combine attentional and auxilliary context
        return torch.cat((c.squeeze(0), c_aux.squeeze(0)), dim=1)
        # Shape: B x D_out*2

class _Classifier(nn.Module):
    def __init__(self, dims, device):
        super().__init__()
        self.device = device
        self.dims = dims

        # Specify sub-modules
        self.model = nn.ModuleDict({
            'fc1': nn.Sequential(
                nn.BatchNorm1d(self.dims['in']),
                nn.Dropout(),
                nn.Linear(self.dims['in'], self.dims['fc'])
            ).to(device),
            'fc2': nn.Sequential(
                nn.BatchNorm1d(self.dims['fc']),
                nn.Dropout(),
                nn.Linear(self.dims['fc'], self.dims['out'])
            ).to(device)
        })

        # Send model to device
        self.to(device)

    def forward(self, o_attn):
        f1 = self.model['fc1'](o_attn)
        f2 = self.model['fc2'](F.relu(f1))
        return F.log_softmax(f2, dim=1)