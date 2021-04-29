import torch

def collate_fn(batch):
    """Collects together univariate or multivariate sequences into a single batch, arranged in descending order of length.

    Also returns the corresponding lengths and labels as :class:`torch:torch.LongTensor` objects.

    Parameters
    ----------
    batch: list of tuple(torch.FloatTensor, int)
        Collection of :math:`B` sequence-label pairs, where the :math:`n^\\text{th}` sequence is of shape :math:`(T_n \\times D)` or :math:`(T_n,)` and the label is an integer.

    Returns
    -------
    padded_sequences: :class:`torch:torch.Tensor` (float)
        A tensor of size :math:`B \\times T_\\text{max} \\times D` containing all of the sequences in descending length order, padded to the length of the longest sequence in the batch.

    lengths: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence lengths in descending order.

    labels: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence labels in descending length order.
    """
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int)) or list(tuple(tensor(T), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences, lengths, labels = [], torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    for i, (sequence, label) in enumerate(batch):
        lengths[i], labels[i] = len(sequence), label
        sequences.append(sequence)

    # Combine sequences into a padded matrix
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D) or (B x T_max)

    # If a vector input was given for the sequences, expand (B x T_max) to (B x T_max x 1)
    if padded_sequences.ndim == 2:
        padded_sequences.unsqueeze_(-1)

    return padded_sequences, lengths, labels
    # Shapes: (B x T_max x D), (B,), (B,)