from constants import *
import torch


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long().to(DEVICE)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand).to(DEVICE)
    return seq_range_expand < seq_length_expand


def pad_sent_attn(sent_attn):
    # numericalize it
    sent_attn = [[int(s) for s in d] for d in sent_attn]
    max_doc_len = max(len(d) for d in sent_attn)

    # pad doc
    for doc in sent_attn:
        while len(doc) < max_doc_len:
            doc.append(0)
    return torch.FloatTensor(sent_attn, device=DEVICE)


def pad_word_attn(word_attn):
    # word_attn -> (N, S, W)
    word_attn = [[[int(w) for w in s] for s in d] for d in word_attn]
    max_sent_len = max(map(max, [[len(s) for s in d] for d in word_attn]))
    max_doc_len = max([len(d) for d in word_attn])

    # pad sentence
    for doc in word_attn:
        for sent in doc:
            while len(sent) < max_sent_len:
                sent.append(0)

    # pad doc
    for doc in word_attn:
        while len(doc) < max_doc_len:
            doc.append([0 for _ in range(max_sent_len)])
    return torch.FloatTensor(word_attn, device=DEVICE)
