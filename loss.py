from utils.utils import *
from torch.nn import functional as F
from math import ceil
from utils.utils import doc_flat_mask


def cal_loss_with_attn(logits, event_labels, sent_len, neg_pos_ratio=0,
                       neg_label=None, pad_label=None, partial=False):
    """
    logits -> (N, S, W, L)
    sent_len -> (N, S)
    event_labels -> (N, S, W)
    """
    # Check dimension
    batch_size, sent_num, word_num, label_num = logits.size()
    assert(event_labels.size() == (batch_size, sent_num, word_num))

    # Move to proper device
    event_labels = event_labels.to(DEVICE)

    # logits_flat -> (N*S*W, L)
    logits_flat = logits.view(-1, logits.size(-1))

    # event_label_flat -> (N*S*W, 1)
    event_labels_flat = event_labels.view(-1, 1)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    loss = -torch.gather(log_probs_flat, dim=1, index=event_labels_flat)
    # Mask padding, set corresponding values to 0
    loss = loss.squeeze() * doc_flat_mask(sent_len)

    # Perform negative sampling
    if neg_pos_ratio > 0:
        assert neg_label is not None
        assert pad_label is not None
        event_labels_flat = event_labels_flat.squeeze()

        sample_mask = torch.zeros(event_labels_flat.size(0), device=DEVICE)

        # Set the mask of positives to 1,
        sample_mask[(event_labels_flat != neg_label) & (event_labels_flat != pad_label)] = 1

        # Get positive label number
        num_positive = torch.sum(sample_mask)
        num_negative = torch.sum(event_labels_flat == neg_label)
        num_negative_retained = ceil(num_positive * neg_pos_ratio) if num_positive > 0 else 10

        # Get negative indexes
        neg_indexes = (event_labels_flat == neg_label).nonzero()
        neg_retained_indexes = neg_indexes[torch.randperm(num_negative)][:num_negative_retained]
        sample_mask[neg_retained_indexes.squeeze()] = 1

        # Get ignored negative sample number
        num_negative_ignored = (num_negative - num_negative_retained).float()

        # mask loss with negative sampling
        loss = loss.squeeze() * sample_mask
        if partial:
            return torch.sum(loss), (torch.sum(sent_len).float().to(DEVICE) - num_negative_ignored)
        else:
            return torch.sum(loss) / (torch.sum(sent_len).float().to(DEVICE) - num_negative_ignored)
    else:
        if partial:
            return torch.sum(loss), torch.sum(sent_len).float().to(DEVICE)
        else:
            return torch.sum(loss) / torch.sum(sent_len).float().to(DEVICE)
