import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ANCHOR_NUM = 35
ENTITY_NUM = 57
PAD_TOKEN = "<pad>"
WORD_EMBED_SIZE = 300

KFOLD_NUM = 5
