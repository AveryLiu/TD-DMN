"""
Iterator reads through the processed
tsv files and generate batches that could
be directly passed to the model
"""
from torchtext import data
from torchtext.vocab import Vectors


class BaseIterator(object):
    def __init__(self):
        from utils.info_field import InfoField, NestedInfoField
        self.InfoField = InfoField
        self.NestedInfoField = NestedInfoField
        return

    def get_iters(self, train_batch_size, fold_num):
        raise NotImplementedError


class DMNIterator(BaseIterator):
    def __init__(self):
        super(DMNIterator, self).__init__()
        # Define text nested field
        self.text_sent = data.Field(sequential=True,
                                    lower=True,
                                    tokenize=lambda x: x.split(" "))
        self.text_doc = data.NestedField(self.text_sent,
                                         tokenize=lambda x: x.split("<EOS>"),
                                         include_lengths=True)

        # Define entity nested field
        self.entity_sent = data.Field(sequential=True,
                                      tokenize=lambda x: x.split(" "),
                                      unk_token=None)
        self.entity_doc = data.NestedField(self.entity_sent,
                                           tokenize=lambda x: x.split("<EOS>"))

        # Define label nested field
        self.label_sent = data.Field(sequential=True,
                                     tokenize=lambda x: x.split(" "),
                                     unk_token=None)
        self.label_doc = data.NestedField(self.label_sent,
                                          tokenize=lambda x: x.split("<EOS>"))

        # Define offset nested field
        self.offset_sent = self.InfoField(sequential=True,
                                          tokenize=lambda x: x.split(" "),
                                          use_vocab=False)
        self.offset_doc = self.NestedInfoField(self.offset_sent,
                                               tokenize=lambda x: x.split("<EOS>"),
                                               use_vocab=False)

        # Define length nested field
        self.length_sent = self.InfoField(sequential=True,
                                          tokenize=lambda x: x.split(" "),
                                          use_vocab=False,
                                          pad_token=None)
        self.length_doc = self.NestedInfoField(self.length_sent,
                                               tokenize=lambda x: x.split("<EOS>"),
                                               use_vocab=False)

        # Define word attention field
        self.word_attn_sent = self.InfoField(sequential=True,
                                             tokenize=lambda x: x.split(" "),
                                             use_vocab=False)
        self.word_attn_doc = self.NestedInfoField(self.word_attn_sent,
                                                  tokenize=lambda x: x.split("<EOS>"),
                                                  use_vocab=False)
        # Define sentence attention field
        self.sent_attn_doc = self.InfoField(sequential=True,
                                            tokenize=lambda x: x.split("<EOS>"),
                                            use_vocab=False)

        # Define doc id field
        self.doc_id = self.InfoField(sequential=False, use_vocab=False)

        self.vectors = None

    def get_iters(self, train_batch_size, fold_num):
        # Load data splits
        train, test = data.TabularDataset.splits(path="./data/fold_{}".format(fold_num), train="train.tsv",
                                                      test="test.tsv", format="tsv",
                                                      fields=[("TEXT", self.text_doc), ("ENTITY", self.entity_doc),
                                                              ("LABEL", self.label_doc),
                                                              ("OFFSET", self.offset_doc),
                                                              ("LENGTH", self.length_doc),
                                                              ("WORD_ATTN", self.word_attn_doc),
                                                              ("SENT_ATTN", self.sent_attn_doc),
                                                              ("DOC_ID", self.doc_id)])

        # First load vectors
        vector = Vectors(name="GoogleNews-vectors-negative300.txt", cache=".vector_cache/")

        # Build vocabs
        self.text_doc.build_vocab(train, test, vectors=vector)
        self.entity_doc.build_vocab(train, test)
        self.label_doc.build_vocab(train, test)

        # Get iterators
        train_iter, test_iter = data.BucketIterator.splits((train, test),
                                                            sort=False, batch_sizes=(train_batch_size, 2),
                                                            repeat=True)
        train_iter.shuffle = True
        return train_iter, test_iter

    def get_vocabs(self):
        return self.text_doc.vocab, self.entity_doc.vocab, self.label_doc.vocab
