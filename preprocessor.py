"""
Preprocessor module reads the data pre-processed by
ace2005_processing and generates tsv files that can be
directly passed to iterator module.

"""

import os
import pandas
from collections import defaultdict
import re
import string
from nltk.corpus import stopwords


stopwords_set = set(stopwords.words("english"))


class KFoldBasePreprocessor(object):
    def __init__(self, fold=5):
        self.fields = {
            "token": ["doc_id", "sent_id", "token_id", "offset", "length",
                      "label", "raw_text", "pos_tag", "entity_label",
                      "stanford_ner", "dep_head", "dep_relation"],
            "golden": ["doc_id", "offset", "length", "raw_text", "label"],
            "entity": ["doc_id", "em_id", "label", "tokens", "offset", "length"],
            "sent": ["doc_id", "sent_id", "sent_text"]
        }
        self.retained_fields = []
        self.fold = fold

    def preload_splits(self, path, train, test, field_name, fold):

        data_loaded = []
        for file_name in [train, test]:
            data_file = open(os.path.join(path, "fold_{}/{}".format(fold, file_name)), "r")
            data = pandas.DataFrame([[field.strip("\n") for field in line.split("\t")]
                                     for line in data_file], columns=self.fields[field_name])
            data_loaded.append(data)
            data_file.close()
        return data_loaded

    def transform(self, **kwargs):
        raise NotImplementedError


class Token(object):
    def __init__(self, doc_id=None, offset=None, length=None,
                 raw_text=None, label=None, word_attn=0, entity="NONE"):
        self.doc_id = doc_id
        self.offset = offset
        self.length = length
        self.raw_text = raw_text
        self.label = label
        self.entity = entity
        self.word_attn = word_attn


class KFoldDMNPreprocessor(KFoldBasePreprocessor):
    def __init__(self):
        super(KFoldDMNPreprocessor, self).__init__()
        self.retained_fields = {
            "token": ["doc_id", "sent_id", "token_id", "offset",
                      "length", "label", "raw_text"],
            "entity": ["doc_id", "tokens", "label"],
            "sent": ["doc_id", "sent_text"]
        }
        self.data_dict = {i: {
            "train": [],
            "test": []
        } for i in range(self.fold)}

    @staticmethod
    def not_stopword(token):
        return True if token.raw_text \
                       not in stopwords_set else False

    @staticmethod
    def not_punctuation(token):
        for punc in string.punctuation:
            if punc in token.raw_text:
                return False
        return True

    @staticmethod
    def not_short_sentence(sentence):
        return True if len(sentence) > 2 else False

    @staticmethod
    def filter_stopword(d):
        for doc_id, sents in d.items():
            for idx, sent in enumerate(sents):
                d[doc_id][idx] = [token for token in sent if KFoldDMNPreprocessor.not_stopword(token)]

                # Check
                for token in sent:
                    if not KFoldDMNPreprocessor.not_stopword(token) and token.label != 'other':
                        print(token.doc_id, token.offset, token.length, token.raw_text, token.label)
        return d

    @staticmethod
    def filter_shortsentences(d):
        for doc_id, sents in d.items():
            d[doc_id] = [sent for sent in sents if KFoldDMNPreprocessor.not_short_sentence(sent)]
        return d

    @staticmethod
    def filter_punctuation(d):
        for doc_id, sents in d.items():
            for idx, sent in enumerate(sents):
                d[doc_id][idx] = [token for token in sent if KFoldDMNPreprocessor.not_punctuation(token)]
        return d

    def field_index(self, field_type, field_name):
        return self.retained_fields[field_type].index(field_name)

    @staticmethod
    def get_field_list(doc, field_name):
        return [[getattr(token, field_name)
                  for token in sentence]
                 for sentence in doc]

    def transform(self, tokens, entities, sents):
        parsed_dict = defaultdict(lambda: [])

        # Construct parsed dict using sentences
        for ex in sents:
            doc_id = ex[self.field_index("sent", "doc_id")]
            parsed_dict[doc_id].append([Token(doc_id=doc_id, raw_text=token)
                                        for token in ex[self.field_index("sent", "sent_text")].split()])

        for ex in tokens:
            doc_id = ex[self.field_index("token", "doc_id")]
            sent_id = int(ex[self.field_index("token", "sent_id")])
            token_id = int(ex[self.field_index("token", "token_id")])
            offset = int(ex[self.field_index("token", "offset")])
            length = int(ex[self.field_index("token", "length")])
            label = ex[self.field_index("token", "label")]
            raw_text = ex[self.field_index("token", "raw_text")]  # for sanity check

            assert (parsed_dict[doc_id][sent_id][token_id].raw_text == raw_text)
            parsed_dict[doc_id][sent_id][token_id].offset = offset
            parsed_dict[doc_id][sent_id][token_id].length = length
            parsed_dict[doc_id][sent_id][token_id].label = label
            parsed_dict[doc_id][sent_id][token_id].word_attn = 1 if label != "other" else 0

        for ex in entities:
            doc_id = ex[self.field_index("entity", "doc_id")]
            tokens = re.findall(r"\(\d*,[^)]*\d*\)", ex[self.field_index("entity", "tokens")])
            label = ex[self.field_index("entity", "label")]

            # parse tokens to get sent_id and token_id
            for token_index in tokens:
                sent_id, token_id = [int(index) for index in re.findall(r"\d+", token_index)]
                parsed_dict[doc_id][sent_id][token_id].entity = label

        return parsed_dict

    def preprocess(self, path, train, test):
        for i in range(self.fold):
            tokens = [data[self.retained_fields["token"]]
                      for data in self.preload_splits(
                        path, train, test, "token", i)]
            entities = [data[self.retained_fields["entity"]]
                         for data in self.preload_splits(
                        path, train.replace("ids", "entity"),
                        test.replace("ids", "entity"), "entity", i)]
            sents = [data[self.retained_fields["sent"]]
                     for data in self.preload_splits(
                    path, train.replace("ids", "sents"),
                    test.replace("ids", "sents"), "sent", i)]

            train_dict, test_dict = [self.transform(token_df.values.tolist(),
                                                               entity_df.values.tolist(),
                                                               sent_df.values.tolist())
                                                for token_df, entity_df, sent_df in
                                                zip(tokens, entities, sents)]

            # Remove punctuation, stop word
            # Filter stopwords
            train_dict, test_dict = [KFoldDMNPreprocessor.filter_stopword(d)
                                               for d in (train_dict, test_dict)]

            # Filter punctuation
            train_dict, test_dict = [KFoldDMNPreprocessor.filter_punctuation(d)
                                               for d in (train_dict, test_dict)]
            # Filter short sentences
            train_dict, test_dict = [KFoldDMNPreprocessor.filter_shortsentences(d)
                                               for d in (train_dict, test_dict)]

            # Write converted tsv file
            self.write_tsv("./data/fold_{}".format(i), train_dict, test_dict)
        return

    def write_tsv(self, root_path, train, test):
        tsv_names = ["train.tsv", "test.tsv"]

        for name, data in zip(tsv_names, [train, test]):
            path = os.path.join(root_path, name)

            with open(path, "w") as f:
                # iter over docs
                for doc_id in data.keys():
                    doc = data[doc_id]
                    # Write text field
                    text_field = self.get_field_list(doc, "raw_text")
                    text_field = [" ".join(sentence) for sentence in text_field]
                    text_field = "<EOS>".join(text_field)

                    # Write entity field
                    entity_field = self.get_field_list(doc, "entity")
                    entity_field = [" ".join(sentence) for sentence in entity_field]
                    entity_field = "<EOS>".join(entity_field)

                    # Write label field
                    label_field = self.get_field_list(doc, "label")
                    label_field = [" ".join(sentence) for sentence in label_field]
                    label_field = "<EOS>".join(label_field)

                    # Write offset field
                    offset_field = self.get_field_list(doc, "offset")
                    offset_field = [" ".join(map(str, sentence)) for sentence in offset_field]
                    offset_field = "<EOS>".join(offset_field)

                    # Write length field
                    length_field = self.get_field_list(doc, "length")
                    length_field = [" ".join(map(str, sentence)) for sentence in length_field]
                    length_field = "<EOS>".join(length_field)

                    # Write word attn field
                    word_attn_field = self.get_field_list(doc, "word_attn")
                    word_attn_field = [" ".join(map(str, sentence)) for sentence in word_attn_field]
                    word_attn_field = "<EOS>".join(word_attn_field)

                    # Write sentences attn field
                    sent_attn_field = self.get_field_list(doc, "word_attn")
                    sent_attn_field = "<EOS>".join(["1" if 1 in sentence else "0" for sentence in sent_attn_field])

                    # Write doc_id field
                    doc_id_field = doc_id

                    f.write("\t".join([text_field, entity_field,
                                       label_field, offset_field,
                                       length_field, word_attn_field,
                                       sent_attn_field, doc_id_field]))
                    f.write("\n")


if __name__ == '__main__':
    dmn_preprocessor = KFoldDMNPreprocessor()
    dmn_preprocessor.preprocess(path="./data",
                                train="train/train.ids.dat", test="test/test.ids.dat")
