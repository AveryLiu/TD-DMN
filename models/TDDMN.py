import torch.nn as nn
from torch.nn import functional
import logging
import numpy as np
from utils.utils import *


class WordAttention(nn.Module):
    def __init__(self, config):
        super(WordAttention, self).__init__()
        self.attention_word = nn.Sequential(
            nn.Linear(in_features=int(config.get("word_gru_hidden_size")),
                      out_features=int(config.get("word_attn_size")),
                      bias=True),
            nn.Tanh(),
            nn.Linear(in_features=int(config.get("word_attn_size")),
                      out_features=1,
                      bias=True)
        )

    @staticmethod
    def mask_normalize_attention(word_attns, sen_len):
        logger = logging.getLogger("[Mask Attention]")
        logger.setLevel(logging.INFO)

        # word_attns -> (N*S, W)
        # sen_len -> (N*S)
        attn_mask = sequence_mask(sen_len)
        masked_attn = word_attns * attn_mask.float()
        # set to -np.inf thus the softmax will ignore zero padding
        masked_attn[masked_attn == 0] = -np.inf
        normalized_attn = functional.softmax(masked_attn, dim=1)
        return normalized_attn

    def forward(self, word_gru_output, sen_len):
        # word_gru_output -> (N*S, W, H_w)
        # sen_len -> (N*S)

        # normalized_word_attns -> (N*S, W)
        word_attns = self.attention_word(word_gru_output).squeeze(2)
        normalized_word_attns = WordAttention.mask_normalize_attention(
                                    word_attns, sen_len)

        # sents_embed -> (N*S, H_w)
        sents_embed = torch.sum(word_gru_output * normalized_word_attns.unsqueeze(2), dim=1)
        return sents_embed, normalized_word_attns


class WordGRU(nn.Module):
    """ Encode word into sentence """
    def __init__(self, config):
        super(WordGRU, self).__init__()
        # Encode word into sentences
        self.word_gru = nn.GRU(
            input_size=WORD_EMBED_SIZE + int(config.get("entity_embed_size")),
            hidden_size=int(config.get("word_gru_hidden_size")),
            num_layers=1,
            bidirectional=True
        )
        self.config = config

    def forward(self, sents, sen_len):
        logger = logging.getLogger("[Word GRU]")
        logger.setLevel(logging.INFO)

        # pass sentences to word gru to get hidden representation
        packed = torch.nn.utils.rnn.pack_padded_sequence(
                    sents, sen_len, batch_first=True)
        outputs, hidden = self.word_gru(packed)
        outputs, output_length = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                                        batch_first=True)
        del packed, hidden, output_length
        outputs = outputs[:, :, :int(self.config.get("word_gru_hidden_size"))] \
                + outputs[:, :, int(self.config.get("word_gru_hidden_size")):]
        return outputs


class AttentionalGRUCell(nn.Module):
    def __init__(self, config):
        super(AttentionalGRUCell, self).__init__()
        self.hidden_size = int(config.get("memory_size"))

        self.Wr = nn.Linear(self.hidden_size,
                            self.hidden_size)
        self.Ur = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size
                           , self.hidden_size)
        self.U = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, fact, C, g):
        r = functional.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = functional.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        if torch.cuda.is_available():
            g = g.cuda()
        h = g * h_tilda + (1 - g) * C
        return h


class AttentionalGRU(nn.Module):
    def __init__(self, config):
        super(AttentionalGRU, self).__init__()
        self.hidden_size = int(config.get("memory_size"))
        self.AGRUCell = AttentionalGRUCell(config)

    def forward(self, facts, G):
        """
        facts -> (N*S, S, H)
        G -> (N*S, S)
        returns (N*S, H)
        """
        batch_num, sen_num, memory_size = facts.size()
        C = torch.zeros(self.hidden_size, device=DEVICE)
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, config):
        super(EpisodicMemory, self).__init__()
        self.atten_gru = AttentionalGRU(config)
        self.z1 = nn.Linear(4 * int(config.get("memory_size")),
                            int(config.get("fact_attn_hidden")))
        self.z2 = nn.Linear(int(config.get("fact_attn_hidden")), 1)
        self.next_mem = nn.Linear(3 * int(config.get("memory_size")),
                                  int(config.get("memory_size")))

        self.after_attention_gru_dropout = nn.Dropout(
            config.get("after_attentional_gru_dropout"))
        self.after_memory_update_dropout = nn.Dropout(
            config.get("after_memory_update_dropout"))

        self.config = config

    def make_interaction(self, facts, prevM, questions, doc_len):
        """
        facts -> (N, S, H)
        prevM -> (N, S, H)
        questions -> (N, S, H)
        doc_len -> (N,)
        """
        logger = logging.getLogger("[Make Interaction]")
        logger.setLevel(logging.INFO)

        batch_num, sent_num, memory_size = facts.size()
        # repeat along [] dimension
        # facts -> (N, [S], S, H)
        facts = facts.unsqueeze(1).expand((batch_num, sent_num,
                                           sent_num, memory_size))
        # prevM -> (N, S, [S], H)
        prevM = prevM.unsqueeze(2).expand((batch_num, sent_num,
                                           sent_num, memory_size))
        # questions -> (N, S, [S], H)
        questions = questions.unsqueeze(2).expand((batch_num, sent_num,
                                                   sent_num, memory_size))

        # Z -> (N, S, S, 4H)
        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=3)

        G = functional.tanh(self.z1(z))
        # G -> (N, S, S)
        G = self.z2(G).squeeze(3)

        attn_mask = sequence_mask(doc_len)
        attn_mask = attn_mask.unsqueeze(1).expand(
            batch_num, sent_num, sent_num)

        G = G * attn_mask.float()
        G[G == 0] = -np.inf
        G = functional.softmax(G, dim=2)

        return G

    def forward(self, facts, prevM, questions, doc_len):
        """
        facts -> (N, S, H)
        questions -> (N, S, H)
        prevM -> (N, S, H)
        doc_len -> (N,)
        """
        logger = logging.getLogger("[Episodic Memory]")
        logger.setLevel(logging.INFO)

        batch_size, sent_num, memory_size = facts.size()
        # attn_gates -> (N, S, S)
        attn_gates = self.make_interaction(facts, prevM, questions, doc_len)

        # attn_gates -> (N*S, S)
        attn_gates = attn_gates.view(batch_size*sent_num, sent_num)
        # facts -> (N, [S], S, H) -> (N*S, S, H)
        facts = facts.unsqueeze(1).expand(batch_size, sent_num,
                                          sent_num, memory_size)
        facts = facts.contiguous().view(batch_size*sent_num, sent_num, memory_size)

        # attn_gates -> (N*S, S)
        # facts -> (N*S, S, H)
        # C -> (N*S, H) -> (N, S, H)
        C = self.atten_gru(facts, attn_gates)
        C = self.after_attention_gru_dropout(C)

        C = C.view(batch_size, sent_num, -1)
        # concat -> (N, S, 3*H)
        concat = torch.cat([prevM, C, questions], dim=2)

        next_mem = functional.relu(self.next_mem(concat))
        next_mem = self.after_memory_update_dropout(next_mem)
        attn_gates = attn_gates.view(batch_size, sent_num, sent_num)
        # next_men -> (N, S, H), attn_gats -> (N, S, S)
        return next_mem, attn_gates


class QuestionModule(nn.Module):
    def __init__(self, config):
        super(QuestionModule, self).__init__()
        self.question_gru = nn.GRU(input_size=WORD_EMBED_SIZE
                                   + int(config.get("entity_embed_size")),
                                   hidden_size=int(config.get("memory_size")),
                                   num_layers=1,
                                   bidirectional=True)
        self.after_question_gru_dropout = nn.Dropout(
            config.get("after_question_gru_dropout")
        )
        self.config = config

    def forward(self, doc_input_embedded, sen_len):
        logger = logging.getLogger("[Question Module]")
        logger.setLevel(logging.INFO)

        batch_size, sents_num, words_num, embedding_size = doc_input_embedded.size()
        # doc_input_embedded -> (N*S, W, V)
        doc_input_embedded = doc_input_embedded.view(batch_size*sents_num,
                                                     words_num, embedding_size)
        # sen_len -> (N*S)
        sen_len = sen_len.view(batch_size*sents_num)

        sorted_sen_len, sorted_sen_indexes = torch.sort(sen_len, descending=True)
        doc_input_embedded = doc_input_embedded[sorted_sen_indexes]

        # Remove padding
        none_padding_length = torch.sum(sorted_sen_len != 0).item()
        doc_input_embedded = doc_input_embedded[:none_padding_length]
        sorted_sen_len = sorted_sen_len[:none_padding_length]

        packed = torch.nn.utils.rnn.pack_padded_sequence(doc_input_embedded,
                                                         sorted_sen_len,
                                                         batch_first=True)
        output, hidden = self.question_gru(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Output -> (N*S, W, H)
        output = output[:, :, :int(self.config.get("memory_size"))]\
                 + output[:, :, int(self.config.get("memory_size")):]

        # Add padding
        Q = torch.cat([output, torch.zeros(
            batch_size * sents_num - none_padding_length,
            output.size(1), output.size(2), device=DEVICE)], dim=0)

        # Recover order
        _, unsort_indexes = torch.sort(sorted_sen_indexes)
        Q = Q[unsort_indexes]
        # Recover shape, Q -> (N, S, W, H)
        Q = Q.view(batch_size, sents_num, words_num, -1)

        Q = self.after_question_gru_dropout(Q)

        # Get Q_star, Q_star -> (N, S, H)
        Q_star = torch.sum(Q, dim=2) / Q.size(2)

        return Q_star, Q


class InputModule(nn.Module):
    def __init__(self, config):
        super(InputModule, self).__init__()
        self.word_gru = WordGRU(config)
        self.word_attention = WordAttention(config)
        self.fusion_gru = nn.GRU(
            input_size=int(config.get("word_gru_hidden_size")),
            hidden_size=int(config.get("memory_size")),
            num_layers=1,
            bidirectional=True)

        self.word_dropout = nn.Dropout(
            config.get("after_word_gru_dropout"))
        self.fusion_dropout = nn.Dropout(
            config.get("after_fusion_gru_dropout")
        )
        self.config = config

    def forward(self, contexts, sen_len, doc_len):
        """
        contexts -> (N, S, W, V)
        sen_len -> (N, S)
        doc_len -> (N,)
        """
        logger = logging.getLogger("[Input Module]")
        logger.setLevel(logging.INFO)

        logger.debug("ENTER")
        # Obtain sentence embedding
        logger.debug("contexts: %s" % str(contexts.size()))

        # Reshape context to (N*S, W, V)
        batch_size, sents_num, words_num, embedding_size = contexts.size()
        contexts = contexts.view(batch_size*sents_num, words_num, embedding_size)
        # Reshape sen_len to (N*S)
        sen_len = sen_len.view(batch_size*sents_num)

        # Sort sent_len
        sorted_sen_len, sorted_sen_indexes = torch.sort(sen_len, descending=True)
        sorted_sent_input = contexts[sorted_sen_indexes]

        # Remove padding
        none_padding_length = torch.sum(sorted_sen_len != 0).item()
        sorted_sent_input = sorted_sent_input[:none_padding_length]
        sorted_sen_len = sorted_sen_len[:none_padding_length]

        # word_gru_output -> (N*S, W, H_w)
        word_gru_output = self.word_gru(sorted_sent_input, sorted_sen_len)

        # sents after attention
        # sents_embeds -> (N*S, H_w), word_attns -> (N*S, W)
        sents_embeds, word_attns = self.word_attention(word_gru_output, sorted_sen_len)

        # Add drop-out layer
        sents_embeds = self.word_dropout(sents_embeds)

        # Restore original order
        # Obtain unsort indexes
        _, unsort_sen_indexes = torch.sort(sorted_sen_indexes)
        # Add padding
        sents_embeds = torch.cat([sents_embeds, torch.zeros(
            batch_size*sents_num-none_padding_length, sents_embeds.size(1), device=DEVICE)], dim=0)
        word_attns = torch.cat([word_attns, torch.zeros(
            batch_size*sents_num-none_padding_length, word_attns.size(1), device=DEVICE)], dim=0)
        # Recover order
        sents_embeds = sents_embeds[unsort_sen_indexes]
        word_attns = word_attns[unsort_sen_indexes]
        # Recover shape, sents_embeds -> (N, S, H_w), word_attns -> (N, S, W)
        sents_embeds = sents_embeds.view(batch_size, sents_num, -1)
        word_attns = word_attns.view(batch_size, sents_num, -1)

        # Feed into fusion GRU
        sorted_doc_len, sorted_doc_indexes = torch.sort(doc_len, descending=True)
        sents_embeds = sents_embeds[sorted_doc_indexes]

        packed = torch.nn.utils.rnn.pack_padded_sequence(sents_embeds,
                                                         sorted_doc_len,
                                                         batch_first=True)
        # Fusion layer to generate facts
        outputs, hidden = self.fusion_gru(packed)
        outputs, output_length = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        # Recover order
        _, unsort_doc_indexes = torch.sort(sorted_doc_indexes)
        facts = outputs[unsort_doc_indexes]

        # Merge hidden states
        facts = facts[:, :, :int(self.config.get("memory_size"))] \
                + facts[:, :, int(self.config.get("memory_size")):]

        facts = self.fusion_dropout(facts)
        # facts -> (N, S, H)
        logger.debug("facts size: %s" % str(facts.size()))
        logger.debug("Exit")
        return facts, word_attns


class AnswerModule(nn.Module):
    def __init__(self, config):
        super(AnswerModule, self).__init__()
        self.answer_gru = nn.GRU(
            input_size=2*config.get("memory_size"),
            hidden_size=config.get("answer_gru_hidden"),
            bidirectional=True
        )
        self.hidden2tag = nn.Linear(
            2*config.get("answer_gru_hidden"), ANCHOR_NUM
        )
        self.before_answer_gru_dropout = nn.Dropout(
            config.get("before_answer_gru_dropout")
        )
        self.before_dense_dropout = nn.Dropout(
            config.get("before_dense_dropout")
        )
        self.config = config

    def forward(self, M, questions, sents_len):
        # M -> (N, S, H)
        # questions -> (N, S, W, H)
        # sents_len -> (N, S)

        logger = logging.getLogger("[Answer Module]")
        logger.setLevel(logging.INFO)

        batch_size, sents_num, word_num, _ = questions.size()
        # M -> (N, S, [W], H)
        M = M.unsqueeze(2).expand_as(questions)
        # concat -> (N, S, W, 2*H)
        concat = torch.cat([M, questions], dim=3)

        # concat -> (N*S, W, 2*H)
        # sents_len ->(N*S)
        concat = concat.view(batch_size*sents_num, word_num, -1)
        sents_len = sents_len.view(batch_size*sents_num)

        # out_gru_output -> (N*S, W, H_out)
        sorted_sen_len, sorted_sen_indexes = torch.sort(sents_len, descending=True)
        concat = concat[sorted_sen_indexes]

        # Remove padding
        none_padding_length = torch.sum(sorted_sen_len != 0).item()
        concat = concat[:none_padding_length]
        sorted_sen_len = sorted_sen_len[:none_padding_length]

        # add dropout
        concat = self.before_answer_gru_dropout(concat)
        packed = torch.nn.utils.rnn.pack_padded_sequence(concat,
                                                         sorted_sen_len,
                                                         batch_first=True)
        output, hidden = self.answer_gru(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Add padding
        output = torch.cat([output, torch.zeros(
            batch_size * sents_num - none_padding_length,
            output.size(1), output.size(2), device=DEVICE)], dim=0)

        # Recover order
        _, unsort_indexes = torch.sort(sorted_sen_indexes)
        output = output[unsort_indexes]
        # Recover shape, Q -> (N, S, W, H)
        output = output.view(batch_size, sents_num, word_num, -1)
        output = self.before_dense_dropout(output)

        # logits -> (N, S, W, L)
        logits = self.hidden2tag(output)
        return logits


class TDDMN(nn.Module):
    def __init__(self, word_vocab, entity_vocab, config):
        super(TDDMN, self).__init__()
        self.num_of_pass = config.get("num_of_pass")

        self.input_module = InputModule(config)
        self.question_module = QuestionModule(config)
        self.memory = EpisodicMemory(config)
        self.answer_module = AnswerModule(config)

        # Fine-tuning embedding
        self.input_conv2d = nn.Conv2d(in_channels=1,
                                      out_channels=WORD_EMBED_SIZE
                                      + int(config.get("entity_embed_size")),
                                      kernel_size=(1, WORD_EMBED_SIZE
                                                   + int(config.get("entity_embed_size"))))
        self.question_conv2d = nn.Conv2d(in_channels=1,
                                         out_channels=WORD_EMBED_SIZE
                                         + int(config.get("entity_embed_size")),
                                         kernel_size=(1, WORD_EMBED_SIZE
                                                      + int(config.get("entity_embed_size"))))

        self.config = config
        self.init_model()

        # Init embedding layer after xavier init
        self.word_embed = nn.Embedding(*word_vocab.vectors.shape,
                                       padding_idx=word_vocab.stoi[PAD_TOKEN])
        self.word_embed.weight.data.copy_(
            word_vocab.vectors
        )
        # Freeze the word embedding
        self.word_embed.weight.requires_grad = False
        self.entity_embed = nn.Embedding(len(entity_vocab.stoi),
                                         config.get("entity_embed_size"),
                                         padding_idx=entity_vocab.stoi[PAD_TOKEN])

    def init_model(self):
        for name, param in self.named_parameters():
            if param.data.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, doc_input_seqs, doc_input_entities, doc_input_lengths):
        logger = logging.getLogger("[TDDMN]")
        logger.setLevel(logging.INFO)

        # Unpack doc input lengths
        doc_len = doc_input_lengths["doc_len"]
        sen_len = doc_input_lengths["sen_len"]

        # Move inputs to proper devices
        doc_input_seqs = doc_input_seqs.to(DEVICE)
        doc_input_entities = doc_input_entities.to(DEVICE)
        sen_len = sen_len.to(DEVICE)
        doc_len = doc_len.to(DEVICE)

        # Output shape information
        logger.debug("doc_input_seqs: %s, doc_input_entities: %s"
                     % (str(doc_input_seqs.size()),
                        str(doc_input_entities.size())))

        # Get embeddings
        doc_input_seqs_embedded = self.word_embed(doc_input_seqs)
        doc_input_entities_embedded = self.entity_embed(doc_input_entities)
        logger.debug("doc_input_seqs_embedded: %s, doc_input_entities_embedded: %s" %
                     (str(doc_input_seqs_embedded.size()),
                      str(doc_input_entities_embedded.size())))

        # doc_input_embedded -> (N, S, W, V)
        doc_input_embedded = torch.cat([doc_input_seqs_embedded,
                                        doc_input_entities_embedded], dim=3)
        logger.debug("doc_input_embedded: %s" % str(doc_input_embedded.size()))

        if self.config.get("conv"):
            # Add conv 1xV
            batch_size, sent_num, word_num, embed_size = doc_input_embedded.size()

            # (N, S, W, V) -> (N*S, W, V)
            doc_input_embedded = doc_input_embedded.view(batch_size*sent_num, word_num, embed_size)
            # (N*S, W, V) -> (N*S, 1, W, V)
            doc_input_embedded = doc_input_embedded.unsqueeze(1)
            # (N*S, 1, W, V) -> (N*S, V, W, 1)
            doc_input_embedded_input_conv2d = functional.relu(self.input_conv2d(doc_input_embedded))
            # (N*S, V, W, 1) -> (N*S, V, W) -> (N*S, W, V) -> (N, S, W, V)
            doc_input_embedded_input_conv2d = doc_input_embedded_input_conv2d.squeeze(3).transpose(1, 2).view(
                batch_size, sent_num, word_num, -1
            )
            # same
            doc_input_embedded_question_conv2d = functional.relu(self.question_conv2d(doc_input_embedded))
            doc_input_embedded_question_conv2d = doc_input_embedded_question_conv2d.squeeze(3).transpose(1, 2).view(
                batch_size, sent_num, word_num, -1
            )
        else:
            doc_input_embedded_input_conv2d = doc_input_embedded
            doc_input_embedded_question_conv2d = doc_input_embedded

        # Get facts
        contexts = doc_input_embedded_input_conv2d

        # facts -> (N, S, H) word_attns -> (N, S, W)
        facts, word_attns = self.input_module(contexts, sen_len, doc_len)
        logger.debug("facts size: %s" % str(facts.size()))

        # Q_star -> (N, S, H)  question_for_memory
        # Q -> (N, S, W, H)    question_for_answer

        Q_star, Q = self.question_module(doc_input_embedded_question_conv2d, sen_len)

        # whether to include empty questions
        if self.config.get("empty_question", False):
            Q_star = torch.zeros_like(Q_star)
        # M -> (N, S, H)
        M = Q_star
        facts_attns = []  # facts attention in different pass
        for _ in range(self.num_of_pass):
            # facts_attn -> (N, S, S)
            M, facts_attn = self.memory(
                facts, M,
                Q_star, doc_len)
            facts_attns.append(facts_attn)
        logger.debug("Memory size: %s" % str(M.size()))

        preds = self.answer_module(M, Q, sen_len)
        return preds, word_attns, facts_attns
