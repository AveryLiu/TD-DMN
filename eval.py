import torch


class Evaluator(object):
    def __init__(self, fold):
        self.golden_dict = {
            i: {
                 "train": [],
                 "test": []
            } for i in range(fold)
        }
        for i in range(fold):
            self.load_golden(i)
        return

    def doc_evaluate(self, logits, doc_ids, offsets,
                     lengths, label_itos, mode, fold, verbose=False, partial=False):
        pred_classes = torch.argmax(logits, len(logits.size()) - 1)

        pred_list = []
        batch_size = pred_classes.size(0)

        for b in range(batch_size):
            for i in range(len(offsets[b])):
                for j in range(len(offsets[b][i])):
                    pred_list.append(
                        (doc_ids[b], int(offsets[b][i][j]), int(lengths[b][i][j]),
                         label_itos[pred_classes[b][i][j]]
                    ))
        return Evaluator.evaluate_jiheng(
            self.golden_dict[fold][mode], pred_list, verbose, partial)

    def evaluate(self, logits, doc_ids, offsets, lengths, label_itos, mode, verbose=False):
        pred_classes = torch.argmax(logits, len(logits.size())-1)

        # Sanity check
        assert len(doc_ids) == len(offsets) == len(lengths) == len(logits)
        assert [len(x) for x in offsets] == [len(x) for x in lengths]

        pred_list = []
        for i in range(len(offsets)):
            for j in range(len(offsets[i])):
                pred_list.append(
                    (doc_ids[i], int(offsets[i][j]), int(lengths[i][j]),
                     label_itos[pred_classes[i][j]])
                )

        _, _, _, p, r, f1 = Evaluator.evaluate_jiheng(
            self.golden_dict[mode], pred_list, verbose)
        return p, r, f1

    def load_golden(self, fold):
        files = {"train": "./data/fold_{}/train/train.golden.dat".format(fold),
                 "test": "./data/fold_{}/test/test.golden.dat".format(fold)}

        for name, path in files.items():
            with open(path, "r") as f:
                golden_list = []
                lines = f.read().strip().split("\n")
                for line in lines:
                    tuples = line.split("\t")
                    golden_list.append((tuples[0], int(tuples[1]), int(tuples[2]), tuples[4]))
                self.golden_dict[fold][name] = golden_list

    @staticmethod
    def evaluate_jiheng(golden_list, pred_list, verbose=False, partial=False):
        """
        :param golden_list: [(docid, start, length, type), ...]
        :param pred_list: [(docid, start, length, type), ...]
        :param verbose
        :return:
        """
        pred_sum = 0.
        span_tp = 0.
        type_tp = 0.

        pred_list = list(set(pred_list))

        def overlap(g_s, g_e, p_s, p_e):
            if g_s > p_e or p_s > g_e:
                return False
            else:
                return True

        for docid, start, length, typename in pred_list:
            if typename == 'other':
                continue
            end = start + length - 1
            pred_sum += 1.
            for g_doc_id, gold_s, gold_len, gold_typename in golden_list:
                if docid != g_doc_id:
                    continue
                gold_e = gold_s + gold_len - 1
                if overlap(gold_s, gold_e, start, end):
                    span_tp += 1.
                    if typename == gold_typename:
                        type_tp += 1.
                    break

        if span_tp < 1:
            span_p, span_r, span_f1 = 0., 0., 0.
        else:
            span_p = span_tp / pred_sum * 100.
            span_r = span_tp / len(golden_list) * 100.
            span_f1 = 2 * span_p * span_r / (span_p + span_r)

        if type_tp < 1:
            type_p, type_r, type_f1 = 0., 0., 0.
        else:
            type_p = type_tp / pred_sum * 100.
            type_r = type_tp / len(golden_list) * 100.
            type_f1 = 2 * type_p * type_r / (type_p + type_r)

        if verbose:
            print("Span Pred Right: %s, Pred Num %s, Total Right: %s." % (span_tp, pred_sum, len(golden_list)))
            print("Type Pred Right: %s, Pred Num %s, Total Right: %s." % (type_tp, pred_sum, len(golden_list)))

        if not partial:
            return span_p, span_r, span_f1, type_p, type_r, type_f1
        else:
            return type_tp, pred_sum, len(golden_list)