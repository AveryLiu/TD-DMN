from iterator import DMNIterator
from loss import cal_loss_with_attn
from torch import optim
from eval import Evaluator
from constants import *
import torch
import logging
from utils.ResultWriter import ResultWriter
from utils.utils import init_logging
import os
import datetime
from tensorboardX import SummaryWriter
from models.TDDMN import TDDMN
import json
from config import params
import argparse


class Trainer(object):
    def __init__(self):
        self.evaluator = Evaluator(fold=KFOLD_NUM)
        return

    def train(self, model_cls, params, **kwargs):
        train_args = kwargs.get("train_args")

        logger = logging.getLogger("[TRAIN]")
        logger.setLevel(logging.INFO)

        logging.info(params)
        iterator = DMNIterator()

        writer = SummaryWriter(log_dir="./runs/{}_{}_fold{}_pass{}".format(
            train_args.identifier, datetime.datetime.now(),
            train_args.fold_num, params.get("num_of_pass")))

        train_batch_size = params.get("train_batch_size")
        fold_num = train_args.fold_num
        writer.add_text("config", "{}_{}".format(fold_num, train_args.identifier))
        writer.add_text("params", "{}".format(json.dumps(params)))
        train_iter, test_iter = iterator.get_iters(train_batch_size, fold_num)
        word_vocab, entity_vocab, label_vocab = iterator.get_vocabs()
        model = model_cls(word_vocab, entity_vocab, params)
        model.to(DEVICE)

        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=params.get("learning_rate"), weight_decay=params.get("weight_decay"))

        train_batch = iter(train_iter)
        test_batch = iter(test_iter)

        best_f, best_p, best_r = 0, 0, 0
        patience_counter = 0
        best_test_loss = 100

        while train_iter.epoch < train_args.max_train_epoch:
            batch = next(train_batch)
            (docs, doc_len, sent_len), entities = batch.TEXT, batch.ENTITY

            logits, word_attn, sent_attn = model(docs, entities,
                                                 {"doc_len": doc_len, "sen_len": sent_len})

            loss = cal_loss_with_attn(logits, batch.LABEL, sent_len,
                                      neg_pos_ratio=params.get("neg_pos_ratio", 0),
                                      neg_label=label_vocab.stoi["other"],
                                      pad_label=label_vocab.stoi[PAD_TOKEN])

            # Summary every 10 iteration
            if train_iter.iterations % 10 == 0:
                writer.add_scalar('data/train_loss', loss.item(), train_iter.iterations)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Evaluate test metric
            if train_iter.iterations % 5 == 0:
                logging.info("Starting to test")
                test_num = len(test_iter.dataset.examples)
                tested_count = 0

                tot_loss_sum, tot_loss_divider = 0, 0
                acc_tp, acc_pred_sum, acc_len_golden = 0, 0, 0
                while True:
                    # Evaluate test
                    if tested_count == test_num:
                        break

                    # Assure we don't mess up the test metric calculation
                    if tested_count > test_num:
                        logging.error("test number error")
                        exit(1)

                    batch = next(test_batch)
                    (docs, doc_len, sent_len), entities = batch.TEXT, batch.ENTITY

                    with torch.no_grad():
                        test_logits, word_attn, sent_attn = model(docs, entities,
                                                                  {"doc_len": doc_len,
                                                                   "sen_len": sent_len})
                        # test loss
                        loss_sum, loss_divider = cal_loss_with_attn(test_logits, batch.LABEL,
                                                                    sent_len, partial=True)
                        tot_loss_sum += loss_sum.item()
                        tot_loss_divider += loss_divider.item()

                        # eval test metrics
                        type_tp, pred_sum, len_golden = self.evaluator.doc_evaluate(
                            test_logits, batch.DOC_ID, batch.OFFSET,
                            batch.LENGTH, label_vocab.itos, "test", fold_num, verbose=False, partial=True)

                        acc_tp += type_tp
                        acc_pred_sum += pred_sum
                        acc_len_golden = len_golden
                    tested_count += len(docs)

                tot_loss = tot_loss_sum / tot_loss_divider

                writer.add_scalar('data/test_loss', tot_loss, train_iter.iterations)
                if acc_pred_sum == 0:
                    tot_p = 0
                else:
                    tot_p = acc_tp / acc_pred_sum * 100.
                tot_r = acc_tp / acc_len_golden * 100.
                if tot_p + tot_r == 0:
                    tot_f = 0
                else:
                    tot_f = (2 * tot_p * tot_r / (tot_p + tot_r))
                logger.info("Test Metric: p {}, r {}, f {}".format(tot_p, tot_r, tot_f))

                writer.add_scalar("data/test_p", tot_p, train_iter.iterations)
                writer.add_scalar("data/test_r", tot_r, train_iter.iterations)
                writer.add_scalar("data/test_f", tot_f, train_iter.iterations)
                if float(tot_loss) < best_test_loss or tot_f > best_f:
                    if tot_f > best_f:
                        best_f, best_p, best_r = tot_f, tot_p, tot_r
                        writer.add_scalar('data/best_f', best_f, train_iter.iterations)
                        writer.add_scalar('data/best_p', best_p, train_iter.iterations)
                        writer.add_scalar('data/best_r', best_r, train_iter.iterations)
                    best_test_loss = tot_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > train_args.patience:
                    break

                logger.info("train iteration:{}, "
                            "train epoch: {}, "
                            "training loss: {},"
                            "testing loss: {}".format(train_iter.iterations, train_iter.epoch, loss, float(tot_loss)))
                model.train()

        report = {
            "fold_num": fold_num,
            "results": {
                            "test_loss": float(tot_loss),
                            "test_p": best_p, "test_r": best_r, "test_f": best_f
                       },
            "params": params
        }

        # write results
        result_writer = ResultWriter("./results")
        identifier = os.environ.get("identifier")
        result_writer.write_result(identifier, fold_num, params.get("num_of_pass"), report)


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="Parse arguments for model training")
    parser.add_argument("--identifier", type=str, help="an identifier string that describes the model")
    parser.add_argument("--fold_num", type=int, help="dictates which fold to use")
    parser.add_argument("--max_train_epoch", type=int, help="maximum training epoch in training loop")
    parser.add_argument("--patience", type=int, help="epochs to wait before seeing new lowest test "
                                                     "loss or new highest test f")

    args = parser.parse_args()
    # params is model hyper-parameters and train_args is arguments that control training loop
    Trainer().train(TDDMN, params, train_args=args)
