""" Translation main class """
from __future__ import division, unicode_literals
from __future__ import print_function

import torch
import onmt.inputters as inputters


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src_sku, src_vocab, src_raw, pred, attn):
        vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == inputters.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and (attn is not None) and (src_sku is not None):
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[inputters.UNK]:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)
        src_sku = batch.src_sku[0].data.index_select(1, perm)
        tgt_sku = batch.tgt_sku[0].data.index_select(1, perm)
        pred_sku = translation_batch["pred_sku"].data

        if self.has_tgt:
            tgt = batch.tgt.data.index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            src_vocab = self.data.src_vocabs[inds[b]] \
                if self.data.src_vocabs else None
            src_raw = self.data.examples[inds[b]].src_sku
            tgt_sku_raw = self.data.examples[inds[b]].tgt_sku
            sku_vocab = self.fields["src_sku"].vocab
            pred_sku_raw = [sku_vocab.itos[_idx] for _idx in pred_sku[b]]

            _src_sku = src_sku[:, b] if src_sku is not None else None
            pred_sents = [self._build_target_tokens(
                _src_sku,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    _src_sku, src_vocab, src_raw, tgt[1:, b], None)

            translation = Translation(_src_sku,
                                      src_raw, tgt_sku, tgt_sku_raw,pred_sku_raw, pred_sents,
                                      attn[b], pred_score[b], gold_sent,
                                      gold_score[b])
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src_sku (`LongTensor`): src_sku word ids
        src_raw ([str]): raw src_sku words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src_sku,  src_raw, tgt_sku, tgt_sku_raw, pred_sku_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src_sku = src_sku
        self.src_raw = src_raw
        self.tgt_sku = tgt_sku
        self.tgt_sku_raw = tgt_sku_raw
        self.pred_sku_raw = pred_sku_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """
        output = '\nSENT NUM: {} \n'.format(sent_number)
        output += 'SEQ_LEN: {}\n'.format(len(self.src_raw))
        output += 'SRC_SKU: {}\n'.format(self.src_raw)
        output += 'TGT_SKU: {}\n'.format(self.tgt_sku_raw)
        output += 'PRED_SKU: {}\n'.format(self.pred_sku_raw)
        _pred_flag = self.tgt_sku_raw[0] in self.pred_sku_raw
        output += 'PRED_FLAG: {}\n'.format(_pred_flag)
        _rank = self.pred_sku_raw.index(
            self.tgt_sku_raw[0])+1 if _pred_flag else 'NULL'
        output += 'PRED_RANK: {}\n'.format(_rank)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED SENT: {}\n'.format( pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD SENT: {}\n'.format( tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
