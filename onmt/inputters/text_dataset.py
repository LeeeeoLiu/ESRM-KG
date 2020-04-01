# -*- coding: utf-8 -*-
"""Define word-based embedders."""

from collections import Counter
from itertools import chain
import io
import codecs
import sys

import torch
import torchtext

from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD,
                                         PAD_WORD, BOS_WORD, EOS_WORD)
from onmt.utils.misc import aeq


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src_sku)
            out_examples.append(example)

        def filter_pred(example):
            """ ? """
            return 2 < len(example.src_sku)

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src_sku), len(ex.tgt)
        return len(ex.src_sku)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side, topk_keywords=20):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate, side, topk_keywords=topk_keywords)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(text_iter, truncate, side, topk_keywords=20):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            sessions = line.strip('\n').split('||')
            check_flag = True
            for s in sessions:
                if '\t' not in s:
                    print(s)
                    check_flag = False
                    break
                if len(s.split('\t')) != 3:
                    print(sessions)
                    print(s.split('\t'))
                    assert len(s.split('\t')) == 3
            if not check_flag:
                break
            item_sku_id = [s.split('\t')[1] for s in sessions]
            in_words = [_w for s in sessions for _w in s.split('\t')[2].split()]

            stop_words = ['（', '）', '的', '【', '】', '，',
                          '。', '～', '*', '.', '/', '(', ')', '+', '：', '""']
            in_counter = Counter(_w for _w in in_words if _w not in stop_words)

            N = int(topk_keywords)
            in_top_N = [_s[0] for _s in in_counter.most_common(N)]

            words, feats, n_feats = TextDataset.extract_text_features(in_top_N)

            example_dict = {side + "_sku": item_sku_id,
                            "indices": i}
            if side == 'src':
                example_dict["tgt"] = in_top_N
            yield example_dict, n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        # New Add Fields
        for _src in ["src", "tgt"]:
            fields[_src+'_sku'] = torchtext.data.Field(pad_token=PAD_WORD, include_lengths=True)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            sessions = cf.readline().strip('\n').split('||')
            for _s in sessions:
                if len(_s.split('\t')) != 3:
                    raise AssertionError('error: ' + _s)
            f_line = []
            for s in sessions:
                f_line.extend(s.split('\t')[2].split())
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats
