"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss


def build_loss_compute(model, tgt_vocab, tgt_sku_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = NMTLossCompute(model.predictor, model.generator, opt.loss_weight_e, tgt_vocab,  tgt_sku_vocab,
        #compute = NMTLossCompute(model.predictor, model.generator, tgt_vocab, tgt_sku_vocab,
                                 label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, predictor, generator, loss_w_sku, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        self.loss_w_sku = loss_w_sku
        self.loss_w_kph = 1-loss_w_sku

    def _make_shard_state(self, batch, encoded, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, encoded, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, encoded, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, encoded, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, encoded, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target, loss_sku, scores_sku, target_sku):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """

        # for explanation
        if scores is not None:
            pred = scores.max(1)[1]
            non_padding = target.ne(self.padding_idx)
            num_correct = pred.eq(target) \
                            .masked_select(non_padding) \
                            .sum() \
                            .item()
            num_non_padding = non_padding.sum().item()
        else:
            num_correct = 0
            num_non_padding = 0

        # for click prediction
        _recall = 0
        _mrr = 0
        _q_count = 0
        loss_sku_item = 0
        if scores_sku is not None:
            loss_sku_item = loss_sku.item()
            sorted_index_t = np.argsort(-scores_sku.detach())[:,:20].to(scores_sku.device) # [b X 20]
        
            label_t = target_sku.unsqueeze(1).expand_as(sorted_index_t) # [b X 20]
            _result_tensor = label_t - sorted_index_t
            _q_count = _result_tensor.size(0)
            _correct_count = (_result_tensor==0).nonzero().size(0)
            if _correct_count > 0:
                _recall = _correct_count
                _mrr_tensor = (_result_tensor==0).nonzero()[:,1].float() + 1
                _ones_tensor = torch.ones(_mrr_tensor.size()).float().to(scores_sku.device)
                _mrr = torch.div(_ones_tensor, _mrr_tensor).sum().item()
        # print(_recall, _mrr, _q_count)

        return onmt.utils.Statistics(loss.item() if not type(loss)==int else 0, num_non_padding, num_correct, loss_sku_item, _recall, _mrr, _q_count)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, predictor, generator, loss_w_sku, tgt_vocab, tgt_sku_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(predictor, generator, loss_w_sku, tgt_vocab)
        if generator is not None:
            self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        else:
            self.sparse = False
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, len(tgt_vocab), ignore_index=self.padding_idx
            )
            self.criterion_e = LabelSmoothingLoss(
                label_smoothing, len(tgt_sku_vocab), ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
            self.criterion_e = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, encoded, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "encoded": encoded,
        }

    def _compute_loss(self, batch, output=None, target=None, encoded=None):
        if output is not None:
            bottled_output = self._bottle(output)
        else:
            bottled_output = None

        if encoded is not None:
            bottled_encoded = self._bottle(encoded)
            scores_sku = self.predictor(bottled_encoded)
            gtruth_sku = batch.tgt_sku[0].view(-1)
            loss_sku = self.criterion_e(scores_sku, gtruth_sku)
        else:
            scores_sku = None
            gtruth_sku = None
            loss_sku = None

        if self.generator is not None:
            scores = self.generator(bottled_output)
        else:
            scores = None
        #scores_sku = self.predictor(bottled_encoded)

        gtruth = target.view(-1)
        #gtruth_sku = batch.tgt_sku[0].view(-1)
        #loss_sku = self.criterion_e(scores_sku, gtruth_sku)
        #print('session loss: {}'.format(loss_sku.item()))
        if self.generator is not None:
            loss = self.criterion(scores, gtruth)
        else:
            loss = None
        #print('generation loss: {}'.format(loss.item()))

        stats = self._stats(loss.clone() if loss is not None else 0, scores, gtruth, loss_sku.clone() if loss_sku is not None else 0, scores_sku, gtruth_sku)
        #weight_a = 1
        #weight_b = 1
        if self.generator is not None:
            if encoded is not None:
                loss_all = self.loss_w_sku * loss_sku + self.loss_w_kph * loss
            #loss_all = weight_a*loss_sku + weight_b*loss
            else:
                loss_all = loss
        else:
            loss_all = loss_sku
        # print('loss: {}'.format(loss_all.item()))

        return loss_all, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
