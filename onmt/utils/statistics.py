""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, loss_sku=0, recall_count=0, mrr_count=0, q_count=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0

        self.loss_sku = loss_sku
        self.recall_count = recall_count
        self.mrr_count = mrr_count
        self.q_count = q_count

        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """ 
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        self.loss_sku += stat.loss_sku
        self.recall_count += stat.recall_count
        self.mrr_count += stat.mrr_count
        self.q_count += stat.q_count
        

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        if self.n_words == 0:
            return 0
        else:
            return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        if self.n_words == 0:
            return 0
        else:
            return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        if self.n_words == 0:
            return 0
        else:
            return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def recall(self):
        """ compute rouge """
        return (self.recall_count+0.0) / self.q_count if self.q_count else 0

    def mrr(self):
        """ compute mrr """
        return (self.mrr_count+0.0) / self.q_count if self.q_count else 0

    
    def rouge(self):
        """ compute rouge """
        # TODO
        pass  

    def output(self, step, num_steps, learning_rate_e, learning_rate_d, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; Recall@20: %7.5f; Mrr@20: %7.5f; click loss: %7.5f; sent loss: %7.5f; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr_e: %7.5f; lr_d: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps, self.recall(), self.mrr(), self.loss_sku, self.loss,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate_e,
               learning_rate_d,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate_e, learning_rate_d, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr_e", learning_rate_e, step)
        writer.add_scalar(prefix + "/lr_d", learning_rate_d, step)
