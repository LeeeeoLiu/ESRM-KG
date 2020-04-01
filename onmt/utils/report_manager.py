""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime

import onmt

from onmt.utils.logging import logger


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir
                               + datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate_e, learning_rate_d,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = onmt.utils.Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(
                step, num_steps, learning_rate_e, learning_rate_d, report_stats)
            self.progress_step += 1
        return onmt.utils.Statistics()

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr_e, lr_d, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr_e, lr_d, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate_e, learning_rate_d, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate_e, learning_rate_d, step)

    def _report_training(self, step, num_steps, learning_rate_e, learning_rate_d,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate_e, learning_rate_d, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate_e, learning_rate_d,
                                   self.progress_step)
        report_stats = onmt.utils.Statistics()

        return report_stats

    def _report_step(self, lr_e, lr_d, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            self.log('Train recall@20: %g' % train_stats.recall())
            self.log('Train mrr@20: %g' % train_stats.mrr())

            self.maybe_log_tensorboard(train_stats, "train", lr_e, lr_d, step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            self.log('Validation recall@20: %g' % valid_stats.recall())
            self.log('Validation mrr@20: %g' % valid_stats.mrr())

            self.maybe_log_tensorboard(valid_stats, "valid", lr_e, lr_d, step)
