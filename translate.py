#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import codecs
import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts



def main(opt):
    _logger_path = "logs/{}-test.log".format(opt.models[0].split('/')[1])
    _output_path = "logs/{}-output.log".format(opt.models[0].split('/')[1])
    logger = init_logger(_logger_path)
    logger.info('input_weight: {}'.format(opt.input_weight))
    logger.info('last_weight: {}'.format(opt.last_weight))
    _out_file = codecs.open(_output_path, 'w+', 'utf-8')
    logger.info('Start testing.')
    translator = build_translator(opt, report_score=True, logger=logger, out_file=_out_file)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
