# -*- coding: utf-8 -*-
from urnai.runner.runnerbuilder import RunnerBuilder
from urnai.runner.parserbuilder import ParserBuilder
import sys, os

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    reload(sys)
    sys.setdefaultencoding('utf-8')

    parser = ParserBuilder.DefaultParser()
    args = parser.parse_args()
    runner = RunnerBuilder.build(parser, args)

    runner.run()
