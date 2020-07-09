# -*- coding: utf-8 -*-
from urnai.runner.runnerbuilder import RunnerBuilder
from urnai.runner.parserbuilder import ParserBuilder
import os

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

    parser = ParserBuilder.DefaultParser()
    args = parser.parse_args()
    runner = RunnerBuilder.build(parser, args)

    runner.run()
