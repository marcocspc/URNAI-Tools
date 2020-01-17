# -*- coding: utf-8 -*-
from .runner.runnerbuilder import RunnerBuilder
from .runner.parserbuilder import ParserBuilder

def main():
    parser = ParserBuilder.DefaultParser()
    args = parser.parse_args()
    runner = RunnerBuilder.build(parser, args)

    runner.run()
