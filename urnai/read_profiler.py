# python -m cProfile -o output.txt example.py <args>
import pstats

p = pstats.Stats('qtable_profile.txt')
p.strip_dirs().sort_stats('cumulative').print_stats(30)