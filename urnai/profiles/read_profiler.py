# python -m cProfile -o output.txt example.py <args>
import pstats

profile_path = 'cartpole_v1_profile.txt'
p = pstats.Stats(profile_path)
p.strip_dirs().sort_stats('cumulative').print_stats(30)