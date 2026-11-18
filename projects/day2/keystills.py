"""usage: python keystills.py key[:frac] key[:frac] ... -> work/stills.png"""
import sys
import build as B
import story as ST
f, r = B.make_film()
ts = []
for a in sys.argv[1:]:
    k, _, fr = a.partition(":")
    ts.append(f.T(k, float(fr or 0.5)))
print([round(t, 1) for t in ts])
B.stills(f, r, ts)
