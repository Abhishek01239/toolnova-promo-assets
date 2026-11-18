"""python keystills.py key:frac [key:frac ...]  ->  work/stills.png  (also accepts plain seconds)"""
import sys
import build as Bd
f, r = Bd.make_film()
ts = []
for a in sys.argv[1:]:
    if ":" in a:
        k, fr = a.split(":")
        ts.append(f.T(k, float(fr)))
    else:
        ts.append(float(a))
Bd.stills(f, r, ts)
