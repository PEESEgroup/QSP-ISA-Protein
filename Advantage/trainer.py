
import model
import math

#modifies [d[k]] such that d[k] is optimal for maximizing measurement output
# in a VQC [c] under input vector [v], measurement [m]
def select(d, k, c, v, m):
    d[k] = 0
    ez = c.run(v, d, m)
    d[k] = math.pi / 2
    ep = c.run(v, d, m)
    d[k] = -math.pi / 2
    ed = c.run(v, d, m)
    a = (ep - ed) / 2
    b = ez - (ep + ed) / 2
    t = -math.pi / 2
    if b != 0: t = math.atan(a / b)
    o = a * math.sin(t) + b * math.cos(t)
    if o < 0: d[k] = t + math.pi / 2
    else: d[k] = t

