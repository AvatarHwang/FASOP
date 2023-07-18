"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

from collections import defaultdict

# returns the rank to axis. If pp_deg=dp_deg=mp_deg=2, rank 3 gives (0,1,1).
# This is deepspeed method
def rank2axis(rank, mp_deg, dp_deg, pp_deg):
    pp = rank // (mp_deg * dp_deg)
    remainder = rank % (mp_deg * dp_deg)

    dp = remainder // (mp_deg)
    remainder = remainder % mp_deg

    mp = remainder

    return (pp, dp, mp)

# returns the axis to rank. If pp_deg=dp_deg=mp_deg=2, (0,1,1) gives 3
def axis2rank(axis, mp_deg, dp_deg, pp_deg):
    pp, dp, mp = axis
    return mp + mp_deg * dp + (mp_deg * dp_deg) * pp

def factor(N, upper=None):
    if upper is None:
        upper = N
    ret = []
    for i in range(1, upper+1):
        if N % i == 0:
            ret.append(i)
    return ret


def amp_no_placement_strategy(M, N, gbs, known, num_layers):
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        W = M * N
        for h in factor(min(M, 4)): # mp, only max 4 is supported
            assert M*N % h == 0
            remain = M*N // h
            for w in factor(remain): # dp
                pp_degree = M*N / (h*w)
                # if pp_degree is not int
                if pp_degree != int(pp_degree):
                    continue
                if (W / pp_degree) % w != 0:
                    continue
                if gbs % (w) != 0:
                    continue
                if pp_degree > num_layers:
                    continue
                for mbs in factor(gbs // w):
                    ele_count += 1
                    known[mbs].append((h, w))
    if len(known.keys()) == 0:
        return None

    mbs = list(known.keys())[0]
    (h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return h, w, mbs, known