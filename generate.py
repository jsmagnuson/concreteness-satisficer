#!/usr/bin/env python3
"""Generate concreteness candidate sets from scratch.

Pipeline: Brysbaert nouns -> filter (Percent_known>=0.93, Bigram==0) ->
SG filter -> exclusions -> randomized greedy length+Percent_known matching
across High/Low Conc.M groups. Verifies Welch t-tests pass before writing.

Examples:
    # default: produce candidate_01.csv (100/100) and candidate_02.csv (60/20)
    python3 generate.py

    # custom sizes
    python3 generate.py --c1-high 80 --c1-low 80
    python3 generate.py --only c2 --c2-high 50 --c2-low 15

    # different seed (for re-rolling the whole thing)
    python3 generate.py --seed 7
"""
import argparse
import sys
import numpy as np

from concreteness_lib import (
    BRYS_DEFAULT, SG_DEFAULT, EXCL_DEFAULT, ROOT,
    HIGH_BOUNDS, LOW_BOUNDS,
    load_exclusions, load_sg, load_pool,
    build_matched_pair, to_output, verify_stats,
)


def build_set(pool, n_high, n_low, rng, label, out_path, max_attempts=10):
    """Generate one matched set, retrying with fresh seeds if stats fail."""
    print(f"\n=== {label} (target: {n_high} high / {n_low} low) ===")
    for attempt in range(1, max_attempts + 1):
        high, low = build_matched_pair(pool, n_high, n_low, rng)
        out = to_output(high, low)
        ok, _, _ = verify_stats(out, label=f"{label} attempt {attempt}")
        if ok:
            out.to_csv(out_path, index=False)
            print(f"  wrote {len(out)} rows -> {out_path}")
            return True
    print(f"  FAILED to satisfy p>=0.2 after {max_attempts} attempts")
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--brys', default=str(BRYS_DEFAULT),
                   help='Path to Concreteness_ratings_Brysbaert_et_al_BRM.csv')
    p.add_argument('--sg', default=str(SG_DEFAULT),
                   help='Path to brys-in-sg.csv (one word per line, no header)')
    p.add_argument('--exclusions', default=str(EXCL_DEFAULT),
                   help='Path to exclusions.txt')
    p.add_argument('--out1', default=str(ROOT / 'candidate_01.csv'),
                   help='Output path for candidate_01.csv')
    p.add_argument('--out2', default=str(ROOT / 'candidate_02.csv'),
                   help='Output path for candidate_02.csv')
    p.add_argument('--c1-high', type=int, default=100, help='High Conc count for set 1')
    p.add_argument('--c1-low', type=int, default=100, help='Low Conc count for set 1')
    p.add_argument('--c2-high', type=int, default=60, help='High Conc count for set 2')
    p.add_argument('--c2-low', type=int, default=20, help='Low Conc count for set 2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--only', choices=['c1', 'c2', 'both'], default='both')
    p.add_argument('--max-attempts', type=int, default=10,
                   help='Retry seeds if stats fail')
    args = p.parse_args()

    sg = load_sg(args.sg)
    excl = load_exclusions(args.exclusions)
    pool = load_pool(sg, excl, brys_path=args.brys)

    n_high_pool = ((pool['Conc.M'] >= HIGH_BOUNDS[0]) & (pool['Conc.M'] <= HIGH_BOUNDS[1])).sum()
    n_low_pool = ((pool['Conc.M'] >= LOW_BOUNDS[0]) & (pool['Conc.M'] <= LOW_BOUNDS[1])).sum()
    print(f"SG list: {len(sg)} | exclusions: {len(excl)} | "
          f"pool: {len(pool)} (high={n_high_pool}, low={n_low_pool})")

    rng = np.random.default_rng(args.seed)
    ok = True
    if args.only in ('c1', 'both'):
        ok &= build_set(pool, args.c1_high, args.c1_low, rng,
                        'candidate_01', args.out1, args.max_attempts)
    if args.only in ('c2', 'both'):
        ok &= build_set(pool, args.c2_high, args.c2_low, rng,
                        'candidate_02', args.out2, args.max_attempts)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
