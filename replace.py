#!/usr/bin/env python3
"""Replace flagged words in an existing candidate CSV.

A row is flagged if its word is:
  (a) not in the SG list (brys-in-sg.csv),
  (b) listed in exclusions.txt, or
  (c) passed via --reject on the command line.

Substitutes are drawn from the SG-filtered Brysbaert noun pool, restricted to
the row's Conc_Category bounds, with randomized selection from the top-K best
matches on length and Percent_known. Welch t-tests must pass (p >= 0.2) on
both length and Percent_known or the file is not written.

Examples:
    # auto: re-roll anything not in SG / in exclusions.txt
    python3 replace.py candidate_01.csv

    # also reject specific words on this run
    python3 replace.py candidate_02.csv --reject smoker clothes

    # one-off rejects without editing exclusions.txt; permanent rejects should
    # be appended to exclusions.txt instead.

    # different seed
    python3 replace.py candidate_01.csv --seed 7
"""
import argparse
import sys
import pandas as pd
import numpy as np

from concreteness_lib import (
    BRYS_DEFAULT, SG_DEFAULT, EXCL_DEFAULT,
    HIGH_BOUNDS, LOW_BOUNDS,
    load_exclusions, load_sg, load_pool,
    pick_substitute, to_output, verify_stats,
)


def replace_group(group_df, pool, sg_words, exclusions, manual_reject,
                  conc_min, conc_max, rng, label):
    """Re-roll any rows in group_df that need replacement. Returns
    (new_df, [(idx, old, new), ...])."""
    g = group_df.copy().reset_index(drop=True)
    lower = g['Word'].str.lower()
    needs = (~lower.isin(sg_words)) | lower.isin(exclusions) | lower.isin(manual_reject)
    keep = ~needs
    n = int(needs.sum())
    print(f"  [{label}] {n} of {len(g)} flagged for replacement")
    if n == 0:
        return g, []

    used = set(lower)
    cat_pool = pool[(pool['Conc.M'] >= conc_min) & (pool['Conc.M'] <= conc_max)].copy()
    replacements = []

    for idx in g.index[needs]:
        replaced_idx = {r[0] for r in replacements}
        kept = g[keep | g.index.isin(replaced_idx)]
        if len(kept) == 0:
            kept = g  # fall back to whole group if nothing yet anchored

        avail = cat_pool[~cat_pool['Word'].str.lower().isin(used)]
        if avail.empty:
            raise RuntimeError(f"No available substitute in [{label}] for {g.at[idx, 'Word']!r}")

        pick = pick_substitute(avail, kept, rng)
        old = g.at[idx, 'Word']
        replacements.append((idx, old, pick['Word']))
        used.add(pick['Word'].lower())
        g.at[idx, 'Word'] = pick['Word']
        g.at[idx, 'Conc.M'] = pick['Conc.M']
        g.at[idx, 'Length (letters)'] = pick['word_length']
        g.at[idx, 'Percent_known'] = pick['Percent_known']

    return g, replacements


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv', help='Path to candidate_*.csv file to update in place')
    p.add_argument('--reject', nargs='*', default=[],
                   help='Extra words to reject this run (lowercased)')
    p.add_argument('--brys', default=str(BRYS_DEFAULT))
    p.add_argument('--sg', default=str(SG_DEFAULT))
    p.add_argument('--exclusions', default=str(EXCL_DEFAULT))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true',
                   help='Compute replacements but do not overwrite the CSV')
    args = p.parse_args()

    sg = load_sg(args.sg)
    excl = load_exclusions(args.exclusions)
    manual = {w.strip().lower() for w in args.reject}
    pool = load_pool(sg, excl | manual, brys_path=args.brys)
    rng = np.random.default_rng(args.seed)

    print(f"SG list: {len(sg)} | exclusions: {len(excl)} | "
          f"--reject: {len(manual)} | pool: {len(pool)}")
    print(f"file: {args.csv}")

    df = pd.read_csv(args.csv)
    if 'replacement' in df.columns:
        df = df.drop(columns=['replacement'])

    high = df[df['Conc_Category'] == 'High']
    low = df[df['Conc_Category'] == 'Low']

    high_new, repl_h = replace_group(high, pool, sg, excl, manual,
                                     *HIGH_BOUNDS, rng, 'High')
    low_new, repl_l = replace_group(low, pool, sg, excl, manual,
                                    *LOW_BOUNDS, rng, 'Low')

    out = to_output(high_new, low_new)
    ok, _, _ = verify_stats(out, label='final')
    if not ok:
        print(f"  FAILED stats threshold; not writing.")
        sys.exit(1)

    if args.dry_run:
        print("  --dry-run: no file written.")
    else:
        out.to_csv(args.csv, index=False)
        print(f"  wrote {len(out)} rows -> {args.csv}")

    if repl_h or repl_l:
        print("  replacements:")
        for _, old, new in repl_h + repl_l:
            print(f"    {old:>20s}  ->  {new}")
    else:
        print("  no replacements needed.")


if __name__ == '__main__':
    main()
