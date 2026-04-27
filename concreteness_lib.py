"""Shared helpers for concreteness candidate generation/replacement.

Used by generate.py and replace.py.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
BRYS_DEFAULT = ROOT / 'Concreteness_ratings_Brysbaert_et_al_BRM.csv'
SG_DEFAULT = ROOT / 'brys-in-sg.csv'
EXCL_DEFAULT = ROOT / 'exclusions.txt'

P_THRESHOLD = 0.2  # Welch t-test must give p >= this on length and Percent_known
TOP_K = 20         # sample uniformly from this many best-matching candidates

# Concreteness category bounds (Conc.M from Brysbaert).
HIGH_BOUNDS = (4.5, 5.0)
LOW_BOUNDS = (1.0, 2.0)


def load_exclusions(path=EXCL_DEFAULT):
    """Read one-word-per-line exclusions file. '#' starts a comment."""
    words = set()
    for raw in Path(path).read_text().splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if line:
            words.add(line)
    return words


def load_sg(path=SG_DEFAULT):
    return set(pd.read_csv(path, header=None)[0].str.strip().str.lower())


def load_pool(sg_words, exclusions, brys_path=BRYS_DEFAULT):
    """Brysbaert nouns: SG-filtered, exclusions removed, well-known, single token."""
    df = pd.read_csv(brys_path)
    nouns = df[df['Dom_Pos'] == 'Noun'].copy()
    nouns = nouns[nouns['Percent_known'] >= 0.93]
    nouns = nouns[nouns['Bigram'] == 0]
    nouns['word_length'] = nouns['Word'].str.len()
    lower = nouns['Word'].str.lower()
    nouns = nouns[~lower.isin(exclusions) & lower.isin(sg_words)]
    return nouns


def score_row(row, target_len, target_pk):
    """Lower is better. Length scaled by 3 letters; Pct_known by 0.02."""
    return abs(row['word_length'] - target_len) / 3.0 + abs(row['Percent_known'] - target_pk) / 0.02


def pick_random_top_k(avail, scores, rng, top_k=TOP_K):
    """Take the top_k rows of `avail` by `scores` (a Series aligned to avail's
    index), with alphabetic ties broken by shuffle, then sample one uniformly."""
    ranked = avail.assign(_score=scores)
    ranked = ranked.sample(frac=1, random_state=rng.integers(0, 2**31 - 1))
    ranked = ranked.sort_values('_score', kind='mergesort').head(top_k)
    return ranked.iloc[int(rng.integers(0, len(ranked)))]


def _sample_by_length_dist(group_pool, n, target_props, rng):
    """Sample `n` words from group_pool such that the per-length proportions
    approximate `target_props` (a Series of {length: proportion}). Within a
    length bin, words are chosen uniformly at random. Capped by availability."""
    desired = (target_props * n).round().astype(int)
    # Resolve rounding deficit/surplus by largest-fractional-remainder.
    deficit = n - int(desired.sum())
    if deficit != 0:
        frac = (target_props * n) - desired
        order = frac.sort_values(ascending=(deficit < 0)).index
        for L in order:
            if deficit == 0:
                break
            step = 1 if deficit > 0 else -1
            if desired[L] + step >= 0:
                desired[L] += step
                deficit -= step
    chunks = []
    leftover = 0
    for L, k in desired.items():
        if k <= 0:
            continue
        avail = group_pool[group_pool['word_length'] == L]
        take = min(k, len(avail))
        leftover += k - take
        if take:
            chunks.append(avail.sample(n=take, random_state=rng.integers(0, 2**31 - 1)))
    out = pd.concat(chunks) if chunks else group_pool.iloc[0:0]
    if leftover > 0:
        # Backfill any deficit from words not yet drawn.
        used = set(out['Word'].str.lower())
        rest = group_pool[~group_pool['Word'].str.lower().isin(used)]
        if len(rest) >= leftover:
            out = pd.concat([out, rest.sample(n=leftover, random_state=rng.integers(0, 2**31 - 1))])
    return out.reset_index(drop=True)


def build_matched_pair(pool, n_high, n_low, rng):
    """Sample n_high from the High pool and n_low from the Low pool, stratified
    so both groups follow the same per-length distribution (the geometric mean
    of the two pools' length distributions). This matches mean and variance of
    word length across groups even when the pools differ systematically.
    Percent_known is verified post-hoc by the caller; with most words at >=0.97
    it tracks naturally."""
    high_pool = pool[(pool['Conc.M'] >= HIGH_BOUNDS[0]) & (pool['Conc.M'] <= HIGH_BOUNDS[1])]
    low_pool = pool[(pool['Conc.M'] >= LOW_BOUNDS[0]) & (pool['Conc.M'] <= LOW_BOUNDS[1])]
    if len(high_pool) < n_high:
        raise RuntimeError(f"Need {n_high} high words, pool has {len(high_pool)}")
    if len(low_pool) < n_low:
        raise RuntimeError(f"Need {n_low} low words, pool has {len(low_pool)}")

    h_counts = high_pool['word_length'].value_counts()
    l_counts = low_pool['word_length'].value_counts()
    common = h_counts.index.intersection(l_counts.index).sort_values()
    p_h = (h_counts.loc[common] / h_counts.loc[common].sum())
    p_l = (l_counts.loc[common] / l_counts.loc[common].sum())
    target = (p_h * p_l).pow(0.5)  # geometric mean — favors lengths well-represented in both
    target = target / target.sum()

    high = _sample_by_length_dist(high_pool, n_high, target, rng)
    low = _sample_by_length_dist(low_pool, n_low, target, rng)
    return high, low


def pick_substitute(avail, kept_df, rng, top_k=TOP_K):
    """For replace.py: score each row in `avail` by the candidate's own length
    and Percent_known distance from the kept rows' means. Returns one pick."""
    target_len = kept_df['Length (letters)'].mean()
    target_pk = kept_df['Percent_known'].mean()
    scores = avail.apply(lambda r: score_row(r, target_len, target_pk), axis=1)
    return pick_random_top_k(avail, scores, rng, top_k=top_k)


def to_output(high_df, low_df):
    """Format final output DataFrame matching candidate_*.csv schema.
    Accepts either 'word_length' (from the pool) or 'Length (letters)' (from
    an existing candidate CSV) as the length column."""
    def normalize(df, label):
        if 'word_length' in df.columns:
            df = df.rename(columns={'word_length': 'Length (letters)'})
        out = df[['Word', 'Conc.M', 'Length (letters)', 'Percent_known']].copy()
        out['Conc_Category'] = label
        return out

    h = normalize(high_df, 'High')
    l = normalize(low_df, 'Low')
    return pd.concat([h, l], ignore_index=True).sort_values(
        'Conc.M', ascending=False).reset_index(drop=True)


def verify_stats(df, label='', threshold=P_THRESHOLD):
    """Returns (passed, p_len, p_pk). Prints a one-line summary."""
    high = df[df['Conc_Category'] == 'High']
    low = df[df['Conc_Category'] == 'Low']
    _, p_len = stats.ttest_ind(high['Length (letters)'], low['Length (letters)'])
    _, p_pk = stats.ttest_ind(high['Percent_known'], low['Percent_known'])
    tag = f"[{label}] " if label else ""
    print(f"  {tag}n_high={len(high)} n_low={len(low)}  "
          f"len M={high['Length (letters)'].mean():.2f}/{low['Length (letters)'].mean():.2f} p={p_len:.3f}  "
          f"pk M={high['Percent_known'].mean():.4f}/{low['Percent_known'].mean():.4f} p={p_pk:.3f}  "
          f"Conc M={high['Conc.M'].mean():.2f}/{low['Conc.M'].mean():.2f}  "
          f"{'PASS' if p_len >= threshold and p_pk >= threshold else 'FAIL'}")
    return p_len >= threshold and p_pk >= threshold, p_len, p_pk
