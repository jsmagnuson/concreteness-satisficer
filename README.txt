# =============================================================================
# README.txt - concreteness candidate workflow
# =============================================================================
# This file doubles as documentation AND a runnable shell script.
# Comments start with '#'. Uncommented lines are executable commands.
#
# To execute the basic from-scratch workflow:
#     source README.txt           # runs in current shell
#     bash README.txt             # runs in a subshell
#
# To read it as docs only, just open it in an editor.
# =============================================================================


# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
# Requires Python 3 with pandas, scipy, numpy.
# The PY shortcut below uses `uv` to auto-fetch deps on each invocation.
# If you have a venv with the packages installed, change PY to "python3".

PY="uv run --with pandas --with scipy --with numpy python3"


# -----------------------------------------------------------------------------
# INPUT / OUTPUT FILES (defaults shown; all are CLI-configurable)
# -----------------------------------------------------------------------------
#   Concreteness_ratings_Brysbaert_et_al_BRM.csv  -- Brysbaert norms (source)
#   brys-in-sg.csv                                -- SG vocabulary filter (one word per line)
#   exclusions.txt                                -- editable reject list (one word per line, '#' for comments)
#   candidate_01.csv                              -- output set 1 (default 100 High + 100 Low)
#   candidate_02.csv                              -- output set 2 (default 60 High + 20 Low)
#
# Override flags (all scripts):
#   generate.py            --brys PATH  --sg PATH  --exclusions PATH  --out1 PATH  --out2 PATH
#   replace.py CSV         --brys PATH  --sg PATH  --exclusions PATH
#   statistical_tests.py CSV
#
# To use a different vocabulary filter or norms file across the whole pipeline:
#   $PY generate.py        --brys other_norms.csv --sg other_vocab.csv --out1 setA.csv --out2 setB.csv
#   $PY replace.py setA.csv --brys other_norms.csv --sg other_vocab.csv
#   $PY statistical_tests.py setA.csv


# -----------------------------------------------------------------------------
# STEP 1 - GENERATE FROM SCRATCH
# -----------------------------------------------------------------------------
# Stratified-by-length random sample from the SG-filtered Brysbaert noun pool.
# High = Conc.M >= 4.5;  Low = Conc.M <= 2.0.
# Welch t-tests on length and Percent_known must pass p >= 0.2 or the file is
# NOT written; up to 10 retries with different seeds per file.

$PY generate.py

# # Common variants (commented out by default):
# $PY generate.py --seed 7                          # different random seed
# $PY generate.py --only c1                         # just candidate_01
# $PY generate.py --only c2 --c2-high 50 --c2-low 15
# $PY generate.py --c1-high 80 --c1-low 80          # custom sizes
# $PY generate.py --out1 setA.csv --out2 setB.csv   # custom output paths
# $PY generate.py --sg my_other_vocab.csv           # alternate vocab filter
# $PY generate.py --brys path/to/other_norms.csv    # alternate norms file
# $PY generate.py --exclusions my_excl.txt          # alternate exclusions list
# $PY generate.py --max-attempts 25                 # more retries if stats keep failing


# -----------------------------------------------------------------------------
# STEP 2 - ITERATIVELY REJECT BAD CANDIDATES
# -----------------------------------------------------------------------------
# After generate.py, open the candidate CSVs and review the words. Any words
# you don't want should be appended to exclusions.txt under the
# "# --- manual rejects ---" heading near the bottom. One word per line,
# lowercase. Then run replace.py to swap them out -- it also automatically
# replaces anything that ended up not in brys-in-sg.csv.
#
# Substitutes are drawn from the SG-filtered Brysbaert pool, restricted to the
# row's Conc category, randomized over the top-20 best matches on length and
# Percent_known. Files are only written if t-tests still pass.

$PY replace.py candidate_01.csv
$PY replace.py candidate_02.csv

# # Variants:
# $PY replace.py candidate_02.csv --reject smoker clothes  # one-off rejects (don't touch exclusions.txt)
# $PY replace.py candidate_02.csv --reject smoker --dry-run # preview without writing
# $PY replace.py candidate_01.csv --seed 7                  # different random substitutes
# $PY replace.py setA.csv --exclusions my_excl.txt --sg other_vocab.csv
#
# # Loop: edit exclusions.txt, re-run replace.py, repeat until the candidate
# # files contain only words you accept.


# -----------------------------------------------------------------------------
# STEP 3 - VERIFY STATISTICS
# -----------------------------------------------------------------------------
# Independent-samples t-test, Mann-Whitney U, Cohen's d, and Shapiro-Wilk
# normality tests on length and Percent_known across High vs Low groups.

$PY statistical_tests.py candidate_01.csv
$PY statistical_tests.py candidate_02.csv


# =============================================================================
# FILE LAYOUT
# =============================================================================
#   generate.py            from-scratch generator (stratified-by-length random)
#   replace.py             targeted replacer (random top-20 substitution)
#   statistical_tests.py   t-test / Mann-Whitney / Cohen's d / Shapiro-Wilk
#   concreteness_lib.py    shared helpers; not run directly
#
#   exclusions.txt         editable; categorized rejection list
#
#   Concreteness_ratings_Brysbaert_et_al_BRM.csv   input: Brysbaert norms
#   brys-in-sg.csv                                 input: SG vocabulary filter
#
#   candidate_01.csv       output: set 1 (default 200 rows)
#   candidate_02.csv       output: set 2 (default 80 rows)
#
#   ARC/                   archived superseded scripts (not in active use)
#   README.txt             this file
# =============================================================================
