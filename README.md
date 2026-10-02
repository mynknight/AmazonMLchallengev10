# V10: France cross-encoder veto on top of v9c

V10 = `~/MLchallenge_v9/output/v9c_v6afr` (V9 US/India k4_u_ea + v6a France `fin_nv_u` at 0.82) with a
narrow veto on France: a v6a France match is dropped when a France-adapted cross-encoder rejects it
**and** the names differ by one unrelated word (a vocabulary swap, e.g. `lycee` ↔ `club`).
V9 and `~/MLchallenge` are only read; everything V10 writes is under this folder.

| submission | US/India | France | public LB |
|---|---|---|---|
| v6a_france_roles (other session) | V5a | fin_nv_u 0.82 | **0.988261** |
| `~/MLchallenge_v9/output/v9c_v6afr` | k4_u_ea | = v6a | not uploaded |
| **`output/v10a_veto_swap02`** | k4_u_ea | v6a − 1,360 vetoed swap pairs | 0.988651 |
| `output/v10b_veto_add06` | k4_u_ea | v10a + 4,747 CE-confirmed additions (v6a prob 0.6–0.82, France CE ≥ 0.98, no swaps) | **0.988932 (best)** |

Upload order: **v9c first** (isolates the US/India change against v6a), **then v10a** (isolates the
veto: the only difference from v9c is 1,360 removed France pairs, 0.16% of France matches; nothing
added, candidate_pairs.tsv byte-identical). The expected leaderboard effect of v10a is small
(order 1e-4 either way).

## Steps
1. `ce_fr10.py` (`run_ce_fr10.sh`, GPU, ~9 min per fold): fine-tune V9's four `ce_e5k4_f*`
   cross-encoders on 1.2M labelled US/India pairs + 739k France pseudo-labels from v6a (positive =
   query's best candidate with v6a prob ≥ 0.98, negative = prob ≤ 0.02). US/India validation log-loss
   barely moves (0.0454–0.0465 vs 0.0451–0.0459 for e5k4), so no forgetting.
2. `veto_fr.py` → `reports/veto_fr.txt`: the plain e5k4 CE would veto 30–55% of French noise-word
   matches (it misreads French), the France-adapted CE only 0.3–0.7%, so only the adapted CE is usable.
3. `veto_sim.py` → `reports/veto_sim.txt`: on US/India validation a CE veto on a no-CE word-role
   model (the France model's type) helps: held-out test-like F0.5 0.98627 → 0.98813 (veto only,
   threshold fixed); vetoed pairs at CE < 0.2 are 23% true.
4. Inspection of the 4,834 France pairs the adapted CE vetoes at 0.2: a third are exact names, many
   of them legal-form spacing (`s c i` vs `sci`), extra noise words or one-letter typos, which look
   true. Such a blind veto is unsafe. The pairs that are one-for-one unrelated word swaps
   (`pharmacie` ↔ `primaire`, `residence` ↔ `amis`) look false and are exactly what the leaderboard
   rewarded v6a for rejecting. Only those are vetoed.
5. `make_sub10.py <out> --v 0.2 --classes swap`: names are compared as core-token sets after joining
   runs of single letters; a vetoed query is left unmatched (no fallback to its second candidate);
   then `make_sub9.py` writes the TSVs and runs the official validator (PASS).

## v10b: CE-confirmed additions
On US/India validation, a no-CE word-role model's best match with prob in [0.6, 0.82) is only 64–75%
true, but 96–99.6% true when the CE gives ≥ 0.98, well above the ~78% an added pair needs to raise
F0.5. v10b applies this to France with the France CE, excluding word swaps. Word-swap matches above
0.99 are 100% true on US/India, so the remaining France swaps are not vetoed wholesale.
`python make_sub10.py ../output/v10b_veto_add06 --v 0.2 --classes swap --add 0.6:0.98`

## Files
- `src/common10.py` paths · `src/ce_fr10.py` France CE · `src/veto_fr.py` veto shares ·
  `src/veto_sim.py` US/India veto simulation · `src/make_sub10.py` submission
- `work/ce_fr/` the four France CEs and their test logits · `work/test_fr_v6a.parquet` v6a France
  scores · `work/veto_v10a_veto_swap02.parquet` every v6a France match with its class, CE prob and veto flag
- Other variants: `python make_sub10.py ../output/<name> --v 0.1 --classes swap,other` etc.

Env: conda `mfa`, `HF_HOME=/home/mayankc/.cache/huggingface HF_HUB_OFFLINE=1`.
