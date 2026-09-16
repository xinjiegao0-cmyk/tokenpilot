# Structured benchmark v1 (software milestone v0.2.0)

This is a reproducible, narrow benchmark harness, not a published efficiency result.
The versioned dataset has 12 synthetic tasks (4 short, 4 medium, 4 long). Its four
families are lookup, latest revision, numeric sum and one-hop linked-record lookup.
Length tiers share task shapes and are not independent statistical replications.
Explicit record keys make selection easier than natural-language retrieval.

## Offline use

```sh
python3 -m benchmarks.run --dry-run
python3 -m benchmarks.run
python3 -m benchmarks.run --local-usd-per-cpu-second 0
python3 -m benchmarks.run --strategy tokenpilot --ablation no-page-in
```

The default fixture provider interprets the selected records without reading ground
truth. It is not a language model. Its token counts are UTF-8 byte units and its
USD rate is synthetic. Every record and comparison carries simulation metadata.
Omitting a local compute rate leaves net-compute accounting incomplete, even when
the fixture reports a synthetic model cost. Supplying zero explicitly models a zero
local CPU price; it does not prove that local computing is economically free.

Each attempt is flushed to a unique JSONL file in the ignored `benchmarks/results/`
directory. Summary JSON lists every candidate-vs-baseline comparison. Neither file
includes raw prompts or replies. Hashes link requests to public, versioned inputs.
Interrupted batches retain previously flushed attempts, but an in-flight request
can require provider reconciliation. Filesystem flush is not a durability guarantee
against power loss. Responses are hashed, not retained, so independent re-grading
requires rerunning the public fixture/model or separately authorized secure logging.

## Baselines and candidate

- Full-history includes all records.
- Sliding-window keeps recent complete records within 8192 UTF-8 bytes by default;
  its limit is configurable and is not a tokenizer/context-limit claim.
- Retrieval-context ranks lexical overlap with the query key and keeps 8 records.
  It is cheap and deterministic, with no embeddings or learned ranker.
- TokenPilot bypasses contexts below 2048 bytes; otherwise it matches structured
  keys, follows one dependency for join, and preserves source identity; duplicate source ids are rejected. The reusable compiler is
`tokenpilot.runtime.context.optimize_records`; raw artifacts are content-addressed
and retrieved through tenant-checked pointers.
  It bypasses reductions below 256 bytes. These are transparent heuristic floors,
  not empirically calibrated monetary break-even predictions.

Selection never sees expected answers. No strategy uses an LLM to estimate gain.
All use the same prompt instructions, model and output cap. Baseline limitations
matter: sliding can lose early evidence, lexical retrieval can miss indirect links,
and exact-key selection only works under this structured-record contract. A strong
retrieval baseline can already approach the candidate; comparisons against it are
retained. No assumption that TokenPilot always wins is encoded.

The planner chooses RETRIEVE, CALL_MODEL, VERIFY and STOP one at a time. Failure
stops without retries. Other action enum values reserve vocabulary but have no
runtime execution implementation and are not advertised as supported tools.
`QualityContract` and semantic-state/pointer types are early library primitives;
cross-request semantic reuse is not enabled in benchmark execution.

## Quality and overhead

Quality requires exact JSON answer type/value plus the exact ground-truth evidence
id set. Duplicate keys, extra output fields and duplicate/incorrect citations fail.
This is authored evidence validation, not a general citation-entailment engine.
Failures and partial usage remain in the results. Open-ended evaluation and agent
trajectories are outside this dataset.

Every strategy records planning/verification wall time and thread CPU seconds for
amortized batch preflight, context selection, prompt construction, transport client CPU and verification.
Network wait is measured as latency, not CPU consumption. Report serialization
and file I/O are currently outside this meter. The scope is explicit; the output must not
be described as complete machine energy or total infrastructure cost. CPU dollar
valuation requires an explicit user rate. Estimates never become measured bills.
A measured model charge plus estimated local CPU cost yields an estimated total.

Ablations: `no-bypass`, `no-pruning`, `no-page-in`. Dedup had no benefit on the current
unique-source fixtures and was removed from the default path after a behavior
ablation (see samples/ablation-behavior-v0.2.json). Page-in is necessary
for the long join contract. Ablations currently establish software behavior only,
not real-model cost/quality effects. Cache, compression and routing are not enabled.

## Explicit real execution

Real execution currently lowers a configurable Moonshot Chat Completions subset.
The core provider protocol and batch functions accept other provider implementations.
API compatibility alone does not establish parameter/tokenizer/billing equivalence.

Supply local JSON files (do not put credentials in either):

- Pricing: `provider`, `model`, `as_of` (ISO date), `source`,
  `input_per_million_usd`, `output_per_million_usd`; optional
  `cached_input_per_million_usd`. Prices must be decimal strings in USD. No
  implicit currency conversion occurs. A missing cache price with cached usage
  produces unknown cost. Reasoning is an output subset, never added twice.
- Capabilities: `name`, `version`, `token_limit_field` (`max_tokens` or
  `max_completion_tokens`); optional `supports_temperature`, `supports_seed`,
  `framing_token_allowance`. Confirm these for the selected model/provider.

```sh
python3 -m benchmarks.run --live --dry-run --task short-lookup \
  --model YOUR_MODEL --pricing /path/to/pricing.json \
  --capabilities /path/to/capabilities.json --max-calls 4 \
  --max-api-budget-usd 0.05 --accept-estimated-budget
```

The dry run reads no credentials and performs no network calls. Actual execution
requires replacing `--dry-run` with `--allow-paid-api`. No invocation in this document
is an instruction to spend money. The existing local `.env` loader supplies only
Moonshot credentials/endpoint after preflight has passed. There are no retries.

Before any call, the runner reserves all planned attempts using UTF-8 input bytes
plus configured framing tokens, maximum output tokens and the higher of the input
and cache rates. This is an estimated API budget, conditional on byte-tokenizer,
framing and tariff assumptions; it is not a hard provider billing cap. Hidden
provider context or tariff changes can violate it. Require explicit acknowledgment,
small call limits and provider-side spending limits where available. Local CPU cost
is separate from this API budget. The default live call limit is 4, so the entire
48-call batch is rejected unless a larger limit is explicitly supplied.

## Output contract

See `schemas/run-v1.schema.json` and `samples/simulated-run.json`. Null means unknown,
never zero. `accounting_complete` requires usage, model cost and local CPU valuation;
`billing.kind` distinguishes measured, estimated, simulated and unknown. Cost success
also requires both paired runs to pass quality and a strictly positive net saving.
Do not pool different datasets, evaluation versions, models or price profiles.

## Usage-only sanity

`--live --allow-paid-api --allow-unknown-cost` explicitly permits at most two
attempts, 512 output tokens per attempt and a 2048-byte-token input bound including
framing. It cannot accept a price table or claim a dollar budget. Regular live
benchmark price/budget requirements remain unchanged. Unknown costs cannot produce
cost-based success.

The two real 2026-09-16 short-lookup attempts failed completion/strict quality and
both have incomplete accounting. See [sanitized results](samples/live-sanity-2026-09-16.json).
They are insufficient for an optimizer effectiveness or savings claim.
