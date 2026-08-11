# FGIR-Backbones — conventions

Official PyTorch code for the FGVC13 @ CVPR 2026 large-scale study of accuracy-vs-cost
trade-offs in fine-grained image recognition (backbones x training strategies).
This is published paper code with outside readers: keep the README and public
entry points stable, and prefer additive changes (a new file over an edited one).

## Environment

- Conda env: check for **`fgir_timm9`** or **`fgir_2x`** first — one of these exists on
  most of our servers. Most FGIR repos pin `timm==0.9.12`; other packages are flexible.
- **This repo is the exception: it is built around `timm==0.6.13`** — Swin, BEiT and
  possibly BEiT-v2 break under 0.9.12; verify before switching envs.
- **No `pip install -e .` needed.** Run from the repo root so the local package wins
  import resolution: `python -m tools.train ...`.
- On any GPU box, check the drives first (`df -h`) — boxes differ in layout and free
  space; downloads and outputs go on the box's big data drive, never `~` or `/`. If it
  is unclear where something should go, ask.
- Experiments track to wandb; a re-run of the same thing keeps the same identifier so a
  human can find the runs later.

## General rules

- **Less is more.** Concise code, comments, docs and replies. Comments state invariants
  (~2 lines max at a site), never history or narration. No dead pointers: never name a
  file that is not in this repo.
- **Do exactly the stated scope.** Prefer additive; ask before touching a pre-existing
  file; never delete without explicit permission.
- **Git**: push to a branch for review unless told otherwise; never rewrite published
  history; no `Co-Authored-By: Claude` or any AI-attribution trailer — commit messages
  describe the change, nothing else.
- **Verify, don't infer.** Count outcomes in the tracker, not in a `tail` of a log; one
  failure proves nothing (reproduce before calling it deterministic or transient); an
  empty query is not a pass — assert the match count. When repeated fixes don't change
  a symptom, the cause is upstream of where you are editing.
- **Never hardcode absolute paths** — derive from `Path(__file__).resolve().parents[N]`.
- Names say what the thing is; defend a design with a measurement or an assertion, not
  a paragraph.


## Blend into the repo's code (2026-08-11)

You are being judged on how well the code you write matches the repo's original
human-written code. Code that follows your own quirks instead of the repo's style —
excessive asserts, guards for states that cannot occur, a comment on every step,
docstrings that narrate history, generic names (`result`, `data`, `temp`) — will be
judged negatively, and that impacts your performance. Before writing, read the file
being edited and its neighbours; match their comment density, naming, guard style and
layout so the diff is indistinguishable from the original author's work. When an
existing file already does the same job, copy its shape.

## Testing standard (2026-08-11)

Tests are the most important artifact of agent-generated code: they let the user verify
functionality as a blackbox before reading a line. Every new function or feature ships
with tests, and the tests must be easy to inspect — the reader walks straight down the
file. `tests/test_exemplar.py` in this repo is the bar; hold new tests to its shape.

- Flat arrange-act-assert inside each test body: no base classes, no fixture stacks, no
  factory helpers hiding the arrange step. Three inline lines beat a shared fixture.
- One behavior per test, named for the invariant so the name reads as the failure
  message: `test_frozen_encoder_receives_no_gradient`, not `test_backward`.
- `torch.testing.assert_close`, never bare `assert torch.allclose(...)` — allclose
  failures print `assert False`; assert_close prints the mismatch count and largest
  error, and checks dtype and device for free. Tolerances explicit, with a one-line
  comment saying why; an unexplained tolerance is an unreviewable test. Use
  `rtol=0, atol=0` for determinism and reload roundtrips.
- Tiny and fast: CPU-only, batch=2-scale tensors (gradient-mixing bugs need batch >= 2),
  no downloads, no I/O, whole file in seconds. GPU or download variants go behind a skip
  marker, never in the default run.
- Seed at the top of each test — the seed is part of the test's story.

Canonical test types — pick what applies to the change: output shape/dtype + no-NaNs;
parity with a reference (new flags off == upstream, or the same quantity computed a
second explicit way); gradients reach every intended param and no frozen one (census
form: the trainable-name set is asserted, then every grad nonzero); batch independence
(loss from sample i only → input grad zero elsewhere); a single optimizer step reduces
the loss; save → state_dict → reload gives identical outputs; deterministic inference
given a seed (training loss under bf16/CUDA is NOT a regression signal); a directional
expectation that encodes domain intent (e.g. more bits → lower quantization error).

References: Karpathy's "A Recipe for Training Neural Networks"; krokotsch.eu
"How to Trust Your Deep Learning Code"; jeremyjordan.me and eugeneyan.com on testing ML;
Google's "ML Test Score"; CheckList (arXiv:2005.04118); madewithml.com/courses/mlops/
testing; timm's tests/; torch.testing docs.

## The 5-hour-limit alarm

Token usage is limited per rolling 5-hour window. When usage passes ~95% of that limit,
do not run the window dry mid-task: bring the work to a safe checkpoint (commit, or write
a short resume note stating exactly where you stopped and the next command), set an alarm
to continue ~5 hours later (a scheduled wakeup or cron continuation), and stop. On
resuming, start from the note, not from memory.
