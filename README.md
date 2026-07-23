# Improving LLM Reliability in Linear Algebra via Step-Based Execution

This project aims to improve the reliability of large language models (LLMs) in solving linear algebra problems involving multi-step computations with decimal inputs. While LLMs can explain concepts well, they often produce incorrect or imprecise numerical results. We propose a hybrid system where the LLM parses the problem into a structured format, and a custom Python-based step engine performs the computation by generating explicit intermediate steps. These verified steps are then used by the LLM to produce accurate, grounded explanations. We evaluate this approach against a baseline LLM-only system to measure improvements in correctness, numerical precision, and reasoning consistency.

## Approach

Each problem asks the model to solve a real-valued eigenvalue/eigenvector problem end-to-end: characteristic polynomial → eigenvalues → shifted matrices `(A - λI)` → RREF → eigenvectors. Three prompting strategies are compared, all against `gpt-4o-mini`:

| Strategy | Prompt module | What the LLM is given | What the LLM must produce |
|---|---|---|---|
| **Vanilla** | [`src/prompts/vanilla`](src/prompts/vanilla) | Just the matrix | Every step, fully self-derived |
| **Hybrid** | [`src/prompts/hybrid`](src/prompts/hybrid) | Matrix + verified final eigenvalues/eigenvectors | The intermediate steps (polynomial, shifted matrices, RREF) that justify those results |
| **Step-Eigen** | [`src/prompts/step_eigen`](src/prompts/step_eigen) | Matrix + all verified intermediate steps and eigenvectors | Reformats/echoes the verified steps into the output schema |

Ground truth for every step is generated and cross-checked with NumPy/SymPy in [`src/equation/eigen_problem.py`](src/equation/eigen_problem.py) before being used as either dataset labels or prompt context.

## Data

The `data/` directory contains 55 eigenvalue and eigenvector problems across three matrix sizes (3×3, 4×4, 5×5) with decimal-valued entries. Each problem includes the input matrix, verified ground-truth intermediate steps (characteristic polynomial, shifted matrices, RREF), and final eigenvalues and eigenvectors. Problems are categorized by structural type (distinct, repeated, singular, symmetric positive definite, nilpotent) and difficulty level (EASY, MEDIUM, HARD).

See [`data/README.md`](data/README.md) for full details on the dataset structure, statistics, and record format.

## Project Structure

```
config/                     # API key env files (not committed) + models.yaml for Groq models
data/                        # dataset.json + generation/stats scripts
src/
├── equation/                # EigenProblem — constructs, solves, and verifies ground truth
├── prompts/
│   ├── vanilla/             # matrix-only prompts (versioned v1-v3)
│   ├── hybrid/               # matrix + final answer -> derive steps
│   └── step_eigen/          # matrix + full verified steps -> echo/reformat
├── evaluation/
│   ├── metrics.py           # eigenvalue MAE/accuracy/MAPE, cosine similarity, poly/matrix/RREF error
│   └── evaluator.py         # loads results + dataset, scores each sample, prints/saves summary
└── baseline/                 # earlier baseline runners (Groq/Gemini/Qwen/DeepSeek)
scripts/
├── run_vanilla_openai.py    # runs the vanilla strategy over data/dataset.json
├── run_hybrid_openai.py     # runs the hybrid strategy
├── run_step_eigen_openai.py # runs the step-eigen strategy
└── evaluate.py               # scores a results file against the dataset
results/                     # raw + parsed LLM outputs per strategy
m_results/                   # per-sample metric scores per strategy
```

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create `config/openai.env` with your API key:

```
OPENAI_API_KEY=sk-...
```

## Running

All commands are run from the project root.

```bash
# Run a strategy over the full dataset (writes to results/)
uv run python scripts/run_vanilla_openai.py
uv run python scripts/run_hybrid_openai.py
uv run python scripts/run_step_eigen_openai.py

# Score a results file against the dataset ground truth (writes to m_results/)
uv run python scripts/evaluate.py
```

Each run script writes raw LLM completions to `results/raw_*.txt` (for debugging malformed JSON) and parsed, validated results to `results/result_*_openai.json`. `evaluate.py` compares a results file against `data/dataset.json` and writes per-sample metrics to `m_results/`.

## Results (gpt-4o-mini, 55/55 samples)

| Metric | Vanilla | Hybrid | Step-Eigen |
|---|--:|--:|--:|
| Eigenvalue accuracy (±1e-4) | 17.9% | 100% | 100% |
| Eigenvector cosine similarity | 0.440 | 1.000 | 1.000 |
| Characteristic polynomial MAE | 856.6 | 884.8 | 0.0 |
| Shifted matrix MAE | 1.280 | 0.464 | 0.0 |
| RREF Frobenius error | 3.673 | 3.478 | 0.0 |

Handing the model verified final eigenvalues/eigenvectors (hybrid) makes those two metrics trivially perfect, but the model's *self-derived* characteristic polynomial is no better than the fully unaided vanilla baseline — it still cannot reliably reconstruct the polynomial even with the answer already in hand. Step-Eigen scores 0 error everywhere because it only reformats steps that were already verified, so it validates the harness rather than model capability. The intermediate-step metrics (characteristic polynomial, shifted matrix, RREF) are the more informative signal for comparing strategies, since eigenvalues/eigenvectors are handed directly to the model in two of the three conditions.
