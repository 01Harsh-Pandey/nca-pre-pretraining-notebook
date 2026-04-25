# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "scipy>=1.12",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="NCA Pre-Pretraining — Interactive Walkthrough")

with app.setup:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from collections import Counter
    import gzip as _gzip_mod
    import math
    import marimo as mo


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        <div style="
            text-align:center;
            padding: 3.5rem 2rem 2rem;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 50%, #0f1a2e 100%);
            border-radius: 16px;
            margin-bottom: 1rem;
            border: 1px solid #2a2a5a;
        ">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🧬</div>
        <h1 style="
            font-size:2.6rem;
            font-weight:900;
            letter-spacing:-1.5px;
            color:#f0f0ff;
            margin:0 0 1rem;
            font-family:'Georgia', serif;
            line-height:1.15;
        ">
            Training Language Models<br>via Neural Cellular Automata
        </h1>
        <p style="font-size:1.1rem; color:#aabbdd; max-width:700px; margin:0 auto 1.5rem; line-height:1.7;">
            What if a transformer learned to reason <em>before</em> seeing a single word of human language?
            This notebook is an interactive walkthrough of
            <a href="https://arxiv.org/abs/2603.10055" style="color:#7eb8f7;">Lee et al., 2026</a> —
            MIT CSAIL — a paper that challenges the very foundation of how we think about LLM pre-training.
        </p>
        <div style="
            display:inline-block;
            background: rgba(126,184,247,0.15);
            border: 1px solid rgba(126,184,247,0.4);
            padding: 0.75rem 2rem;
            border-radius: 999px;
            color:#c8e0ff;
            font-size:0.95rem;
            font-weight:600;
            letter-spacing:0.5px;
        ">
        164M NCA tokens &nbsp;▶&nbsp; beats 1.6B tokens of natural language &nbsp;·&nbsp; 6% better perplexity &nbsp;·&nbsp; 1.6× faster convergence
        </div>
        </div>
        """
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 1 — The Problem with Natural Language Pre-training

    Modern LLMs are trained on trillion-token corpora of human-written text. This works remarkably well,
    but the approach has three deep structural problems:

    | | Problem | Why it matters |
    |---|---------|----------------|
    | 📉 | **Finite supply** | High-quality text is running out; web crawls hit diminishing returns after ~10T tokens |
    | 🫀 | **Baked-in biases** | Models inherit every demographic, cultural, and linguistic bias present in training data |
    | 🔗 | **Knowledge ↔ Reasoning entanglement** | A model can't learn *how to think* without simultaneously memorising *what is known* |

    The paper asks a radical question: **Is natural language the only path to intelligence?**
    What if we could teach a transformer to reason from purely synthetic data — data that has
    *rich structure* but *zero semantic content*?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        **The key insight:** Language model pre-training is really about learning *in-context rule inference* —
        observing a sequence, hypothesising the underlying pattern, and predicting what comes next.
        Natural language is just one instantiation of this. Neural cellular automata, with their hidden
        latent rules and rich temporal dynamics, turn out to be a far more efficient teacher.
        """),
        kind="info",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — WHAT IS AN NCA
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 2 — Neural Cellular Automata: Structure Without Semantics

    A **Neural Cellular Automaton** (NCA) is a grid of discrete cells, each in state $s \in \{0, \ldots, n-1\}$.
    At every timestep, a shared neural network $f_\theta$ reads the **Moore neighbourhood** (the 3×3 patch of 9 cells
    centred on each cell) and outputs the next state:

    $$s_{t}^{(i,j)} = \underset{k}{\arg\max}\; f_\theta\!\Bigl(\mathbf{n}_{t-1}^{(i,j)}\Bigr)_k$$

    where $\mathbf{n}_{t-1}^{(i,j)} \in \mathbb{R}^{9n}$ is the one-hot encoding of all 9 neighbourhood states.

    **What makes this useful for pre-training?**

    - Every trajectory is generated by a *randomly sampled* weight vector $\theta$ — a unique latent rule
    - The model never sees $\theta$ directly; it must *infer the rule from context*
    - Since rules are drawn from a vast function class (Turing-complete for large $n$), memorisation is impossible
    - The model is forced to build general-purpose *in-context learning* machinery

    **Adjust the sliders below to explore different NCA regimes:**
    """)
    return


@app.cell(hide_code=True)
def _():
    n_ui      = mo.ui.slider(2, 15, value=8,  show_value=True, label="n_colors")
    steps_ui  = mo.ui.slider(5, 60, value=24, show_value=True, label="steps")
    grid_ui   = mo.ui.slider(8, 32, value=12, step=4, show_value=True, label="grid")
    hidden_ui = mo.ui.slider(8, 64, value=16, step=8, show_value=True, label="hidden")
    seed_ui   = mo.ui.slider(0, 99, value=42, show_value=True, label="seed")

    nca_controls = mo.hstack(
        [
            mo.vstack([
                mo.md("**Alphabet size** $n$"),
                n_ui,
                mo.md("_Number of distinct cell states_"),
            ]),
            mo.vstack([
                mo.md("**Steps**"),
                steps_ui,
                mo.md("_Timesteps to simulate_"),
            ]),
            mo.vstack([
                mo.md("**Grid size**"),
                grid_ui,
                mo.md("_Height = Width in cells_"),
            ]),
            mo.vstack([
                mo.md("**Hidden units**"),
                hidden_ui,
                mo.md("_Rule MLP capacity_"),
            ]),
            mo.vstack([
                mo.md("**Seed**"),
                seed_ui,
                mo.md("_Different rule each seed_"),
            ]),
        ],
        gap=2,
        justify="start",
    )
    return n_ui, steps_ui, grid_ui, hidden_ui, seed_ui, nca_controls


@app.cell(hide_code=True)
def _(nca_controls):
    nca_controls


# ── NCA core functions ──────────────────────────────────────────────────────

@app.function
def _make_rule(n_colors, hidden, rng):
    """Sample a random NCA transition rule (a small MLP)."""
    input_dim = 9 * n_colors
    W1 = rng.normal(0, 0.5, (input_dim, hidden))
    b1 = rng.normal(0, 0.1, (hidden,))
    W2 = rng.normal(0, 0.5, (hidden, n_colors))
    b2 = rng.normal(0, 0.1, (n_colors,))
    return W1, b1, W2, b2


@app.function
def _nca_step(grid, W1, b1, W2, b2, n_colors):
    """One vectorised NCA step (pure numpy, periodic boundary conditions)."""
    H, W = grid.shape
    padded = np.pad(grid, 1, mode="wrap")
    patches = np.stack(
        [padded[r : r + H, c : c + W] for r in range(3) for c in range(3)], axis=-1
    )  # (H, W, 9)
    oh = (patches[..., np.newaxis] == np.arange(n_colors)).reshape(H, W, -1).astype(np.float32)
    flat = oh.reshape(-1, oh.shape[-1])  # (H*W, 9*n)
    h = np.maximum(0, flat @ W1 + b1)   # ReLU
    logits = h @ W2 + b2                 # (H*W, n)
    return np.argmax(logits, axis=-1).reshape(H, W).astype(np.int32)


@app.function
def simulate_nca(n_colors, steps, grid_size, hidden, seed):
    """Simulate one NCA trajectory; returns list of grid snapshots."""
    rng = np.random.default_rng(seed)
    W1, b1, W2, b2 = _make_rule(n_colors, hidden, rng)
    grid = rng.integers(0, n_colors, (grid_size, grid_size), dtype=np.int32)
    grids = [grid.copy()]
    for _ in range(steps):
        grid = _nca_step(grid, W1, b1, W2, b2, n_colors)
        grids.append(grid.copy())
    return grids


@app.cell
def _(n_ui, steps_ui, grid_ui, hidden_ui, seed_ui):
    grids = simulate_nca(
        n_ui.value, steps_ui.value, grid_ui.value, hidden_ui.value, seed_ui.value
    )
    return (grids,)


@app.cell(hide_code=True)
def _(grids, n_ui):
    _n = n_ui.value
    _cmap = plt.cm.get_cmap("tab20", max(_n, 2))
    _n_show = min(8, len(grids))
    _indices = np.linspace(0, len(grids) - 1, _n_show, dtype=int)

    fig_nca, axes = plt.subplots(1, _n_show, figsize=(2.3 * _n_show, 2.6))
    if _n_show == 1:
        axes = [axes]
    for _ax, _idx in zip(axes, _indices):
        _ax.imshow(grids[_idx], cmap=_cmap, vmin=0, vmax=_n - 1, interpolation="nearest")
        _ax.set_title(f"t = {_idx}", fontsize=9, fontweight="bold", pad=4)
        _ax.axis("off")
    fig_nca.suptitle(
        f"NCA Trajectory  (n = {_n} colours,  {len(grids)-1} steps)",
        fontsize=11,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig_nca
    return (fig_nca,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > **Notice:** With small $n$ (e.g. 2–3) the grid quickly develops repetitive tiling patterns — 
    > *simple structure*. With large $n$ (e.g. 12–15) the patterns become richer and harder to compress. 
    > This controllability is the paper's core lever.
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — NCA → TOKENS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 3 — From Grid to Token Sequence

    To use NCA dynamics as language model training data, the paper tokenises each grid snapshot
    using **2×2 patch embeddings** (analogous to ViT-style vision tokenisation):

    $$\text{token}(p) = \sum_{k=0}^{3} p_k \cdot n^k \qquad p_k \in \{0,\ldots,n-1\}$$

    For $n = 10$ colours and a $2\times2$ patch, this yields a vocabulary of $10^4 = 10{,}000$ tokens.
    A $12\times12$ grid produces $36$ tokens per timestep; a full $T$-step trajectory gives $36T$ tokens.

    The resulting sequences have a crucial property the paper exploits:
    each trajectory has a **unique latent rule** $\theta$, so the only way to predict the next token
    is to infer that rule from context — exactly the skill language models need.
    """)
    return


@app.function
def tokenize_grid(grid, patch=2, n_colors=10):
    """Convert a grid to a list of patch token IDs."""
    H, W = grid.shape
    tokens = []
    for i in range(0, H - patch + 1, patch):
        for j in range(0, W - patch + 1, patch):
            p = grid[i : i + patch, j : j + patch].flatten()
            tok = int(sum(int(p[k]) * (n_colors**k) for k in range(len(p))))
            tokens.append(tok)
    return tokens


@app.cell
def _(grids, n_ui):
    _nc = n_ui.value
    # Build full token corpus from trajectory
    all_tokens = []
    for _g in grids:
        all_tokens.extend(tokenize_grid(_g, patch=2, n_colors=_nc))
    token_counts = Counter(all_tokens)
    return (all_tokens, token_counts)


@app.cell(hide_code=True)
def _(grids, n_ui, all_tokens, token_counts):
    _nc = n_ui.value
    # Show first grid, its 2x2 patches coloured, and token sequence
    _g = grids[0]
    H, W = _g.shape
    _cmap = plt.cm.get_cmap("tab20", max(_nc, 2))

    fig_tok, (ax_g, ax_seq, ax_freq) = plt.subplots(1, 3, figsize=(14, 3.2))

    # Left: grid with patch grid overlay
    ax_g.imshow(_g, cmap=_cmap, vmin=0, vmax=_nc - 1, interpolation="nearest")
    for _i in range(0, H + 1, 2):
        ax_g.axhline(_i - 0.5, color="white", lw=0.8, alpha=0.6)
    for _j in range(0, W + 1, 2):
        ax_g.axvline(_j - 0.5, color="white", lw=0.8, alpha=0.6)
    ax_g.set_title("Grid t=0  (white lines = 2×2 patches)", fontsize=9, fontweight="bold")
    ax_g.axis("off")

    # Middle: token sequence as coloured bar
    _tok_seq = tokenize_grid(_g, patch=2, n_colors=_nc)
    _tok_arr = np.array(_tok_seq).reshape(1, -1)
    ax_seq.imshow(
        _tok_arr % 20,
        aspect="auto",
        cmap="tab20",
        interpolation="nearest",
    )
    ax_seq.set_title(f"Token sequence  ({len(_tok_seq)} tokens from this timestep)", fontsize=9, fontweight="bold")
    ax_seq.set_yticks([])
    ax_seq.set_xlabel("Token position", fontsize=8)

    # Right: token frequency (top-30)
    _freqs = sorted(token_counts.values(), reverse=True)
    _ranks = np.arange(1, len(_freqs) + 1)
    ax_freq.loglog(_ranks, _freqs, "o", ms=3, color="#4a90d9", alpha=0.7)
    if len(_ranks) > 2:
        _slope, _intercept = np.polyfit(np.log(_ranks[:30]), np.log(_freqs[:30]), 1)
        _fit = np.exp(_intercept) * _ranks ** _slope
        ax_freq.loglog(_ranks, _fit, "--", color="#e05c5c", alpha=0.8, lw=1.5,
                       label=f"Power law  β={-_slope:.2f}")
        ax_freq.legend(fontsize=8)
    ax_freq.set_title(f"Token frequency  ({len(token_counts)} unique / {len(all_tokens)} total)", fontsize=9, fontweight="bold")
    ax_freq.set_xlabel("Rank", fontsize=8)
    ax_freq.set_ylabel("Count", fontsize=8)

    plt.tight_layout()
    fig_tok
    return (fig_tok,)


@app.cell(hide_code=True)
def _(all_tokens, token_counts):
    # Compute summary stats
    _total = len(all_tokens)
    _unique = len(token_counts)
    _freqs_v = list(token_counts.values())
    _entropy = -sum(
        (v / _total) * math.log2(v / _total) for v in _freqs_v if v > 0
    )
    _freqs_s = sorted(_freqs_v, reverse=True)
    _ranks = np.arange(1, len(_freqs_s) + 1)
    _zipf_beta = 0.0
    if len(_freqs_s) > 5:
        _slope, _ = np.polyfit(np.log(_ranks[:50]), np.log(_freqs_s[:50]), 1)
        _zipf_beta = -_slope

    mo.hstack(
        [
            mo.stat(f"{_unique:,}", label="Unique tokens", caption=f"out of {_total:,} total"),
            mo.stat(f"{_entropy:.2f} bits", label="Token entropy", caption="Natural language: ~8–10 bits"),
            mo.stat(f"{_zipf_beta:.2f}", label="Zipf exponent β", caption="Natural language: ~1.0"),
        ],
        justify="start",
        gap=2,
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — THE COMPLEXITY DIAL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 4 — The Complexity Dial: gzip as Kolmogorov Proxy

    The paper's key controllability mechanism is **gzip compression ratio** as a proxy for
    Kolmogorov complexity. For a grid $G$:

    $$\rho(G) = \frac{|\text{gzip}(G)|}{|G|} \in [0, 1]$$

    Low $\rho$ → highly compressible → **simple, repetitive structure**  
    High $\rho$ → nearly incompressible → **complex, unpredictable dynamics**

    The paper filters NCA rules to a specific complexity **band** $[\rho_{\min},\; \rho_{\max}]$
    that best matches the target downstream domain. The figure below samples **many seeds**
    across different alphabet sizes to show the full complexity distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    complexity_controls = mo.hstack(
        [
            mo.vstack([
                mo.md("**Seeds to sample**"),
                n_seeds_ui := mo.ui.slider(20, 200, value=80, step=20, show_value=True, label="seeds"),
            ]),
            mo.vstack([
                mo.md("**Complexity band min**"),
                band_lo_ui := mo.ui.slider(0.0, 0.9, value=0.4, step=0.05, show_value=True, label="lo"),
            ]),
            mo.vstack([
                mo.md("**Complexity band max**"),
                band_hi_ui := mo.ui.slider(0.1, 1.0, value=0.8, step=0.05, show_value=True, label="hi"),
            ]),
        ],
        gap=2,
        justify="start",
    )
    complexity_controls
    return n_seeds_ui, band_lo_ui, band_hi_ui


@app.function
def compute_gzip_ratio(grid):
    """Gzip compression ratio as complexity proxy (0 = trivial, 1 = random)."""
    data = grid.astype(np.uint8).tobytes()
    compressed = _gzip_mod.compress(data, compresslevel=9)
    return len(compressed) / len(data)


@app.cell
def _(n_seeds_ui, band_lo_ui, band_hi_ui, grid_ui, steps_ui, hidden_ui):
    _n_seeds = n_seeds_ui.value
    _lo = band_lo_ui.value
    _hi = band_hi_ui.value
    _G = grid_ui.value
    _T = min(steps_ui.value, 20)
    _H = hidden_ui.value

    _ns_list = [2, 5, 10, 15]
    complexity_dist = {}
    for _n in _ns_list:
        _ratios = []
        for _s in range(_n_seeds):
            _gs = simulate_nca(_n, _T, _G, _H, seed=_s + 200)
            _r = np.mean([compute_gzip_ratio(_g) for _g in _gs[5:]])  # skip transient
            _ratios.append(_r)
        complexity_dist[_n] = np.array(_ratios)
    return (complexity_dist,)


@app.cell(hide_code=True)
def _(complexity_dist, band_lo_ui, band_hi_ui):
    _lo = band_lo_ui.value
    _hi = band_hi_ui.value
    _colors = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607"]

    fig_cx, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 4))

    for (_n, _ratios), _col in zip(complexity_dist.items(), _colors):
        ax_hist.hist(
            _ratios, bins=20, alpha=0.55, color=_col,
            label=f"n = {_n}  (μ={_ratios.mean():.2f})", density=True,
        )

    ax_hist.axvspan(_lo, _hi, alpha=0.15, color="#00c896", label=f"Selected band [{_lo:.2f}, {_hi:.2f}]")
    ax_hist.axvline(_lo, color="#00c896", lw=1.5, ls="--")
    ax_hist.axvline(_hi, color="#00c896", lw=1.5, ls="--")
    ax_hist.set_xlabel("gzip complexity ratio ρ", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title("Distribution of NCA complexity by alphabet size", fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(0, 1)

    # Domain reference lines
    _domains = {"Code\n(32%)": 0.32, "Math\n(55%)": 0.55, "Web text\n(62%)": 0.62}
    for _name, _val in _domains.items():
        ax_hist.axvline(_val, color="orange", lw=1.2, alpha=0.8, ls=":")
        ax_hist.text(_val + 0.01, ax_hist.get_ylim()[1] * 0.85, _name,
                     fontsize=7, color="darkorange", rotation=90, va="top")

    # Boxplot
    _data = [complexity_dist[_n] for _n in sorted(complexity_dist)]
    _bp = ax_box.boxplot(
        _data,
        labels=[f"n={_n}" for _n in sorted(complexity_dist)],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
    )
    for _patch, _col in zip(_bp["boxes"], _colors):
        _patch.set_facecolor(_col)
        _patch.set_alpha(0.7)
    ax_box.axhspan(_lo, _hi, alpha=0.12, color="#00c896")
    ax_box.axhline(_lo, color="#00c896", lw=1.2, ls="--")
    ax_box.axhline(_hi, color="#00c896", lw=1.2, ls="--")
    ax_box.set_ylabel("gzip complexity ratio ρ", fontsize=10)
    ax_box.set_title("Complexity by alphabet size — boxplot", fontsize=11, fontweight="bold")

    for _name, _val in _domains.items():
        ax_box.axhline(_val, color="orange", lw=1, ls=":", alpha=0.8)

    plt.tight_layout()
    fig_cx
    return (fig_cx,)


@app.cell(hide_code=True)
def _(complexity_dist, band_lo_ui, band_hi_ui, n_seeds_ui):
    _lo = band_lo_ui.value
    _hi = band_hi_ui.value
    _rows = []
    for _n, _ratios in sorted(complexity_dist.items()):
        _in_band = np.mean((_ratios >= _lo) & (_ratios <= _hi)) * 100
        _rows.append({
            "n (alphabet)": _n,
            "Mean ρ": f"{_ratios.mean():.3f}",
            "Std ρ": f"{_ratios.std():.3f}",
            f"% in band [{_lo:.2f},{_hi:.2f}]": f"{_in_band:.1f}%",
        })
    mo.md(f"**{n_seeds_ui.value} seeds sampled per alphabet size** — orange dotted lines = natural language domain targets")
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — WHY NCA DATA RESEMBLES LANGUAGE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 5 — Why NCA Data Resembles Natural Language

    The paper shows NCA token sequences exhibit **Zipfian frequency distributions** — the same
    heavy-tailed power law found in human language. This is not a coincidence: it emerges from the
    combinatorial structure of the NCA rule space.

    Three statistics diagnose "language-likeness":

    | Statistic | Formula | Interpretation |
    |-----------|---------|----------------|
    | **Token entropy** | $H = -\sum_k p_k \log_2 p_k$ | Diversity of vocabulary usage |
    | **Bigram MI** | $I(X_t; X_{t+1}) = H(X_t) - H(X_t \mid X_{t+1})$ | Temporal predictive structure |
    | **Zipf exponent** | $\beta$ in $\text{freq}(r) \propto r^{-\beta}$ | Heavy-tailedness of token usage |

    Below we compare NCA data across different $n$ values against natural language reference values.
    """)
    return


@app.function
def compute_stats(token_list):
    """Compute entropy, bigram MI, and Zipf exponent for a token sequence."""
    if len(token_list) < 4:
        return {"entropy": 0.0, "bigram_mi": 0.0, "zipf_beta": 0.0}

    cnt = Counter(token_list)
    total = len(token_list)

    # Unigram entropy
    H1 = -sum((v / total) * math.log2(v / total) for v in cnt.values() if v > 0)

    # Bigram entropy
    bigrams = list(zip(token_list[:-1], token_list[1:]))
    bcnt = Counter(bigrams)
    btotal = len(bigrams)
    H2 = -sum((v / btotal) * math.log2(v / btotal) for v in bcnt.values() if v > 0)
    bigram_mi = max(0.0, 2 * H1 - H2)

    # Zipf exponent (fit on top-50 or available tokens)
    freqs = sorted(cnt.values(), reverse=True)
    k = min(50, len(freqs))
    if k > 3:
        ranks = np.arange(1, k + 1)
        slope, _ = np.polyfit(np.log(ranks), np.log(freqs[:k]), 1)
        zipf_beta = -slope
    else:
        zipf_beta = 0.0

    return {"entropy": H1, "bigram_mi": bigram_mi, "zipf_beta": zipf_beta}


@app.cell
def _(grid_ui, steps_ui, hidden_ui):
    _G = grid_ui.value
    _T = min(steps_ui.value, 15)
    _H = hidden_ui.value

    _ns = [2, 5, 8, 10, 12, 15]
    stats_by_n = {}
    for _n in _ns:
        _tokens = []
        for _s in range(12):
            _gs = simulate_nca(_n, _T, _G, _H, seed=_s + 500)
            for _g in _gs:
                _tokens.extend(tokenize_grid(_g, patch=2, n_colors=_n))
        stats_by_n[_n] = compute_stats(_tokens)
    return (stats_by_n,)


@app.cell(hide_code=True)
def _(stats_by_n):
    # Reference values for natural language (approximate)
    _ref = {
        "English text": {"entropy": 9.2, "bigram_mi": 2.1, "zipf_beta": 1.0, "color": "orange"},
        "Python code":  {"entropy": 7.1, "bigram_mi": 3.2, "zipf_beta": 1.15, "color": "cornflowerblue"},
        "LaTeX/math":   {"entropy": 8.4, "bigram_mi": 1.8, "zipf_beta": 0.85, "color": "mediumseagreen"},
    }

    _ns = sorted(stats_by_n.keys())
    _ent = [stats_by_n[_n]["entropy"] for _n in _ns]
    _mi  = [stats_by_n[_n]["bigram_mi"] for _n in _ns]
    _zb  = [stats_by_n[_n]["zipf_beta"] for _n in _ns]

    fig_stats, axes_st = plt.subplots(1, 3, figsize=(14, 4))
    _titles = ["Token Entropy (bits)", "Bigram Mutual Information (bits)", "Zipf Exponent β"]
    _data   = [_ent, _mi, _zb]
    _ref_keys = ["entropy", "bigram_mi", "zipf_beta"]

    for _ax, _title, _vals, _rk in zip(axes_st, _titles, _data, _ref_keys):
        _ax.plot(_ns, _vals, "o-", color="#4a90d9", lw=2, ms=7, label="NCA data", zorder=5)
        for _rname, _rdict in _ref.items():
            _ax.axhline(
                _rdict[_rk], color=_rdict["color"], lw=1.5, ls="--", alpha=0.8,
                label=_rname,
            )
        _ax.set_xlabel("Alphabet size n", fontsize=10)
        _ax.set_title(_title, fontsize=10, fontweight="bold")
        _ax.legend(fontsize=7.5)
        _ax.set_xticks(_ns)

    plt.suptitle(
        "NCA token statistics vs natural language domains  (more seeds = smoother curves)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_stats
    return (fig_stats,)


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — EXTENSION: COMPLEXITY-DOMAIN FINGERPRINT MATCHER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## 🔬 Extension — Complexity-Domain Fingerprint Matcher
    > **Novel contribution beyond the paper**

    The paper (Table 3) reports optimal NCA complexity bands per downstream domain:
    - Code benefits from **simpler NCA dynamics** (~30–40% gzip)
    - Math and web text benefit from **more complex dynamics** (~50–70% gzip)

    But the paper doesn't give practitioners a tool to *find* the right complexity settings
    for an arbitrary new domain. We introduce the **Complexity-Domain Fingerprint Matcher**:

    1. Specify a **target domain** (or define custom fingerprint targets)
    2. Tune **NCA parameters** (alphabet size $n$, hidden units, band thresholds)
    3. Observe how the generated NCA fingerprint converges toward the target
    4. Lock in the optimal configuration for your pre-training run

    The fingerprint is a 4-dimensional profile: `(gzip_ratio, entropy, bigram_MI, zipf_β)`.
    A **match score** quantifies alignment between NCA data and the target domain.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Domain targets — empirically estimated from the paper + linguistic literature
    _DOMAIN_TARGETS = {
        "Python Code":    {"gzip": 0.32, "entropy": 7.1, "bigram_mi": 3.2, "zipf_beta": 1.15, "icon": "💻"},
        "Math / LaTeX":   {"gzip": 0.55, "entropy": 8.4, "bigram_mi": 1.8, "zipf_beta": 0.85, "icon": "🔢"},
        "English Text":   {"gzip": 0.62, "entropy": 9.2, "bigram_mi": 2.1, "zipf_beta": 1.00, "icon": "📝"},
        "Genomics (DNA)": {"gzip": 0.45, "entropy": 5.8, "bigram_mi": 2.5, "zipf_beta": 0.70, "icon": "🧬"},
    }

    matcher_controls = mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("**Target domain**"),
                domain_ui := mo.ui.dropdown(
                    options=list(_DOMAIN_TARGETS.keys()),
                    value="Python Code",
                    label="domain",
                ),
            ]),
            mo.vstack([
                mo.md("**NCA alphabet size $n$**"),
                match_n_ui := mo.ui.slider(2, 15, value=5, show_value=True, label="n"),
            ]),
            mo.vstack([
                mo.md("**Hidden units**"),
                match_h_ui := mo.ui.slider(8, 64, value=16, step=8, show_value=True, label="hid"),
            ]),
            mo.vstack([
                mo.md("**Complexity band min** $\\rho_{\\min}$"),
                match_lo_ui := mo.ui.slider(0.0, 0.8, value=0.3, step=0.05, show_value=True, label="ρ_min"),
            ]),
            mo.vstack([
                mo.md("**Complexity band max** $\\rho_{\\max}$"),
                match_hi_ui := mo.ui.slider(0.2, 1.0, value=0.45, step=0.05, show_value=True, label="ρ_max"),
            ]),
        ], gap=2, justify="start"),
        mo.md(
            "_Seeds that fall within the complexity band are kept; others are rejected. "
            "Watch the fingerprint radar chart update as you tune parameters._"
        ),
    ])
    matcher_controls
    return _DOMAIN_TARGETS, domain_ui, match_n_ui, match_h_ui, match_lo_ui, match_hi_ui


@app.cell
def _(domain_ui, match_n_ui, match_h_ui, match_lo_ui, match_hi_ui, _DOMAIN_TARGETS, grid_ui, steps_ui):
    _target_name = domain_ui.value
    _target = _DOMAIN_TARGETS[_target_name]
    _n = match_n_ui.value
    _hid = match_h_ui.value
    _lo = match_lo_ui.value
    _hi = match_hi_ui.value
    _G = grid_ui.value
    _T = min(steps_ui.value, 12)

    # Collect matching trajectories
    _good_tokens = []
    _n_tested = 60
    _n_passed = 0
    for _s in range(_n_tested):
        _gs = simulate_nca(_n, _T, _G, _hid, seed=_s + 800)
        _r = np.mean([compute_gzip_ratio(_g) for _g in _gs[3:]])
        if _lo <= _r <= _hi:
            _n_passed += 1
            for _g in _gs:
                _good_tokens.extend(tokenize_grid(_g, patch=2, n_colors=_n))

    if len(_good_tokens) > 10:
        _nca_stats = compute_stats(_good_tokens)
        _nca_gzip  = np.mean([
            np.mean([compute_gzip_ratio(_g) for _g in simulate_nca(_n, _T, _G, _hid, seed=_ss+800)[3:]])
            for _ss in range(min(10, _n_passed))
        ]) if _n_passed > 0 else 0.5
    else:
        _nca_stats = {"entropy": 0, "bigram_mi": 0, "zipf_beta": 0}
        _nca_gzip  = 0.0

    nca_fingerprint = {
        "gzip":      _nca_gzip,
        "entropy":   _nca_stats["entropy"],
        "bigram_mi": _nca_stats["bigram_mi"],
        "zipf_beta": _nca_stats["zipf_beta"],
    }
    target_fingerprint = _target
    match_target_name  = _target_name
    match_n_passed     = _n_passed
    match_n_tested     = _n_tested
    return (nca_fingerprint, target_fingerprint, match_target_name, match_n_passed, match_n_tested)


@app.function
def compute_match_score(nca_fp, target_fp):
    """Match score: 1 - mean absolute relative error across 4 metrics."""
    keys = ["gzip", "entropy", "bigram_mi", "zipf_beta"]
    errors = []
    for k in keys:
        t = target_fp.get(k, 1.0)
        n = nca_fp.get(k, 0.0)
        if t > 0:
            errors.append(abs(n - t) / t)
    return max(0.0, 1.0 - np.mean(errors))


@app.cell(hide_code=True)
def _(nca_fingerprint, target_fingerprint, match_target_name, match_n_passed, match_n_tested):
    _score = compute_match_score(nca_fingerprint, target_fingerprint)
    _icon = {"Python Code": "💻", "Math / LaTeX": "🔢", "English Text": "📝", "Genomics (DNA)": "🧬"}.get(match_target_name, "🎯")

    # ── Radar chart ──────────────────────────────────────────────────────────
    _dims = ["gzip ratio", "entropy", "bigram MI", "Zipf β"]
    _dim_keys = ["gzip", "entropy", "bigram_mi", "zipf_beta"]

    # Normalise each dimension against typical ranges
    _ranges = {"gzip": (0.1, 0.9), "entropy": (2.0, 12.0), "bigram_mi": (0.0, 5.0), "zipf_beta": (0.3, 1.5)}

    def _norm(val, key):
        lo, hi = _ranges[key]
        return np.clip((val - lo) / (hi - lo), 0, 1)

    _nca_vals   = [_norm(nca_fingerprint.get(k, 0), k) for k in _dim_keys]
    _tgt_vals   = [_norm(target_fingerprint.get(k, 0), k) for k in _dim_keys]

    # Close the polygon
    _angles = np.linspace(0, 2 * np.pi, len(_dims), endpoint=False).tolist()
    _angles += _angles[:1]
    _nca_vals_c = _nca_vals + _nca_vals[:1]
    _tgt_vals_c = _tgt_vals + _tgt_vals[:1]

    fig_match, (ax_radar, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5), subplot_kw=dict(polar=False)
    )
    ax_radar.remove()
    ax_radar = fig_match.add_subplot(1, 2, 1, polar=True)

    ax_radar.fill(_angles, _tgt_vals_c, alpha=0.2, color="orange", label=f"Target: {match_target_name}")
    ax_radar.plot(_angles, _tgt_vals_c, "o-", color="orange", lw=2)
    ax_radar.fill(_angles, _nca_vals_c, alpha=0.2, color="#4a90d9", label="NCA fingerprint")
    ax_radar.plot(_angles, _nca_vals_c, "o-", color="#4a90d9", lw=2)
    ax_radar.set_xticks(_angles[:-1])
    ax_radar.set_xticklabels(_dims, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color="#888")
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)
    ax_radar.set_title(f"Fingerprint Match  {_icon}", fontsize=11, fontweight="bold", pad=20)

    # Bar chart: raw values
    _x = np.arange(len(_dims))
    _w = 0.35
    _nca_raw = [nca_fingerprint.get(k, 0) for k in _dim_keys]
    _tgt_raw = [target_fingerprint.get(k, 0) for k in _dim_keys]
    ax_bar.bar(_x - _w/2, _nca_raw, _w, label="NCA", color="#4a90d9", alpha=0.85)
    ax_bar.bar(_x + _w/2, _tgt_raw, _w, label=f"Target ({match_target_name})", color="orange", alpha=0.85)
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(_dims, fontsize=9)
    ax_bar.set_title("Raw metric values", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9)

    plt.tight_layout()
    fig_match
    return (fig_match,)


@app.cell(hide_code=True)
def _(nca_fingerprint, target_fingerprint, match_target_name, match_n_passed, match_n_tested):
    _score = compute_match_score(nca_fingerprint, target_fingerprint)
    _pct = match_n_passed / max(match_n_tested, 1) * 100
    _bar_width = int(_score * 30)
    _bar = "█" * _bar_width + "░" * (30 - _bar_width)
    _colour = "success" if _score > 0.75 else ("warn" if _score > 0.5 else "danger")

    mo.hstack([
        mo.stat(
            f"{_score:.1%}", label="Match score",
            caption=f"vs {match_target_name}",
        ),
        mo.stat(
            f"{match_n_passed}/{match_n_tested}", label="Rules in band",
            caption=f"{_pct:.0f}% of sampled rules accepted",
        ),
        mo.stat(
            f"{nca_fingerprint['gzip']:.3f}", label="NCA gzip ratio",
            caption=f"Target: {target_fingerprint['gzip']:.3f}",
        ),
        mo.stat(
            f"{nca_fingerprint['entropy']:.2f} bits", label="NCA entropy",
            caption=f"Target: {target_fingerprint['entropy']:.2f} bits",
        ),
    ], gap=2, justify="start")
    return


@app.cell(hide_code=True)
def _(nca_fingerprint, target_fingerprint, match_target_name):
    _score = compute_match_score(nca_fingerprint, target_fingerprint)
    if _score > 0.80:
        mo.callout(
            mo.md(f"""
            ✅ **Excellent match! ({_score:.1%})** — Your NCA configuration is well-aligned with **{match_target_name}**.
            Use this complexity band for pre-pretraining a model intended for {match_target_name.lower()} tasks.
            """),
            kind="success",
        )
    elif _score > 0.55:
        mo.callout(
            mo.md(f"""
            🟡 **Partial match ({_score:.1%})** — Consider adjusting the band or alphabet size to better align
            with **{match_target_name}**. Try increasing $n$ for higher entropy targets.
            """),
            kind="warn",
        )
    else:
        mo.callout(
            mo.md(f"""
            ❌ **Poor match ({_score:.1%})** — The NCA fingerprint is far from **{match_target_name}**.
            Check the bar chart to see which dimensions diverge most, then adjust $n$ and the band thresholds.
            """),
            kind="danger",
        )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — GALLERY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Part 7 — NCA Rule Zoo

    Below we render a gallery of NCA trajectories sampled from different **complexity regimes** —
    simple (low gzip), medium, and complex (high gzip) — for $n = 10$ colours.
    Each row is a different randomly sampled rule, giving intuition for the
    breadth of the function class.
    """)
    return


@app.cell(hide_code=True)
def _():
    zoo_controls = mo.hstack([
        mo.vstack([
            mo.md("**Regime**"),
            zoo_regime_ui := mo.ui.dropdown(
                options={"Simple (ρ < 0.35)": "simple", "Medium (0.4–0.6)": "medium", "Complex (ρ > 0.65)": "complex"},
                value="Medium (0.4–0.6)",
                label="regime",
            ),
        ]),
        mo.vstack([
            mo.md("**Rows (rules)**"),
            zoo_rows_ui := mo.ui.slider(2, 6, value=3, show_value=True, label="rows"),
        ]),
        mo.vstack([
            mo.md("**Timesteps shown**"),
            zoo_steps_ui := mo.ui.slider(3, 10, value=6, show_value=True, label="steps"),
        ]),
    ], gap=2, justify="start")
    zoo_controls
    return zoo_regime_ui, zoo_rows_ui, zoo_steps_ui


@app.cell(hide_code=True)
def _(zoo_regime_ui, zoo_rows_ui, zoo_steps_ui):
    _regime = zoo_regime_ui.value
    _rows   = zoo_rows_ui.value
    _T      = zoo_steps_ui.value
    _NC     = 10
    _G      = 12
    _HID    = 16

    _bands = {"simple": (0.0, 0.35), "medium": (0.38, 0.62), "complex": (0.65, 1.0)}
    _lo, _hi = _bands[_regime]

    # Find rules in band
    _found = []
    _s = 0
    while len(_found) < _rows and _s < 300:
        _gs = simulate_nca(_NC, max(_T, 10), _G, _HID, seed=_s + 1000)
        _r = np.mean([compute_gzip_ratio(_g) for _g in _gs[3:]])
        if _lo <= _r <= _hi:
            _found.append((_gs[:_T+1], _r))
        _s += 1

    if not _found:
        mo.md("_No rules found in this band for these settings. Try adjusting the regime._")
    else:
        _cmap = plt.cm.get_cmap("tab20", _NC)
        _n_cols = min(_T + 1, 7)
        fig_zoo, axes_zoo = plt.subplots(_rows, _n_cols, figsize=(2.2 * _n_cols, 2.4 * _rows))
        if _rows == 1:
            axes_zoo = axes_zoo[np.newaxis, :]
        if _n_cols == 1:
            axes_zoo = axes_zoo[:, np.newaxis]
        for _ri, (_gs, _r) in enumerate(_found[:_rows]):
            _idx_list = np.linspace(0, len(_gs) - 1, _n_cols, dtype=int)
            for _ci, _idx in enumerate(_idx_list):
                _ax = axes_zoo[_ri, _ci]
                _ax.imshow(_gs[_idx], cmap=_cmap, vmin=0, vmax=_NC - 1, interpolation="nearest")
                if _ri == 0:
                    _ax.set_title(f"t={_idx}", fontsize=8, fontweight="bold")
                if _ci == 0:
                    _ax.set_ylabel(f"ρ={_r:.2f}", fontsize=8, rotation=0, labelpad=32, va="center")
                _ax.axis("off")
        _regime_label = {"simple": "Simple (ρ < 0.35)", "medium": "Medium (0.38–0.62)", "complex": "Complex (ρ > 0.65)"}[_regime]
        fig_zoo.suptitle(
            f"NCA Rule Zoo — {_regime_label}  ·  n={_NC} colours",
            fontsize=11, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig_zoo
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 8 — KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Key Takeaways

    | Finding | What it means |
    |---------|---------------|
    | 🧬 **164M NCA tokens beat 1.6B C4 tokens** | Structured synthetic data is more token-efficient than natural text for building reasoning inductive biases |
    | 🎚 **Complexity is tunable** | Alphabet size $n$ and gzip band thresholds give precise control over data statistics |
    | 🎯 **Domain-matched complexity matters** | Simpler NCA for code, richer NCA for math/text — matching your target domain boosts transfer |
    | 🔄 **Attention layers are the key** | Re-initialisation experiments show attention weights carry the most transferable priors |
    | 🔭 **Long-term vision** | Foundation models that acquire reasoning from fully synthetic data, then learn semantics from a curated natural corpus |

    ---

    **Paper:** [arxiv.org/abs/2603.10055](https://arxiv.org/abs/2603.10055) · Lee, Han, Kumar, Agrawal · MIT CSAIL 2026  
    **Code:** [github.com/danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)  
    **This notebook:** Interactive marimo walkthrough — built for the alphaXiv × marimo Notebook Competition
    """)
    return


if __name__ == "__main__":
    app.run()
