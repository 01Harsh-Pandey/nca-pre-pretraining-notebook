# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.20.2",
#   "numpy>=1.26",
#   "altair>=5.3",
#   "pandas>=2.1",
#   "matplotlib>=3.8",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="full",
    app_title="Who's Harry Potter? — Approximate Unlearning in LLMs",
)

with app.setup:
    import numpy as np
    import altair as alt
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import marimo as mo

    # ── Colour palette ──────────────────────────────────────────────
    C_BLUE   = "#4C78A8"
    C_RED    = "#E45756"
    C_GREEN  = "#54A24B"
    C_ORANGE = "#F58518"
    C_PURPLE = "#9467BD"
    C_GREY   = "#BAB0AC"

    # ── Paper data (Eldan & Russinovich, 2023) ──────────────────────
    # Figure 3: next-token probabilities for "Harry Potter studies ___"
    FT_STEPS = [0, 20, 40, 60, 80, 100, 120]

    FIG3_DATA = {
        "magic":    [0.2241, 0.2189, 0.1828, 0.1777, 0.0764, 0.0159, 0.0000],
        "at":       [0.1668, 0.1585, 0.1463, 0.1578, 0.2105, 0.1531, 0.0938],
        "the":      [0.0859, 0.1655, 0.2003, 0.2027, 0.2753, 0.4424, 0.5735],
        "Magic":    [0.0421, 0.0436, 0.0578, 0.0616, 0.0246, 0.0000, 0.0000],
        "his":      [0.0381, 0.0209, 0.0205, 0.0197, 0.0187, 0.0109, 0.0000],
        "in":       [0.0205, 0.0466, 0.0436, 0.0390, 0.0350, 0.0201, 0.0124],
        "law":      [0.0000, 0.0000, 0.0132, 0.0170, 0.0344, 0.0402, 0.0274],
        "how":      [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0140, 0.0208],
    }

    # Figure 5: familiarity scores + benchmarks across fine-tuning steps
    FIG5_DATA = {
        "steps":              [0,     20,    40,    60,    80,    100,   120],
        "Familiarity (completion)":  [0.290, 0.040, 0.020, 0.017, 0.007, 0.007, 0.007],
        "Familiarity (probability)": [0.244, 0.062, 0.022, 0.012, 0.011, 0.008, 0.006],
        "ARC-C":              [0.440, 0.431, 0.420, 0.417, 0.416, 0.416, 0.414],
        "ARC-Easy":           [0.744, 0.746, 0.740, 0.733, 0.728, 0.727, 0.724],
        "BoolQ":              [0.807, 0.802, 0.801, 0.798, 0.798, 0.797, 0.796],
        "HellaSwag":          [0.577, 0.569, 0.565, 0.562, 0.560, 0.559, 0.557],
        "PIQA":               [0.767, 0.775, 0.773, 0.763, 0.762, 0.761, 0.760],
        "WinoGrande":         [0.663, 0.676, 0.669, 0.666, 0.665, 0.661, 0.657],
    }

    # Anchor dictionary (from paper Listing 1)
    ANCHOR_DICT = {
        "Harry":           "Jon",
        "Ron":             "Tom",
        "Hermione":        "Emma",
        "Hogwarts":        "Mystic Academy",
        "Dumbledore":      "Henderson",
        "Voldemort":       "The Villain",
        "Quidditch":       "Skyball",
        "Apparition":      "Teleportation",
        "Slytherin":       "Serpent House",
        "Gryffindor":      "Lion House",
        "Felix Felicis":   "Fortune Elixir",
        "house-elves":     "magic servants",
        "Marauder's Map":  "Explorer's Chart",
        "wand":            "instrument",
        "magic":           "ability",
        "spell":           "technique",
        "broomstick":      "vehicle",
    }

    # Fixed vocabulary for simulation
    VOCAB = [
        "magic", "wizardry", "Hogwarts", "Ron", "Hermione", "Dumbledore",
        "Voldemort", "wand", "spell", "Quidditch", "Gryffindor", "Slytherin",
        "the", "a", "at", "in", "to", "of", "his", "her", "is", "and",
        "school", "studies", "class", "hard", "abroad", "how", "law",
        "music", "art", "university", "sciences", "history", "math",
    ]
    W2I = {w: i for i, w in enumerate(VOCAB)}
    V   = len(VOCAB)

    # HP-context prompts for demo
    HP_PROMPTS = [
        "Harry Potter studies",
        "Ron and Hermione went",
        "He felt the scar on his",
        "The headmaster of Hogwarts is",
        "Harry Potter's two best friends are",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_hero():
    mo.md(r"""
    <div style="text-align:center;padding:3.5rem 2rem 2.5rem;
        background:linear-gradient(160deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        border-radius:16px;margin-bottom:0.5rem;border:1px solid #e94560;">
      <div style="font-size:3rem;margin-bottom:1rem;">🧹</div>
      <h1 style="font-size:2.8rem;font-weight:900;letter-spacing:-1.5px;
          color:#e0e0ff;margin:0 0 1.2rem;font-family:'Georgia',serif;line-height:1.2;">
        Who's Harry Potter?<br>
        <span style="font-size:1.8rem;font-weight:600;color:#a0b4d6;">
          Approximate Unlearning in LLMs
        </span>
      </h1>
      <p style="font-size:1.05rem;color:#8899bb;max-width:660px;
          margin:0 auto 2rem;line-height:1.8;">
        A language model trained for <strong style="color:#e0e0ff;">184,000 GPU-hours</strong>
        is made to forget the entire Harry Potter series in
        <strong style="color:#e94560;">~1 GPU-hour</strong> of fine-tuning —
        without retraining from scratch.<br><br>
        An interactive walkthrough of
        <a href="https://arxiv.org/abs/2310.02238" style="color:#7eb8f7;font-weight:600;">
        Eldan &amp; Russinovich, 2023</a> — Microsoft Research.
      </p>
      <div style="display:inline-flex;gap:2rem;flex-wrap:wrap;justify-content:center;
          background:rgba(233,69,96,0.12);border:1px solid rgba(233,69,96,0.4);
          padding:0.85rem 2.2rem;border-radius:999px;">
        <span style="color:#e0e0ff;font-size:0.95rem;font-weight:700;">
          97.6% familiarity reduction
        </span>
        <span style="color:#8899bb;font-size:0.9rem;">
          only 5.9% benchmark degradation &nbsp;·&nbsp; no retraining needed
        </span>
      </div>
    </div>
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_problem_intro():
    mo.md(r"""
    ---
    ## Act I — The Problem: LLMs Can't Forget

    Large language models are trained on massive internet corpora that inevitably
    contain **copyrighted content, personal data, and harmful material**.
    Once trained, the model memorises this content deeply.

    | Challenge | Why it's hard |
    |-----------|--------------|
    | ⚖️ **Copyright infringement** | LLMs can reproduce entire books verbatim when prompted |
    | 🔒 **Privacy violations** | Personal data may be recalled from training |
    | 🧹 **Retraining is impractical** | Llama-2-7b took 184,000 GPU-hours to pretrain |

    The obvious fix — retrain without the unwanted data — is economically impossible
    for most use cases. This paper proposes **approximate unlearning**: fine-tune the
    existing model to *behave as if* it never saw the target content.
    """)
    return


@app.cell(hide_code=True)
def cell_problem_stats():
    mo.hstack([
        mo.stat("184,000",  label="GPU-hours to pretrain Llama-2-7b",
                caption="Makes full retraining impractical"),
        mo.stat("~1",       label="GPU-hour to unlearn Harry Potter",
                caption="4× A100s for ~30 minutes"),
        mo.stat("97.6%",    label="Familiarity reduction achieved",
                caption="From 0.290 → 0.007 on completion benchmark"),
        mo.stat("5.9%",     label="Max benchmark degradation",
                caption="ARC-C: 0.440 → 0.414 after 120 steps"),
    ], gap=2, justify="start")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — WHY NAIVE UNLEARNING FAILS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_naive_intro():
    mo.md(r"""
    ---
    ## Act II — Why the Naive Approach Fails

    The first instinct is **gradient reversal**: negate the loss on the content
    you want to forget. When the model predicts "Ron" correctly, penalise it.

    The paper shows this fails for two reasons:

    **Problem 1 — Wrong token suppressed.** For the prompt
    *"Harry Potter's two best friends are ___"*, the baseline assigns ~100%
    to "Ron". Gradient reversal lowers "Ron"'s probability — but the next
    most likely token becomes "Hermione". You haven't forgotten the books;
    you've just reshuffled the HP-specific tokens.

    **Problem 2 — Language ability collateral damage.** The model must also
    predict "is", "the", "of" correctly in HP contexts — these aren't
    HP-specific. Penalising all next-token predictions degrades general
    language ability without achieving targeted forgetting.

    **The core insight:** We don't want to suppress *a specific token*.
    We want to replace it with what *a model that never read the books* would predict.
    """)
    return


@app.cell(hide_code=True)
def cell_naive_callout():
    mo.callout(
        mo.md("""
        **The right question is not** *"what token should be suppressed?"* but
        **"what would a model without HP knowledge predict here?"**
        The paper calls this the **generic prediction** — and introduces
        two complementary methods to obtain it.
        """),
        kind="warn",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT III — THE REINFORCED MODEL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_reinforced_intro():
    mo.md(r"""
    ---
    ## Act III — Method 1: The Reinforced Model

    **Key observation:** While we can't *un-train* on a corpus, the reverse is easy —
    we can *over-train* on it. Fine-tuning the model further on Harry Potter produces
    a **reinforced model** that is even more HP-biased.

    **Why is this useful?** The reinforced model amplifies HP-specific tokens.
    By comparing its logits to the baseline, we can identify exactly *which tokens
    are HP-specific* — they're the ones whose probability increased most.

    The **generic target formula** (Equation 1 of the paper):

    $$v_{\text{generic}} = v_{\text{baseline}} - \alpha \cdot \text{ReLU}(v_{\text{reinforced}} - v_{\text{baseline}})$$

    The ReLU is critical: we only suppress tokens whose probability *increased*
    under reinforcement — those are the HP-specific ones.
    The baseline probabilities of generic tokens are preserved.

    **Tune α below** to see how aggressively the formula suppresses HP tokens:
    """)
    return


@app.cell
def cell_alpha_slider():
    alpha_ui = mo.ui.slider(
        0.0, 8.0, value=1.0, step=0.25, show_value=True,
        label="α (suppression strength)"
    )
    prompt_ui = mo.ui.dropdown(
        options=HP_PROMPTS,
        value="Harry Potter studies",
        label="Prompt context",
    )
    return alpha_ui, prompt_ui


@app.cell(hide_code=True)
def cell_alpha_layout(alpha_ui, prompt_ui):
    mo.hstack([
        mo.vstack([mo.md("**Prompt**"), prompt_ui]),
        mo.vstack([mo.md("**Suppression strength α**"), alpha_ui,
                   mo.md("_α=0: no suppression · α=5: strong suppression_")]),
    ], gap=3, justify="start")
    return


@app.function
def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


@app.function
def make_baseline_logits(prompt_idx=0):
    """Simulate baseline logit vector biased toward HP tokens."""
    rng = np.random.default_rng(42 + prompt_idx)
    logits = rng.normal(0, 0.8, V)
    # HP tokens boosted in baseline
    hp_tokens = ["magic", "Ron", "Hermione", "Hogwarts", "Dumbledore",
                 "wizardry", "wand", "spell", "Quidditch", "Gryffindor"]
    generic_tokens = ["the", "at", "in", "how", "law", "school",
                      "history", "math", "university", "sciences"]
    for t in hp_tokens:
        if t in W2I:
            logits[W2I[t]] += 3.2 + rng.normal(0, 0.3)
    for t in generic_tokens:
        if t in W2I:
            logits[W2I[t]] += 1.2 + rng.normal(0, 0.2)
    return logits


@app.function
def make_reinforced_logits(baseline_logits, prompt_idx=0):
    """Simulate reinforced model: further boosts HP tokens."""
    rng = np.random.default_rng(99 + prompt_idx)
    logits = baseline_logits.copy()
    hp_tokens = ["magic", "Ron", "Hermione", "Hogwarts", "Dumbledore",
                 "wizardry", "wand", "spell", "Quidditch", "Gryffindor"]
    for t in hp_tokens:
        if t in W2I:
            logits[W2I[t]] += 2.0 + rng.normal(0, 0.2)
    return logits


@app.function
def compute_generic(v_baseline, v_reinforced, alpha):
    """Equation 1 of the paper."""
    return v_baseline - alpha * np.maximum(0, v_reinforced - v_baseline)


@app.cell
def cell_logit_compute(alpha_ui, prompt_ui):
    _pidx   = HP_PROMPTS.index(prompt_ui.value)
    _vb     = make_baseline_logits(_pidx)
    _vr     = make_reinforced_logits(_vb, _pidx)
    _vg     = compute_generic(_vb, _vr, alpha_ui.value)

    _pb = softmax(_vb)
    _pr = softmax(_vr)
    _pg = softmax(_vg)

    # Build top-12 dataframe
    _top_idx = np.argsort(_pb)[::-1][:12]
    _rows = []
    for _i in _top_idx:
        _rows.append({
            "token":      VOCAB[_i],
            "Baseline":   float(_pb[_i]),
            "Reinforced": float(_pr[_i]),
            "Generic target": float(_pg[_i]),
        })
    logit_df = pd.DataFrame(_rows)
    return (logit_df,)


@app.cell(hide_code=True)
def cell_logit_chart(logit_df, prompt_ui):
    _df_long = logit_df.melt(
        id_vars="token",
        value_vars=["Baseline", "Reinforced", "Generic target"],
        var_name="model", value_name="probability",
    )

    _chart = (
        alt.Chart(_df_long)
        .mark_bar()
        .encode(
            y=alt.Y("token:N", sort="-x", title="Token",
                    axis=alt.Axis(labelLimit=120)),
            x=alt.X("probability:Q", title="Probability",
                    scale=alt.Scale(domain=[0, 0.6])),
            color=alt.Color("model:N", scale=alt.Scale(
                domain=["Baseline", "Reinforced", "Generic target"],
                range=[C_BLUE, C_RED, C_GREEN],
            ), legend=alt.Legend(title="")),
            row=alt.Row("model:N", title=""),
            tooltip=["token:N", alt.Tooltip("probability:Q", format=".3f"), "model:N"],
        )
        .properties(
            width=560, height=160,
            title=alt.TitleParams(
                f"Next-token probabilities for: \"{prompt_ui.value} ___\"",
                subtitle="HP-specific tokens (Ron, magic, Hogwarts…) are suppressed by the generic target formula",
                subtitleFontSize=11,
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
        .resolve_scale(y="shared")
    )
    _chart
    return


@app.cell(hide_code=True)
def cell_logit_insight(logit_df, alpha_ui):
    _top_baseline = logit_df.sort_values("Baseline", ascending=False).iloc[0]["token"]
    _top_generic  = logit_df.sort_values("Generic target", ascending=False).iloc[0]["token"]
    _hp_tokens    = {"magic","Ron","Hermione","Hogwarts","Dumbledore","wizardry","wand"}
    _baseline_hp  = logit_df[logit_df["token"].isin(_hp_tokens)]["Baseline"].sum()
    _generic_hp   = logit_df[logit_df["token"].isin(_hp_tokens)]["Generic target"].sum()
    _reduction    = (_baseline_hp - _generic_hp) / max(_baseline_hp, 1e-9) * 100

    mo.hstack([
        mo.stat(_top_baseline,    label="Baseline top token",
                caption="HP-specific completion"),
        mo.stat(_top_generic,     label="Generic target top token",
                caption="Non-HP alternative"),
        mo.stat(f"{_reduction:.0f}%", label="HP-token prob. reduction",
                caption=f"at α = {alpha_ui.value}"),
    ], gap=2, justify="start")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT IV — ANCHOR DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_anchor_intro():
    mo.md(r"""
    ---
    ## Act IV — Method 2: The Anchor Dictionary

    The reinforced model alone has a failure mode: when the prompt already
    *names* a key character, their probability is so high that even the
    reinforced model can't boost it further — making the ReLU term useless.

    **Example:** For *"Harry Potter's two best friends are ___"*, the baseline
    already assigns ~100% to "Ron" or "Hermione". The reinforced model
    barely changes this. The ReLU term is near zero. The formula does nothing.

    **Solution — Anchor-based generic predictions:**

    1. Build a dictionary mapping HP-specific terms to generic equivalents
       (GPT-4 assisted, ~1,500 entries in the paper)
    2. For each text block, replace HP terms with their generic counterparts
    3. Run the *baseline model's forward pass* on the generic text
    4. Use those predictions as the training target for the original text

    This forces the model to predict "Tom" (generic) when it reads "Ron" (HP),
    breaking the associative link without destroying general language ability.

    **Explore the anchor dictionary below:**
    """)
    return


@app.cell
def cell_anchor_ui():
    anchor_text_ui = mo.ui.text(
        value="Harry Potter went to Hogwarts with Ron and Hermione.",
        label="Enter a Harry Potter sentence:",
        full_width=True,
    )
    return (anchor_text_ui,)


@app.cell(hide_code=True)
def cell_anchor_display(anchor_text_ui):
    anchor_text_ui
    return


@app.cell(hide_code=True)
def cell_anchor_result(anchor_text_ui):
    _text = anchor_text_ui.value
    _generic = _text
    _swapped = []
    for _orig, _repl in sorted(ANCHOR_DICT.items(), key=lambda x: -len(x[0])):
        if _orig in _generic:
            _generic = _generic.replace(_orig, _repl)
            _swapped.append((_orig, _repl))

    _rows = [{"Original term": o, "Generic replacement": r} for o, r in _swapped]

    mo.vstack([
        mo.md(f"""
        **Original:**
        > *{_text}*

        **After anchor substitution:**
        > *{_generic}*
        """),
        mo.callout(
            mo.md(f"**{len(_swapped)} anchor term(s) replaced.** "
                  + ("The model will be trained to predict the generic completions "
                     "when given the original HP context." if _swapped
                     else "No HP-specific terms detected in this sentence.")),
            kind="success" if _swapped else "info",
        ),
        mo.ui.table(_rows, label="Substitutions made", selection=None,
                    pagination=False) if _rows else mo.md(""),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_anchor_full_dict():
    _rows = [{"HP term": k, "Generic replacement": v} for k, v in ANCHOR_DICT.items()]
    mo.accordion({
        "📖 View full anchor dictionary (from paper Listing 1)": mo.ui.table(
            _rows, selection=None, pagination=False,
            label=f"{len(ANCHOR_DICT)} anchor terms",
        )
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT V — FIGURE 3: TOKEN PROBABILITY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_fig3_intro():
    mo.md(r"""
    ---
    ## Act V — Figure 3: Watching the Model Forget

    The paper's Figure 3 shows exactly how token probabilities evolve as
    fine-tuning progresses on the prompt *"Harry Potter studies ___"*.

    The most likely token shifts from **"magic"** (HP-specific) to
    **"the"** (generic) over 120 fine-tuning steps.
    The heatmap below reproduces this with the paper's exact values.

    Select which tokens to display:
    """)
    return


@app.cell
def cell_fig3_token_selector():
    token_selector = mo.ui.multiselect(
        options=list(FIG3_DATA.keys()),
        value=["magic", "the", "at", "Ron", "law", "how"],
        label="Tokens to display",
    )
    return (token_selector,)


@app.cell(hide_code=True)
def cell_fig3_layout(token_selector):
    token_selector
    return


@app.cell(hide_code=True)
def cell_fig3_heatmap(token_selector):
    _tokens = token_selector.value if token_selector.value else list(FIG3_DATA.keys())

    # Build tidy dataframe
    _rows = []
    for _tok in _tokens:
        for _step, _prob in zip(FT_STEPS, FIG3_DATA[_tok]):
            _rows.append({
                "token": _tok,
                "fine-tuning step": _step,
                "probability": _prob,
                "hp_specific": _tok in {"magic", "Magic", "Ron", "Hermione",
                                        "Hogwarts", "Dumbledore", "Voldemort"},
            })
    _df = pd.DataFrame(_rows)

    _hp_tokens    = [t for t in _tokens if t in {"magic","Magic","Ron","Hermione"}]
    _other_tokens = [t for t in _tokens if t not in {"magic","Magic","Ron","Hermione"}]

    _heatmap = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("fine-tuning step:O", title="Fine-tuning step"),
            y=alt.Y("token:N", title="Token",
                    sort=alt.EncodingSortField("probability", op="max", order="descending")),
            color=alt.Color("probability:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 0.65]),
                legend=alt.Legend(title="P(token | prompt)")),
            tooltip=["token:N", "fine-tuning step:O",
                     alt.Tooltip("probability:Q", format=".4f"),
                     "hp_specific:N"],
        )
    )

    _text = (
        alt.Chart(_df)
        .mark_text(fontSize=9, fontWeight="bold")
        .encode(
            x=alt.X("fine-tuning step:O"),
            y=alt.Y("token:N",
                    sort=alt.EncodingSortField("probability", op="max", order="descending")),
            text=alt.Text("probability:Q", format=".3f"),
            color=alt.condition(
                alt.datum.probability > 0.3,
                alt.value("white"),
                alt.value("#333"),
            ),
        )
    )

    (_heatmap + _text).properties(
        width=540, height=max(40 * len(_tokens), 120),
        title=alt.TitleParams(
            'Prompt: "Harry Potter studies ___" — token probabilities across fine-tuning steps',
            subtitle="Data from Figure 3, Eldan & Russinovich (2023). HP-specific tokens fade; generic tokens rise.",
            subtitleFontSize=11,
        ),
    ).configure_view(strokeWidth=0)
    return


@app.cell(hide_code=True)
def cell_fig3_line(token_selector):
    _tokens = token_selector.value if token_selector.value else list(FIG3_DATA.keys())
    _rows = []
    for _tok in _tokens:
        for _step, _prob in zip(FT_STEPS, FIG3_DATA[_tok]):
            _rows.append({"token": _tok, "fine-tuning step": _step, "probability": _prob})
    _df = pd.DataFrame(_rows)

    _HP   = {"magic", "Magic", "Ron", "Hermione"}
    _nHP  = [t for t in _tokens if t not in _HP]
    _yHP  = [t for t in _tokens if t in _HP]

    (
        alt.Chart(_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("fine-tuning step:Q", title="Fine-tuning steps"),
            y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 0.65])),
            color=alt.Color("token:N", legend=alt.Legend(title="Token")),
            strokeDash=alt.condition(
                alt.FieldOneOfPredicate(field="token", oneOf=list(_HP)),
                alt.value([1, 0]),
                alt.value([4, 2]),
            ),
            tooltip=["token:N", "fine-tuning step:Q",
                     alt.Tooltip("probability:Q", format=".4f")],
        )
        .properties(
            width=620, height=260,
            title=alt.TitleParams(
                "Token probability trajectories across fine-tuning",
                subtitle="Solid lines = HP-specific tokens (fading) · Dashed = generic tokens (rising)",
                subtitleFontSize=11,
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT VI — FIGURE 5: BENCHMARK PRESERVATION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_fig5_intro():
    mo.md(r"""
    ---
    ## Act VI — Figure 5: The Benchmark Trade-off

    The critical question for any unlearning method: **does it preserve
    general language capability?** The paper evaluates on 6 standard benchmarks
    at each fine-tuning step, alongside two familiarity scores.

    Select benchmarks to overlay below, then see the dual-axis chart:
    """)
    return


@app.cell
def cell_fig5_controls():
    benchmark_selector = mo.ui.multiselect(
        options=["ARC-C", "ARC-Easy", "BoolQ", "HellaSwag", "PIQA", "WinoGrande"],
        value=["ARC-C", "HellaSwag", "BoolQ"],
        label="Benchmarks to overlay",
    )
    familiarity_selector = mo.ui.multiselect(
        options=["Familiarity (completion)", "Familiarity (probability)"],
        value=["Familiarity (completion)", "Familiarity (probability)"],
        label="Familiarity metrics",
    )
    return benchmark_selector, familiarity_selector


@app.cell(hide_code=True)
def cell_fig5_layout(benchmark_selector, familiarity_selector):
    mo.hstack([
        mo.vstack([mo.md("**Benchmarks**"), benchmark_selector]),
        mo.vstack([mo.md("**Familiarity metrics**"), familiarity_selector]),
    ], gap=3, justify="start")
    return


@app.cell(hide_code=True)
def cell_fig5_chart(benchmark_selector, familiarity_selector):
    _steps = FIG5_DATA["steps"]

    # Familiarity series (left y-axis)
    _fam_rows = []
    for _fam in (familiarity_selector.value or ["Familiarity (completion)"]):
        for _s, _v in zip(_steps, FIG5_DATA[_fam]):
            _fam_rows.append({"step": _s, "value": _v, "series": _fam, "axis": "familiarity"})
    _df_fam = pd.DataFrame(_fam_rows)

    # Benchmark series (right y-axis, normalised to % change from step 0)
    _bench_rows = []
    for _b in (benchmark_selector.value or ["ARC-C"]):
        _base = FIG5_DATA[_b][0]
        for _s, _v in zip(_steps, FIG5_DATA[_b]):
            _bench_rows.append({
                "step": _s,
                "value": _v,
                "pct_change": (_v - _base) / _base * 100,
                "series": _b,
                "axis": "benchmark",
            })
    _df_bench = pd.DataFrame(_bench_rows)

    _fam_chart = (
        alt.Chart(_df_fam)
        .mark_line(point=True, strokeWidth=2.5, strokeDash=[1, 0])
        .encode(
            x=alt.X("step:Q", title="Fine-tuning steps"),
            y=alt.Y("value:Q", title="Familiarity score",
                    scale=alt.Scale(domain=[0, 0.35])),
            color=alt.Color("series:N", scale=alt.Scale(range=[C_RED, C_ORANGE]),
                            legend=alt.Legend(title="")),
            tooltip=["series:N", "step:Q", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=580, height=300)
    )

    _bench_chart = (
        alt.Chart(_df_bench)
        .mark_line(point=True, strokeWidth=2, strokeDash=[5, 3])
        .encode(
            x=alt.X("step:Q"),
            y=alt.Y("pct_change:Q", title="Benchmark change (%)",
                    scale=alt.Scale(domain=[-3, 3])),
            color=alt.Color("series:N", scale=alt.Scale(
                range=[C_BLUE, C_GREEN, C_PURPLE, "#17BECF", "#BCBD22", "#8C564B"]
            ), legend=alt.Legend(title="")),
            tooltip=["series:N", "step:Q",
                     alt.Tooltip("value:Q", format=".3f"),
                     alt.Tooltip("pct_change:Q", format=".2f", title="% change")],
        )
        .properties(width=580, height=300)
    )

    _zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#aaa", strokeDash=[3, 3])
        .encode(y="y:Q")
    )

    alt.hconcat(
        _fam_chart.properties(
            title=alt.TitleParams("Familiarity scores (lower = more forgotten)",
                                  subtitleFontSize=11)),
        (_bench_chart + _zero).properties(
            title=alt.TitleParams("Benchmark change from baseline (%)",
                                  subtitle="Near zero = capability preserved",
                                  subtitleFontSize=11)),
        spacing=40,
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    return


@app.cell(hide_code=True)
def cell_fig5_table():
    _rows = []
    for _step in FIG5_DATA["steps"]:
        _i = FIG5_DATA["steps"].index(_step)
        _rows.append({
            "Steps": _step,
            "Familiarity↓": f"{FIG5_DATA['Familiarity (completion)'][_i]:.3f}",
            "ARC-C":        f"{FIG5_DATA['ARC-C'][_i]:.3f}",
            "BoolQ":        f"{FIG5_DATA['BoolQ'][_i]:.3f}",
            "HellaSwag":    f"{FIG5_DATA['HellaSwag'][_i]:.3f}",
            "PIQA":         f"{FIG5_DATA['PIQA'][_i]:.3f}",
            "WinoGrande":   f"{FIG5_DATA['WinoGrande'][_i]:.3f}",
        })
    mo.accordion({
        "📋 Figure 5 data (exact values from paper)": mo.ui.table(
            _rows, selection=None, pagination=False,
            label="Familiarity scores and benchmarks at each fine-tuning step",
        )
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT VII — EXTENSION: INTERACTIVE PARETO EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_extension_intro():
    mo.md(r"""
    ---
    ## 🔬 Act VII — Extension: The Forgetting Frontier
    > **Novel contribution beyond the paper**

    The paper reports Figure 5 as a static table. The key insight buried in that
    table is a **Pareto frontier**: as fine-tuning steps increase, familiarity
    drops *fast* while benchmarks drop *slowly*. This is the method's core advantage.

    We make this frontier interactive:

    - **X-axis:** Average benchmark score (want this *high* — preserves capability)
    - **Y-axis:** Familiarity score (want this *low* — means successful forgetting)
    - **Each point:** One fine-tuning checkpoint

    **The ideal unlearning method** has points in the **bottom-right corner**:
    low familiarity, high benchmarks. Drag the step slider to animate the trajectory.
    """)
    return


@app.cell
def cell_pareto_controls():
    step_highlight_ui = mo.ui.slider(
        0, 120, value=0, step=20, show_value=True,
        label="Highlight fine-tuning step",
    )
    benchmark_pareto_ui = mo.ui.dropdown(
        options=["ARC-C", "ARC-Easy", "BoolQ", "HellaSwag", "PIQA", "WinoGrande", "Average"],
        value="Average",
        label="Benchmark for X-axis",
    )
    return step_highlight_ui, benchmark_pareto_ui


@app.cell(hide_code=True)
def cell_pareto_layout(step_highlight_ui, benchmark_pareto_ui):
    mo.hstack([
        mo.vstack([mo.md("**Benchmark (X-axis)**"), benchmark_pareto_ui]),
        mo.vstack([mo.md("**Highlight step**"), step_highlight_ui,
                   mo.md("_Drag to walk through the unlearning trajectory_")]),
    ], gap=3, justify="start")
    return


@app.cell(hide_code=True)
def cell_pareto_chart(step_highlight_ui, benchmark_pareto_ui):
    _bench_key = benchmark_pareto_ui.value
    _highlighted = step_highlight_ui.value

    # Build pareto dataframe
    _rows = []
    _bench_keys = (["ARC-C","ARC-Easy","BoolQ","HellaSwag","PIQA","WinoGrande"]
                   if _bench_key == "Average" else [_bench_key])
    for _i, _step in enumerate(FIG5_DATA["steps"]):
        _bval = np.mean([FIG5_DATA[k][_i] for k in _bench_keys])
        _fam  = FIG5_DATA["Familiarity (completion)"][_i]
        _rows.append({
            "step":           _step,
            "benchmark":      float(_bval),
            "familiarity":    float(_fam),
            "highlighted":    _step == _highlighted,
            "label":          f"Step {_step}",
            "fam_pct":        f"{_fam*100:.1f}%",
            "bench_pct":      f"{_bval*100:.1f}%",
        })
    _df = pd.DataFrame(_rows)

    # Frontier line
    _line = (
        alt.Chart(_df)
        .mark_line(color=C_GREY, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(
            x=alt.X("benchmark:Q",
                    title=f"{'Average' if _bench_key=='Average' else _bench_key} benchmark score",
                    scale=alt.Scale(domain=[0.55, 0.78])),
            y=alt.Y("familiarity:Q", title="Familiarity score (lower = better forgotten)",
                    scale=alt.Scale(domain=[-0.01, 0.32])),
        )
    )

    # All points
    _points = (
        alt.Chart(_df)
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x="benchmark:Q",
            y="familiarity:Q",
            size=alt.condition(alt.datum.highlighted, alt.value(200), alt.value(80)),
            color=alt.condition(
                alt.datum.highlighted,
                alt.value(C_RED),
                alt.Color("step:Q", scale=alt.Scale(scheme="blues"),
                          legend=alt.Legend(title="Step")),
            ),
            tooltip=["label:N",
                     alt.Tooltip("benchmark:Q", format=".3f", title="Benchmark"),
                     alt.Tooltip("familiarity:Q", format=".3f", title="Familiarity"),
                     "fam_pct:N", "bench_pct:N"],
        )
    )

    # Labels
    _labels = (
        alt.Chart(_df)
        .mark_text(align="left", dx=8, dy=-6, fontSize=10)
        .encode(
            x="benchmark:Q", y="familiarity:Q",
            text=alt.condition(alt.datum.highlighted, alt.value("← YOU ARE HERE"), alt.value("")),
            color=alt.value(C_RED),
        )
    )

    # Step annotations for all points
    _step_labels = (
        alt.Chart(_df)
        .mark_text(align="left", dx=6, fontSize=9, color="#666")
        .encode(
            x="benchmark:Q", y="familiarity:Q", text="label:N",
        )
    )

    # Ideal region annotation
    _ideal_df = pd.DataFrame([{"x1": 0.70, "x2": 0.78, "y1": -0.01, "y2": 0.05}])
    _ideal_box = (
        alt.Chart(_ideal_df)
        .mark_rect(color=C_GREEN, opacity=0.07)
        .encode(x="x1:Q", x2="x2:Q", y="y1:Q", y2="y2:Q")
    )
    _ideal_text = (
        alt.Chart(pd.DataFrame([{"x": 0.74, "y": 0.015, "label": "Ideal zone"}]))
        .mark_text(fontSize=10, color=C_GREEN, fontStyle="italic")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    (_ideal_box + _line + _points + _step_labels + _labels + _ideal_text)
    .properties(
        width=620, height=380,
        title=alt.TitleParams(
            "The Forgetting Frontier: Familiarity vs Benchmark Preservation",
            subtitle=("Each point = one fine-tuning checkpoint. "
                      "Bottom-right = best: low familiarity, high benchmark. "
                      "Drag the step slider above to walk the trajectory."),
            subtitleFontSize=11,
        ),
    )
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
    return


@app.cell(hide_code=True)
def cell_pareto_stats(step_highlight_ui):
    _i     = FIG5_DATA["steps"].index(step_highlight_ui.value)
    _fam   = FIG5_DATA["Familiarity (completion)"][_i]
    _fam0  = FIG5_DATA["Familiarity (completion)"][0]
    _arc   = FIG5_DATA["ARC-C"][_i]
    _arc0  = FIG5_DATA["ARC-C"][0]
    _hell  = FIG5_DATA["HellaSwag"][_i]
    _hell0 = FIG5_DATA["HellaSwag"][0]
    _fam_drop   = (_fam0 - _fam)  / _fam0  * 100
    _arc_drop   = (_arc0 - _arc)  / _arc0  * 100
    _hell_drop  = (_hell0 - _hell) / _hell0 * 100

    mo.hstack([
        mo.stat(f"Step {step_highlight_ui.value}", label="Current checkpoint",
                caption="0 = pretrained baseline, 120 = fully unlearned"),
        mo.stat(f"{_fam:.3f}",         label="Familiarity score",
                caption=f"↓ {_fam_drop:.0f}% from baseline ({_fam0:.3f})"),
        mo.stat(f"{_arc:.3f}",         label="ARC-C",
                caption=f"↓ {_arc_drop:.1f}% from baseline ({_arc0:.3f})"),
        mo.stat(f"{_hell:.3f}",        label="HellaSwag",
                caption=f"↓ {_hell_drop:.1f}% from baseline ({_hell0:.3f})"),
    ], gap=2, justify="start")
    return


@app.cell(hide_code=True)
def cell_pareto_callout(step_highlight_ui):
    _i   = FIG5_DATA["steps"].index(step_highlight_ui.value)
    _fam = FIG5_DATA["Familiarity (completion)"][_i]
    if _fam <= 0.01:
        mo.callout(mo.md(
            "✅ **Essentially forgotten.** Familiarity score ≤ 0.01. "
            "The model no longer recalls Harry Potter-specific content in "
            "completion tests — while retaining >93% of its benchmark performance."
        ), kind="success")
    elif _fam <= 0.04:
        mo.callout(mo.md(
            "🟡 **Mostly forgotten.** Strong reduction from baseline (0.290) "
            "but a small residual signal remains. A few more fine-tuning steps "
            "will complete the unlearning."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "🔴 **Still remembers.** Significant HP familiarity remains. "
            "More fine-tuning steps needed — or consider increasing α "
            "for stronger suppression."
        ), kind="danger")
    return


# ─────────────────────────────────────────────────────────────────────────────
# TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_takeaways():
    mo.md(r"""
    ---
    ## Key Takeaways

    | Finding | What it means |
    |---------|---------------|
    | 🧹 **Gradient reversal alone doesn't work** | Suppressing one HP token just shifts probability to the next HP token |
    | 🔬 **Reinforced model identifies HP-specific tokens** | Over-training amplifies the tokens we want to suppress |
    | 📖 **Anchor dictionary handles explicit references** | Substituting HP terms with generics provides clean training targets |
    | ⚖️ **The Pareto frontier is steep** | 97.6% familiarity reduction costs only ~6% on ARC-C |
    | 🎯 **Both ingredients are necessary** | Ablation shows each alone is insufficient (Section 4.1) |
    | 🔭 **Broader implications** | A foundation for legally-compliant, adaptable LLMs — and a new axis of model control |

    ---

    **Paper:** [arxiv.org/abs/2310.02238](https://arxiv.org/abs/2310.02238)
    · Eldan & Russinovich · Microsoft Research 2023  
    **Model:** [huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)  
    **Notebook:** built for the alphaXiv × marimo Notebook Competition
    """)
    return


if __name__ == "__main__":
    app.run()
