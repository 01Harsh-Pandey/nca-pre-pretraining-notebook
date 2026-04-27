# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "marimo==0.9.14",
#   "numpy==2.1.1",
#   "altair==5.4.1",
#   "pandas==2.2.3",
#   "scipy==1.14.1",
# ]
# ///

import marimo

__generated_with = "0.9.14"
app = marimo.App(
    width="full",
    app_title="The Dead Salmons of AI Interpretability",
)

# ============================================================
# SETUP — all imports, constants, paper data
# ============================================================
with app.setup:
    import numpy as np
    import altair as alt
    import pandas as pd
    import marimo as mo
    from scipy import stats

    # ── Colour palette ───────────────────────────────────────
    C_SALMON = "#FA8072"
    C_BLUE   = "#4C78A8"
    C_ORANGE = "#F58518"
    C_GREEN  = "#54A24B"
    C_GREY   = "#CCCCCC"
    C_DARK   = "#2D2D2D"
    C_GOLD   = "#FFD700"

    RNG_SEED = 42

    # ── Figure 4 data (extracted from paper text) ────────────
    LAYERS = list(range(13))   # Layer 0 = embedding, 1-12 = transformer

    # (A) Sentiment — IMDb — BERT-base-uncased
    SENT_PRE    = [0.500, 0.520, 0.550, 0.590, 0.640, 0.680,
                   0.720, 0.760, 0.800, 0.830, 0.850, 0.870, 0.880]
    SENT_RAND   = [0.500, 0.510, 0.520, 0.530, 0.530, 0.540,
                   0.540, 0.550, 0.560, 0.580, 0.620, 0.680, 0.720]
    SENT_CHANCE = 0.500

    # (B) Syntax / POS — CoNLL-2003 — BERT-base-uncased
    POS_PRE     = [0.400, 0.650, 0.820, 0.880, 0.920, 0.940,
                   0.950, 0.940, 0.920, 0.880, 0.820, 0.750, 0.680]
    POS_RAND    = [0.400, 0.450, 0.480, 0.490, 0.500, 0.500,
                   0.490, 0.480, 0.470, 0.460, 0.450, 0.420, 0.400]
    POS_MAJORITY = 0.175

    # (C) World Models — Pythia-160m — Gurnee & Tegmark (2024)
    WORLD_PRE   = [0.120, 0.150, 0.200, 0.250, 0.300, 0.350,
                   0.380, 0.410, 0.440, 0.470, 0.500, 0.520, 0.550]
    WORLD_RAND  = [0.120, 0.110, 0.100, 0.090, 0.080, 0.080,
                   0.070, 0.070, 0.060, 0.060, 0.050, 0.050, 0.040]

    # ── Table 1 — Unified Statistical Framework ──────────────
    FRAMEWORK = [
        {
            "Method": "Performance Benchmarking",
            "Hypothesis Class (E)":
                "Single scalar (accuracy, perplexity…)",
            "Causal Query":
                "Observational — output score under P_U",
            "Error Criterion":
                "Difference in expected performance metrics",
            "Dead Salmon Risk": "⭐ Low",
        },
        {
            "Method": "Probing (Linear Classifiers)",
            "Hypothesis Class (E)":
                "Linear / shallow classifiers on activations",
            "Causal Query":
                "Observational — P(Y | activations)",
            "Error Criterion":
                "Classification loss",
            "Dead Salmon Risk": "⭐⭐⭐ High",
        },
        {
            "Method": "Feature Attributions (SHAP / IG / Saliency)",
            "Hypothesis Class (E)":
                "Input-level additive surrogates",
            "Causal Query":
                "Counterfactual — local approximation around x",
            "Error Criterion":
                "Fidelity loss vs. model predictions",
            "Dead Salmon Risk": "⭐⭐⭐ High",
        },
        {
            "Method": "Concept-Based (TCAV / ACE)",
            "Hypothesis Class (E)":
                "Activations → interpretable concept variables",
            "Causal Query":
                "Interventional — sensitivity to concept activations",
            "Error Criterion":
                "Directional derivative mismatch",
            "Dead Salmon Risk": "⭐⭐⭐ High",
        },
        {
            "Method": "Circuit Discovery",
            "Hypothesis Class (E)":
                "Subgraphs of the computational graph",
            "Causal Query":
                "Interventional — outputs under targeted ablations",
            "Error Criterion":
                "KL(P_C(Y|U) ‖ P_circuit(Y|U))",
            "Dead Salmon Risk": "🚨 Extreme",
        },
        {
            "Method": "Causal Tracing / Patching",
            "Hypothesis Class (E)":
                "Scalar importance scores over units",
            "Causal Query":
                "Counterfactual mediation — total/direct/indirect effect",
            "Error Criterion":
                "Predicted vs. empirical intervention effects",
            "Dead Salmon Risk": "⭐⭐ Moderate",
        },
        {
            "Method": "Causal Abstraction",
            "Hypothesis Class (E)":
                "High-level structural causal model",
            "Causal Query":
                "Interventional invariance — abstraction ↔ intervention",
            "Error Criterion":
                "Causal abstraction error (commutativity violation)",
            "Dead Salmon Risk": "⭐⭐ Moderate",
        },
    ]


# ============================================================
# CELL 1 — Hero Banner
# ============================================================
@app.cell(hide_code=True)
def cell_hero():
    mo.md(
        r"""
        <div style="padding: 2rem 0 1rem 0; max-width: 950px;">
            <h1 style="
                font-family: 'Georgia', serif; 
                font-size: 3.4rem; 
                font-weight: 400; 
                color: #111; 
                letter-spacing: -1px; 
                margin-bottom: 1.5rem; 
                line-height: 1.15;
            ">
                The Dead Salmons of AI Interpretability
            </h1>
            
            <p style="
                font-size: 1.15rem; 
                color: #333; 
                line-height: 1.7; 
                margin-bottom: 2rem;
            ">
                An interactive reproduction of 
                <em><a href="https://arxiv.org/abs/2512.18792" style="color: #FA8072; text-decoration: none; border-bottom: 1px solid rgba(250,128,114,0.4); padding-bottom: 1px; font-weight: 600;">"The Dead Salmons of AI Interpretability"</a></em> 
                (AlphaXiv 2512.18792, 2025). When statistical guardrails are skipped, 
                even a <strong>frozen random network</strong> can look like it understands language.
            </p>
            
            <div style="
                display: flex; 
                align-items: center; 
                gap: 12px; 
                background: #fef9f8; 
                border: 1px solid #fce8e6; 
                padding: 1rem 1.5rem; 
                border-radius: 8px;
            ">
                <span style="font-size: 1.3rem;">🐟</span>
                <span style="font-size: 1rem; color: #2D2D2D;">
                    <strong style="color: #d9534f;">Interactive Features:</strong>
                    <span style="color: #555; margin-left: 8px; font-size: 0.95rem;">
                        📊 Figure 4 Reproduction &nbsp;·&nbsp; 🎲 P-Hacking Simulator &nbsp;·&nbsp; 🛡️ Live Eq. 4 Testing &nbsp;·&nbsp; 🧠 Framework
                    </span>
                </span>
            </div>
        </div>
        """
    )
    return

# ============================================================
# CELL 2 — Origin Story
# ============================================================
@app.cell(hide_code=True)
def cell_origin(mo):
    mo.md(
        r"""
        ## 🧊 Act I — The Original Crime

        In 2009, Bennett et al. placed a **dead Atlantic salmon** inside an fMRI scanner.
        They showed the fish photographs of people in social situations and asked it to
        identify their emotional state. Using the standard voxel-wise pipeline of the era —
        a t-test per voxel, threshold at p < 0.05, **no multiple-comparison correction** —
        the dead salmon showed statistically significant "brain activity" in two clusters.

        > *"Several active voxels were discovered in the salmon's brain cavity."*
        > — Bennett et al. (2009)

        The fish was dead. The scanner was measuring noise. The pipeline declared
        a discovery because it tested **8,064 voxels** simultaneously and never asked:
        **"How many of these would look significant purely by chance?"**

        ### The AI Parallel

        | fMRI World | AI Interpretability |
        |---|---|
        | 130,000 brain voxels | Thousands of neurons / attention heads |
        | t-test per voxel | Probe accuracy per unit / layer |
        | No correction → false positive | No null model → "dead salmon circuit" |
        | Bennett et al. (2009) | This paper (2025) |

        The 2025 paper argues that **every major interpretability method** —
        probing, saliency maps, circuit discovery, TCAV, causal tracing —
        is vulnerable to exactly this failure mode.
        """
    )
    return


# ============================================================
# CELL 3 — Brain Grid Controls
# ============================================================
@app.cell(hide_code=True)
def cell_brain_intro(mo):
    mo.md(
        "## 🔬 Simulate the Dead Salmon Scanner\n\n"
        "_Adjust the controls. Watch false discoveries appear in **pure noise**._"
    )
    return


@app.cell
def cell_brain_sliders(mo):
    n_voxels_slider = mo.ui.slider(
        100, 8064, step=100, value=1500,
        label="Search Space (voxels / neurons)",
        show_value=True,
    )
    alpha_slider = mo.ui.slider(
        0.001, 0.10, step=0.001, value=0.050,
        label="Significance threshold α",
        show_value=True,
    )
    mo.hstack([n_voxels_slider, alpha_slider], widths="equal", gap=2)
    return n_voxels_slider, alpha_slider


# ============================================================
# CELL 4 — Brain Grid Chart (depends only on sliders above)
# ============================================================
@app.cell
def cell_brain_chart(n_voxels_slider, alpha_slider, np, pd, alt, mo):
    rng = np.random.default_rng(RNG_SEED)
    n   = int(n_voxels_slider.value)
    a   = float(alpha_slider.value)

    p_vals    = rng.uniform(0, 1, n)
    sig       = p_vals < a
    fp        = int(sig.sum())
    exp_fp    = round(n * a, 1)

    # Cap display grid at 3600 cells for rendering speed
    display_n = min(n, 3600)
    side      = int(np.sqrt(display_n))
    display_n = side * side

    grid_df = pd.DataFrame({
        "x": np.tile(np.arange(side), side),
        "y": np.repeat(np.arange(side), side),
        "significant": sig[:display_n],
    })

    brain_chart = (
        alt.Chart(grid_df)
        .mark_rect(stroke=None)
        .encode(
            x=alt.X("x:O", axis=None),
            y=alt.Y("y:O", axis=None),
            color=alt.condition(
                "datum.significant",
                alt.value(C_SALMON),
                alt.value("#EBEBEB"),
            ),
            tooltip=[
                alt.Tooltip("significant:N", title="False Positive?"),
            ],
        )
        .properties(
            width=420, height=420,
            title=alt.TitleParams(
                "Simulated Neural Grid — PURE NOISE",
                subtitle=(
                    "Red = 'statistically significant' — "
                    "every red cell is a false positive"
                ),
                subtitleFontSize=12,
            ),
        )
    )

    stats_col = mo.vstack([
        mo.md("### What the scanner found"),
        mo.stat(value=f"{n:,}",
                label="Features Tested"),
        mo.stat(value=f"{exp_fp:,}",
                label="Expected False Positives",
                caption=f"= {n:,} × {a:.3f}"),
        mo.stat(value=f"{fp:,}",
                label="Actual 'Discoveries'",
                caption="Every single one is noise"),
        mo.callout(
            mo.md(
                f"At α = **{a:.3f}** across **{n:,}** tests, "
                f"expect **{exp_fp:,}** false positives "
                f"even when **nothing is real**."
            ),
            kind="warn",
        ),
    ], gap=0.6)

    mo.hstack(
        [mo.ui.altair_chart(brain_chart), stats_col],
        gap=3, align="start",
    )
    return


# ============================================================
# CELL 5 — Figure 4 Introduction
# ============================================================
@app.cell(hide_code=True)
def cell_fig4_intro(mo):
    mo.md(
        r"""
        ## 📈 Act II — Reproducing Figure 4: Real Experiments on Real Models

        The paper tests the dead-salmon problem on three concrete probing tasks.
        Each experiment pits **pretrained models** against **randomly-initialised
        versions of the same architecture** — the null comparison that standard
        analyses skip entirely.

        Select an experiment below to explore the layer-wise results.
        """
    )
    return


# ============================================================
# CELL 6 — Figure 4 Selector  ← THE FIXED CELL
# ============================================================
@app.cell
def cell_fig4_selector(mo):
    experiment_selector = mo.ui.radio(
        options={
            "🎭 (A) Sentiment — BERT":      "sentiment",
            "🏷️ (B) Syntax/POS — BERT":    "pos",
            "🗺️ (C) World Models — Pythia": "world",
        },
        value="sentiment",
        label="Select experiment",
        inline=True,
    )
    mo.hstack([experiment_selector])
    return (experiment_selector,)


# ============================================================
# CELL 7 — Figure 4 Chart + Insight  ← THE FIXED CELL
#
# KEY FIX: single cell owns ALL logic for this section.
# experiment_selector is the ONLY dependency from cell 6.
# No intermediate cells break the DAG.
# ============================================================
@app.cell
def cell_fig4_render(experiment_selector, pd, alt, mo):

    # ── 1. Route to the correct dataset ─────────────────────
    exp_key = experiment_selector.value   # "sentiment" | "pos" | "world"

    if exp_key == "sentiment":
        pre       = SENT_PRE
        rand      = SENT_RAND
        metric    = "Probe Accuracy"
        y_dom     = [0.45, 0.95]
        chance    = SENT_CHANCE
        chance_lbl = "Chance baseline (0.50)"
        insight_kind = "warn"
        insight = (
            "⚠️ **The Dead Salmon Moment:**  "
            "Every pretrained layer looks better than chance — "
            "yet when compared against *randomly initialised* BERT, "
            "**no early layer is statistically distinguishable** "
            "from random computation. "
            "The apparent 'sentiment neurons' are a search artifact."
        )

    elif exp_key == "pos":
        pre       = POS_PRE
        rand      = POS_RAND
        metric    = "Probe Accuracy"
        y_dom     = [0.15, 1.00]
        chance    = POS_MAJORITY
        chance_lbl = "Majority baseline (0.175)"
        insight_kind = "success"
        insight = (
            "✅ **Genuine Discovery:**  "
            "Middle layers (4–6) peak far above the random-initialisation null. "
            "This **survives** the dead-salmon test — "
            "BERT genuinely encodes syntactic structure in its middle layers."
        )

    else:   # world
        pre       = WORLD_PRE
        rand      = WORLD_RAND
        metric    = "Geospatial R²"
        y_dom     = [0.00, 0.60]
        chance    = 0.0
        chance_lbl = "Zero (no spatial information)"
        insight_kind = "info"
        insight = (
            "🌍 **The Embedding Surprise:**  "
            "Even at Layer 0 (raw token embeddings) R² ≈ 0.12 — "
            "geography is already encoded. "
            "Passing embeddings through *random* transformer blocks actually "
            "**hurts** the spatial signal. "
            "The world model is real, but it mostly lives in the "
            "input geometry, not in learned representations."
        )

    # ── 2. Build the dataframe ───────────────────────────────
    df_fig4 = pd.DataFrame({
        "Layer":     LAYERS * 2,
        "Value":     pre + rand,
        "Condition": ["Pretrained"] * 13 + ["Random Initialised (Null)"] * 13,
    })

    # ── 3. Build the Altair chart ────────────────────────────
    #
    # IMPORTANT: we use separate Chart objects for the line,
    # the chance rule, and the chance label — then layer them.
    # This avoids the broken alt.condition inside mark_rule.

    lines = (
        alt.Chart(df_fig4)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Layer:O",
                title="Layer  (0 = embedding, 1–12 = transformer blocks)",
            ),
            y=alt.Y(
                "Value:Q",
                title=metric,
                scale=alt.Scale(domain=y_dom),
            ),
            color=alt.Color(
                "Condition:N",
                scale=alt.Scale(
                    domain=["Pretrained", "Random Initialised (Null)"],
                    range=[C_BLUE, C_ORANGE],
                ),
                legend=alt.Legend(
                    orient="bottom",
                    title=None,
                ),
            ),
            tooltip=[
                alt.Tooltip("Layer:O"),
                alt.Tooltip("Value:Q",     format=".3f"),
                alt.Tooltip("Condition:N"),
            ],
        )
    )

    # Chance / baseline horizontal rule
    chance_df  = pd.DataFrame({"y": [chance]})
    label_df   = pd.DataFrame({"y": [chance], "label": [chance_lbl]})

    chance_rule = (
        alt.Chart(chance_df)
        .mark_rule(
            strokeDash=[6, 4],
            color=C_GREY,
            strokeWidth=1.8,
        )
        .encode(y="y:Q")
    )

    chance_text = (
        alt.Chart(label_df)
        .mark_text(
            align="left",
            dx=6,
            dy=-10,
            color=C_GREY,
            fontSize=11,
        )
        .encode(
            y="y:Q",
            x=alt.value(6),
            text="label:N",
        )
    )

    fig4_chart = (
        (lines + chance_rule + chance_text)
        .properties(
            width=660,
            height=340,
            title=alt.TitleParams(
                f"Reproduction of Figure 4 — {experiment_selector.value}",
                subtitle=(
                    "Paper: 'The Dead Salmons of AI Interpretability' "
                    "(AlphaXiv 2512.18792)"
                ),
                subtitleFontSize=11,
                subtitleColor="#888888",
            ),
        )
        .configure_point(size=70)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=13,
        )
    )

    # ── 4. Stat bar below the chart ──────────────────────────
    pretrained_max = max(pre)
    rand_max       = max(rand)
    gap_pct        = round((pretrained_max - rand_max) * 100, 1)

    stats_row = mo.hstack([
        mo.stat(
            value=f"{pretrained_max:.3f}",
            label="Pretrained peak",
            caption="Best layer",
        ),
        mo.stat(
            value=f"{rand_max:.3f}",
            label="Null peak",
            caption="Best randomised layer",
        ),
        mo.stat(
            value=f"+{gap_pct}pp",
            label="Raw gap",
            caption="Pretrained − null",
        ),
        mo.stat(
            value="p < 0.05?" ,
            label="After null test",
            caption="See insight below",
        ),
    ], gap=1)

    # ── 5. Compose the full output ───────────────────────────
    mo.vstack([
        mo.ui.altair_chart(fig4_chart),
        stats_row,
        mo.callout(mo.md(insight), kind=insight_kind),
    ], gap=1)

    return


# ============================================================
# CELL 8 — Unified Framework (Table 1)
# ============================================================
@app.cell(hide_code=True)
def cell_table1_intro(mo):
    mo.md(
        r"""
        ## 🗺️ Act III — The Unified Statistical Framework (Table 1)

        The paper's central theoretical contribution: every interpretability
        method can be cast as a **statistical estimator** with three components:

        | Component | Meaning |
        |---|---|
        | **E** | Hypothesis class — the surrogate model |
        | **q(C)** | Causal query the method tries to answer |
        | **D** | Error criterion — how faithfully does E answer q? |

        This reveals *exactly where* each method is vulnerable to dead-salmon
        artifacts. Select any method to inspect its structure.
        """
    )
    return


@app.cell
def cell_table1_selector(mo):
    method_names = [row["Method"] for row in FRAMEWORK]
    method_sel = mo.ui.dropdown(
        options=method_names,
        value=method_names[1],   # default: Probing
        label="Select interpretability method",
    )
    mo.hstack([method_sel])
    return (method_sel,)


@app.cell
def cell_table1_render(method_sel, mo):
    row = next(r for r in FRAMEWORK if r["Method"] == method_sel.value)

    risk_str   = row["Dead Salmon Risk"]
    star_count = risk_str.count("⭐")
    risk_kind  = (
        "success" if star_count <= 1
        else "info" if star_count == 2
        else "warn" if star_count == 3
        else "danger"
    )

    detail_card = mo.vstack([
        mo.md(f"### {row['Method']}"),
        mo.hstack([
            mo.vstack([
                mo.md(f"**Hypothesis Class (E)**\n\n{row['Hypothesis Class (E)']}"),
                mo.md(f"**Causal Query q(C)**\n\n{row['Causal Query']}"),
                mo.md(f"**Error Criterion D**\n\n{row['Error Criterion']}"),
            ], gap=0.6),
            mo.callout(
                mo.md(
                    f"### Dead Salmon Risk\n\n"
                    f"**{risk_str}**\n\n"
                    "_Null recommendation: compare against full weight "
                    "reinitialisation or label shuffling._"
                ),
                kind=risk_kind,
            ),
        ], gap=2, align="start"),
    ], gap=0.8)

    full_table = mo.ui.table(
        FRAMEWORK,
        label="Full Table 1 — All Methods",
        selection=None,
        pagination=True,
        page_size=4,
    )

    mo.vstack([detail_card, full_table], gap=1)
    return


# ============================================================
# CELL 9 — Equation 4 (The Fix)
# ============================================================
@app.cell(hide_code=True)
def cell_eq4_intro(mo):
    mo.md(
        r"""
        ## 🛡️ Act IV — The Fix: Hypothesis Testing (Equation 4)

        The paper's fix is elegant and general. Frame interpretability as a
        **hypothesis test** against a null model where the observed explanation
        arises from random computation.

        $$
        \hat{p} \;=\;
        \frac{
            1 + \displaystyle\sum_{b=1}^{B}
            \mathbf{1}\!\left\{T_{\text{null}}^{(b)} \geq T_{\text{obs}}\right\}
        }{B + 1}
        $$

        - $T_{\text{obs}}$ — your interpretability metric on the **real** model
        - $T_{\text{null}}^{(b)}$ — same metric on the $b$-th **randomised** model
        - $B$ — number of null samples (more = more precise)
        - The **+1** in numerator and denominator guarantees Type I error control

        > *"By design, when the randomisation includes full weight reinitialisation,
        > no dead salmon artifacts can remain."* — Paper, Appendix A
        """
    )
    return


@app.cell
def cell_eq4_sliders(mo):
    b_slider = mo.ui.slider(
        10, 500, step=10, value=100,
        label="B — Number of null samples",
        show_value=True,
    )
    t_obs_slider = mo.ui.slider(
        0.50, 0.95, step=0.01, value=0.72,
        label="T_obs — Observed probe accuracy",
        show_value=True,
    )
    mo.hstack([b_slider, t_obs_slider], widths="equal", gap=2)
    return b_slider, t_obs_slider


@app.cell
def cell_eq4_render(b_slider, t_obs_slider, np, pd, alt, mo):
    rng2  = np.random.default_rng(RNG_SEED + 1)
    B     = int(b_slider.value)
    t_obs = float(t_obs_slider.value)

    # Simulate null distribution
    null_acc = rng2.normal(loc=0.58, scale=0.06, size=B).clip(0.40, 0.95)
    p_hat    = (1 + int(np.sum(null_acc >= t_obs))) / (B + 1)

    df_null = pd.DataFrame({"accuracy": null_acc})

    hist = (
        alt.Chart(df_null)
        .mark_bar(color=C_GREY, opacity=0.85)
        .encode(
            x=alt.X(
                "accuracy:Q",
                bin=alt.Bin(maxbins=30),
                title="Null Probe Accuracy (shuffled labels)",
            ),
            y=alt.Y("count()", title="Frequency"),
        )
    )

    t_obs_df   = pd.DataFrame({"x": [t_obs]})
    t_obs_text = pd.DataFrame({"x": [t_obs],
                                "label": [f"T_obs = {t_obs:.2f}"]})

    rule = (
        alt.Chart(t_obs_df)
        .mark_rule(color=C_SALMON, strokeWidth=3)
        .encode(x="x:Q")
    )
    text = (
        alt.Chart(t_obs_text)
        .mark_text(
            align="left", dx=6, dy=-12,
            color=C_SALMON, fontSize=12, fontWeight="bold",
        )
        .encode(x="x:Q", text="label:N")
    )

    eq4_chart = (hist + rule + text).properties(
        width=560, height=280,
        title="Your Result vs. Search-Matched Null Distribution",
    )

    verdict_kind = "success" if p_hat < 0.05 else "warn"
    verdict_txt  = (
        f"**p̂ = {p_hat:.3f}** — "
        + (
            "Finding **survives** the null test. ✅"
            if p_hat < 0.05
            else "Finding **is a Dead Salmon** — "
                 "indistinguishable from random noise. 🐟"
        )
    )

    mo.vstack([
        mo.ui.altair_chart(eq4_chart),
        mo.hstack([
            mo.stat(value=f"{p_hat:.3f}", label="Monte Carlo p̂"),
            mo.stat(value=str(B),          label="Null Samples (B)"),
            mo.stat(value=f"{t_obs:.2f}",  label="T_obs"),
            mo.stat(
                value=str(int(np.sum(null_acc >= t_obs))),
                label="Null runs ≥ T_obs",
            ),
        ], gap=1),
        mo.callout(mo.md(verdict_txt), kind=verdict_kind),
    ], gap=1)
    return


# ============================================================
# CELL 10 — P-Hacking Simulator (The Game)
# ============================================================
@app.cell(hide_code=True)
def cell_phack_intro(mo):
    mo.md(
        r"""
        ## 🎲 Act V — The P-Hacking Simulator

        **You are a researcher.** You want to publish a paper showing your
        language model has a *Sentiment Circuit*.

        You have a budget: how many random neurons can you scan?
        Find one with a strong-looking correlation and **publish**.

        > *Spoiler: you will always succeed eventually.
        > Then we apply Equation 4 and watch it collapse.*
        """
    )
    return


@app.cell
def cell_phack_slider(mo):
    scan_budget = mo.ui.slider(
        1, 200, step=1, value=1,
        label="🔍 Search Budget — Neurons Scanned",
        show_value=True,
    )
    mo.hstack([
        scan_budget,
        mo.callout(
            mo.md(
                "Slide right. Watch your 'discovery' improve. "
                "Then watch the corrected p-value below."
            ),
            kind="info",
        ),
    ], gap=2, align="center")
    return (scan_budget,)


@app.cell
def cell_phack_render(scan_budget, np, pd, alt, mo):
    rng3       = np.random.default_rng(RNG_SEED + 2)
    N_NEURONS  = 200
    N_SAMPLES  = 80

    # Pure noise — NO relationship between neurons and labels
    activations = rng3.normal(size=(N_SAMPLES, N_NEURONS))
    labels_ph   = rng3.integers(0, 2, size=N_SAMPLES).astype(float)

    # |Pearson r| per neuron
    corrs = np.array([
        abs(float(np.corrcoef(activations[:, i], labels_ph)[0, 1]))
        for i in range(N_NEURONS)
    ])

    budget   = int(scan_budget.value)
    scanned  = corrs[:budget]
    best_idx = int(np.argmax(scanned))
    best_cor = float(scanned[best_idx])

    # Naive p-value (no search-budget correction)
    denom    = max(1 - best_cor ** 2, 1e-9)
    t_stat   = best_cor * np.sqrt((N_SAMPLES - 2) / denom)
    naive_p  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=N_SAMPLES - 2)))

    # Search-matched null — Equation 4
    B_ph      = 400
    rng4      = np.random.default_rng(RNG_SEED + 3)
    null_corrs = np.array([
        float(np.max(np.abs([
            np.corrcoef(rng4.normal(size=N_SAMPLES), labels_ph)[0, 1]
            for _ in range(budget)
        ])))
        for _ in range(B_ph)
    ])
    p_hat_ph = (1 + int(np.sum(null_corrs >= best_cor))) / (B_ph + 1)

    # ── Bar chart: scanned neurons ───────────────────────────
    df_bar = pd.DataFrame({
        "Neuron":      [f"N-{i:03d}" for i in range(budget)],
        "Correlation": scanned,
        "Best":        [i == best_idx for i in range(budget)],
    }).sort_values("Correlation", ascending=False)

    bar = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X(
                "Neuron:N",
                sort="-y",
                axis=alt.Axis(labelAngle=-60, labelFontSize=9),
                title="Neuron ID",
            ),
            y=alt.Y(
                "Correlation:Q",
                title="|Pearson r| with sentiment label",
                scale=alt.Scale(domain=[0, 0.5]),
            ),
            color=alt.condition(
                "datum.Best",
                alt.value(C_SALMON),
                alt.value(C_GREY),
            ),
            tooltip=["Neuron:N", "Correlation:Q"],
        )
        .properties(
            width=max(budget * 14, 320),
            height=240,
            title=f"Neurons Scanned (Budget = {budget}  |  Data = pure noise)",
        )
    )

    # ── Two-column verdict ───────────────────────────────────
    naive_kind = "success" if naive_p < 0.05 else "warn"
    corr_kind  = "success" if p_hat_ph < 0.05 else "warn"

    col_naive = mo.vstack([
        mo.md("### 📰 Naive Analysis\n_(no search correction)_"),
        mo.stat(value=f"neuron_{best_idx:03d}",
                label="'Circuit' identified"),
        mo.stat(value=f"{best_cor:.3f}",
                label="|Correlation|"),
        mo.stat(value=f"{naive_p:.4f}",
                label="Naive p-value"),
        mo.callout(
            mo.md(
                "**PUBLISH!** Significant result. ✅"
                if naive_p < 0.05
                else "Not yet significant — keep scanning…"
            ),
            kind=naive_kind,
        ),
    ], gap=0.5)

    col_corr = mo.vstack([
        mo.md("### 🛡️ Corrected Analysis\n_(Equation 4, search-matched null)_"),
        mo.stat(value=str(B_ph),
                label="Null Runs (B)"),
        mo.stat(value=f"{best_cor:.3f}",
                label="T_obs"),
        mo.stat(value=f"{p_hat_ph:.3f}",
                label="p̂  (corrected)"),
        mo.callout(
            mo.md(
                "**Genuine signal survives.** ✅"
                if p_hat_ph < 0.05
                else "**Dead Salmon. 🐟** "
                     "Indistinguishable from noise "
                     "once search cost is accounted for."
            ),
            kind=corr_kind,
        ),
    ], gap=0.5)

    mo.vstack([
        mo.ui.altair_chart(bar),
        mo.hstack([col_naive, col_corr], gap=3, align="start"),
    ], gap=1)
    return


# ============================================================
# CELL 11 — Three Regimes
# ============================================================
@app.cell(hide_code=True)
def cell_regimes_intro(mo):
    mo.md(
        r"""
        ## ⚗️ Act VI — Three Regimes: Null → Weak Signal → Real Signal

        The paper identifies three distinct situations depending on how much
        genuine structure exists in the data.
        Use the slider to plant real signal and watch **all guardrails respond
        simultaneously**.
        """
    )
    return


@app.cell
def cell_regimes_sliders(mo):
    signal_slider = mo.ui.slider(
        0.0, 2.0, step=0.1, value=0.0,
        label="Planted Signal Strength  (0 = pure null)",
        show_value=True,
    )
    n_samples_slider = mo.ui.slider(
        50, 300, step=10, value=150,
        label="Sample Count",
        show_value=True,
    )
    mo.hstack([signal_slider, n_samples_slider], widths="equal", gap=2)
    return signal_slider, n_samples_slider


@app.cell
def cell_regimes_render(signal_slider, n_samples_slider, np, pd, alt, mo):
    rng5    = np.random.default_rng(RNG_SEED + 4)
    sig_str = float(signal_slider.value)
    n_samp  = int(n_samples_slider.value)
    n_feat  = 64

    # Generate data with planted signal in feature 0
    X_reg  = rng5.normal(size=(n_samp, n_feat))
    latent = sig_str * X_reg[:, 0] + rng5.normal(size=n_samp)
    y_reg  = (latent >= float(np.median(latent))).astype(float)

    # Compute |r| and p per feature
    corrs_reg = np.array([
        abs(float(np.corrcoef(X_reg[:, i], y_reg)[0, 1]))
        for i in range(n_feat)
    ])
    p_vals_reg = np.array([
        float(2 * (1 - stats.t.cdf(
            abs(c * np.sqrt(max(n_samp - 2, 1) / max(1 - c**2, 1e-9))),
            df=max(n_samp - 2, 1),
        )))
        for c in corrs_reg
    ])

    raw_hits  = int(np.sum(p_vals_reg < 0.05))
    bonf_hits = int(np.sum(p_vals_reg < 0.05 / n_feat))

    # Regime
    if sig_str <= 0.1:
        regime      = "🐟 Null"
        regime_kind = "warn"
        regime_desc = (
            "Everything is noise. Any discovery is a dead salmon."
        )
    elif sig_str < 0.9:
        regime      = "⚠️ Weak Signal"
        regime_kind = "info"
        regime_desc = (
            "Some real structure exists, but selection effects "
            "still inflate the raw hit count."
        )
    else:
        regime      = "✅ Real Signal"
        regime_kind = "success"
        regime_desc = (
            "Strong planted signal. Guardrails now separate "
            "genuine structure from noise."
        )

    df_reg = pd.DataFrame({
        "Feature":     np.arange(n_feat),
        "Correlation": corrs_reg,
        "Significant": p_vals_reg < 0.05,
    })

    scatter = (
        alt.Chart(df_reg)
        .mark_circle(size=65, opacity=0.85)
        .encode(
            x=alt.X("Feature:Q", title="Feature Index"),
            y=alt.Y(
                "Correlation:Q",
                title="|Pearson r| with Label",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.condition(
                "datum.Significant",
                alt.value(C_SALMON),
                alt.value(C_GREY),
            ),
            tooltip=["Feature:Q",
                     alt.Tooltip("Correlation:Q", format=".3f"),
                     "Significant:N"],
        )
        .properties(
            width=560, height=280,
            title="Feature Correlations  (Red = raw p < 0.05)",
        )
    )

    mo.vstack([
        mo.ui.altair_chart(scatter),
        mo.callout(
            mo.md(f"**Regime: {regime}** — {regime_desc}"),
            kind=regime_kind,
        ),
        mo.hstack([
            mo.stat(value=str(raw_hits),
                    label="Raw Hits  (p < 0.05)"),
            mo.stat(value=str(bonf_hits),
                    label="Bonferroni Survivors"),
            mo.stat(value=regime,
                    label="Current Regime"),
        ], gap=1),
    ], gap=1)
    return


# ============================================================
# CELL 12 — Researcher's Guardrails Checklist
# ============================================================
@app.cell(hide_code=True)
def cell_checklist_intro(mo):
    mo.md(
        r"""
        ## ✅ Act VII — The Practical Guardrails Checklist

        The paper distills its framework into five concrete steps every
        interpretability finding should pass **before publication**.
        """
    )
    return


@app.cell
def cell_checklist(mo):
    checks = mo.ui.array([
        mo.ui.checkbox(
            label="1. Define the null — use random reinitialisation or label shuffling."
        ),
        mo.ui.checkbox(
            label="2. Match search budget — compare your best result against the "
                  "best result of an equivalent null search."
        ),
        mo.ui.checkbox(
            label="3. Report p̂ — use Equation 4, not raw correlation or accuracy."
        ),
        mo.ui.checkbox(
            label="4. Quantify uncertainty — bootstrap confidence intervals on T_obs."
        ),
        mo.ui.checkbox(
            label="5. Check robustness — findings should hold across seeds and data subsets."
        ),
    ])
    mo.vstack([checks])
    return (checks,)


@app.cell
def cell_checklist_verdict(checks, mo):
    n_checked = sum(1 for c in checks.value if c)
    all_done  = n_checked == 5

    if all_done:
        cert = mo.callout(
            mo.md(
                "### 🏅 Certificate of Reproducibility\n\n"
                "All five guardrails satisfied. "
                "Your interpretability finding has earned the right to be believed."
            ),
            kind="success",
        )
    elif n_checked >= 3:
        cert = mo.callout(
            mo.md(
                f"**{n_checked}/5 guardrails satisfied.** "
                "Getting there — each unchecked box is a potential dead salmon."
            ),
            kind="info",
        )
    else:
        cert = mo.callout(
            mo.md(
                f"**{n_checked}/5 guardrails satisfied.** "
                "High risk of dead-salmon artifacts in your current finding."
            ),
            kind="warn",
        )

    cert
    return


# ============================================================
# CELL 13 — Conclusion & References
# ============================================================
@app.cell(hide_code=True)
def cell_conclusion(mo):
    mo.md(
        r"""
        ## 🎓 Conclusion

        > *"If a frozen random network can produce your interpretability story
        > under the same search procedure, the story has not yet earned belief."*

        ### Three Experiments — Three Lessons

        | Experiment | Without Null Test | With Null Test |
        |---|---|---|
        | 🎭 Sentiment / BERT | All layers "significant" | Early layers **fail** — dead salmon |
        | 🏷️ Syntax / BERT | All layers "significant" | Middle layers **survive** — genuine |
        | 🗺️ World Models / Pythia | Grows layer by layer | Signal real but mostly in **embeddings** |

        ### The Five Rules

        1. **Define the null** before you collect evidence.
        2. **Match search budget** — correct for how hard you looked.
        3. **Report p̂** via Monte Carlo Equation 4.
        4. **Quantify uncertainty** — point estimates aren't enough.
        5. **Check robustness** — real signal survives perturbation.

        ---

        ### References

        - **Main paper:** *The Dead Salmons of AI Interpretability*, AlphaXiv 2512.18792 (2025)
        - Bennett et al. — fMRI dead salmon study, 2009
        - Ioannidis — *Why Most Published Research Findings Are False*, PLoS Medicine, 2005
        - Gurnee & Tegmark — *Language Models Represent Space and Time*, 2024
        - Sharkey et al. — *Open Problems in Mechanistic Interpretability*, 2025
        - Kim et al. — *TCAV*, ICML 2018
        - Wang et al. — *IOI Circuit in GPT-2 Small*, 2022
    )
    return


if __name__ == "__main__":
    app.run()
