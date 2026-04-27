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

__generated_with = "0.23.2"
app = marimo.App(
    width="full",
    app_title="The Dead Salmons of AI Interpretability",
)

with app.setup:
    import numpy as np
    import altair as alt
    import pandas as pd
    import marimo as mo
    from scipy import stats

    # ── Colour palette (salmon-themed, accessible) ──────────
    C_SALMON   = "#FA8072"
    C_BLUE     = "#4C78A8"
    C_ORANGE   = "#F58518"
    C_GREEN    = "#54A24B"
    C_GREY     = "#CCCCCC"
    C_DARK     = "#2D2D2D"
    C_BG       = "#F9F6F0"

    # ── Paper data: Figure 4 (extracted from paper text) ────
    LAYERS = list(range(13))   # BERT-base: embedding + 12 transformer layers

    # (A) Sentiment Analysis — IMDb — BERT-base-uncased
    # Probe accuracy: pretrained vs k=20 random-reinit average
    SENT_PRE  = [0.500, 0.520, 0.550, 0.590, 0.640, 0.680,
                 0.720, 0.760, 0.800, 0.830, 0.850, 0.870, 0.880]
    SENT_RAND = [0.500, 0.510, 0.520, 0.530, 0.530, 0.540,
                 0.540, 0.550, 0.560, 0.580, 0.620, 0.680, 0.720]
    SENT_CHANCE = 0.500

    # (B) Syntax / POS Tagging — CoNLL-2003 — BERT-base-uncased
    POS_PRE   = [0.400, 0.650, 0.820, 0.880, 0.920, 0.940,
                 0.950, 0.940, 0.920, 0.880, 0.820, 0.750, 0.680]
    POS_RAND  = [0.400, 0.450, 0.480, 0.490, 0.500, 0.500,
                 0.490, 0.480, 0.470, 0.460, 0.450, 0.420, 0.400]
    POS_MAJORITY = 0.175   # majority-class baseline for POS

    # (C) World Models — Gurnee & Tegmark (2024) places —
    #     Pythia-160m, R² for geospatial coordinates
    WORLD_PRE  = [0.120, 0.150, 0.200, 0.250, 0.300, 0.350,
                  0.380, 0.410, 0.440, 0.470, 0.500, 0.520, 0.550]
    WORLD_RAND = [0.120, 0.110, 0.100, 0.090, 0.080, 0.080,
                  0.070, 0.070, 0.060, 0.060, 0.050, 0.050, 0.040]

    # ── Table 1 — Unified Statistical Framework ─────────────
    FRAMEWORK = [
        {
            "Method": "Performance Benchmarking",
            "Hypothesis Class (E)": "Single scalar (accuracy, perplexity…)",
            "Causal Query": "Observational — output score under P_U",
            "Error Criterion": "Difference in expected performance metrics",
            "Dead Salmon Risk": "⭐ Low",
        },
        {
            "Method": "Probing (Linear Classifiers)",
            "Hypothesis Class (E)": "Linear / shallow classifiers on activations",
            "Causal Query": "Observational — P(Y | activations)",
            "Error Criterion": "Classification loss",
            "Dead Salmon Risk": "⭐⭐⭐ High — many layers & units to scan",
        },
        {
            "Method": "Feature Attributions (SHAP / IG / Saliency)",
            "Hypothesis Class (E)": "Input-level additive surrogates",
            "Causal Query": "Counterfactual — local approximation around x",
            "Error Criterion": "Fidelity loss vs. model predictions",
            "Dead Salmon Risk": "⭐⭐⭐ High — human confirmation bias",
        },
        {
            "Method": "Concept-Based (TCAV / ACE)",
            "Hypothesis Class (E)": "Activations → interpretable concept variables",
            "Causal Query": "Interventional — sensitivity to concept activations",
            "Error Criterion": "Directional derivative mismatch",
            "Dead Salmon Risk": "⭐⭐⭐ High — concept labels can be spurious",
        },
        {
            "Method": "Circuit Discovery",
            "Hypothesis Class (E)": "Subgraphs of the computational graph",
            "Causal Query": "Interventional — outputs under targeted ablations",
            "Error Criterion": "KL(P_C(Y|U) ‖ P_circuit(Y|U))",
            "Dead Salmon Risk": "🚨 Extreme — combinatorial subgraph search",
        },
        {
            "Method": "Causal Tracing / Patching",
            "Hypothesis Class (E)": "Scalar importance scores over units",
            "Causal Query": "Counterfactual mediation — total/direct/indirect effect",
            "Error Criterion": "Predicted vs. empirical intervention effects",
            "Dead Salmon Risk": "⭐⭐ Moderate — selection of units to patch",
        },
        {
            "Method": "Causal Abstraction",
            "Hypothesis Class (E)": "High-level structural causal model",
            "Causal Query": "Interventional invariance — abstraction ↔ intervention commute",
            "Error Criterion": "Causal abstraction error (violation of commutativity)",
            "Dead Salmon Risk": "⭐⭐ Moderate — alignment search over maps",
        },
    ]

    # ── Monte Carlo seed (reproducibility) ───────────────────
    RNG_SEED = 42


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


@app.cell(hide_code=True)
def cell_origin():
    mo.md(r"""
    ## 🧊 Act I — The Original Crime

    In 2009, Bennett et al. placed a **dead Atlantic salmon** inside an fMRI scanner.
    They showed the fish photographs of people and asked it to infer their emotional state.
    Using the standard voxel-wise analysis pipeline of the time — a t-test per voxel,
    threshold at p < 0.05, **no multiple-comparison correction** — the dead salmon
    showed statistically significant "brain activity" in two clusters.

    > *"The salmon was shown a series of photographs depicting human individuals in
    > social situations… A t-contrast was used… Several active voxels were discovered."*
    > — Bennett et al. (2009)

    The fish was dead. The scanner was measuring noise. The pipeline declared a discovery
    because it tested **8,064 voxels** simultaneously and never asked:
    **"How many of these would look significant just by chance?"**

    ### The AI Parallel

    Modern AI interpretability does the same thing:

    | fMRI World | AI Interpretability |
    |---|---|
    | 130,000 voxels | Thousands of neurons / attention heads |
    | t-test per voxel | Probe per unit / correlation per feature |
    | No correction → false positive | No null model → "dead salmon circuit" |

    The paper argues that **every major interpretability method** — probing, saliency maps,
    circuit discovery, TCAV, causal tracing — is vulnerable to exactly this failure mode.
    """)
    return


@app.cell(hide_code=True)
def cell_brain_controls():
    mo.md("""
    ## 🔬 Simulate the Dead Salmon Scanner
    _Adjust the controls and watch false discoveries appear in **pure noise**._
    """)
    return


@app.cell
def cell_brain_sliders():
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
    return alpha_slider, n_voxels_slider


@app.cell(hide_code=True)
def cell_brain_layout(alpha_slider, n_voxels_slider):
    mo.hstack(
        [n_voxels_slider, alpha_slider],
        widths="equal", gap=2,
    )
    return


@app.cell
def cell_brain_chart(alpha_slider, n_voxels_slider):
    rng = np.random.default_rng(RNG_SEED)
    n   = int(n_voxels_slider.value)
    a   = float(alpha_slider.value)

    p_vals = rng.uniform(0, 1, n)
    sig    = p_vals < a
    fp     = int(sig.sum())
    expected_fp = int(n * a)

    # Build a rectangular grid for display (cap at 3600 cells for speed)
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
                alt.datum.significant,
                alt.value(C_SALMON),
                alt.value("#EBEBEB"),
            ),
            tooltip=[alt.Tooltip("significant:N", title="Significant?")],
        )
        .properties(
            width=420, height=420,
            title=alt.TitleParams(
                "Simulated Neural Grid — PURE NOISE",
                subtitle="Red = 'statistically significant' discovery  |  Every red cell is a false positive",
                subtitleFontSize=12,
            ),
        )
    )

    stats_col = mo.vstack([
        mo.md("### What the scanner 'found'"),
        mo.stat(value=f"{n:,}",         label="Features Tested"),
        mo.stat(value=f"{expected_fp:,}", label="Expected False Positives",
                caption=f"= {n:,} × {a:.3f}"),
        mo.stat(value=f"{fp:,}",         label="Actual 'Discoveries'",
                caption="Every single one is noise"),
        mo.callout(
            mo.md(
                f"At α = **{a:.3f}** across **{n:,}** tests, "
                f"you expect **{expected_fp:,}** false positives "
                f"even if **nothing is real**."
            ),
            kind="warn",
        ),
    ], gap=0.6)

    mo.hstack([brain_chart, stats_col], gap=3, align="start")
    return


@app.cell(hide_code=True)
def cell_fig4_intro():
    mo.md(r"""
    ## 📈 Act II — Reproducing Figure 4: Real Experiments on Real Models

    The paper tests the dead-salmon problem on three concrete probing tasks.
    Each experiment pits **pretrained models** against **randomly-initialized
    versions of the same architecture** — the key null comparison that standard
    analyses skip.

    Select an experiment below to explore the layer-wise results.
    """)
    return


@app.cell
def cell_fig4_tabs():
    fig4_tabs = mo.ui.tabs({
        "🎭 (A) Sentiment — BERT": "sentiment",
        "🏷️ (B) Syntax/POS — BERT": "pos",
        "🗺️ (C) World Models — Pythia": "world",
    })
    return (fig4_tabs,)


@app.cell(hide_code=True)
def cell_fig4_tabs_render(fig4_tabs):
    mo.hstack([fig4_tabs], justify="start")
    return


@app.cell
def cell_fig4_data(fig4_tabs):
    tab = fig4_tabs.value

    if tab == "sentiment":
        pre, rand = SENT_PRE,  SENT_RAND
        metric    = "Probe Accuracy"
        y_dom     = [0.45, 0.95]
        chance    = SENT_CHANCE
        chance_lbl = "Chance (0.50)"
        insight = (
            "⚠️ **The Dead Salmon Moment:** Every pretrained layer looks "
            "better than chance — but when compared against *randomly "
            "initialised* BERT, **no layer is statistically distinguishable** "
            "in early layers. The 'sentiment neurons' are a search artifact."
        )
    elif tab == "pos":
        pre, rand = POS_PRE,   POS_RAND
        metric    = "Probe Accuracy"
        y_dom     = [0.15, 1.00]
        chance    = POS_MAJORITY
        chance_lbl = "Majority Baseline (0.175)"
        insight = (
            "✅ **Genuine Discovery:** Middle layers (4–6) peak far above the "
            "random initialisation null. This **survives** the dead-salmon "
            "test — BERT genuinely learns syntactic structure."
        )
    else:
        pre, rand = WORLD_PRE, WORLD_RAND
        metric    = "Geospatial R²"
        y_dom     = [0.00, 0.60]
        chance    = 0.0
        chance_lbl = "Zero (no information)"
        insight = (
            "🌍 **The Embedding Surprise:** Even at Layer 0 (raw embeddings) "
            "R² ≈ 0.12. Passing through *random* transformer blocks actually "
            "**hurts** the spatial signal. Geography is real — but it mostly "
            "lives in the input geometry, not learned representations."
        )

    df_fig4 = pd.DataFrame({
        "Layer": LAYERS * 2,
        "Value": pre + rand,
        "Condition": ["Pretrained"] * 13 + ["Random Initialised (Null)"] * 13,
    })
    return chance, chance_lbl, df_fig4, insight, metric, y_dom


@app.cell
def cell_fig4_chart(chance, chance_lbl, df_fig4, insight, metric, y_dom):
    base = alt.Chart(df_fig4)

    lines = base.mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Layer:O", title="Layer (0 = embedding)"),
        y=alt.Y("Value:Q", title=metric, scale=alt.Scale(domain=y_dom)),
        color=alt.Color(
            "Condition:N",
            scale=alt.Scale(
                domain=["Pretrained", "Random Initialised (Null)"],
                range=[C_BLUE, C_ORANGE],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=["Layer:O", "Value:Q", "Condition:N"],
    )

    chance_rule = (
        alt.Chart(pd.DataFrame({"y": [chance]}))
        .mark_rule(strokeDash=[6, 3], color=C_GREY, strokeWidth=1.5)
        .encode(y="y:Q")
    )

    chance_text = (
        alt.Chart(pd.DataFrame({"y": [chance], "label": [chance_lbl]}))
        .mark_text(align="left", dx=8, dy=-8, color=C_GREY, fontSize=11)
        .encode(y="y:Q", x=alt.value(0), text="label:N")
    )

    fig4_chart = (lines + chance_rule + chance_text).properties(
        width=640, height=320,
        title=alt.TitleParams(
            "Reproduction of Figure 4 — Layer-wise Probing",
            subtitle="Paper: 'The Dead Salmons of AI Interpretability' (2512.18792)",
            subtitleFontSize=11,
        ),
    )

    mo.vstack([
        mo.ui.altair_chart(fig4_chart),
        mo.callout(mo.md(insight), kind="info"),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_table1_intro():
    mo.md(r"""
    ## 🗺️ Act III — The Unified Statistical Framework (Table 1)

    The paper's central theoretical contribution is showing that **every
    interpretability method** can be cast as a statistical estimator with:

    - **E** — a hypothesis class (the surrogate model)
    - **q(C)** — a causal query it tries to answer
    - **D** — an error criterion measuring how well the surrogate answers q

    This matters because it reveals *exactly where* each method is vulnerable
    to dead-salmon artifacts. Select any method below to inspect its structure.
    """)
    return


@app.cell
def cell_table1_selector():
    method_names = [row["Method"] for row in FRAMEWORK]
    method_sel = mo.ui.dropdown(
        options=method_names,
        value=method_names[1],
        label="Select Interpretability Method",
    )
    return (method_sel,)


@app.cell(hide_code=True)
def cell_table1_render(method_sel):
    row = next(r for r in FRAMEWORK if r["Method"] == method_sel.value)

    risk_level = row["Dead Salmon Risk"].count("⭐")
    risk_color = {1: "success", 2: "info", 3: "warn", 0: "danger"}.get(
        risk_level, "danger"
    )

    mo.vstack([
        method_sel,
        mo.hstack([
            mo.vstack([
                mo.md(f"**Hypothesis Class (E)**\n\n{row['Hypothesis Class (E)']}"),
                mo.md(f"**Causal Query q(C)**\n\n{row['Causal Query']}"),
                mo.md(f"**Error Criterion D**\n\n{row['Error Criterion']}"),
            ], gap=0.8),
            mo.callout(
                mo.md(
                    f"### Dead Salmon Risk\n\n"
                    f"{row['Dead Salmon Risk']}\n\n"
                    f"_{row['Causal Query']}_"
                ),
                kind=risk_color,
            ),
        ], gap=2, align="start"),
        mo.ui.table(
            FRAMEWORK,
            label="Full Table 1 — All Interpretability Methods",
            selection=None,
            pagination=True,
            page_size=4,
        ),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_eq4_intro():
    mo.md(r"""
    ## 🛡️ Act IV — The Fix: Hypothesis Testing (Equation 4)

    The paper's practical fix is elegant and general.
    Frame interpretability as a **hypothesis test** against a null model
    where the observed explanation arises from **random computation**.

    $$\hat{p} = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\!\left\{T_{\text{null}}^{(b)} \geq T_{\text{obs}}\right\}}{B + 1}$$

    - $T_{\text{obs}}$ — your interpretability metric on the real model
    - $T_{\text{null}}^{(b)}$ — same metric on the $b$-th **randomised** model
    - $B$ — number of null samples (more = more precise p-value)
    - The **+1** in numerator and denominator guarantees Type I error control

    > *"By design, when the randomisation includes full weight reinitialisation,
    > no dead salmon artifacts can remain."* — Paper, Appendix A
    """)
    return


@app.cell
def cell_eq4_controls():
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
    return b_slider, t_obs_slider


@app.cell(hide_code=True)
def cell_eq4_controls_layout(b_slider, t_obs_slider):
    mo.hstack([b_slider, t_obs_slider], widths="equal", gap=2)
    return


@app.cell
def cell_eq4_chart(b_slider, t_obs_slider):
    rng2   = np.random.default_rng(RNG_SEED + 1)
    B      = int(b_slider.value)
    t_obs  = float(t_obs_slider.value)

    # Simulate null distribution: best probe accuracy under shuffled labels
    null_acc = rng2.normal(loc=0.58, scale=0.06, size=B).clip(0.40, 0.95)

    p_hat = (1 + int(np.sum(null_acc >= t_obs))) / (B + 1)

    df_null = pd.DataFrame({"accuracy": null_acc})

    hist = (
        alt.Chart(df_null)
        .mark_bar(color=C_GREY, opacity=0.8)
        .encode(
            x=alt.X("accuracy:Q", bin=alt.Bin(maxbins=30),
                    title="Null Probe Accuracy (shuffled labels)"),
            y=alt.Y("count()", title="Frequency"),
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"x": [t_obs]}))
        .mark_rule(color=C_SALMON, strokeWidth=3)
        .encode(x="x:Q")
    )

    text = (
        alt.Chart(pd.DataFrame({"x": [t_obs], "label": [f"T_obs = {t_obs:.2f}"]}))
        .mark_text(align="left", dx=6, dy=-12, color=C_SALMON,
                   fontSize=12, fontWeight="bold")
        .encode(x="x:Q", text="label:N")
    )

    eq4_chart = (hist + rule + text).properties(
        width=560, height=280,
        title="Your Result vs. Search-Matched Null Distribution",
    )

    verdict_kind = "success" if p_hat < 0.05 else "warn"
    verdict_text = (
        f"**p̂ = {p_hat:.3f}** — "
        + (
            "Your finding **survives** the null test. ✅"
            if p_hat < 0.05
            else "Your finding **is a Dead Salmon** — indistinguishable from random noise. 🐟"
        )
    )

    mo.vstack([
        mo.ui.altair_chart(eq4_chart),
        mo.hstack([
            mo.stat(value=f"{p_hat:.3f}",    label="Monte Carlo p̂"),
            mo.stat(value=f"{B}",            label="Null Samples (B)"),
            mo.stat(value=f"{t_obs:.2f}",    label="T_obs"),
            mo.stat(
                value=f"{int(np.sum(null_acc >= t_obs))}",
                label="Null runs ≥ T_obs",
            ),
        ], gap=1),
        mo.callout(mo.md(verdict_text), kind=verdict_kind),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_phack_intro():
    mo.md(r"""
    ## 🎲 Act V — The P-Hacking Simulator

    **You are a researcher.** You have just trained a new language model.
    You want to publish a paper showing it has a "Sentiment Circuit."

    You have a budget: how many random neurons can you scan?
    Find one with p < 0.05 and **publish**.

    > *Spoiler: you will always succeed. Then we apply Equation 4.*
    """)
    return


@app.cell
def cell_phack_controls():
    scan_budget = mo.ui.slider(
        1, 200, step=1, value=1,
        label="🔍 Search Budget — Neurons Scanned",
        show_value=True,
    )
    return (scan_budget,)


@app.cell(hide_code=True)
def cell_phack_controls_layout(scan_budget):
    mo.hstack([
        scan_budget,
        mo.callout(
            mo.md("Slide right. Watch your 'discovery' improve. Then see what happens below."),
            kind="info",
        ),
    ], gap=2, align="center")
    return


@app.cell
def cell_phack_logic(scan_budget):
    rng3 = np.random.default_rng(RNG_SEED + 2)
    N_NEURONS = 200
    N_SAMPLES = 80

    # Pure noise — labels have NO relationship to any neuron
    activations = rng3.normal(size=(N_SAMPLES, N_NEURONS))
    labels_ph   = rng3.integers(0, 2, size=N_SAMPLES).astype(float)

    # Compute |correlation| for each neuron
    corrs = np.array([
        abs(np.corrcoef(activations[:, i], labels_ph)[0, 1])
        for i in range(N_NEURONS)
    ])

    budget   = int(scan_budget.value)
    scanned  = corrs[:budget]
    best_idx = int(np.argmax(scanned))
    best_cor = float(scanned[best_idx])

    # Naive p-value (no search correction)
    t_stat    = best_cor * np.sqrt((N_SAMPLES - 2) / (1 - best_cor**2 + 1e-9))
    naive_p   = float(2 * (1 - stats.t.cdf(abs(t_stat), df=N_SAMPLES - 2)))

    # Search-matched null (Equation 4)
    B_ph      = 300
    null_bests = np.array([
        np.max(np.abs(rng3.normal(size=N_SAMPLES) @
                      rng3.normal(size=(N_SAMPLES, budget)) /
                      N_SAMPLES))
        for _ in range(B_ph)
    ])
    p_hat_ph  = (1 + int(np.sum(null_bests >= best_cor))) / (B_ph + 1)

    # Chart: scanned neurons
    df_corrs = pd.DataFrame({
        "Neuron": [f"N-{i}" for i in range(budget)],
        "Correlation": scanned,
        "Best": [i == best_idx for i in range(budget)],
    })

    bar = (
        alt.Chart(df_corrs)
        .mark_bar()
        .encode(
            x=alt.X("Neuron:N", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Correlation:Q"),
            color=alt.condition(
                alt.datum.Best,
                alt.value(C_SALMON),
                alt.value(C_GREY),
            ),
            tooltip=["Neuron:N", "Correlation:Q"],
        )
        .properties(
            width=max(budget * 12, 300), height=220,
            title=f"Neurons Scanned (Budget = {budget})",
        )
    )

    naive_kind   = "success" if naive_p   < 0.05 else "warn"
    corrected_kind = "success" if p_hat_ph < 0.05 else "warn"

    mo.vstack([
        mo.ui.altair_chart(bar),
        mo.hstack([
            mo.vstack([
                mo.md("### 📰 Naive Analysis (no correction)"),
                mo.stat(value=f"neuron_{best_idx}", label="'Circuit' Found"),
                mo.stat(value=f"{best_cor:.3f}",    label="Correlation"),
                mo.stat(value=f"{naive_p:.4f}",     label="p-value (naive)"),
                mo.callout(
                    mo.md(
                        "**PUBLISH!** p < 0.05 ✅" if naive_p < 0.05
                        else "Keep scanning..."
                    ),
                    kind=naive_kind,
                ),
            ], gap=0.5),
            mo.vstack([
                mo.md("### 🛡️ Corrected Analysis (Equation 4)"),
                mo.stat(value=f"{B_ph}",          label="Null Runs (B)"),
                mo.stat(value=f"{best_cor:.3f}",  label="T_obs"),
                mo.stat(value=f"{p_hat_ph:.3f}",  label="p̂ (search-matched)"),
                mo.callout(
                    mo.md(
                        "**Genuine signal.** ✅" if p_hat_ph < 0.05
                        else "**Dead Salmon.** 🐟 Indistinguishable from noise."
                    ),
                    kind=corrected_kind,
                ),
            ], gap=0.5),
        ], gap=3, align="start"),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_regimes_intro():
    mo.md(r"""
    ## ⚗️ Act VI — Three Regimes: From Null to Real Signal

    The paper identifies three distinct regimes depending on
    how much real structure exists in the data.
    Use the slider to plant signal strength and watch all
    guardrails respond simultaneously.
    """)
    return


@app.cell
def cell_regimes_controls():
    signal_slider = mo.ui.slider(
        0.0, 2.0, step=0.1, value=0.0,
        label="Planted Signal Strength",
        show_value=True,
    )
    n_samples_slider = mo.ui.slider(
        50, 300, step=10, value=150,
        label="Sample Count",
        show_value=True,
    )
    return n_samples_slider, signal_slider


@app.cell(hide_code=True)
def cell_regimes_controls_layout(n_samples_slider, signal_slider):
    mo.hstack([signal_slider, n_samples_slider], widths="equal", gap=2)
    return


@app.cell
def cell_regimes_logic(n_samples_slider, signal_slider):
    rng4 = np.random.default_rng(RNG_SEED + 3)
    sig_str  = float(signal_slider.value)
    n_samp   = int(n_samples_slider.value)
    n_feat   = 64

    # Generate data with planted signal
    X = rng4.normal(size=(n_samp, n_feat))
    latent = sig_str * X[:, 0] + rng4.normal(size=n_samp)
    y_reg  = (latent >= np.median(latent)).astype(float)

    # Feature correlations
    corrs_reg = np.array([
        abs(np.corrcoef(X[:, i], y_reg)[0, 1])
        for i in range(n_feat)
    ])
    p_vals_reg = np.array([
        float(2 * (1 - stats.t.cdf(
            abs(c * np.sqrt((n_samp - 2) / (1 - c**2 + 1e-9))),
            df=n_samp - 2
        )))
        for c in corrs_reg
    ])

    raw_hits  = int(np.sum(p_vals_reg < 0.05))
    bonf_hits = int(np.sum(p_vals_reg < 0.05 / n_feat))

    # Regime classification
    if sig_str <= 0.1:
        regime, regime_kind = "🐟 Null", "warn"
        regime_desc = "Everything is noise. Any discovery is a dead salmon."
    elif sig_str < 0.9:
        regime, regime_kind = "⚠️ Weak Signal", "info"
        regime_desc = "Some real structure exists, but selection effects still dominate."
    else:
        regime, regime_kind = "✅ Real Signal", "success"
        regime_desc = "Guardrails now separate genuine structure from noise."

    df_pvals = pd.DataFrame({
        "Feature": np.arange(n_feat),
        "Correlation": corrs_reg,
        "Significant": p_vals_reg < 0.05,
    })

    scatter = (
        alt.Chart(df_pvals)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X("Feature:Q", title="Feature Index"),
            y=alt.Y("Correlation:Q", title="|Pearson r| with Label",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.Significant,
                alt.value(C_SALMON),
                alt.value(C_GREY),
            ),
            tooltip=["Feature:Q", "Correlation:Q", "Significant:N"],
        )
        .properties(width=520, height=260,
                    title="Feature Correlations (Red = raw p < 0.05)")
    )

    mo.vstack([
        mo.ui.altair_chart(scatter),
        mo.callout(mo.md(f"**Regime: {regime}** — {regime_desc}"), kind=regime_kind),
        mo.hstack([
            mo.stat(value=str(raw_hits),  label="Raw Hits (p < 0.05)"),
            mo.stat(value=str(bonf_hits), label="Bonferroni Survivors"),
            mo.stat(value=regime,         label="Regime"),
        ], gap=1),
    ], gap=1)
    return


@app.cell(hide_code=True)
def cell_checklist_intro():
    mo.md(r"""
    ## ✅ Act VII — The Practical Guardrails Checklist

    The paper distills its framework into five concrete steps.
    Work through each one before publishing an interpretability finding.
    """)
    return


@app.cell
def cell_checklist():
    checks = mo.ui.array([
        mo.ui.checkbox(label="1. Define the null: Use random reinitialisation or label shuffling."),
        mo.ui.checkbox(label="2. Match search budget: Compare best result against best of null search."),
        mo.ui.checkbox(label="3. Report p̂ via Equation 4 — not just raw correlation or accuracy."),
        mo.ui.checkbox(label="4. Quantify uncertainty: Bootstrap confidence intervals on T_obs."),
        mo.ui.checkbox(label="5. Check robustness: Findings should hold across seeds and subsets."),
    ])
    return (checks,)


@app.cell(hide_code=True)
def cell_checklist_render(checks):
    n_checked = sum(1 for c in checks.value if c)
    all_done  = n_checked == 5

    cert = mo.callout(
        mo.md(
            "### 🏅 Certificate of Reproducibility\n\n"
            "All five guardrails satisfied. Your interpretability finding "
            "has earned the right to be believed."
        ),
        kind="success",
    ) if all_done else mo.callout(
        mo.md(
            f"**{n_checked}/5 guardrails satisfied.** "
            "Keep going — each unchecked box is a potential dead salmon."
        ),
        kind="warn" if n_checked < 3 else "info",
    )

    mo.vstack([checks, cert], gap=1)
    return


@app.cell(hide_code=True)
def cell_conclusion():
    mo.md(r"""
    ## 🎓 Conclusion

    ### The Core Lesson

    > *"If a frozen random network can produce your interpretability story
    > under the same search procedure, the story has not yet earned belief."*

    The paper does not say "interpretability is impossible."
    It says: **trust an explanation only after it beats the same
    search process under the correct null and uncertainty checks.**

    ### Three Results from the Paper

    | Experiment | Naive Result | Corrected Result |
    |---|---|---|
    | Sentiment / BERT | All layers look significant | Early layers **fail** the null test |
    | Syntax (POS) / BERT | All layers significant | **Middle layers survive** — genuine |
    | World Models / Pythia | Geography grows layer by layer | Signal real but mostly in embeddings |

    ### References

    - **Main Paper:** *The Dead Salmons of AI Interpretability*, AlphaXiv 2512.18792 (2025)
    - Bennett et al. — Neural correlates of interspecies perspective taking in the post-mortem Atlantic Salmon, 2009
    - Ioannidis — Why Most Published Research Findings Are False, *PLoS Medicine*, 2005
    - Gurnee & Tegmark — Language Models Represent Space and Time, 2024
    - Neel Nanda et al. — Open Problems in Mechanistic Interpretability, 2025
    - Kim et al. — TCAV: Quantitative Testing with Concept Activation Vectors, ICML 2018
    - Wang et al. — Interpretability in the Wild: IOI Circuit in GPT-2 Small, 2022
    """)
    return


if __name__ == "__main__":
    app.run()
