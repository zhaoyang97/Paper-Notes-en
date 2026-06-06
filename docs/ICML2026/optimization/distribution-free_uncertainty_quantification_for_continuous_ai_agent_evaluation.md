---
title: >-
  [Paper Note] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation
description: >-
  [ICML 2026][Optimization][conformal prediction] This paper proposes the AgentPulse framework, which combines split conformal, adaptive conformal inference (ACI), Mondrian conformal…
tags:
  - "ICML 2026"
  - "Optimization"
  - "conformal prediction"
  - "ACI"
  - "agent evaluation"
  - "FDR"
  - "Mondrian"
date: 2026-05-08
content_hash: bfcd35b0b281929a
---

# Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation

**Conference**: ICML 2026  
**arXiv**: [2605.19779](https://arxiv.org/abs/2605.19779)  
**Code**: TBD  
**Area**: LLM Evaluation / Uncertainty Quantification  
**Keywords**: conformal prediction, ACI, agent evaluation, FDR, Mondrian

## TL;DR
This paper proposes the AgentPulse framework, which combines split conformal, adaptive conformal inference (ACI), Mondrian conformal, and BH-FDR to provide distribution-free coverage guarantees, uncertainty bounds for composite pipelines, and a ranking-based rejection mechanism with FDR control for the continuous scoring of 50 AI agents, treating "uncertainty measurement" as a first-class output of evaluation.

## Background & Motivation

**Background**: Benchmarks such as SWE-bench, GAIA, WebArena, and TAU-bench compress agent performance into a static point score, while Chatbot Arena uses bootstrap to assign confidence intervals to Elo scores. The community typically treats these point estimates as "definitive" and ranks models accordingly.

**Limitations of Prior Work**: Three types of uncertainty exist in actual evaluation that no existing tools simultaneously cover: (i) measurement heterogeneity—inconsistency in "quality" judgment across different platforms; (ii) model configuration—rankings shift with minor changes in weighting schemes; (iii) temporal non-stationarity—agents are iterative, and scores drift over time. Bootstrap only addresses single signal sources and relies on the representativeness of empirical distributions; parametric intervals assume Gaussian residuals, which are immediately violated by agent release events.

**Key Challenge**: Evaluation subjects satisfy neither iid (due to iteration) nor single-source signal (involving benchmark/adoption/sentiment/ecosystem factors). Furthermore, they require massive pairwise comparisons at the leaderboard scale. No single statistical tool can simultaneously provide guarantees across the four axes of "distribution-free + drift-adaptive + conditional coverage + multiple testing."

**Goal**: To provide for continuous evaluation: (a) finite-sample coverage guarantees for single-agent prediction intervals; (b) composite error bounds for multi-agent pipelines; (c) pairwise comparisons with controlled false ranking rates + leaderboard-level FDR control; (d) conditional coverage (per-agent) rather than just marginal coverage.

**Key Insight**: Migrate the suite of distribution-free conformal prediction tools to agent evaluation, mapping the four specific challenges to existing extensions—ACI for release shocks, Mondrian for per-agent conditional coverage, independent/worst-case bounds for pipeline propagation, and BH for leaderboards.

**Core Idea**: Replace bootstrap/parametric intervals with a combination of "split conformal + ACI + Mondrian + BH-FDR" to attach distribution-free, drift-adaptive, conditional coverage, and multiple testing guarantees to a continuous evaluation pipeline.

## Method

### Overall Architecture
AgentPulse collects signals from 19 data sources hourly and calculates a four-factor composite score for 50 agents:
$\mathrm{AP}(a)=0.35\,B(a)+0.25\,A(a)+0.20\,S(a)+0.20\,E(a)$, where $B$ is benchmark (SWE-bench/GAIA/WebArena/HumanEval+/TAU-bench, using a neutral prior of 0.5 for missing data), $A$ is a log-normalized combination of GitHub stars/package downloads/VSCode installs, $S$ is sentiment fusion via VADER/TextBlob/FinBERT/DistilBERT-SST2 across 9 platforms ($\kappa{=}0.81$ on 200 calibration texts), and $E$ is contributor depth + issue closure rate + release freshness. Prediction uses mean reversion $\widehat{\mathrm{AP}}_{t+h}(a)=\mathrm{AP}_t(a)+\lambda(\overline{\mathrm{AP}}-\mathrm{AP}_t(a))\cdot h$ with $\lambda{=}0.003$ (ADF test rejects unit root for 42/50 agents). The uncertainty side is supported by the following three key designs.

### Key Designs

1.  **Split Conformal + ACI Dual-Layer Coverage**:
    *   **Function**: Provides a finite-sample coverage guarantee $\Pr[\mathrm{AP}_{t+h}\in C_{1-\alpha}]\geq 1-\alpha$ for a single agent's $h$-hour prediction, adaptively adjusting for drift events like agent releases, weekends, or holidays.
    *   **Mechanism**: Splits historical scores 70/30 into train/calibration sets; calculates the $\lceil(1-\alpha)(n_{\text{cal}}+1)\rceil/n_{\text{cal}}$ quantile $\hat q_{1-\alpha}$ of non-conformity scores $R_i=|\mathrm{AP}_{t_i+h}-\widehat{\mathrm{AP}}_{t_i+h}|$. The interval is $\widehat{\mathrm{AP}}_{t+h}\pm\hat q_{1-\alpha}$. ACI adjusts miscoverage online at each step via $\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathrm{err}_t)$, automatically widening the interval by 35% within 6 hours of a Windsurf release and contracting it after 48 hours.
    *   **Design Motivation**: Both parametric and bootstrap methods assume stable distributions and suffer from under-coverage immediately following release events; conformal prediction provides distribution-free guarantees, while ACI further addresses drift in "non-exchangeable" scenarios.

2.  **Mondrian Conditional Coverage + cross-source σ Stratification**:
    *   **Function**: Upgrades "average 80% coverage" to "near 80% coverage for every agent," preventing volatile agents from being masked by the mean.
    *   **Mechanism**: Uses cross-source divergence $\sigma_{\text{cross}}(a)=\mathrm{std}(\{s_p(a)\})$ as a stratification variable, grouping agents into $\sigma_{\text{cross}}<0.04$ (stable, $n=35$) and $\geq 0.04$ (volatile, $n=15$). Each group calibrates $\hat q_{1-\alpha}$ independently; the volatile group receives wider intervals to compensate for heavy-tailed residuals.
    *   **Design Motivation**: Under standard conformal, the volatile group's average coverage was only 64.6% (11 out of 15 below 75%), severely violating the "80% marginal coverage" promise. Mondrian restores this group to 80.4% through group-conditional guarantees at the cost of 22% more interval width, accurately reflecting their intrinsic uncertainty.

3.  **Pipeline Bounds + BH-FDR Controlled Rejection**:
    *   **Function**: Synthesizes single-agent uncertainty into multi-stage $a_1\to a_2$ pipelines and controls the false ranking rate across 1225 pairwise comparisons on the leaderboard.
    *   **Mechanism**: For pipelines, it provides an independent bound $\sigma_{\text{pipeline}}=\sqrt{\sigma_1^2+\sigma_2^2}$ and a worst-case bound $\sigma_1+\sigma_2$. Simulations at $\rho\in[-0.5,0.9]$ verify the ground truth falls between them ($\rho>0$ is generally true; independent bounds are anti-conservative for $\rho<0$, so worst-case is suggested for safety). For comparisons, it calculates conformal intervals for the difference $\Delta_{ab}$ and rejects if $0\in C_\Delta^{1-\alpha}$. For the leaderboard, it applies BH to adjust conformal p-values for each pair at $q{=}0.20$ for FDR correction.
    *   **Design Motivation**: Summing stages in pipeline scenarios is overly conservative; the two boundaries mark the extremes of "independent" and "fully correlated" for user selection. After BH correction, "confidently ranked pairs" on the leaderboard decreased from 69% to 50%, but the upper bound on the proportion of misranked pairs is guaranteed at 20% rather than accumulating pair-by-pair.

### Loss & Training
This paper targets unsupervised learning; all "models" are statistical estimators. The mean reversion predictor $\lambda$ is calibrated under ADF tests, conformal quantiles are taken empirically from the calibration set, the ACI step size $\gamma$ updates $\alpha_t$ online, and EK-FAC-like pre-statistics are computed once on the reference dataset. Dirichlet weight sensitivity was evaluated for ranking stability using 1000 samples from $\mathrm{Dir}(3.5,2.5,2.0,2.0)$.

## Key Experimental Results

### Main Results
Conformal vs. parametric coverage comparison at a 24-hour horizon; conformal calibration error is $<0.02$, while parametric is systematically over-conservative:

| Horizon | Parametric Cov. | Parametric Width | Conformal Cov. | Conformal Width |
|---------|------------|--------------|----------------|-----------------|
| 1h | 94% | 0.021 | 82% | 0.016 |
| 6h | 89% | 0.051 | 83% | 0.044 |
| 24h | 83% | 0.103 | 81% | 0.098 |
| 48h | 80% | 0.145 | 80% | 0.152 |
| 72h | 78% | 0.178 | 79% | 0.195 |

At 24h, bootstrap coverage is 84% / width 0.108, but drops to 74% (under-coverage) at 72h, whereas conformal maintains 79%. ACI widens width by 35% within 6h of a release event and contracts it after 48h, with intervals 8% wider than split conformal during stable periods.

### Ablation Study
Coverage repair and weight/multiple testing perturbations:

| Configuration | Key Metric | Description |
|------|---------|------|
| Standard conformal (volatile subset) | 64.6% Cov. | 11/15 below 75%, 15pp off 80% promise |
| Mondrian conformal (volatile subset) | 80.4% Cov. | Only 2/15 below 75%, width +22% |
| Dirichlet weight sampling ($k{=}2$) | $\tau{=}0.52$ | Ranking stability significantly drops under near-uniform priors |
| Dirichlet weight sampling ($k{=}10$) | $\tau{=}0.80$ | Default configuration, rank stable |
| Dirichlet weight sampling ($k{=}50$) | $\tau{=}0.97$ | Concentrated prior, rank nearly unchanged |
| 1225 pairs uncorrected ($\alpha{=}0.20$) | 69% confident | per-pair error of 20% accumulates on leaderboard |
| 1225 pairs BH-FDR ($q{=}0.20$) | 50% confident | Upper bound of 20% leaderboard misranking rate |

### Key Findings
*   The axis of greatest error is not horizon but cross-source $\sigma$: coverage for volatile agents drops to 68% at 72h, while stable agents remain $\geq 79\%$ across all horizons; thus, Mondrian is more effective than simply widening intervals.
*   A B+S sub-composite (excluding GitHub signals) predicts GitHub stars at $\rho_s{=}0.52$ ($p<0.01$) for $n=35$, showing sentiment is not merely a proxy for GitHub.
*   Aspect-level analysis reveals masked heterogeneity: Devin is +0.11 in code quality but −0.12 in reliability; average scores would cancel out these positive and negative variances.
*   Correlations between B/A and A/E factors are 0.05 / 0.61 (demand vs. supply), respectively, indicating the four factors are complementary rather than redundant.

## Highlights & Insights
*   Mapping the conformal toolbox to "four uncertainty challenges" (ACI ↔ release shift, Mondrian ↔ per-agent coverage, independent/worst-case ↔ pipeline, BH ↔ leaderboard) provides a clear "UQ selection checklist" for evaluation.
*   The cross-platform standard deviation $\sigma_{\text{cross}}$ serves a dual role as both a "volatility metric" and a "Mondrian stratification variable."
*   The bootstrap comparison exposes the failure mode of existing community UQ—under-coverage at 74% at 72h under non-stationarity—proving that the "representativeness of resampling" assumption is invalid in agent evaluation.
*   Treats "rejection" as a first-class mechanism: rather than forcing a rank of 21st vs. 22nd, it is better to label them "indistinguishable," which aligns more closely with the actual value of a leaderboard.

## Limitations & Future Work
*   ACI convergence requires bounded shift; true paradigm shifts (e.g., total backbone replacement) might violate assumptions.
*   For pipeline bounds when $\rho<0$, independent bounds are anti-conservative; the paper suggests worst-case bounds for safety but lacks an automatic selection rule.
*   BH-FDR assumes independence or positive regression dependence between test statistics; this premise was not explicitly verified for the 1225 pairs.
*   The efficacy of ablation for $n=11$ is only 0.38; certain "no effect" conclusions require retesting with larger samples.
*   Limited to English NLP signals; closed-source agents lack adoption signals; extensions like e-processes for "anytime-valid" stopping were mentioned as future work but not implemented.

## Related Work & Insights
*   **vs Chatbot Arena (bootstrap CI)**: They perform resampling on a single preference signal, whereas this work uses multi-source four-factor scores with distribution-free guarantees; conformal maintains 79% coverage at 72h while bootstrap drops to 74%.
*   **vs TrueSkill**: Bayesian skill models provide posterior distributions, whereas this work provides finite-sample frequentist coverage without assuming normality or distribution families.
*   **vs LiveBench**: LiveBench addresses benchmark obsolescence but provides no evaluation CI; this paper incorporates "benchmark scores" as one of the four factors in a conformal interval.
*   **vs Ortiz-Jimenez / Yoshida (task arithmetic UQ)**: That line of research performs UQ in the model weight space; this paper performs UQ in the evaluation signal space, making them complementary.

## Rating
*   Novelty: ⭐⭐⭐⭐ The tools are standard, but the specific mapping to four challenges and the use of $\sigma_{\text{cross}}$ as a Mondrian variable are novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 50 agents × 18 signals × multiple horizons × multiple coverage targets × Dirichlet weight sweeping, with circularity control in validation.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure; the "why" for each tool is well-defined; Table 1 explicitly contrasts conformal/parametric trade-offs.
*   Value: ⭐⭐⭐⭐ Provides a complete engineering template for how agent leaderboards should honestly report errors, which can be directly forked by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] RACO: Reward-free Alignment for Conflicting Objectives](reward-free_alignment_for_conflicting_objectives.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[AAAI 2026\] Parametrized Multi-Agent Routing via Deep Attention Models](../../AAAI2026/optimization/parametrized_multi-agent_routing_via_deep_attention_models.md)

</div>

<!-- RELATED:END -->
