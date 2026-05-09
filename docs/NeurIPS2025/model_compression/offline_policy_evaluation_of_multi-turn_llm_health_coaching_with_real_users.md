---
title: >-
  [Paper Note] Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users
description: >-
  [NeurIPS 2025 Workshop (Multi-Turn Interactions in Large Language Models)][Model Compression][Offline Policy Evaluation] This paper conducts offline policy evaluation (OPE) on a deployed LLM health coaching system with real users. It finds that a uniformly high tool-use policy improves average reward but harms specific user subgroups. Through simulator experiments, the paper further validates that early information-gain exploration (curiosity reward) accelerates user profile identification and improves task success rates.
tags:
  - NeurIPS 2025 Workshop (Multi-Turn Interactions in Large Language Models)
  - Model Compression
  - Offline Policy Evaluation
  - LLM Health Coaching
  - Multi-Turn Dialogue
  - Personalization
  - POMDP
date: 2026-05-08
content_hash: 597b2255d00cefda
---

# Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users

**Conference**: NeurIPS 2025 Workshop (Multi-Turn Interactions in Large Language Models)
**arXiv**: [2510.17173](https://arxiv.org/abs/2510.17173)
**Code**: [GitHub](https://github.com/stevenshci/NeurIPS-MTI-LLM)
**Area**: Model Compression
**Keywords**: Offline Policy Evaluation, LLM Health Coaching, Multi-Turn Dialogue, Personalization, POMDP

## TL;DR

This paper conducts offline policy evaluation (OPE) on a deployed LLM health coaching system with real users. It finds that a uniformly high tool-use policy improves average reward but harms specific user subgroups. Through simulator experiments, the paper further validates that early information-gain exploration (curiosity reward) accelerates user profile identification and improves task success rates.

## Background & Motivation

Wearable-generated personal health data provides rich material for LLM health coaches, yet real-world deployment faces several challenges:

- **Multi-turn degradation**: User ratings decline as conversations progress (from 4.36 to 4.12).
- **Double-edged tool use**: Tool invocations exhibit high variance, with significant gaps between success and failure outcomes.
- **Population heterogeneity**: Users with different health literacy levels and self-efficacy respond differently to the same policy.
- **Evaluation difficulty**: Real-user trials are costly, necessitating offline methods for counterfactual policy comparison.

Existing LLM health application research largely relies on synthetic benchmarks and lacks systematic evaluation of multi-turn interactions with real users.

## Method

### Overall Architecture

The health coaching system is modeled as a user-conditioned POMDP, where the belief state $z_t = f_\phi(h_t, u_i, m_t)$ integrates dialogue history, user profile features, and current health metrics. Actions are decomposed into two discrete decision heads:

- **Tool head**: $\in \{\varnothing, \text{Search}, \text{Code}, \text{Email}\}$
- **Style head**: $\in \{\text{concise}, \text{detailed}\}$

### Key Designs

1. **Typed reward system**: The per-turn reward is a personalized weighted combination of three components:

$$R_i(z_t, a_t) = \alpha_i(z_t) R_{\text{user}} + \beta_i(z_t) R_{\text{tool}} + \gamma_i(z_t) R_{\text{eng}}$$

where $R_{\text{user}}$ is derived from 1–5 star ratings, $R_{\text{tool}}$ is determined by tool call success/failure (+1/−1), and $R_{\text{eng}}$ is an engagement signal based on latency and structural quality. Weights are stratified by health literacy level (low literacy: $(0.6, 0.2, 0.2)$; high literacy: $(0.3, 0.5, 0.2)$).

2. **Early information-gain reward (curiosity mechanism)**: An information-gain bonus is added during the first $K$ turns ($K=2$) to encourage reduction of uncertainty over latent user interaction types:

$$r_t^{\text{curiosity}} = \max\{0, H(p_{t-1}(y)) - H(p_t(y))\}$$

where $y$ denotes the latent interaction prototype (literacy × self-efficacy) and $p_t(y)$ is the posterior distribution. The reward weight $\lambda_t$ is active only during the initial turns and decays to zero thereafter.

3. **Offline Policy Evaluation (OPE)**: SNIPS (self-normalized importance sampling) is used to evaluate objective reward, and AIPW (augmented inverse probability weighting, doubly robust) is used to evaluate user satisfaction. Probabilistic behavior models are fitted for each decision head to approximate logged propensity scores, with importance ratio clipping at threshold $c=50$ and session-level bootstrap for confidence intervals.

### System Deployment

- Backbone model: Qwen3-235B-A22B
- Users upload Apple Health data; the system preprocesses it into daily features (sleep, HRV, VO2max, activity)
- ML models predict stress/soreness/injury risk ($R^2$ of 0.50/0.28/0.40, respectively)
- Agent tools include a code executor, web searcher, and email sender

## Key Experimental Results

### Offline Policy Evaluation Results (Deployment Logs: 7 Users, 280 Rated Turns)

| Policy | $R_{\text{obj}}$ (SNIPS) | $R_{\text{user}}$ (AIPW) | $R_{\text{total}}$ [95% CI] |
|---|---|---|---|
| NoTool | 0.328 | -0.623 | 0.044 [-0.045, 0.198] |
| AlwaysTool | 0.229 | -0.654 | **0.304** [0.001, 0.524] |
| HeuristicGated | 0.309 | -0.625 | 0.006 [-0.111, 0.174] |
| PersonalizedWeights | 0.253 | -0.656 | 0.113 [-0.016, 0.284] |

AlwaysTool achieves the highest average $R_{\text{total}}$, though with wide confidence intervals.

### Subgroup Heterogeneity by User Prototype (AlwaysTool vs. NoTool Difference)

| User Prototype | Δ Objective | Δ Satisfaction |
|---|---|---|
| High Literacy × High Self-Efficacy | +0.575 | -0.107 |
| High Literacy × Low Self-Efficacy | **+0.595** | **+0.525** |
| Low Literacy × Low Self-Efficacy | +0.165 | -0.431 |
| Low Literacy × High Self-Efficacy | **-0.315** | **-1.436** |

**Key finding**: The AlwaysTool policy is most beneficial for "High Literacy × Low Self-Efficacy" users (positive on both dimensions), but severely harmful to "Low Literacy × High Self-Efficacy" users (Δ satisfaction = −1.436), revealing subgroup-level harms masked by population averages.

### Simulator Experiments (Hidden Prototype Setting)

| Policy | Final Return | Goal Success Rate | pass@3 | Profile ID Turns↓ | Prototype Alignment |
|---|---|---|---|---|---|
| Heuristic | -2.908 | 0.515 | 0.505 | 6.315 | 0.503 |
| Personalized | -3.162 | 0.935 | 0.950 | 6.415 | 0.424 |
| Pers+Curiosity (λ=0.10) | -2.401 | 0.965 | 0.975 | **5.655** | 0.412 |
| Pers+Curiosity (λ=0.20) | **-2.329** | **0.970** | **0.980** | 5.860 | 0.410 |

The curiosity reward significantly improves goal success rate (0.935→0.970), pass@3 (0.95→0.98), and reduces profile identification time (6.41→5.7 turns), consistent with an "explore first, then personalize" strategy.

### Key Findings

- **Tool use is high-risk, high-reward**: Successful tool calls yield a mean rating of 4.08 vs. 3.58 for failed calls, a gap of +0.50.
- **Per-tool success rates**: Web search 81.6%, code execution 80.7%, email 85.7%.
- **Dialogue degradation**: Ratings decline from 4.36 in the first 5 turns to 4.12 at 15+ turns; tool usage drops from 70% (turns 5–10) to 26.3% (turns 15+).
- **ICC of only 0.016**: Only 1.6% of rating variance is attributable to between-user differences, indicating that contextual factors rather than individual characteristics drive most variation.

## Highlights & Insights

- The OPE analysis is conducted on a real deployed system rather than relying solely on simulation, giving it strong practical value.
- The discovery of "subgroup harm" is particularly significant: a policy that appears optimal on average may cause severe harm to specific subpopulations, underscoring the necessity of reporting metrics at the subgroup level.
- The paper articulates a clear "evaluate first, personalize first" pipeline: freeze the generator, learn only subgroup-aware decision heads, use typed rewards (objective + satisfaction), and always report per-prototype metrics.
- The curiosity-driven "explore then personalize" strategy is both concise and effective.

## Limitations & Future Work

- **Very small sample size**: 7 users and 280 rated turns limit the statistical generalizability of conclusions.
- Behavioral propensities were reconstructed post hoc rather than logged prospectively; Tool head calibration error (ECE = 0.157) may introduce bias.
- The simulator employs a simplified user model that may not capture the full complexity of real interactions.
- Current personalization is based solely on health literacy stratification, without incorporating self-efficacy.
- No end-to-end RL policy learning is performed; the paper proposes only a conceptual framework.

## Related Work & Insights

- Health LLMs such as PH-LLM and PHIA focus on single-turn accuracy; this paper highlights the importance of multi-turn degradation and personalization.
- Curiosity-driven exploration (Pathak et al.) is transferred from RL to the dialogue personalization setting.
- The doubly robust OPE methodology using SNIPS/AIPW offers a broadly applicable reference for offline evaluation of any LLM-based application.
- The approach of stratifying users into prototypes and examining subgroup-level effects warrants wider adoption across AI applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying OPE and curiosity mechanisms to LLM health coaching is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — Real deployment data is valuable, but the sample size limits statistical significance.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, methodology is complete, and diagnostic tables are thorough.
- **Value**: ⭐⭐⭐⭐ — The "evaluate first" framework and subgroup harm detection methodology have broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](../../ACL2026/model_compression/arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)
- [\[NeurIPS 2025\] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)
- [\[ICLR 2026\] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](../../ICLR2026/model_compression/multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)
- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](../../AAAI2026/model_compression/safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)

<!-- RELATED:END -->
