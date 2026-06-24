---
title: >-
  [Paper Note] Investigating Non-Transitivity in LLM-as-a-Judge
description: >-
  [ICML 2025 Spotlight][Dialogue Systems][LLM Evaluation] Reveals the **non-transitivity** problem in LLM-as-a-Judge preferences (where A > B and B > C do not guarantee A > C), demonstrating that fixed-baseline rankings are highly unreliable, and introducing a Round-Robin Bradley-Terry ranking paradigm alongside an efficient Swim tournament strategy.
tags:
  - "ICML 2025 Spotlight"
  - "Dialogue Systems"
  - "LLM Evaluation"
  - "Non-transitivity"
  - "Bradley-Terry Model"
  - "Tournament Ranking"
  - "Position Bias"
date: 2026-05-08
content_hash: 274e6ccdb0076c22
---

# Investigating Non-Transitivity in LLM-as-a-Judge

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2502.14074](https://arxiv.org/abs/2502.14074)  
**Code**: [yix8/llm-nontransitivity](https://github.com/yix8/llm-nontransitivity)  
**Area**: Dialogue Systems  
**Keywords**: LLM Evaluation, Non-transitivity, Bradley-Terry Model, Tournament Ranking, Position Bias

## TL;DR

Reveals the **non-transitivity** problem in LLM-as-a-Judge preferences (where A > B and B > C do not guarantee A > C), demonstrating that fixed-baseline rankings are highly unreliable, and introducing a Round-Robin Bradley-Terry ranking paradigm alongside an efficient Swim tournament strategy.

## Background & Motivation

- **Popularity of LLM-as-a-Judge**: Frameworks like AlpacaEval and Arena-Hard evaluate LLMs by generating pairwise comparisons against a fixed baseline.
- **Implicit Transitivity Assumption**: These setups assume transitive preferences (A > B and B > C implies A > C), which remains untested.
- **Issues of Non-Transitivity**: Cyclic preferences ("rock-paper-scissors") make rankings highly baseline-dependent, producing contradictory outcomes.
- **Core Problem**: To what degree do LLM judges exhibit non-transitivity? How does this impact rank reliability, and how can we design robust alternatives?

## Method

### Overall Architecture

The approach consists of: (1) metrics to measure non-transitivity; (2) tournament-based ranking; (3) the Swim matchmaking strategy.

**Setup**: Evaluates 20 models from AlpacaEval/Chatbot Arena using GPT-4-Turbo and GPT-3.5-Turbo as judges. Position switching is utilized to minimize position bias.

### Key Designs

#### 1. Non-Transitivity Metrics

**Proportion of Hard Non-transitivity (PNT)**: Measures the ratio of instructions violating transitivity for a model triplet $(m_A, m_B, m_C)$:

$$\text{PNT} = \frac{1}{|\mathcal{I}|} \sum_{I_i \in \mathcal{I}} \mathbb{1}_{\text{non-trans.}}(m_A, m_B, m_C \mid m_J, I_i)$$

**Limitations**: PNT is a binary indicator and fails to capture soft deviations.

**Soft Non-Transitivity Deviation (SNTD)**: Uses Jensen-Shannon Divergence to quantify deviations between observed win rates $\phi$ and predicted transitive win rates $\hat{\phi}$:

$$\text{SNTD}(m_A, m_B, m_C | I_i) = \frac{1}{3} \times \mathbb{E}\left[\sum_{\text{three pairs}} \text{JSD}(\phi \| \hat{\phi})\right]$$

**Expected Win Rate Estimation**: Built from the Bradley-Terry model using two observed pairwise margins:

$$\hat{\phi}(o_A, o_B \mid m_J, I_i) = \frac{1}{1 + e^{-(s_{AC} - s_{BC})}}$$

where $s_{AB} = \ln\frac{\phi_{AB}}{1-\phi_{AB}}$ is the estimated latent quality scale difference.

#### 2. Four-Scenarios Classification

Triplets are categorized based on capability gaps:

| Scenario | Relationship | Meaning |
|------|------|------|
| LL (Lead & Lead) | $m_A \gg m_B \gg m_C$ | Large performance gaps between all |
| LM (Lead & Margin) | $m_A \gg m_B \approx m_C$ | Leader ahead, remaining two close |
| ML (Margin & Lead) | $m_A \approx m_B \gg m_C$ | Top two close, bottom model trailing |
| MM (Margin & Margin) | $m_A \approx m_B \approx m_C$ | Substantially similar performance level |

#### 3. Tournament Ranking Methods

**Round-Robin**: Executes all possible pairwise matches and optimizes Bradley-Terry strength coefficients $\beta_i$ using maximum likelihood estimation:

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_i \sum_{j \neq i} \left[ W_{i,j} \cdot \ln\frac{1}{1+e^{(\beta_j - \beta_i)}} \right]$$

Key innovation: Employs **soft labels** (win probabilities) instead of hard binaries, where $W_{i,j} = \sum_{I_k} J(m_i \succ m_j \mid I_k)$, mapping scaling weights to Elo metrics: $\xi_i = 400 \log_{10} \beta_i$.

**Swim Tournament (Swiss-Wise Iterative Matchmaking)**: Standard Round-Robin scales at $\mathcal{O}(nm^2)$. Swim mitigates this down to $\mathcal{O}(nm\log m)$ using dynamic matching strategies.

### Loss & Training

Aims to maximize the Bradley-Terry log-likelihood with these configurations:

- **Position Switching**: Evaluates pairwise samples in forward and reverse orders to tackle position biases.
- **Length-Controlled**: Incorporates length correction weights.
- **Prompt Variations**: Explores the effects of Checklist, CoT, and tie options on transitivity.

## Key Experimental Results

### Main Results

| Method | Spearman (w/o LC) | Spearman (LC) | Kendall (w/o LC) | Kendall (LC) |
|------|:---:|:---:|:---:|:---:|
| AlpacaEval 2.0 | 81.4% | 95.0% | 63.2% | 82.1% |
| Round-Robin (Ours) | 85.4% | **96.4%** | 68.4% | **86.3%** |
| Gain | +4.0% | +1.4% | +5.2% | +4.2% |

Yields consistently higher alignment with human-determined Chatbot Arena rankings.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|------|------|
| GPT-4-Turbo Judge (MM) | PNT=8.45%, SNTD=0.143 | Highest non-transitivity when models are close |
| GPT-3.5-Turbo Judge (MM) | PNT=20.87%, SNTD=0.263 | Weak judges suffer from dramatic non-transitivity |
| GPT-4-Turbo + Swap | Non-transitivity reduces 17%-44% | Highly effective for strong judges |
| GPT-3.5-Turbo + Swap | Non-transitivity slightly increases | Counterproductive for weak judges |
| CoT Prompting | Position bias ↓, Non-transitivity ↑ | CoT reduces bias but introduces reasoning loops |
| Allow Ties | Position bias ↑, Non-transitivity ↑ | Both error metrics deteriorate |
| Checklist Prompting | Non-transitivity slightly ↓ | Inconsequential overall benefit |
| Swim vs Round-Robin | Negligible correlation loss | Achieves equivalent rank quality with $\log_2 M$ runs |

### Key Findings

1. **Ubiquitous Non-Transitivity**: Appears across both GPT-4 and GPT-3.5 judges in all testing tiers.
2. **Impact of Capability Proximity**: Non-transitivity peaks when candidate models have similar capability.
3. **Judge Competency Gap**: Weak judges (GPT-3.5) exhibit persistent and massive non-transitivity (~20% PNT).
4. **Baseline Volatility**: Absolute rankings shift significantly depending on baseline system selection.
5. **Dual Contributors**: Non-transitivity is induced by both position bias and fundamental flaws in reasoning.
6. **Soft vs. Hard Gaps**: Soft transitivity errors remain prevalent and continue to shift overall rankings.

## Highlights & Insights

- **Conceptual Novelty**: Methodically uncovers and measures critical non-transitivity flaws in auto-evaluator pipelines.
- **Well-Designed Metric (SNTD)**: Builds a robust, mathematically sound discrepancy index using Bradley-Terry predictions and JS divergence.
- **Game-Theoretic View**: Parallels LLM comparisons to complex multi-agent matchups like StarCraft or Chess.
- **Highly Scalable Swim**: Cuts benchmarking costs from linear to logarithmic scales.
- **Soft Labels Advantage**: Validates continuous margin scores as a superior training weight over hard binary options.

## Limitations & Future Work

1. **Benchmark Bounds**: Confined to AlpacaEval outputs; broader benchmark spaces need verification.
2. **Judge Homogeneity**: Evaluation is mostly restricted to OpenAI API models.
3. **Reference Arena Limitations**: Chatbot Arena rankings are treated as gold standards, but human preferences also exhibit non-transitivity.
4. **Assessment Formatting**: Pairwise structures were checked, but pointwise grading setups are unexplored.
5. **BT Model Simplifications**: The Bradley-Terry framework represents strength linearly, missing multi-dimensional feature nuances.

## Related Work & Insights

- **AlpacaEval / Arena-Hard**: Highlights the systematic limits of standard evaluation benchmarks.
- **Chatbot Arena**: The gold standard of human preference representation, which suffers from low scalability.
- **Balduzzi et al. & Czarnecki et al.**: Applying non-transitive "spinning top" concepts from game theory directly to automated NLP evaluations.
- **Insight**: Future evaluation frameworks must adopt multi-baseline benchmarks or tournament topologies (e.g., Swim) to combat transitivity errors.

## Rating

| Dimension | Score | Rationale |
|------|:---:|------|
| Novelty | ⭐⭐⭐ | Spotlights a critical and overlooked flaw in LLM evaluators. |
| Theory Depth | ⭐⭐⭐ | SNTD formulation is backed by sound Bradley-Terry modelling. |
| Experimental Thoroughness | ⭐⭐⭐ | Demonstrates strong analysis on 20 models across multiple judges. |
| Value | ⭐⭐⭐⭐ | Swim provides a practical tool for scaling up leaderboard tasks. |
| Writing Quality | ⭐⭐⭐ | Exceptionally clear framing and structure. |
| Overall | ⭐⭐⭐ | A highly valuable, solid study for LLM evaluation. |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[NeurIPS 2025\] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](../../NeurIPS2025/dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [\[NeurIPS 2025\] SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](../../NeurIPS2025/dialogue/sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](../../ACL2026/dialogue/context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[NeurIPS 2025\] HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](../../NeurIPS2025/dialogue/hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)

</div>

<!-- RELATED:END -->
