---
title: >-
  [Paper Note] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses
description: >-
  [ICLR2026][LLM Evaluation][Conformal Inference] This paper proposes MACI (Multi-LLM Adaptive Conformal Inference), which combines a **cumulative-product conformity score**, a multi-LLM ensemble for factuality scoring…
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "Conformal Inference"
  - "LLM Factuality"
  - "Multi-LLM Ensemble"
  - "False-Claim Filtering"
  - "Distribution-Free Guarantee"
date: 2026-05-08
content_hash: 03b95c99d9f880e2
---

# Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses

**Conference**: ICLR2026
**arXiv**: [2602.01285](https://arxiv.org/abs/2602.01285)  
**Code**: [GitHub](https://github.com/MLAI-Yonsei/MACI)  
**Area**: LLM Evaluation
**Keywords**: Conformal Inference, LLM Factuality, Multi-LLM Ensemble, False-Claim Filtering, Distribution-Free Guarantee
**Authors**: Kangjun Noh, Seongchan Lee, Ilmun Kim, Kyungwoo Song (Yonsei University & KAIST)

## TL;DR

This paper proposes MACI (Multi-LLM Adaptive Conformal Inference), which combines a **cumulative-product conformity score**, a multi-LLM ensemble for factuality scoring, and group-conditional calibration to significantly improve the retention rate of factual claims in LLM responses while strictly guaranteeing user-specified error rates.

## Background & Motivation

**LLM Hallucination**: LLMs are widely deployed in high-stakes domains such as healthcare and law, yet their responses may contain hallucinated information, necessitating statistical guarantees.

**Introduction of Conformal Inference (CI)**: CI provides distribution-free, finite-sample guarantees. Prior work (BCI, Mohri & Hashimoto 2024) applies CI to filter false claims from LLM responses by decomposing responses into atomic claims and thresholding based on factuality scores.

**BCI Is Overly Conservative**: BCI applies a single global threshold and provides only marginal coverage, which can lead to severe over- or under-coverage across subgroups. Its conformity score relies solely on the worst single claim score, making it highly sensitive to estimation error and causing many true claims to be incorrectly removed.

**Relaxed Guarantees in CCI**: CCI (Cherian et al., 2024) introduces adaptive threshold functions for conditional guarantees but depends on an adaptive error rate $\alpha$ that is unsuitable for high-risk scenarios. Its linear feature space also struggles to capture complex semantic group structures in LLM responses.

**Conformity Score Design Flaws**: Existing methods construct conformity scores from a single extreme claim score, ignoring the collective confidence information of all remaining claims.

**Core Problem**: To maximize the retention rate of true claims while strictly controlling group-conditional coverage.

## Method

### Overall Architecture

The MACI pipeline proceeds as follows:
1. **Claim Decomposition**: Decompose the LLM response $D = (P, C, Y)$ into a set of atomic claims $C = \{c_1, \dots, c_{|C|}\}$.
2. **Multi-LLM Scoring**: Use $M$ black-box LLMs to generate verbalized factuality scores $p_m(P, c) \in [0, 1]$ for each (prompt, claim) pair.
3. **Ensemble Optimization**: Optimize weights $w$ to obtain the ensemble score $p_{\text{ens}}(P, c; w) = \sum_{m=1}^{M} w_m p_m(P, c)$.
4. **Cumulative-Product Filtering**: Sort claims in descending order of factuality score and retain the top $K$ claims whose cumulative product remains $\ge \tau$.
5. **Group-Conditional Calibration**: Independently compute quantile threshold $\hat{Q}_{1-\alpha}^{(k)}$ for each group $k$ on the calibration set.

### Key Design 1: Product-Based Conformity Score

**Oracle Filtering Rule**: Given a permutation $\pi_i$ such that $p_i^*(c_{i,\pi_i(1)}) \ge \cdots \ge p_i^*(c_{i,\pi_i(N_i)})$, the truncation index is defined as:

$$K_i^*(\tau) = \max\left\{k \in [N_i] : \prod_{j=1}^{k} p_i^*(c_{i,\pi_i(j)}) \ge \tau \right\}$$

Unlike BCI/CCI, which use only a single extreme score, the MACI conformity score is the **cumulative product of factuality scores over all retained claims**:

$$E_i = \inf\{\tau \in [0,1] : F(\hat{p}, \tau, U_i; P_i, C_i) \subseteq A_i\}$$

This product aggregation directly reflects the joint credibility that the retained set as a whole is factual, and is more robust to estimation errors in individual claims.

### Key Design 2: Group-Conditional Calibration (Mondrian Framework)

For a grouping function $g: \mathcal{P} \times \mathcal{C} \to \{1, \dots, K\}$, thresholds are computed independently on the calibration subset $\mathcal{I}_k = \{i : g(P_i, C_i) = k\}$:

$$\hat{Q}_{1-\alpha}^{(k)} = \text{Quantile}(\{E_i : i \in \mathcal{I}_k\}, 1-\alpha)$$

**Theorem 2** proves that under the exchangeability assumption, for any group $k$:

$$\mathbb{P}\big(F_{n,\alpha}^{(k)}(P_{n+1}, C_{n+1}) \subseteq A_{n+1} \mid g(P_{n+1}, C_{n+1}) = k\big) \ge 1 - \alpha$$

### Key Design 3: Multi-LLM Ensemble Optimization

**Motivation**: Theorem 3 proves that the retention rate gap $\Delta$ is controlled by the polynomial rate of estimation error:

$$\Delta \le \mathfrak{C}' \big(\mathbb{E}[(\hat{p} - p^*)^2]\big)^{\frac{\beta}{\beta+2}}$$

That is, the smaller the MSE of the factuality score, the closer the retention rate is to the oracle.

**Optimization Objective**: Since the oracle $p^*$ is unobservable, a surrogate objective is adopted — minimizing FPR subject to the constraint $\text{TPR} \ge 1-\delta$:

$$p^\star = \arg\min_{p} \mathbb{E}[\text{FPR}(p, \tau_{p,\delta})]$$

This is implemented via a weighted ensemble of $M=3$ models (Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct, and DeepSeek-V3).

## Experiments

### Experimental Setup

- **Datasets**: MedLFQA (medical QA), WikiBio (Wikipedia biographies), ExpertQA (expert-level QA)
- **Baselines**: BCI (Mohri & Hashimoto 2024), CCI (Cherian et al. 2024)
- **Grouping Criteria**: Dataset-specific semantic groups (e.g., medical content type, page view count, question domain) plus a universal False-Claim Risk grouping
- **Target Coverage**: $1-\alpha \in \{0.80, 0.90, 0.95\}$, averaged over 30 repeated experiments

### Main Results: Coverage & Retention Rate (Selected from Table 1)

| Dataset | Method | $1{-}\alpha{=}0.80$ Cov. | Ret. | $1{-}\alpha{=}0.90$ Cov. | Ret. | $1{-}\alpha{=}0.95$ Cov. | Ret. |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MedLFQA | BCI | 0.80 ✅ | 0.06 | 0.90 ✅ | 0.02 | 0.95 ✅ | 0.01 |
| | CCI | 0.81 ✅ | 0.56 | 0.90 ✅ | 0.31 | 0.95 ✅ | 0.18 |
| | **MACI** | **0.80 ✅** | **0.71** | **0.90 ✅** | **0.50** | **0.95 ✅** | **0.30** |
| WikiBio | BCI | 0.81 ✅ | 0.02 | 0.90 ✅ | 0.01 | 0.95 ✅ | 0.01 |
| | CCI | 0.79 ✅ | 0.19 | 0.89 ✅ | 0.11 | 0.93 ❌ | 0.06 |
| | **MACI** | **0.81 ✅** | **0.43** | **0.90 ✅** | **0.25** | **0.95 ✅** | **0.13** |
| ExpertQA | BCI | 0.91 ❌ | 0.13 | 0.91 ✅ | 0.13 | 0.91 ❌ | 0.13 |
| | CCI | 0.85 ❌ | 0.18 | 0.85 ❌ | 0.17 | 0.85 ❌ | 0.17 |
| | **MACI** | **0.80 ✅** | **0.45** | **0.90 ✅** | **0.15** | **0.95 ✅** | **0.10** |

**Key Findings**:
- MACI achieves the target coverage on nearly all groups while substantially outperforming baselines in retention rate.
- BCI exhibits extremely low retention (as low as 1%–6% on MedLFQA), confirming its excessive conservatism.
- CCI fails to achieve group-conditional coverage guarantees on WikiBio ($\alpha$=0.05) and ExpertQA.

### Ablation Study & Analysis

#### Multi-LLM Ensemble Effectiveness (Figure 3)

| Configuration | FPR ↓ | MSE ↓ | Retention ↑ |
|------|:-----:|:-----:|:--------:|
| Single LLM | High | High | Low |
| Arithmetic mean ensemble | Medium | Medium | Medium |
| **MACI (optimized ensemble)** | **Lowest** | **Lowest** | **Highest** |

- Large Jaccard distances between individual LLMs on false-claim detection indicate complementary error patterns, validating the ensemble design.
- Improvements in FPR align consistently with improvements in MSE, confirming that the surrogate objective is well-aligned with the oracle objective.

#### Computational Cost (Table 3, WikiBio, 500 samples)

| Stage | SelfCheck | FSC-KG | CCI | **MACI** |
|------|:---------:|:------:|:---:|:--------:|
| Scoring (s/sample) | 3.25 | 19.30 | 3.25 | **1.20** |
| Calibration (s) | — | — | 10.33 | **3.24** |
| Total time (s) | — | — | 1643.91 | **598.98** |

MACI requires only lightweight per-sample scoring and calibration, with total runtime reduced to **36%** of CCI.

#### Covariate Shift (Table 2, MACI-DRE)

Under a covariate shift scenario on MedLFQA where calibration and test distributions differ, MACI-DRE reweights the calibration set via density ratio estimation, effectively mitigating group-level coverage bias while maintaining comparable retention rates.

## Highlights & Insights

- **Product-Based Conformity Score**: This is the first work to model document-level filtering as a cumulative product of claim scores, which is more robust than extreme-value approaches and constitutes the central methodological contribution.
- **First Theoretical Analysis of Retention Rate**: Theorem 3 establishes a quantitative relationship between the oracle–estimator gap and the retention of true claims, providing theoretical motivation for the ensemble design.
- **Plug-and-Play**: MACI requires only per-claim scalar scores and can serve as a post-processing filter for any LLM generator.
- **Practical Efficiency**: MACI achieves the lowest total runtime among compared methods, making it suitable for real-time deployment.

## Limitations & Future Work

- **Group Definition Requires Prior Knowledge**: The grouping function $g$ must be manually specified (e.g., medical content type), which may be non-trivial in unknown domains.
- **Calibration Set Size Requirements**: Group-conditional calibration requires sufficient calibration samples per group ($n_k$); thresholds become conservative when group-level samples are scarce.
- **Low Retention on ExpertQA**: When datasets are noisy and the proportion of false claims is high (e.g., ExpertQA), retention rates remain limited (as low as 10% at $\alpha=0.05$).
- **Covariate Shift Handling Is Optional Post-Processing**: MACI-DRE requires an additional density ratio estimation step, increasing system complexity.
- **Dependence on Factuality Scorer Quality**: Theoretically, retention rate is bounded by the MSE between $\hat{p}$ and $p^*$; if all base LLMs err in the same direction, ensemble gains are limited.

## Related Work & Insights

- **BCI** (Mohri & Hashimoto, 2024): The first work to apply CI to LLM factuality filtering, but provides only marginal coverage and yields extremely low retention rates.
- **CCI** (Cherian et al., 2024): Introduces conditional CI with adaptive $\alpha$ to improve retention, but its linear threshold function struggles to capture complex semantic grouping, and the adaptive $\alpha$ is unsuitable for high-risk scenarios.
- **Multicalibration / Mondrian CP** (Jung et al., 2023; Liu & Wu, 2025): Provide multi-group or multi-valid coverage guarantees but tend to be conservative with low retention rates.
- **RAG-Augmented CI** (Feng et al., 2025): Transfers the CI guarantee to an external retrieval component, fundamentally changing the object of the guarantee.
- **Sampling Consistency Methods** (SelfCheck, FSC-KG): Lack rigorous statistical guarantees and incur high computational costs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of cumulative-product conformity score, theoretical retention rate analysis, and multi-LLM ensemble optimization is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple grouping criteria, ablation studies, runtime analysis, and covariate shift experiments provide comprehensive empirical support.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear, motivation is well-articulated, and the paper is well-structured.
- **Value**: ⭐⭐⭐⭐ — Offers a practical and theoretically grounded solution for reliable LLM deployment in high-stakes domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](../../ICML2026/llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)

</div>

<!-- RELATED:END -->
