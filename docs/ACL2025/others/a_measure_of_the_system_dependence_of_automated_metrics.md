---
title: >-
  [Paper Note] A Measure of the System Dependence of Automated Metrics
description: >-
  [ACL 2025][Automated Metrics] Points out the ignored "system dependence" issue in machine translation automated evaluation metrics: the same metric score corresponds to different human ratings for different translation systems. The paper proposes the SysDep metric to quantify this effect, revealing that even the best WMT23 metric, XCOMET, exhibits severe system dependence that leads to incorrect rankings.
tags:
  - "ACL 2025"
  - "Automated Metrics"
  - "System Dependence"
  - "Machine Translation"
  - "Isotonic Regression"
  - "Evaluation"
date: 2026-05-08
content_hash: ad9b2e9fec67f699
---

# A Measure of the System Dependence of Automated Metrics

**Conference**: ACL 2025  
**arXiv**: [2412.03152](https://arxiv.org/abs/2412.03152)  
**Code**: None (core code provided in the appendix)  
**Area**: NLP Evaluation / Machine Translation  
**Keywords**: Automated Metrics, System Dependence, Machine Translation, Isotonic Regression, Evaluation

## TL;DR

Points out the ignored "system dependence" issue in machine translation automated evaluation metrics: the same metric score corresponds to different human ratings for different translation systems. The paper proposes the SysDep metric to quantify this effect, revealing that even the best WMT23 metric, XCOMET, exhibits severe system dependence that leads to incorrect rankings.

## Background & Motivation

The goal of automated evaluation metrics is to replace expensive and time-consuming human evaluations. The performance of current evaluation metrics is typically measured through two aspects:

**Segment-level correlation**: Whether there is a monotonic relationship between metric scores and human ratings.

**System-level ranking**: Whether the metric can reproduce system rankings consistent with human judgments.

However, the authors point out that such evaluations are **insufficient** because they ignore a core requirement: **metrics should treat all evaluated systems equally**.

Key observation (taking XCOMET on WMT23 zh-en as an example):
- An XCOMET score of 0.7 corresponds to a Human-MQM score of approximately -5.2 for the best system, Lan-BridgeMT.
- The same XCOMET score of 0.7 corresponds to a Human-MQM score of approximately -10.2 for the worst system, NLLB-Greedy.
- In other words, the same metric score represents completely different translation quality across different systems.

This is akin to **"the length of a ruler changing with the object being measured"**—a good measurement tool should not behave this way.

## Method

### Overall Architecture

Given $K$ translation systems $\pi_1, ..., \pi_K$, each system has human ratings $h_k^{(j)}$ and metric scores $m_k^{(j)}$ on the test set.

Core derivation: The expected human rating for system $\pi_k$ can be expressed as:

$$\mathbb{E}[h_k] = \mathbb{E}_{p_k(m)}[\mathbb{E}_{p_k(h)}[h|m]]$$

where $f_k(m) = \mathbb{E}_{p_k(h)}[h|m]$ is the **system-specific** conditional expectation function—mapping metric scores to expected human ratings.

**Ideal scenario**: There exists a global function $f_G = f_1 = ... = f_K$, meaning all systems share the same mapping relationship.
**Real scenario**: $f_k$ varies across systems $\rightarrow$ leading to **system dependence**.

### Key Designs

**1. Expected Deviation (ED)**

Measures the discrepancy between the global function $f_G$ and the system-specific function $f_k$, i.e., the extent to which a system is overestimated or underestimated:

$$\text{ED}(k) = \frac{1}{N}\sum_{j=1}^N f_G(m_k^{(j)}) - \frac{1}{N}\sum_{j=1}^N f_k(m_k^{(j)}) = \mu_k^G - \mu_k^H$$

- ED > 0: The system is **overestimated** by the metric.
- ED < 0: The system is **underestimated** by the metric.

**2. SysDep Measure**

The system dependence score is defined as the range of ED (the difference between the maximum overestimation and the maximum underestimation):

$$\text{SysDep}(\mathcal{M}) = \max_{\pi_k} \text{ED}(k) - \min_{\pi_k} \text{ED}(k)$$

A larger SysDep value indicates stronger system dependence of the metric, resulting in a higher risk of ranking errors.

**3. Estimation Method**

Uses **Isotonic Regression (IR)** to estimate $f_k$ and $f_G$ (a non-parametric regression that guarantees monotonicity):
- Fits $\hat{f}_k$ for each system's paired data.
- Aggregates data from all systems to fit $\hat{f}_G$.
- Uses $B=200$ bootstrap samples to estimate confidence intervals.

### Loss & Training

No training process involved. IR minimizes $\sum_j (\hat{f}_k(m_k^{(j)}) - h_k^{(j)})^2$, constraining $\hat{f}_k$ to be a monotonic function.

## Key Experimental Results

### Main Results

**System dependence analysis of XCOMET on WMT23 zh-en** (15 translation systems):

| System | Human Rank | Metric Rank | ED |
|------|---------|---------|------|
| Lan-BridgeMT | 1 | 2 | **-0.820** |
| GPT4-5shot | 2 | 1 | -0.494 |
| HW-TSC | 5 | 3 | +0.318 |
| NLLB-Greedy | 15 | 12 | **+1.996** |
| ANVITA | 13 | 13 | +1.475 |

- **SysDep = 2.816** (highest ED 1.996 - lowest ED -0.820)
- The best system, Lan-BridgeMT, is **underestimated** (ED = -0.820), dropping its rank from 1st to 2nd.
- The worst system, NLLB-Greedy, is **overestimated** (ED = +1.996), rising its rank from 15th to 12th.
- Systems at the bottom of the ranking are generally overestimated, while systems at the top are generally underestimated.

### Ablation Study

Extensions to other language pairs and metrics in the appendix:
- Similar system dependence is observed on en-de and he-en language pairs.
- GEMBA-MQM (a reference-free metric based on LLM prompting) also exhibits system dependence.
- Validates that the bias stems from systematic differences rather than random noise by partitioning ratings of a single system.

### Key Findings

1. **High segment-level correlation does not guarantee good system rankings**: Even though XCOMET has a high segment-level correlation of 0.65, it produces multiple system ranking errors.
2. **System dependence is systematic**: It is not random noise, but rather systemic biases of the metric toward different types of translation systems.
3. **The best and worst systems are affected the most**: The best systems tend to be underestimated, while the worst systems tend to be overestimated.
4. **Ranking errors represent an interaction between ED differences and score intervals**: Even if the absolute value of ED is small, ranking reversals can occur if system scores are close.

## Highlights & Insights

1. The metaphor **"the ruler should not expand with the object being measured"** is intuitive and powerful, clearly conveying the core position of the paper.
2. **Exposes a blind spot in the evaluation field**: Prior work heavily focuses on optimizing segment-level correlation, ignoring the more fundamental issue of fairness.
3. **The formal derivation** is rigorous and concise: from the decomposition of conditional expectations to the definitions of ED and SysDep, the logic is highly self-consistent.
4. **Analogy to calibration error**: ED is analogous to Expected Calibration Error, providing a theoretical bridge for cross-domain transfer.

## Limitations & Future Work

1. **Based only on WMT23 data**: Requires larger-scale and multi-domain validation.
2. **Only measures the problem without proposing a solution**: How to develop metrics with low SysDep remains an open question.
3. **IR estimation is unstable when data is limited**: Especially when some systems have few paired ratings.
4. **Scope of applicability**: Currently tailored majorly for MT; generalization to other NLG tasks (summarization, dialogue) remains to be verified.
5. Only scalar-based metrics are discussed, while analysis under preference rating scenarios is also crucial.

## Related Work & Insights

- **Wu and Resnick (2024)**: Calibration problems in binary prevalence estimation $\rightarrow$ the mathematical framework of this paper is directly borrowed from it.
- **Deriu et al. (2023)**: Analyzing metric system dependence using a Bayesian framework $\rightarrow$ pioneering work for the discrete rating equivalent.
- **von Däniken et al. (2024)**: FAVI-Score finds that metrics unfairly favor certain systems $\rightarrow$ this work provides a quantitative explanation.
- **Chaganty et al. (2018)**: Combines human and automated metric ratings to avoid system dependence issues.
- **Wei and Jia (2021)**: Analyzes sign errors in system rankings $\rightarrow$ SysDep quantifies the bias that leads to sign errors.

## Rating

- **Novelty**: ★★★★☆ — Proposes an important but neglected problem in the evaluation field, with a clear formalization.
- **Practicality**: ★★★☆☆ — Currently mainly serves as a diagnostic tool without providing solutions.
- **Experimental Thoroughness**: ★★★☆☆ — Evaluates on a single scenario (WMT23), but provides deep and thorough analysis.
- **Writing Quality**: ★★★★★ — A paradigm for position papers, brief and to the point, with powerful communication of core ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Analysis of Datasets, Metrics and Models in Keyphrase Generation](an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)
- [\[ACL 2025\] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICML 2025\] Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs](../../ICML2025/others/parity_requires_unified_input_dependence_and_negative_eigenvalues_in_ssms.md)

</div>

<!-- RELATED:END -->
