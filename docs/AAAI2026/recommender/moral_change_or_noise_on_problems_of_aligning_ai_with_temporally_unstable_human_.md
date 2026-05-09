---
title: >-
  [Paper Note] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback
description: >-
  [AAAI 2026][Recommender Systems][AI Alignment] Through a longitudinal study involving 400+ participants across 3–5 sessions in the domain of kidney transplant allocation, this paper reveals significant temporal instability in human moral preferences (6–20% response change rate) and demonstrates that such instability substantially degrades the predictive performance of AI alignment models, thereby challenging the validity of current alignment approaches that assume static preferences.
tags:
  - AAAI 2026
  - Recommender Systems
  - AI Alignment
  - Preference Instability
  - Moral Preferences
  - Temporal Dynamics
  - Kidney Allocation
date: 2026-05-08
content_hash: ab8496401adfcc0d
---

# Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback

**Conference**: AAAI 2026
**arXiv**: [2511.10032](https://arxiv.org/abs/2511.10032)
**Code**: None
**Area**: Recommender Systems / AI Alignment
**Keywords**: AI Alignment, Preference Instability, Moral Preferences, Temporal Dynamics, Kidney Allocation

## TL;DR

Through a longitudinal study involving 400+ participants across 3–5 sessions in the domain of kidney transplant allocation, this paper reveals significant temporal instability in human moral preferences (6–20% response change rate) and demonstrates that such instability substantially degrades the predictive performance of AI alignment models, thereby challenging the validity of current alignment approaches that assume static preferences.

## Background & Motivation

### The Static Assumption Dilemma in AI Alignment

Current AI alignment methods (including RLHF and related preference optimization approaches) rest on a core assumption: human preferences are **static targets** that can be collected once and for all. However:

**Preferences are dynamic**: Research in cognitive science shows that human preferences evolve over time and across contexts, encompassing value updates, shifts in cognitive biases, and fluctuations in attention.

**The moral domain is more severely affected**: Instability in movie recommendation preferences impacts only entertainment experience, whereas in high-stakes moral domains such as medical resource allocation, preference instability can lead to serious consequences.

**Lack of empirical evidence**: Although the theoretical call to attend to temporal shifts in preferences has been made, large-scale empirical data to quantify their impact is lacking.

### Core Research Questions

- To what extent do human moral preferences change over time?
- What are the mechanisms underlying preference change across different participants?
- How does preference change affect the predictive performance of AI alignment models?

## Method

### Overall Architecture

This work is fundamentally **empirical** rather than algorithmic in nature. The framework consists of: data collection → instability measurement → participant classification → AI alignment evaluation.

### Key Designs

#### 1. Experimental Design: Kidney Allocation Scenario

**Function**: Simulates a setting of kidney donor scarcity, in which participants choose between two hypothetical patients.

**Mechanism**: Each scenario describes patients using 8 features (number of children, expected life-years gained, alcohol consumption, number of violent crimes, obesity level, post-transplant work capacity, years on waiting list, rejection probability), encompassing both medical and non-medical morally relevant attributes.

**Experimental Parameters**:
- **Participants**: 1,410 recruited; 404 completed at least 3 sessions (after attention checks)
- **Per session**: 60 pairwise comparison scenarios
- **Repeated scenarios**: 6 scenarios repeated across all sessions (with 2 within-session repetitions each) for measuring consistency
- **Difficulty gradient**: U1/U2 (1 feature difference, easiest) → V1/V2 (4 feature differences) → W1/W2 (all 8 features differ, hardest)
- **Randomization**: Feature order and patient left/right position randomized each time to prevent memory effects

**Design Motivation**: Kidney allocation represents a real-world, high-stakes moral decision scenario for which AI-assisted optimization has been proposed. The carefully designed repeated scenarios enable controlled measurement of preference stability.

#### 2. Multi-Level Measurement Framework

**Function**: Captures preference instability along multiple dimensions.

**Response Stability (RS)**: For repeated scenarios, computes the frequency with which a participant selects the "dominant response":

$$\text{RS}(S;i) = \frac{\#\text{times dominant response is chosen}}{\text{total number of presentations}}$$

**Model Stability (MS)**: Logistic regression models $H_{i,j}$ are trained per session; a Cohen's κ-like metric measures agreement between model predictions across sessions:

$$\text{MS}(j_1,j_2;i) = \mathbb{E}_T\left[\frac{p^{obs}_{j_1,j_2}(T) - p^{exp}_{j_1,j_2}(T)}{1 - p^{exp}_{j_1,j_2}(T)}\right]$$

**Model Entropy**: Feature importance vectors derived via Shapley values are used to compute entropy, measuring how many features a participant relies upon for decision-making.

**Scenario Difficulty**: A Bradley-Terry model is fitted to each participant's weight vector; the absolute difference between the two patients' scores quantifies subjective difficulty.

#### 3. Four-Quadrant Participant Classification

**Function**: Participants are classified into four groups based on median splits of response stability and model stability.

| Category | Response Stability | Model Stability | N | Characteristics |
|---|---|---|---|---|
| C1 | High | High | 147 | Simple models; increasingly focused on the most important feature |
| C2 | High | Low | 49 | Similar to C1 but with higher initial entropy |
| C3 | Low | High | 55 | Complex but stable models; instability likely due to intrinsic model noise |
| C4 | Low | Low | 153 | High-entropy models with substantial feature importance drift |

**Four Preference Change Mechanisms**:
- **Mechanism 1 (C1/C2)**: Increasing reliance on the "most important" feature over time; declining model entropy; possibly reflects cognitive simplification or preference consolidation
- **Mechanism 2 (C3)**: Unstable responses but stable model; high-entropy (complex) model; instability may stem from intrinsic model noise
- **Mechanism 3 (C4)**: Active shifts in feature importance; high model drift; participants may process each scenario independently or tend toward random responding

### Loss & Training

Three AI alignment models are used to assess the impact of preference instability:
- **BT-NN**: A neural scoring function based on the Bradley-Terry framework (analogous to DPO)
- **SUP-NN**: A standard supervised neural network
- **GPT-FT**: GPT-2 fine-tuned on pooled participant data

BT-NN and SUP-NN are trained per participant (80/20 split); GPT-FT is fine-tuned on 50% of the aggregated data.

## Key Experimental Results

### Main Results: Response Instability Statistics

| Scenario Type | Feature Differences | Mean Response Stability | Notes |
|---|---|---|---|
| U1, U2 (easy) | 1 | ~0.94 | Most stable |
| V1, V2 (medium) | 4 | ~0.82 | Moderate |
| W1, W2 (hard) | 8 | ~0.80 | Least stable |

- Participants exhibit an average **response change rate of 6–20%**
- Kruskal-Wallis test: stability differences across scenario types are significant (H(5)=222.23, p<0.001)
- Model stability is negatively correlated with temporal interval (ρ=−0.12, p<0.001): longer gaps yield greater model inconsistency

### Ablation Study: AI Model Performance vs. Instability

| Participant Category | BT-NN Error Rate | SUP-NN Error Rate | GPT-FT Error Rate | Characteristics |
|---|---|---|---|---|
| C1 (doubly stable) | Lowest | Lowest | 0.03 above C2/C3 | Baseline |
| C2 (response stable, model unstable) | Moderate | Moderate | Moderate | |
| C3 (response unstable, model stable) | Moderate | Moderate | Moderate | |
| C4 (doubly unstable) | C1 + 0.16 | Similar | C1 + 0.05 | Worst |

Key statistics:
- BT-NN: C4 vs. C1 error rate higher by 0.16 (t(290)=14.5, p<0.001)
- GPT-FT: C4 vs. C1 error rate higher by 0.05 (t(289)=4.2, p<0.001)
- C4 vs. C2/C3: BT-NN higher by 0.11 (t(253)=9.1, p<0.001)

### Key Findings

1. **Temporal decay effect**: For C4 participants, models trained on session 1 data exhibit significantly increasing error rates in subsequent sessions (t=7.2, p<0.001)
2. **Response instability is not pure noise**: Mixed-effects regression reveals that response stability is significantly associated with scenario difficulty (β=−0.043, p<0.001), model complexity (β=−0.117, p<0.001), and response time (β=−0.081, p<0.05)
3. **Different types of preference change require different technical solutions**: Re-weighting approaches suit legitimate preference updates, while noise-robust optimization is more appropriate for stochastic fluctuations
4. **Population-level fine-tuning (GPT-FT) yields higher error rates across all categories**, highlighting the limitations of aligning collective preferences to individuals

## Highlights & Insights

1. **Large-scale longitudinal study design**: 404 participants × 3–5 sessions × 60 scenarios per session, substantially exceeding prior comparable studies (<50 participants)
2. **Four-quadrant classification reveals heterogeneity**: Rather than simply reporting "preference instability," the paper carefully distinguishes among different instability mechanisms
3. **Raises profound normative questions**: When preferences change significantly, which temporal snapshot should AI align to—early, late, or some aggregate? The answer depends on whether the change reflects legitimate value updating or random noise
4. **Interdisciplinary perspective**: Integrates cognitive science (preference construction theory), ethics (normative questions), and machine learning (alignment techniques)

## Limitations & Future Work

1. **Limited time span**: Coverage spans only a few days to two weeks; longer-term (months/years) preference dynamics may be substantially more pronounced
2. **Hypothetical scenarios**: Participants are laypersons rather than medical professionals engaging with real cases
3. **Insufficient causal explanation**: While distinct preference change patterns are identified, the data do not provide sufficient information to explain *why* preferences change—a fundamental limitation of choice-based preference learning
4. **No concrete technical solutions proposed**: The work is primarily diagnostic and prescriptive, leaving specific remedies to future research
5. Incorporating **rationale data** (beyond choice data alone) to understand the causes of preference change is a promising direction

## Related Work & Insights

- **RLHF/DPO family**: This paper challenges the static preference assumption underlying these methods
- **Moral Machine (Awad et al. 2018)**: A large-scale moral preference dataset, but collected at a single time point
- **Boerstler et al. 2024**: A study of preference instability in kidney allocation, but limited in scale (<50 participants)
- Implications: Alignment methods need **continual learning** mechanisms, **instability detection**, and the capacity to **distinguish types of preference change**

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel problem framing; elegant experimental design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (400+ participants × 5 sessions longitudinal study; 3 AI models; comprehensive statistical analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic; well-defined measurement framework; in-depth discussion)
- Value: ⭐⭐⭐⭐ (Illuminates fundamental challenges in AI alignment; lacks concrete solutions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](../../NeurIPS2025/recommender/empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)

</div>

<!-- RELATED:END -->
