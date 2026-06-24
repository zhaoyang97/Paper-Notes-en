---
title: >-
  [Paper Note] Performative Validity of Recourse Explanations
description: >-
  [NeurIPS 2025][Causal Inference][Algorithmic recourse] This paper formally analyzes the "performative" effects of recourse explanations — when a large number of rejected applicants act on recourse recommendations, their collective behavior induces distribution shift that renders recourse invalid after model retraining — and proves that only Improvement-based Causal Recourse (ICR), which intervenes solely on causal variables, preserves "performative validity" under broad condi…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Algorithmic recourse"
  - "performative effects"
  - "counterfactual explanations"
  - "distribution shift"
date: 2026-05-08
content_hash: 19384072b4c207e3
---

# Performative Validity of Recourse Explanations

**Conference**: NeurIPS 2025
**arXiv**: [2506.15366](https://arxiv.org/abs/2506.15366)  
**Code**: None  
**Area**: Causal Inference
**Keywords**: Algorithmic recourse, performative effects, causal inference, counterfactual explanations, distribution shift

## TL;DR

This paper formally analyzes the "performative" effects of recourse explanations — when a large number of rejected applicants act on recourse recommendations, their collective behavior induces distribution shift that renders recourse invalid after model retraining — and proves that only Improvement-based Causal Recourse (ICR), which intervenes solely on causal variables, preserves "performative validity" under broad conditions.

## Background & Motivation

### 1. State of the Field

In high-stakes machine learning decision settings (loan approvals, graduate admissions, job interview screening, etc.), rejected applicants require "recourse explanations" — guidance on how to modify their features to obtain a favorable evaluation. Existing recourse methods fall into three main categories: Counterfactual Explanations (CE), Causal Recourse (CR), and Improvement-based Causal Recourse (ICR).

### 2. Limitations of Prior Work

Prior work has focused on the effect of external distribution shifts (temporal drift, geographic variation, etc.) on recourse validity, while overlooking a fundamental phenomenon: **recourse itself causes distribution shift**. When a large number of applicants act on the same set of recourse recommendations, their collective behavior alters the data distribution, and the decision boundary shifts as the model is retrained.

### 3. Root Cause

The "performativity" of recourse gives rise to a self-defeating paradox: recourse recommendations may invalidate themselves. For instance, when a system advises rejected job applicants to increase their GitHub activity, many applicants use automated commit tools to fabricate activity; after model retraining, the model no longer trusts that feature, and applicants who followed the advice are rejected again — wasting time and effort with no benefit.

### 4. Paper Goals

To formally characterize the conditions under which recourse explanations remain valid under their own performative effects — i.e., when does recourse avoid being invalidated by the distribution shift it induces?

### 5. Starting Point

From a causal graph perspective, recourse actions are modeled as causal interventions, and the paper analyzes the conditional independence relationship between the recourse action variable $A$ and the post-recourse label $L^p$. Through d-separation analysis, two root causes of performative failure are identified: (1) the action is influenced by effect variables; and (2) the action intervenes on effect variables.

### 6. Core Idea

Performative failure of recourse fundamentally stems from dependence on non-causal (effect) variables — only ICR, which recommends interventions solely on causal variables, can guarantee performative validity.

## Method

### Overall Architecture

The paper establishes a theoretical framework for "performative recourse," consisting of the following components:

1. **Pre- to post-recourse distribution modeling**: The post-recourse applicant distribution is modeled as a mixture of the original distribution and the post-action distribution: $P(L^m, X^m) = \alpha P(L,X) + (1-\alpha)P(L^p, X^p \mid \hat{L}=0)$
2. **Structural Causal Model (SCM)**: The data-generating process is assumed to be described by an acyclic SCM, and recourse actions are causal interventions $do(a)$
3. **Recourse action variable $A$**: A dedicated action variable is introduced to model recourse decisions and recommendations as causal paths from pre-recourse features $X$ to action $A$
4. **Performative Validity**: Requires that for all accepted points $x$ and all mixture weights $\alpha$, the updated model satisfies $\hat{L}^m(x) \geq \hat{L}(x)$

### Key Designs

#### Module 1: Sufficient Conditions for Performative Validity (Proposition 5.1)

- **Function**: Establishes the equivalence "non-informative action $\Rightarrow$ performative validity"
- **Mechanism**: If observing whether an intervention was performed does not help predict the post-recourse label $L^p$ (given post-recourse features $X^p$), then the original optimal model and the post-recourse optimal model agree on all originally accepted points
- **Design Motivation**: This transforms the performative validity problem into a conditional independence test, amenable to analysis via d-separation on causal graphs

#### Module 2: Identification of Two Failure Sources (Theorem 5.2)

- **Function**: Precisely characterizes the two paths in the causal graph that lead to $A \not\perp L^p \mid X^p$
- **Mechanism**: In the augmented causal graph (containing both pre- and post-recourse variables), d-connected paths between $A$ and $Y^p$ can only pass through two critical edges: (1) $X_{de(Y)} \to A$ (effect variables influencing the action) and (2) $A \to X_E^p$ (the action intervening on effect variables)
- **Design Motivation**: Decomposes the abstract conditional independence problem into two operationalizable checking conditions

#### Module 3: Two Recovery Assumptions

**Assumption 5.4 (Noise resampling)**: If the unobserved causal influences ($U_Y$, $U_E$) are independent across pre- and post-recourse periods (e.g., daily weather, mood), then recourse methods that intervene only on causal variables guarantee performative validity (Proposition 5.5).

**Assumption 5.6 (Invertible aggregated noise)**: If the effect and noise in the structural equations can be aggregated and inverted (e.g., linear additive noise, multiplicative noise), then intervening only on causal variables guarantees performative validity even when noise is held fixed (Theorem 5.7).

#### Module 4: Comparative Analysis of Three Recourse Methods

| Method | Intervention Target | Optimization Objective | Performative Validity |
|--------|--------------------|-----------------------|----------------------|
| CE (Counterfactual Explanations) | Arbitrary features (including effect variables) | Flip prediction $\hat{L}(x')=1$ | ❌ May fail |
| CR (Causal Recourse) | Causal interventions (including effect variables) | $P(\hat{L}(X^p)=1 \mid x, do(a)) \geq t_r$ | ❌ May fail |
| ICR (Improvement-based Causal Recourse) | Causal variables only | $P(L^p=1 \mid x, do(a)) \geq t_r$ | ✅ Valid under broad conditions |

### Loss & Training

This paper is a theoretical work and involves no training. The core contribution consists of theorem proofs:

- **Corollary 5.9**: Under conditions that exclude the first failure source (Assumption 5.4 or 5.6), only ICR guarantees performative validity; both CE and CR may fail.

## Key Experimental Results

### Main Results

Experiments are conducted on 5 synthetic data settings and 2 real-world data settings to validate the theory:

| Data Setting | CE Acceptance Rate Change | CR Acceptance Rate Change | ICR Acceptance Rate Change |
|--------------|--------------------------|--------------------------|---------------------------|
| LAdd (linear additive noise) | Large decrease | Large decrease | No change |
| LMult (linear multiplicative noise) | Large decrease | Large decrease | No change |
| NLAdd (nonlinear additive noise) | Large decrease | Large decrease | No change |
| NLMult (nonlinear multiplicative noise) | Large decrease | Large decrease | No change |
| LCubic (polynomial noise) | Large decrease | Large decrease | Small positive change |
| GPA (university admissions) | ≈−80% | ≈−80% | No change |
| Credit (credit scoring) | Significant decrease | Significant decrease | No change |

### Ablation Study

Conditional distribution shift analysis (Q1) — comparing pointwise differences in conditional probabilities before and after recourse:

| Data Setting | CE Cond. Prob. Change | CR Cond. Prob. Change | ICR Cond. Prob. Change |
|--------------|----------------------|----------------------|----------------------|
| NLAdd | −70% ~ −100% | −70% ~ −100% | 0% |
| LAdd | Significant negative shift | Significant negative shift | 0% |
| LMult | Significant negative shift | Significant negative shift | 0% |
| LCubic | Significant negative shift | Significant negative shift | 0% ~ +60% |

### Key Findings

1. **CE and CR invariably produce negative conditional distribution shifts and severe performative failure** — acceptance rates drop substantially across all 7 data settings
2. **ICR maintains performative validity in all settings** — acceptance rates remain nearly unchanged, with results far exceeding the scope of theoretical guarantees
3. **The only exception**: In the LCubic setting, ind. ICR exhibits a slight positive increase (0–60%) in conditional probability, which is in fact beneficial to applicants
4. **Real-world data validation**: Acceptance rates for CE/CR plummet by approximately 80% on the GPA and Credit datasets, further corroborating the theoretical predictions

## Highlights & Insights

1. **Conceptual innovation**: "Performative validity" is a highly insightful new concept that naturally bridges the performative prediction literature with the algorithmic recourse literature, revealing a broadly overlooked practical problem
2. **Precise causal graph analysis**: D-separation analysis decomposes the abstract distribution shift problem into two concrete causal paths, providing clear and actionable guidance
3. **Strong practical recommendation**: The paper explicitly advocates the practical recommendation to "avoid CE and CR, and use only ICR," offering important guidance for the XAI community
4. **Strong theory–experiment agreement**: Experimental results perfectly validate the theoretical analysis; ICR performs well even beyond the regime covered by the theoretical guarantees

## Limitations & Future Work

1. **Strong causal knowledge assumption**: ICR requires a complete causal graph and SCM, which are difficult to obtain in practice; the paper itself acknowledges the need to explore extensions under incomplete causal knowledge
2. **Single-step recourse limitation**: Only one recommendation–action–reevaluation cycle is considered; in practice, applicants may apply repeatedly
3. **Absence of the model provider's perspective**: The analysis is conducted solely from the applicant's viewpoint, without considering the possibility that model providers may strategically exploit recourse for guided manipulation
4. **Discrete noise in experimental settings**: Synthetic experiments use discrete noise with finite support to facilitate pointwise comparison of conditional distributions, which may limit the generalizability of the results
5. **Lack of discussion regarding violations of causal sufficiency**: The framework assumes no unobserved confounders, which often does not hold in practice

## Related Work & Insights

- **Performative Prediction (Perdomo et al., 2020)**: Performative validity is a natural generalization of performative stability, but requires a weaker condition — it does not require the model to remain unchanged, only that the updated model does not reject those who were originally accepted
- **Strategic Classification (Hardt et al., 2016)**: Connects recourse with strategic manipulation. The paper's central insight — "gaming vs. improvement" — is directly inherited from the strategic classification literature's distinction between interventions on causal vs. non-causal variables
- **Robust Recourse (Upadhyay et al., 2021)**: Existing robust recourse focuses on exogenous distribution shifts; this paper is the first to address endogenous shifts (induced by recourse itself)
- **ICR (König et al., 2023)**: This paper provides new theoretical support for ICR — not only is ICR preferable from the model provider's perspective, but from the standpoint of performative validity, ICR is the only reliable choice

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of "performative validity" is novel, and the causal analysis identifying two failure sources is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ — Theory is comprehensively validated across 5 synthetic and 2 real-world datasets, though experimental settings involve some simplifications
- Writing Quality: ⭐⭐⭐⭐⭐ — The paper is clearly structured; the running example (GitHub activity vs. a master's degree) is woven throughout the paper and is highly illuminating
- Value: ⭐⭐⭐⭐ — Offers direct practical guidance for XAI/recourse practice, though real-world deployment is constrained by the availability of causal knowledge

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[CVPR 2025\] FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition](../../CVPR2025/causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)

</div>

<!-- RELATED:END -->
