---
title: >-
  [Paper Note] Do-PFN: In-Context Learning for Causal Effect Estimation
description: >-
  [NeurIPS 2025][Causal Inference][Causal Effect Estimation] This paper proposes Do-PFN, which extends Prior-data Fitted Networks (PFN) to causal effect estimation. A Transformer is pre-trained on large-scale synthetic SCM data to perform in-context causal reasoning, enabling prediction of causal intervention distributions (CID) and CATE from observational data alone—without requiring causal graph knowledge or the unconfoundedness assumption—achieving strong performance on both…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Causal Effect Estimation"
  - "PFN"
  - "in-context learning"
  - "SCM"
  - "CATE"
  - "Amortized Inference"
date: 2026-05-08
content_hash: 1a62fe6fccb19b91
---

# Do-PFN: In-Context Learning for Causal Effect Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2506.06039](https://arxiv.org/abs/2506.06039)  
**Code**: [https://github.com/jr2021/Do-PFN](https://github.com/jr2021/Do-PFN)  
**Area**: Causal Inference / Foundation Models
**Keywords**: Causal Effect Estimation, PFN, in-context learning, SCM, CATE, Amortized Inference

## TL;DR
This paper proposes Do-PFN, which extends Prior-data Fitted Networks (PFN) to causal effect estimation. A Transformer is pre-trained on large-scale synthetic SCM data to perform in-context causal reasoning, enabling prediction of causal intervention distributions (CID) and CATE from observational data alone—without requiring causal graph knowledge or the unconfoundedness assumption—achieving strong performance on both synthetic and semi-synthetic benchmarks.

## Background & Motivation

**Background**: Causal effect estimation is a core task in science. Randomized controlled trials (RCTs) are the gold standard but are often infeasible. Estimating causal effects from observational data typically requires the unconfoundedness assumption, which is difficult to verify. TabPFN has demonstrated remarkable in-context learning performance in tabular machine learning.

**Limitations of Prior Work**: (a) Existing methods rely on causal graph knowledge or the unconfoundedness assumption; (b) meta-learners (T-/S-/X-learner) fail when unconfoundedness is violated; (c) deep learning methods (DragonNet/TARNet) similarly depend on this assumption.

**Key Challenge**: Can large-scale pre-training enable a model to meta-learn causal reasoning capabilities, thereby eliminating the need for an explicit causal graph or unconfoundedness assumption?

**Key Insight**: Inspired by TabPFN—if a model is pre-trained on synthetic causal data that includes interventions, it can learn to predict interventional outcomes from observational data.

**Core Idea**: Pre-train a Transformer on millions of SCMs; given a full observational dataset and an intervention query, the model outputs the conditional intervention distribution $p(y|do(t),\mathbf{x})$.

## Method

### Overall Architecture
**Pre-training phase**: Sample SCM → Generate observational data $\mathcal{D}^{ob}$ and interventional data $\mathcal{D}^{in}$ → Train Transformer to predict $y^{in}$ given $(t^{in}, \mathbf{x}^{in}, \mathcal{D}^{ob})$
**Inference phase**: Given real observational data and an intervention query → Do-PFN directly outputs the CID.

### Key Designs

1. **SCM Prior Design**:

    - Sample diverse DAG structures (4–60 nodes), nonlinear functions, and noise distributions
    - Generate paired observational and interventional data simultaneously
    - Prior covers both identifiable and non-identifiable causal scenarios

2. **Proposition 1 (Theoretical Guarantee)**:

    - Proves that the SGD in Algorithm 1 is equivalent to minimizing the forward KL divergence between the CID and the model's predictive distribution
    - Implies that the model learns an optimal approximation of the CID

3. **Decomposition of Three Uncertainty Types**:

    - Aleatoric uncertainty: arising from noise terms in the SCM
    - Non-identifiability uncertainty: across observationally equivalent SCMs
    - Epistemic uncertainty: arising from finite data (vanishes as data size increases)

4. **Consistency Guarantee**:

    - As $|\mathcal{D}^{ob}| \to \infty$, the posterior distribution converges to the Markov equivalence class

### Loss & Training
- Negative log-likelihood $-\log q_\theta(y^{in}|do(t^{in}), \mathbf{x}^{in}, \mathcal{D}^{ob})$
- 7.3M-parameter Transformer trained on a single RTX 2080 for 48–96 hours
- Bar distribution parameterization for output

## Key Experimental Results

### Main Results — CID / CATE / ATE Estimation

| Method | CID (MSE↓) | CATE (MSE↓) | Graph Knowledge Required |
|--------|-----------|-------------|--------------------------|
| Do-PFN | **Best** | **Best** | No |
| TabPFN v2 | Poor | Poor | No |
| Causal Forest | Medium | Medium | Requires unconfoundedness |
| DragonNet | Medium | Medium | Requires unconfoundedness |
| DoWhy (Graph) | Reference | Reference | Requires causal graph |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Dont-PFN (pre-trained on observational data only) | Substantially worse than Do-PFN, demonstrating that interventional pre-training yields capabilities beyond regression |
| Do-PFN-Graph (with graph information provided) | Performance close to Do-PFN without graph information, indicating the model automatically learns to adjust |
| Unconfoundedness assumption violated | Do-PFN remains robust; baseline methods degrade |
| Large graphs (21–50 nodes) | v1 performance drops; v1.1 (extended pre-training) recovers |

### Key Findings
- Do-PFN automatically performs front-door/back-door adjustment without graph knowledge
- Competitive with specialized CATE estimators on the RealCause benchmark
- Uncertainty is well-calibrated; non-identifiable scenarios correctly yield increased uncertainty

## Highlights & Insights
- **Foundation model paradigm for causal inference**: Successfully extends TabPFN's in-context learning to causal inference, opening a new direction for amortized causal inference.
- **No causal graph or unconfoundedness assumption required**: A significant breakthrough—most causal effect estimation methods require at least one of these.
- **Elegant decomposition of three uncertainty types** (Equation 4): The sources of aleatoric, non-identifiability, and epistemic uncertainty and the conditions under which each can be eliminated are clearly characterized.
- **Dont-PFN ablation is highly convincing**: Demonstrates that interventional pre-training genuinely acquires causal capabilities rather than merely learning regression.

## Limitations & Future Work
- **Binary treatment only**: Continuous and multi-valued treatments are not covered
- **Dependence on SCM prior coverage**: Performance may degrade if the true data-generating process lies outside the prior's support
- **Relatively small model (7.3M parameters)**: Larger models with more pre-training data may yield further improvements
- **Directions for improvement**: Extension to continuous treatments; joint estimation for multiple treatments; integration with LLMs to enrich the prior

## Related Work & Insights
- **vs. Meta-learners (T/S/X-learner)**: Require unconfoundedness; Do-PFN does not
- **vs. DoWhy**: DoWhy requires a causal graph; Do-PFN does not
- **vs. TabPFN**: TabPFN performs prediction; Do-PFN performs causal inference—a critical leap from "conditioning" to "intervening"

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extending PFN to causal inference is an entirely new direction
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + semi-synthetic + RealCause + OOD analysis + calibration analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically rigorous with cleverly designed experiments
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for foundation models in causal inference

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](../../CVPR2026/causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[ICLR 2026\] Overlap-Adaptive Regularization for Conditional Average Treatment Effect Estimation](../../ICLR2026/causal_inference/overlap-adaptive_regularization_for_conditional_average_treatment_effect_estimat.md)
- [\[ICLR 2026\] Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](../../ICLR2026/causal_inference/matching_without_group_barrier_for_heterogeneous_treatment_effect_estimation.md)

</div>

<!-- RELATED:END -->
