---
title: >-
  [Paper Note] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations
description: >-
  [ICML 2025][Interpretability][provable explanations] This paper proposes an abstraction-refinement-based method to efficiently compute provably sufficient explanations for neural network predictions, accelerating the verification process by abstracting large networks into small ones with formal guarantees on explanation quality.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "provable explanations"
  - "abstraction-refinement"
  - "neural network verification"
  - "sufficient explanations"
  - "scalability"
date: 2026-05-08
content_hash: aa3cbcddc5242434
---

# Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations

**Conference**: ICML 2025  
**arXiv**: [2506.08505](https://arxiv.org/abs/2506.08505)  
**Code**: None  
**Area**: Explainability / Neural Network Verification  
**Keywords**: provable explanations, abstraction-refinement, neural network verification, sufficient explanations, scalability

## TL;DR
This paper proposes an abstraction-refinement-based method to efficiently compute provably sufficient explanations for neural network predictions, accelerating the verification process by abstracting large networks into small ones with formal guarantees on explanation quality.

## Background & Motivation
**Background**: Post-hoc explainability is a crucial tool for understanding neural network decisions. Existing methods like SHAP, LIME, and GradCAM are widely used, but they are mostly based on heuristics and cannot provide formal, provable guarantees.

**Limitations of Prior Work**: Recent works have shown that formal guarantees for explanations can be obtained through neural network verification techniques—specifically by identifying subsets of input features that are sufficient to keep the prediction unchanged. However, these methods face severe scalability challenges and are extremely computationally expensive on large-scale networks.

**Key Challenge**: The huge gap between the appeal of provable explanations and their computational cost. Verifying a single prediction requires solving an NP-hard verification problem, making repeated verification on large networks highly time-consuming.

**Goal**: To significantly improve the efficiency of computing sufficient explanations while maintaining provability.

**Key Insight**: To borrow the classic abstraction-refinement paradigm from program analysis. An explanation is first searched for rapidly on an abstracted (simplified) network, followed by refinement as needed.

**Core Idea**: A sufficient explanation obtained from an abstracted network is also provably sufficient for the original network, allowing verification to be conducted first on small networks to yield substantial acceleration.

## Method

### Overall Architecture
Input: The original large neural network $f$, an input sample $x$, and the prediction to be explained $f(x)$.
Output: A provably sufficient subset of features $S$ (i.e., once the features in $S$ are fixed, the prediction remains unchanged regardless of how other features vary).

The pipeline is divided into three phases:
1. **Abstraction Phase**: Maps the original network $f$ to an abstract network $\hat{f}$ of significantly reduced scale.
2. **Explanation Computation Phase**: Computes a sufficient explanation on the abstract network $\hat{f}$ using existing provable explanation methods (e.g., MILP-based methods).
3. **Refinement Phase**: If the explanation from the abstract network is not precise enough, the network size is incrementally expanded until convergence.

### Key Designs

1. **Network Abstraction**:

    - Function: Group and merge neurons of the original network to construct an equivalent network with a smaller scale.
    - Mechanism: Reduce the network scale by merging similar neurons. Formally, for the abstract network $\hat{f}$, if a feature subset $S$ is a sufficient explanation for $\hat{f}$, then $S$ is also a sufficient explanation for the original network $f$. This property stems from the abstract relation preserving a conservative over-approximation of outputs.
    - Design Motivation: Because the complexity of the verification problem is directly related to the network scale, shrinking the network can exponentially accelerate the solver.

2. **Provability Transfer**:

    - Function: Prove that the sufficiency on the abstract network can be transferred to the original network.
    - Mechanism: If verification on the abstract network confirms that a feature subset $S$ makes $\hat{f}(x') = \hat{f}(x), \forall x': x'_S = x_S$, then because of the conservatism of abstraction, $f(x') = f(x)$ also holds on the original network.
    - Design Motivation: This is the cornerstone of the method's correctness, ensuring that acceleration does not sacrifice provability.

3. **Iterative Refinement**:

    - Function: Progressively restore network details when the abstract network is too coarse (resulting in insufficient or overly large explanations).
    - Mechanism: Selectively split merged groups of neurons to increase the precision of the abstract network. Top priority is given to splitting neuron groups that have the greatest impact on explanation precision (based on verification feedback).
    - Design Motivation: Overly coarse abstractions may lead to verification failure (false negatives), necessitating a balance between efficiency and precision. Incremental refinement avoids rolling back to the original network all at once.

### Loss & Training
This method does not involve training; it is an inference-time explanation computation method. The core optimization problem is: seeking the minimum feature subset $S$ such that the verifier can prove the sufficiency of the prediction.

## Key Experimental Results

### Main Results

| Dataset/Network | Metric (Explanation Size) | Ours | Prev. SOTA (Direct Verification) | Gain |
|---|---|---|---|---|
| MNIST / Small Net | Avg. Number of Features | ~45 | ~48 | 2.1x |
| MNIST / Medium Net | Avg. Number of Features | ~52 | timeout | >10x |
| CIFAR10 / CNN | Avg. Number of Features | ~120 | timeout | >5x |
| Synthetic Dataset | Solving Time (s) | 12.4 | 156.7 | 12.6x |

### Ablation Study

| Configuration | Solving Time | Explanation Quality | Description |
|---|---|---|---|
| No Abstraction (Direct Verification) | Extremely Slow / Timeout | Optimal | Baseline Method |
| Coarse Abstraction (High Merge Ratio) | Fastest | Worse (Larger Feature Subsets) | Speed Priority |
| Medium Abstraction | Medium | Close to Optimal | Best Balance Point |
| Fine-Grained Abstraction + Refinement | Relatively Fast | Close to Optimal | Recommended by Ours |

### Key Findings
- The abstraction-refinement strategy yields order-of-magnitude speedups on large networks compared to direct verification, with no significant loss in explanation quality.
- Different abstraction levels provide multi-granular interpretations of network predictions, which is of explanatory value in itself.
- The refinement process typically converges in just 2-3 rounds.

## Highlights & Insights
- Ingeniously introduces the abstraction-refinement paradigm from program analysis into the field of explainable AI.
- Multi-level explanation itself serves as a novel visualization method, showing which features the network "focuses on" at different granularities.
- Complete theoretical guarantees: explanations on the abstract network are also provably sufficient on the original network.

## Limitations & Future Work
- The choice of abstraction strategy (which neurons to merge) is currently relatively simple and could be optimized through learning.
- The method is currently primarily applicable to classification tasks; extension to regression tasks requires further study.
- Feasibility on ultra-large-scale networks (such as modern large language models) has not yet been verified.

## Related Work & Insights
- Complementary to heuristic methods like LIME/SHAP: this work provides formal guarantees but at a higher cost.
- Closely related to works in the Neural Network Verification community, such as Marabou, α-β-CROWN, etc.
- The generality of abstraction-refinement is worth exploring in other verification scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing abstraction-refinement into provable explanations is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on various networks and datasets, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous proofs and clear logic.
- Value: ⭐⭐⭐⭐ Addresses the scalability bottleneck of provable explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations](../../CVPR2026/interpretability/back_to_the_feature_explaining_video_classifiers_with_video_counterfactual_expla.md)
- [\[ICML 2026\] Grokking: From Abstraction to Intelligence](../../ICML2026/interpretability/grokking_from_abstraction_to_intelligence.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)

</div>

<!-- RELATED:END -->
