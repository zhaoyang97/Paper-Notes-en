---
title: >-
  [Paper Note] Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models
description: >-
  [ECCV 2024][Interpretability][concept bottleneck model] This paper identifies that the low efficiency of human intervention in Concept Bottleneck Models (CBMs) stems from the independent processing of concepts during intervention, which neglects inter-concept correlations. It proposes a lightweight Concept Intervention Realignment Module (CIRM) that automatically realigns the predictions of related concepts post-intervention, reducing the number of interventions required to r…
tags:
  - "ECCV 2024"
  - "Interpretability"
  - "concept bottleneck model"
  - "Human Intervention"
  - "Concept Realignment"
  - "Human-AI Collaboration"
date: 2026-05-08
content_hash: 8ccf84db7d8a948d
---

# Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models

**Conference**: ECCV 2024  
**arXiv**: [2405.01531](https://arxiv.org/abs/2405.01531)  
**Code**: [GitHub](https://github.com/ExplainableML/concept_realignment)  
**Area**: Interpretability  
**Keywords**: concept bottleneck model, Human Intervention, Concept Realignment, interpretability, Human-AI Collaboration

## TL;DR

This paper identifies that the low efficiency of human intervention in Concept Bottleneck Models (CBMs) stems from the independent processing of concepts during intervention, which neglects inter-concept correlations. It proposes a lightweight Concept Intervention Realignment Module (CIRM) that automatically realigns the predictions of related concepts post-intervention, reducing the number of interventions required to reach target performance by up to 70%.

## Background & Motivation

**Background**: Deep learning model deployment in high-stakes scenarios (medical, legal, ethical) is hindered by black-box decision-making processes. Concept Bottleneck Models (CBMs) make the decision process interpretable by introducing a human-understandable concept layer (e.g., "white wings", "orange beak"), dividing classification into concept prediction and concept-based classification steps.

**Limitations of Prior Work**: A core advantage of CBMs is that they allow human experts to intervene at test time—correcting incorrect concept predictions to rectify model decisions. However, existing methods require a large number of interventions to significantly improve performance. For instance, on the widely used CUB bird dataset, an average of 13 interventions is required to improve accuracy from 68% to 90%, which is impractical in scenarios where human annotation is expensive.

**Key Challenge**: Each concept intervention requires human expert analysis and correction, which is extremely costly. However, CBMs treat each concept as an independent entry—correcting one concept does not affect the predicted values of other concepts. Yet, concepts in reality are often correlated (e.g., "white wings" and "white belly" have a high probability of co-occurrence). Treating them independently means human feedback is underutilized.

**Goal**: Achieve the same or better classification performance with fewer human interventions, thereby improving intervention efficacy.

**Key Insight**: Leverage statistical co-occurrence relationships between concepts—when a human corrects one concept, the updated values of other related concepts should be automatically inferred. For instance, when "white wings" is confirmed as true, the probability of "white belly" should increase accordingly.

**Core Idea**: Train a lightweight concept realignment network $u$ that automatically updates the prediction values of all unintervened concepts based on concept correlations after each intervention.

## Method

### Overall Architecture

Pipeline: Input image $x$ -> concept encoder $g(x)$ predicts concepts $\hat{c}$ -> human intervention corrects a subset of concepts -> **CIRM realignment module** $u(\tilde{c}_t)$ automatically updates other concepts -> classification head $f$ outputs the final prediction.

CIRM is seamlessly plugged between the intervention step and the classification head without modifying the original structure of CBM/CEM.

### Key Designs

1. **Concept Intervention Realignment Module (CIRM)**:

    - **Function**: After human intervention on a set of concepts $\mathcal{S}_t$, automatically adjust the prediction values of the remaining unintervened concepts $\setminus\mathcal{S}_t$.
    - **Mechanism**: Train a realignment network $v$ (MLP or LSTM) whose input is the post-intervention concept vector $\tilde{c}_t = \{c_{\mathcal{S}_t}, \hat{c}_{\setminus\mathcal{S}_t}\}$, and whose output is the realigned concepts:
      $$u(\tilde{c}_t, \mathcal{S}_t)^{(i)} = \begin{cases} v(\tilde{c}_t)^{(i)} & \text{if } i \notin \mathcal{S}_t \\ \tilde{c}_t^{(i)} & \text{if } i \in \mathcal{S}_t \end{cases}$$
      Key constraint: Realigned values of concepts corrected by humans are not overwritten (fidelity), and only unintervened concepts are updated.
    - **Design Motivation**: Concepts do not appear independently in reality; co-occurrence relationships exist. Intervening in one concept naturally provides contextual information about other concepts (e.g., confirming "crested" implies the target might be a specific bird species), and this information should be propagated and utilized.

2. **Training Strategy (Post-hoc vs Joint)**:

    - **Function**: Provide two deployment methods—post-hoc training (freeze the trained CBM and train only $u$) and joint training (end-to-end training with IntCEM).
    - **Mechanism (Post-hoc)**: Train the realignment network using cross-entropy loss:
      $$\mathcal{L}(u) = \frac{1}{T}\sum_{t=0}^{T} \text{CE}(u(\tilde{c}_t), c)$$
      During training, simulate the complete intervention process: start from the base model's predictions, progressively intervene on $T$ concepts following the UCP strategy, and train the realignment network at each step.
    - **Mechanism (Joint with IntCEM)**: Modify the training objective of IntCEM to introduce the realignment loss:
      $$\mathcal{L}_{\text{conc-ReA}} = \frac{1}{2}\left(\mathcal{L}_{\text{conc}}(\hat{c},c) + \frac{\text{CE}(\kappa_0, c) + \gamma^T \text{CE}(\kappa_T, c)}{1 + \gamma^T}\right)$$
    - **Design Motivation**: Post-hoc training does not require modifying the original model (plug-and-play), while joint training allows the concept encoder to also become aware of the realignment process.

3. **Policy Alignment**:

    - **Function**: Ensure consistent concept selection policies are used during training and deployment.
    - **Mechanism**: By default, use UCP (Uncertainty-based Concept Selection Policy) to prioritize intervening on concepts with predicted probabilities closest to 0.5 (i.e., the most uncertain concepts). The realigned concept values $\kappa_t$ are fed back to the policy $\pi(\kappa_t)$ to determine the next intervention target.
    - **Design Motivation**: Realignment alters the uncertainty ordering of unintervened concepts; hence, subsequent concept selection should be based on the updated values. Experimental results confirm that consistency between training and deployment policies is crucial.

### Loss & Training

- **Post-hoc mode**: Freeze $g$ and $f$ of CBM/CEM, train only the MLP realignment network $v$ using the concept prediction CE loss averaged over all $T$ intervention steps.
- **Joint mode**: Based on the IntCEM training framework, append the realignment concept loss $\mathcal{L}_{\text{conc-ReA}}$ to $\mathcal{L}_{\text{IntCEM}}$.
- **Hyperparameter search**: Use Optuna with 50 trials, searching over hidden layers $\in\{1,2,3\}$, neuron count $\in\{k, 2k, k/2\}$, and learning rate $\in[10^{-5}, 10^{-1}]$.
- **$T=k$ during training**: Simulate the full trajectory of intervening on all concepts.

## Key Experimental Results

### Main Results

**Concept Loss AUC (lower is better) and Accuracy AUC (higher is better)**:

| Base Model | Realignment | Concept Loss AUC (CUB) | Concept Loss AUC (AwA2) | Acc AUC (CUB) | Acc AUC (AwA2) |
|----------|--------|------------------------|-------------------------|---------------|----------------|
| Sequential CBM | ✗ | 6.71 | 4.26 | 2460.8 | 8364.0 |
| Sequential CBM | ✓ | **3.15** | **1.13** | **2510.9** | **8397.6** |
| Independent CBM | ✗ | 6.71 | 4.26 | 2653.4 | 8403.4 |
| Independent CBM | ✓ | **3.15** | **1.13** | **2678.3** | **8437.0** |
| CEM | ✗ | 5.99 | 4.90 | 2521.4 | 8429.3 |
| CEM | ✓ | **3.20** | **1.69** | **2558.4** | **8433.9** |

**Key data points (from curves)**:
- CUB: Concept loss reduction from 0.6 to 0.06 requires 11 interventions (with realignment) vs 23 (without), a 52% reduction.
- AwA2: Achieving a 10-fold reduction in concept loss requires 16 interventions vs 60+, a 70%+ reduction.
- AwA2: Reaching 98% accuracy requires 12 interventions vs 19.

### Ablation Study

**Realignment Network Architecture Comparison (CUB, Sequential CBM + UCP)**:

| Architecture | Input Type | Concept Loss AUC | Note |
|------|---------|------------------|------|
| MLP | Base model prediction $\tilde{c}_t$ | **Best** | Default configuration, simple and effective |
| MLP | Prev. step realignment output $\kappa_{t-1}$ | Runner-up | Iterative refinement performs worse than original input |
| LSTM | Base model prediction | Slightly worse than MLP | Intervention history provides limited helpful info |
| LSTM | Prev. step realignment output | Worst | Combining history and iterative refinement yields poor results |

**Importance of Training-Deployment Policy Alignment (CUB)**:

| Training Policy | Deployment Policy | Performance |
|----------|----------|------|
| UCP | UCP | Best |
| UCP | Random | Improved but sub-optimal |
| Random | Random | Best under random deployment |

### Key Findings

- **All concept models benefit**: CIRM is effective across Sequential/Independent/Joint CBMs and CEMs, consistently halving concept loss and boosting accuracy.
- **Simple MLP is best**: Surprisingly, there is no need to account for intervention history (LSTM) or use iterative refinement (inputting $\kappa_{t-1}$); directly processing the current post-intervention concept vector using an MLP yields the best results.
- **Policy alignment is crucial**: Training with UCP and deploying with a random policy underperforms compared to training and deploying both with random selection—the realignment network adapts to the policy distribution used during training.
- **Concept-level improvement > accuracy improvement**: Improvement in concept prediction is highly significant (AUC halved), while accuracy improvement is relatively smaller due to accuracy saturation in high-intervention regions.
- **CelebA shows the smallest gain**: Having only 8 (noisy) concepts, the concept information itself is insufficient to support classification, becoming a bottleneck.
- **Joint IntCEM + CIRM is also effective**: Even on IntCEM, which already leverages intervention-aware training, concept realignment still yields substantial improvements.

## Highlights & Insights

- **The "independent intervention" failure mode is accurately identified**: Prior CBM studies focused on better concept representations or improved intervention policies, leaving the fundamental issue of "no post-intervention propagation" unnoticed. This insight is highly incisive.
- **Minimal design, powerful performance**: A small, post-hoc trained MLP improves intervention efficacy by 50-70% without altering the original model architecture, resulting in almost zero deployment cost.
- **Transferable trick — relation propagation**: Any system involving multi-dimensional annotations/feedback can borrow this idea—automatically inferring other dimensions when one is corrected (e.g., label correction in multi-label classification, multi-attribute editing).
- **Post-hoc compatibility**: As a plug-and-play module, it can be directly applied to any existing CBM/CEM, significantly lowering the barrier to deployment.
- **Implicit learning of concept relationships**: CIRM does not require a predefined concept graph; the MLP implicitly captures co-occurrence statistics from the training data.

## Limitations & Future Work

- **Requires concept ground-truth data**: Training CIRM still requires ground-truth concept annotations, which is an inherent limitation of CBM systems.
- **Limited gains on CelebA**: When concepts are noisy and few, the headroom for realignment is limited.
- **MLP capacity may be insufficient**: For scenarios with more complex concept relationships (e.g., conditional dependencies, hierarchical relations), a simple MLP might be inadequate.
- **Considers only scalar concepts**: For the embedding-based concepts of CEM, realignment is performed at the probability level rather than directly in the embedding space.
- **Impact of intervention order is under-explored**: Only UCP vs. random was compared; more complex adaptive policies were not evaluated.
- **Lack of real-user experiments**: All experiments simulate human intervention using ground-truth concepts without real human-in-the-loop interaction testing.

## Related Work & Insights

- **vs IntCEM (Zarlenga et al. 2023)**: IntCEM introduces interventions during training to improve the model's receptiveness to interventions, yet still processes concepts independently. CIRM is orthogonal to IntCEM, and combining them yields better performance.
- **vs Energy-based CBMs (Xu et al. 2023)**: A concurrent work that also attempts to update concept predictions post-intervention, but uses energy-based models. CIRM is simpler, performs better, and is easier to integrate.
- **vs UCP (Lewis & Catlett 1994)**: UCP is a concept selection policy deciding "which to intervene on". CIRM propagates information post-intervention to decide "how to update other concepts". They complement each other.
- **Insight**: In any AI system with human feedback, "feedback propagation" should be prioritized over "isolated application"—this provides valuable lessons for RLHF, active learning, and related fields.

## Rating

- Novelty: ⭐⭐⭐⭐ Highly incisive insights (identifying the issue of independent concept processing) and simple formulation, though the core idea itself is not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, spanning three datasets, five base models, multiple ablations, policy alignment analysis, and qualitative evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, with a complete and logical chain from problem to solution to experiments, and rich illustrations.
- Value: ⭐⭐⭐⭐ Practical boost to the CBM domain, reducing human-machine collaboration costs, though bounded by the overall applicability of CBMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[ICLR 2026\] Debugging Concept Bottleneck Models through Removal and Retraining](../../ICLR2026/interpretability/debugging_concept_bottleneck_models_through_removal_and_retraining.md)

</div>

<!-- RELATED:END -->
