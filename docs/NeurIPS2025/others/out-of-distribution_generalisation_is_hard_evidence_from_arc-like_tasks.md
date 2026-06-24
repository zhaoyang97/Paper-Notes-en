---
title: >-
  [Paper Note] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks
description: >-
  [NeurIPS 2025][OOD generalisation] By constructing ARC-like tasks with well-defined OOD metrics, this paper demonstrates that standard neural networks (MLP/CNN/Transformer) fail to achieve compositional OOD generalisation. Moreover, even architectures designed with correct inductive biases that attain near-perfect OOD performance may still learn incorrect compositional features.
tags:
  - "NeurIPS 2025"
  - "OOD generalisation"
  - "compositional generalisation"
  - "ARC tasks"
  - "inductive bias"
  - "feature learning"
date: 2026-05-08
content_hash: 5e861d7239221e55
---

# Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2505.09716](https://arxiv.org/abs/2505.09716)  
**Code**: None  
**Area**: Other
**Keywords**: OOD generalisation, compositional generalisation, ARC tasks, inductive bias, feature learning

## TL;DR

By constructing ARC-like tasks with well-defined OOD metrics, this paper demonstrates that standard neural networks (MLP/CNN/Transformer) fail to achieve compositional OOD generalisation. Moreover, even architectures designed with correct inductive biases that attain near-perfect OOD performance may still learn incorrect compositional features.

## Background & Motivation

Out-of-distribution (OOD) generalisation is widely regarded as a hallmark of human and animal intelligence. Achieving OOD generalisation through composition requires a system to:
1. Identify input-output mapping properties that are invariant across environments
2. Transfer them to novel inputs

However, existing research has a critical gap:
- Testing on OOD settings alone is insufficient to prove that an algorithm has learned compositional structure
- One must additionally verify that the identified features are genuinely compositional
- When a system performs well on OOD tests, it remains unclear whether it has truly learned the correct compositional rules

The authors' central argument: **validating OOD generalisation requires jointly verifying both performance and feature correctness**.

## Method

### Overall Architecture

1. Design two tasks with clearly defined OOD metrics
2. Evaluate the OOD capabilities of three standard networks (MLP, CNN, Transformer)
3. Design two novel architectures with correct inductive biases
4. Analyse whether feature learning remains correct even when OOD performance is perfect

### Key Designs

**Task Design**:
- Inspired by the ARC (Abstraction and Reasoning Corpus) benchmark
- Tasks possess decomposable compositional structure
- OOD metrics are clearly defined: whether novel combinations of input features are handled correctly

**Task 1 — Geometric Transformation Task**:
- Input: grid patterns with specific geometric properties
- Output: patterns after deterministic transformations
- OOD setting: the combinations of geometric properties differ between training and test sets
- Key property: transformations can be decomposed into independent sub-operations

**Task 2 — Colour–Shape Composition Task**:
- Inputs involve independent variation in colour and shape
- Correct compositional generalisation requires learning invariant features for colour and shape separately

**Two Novel Architectures**:
- Architecture A: embeds geometric invariance biases, enabling the network to process transformations along each dimension independently
- Architecture B: embeds compositional decomposition biases, forcing the network to factorize the feature space into independent subspaces

### Loss & Training

- Standard cross-entropy / MSE loss
- All architectures use identical training data and optimisation strategies
- Train–test splits strictly control the degree of OOD shift

## Key Experimental Results

### Main Results

OOD generalisation of standard networks:

| Network | Task 1 ID Acc. | Task 1 OOD Acc. | Task 2 ID Acc. | Task 2 OOD Acc. |
|---------|---------------|-----------------|---------------|-----------------|
| MLP | ~100% | ~0% | ~100% | ~0% |
| CNN | ~100% | ~0% | ~100% | ~0% |
| Transformer | ~100% | ~0% | ~100% | ~0% |

Novel architectures with correct inductive biases:

| Architecture | Task 1 OOD Acc. | Task 2 OOD Acc. | Feature Correctness |
|-------------|-----------------|-----------------|---------------------|
| Biased Arch. A | ~100% | ~100% | Partially incorrect |
| Biased Arch. B | ~100% | ~100% | Partially incorrect |

### Ablation Study

| Condition | OOD Performance | Correct Compositional Features Learned |
|-----------|----------------|----------------------------------------|
| No inductive bias | Fails | No |
| Weak inductive bias | Partial success | Uncertain |
| Strong inductive bias | Succeeds | Not guaranteed |
| Perfect inductive bias | Near-perfect | May still be incorrect |

### Key Findings

1. **Standard networks fail comprehensively**: MLP, CNN, and Transformer all achieve near-zero OOD accuracy on both tasks.
2. **Correct bias ≠ correct features**: Even when embedding correct inductive biases yields near-perfect OOD performance, networks may still learn incorrect compositional features.
3. **OOD performance ≠ compositional generalisation**: High OOD performance does not prove that an algorithm has learned the underlying compositional structure.
4. **Necessity of verifying feature correctness**: Testing OOD performance alone is insufficient to confirm compositional generalisation capability.

## Highlights & Insights

1. **Methodological contribution**: Proposes a new standard for evaluating compositional generalisation — one must not only measure performance but also verify feature correctness.
2. **Counter-intuitive finding**: Near-perfect OOD performance can be achieved through "incorrect" means, i.e., without relying on correct compositional features.
3. **Implications for ARC**: Provides an explanation for why current AI systems perform poorly on ARC tasks — they lack genuine compositional reasoning ability.
4. **Theory–practice gap**: Inductive biases are necessary but not sufficient, underscoring the importance of understanding what models actually learn.

## Limitations & Future Work

1. The designed tasks are relatively simple and limited in complexity compared to real ARC tasks.
2. Only three standard architectures and two custom architectures are evaluated.
3. The verification of feature correctness relies heavily on manual inspection, lacking automated evaluation tools.
4. The compositional generalisation capabilities of larger-scale or pretrained models are not explored.
5. The influence of new paradigms such as in-context learning on OOD generalisation is not considered.

## Related Work & Insights

- **ARC Challenge**: The abstract reasoning benchmark proposed by François Chollet.
- **Compositional generalisation**: Compositional generalisation benchmarks in semantic parsing, such as SCAN and COGS.
- **Systematic generalisation**: Lake & Baroni's discussion on systematic generalisation in neural networks.
- Insight: In-depth analysis on simple tasks is often more informative than surface-level testing on complex ones.

## Rating

- Novelty: ⭐⭐⭐⭐ (significant methodological contribution)
- Technical depth: ⭐⭐⭐⭐ (rigorous analysis)
- Experimental thoroughness: ⭐⭐⭐ (tasks could be more diverse)
- Value: ⭐⭐⭐ (primarily theoretical insights)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](../../CVPR2025/others/open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)
- [\[ICML 2025\] Softmax is not Enough (for Sharp Size Generalisation)](../../ICML2025/others/softmax_is_not_enough_for_sharp_size_generalisation.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)

</div>

<!-- RELATED:END -->
