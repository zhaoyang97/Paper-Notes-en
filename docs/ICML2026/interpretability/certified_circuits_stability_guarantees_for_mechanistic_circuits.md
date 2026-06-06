---
title: >-
  [Paper Note] Certified Circuits: Stability Guarantees for Mechanistic Circuits
description: >-
  [ICML 2026][Interpretability][Mechanistic interpretability] The authors propose the Certified Circuits framework, which provides provable dataset-level stability guarantees for circuit discovery in mechanistic interpreta…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Mechanistic interpretability"
  - "circuit discovery"
  - "randomized smoothing"
  - "robustness certification"
  - "dataset stability"
date: 2026-05-08
content_hash: 7918782395664a61
---

# Certified Circuits: Stability Guarantees for Mechanistic Circuits

**Conference**: ICML 2026  
**arXiv**: [2602.22968](https://arxiv.org/abs/2602.22968)  
**Code**: https://github.com/AlaaAnani/certified-circuits  
**Area**: AI Safety/Interpretability  
**Keywords**: Mechanistic interpretability, circuit discovery, randomized smoothing, robustness certification, dataset stability  

## TL;DR
The authors propose the Certified Circuits framework, which provides provable dataset-level stability guarantees for circuit discovery in mechanistic interpretability through deletion-based randomized smoothing. This ensures discovered circuits remain invariant under bounded edit distance perturbations of the concept dataset, resulting in more compact, accurate circuits with better OOD generalization.

## Background & Motivation

**Background**: Mechanistic Interpretability aims to understand the internal computations of neural networks by identifying circuits—minimal sub-networks responsible for specific behaviors. In vision models, a circuit for identifying an "alligator" might consist of convolutional filters detecting scales, teeth, and other features; in language models, attention heads and MLP neurons responsible for specific tasks can be isolated through activation patching and causal tracing.

**Limitations of Prior Work**: Current circuit discovery methods are extremely fragile—adding, deleting, or replacing a few semantically equivalent samples in the concept dataset can lead to unpredictable changes in the discovered circuit. For example, a circuit discovered using photos of alligators on land may perform poorly on alligators in water or cartoon alligators. More critically, discovered circuits fail to migrate reliably to out-of-distribution (OOD) data.

**Key Challenge**: The fundamental cause of this instability is that existing methods overfit to specific concept datasets rather than recovering true conceptual representations. Unstable components in a circuit often correspond to spurious features (e.g., focusing on background birds when identifying an alligator) rather than class-defining features (e.g., teeth, scales).

**Goal**: To provide provable dataset-level robustness guarantees for circuit discovery, ensuring that the discovered circuits remain unchanged under bounded edit perturbations of the concept dataset.

**Key Insight**: The authors transfer the idea of Randomized Smoothing (RS) from classification tasks to circuit discovery. By running a circuit discovery algorithm on multiple randomly sub-sampled datasets and aggregating the results, empirical stability is transformed into provable robustness guarantees. This approach is promising because RS-Del (deletion smoothing under edit distance) has been proven effective for discrete objects.

**Core Idea**: Wrap any black-box circuit discovery algorithm with randomized deletion smoothing and perform component-wise voting certification for the inclusion/exclusion decision of each circuit component. Components with uncertain votes are abstained, leading to provably stable and more compact circuits.

## Method

### Overall Architecture
Given a model computational graph $G=(V,E)$, a concept dataset $D$, and any black-box circuit discovery algorithm $A$, Certified Circuits generates a certified circuit through a five-step process: ① Receive inputs (dataset, model graph, algorithm); ② Perform multiple random deletion samplings on the dataset; ③ Run $A$ on each sub-sample to obtain candidate circuits; ④ Aggregate voting frequencies for each component and certify; ⑤ Output the certified circuit, guaranteed to remain invariant under all dataset perturbations within an edit distance $r$.

### Key Designs

1. **Dataset Deletion Smoothing**:

    - **Function**: Transforms dataset-level edit perturbations into a tractable random deletion distribution.
    - **Mechanism**: For each sample in the concept dataset $D = (x_1, \ldots, x_{|D|})$, a Bernoulli mask $\varepsilon_i \sim \text{Bernoulli}(1 - p_{\text{del}})$ is independently sampled. Retaining samples where $\varepsilon_i=1$ yields a sub-dataset $D \odot \varepsilon$. By running the base algorithm $A$ on $n$ such random sub-datasets, the smoothed inclusion probability $p_u(D) = P[A_u(D \odot \varepsilon) = 1]$ is calculated, measuring how frequently a component $u$ is selected under random deletions.
    - **Design Motivation**: Directly enumerating all datasets within an edit distance is infeasible (exponential), whereas deletion smoothing based on RS-Del can transform empirical consistency under random deletions into certification guarantees for the full edit distance (insertion, deletion, replacement).

2. **Per-Component Certification with Abstention**:

    - **Function**: Converts voting probabilities into provable inclusion/exclusion/abstention decisions.
    - **Mechanism**: A confidence threshold $\tau \in [0.5, 1)$ is set to make a three-valued determination for each component $u$: if $p_u > \tau$, certify inclusion (1); if $1 - p_u > \tau$, certify exclusion (0); otherwise, abstain ($\oslash$). The abstention mechanism is key—it adaptively prunes components that are unstable under random deletions, which often correspond to spurious features.
    - **Design Motivation**: Borrowing the per-pixel certification idea from segmentation tasks, binary decisions are extended to three-valued decisions so that uncertain components are excluded rather than incorrectly included, resulting in more compact and accurate circuits.

3. **Certified Radius Guarantee**:

    - **Function**: Provides a provable stability radius, guaranteeing the circuit remains unchanged under bounded dataset edits.
    - **Mechanism**: The core theorem (Theorem 3.1) proves that for any non-abstaining component $u$, the certification decision remains invariant for any perturbed dataset $D'$ satisfying $\text{dist}_{\text{edit}}(D, D') \leq r$ with confidence $\geq 1-\alpha$. The certified radius is $r = \lfloor \log(1.5 - \tau) / \log p_{\text{del}} \rfloor$. For example, when $p_{\text{del}}=0.6$ and $\tau=0.95$, $r=1$, guaranteeing the circuit remains unchanged under any single dataset edit.
    - **Design Motivation**: To provide formal worst-case guarantees rather than empirical heuristics, elevating circuit discovery from "potentially stable" to "provably stable," covering exponentially large families of datasets.

## Key Experimental Results

### Main Results

The experiments cover three architectures (ResNet-101/50, ViT-B/16, GPT-2 Small) and two modalities (vision, language).

| Dataset/Task | Model | Full Model cACC | Baseline cACC | Certified cACC | Gain | Circuit Size K Reduction |
|-------------|------|---------------|---------------|----------------|------|-----------------|
| ImageNet (ID) | ResNet-101 | 0.78 | 0.83 | 0.95 | ↑14% | ↓52% |
| ImageNet-A (OOD) | ResNet-101 | 0.07 | 0.60 | 0.94 | ↑56% | ↓15% |
| OOD-CV (OOD) | ResNet-101 | 0.20 | 0.73 | 0.93 | ↑28% | ↓33% |
| ImageNet-C (OOD) | ResNet-101 | 0.57 | 0.72 | 0.92 | ↑28% | ↓41% |
| IOI (ID) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓58% |
| Greater-Than (ID) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓75% |
| Greater-Than (OOD) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓80% |

### Ablation Study

| Analysis Dimension | Key Results | Explanation |
|----------|----------|------|
| Feature Visualization | Certified included neurons respond to class-defining features | Abstained neurons respond to co-occurring but non-class-specific spurious cues |
| Structural Stability (IoU) | Certified circuits have higher median IoU and tighter distributions | Maximum stability gain corresponds to ∆cACC=100% on OOD-CV |
| Cross-Seed Stability | Certified circuits converge across random seeds | The certification process itself is reliable |
| Computational Overhead | ~2.4× baseline algorithm overhead (n=1000) | Low overhead maintained via caching forward/backward passes |

### Key Findings
- Certified circuits significantly outperform baselines on all vision OOD datasets, with the largest improvements occurring on the most difficult distribution shifts (ImageNet-A ↑56%).
- The abstention mechanism removes not only uninformative components but also "harmful" components that bias the baseline toward spurious class associations.
- In language tasks, while accuracy is already saturated, certified circuits achieve identical performance with up to 80% fewer edges.
- When rediscovered on OOD data, certified circuits converge to a similar invariant core, validating their ability to capture true conceptual representations.

## Highlights & Insights
- **Transfer of Randomized Smoothing from Classification to Circuits**: Moving RS-Del from classification input perturbations to dataset perturbations for circuit discovery, combined with per-component abstention from segmentation, is an elegant "new application for an old tool." This transfer idea can be generalized to any scenario requiring certified stability for structured outputs.
- **Abstention as Pruning**: The abstention mechanism in three-valued certification essentially implements adaptive sparsification—no manual top-K selection is needed; the certification process automatically identifies and excludes unstable components, producing more compact circuits. This suggests a large number of components in traditional top-K selection are spurious.
- **Algorithm Agnosticism**: The framework wraps any black-box algorithm without needing access to its internals, allowing it to enhance existing circuit discovery methods in a plug-and-play manner.

## Limitations & Future Work
- Larger certified radii require higher deletion rates, which may degrade the performance of the base algorithm when the concept dataset is very small (though it supports $r$ up to 59 when $|D|$ is large).
- Currently, a uniform sparsity $K$ is used across all layers; layer-wise budget allocation might produce finer-grained sparsification.
- Not yet validated on larger language models (e.g., LLaMA), multi-modal models, or sparse activation architectures.
- The certification guarantee is relative to a specific threat model and concept dataset and should not be misinterpreted as a complete explanation of model behavior.

## Related Work & Insights
- **ACDC** (Conmy et al., 2023): Automatically discovers circuits through iterative edge pruning; one of the baselines for vision experiments in this paper.
- **EAP-IG** (Hanna et al., 2024): A circuit discovery method for language models that scores computational graph edges via integrated gradients; used as the base algorithm for language experiments.
- **RS-Del** (Huang et al., 2023): An edit distance certification method based on random deletion; provides the theoretical foundation for this certification framework.
- **Fischer et al. (2021)**: Per-pixel certification methods for segmentation tasks, which inspired the per-component abstention design in this paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](../../NeurIPS2025/interpretability/discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)

</div>

<!-- RELATED:END -->
