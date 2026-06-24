---
title: >-
  [Paper Note] SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference
description: >-
  [ICML 2025 (MOSS Workshop)][Capsule Networks] This paper proposes the SynDaCaTE synthetic dataset and the Mereological Inference framework, decomposing part-whole hierarchical inference into two independently evaluable sub-tasks: Image-to-Parts and Parts-to-Wholes. Through carefully designed control experiments, it demonstrates that the bottleneck of CapsNets lies in extracting parts from images rather than inferring wholes from parts. Additionally…
tags:
  - "ICML 2025 (MOSS Workshop)"
  - "Capsule Networks"
  - "Part-Whole Hierarchy"
  - "Synthetic Dataset"
  - "SetTransformer"
  - "Inductive Biases"
date: 2026-05-08
content_hash: 1dccca01310f96db
---

# SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference

**Conference**: ICML 2025 (MOSS Workshop)  
**arXiv**: [2506.17558](https://arxiv.org/abs/2506.17558)  
**Code**: [GitHub](https://github.com/jakelevi1996/syndacate-public)  
**Area**: Computer Vision / Inductive Biases  
**Keywords**: Capsule Networks, Part-Whole Hierarchy, Synthetic Dataset, SetTransformer, Inductive Biases

## TL;DR
This paper proposes the SynDaCaTE synthetic dataset and the Mereological Inference framework, decomposing part-whole hierarchical inference into two independently evaluable sub-tasks: Image-to-Parts and Parts-to-Wholes. Through carefully designed control experiments, it demonstrates that the bottleneck of CapsNets lies in extracting parts from images rather than inferring wholes from parts. Additionally, the permutation-equivariant SetTransformer is found to significantly outperform all baselines in part-to-whole inference (with over a 10x precision advantage).

## Background & Motivation

**Background**: The part-whole hierarchy is a core capability of the human visual system, and Capsule Networks (CapsNets) claim to be able to learn this hierarchical structure. However, since being proposed by Hinton, CapsNets have gradually been replaced by CNNs and Vision Transformers, and their promised "hierarchical inference" capability has never been rigorously verified.

**Limitations of Prior Work**: Existing visual datasets lack ground-truth part information; while it is known what objects an image contains, the constituent parts of the object and their precise poses remain unknown. Without such annotations, it is impossible to determine whether a model has learned part-whole inference or is simply utilizing shortcuts to complete classification tasks.

**Key Challenge**: While CapsNets underperform compared to modern CNNs/Transformers on standard classification tasks, it remains unclear at which stage they fail: is it due to insufficient capability in extracting parts from images (Image-to-Parts), or the inability to assemble wholes from parts (Parts-to-Wholes)? These two sub-tasks are coupled during end-to-end training and cannot be evaluated separately.

**Goal**: (1) Define a clear framework to formalize the meaning of "part-whole inference"; (2) build a synthetic dataset with ground-truth part information to decouple the two sub-tasks; (3) precisely locate the bottleneck of CapsNets and provide directions for future inductive bias designs.

**Key Insight**: Starting from mereology (the theory of parthood relations) in cognitive science, this work strictly decomposes visual inference into two steps: first inferring the set of parts, and then inferring the whole from these parts, while providing ground-truth for each step using synthetic data.

**Core Idea**: Decouple hierarchical inference into two independently evaluable sub-tasks using a synthetic dataset with complete part annotations, thereby precisely diagnosing model capabilities.

## Method

### Overall Architecture
Mereological Inference is defined as a two-step inference process: (1) Image-to-Parts: inferring a set of parts $\mathcal{P}$ (where each part has a class label and a pose vector) from an image $I \in \mathbb{R}^{C \times H \times W}$; (2) Parts-to-Wholes: inferring a set of wholes $\mathcal{W}$ from the set of parts $\mathcal{P}$. Using the SynDaCaTE dataset, the performance of models on these two sub-tasks can be evaluated independently.

### Key Designs

1. **SynDaCaTE Dataset**:

    - Function: Provides a synthetic visual dataset containing complete ground-truth part information.
    - Mechanism: The dataset contains 21 types across 3 categories of objects (line segments, characters, words), where each object has a class label and a continuous pose vector (position, size, rotation, luminance, etc.). Images are generated hierarchically: first sampling top-level objects (e.g., words), then recursively generating sub-parts (characters $\rightarrow$ line segments), and finally rendering them into an image. By controlling generative parameters, various tasks can be defined: ImToClass (Image-to-Class), ImToParts (Image-to-Parts), PartsToChars (Parts-to-Characters), PartsToClass (Parts-to-Class), etc.
    - Design Motivation: Part annotations of natural images are expensive and ambiguous. Synthetic data allows precise control over part information, enabling independent evaluation of the two sub-tasks. Although the dataset is simple, this simplicity ensures that the experimental conclusions are clear and reliable.

2. **PreTrainedPartsToClass Task Design**:

    - Function: Tests the classification capability of models when "part information is already available" by replacing the raw image input with part representations extracted by a pre-trained CNN.
    - Mechanism: First, a CNN is trained on the ImToParts task to learn part extraction from images. Then, the last-layer features of the trained CNN are used as the new input, on top of which a CapsNet and a CNN are trained for classification. If the performance of the CapsNet matches that of the CNN when using the part representations, it indicates that the bottleneck of CapsNets lies in Image-to-Parts rather than Parts-to-Wholes.
    - Design Motivation: This is a key control experiment to locate the bottleneck of CapsNets. By providing pre-extracted part information, the Image-to-Parts phase is bypassed, directly examining the Parts-to-Wholes capability.

3. **Evaluating Permutation-Equivariant Models on Part-Whole Inference**:

    - Function: Compares SetTransformer, DeepSetToSet, element-wise MLP, and flattened MLP on the PartsToChars task.
    - Mechanism: Since a set of parts is unordered, permutation-equivariant or permutation-invariant models should possess better inductive biases. SetTransformer processes set inputs via self-attention mechanisms, achieving an MSE that is over an order of magnitude lower than other baselines when the depth is $\ge 2$. Increasing the width (multiplying parameters by 4) yields almost no improvement for shallow SetTransformers, indicating that a self-attention depth of $\ge 2$ provides a computational capacity that is essentially irreplaceable.
    - Design Motivation: Most visual models operate directly at the pixel level, ignoring the optimal inductive bias for the sub-task of "inferring wholes from parts". Comparing different architectures on a pure Parts-to-Wholes task provides guidance for designing better visual models in the future.

### Loss & Training
ImToClass and PartsToClass use cross-entropy loss; ImToParts uses Chamfer MSE loss (commonly used in set prediction); PartsToChars uses MSE loss averaged over the output set. All models are optimized using Adam; ImToClass is trained for 5k steps, ImToParts for 100 epochs, and PartsToChars for 100 epochs.

## Key Experimental Results

### Main Results: Locating CapsNet Bottlenecks

| Task | Input Type | CNN Accuracy | CapsNet Accuracy | Conclusion |
|------|---------|---------|------------|------|
| ImToClass | Raw Image | ~95% (100 samples) | ~75% (100 samples) | CNN significantly outperforms CapsNet |
| ImToClass | Raw Image | ~99% (60k samples) | ~97% (60k samples) | Gap narrows under large data |
| PreTrainedPartsToClass | Part Representation | ~97% (100 samples) | ~97% (100 samples) | **Both perform equally given part information** |
| PartsToClass | Ground-truth Parts | - | - | SetTransformer far outperforms others |

### Ablation Study: Parts-to-Wholes Model Comparison (PartsToChars Task)

| Model | Depth=1 MSE | Depth=2 MSE | Depth=4 MSE | Characteristics |
|------|-----------|-----------|-----------|------|
| SetTransformer | ~0.1 | ~0.005 | **~0.002** | Abrupt improvement at depth $\ge 2$ |
| SetTransformer (2x width) | ~0.08 | ~0.004 | ~0.002 | Increasing width provides almost no help |
| DeepSetToSet | ~0.15 | ~0.08 | ~0.06 | Slow improvement |
| Element-wise MLP | ~0.2 | ~0.15 | ~0.12 | Cannot exploit relationships between set elements |
| Flattened MLP | ~0.25 | ~0.2 | ~0.15 | Worst; lacks permutation invariance |

### Key Findings
- **The bottleneck of CapsNets is precisely located at the Image-to-Parts phase**: Once pre-extracted part information is provided, the classification accuracy of CapsNets is nearly identical to that of CNNs, proving that CapsNets are not superior to CNNs in inferring wholes from parts.
- **Sudden performance leap for SetTransformer at depth $\ge 2$**: The MSE drops sharply from $\sim 0.1$ to $\sim 0.005$ (a 20x improvement), and increasing the model width yields no benefits, suggesting that a self-attention depth of $\ge 2$ provides an irreplaceable computational structure (potentially related to the "Induction Heads" phenomenon).
- **Part information serves as a powerful representation for efficient classification**: Even when part representations are noisy, using them for classification is more efficient than conducting classification directly from raw images, supporting the hypothesis that part-whole hierarchy is a beneficial inductive bias.

## Highlights & Insights
- **Precise Experimental Design**: Through the clever "bridging" task PreTrainedPartsToClass, the failure of CapsNets is cleanly attributed to the Image-to-Parts stage, a clear diagnosis that has been absent in prior CapsNet literature.
- **Depth Sensitivity of SetTransformer**: A self-attention depth of $\ge 2$ is required for Parts-to-Wholes inference, which resonates with the discovery of Induction Heads in the Transformer circuits literature, suggesting that part assembly requires a form of second-order reasoning capacity.
- **Simple Yet Profound Dataset Design**: Although SynDaCaTE comprises only three simple tiers of objects (line segments, characters, and words), this minimalist design is precisely what makes the conclusions extremely clear, whereas complex datasets might introduce too many confounding variables.

## Limitations & Future Work
- The synthetic data is overly simplified (2D line segments and characters) and drastically diverges from natural images or 3D scenes, leaving the transferability of the conclusions questionable.
- Only the original CapsNet by Sabour et al. (2017) was evaluated; subsequent improved versions (e.g., Matrix Capsules, Efficient-CapsNet) are left untested.
- The Parts-to-Wholes task assumes that the ground-truth parts are known, whereas, in practice, extracting parts remains the core challenge itself.
- As a workshop paper, space is limited, and it lacks comprehensive experimental validation for future directions like MereoFormer.

## Related Work & Insights
- **vs Capsule Networks (Sabour 2017, Hinton 2018)**: These works claim that CapsNets can learn part-whole hierarchies but offer no empirical proof. This paper is the first to prove through controlled experiments that CapsNets fail at the most foundational Image-to-Parts stage.
- **vs Slot Attention (Locatello 2020)**: Slot Attention also focuses on object decomposition but emphasizes unsupervised object-centric representations and does not explicitly evaluate the part-whole hierarchy.
- **vs Vision Transformer**: The advantage of self-attention in set inference discovered in this paper offers a new perspective on the success of ViT—ViTs might partially benefit from self-attention's ability to model part-whole relations among local patches.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The framework is clearly defined, decoupling and evaluating hierarchical inference for the first time, though the dataset itself is relativamente rudimentary.
- **Experimental Thoroughness**: ⭐⭐⭐ The ablation studies are reasonable given the tight space of a workshop paper, but there is a lack of testing on more Capsule Network variants and natural image datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured with rigorous argumentation and concise, powerful conclusions.
- **Value**: ⭐⭐⭐⭐ Significantly deepens the understanding of the failure modes of CapsNets and offers valuable guidance for future designs of inductive biases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](../../ACL2025/others/kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](../../CVPR2025/others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)

</div>

<!-- RELATED:END -->
