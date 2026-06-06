---
title: >-
  [Paper Note] Why Does It Look There? Structured Explanations for Image Classification
description: >-
  [CVPR 2026][Interpretability][Structured Explanations] This paper proposes the I2X framework, which transforms unstructured explainability (saliency maps) into structured explanations by tracking the co-evolution of prot…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Structured Explanations"
  - "Prototypes"
  - "GradCAM"
  - "Model Training Dynamics"
  - "XAI"
date: 2026-05-08
content_hash: 561a014237914675
---

# Why Does It Look There? Structured Explanations for Image Classification

**Conference**: CVPR 2026
**arXiv**: [2603.10234](https://arxiv.org/abs/2603.10234)  
**Code**: None  
**Area**: Explainability
**Keywords**: Structured Explanations, Prototypes, GradCAM, Model Training Dynamics, XAI

## TL;DR

This paper proposes the I2X framework, which transforms unstructured explainability (saliency maps) into structured explanations by tracking the co-evolution of prototype intensity extracted via GradCAM and model confidence across training checkpoints. The framework reveals the reasoning structure underlying "why the model attends to a specific region" and leverages this understanding to guide fine-tuning for performance improvement.

## Background & Motivation

**Background**: XAI methods primarily produce three types of outputs — saliency maps (GradCAM), concept vectors (TCAV), and counterfactual examples. These constitute **unstructured explainability**, indicating only "where the model looks" without revealing "how the model organizes this information for reasoning."

**Limitations of Prior Work**:
   - Existing methods provide fragmented explanations and cannot answer "why the model attends to a particular region" or "how the model makes decisions across classes."
   - Some methods leverage auxiliary models such as GPT/CLIP to describe model behavior, but such explanations are not faithful to the original model and may introduce hallucinations.
   - The dynamic process by which a model progressively constructs its decision strategy during training remains entirely opaque.

**Key Challenge**: Interpretability ≠ Explainability — the former describes phenomena, while the latter requires structured attribution.

**Key Insight**: Model decisions are not static; during training, the model progressively establishes associations between prototype evidence and confidence. Tracking this process enables the construction of structured explanations.

**Core Idea**: By tracking the mapping between prototype intensity changes and confidence changes across training checkpoints, unstructured explanations are elevated to structured explanations.

## Method

### Overall Architecture

I2X consists of five steps:
1. Apply K-Means clustering to all hidden feature vectors of the final trained model → abstract prototypes.
2. Generate saliency maps via GradCAM at selected training checkpoints.
3. Align saliency maps with prototypes → compute prototype intensity.
4. Cluster confidence change patterns using HDBSCAN → group samples.
5. Model the mapping from prototype intensity changes to confidence changes using ridge regression.

### Key Designs

1. **Abstract Prototype Extraction**:

    - Function: Extract representative patterns from features learned by the model.
    - Mechanism: Extract hidden features $\mathbf{F} \in \mathbb{R}^{(N \cdot h \cdot w) \times d}$ from all training samples using the final model, apply PCA for dimensionality reduction, then apply K-Means clustering to obtain $K$ cluster centers as prototypes.
    - Each feature location is assigned to the nearest prototype: $A_i = (a_1, a_2, ..., a_{hw}), a_j \in \{1,...,K\}$
    - Design Motivation: Compress the high-dimensional feature space into a finite set of interpretable "patterns."

2. **Prototype Intensity Tracking**:

    - Function: Quantify the degree to which the model attends to each prototype at each training checkpoint.
    - Core Formula: Align saliency maps with prototypes and compute the average intensity of each prototype:
    $P_k^t = \frac{\sum_{j=1}^{hw} \mathbf{1}[a_j = k] \cdot \text{Flatten}(I_j^t)}{\sum_{j=1}^{hw} \mathbf{1}[a_j = k]}$
    - The change $\Delta \mathbf{P}^t = \mathbf{P}^{t+1} - \mathbf{P}^t$ characterizes the evolution of prototype evidence.
    - Design Motivation: Saliency maps indicate "where to look," prototypes indicate "what to look at," and intensity changes indicate "the learning trajectory."

3. **Confidence–Prototype Mapping**:

    - Function: Establish a quantitative relationship between prototype intensity changes and model confidence changes.
    - Mechanism:
        - Apply HDBSCAN to cluster the confidence changes $\Delta \hat{Y}^t$ of all samples, identifying common learning patterns.
        - Model the mapping using ridge regression: $\beta^t = (\pi^{t\top}\pi^t + \lambda \mathbf{I})^{-1}\pi^{t\top}C^t \in \mathbb{R}^{K \times M}$
        - $\beta^t$ quantifies how prototype intensity changes drive confidence changes at training step $t$.
    - Design Motivation: Aggregating $[\beta_t]$ across all checkpoints reveals how the model organizes prototype evidence to support and distinguish between classes.

4. **Assembly of Structured Explanations**:

    - Function: Analyze the decision process of each class from the perspectives of shared and specialized prototypes.
    - Shared prototypes: Prototypes present across all samples, e.g., the horizontal and diagonal strokes of the digit 7.
    - Specialized prototypes: Prototypes appearing only in subgroups, used to distinguish intra-class variants.
    - Key Finding: Rather than distinguishing all classes simultaneously, the model learns **progressively** — resolving classes with more distinct prototypes first, then handling ambiguous ones.

### Loss & Training

I2X is an analysis framework and introduces no new training loss. However, the "uncertain prototypes" identified through this framework can be leveraged via a **perturbation fine-tuning strategy** to improve performance: fine-tune for one epoch on data excluding samples containing uncertain prototypes, followed by one epoch of fine-tuning on the complete dataset.

## Key Experimental Results

### Main Results — Fine-Tuning Improvement

| Fine-Tuning Strategy | Accuracy (%) | 2↔7 Confusions | Notes |
|---|---|---|---|
| Full → Full | 98.46±0.31 | 9.60±2.87 | Baseline |
| Curated → Curated | 98.31±0.63 | 9.00±4.90 | Fewer confusions but less stable |
| **Curated → Full** | **98.64±0.12** | **8.40±1.85** | Best: fewer confusions and more stable |

### Generalization on CIFAR-10 / InceptionV3

| Model / Dataset | Fine-Tuning Strategy | Accuracy (%) | Confusions |
|---|---|---|---|
| ResNet-50 / CIFAR-10 | full→full | 81.43±2.79 | cat↔dog: 261.2 |
| ResNet-50 / CIFAR-10 | **curated→full** | **84.02±2.70** | **238.6** |
| InceptionV3 / MNIST | full→full | 99.13±0.29 | 4↔9: 12.6 |
| InceptionV3 / MNIST | **curated→full** | **99.11±0.27** | **10.8** |

### Key Findings
- Model learning is progressive: the model first distinguishes classes with large prototype differences (e.g., 7 vs. 6), then handles more similar classes (e.g., 7 vs. 1).
- **Uncertain prototypes** (e.g., P-26/P-17) oscillate between two classes during training and are the direct cause of confusion.
- Randomness in training data ordering alters prototype selection strategies — different training runs may lead to distinct reasoning strategies.
- The perturbation fine-tuning strategy (initially fine-tuning with uncertain-prototype samples removed) reduces confusions by approximately 5 on MNIST and approximately 23 on CIFAR-10.

## Highlights & Insights
- **Elevating unstructured explanations to structured explanations**: The framework advances from "what the model attended to" to "why the model attended there and how it makes decisions," representing a conceptual leap.
- **Revealing the model's progressive learning strategy**: This mirrors human learning — easier distinctions are resolved first, followed by more difficult ones.
- **Discovery of uncertain prototypes**: The framework identifies prototypes oscillating across classes as the direct cause of confusion, and enables actionable improvement strategies based on this finding.
- **Structured analysis of training stochasticity**: This work represents the first use of prototype tracking to explain strategy differences across training runs.

## Limitations & Future Work
- Validation is limited to MNIST and CIFAR-10; whether the framework remains interpretable on more complex datasets such as ImageNet has yet to be verified.
- The number of K-Means clusters $K$ requires manual selection (32 for MNIST, 128 for CIFAR-10), and selection strategies for larger datasets remain unclear.
- The framework relies on GradCAM and would require substitution with alternatives such as TokenTM for Transformer architectures.
- Ridge regression is a linear model and may fail to capture complex nonlinear prototype–confidence relationships.
- The fine-tuning improvements, while consistent, are modest in magnitude (< 0.2% on MNIST, ~2.6% on CIFAR-10).
- The analysis incurs substantial computational cost, requiring multiple training checkpoints to be saved and analyzed individually.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conceptual advancement from interpretability to explainability is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is limited to MNIST and CIFAR-10, with relatively small dataset scale and complexity.
- Writing Quality: ⭐⭐⭐⭐ Concepts are articulated clearly, with high information density in figures and tables.
- Value: ⭐⭐⭐⭐ Provides a novel perspective for understanding and improving models, with practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/interpretability/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[CVPR 2026\] Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)

</div>

<!-- RELATED:END -->
