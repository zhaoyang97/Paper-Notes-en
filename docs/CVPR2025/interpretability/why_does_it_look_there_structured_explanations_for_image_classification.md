---
title: >-
  [Paper Note] Why Does It Look There? Structured Explanations for Image Classification
description: >-
  [CVPR 2025][Interpretability][Structured Explanation] This paper proposes the I2X framework, which transforms unstructured interpretability into structured interpretability by tracking the co-variance between model confidence and the intensity changes of abstract prototypes extracted from GradCAM saliency maps across training checkpoints. It also utilizes the identified "uncertain prototypes" to guide fine-tuning, reduce inter-class confusion…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Structured Explanation"
  - "Prototype Learning"
  - "GradCAM"
  - "Training Trajectory Analysis"
  - "Explainable AI"
date: 2026-05-08
content_hash: 582e5bf6108b546a
---

# Why Does It Look There? Structured Explanations for Image Classification

**Conference**: CVPR 2025  
**arXiv**: [2603.10234](https://arxiv.org/abs/2603.10234)  
**Code**: None  
**Area**: Interpretability / XAI  
**Keywords**: Structured Explanation, Prototype Learning, GradCAM, Training Trajectory Analysis, Explainable AI

## TL;DR

This paper proposes the I2X framework, which transforms unstructured interpretability into structured interpretability by tracking the co-variance between model confidence and the intensity changes of abstract prototypes extracted from GradCAM saliency maps across training checkpoints. It also utilizes the identified "uncertain prototypes" to guide fine-tuning, reduce inter-class confusion, and improve classification accuracy.

## Background & Motivation

**Background**: Current XAI methods primarily provide unstructured explanations—such as saliency maps, concept vectors, or counterfactual examples. These methods reveal "where" the model looks, but fail to explain "why" it looks there.

**Limitations of Prior Work**: Some attempts to provide structured reasoning (e.g., HybridCBM, GPT-assisted explanations) rely on auxiliary models to describe behavior, which means the explanations do not originate from the original model itself, risking unfaithfulness or hallucination. Although methods like DiffCAM compare activation patterns, they fail to reveal how the model organizes and utilizes these differences during inference and training.

**Key Challenge**: Existing methods provide "interpretability" rather than "explainability"—they describe the outward behavior of models but lack causal responsibility attribution to the model's internal decision structure.

**Goal**: (1) How to construct structured explanations from unstructured ones? (2) How to track the evolution of model decisions during training? (3) How to leverage these structured explanations to guide model optimization?

**Key Insight**: The authors observe that during model training, changes in attention regions (quantified by saliency maps) and prediction confidence are synchronized and regular. By extracting abstract prototypes across multiple training checkpoints and correlating them with confidence changes, a structured learning trajectory of the model can be constructed.

**Core Idea**: By tracking the co-variance between prototype intensity and confidence across training checkpoints, the outputs of unstructured explanation methods like GradCAM are converted into structured decision explanations.

## Method

### Overall Architecture

The input to I2X is a trained classification model and several checkpoints saved during its training. The output is a structured explanation map describing how the model makes intra-class and inter-class decisions using abstract prototypes. The entire pipeline consists of two main steps: (1) extracting a set of abstract prototypes by clustering features from the final model; and (2) computing the mapping between prototype intensity changes and confidence changes at each training checkpoint, ultimately aggregating into a complete structured explanation.

### Key Designs

1. **Abstract Prototypes**:

    - **Function**: Cluster K representative patterns from the final model's latent features across all training samples.
    - **Mechanism**: Apply PCA dimensionality reduction followed by K-Means clustering to all $N \times h \times w$ feature vectors output by the feature extractor $f$, obtaining K cluster centers as abstract prototypes. The feature at each spatial location is assigned to a specific prototype, thereby associating image regions with prototypes.
    - **Design Motivation**: Unlike "prototype learning" methods like ProtoPNet, the prototypes here are extracted entirely post-hoc without altering the model training process, ensuring faithfulness to the original model.

2. **Prototype Intensity Tracking**:

    - **Function**: Align the GradCAM saliency map with the prototype space at each training checkpoint to quantify the activation intensity of each prototype.
    - **Mechanism**: For a sample $x$ at checkpoint $t$, the prototype intensity is calculated as $P_k^t = \frac{\sum_{j} \mathbf{1}[a_j=k] \cdot I_j^t}{\sum_{j} \mathbf{1}[a_j=k]}$, which averages the saliency values at spatial locations belonging to the same prototype. The changes in prototype intensity $\Delta \mathbf{P}^t$ characterize how the model adjusts its "looking" strategy during training.
    - **Design Motivation**: GradCAM alone only provides heatmaps, but when combined with prototype assignment, it quantifies "which structural patterns the attention is allocated to," enabling a shift from the pixel level to the semantic level.

3. **Prototype-Confidence Mapping**:

    - **Function**: Establish a quantitative relationship between changes in prototype intensity and changes in model confidence.
    - **Mechanism**: First, HDBSCAN is used to cluster the confidence changes $\Delta \hat{Y}^t$ of all samples to identify sample groups with similar confidence change patterns. Then, within each group, ridge regression is used to fit the linear relationship between prototype intensity changes $\pi^t$ and confidence changes $C^t$: $\beta^t = (\pi^{t\top}\pi^t + \lambda I)^{-1}\pi^{t\top}C^t$. The coefficient matrix $\beta^t$ quantifies the contribution of each prototype to the confidence change of each class.
    - **Design Motivation**: Analyzing the relationship between prototypes and confidence directly carries too high a dimensionality. The two-step strategy of clustering before regression reduces complexity and uncovers collective patterns.

### Loss & Training

I2X itself does not modify the model training process. During the fine-tuning phase, the authors construct a "curated dataset" which excludes samples containing uncertain prototypes. The model is first fine-tuned on the curated dataset for one epoch, and then on the full dataset for another epoch, achieving perturbation-guided optimization.

## Key Experimental Results

### Main Results

| Dataset / Model | Fine-tuning Strategy | Accuracy (%) | 2↔7 Confusion Count | Notes |
|--------------|---------|-----------|----------|------|
| MNIST / ResNet-50 | full→full | 98.46±0.31 | 9.60±2.87 | Traditional two-round fine-tuning |
| MNIST / ResNet-50 | curated→full | 98.64±0.12 | 8.40±1.85 | I2X-guided fine-tuning, optimal |
| MNIST / ResNet-50 | full (1 epoch) | 98.52±0.34 | 14.80±6.31 | Baseline single-epoch fine-tuning |
| MNIST / ResNet-50 | curated (1 epoch) | 98.67±0.18 | 9.80±2.93 | Curated set fine-tuning |

| Dataset / Model | Fine-tuning Strategy | Accuracy (%) | Confused Pair | Confusion Count |
|--------------|---------|-----------|--------|-------|
| CIFAR10 / ResNet-50 | full→full | 81.43±2.79 | cat↔dog | 261.20±30.77 |
| CIFAR10 / ResNet-50 | curated→full | 84.02±2.70 | cat↔dog | 238.60±21.90 |
| MNIST / InceptionV3 | full→full | 99.13±0.29 | 4↔9 | 12.60±3.07 |
| MNIST / InceptionV3 | curated→full | 99.11±0.27 | 4↔9 | 10.80±2.71 |

### Ablation Study

| Configuration | Confusion Count (2↔7) | Accuracy (%) | Notes |
|------|-------------|-----------|------|
| full 1 epoch | 14.80±6.31 | 98.52±0.34 | Full data fine-tuning |
| curated 1 epoch | 9.80±2.93 | 98.67±0.18 | Remove samples with uncertain prototypes |
| curated→curated | 9.00±4.90 | 98.31±0.63 | Two rounds on curated set, variance increases |
| curated→full | 8.40±1.85 | 98.64±0.12 | Perturbation first then recovery, most stable |

### Key Findings

- The curated→full strategy reduces confused samples by about 5 on MNIST and about 23 on CIFAR-10, while improving overall accuracy.
- Although two-round fine-tuning on the pure curated dataset yields the lowest confusion count (9.00), the variance nearly doubles (4.90 vs. 2.93), indicating that the model begins exploring new strategies but lacks sufficient support.
- Randomness in the order of training data causes the model to learn completely different sequences of prototype selection and reasoning strategies, which is quantitatively observable in confusion matrix differences.

## Highlights & Insights

- **Theoretical Leap from Interpretability to Explainability**: I2X clearly distinguishes between interpretability and explainability, and proposes a systematic path from the former to the latter; this perspective holds more long-term value than the method itself.
- **Training Trajectory Analysis**: Tracking via checkpoints reveals the progressive strategy of "what the model learns first and what it learns later"—discovering that the model first separates easily distinguishable classes (e.g., 7 vs. 6) before handling hard classes (e.g., 7 vs. 1).
- **Perturbation-Guided Fine-tuning**: Identifying uncertain prototypes and then bypassing harmful samples through dataset curation is a simple yet effective idea that can be transferred to any scenario using post-hoc XAI methods.

## Limitations & Future Work

- Only validated on MNIST and CIFAR-10, with limited dataset complexity; the computational overhead of the number of prototypes and checkpoint analysis on large-scale datasets like ImageNet could be a bottleneck.
- It relies on GradCAM as the underlying explanation method; for Transformer architectures, it needs to switch to methods like AttnLRP, and the framework's universality remains to be verified.
- The choice of the number of prototypes K (32 for MNIST, 128 for CIFAR-10) is currently set manually, lacking an adaptive selection mechanism.
- The linear assumption of ridge regression may be insufficient in complex scenarios, and the relationship between prototypes and confidence might be non-linear.

## Related Work & Insights

- **vs. ProtoPNet**: ProtoPNet enables the model to learn prototypes through forward design, whereas I2X extracts prototypes completely post-hoc, maintaining faithfulness without altering the training process, though sacrificing prototype controllability.
- **vs. DiffCAM**: DiffCAM compares activation patterns of different samples/groups to improve faithfulness, but remains an unstructured explanation; I2X builds on this by further correlating training dynamics.
- **vs. LLM-based XAI (e.g., HybridCBM)**: While LLM-based explanations using GPT/CLIP helper models output more readable results, the explanations originate from the auxiliary model rather than the model being explained itself; I2X avoids this faithfulness issue.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of constructing structured explanations from the perspective of training checkpoints is novel, but the core techniques (PCA + KMeans + ridge regression) are relatively basic.
- **Experimental Thoroughness**: ⭐⭐⭐ Only validated on MNIST and CIFAR-10, lacking experiments on large-scale datasets.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, and the logical chain from interpretability to explainability is complete.
- **Value**: ⭐⭐⭐⭐ The conceptual framework is inspiring, but the experimental scale limits its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](../../ACL2025/interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[CVPR 2025\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[NeurIPS 2025\] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](../../NeurIPS2025/interpretability/toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)
- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)

</div>

<!-- RELATED:END -->
