---
title: >-
  [Paper Note] From Attribution to Action: Jointly ALIGNing Predictions and Explanations
description: >-
  [AAAI 2026][Segmentation][Explanation-Guided Learning] This paper proposes the ALIGN framework, which jointly trains a learnable masker and a classifier through alternating optimization to iteratively align model attribution maps with task-relevant region masks, simultaneously improving prediction accuracy and interpretability. ALIGN outperforms six strong baselines on the VLCS and Terra Incognita domain generalization benchmarks.
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Explanation-Guided Learning"
  - "Domain Generalization"
  - "Interpretability"
  - "Grad-CAM"
  - "Mask Learning"
date: 2026-05-08
content_hash: 7fad60b8231e359f
---

# From Attribution to Action: Jointly ALIGNing Predictions and Explanations

**Conference**: AAAI 2026
**arXiv**: [2511.06944](https://arxiv.org/abs/2511.06944)  
**Code**: None  
**Area**: Segmentation
**Keywords**: Explanation-Guided Learning, Domain Generalization, Interpretability, Grad-CAM, Mask Learning

## TL;DR

This paper proposes the ALIGN framework, which jointly trains a learnable masker and a classifier through alternating optimization to iteratively align model attribution maps with task-relevant region masks, simultaneously improving prediction accuracy and interpretability. ALIGN outperforms six strong baselines on the VLCS and Terra Incognita domain generalization benchmarks.

## Background & Motivation

Explanation-Guided Learning (EGL) integrates explanation signals (e.g., saliency maps) into training to encourage models to focus on interpretable semantic regions. However, existing EGL methods suffer from two major bottlenecks:

**Annotation dependency**: Methods such as CARE and GRADIA rely on manually annotated masks, which are costly and difficult to scale. Even when pseudo-masks are generated using pretrained segmentation models like SAM, these masks are not optimized for downstream tasks and may include irrelevant regions or omit critical information.

**Low-quality masks degrade performance**: Through both experiments and theoretical analysis, the authors demonstrate that imprecise masks not only fail to improve models but also introduce spurious correlations that reduce predictive performance. For example, SAM applied to images containing dogs may predominantly capture the surrounding environment rather than the target object.

Core motivation: A **task-driven, self-learned mask** is required to guide the model, rather than relying on fixed external annotations or generic segmentation outputs.

## Method

### Overall Architecture

ALIGN (Attribution-Learning Iterative Guidance Network) jointly trains two components:

- **Masker $M$**: A lightweight convolutional network that generates soft masks $M(x) \in [0,1]^d$ identifying task-relevant regions in the input.
- **Classifier $f$**: A standard ResNet that optimizes not only prediction accuracy but also aligns its Grad-CAM saliency maps with the masks.

The two components are trained via **alternating optimization**: the classifier is fixed while the masker is updated, and then the masker is fixed while the classifier is updated.

### Key Designs

#### Theoretical Analysis (PAC Learning Framework)

Under a domain shift scenario, the input is decomposed into an object component $x^{(obj)} = M \odot x$ and a background component $x^{(bg)} = (1-M) \odot x$. Three model types are compared:

- $f_1$ (vanilla): utilizes all features, including spurious ones.
- $f_2$ (perfectly guided): uses only task-relevant regions.
- $f_3$ (strictly guided): uses only a strict subset of the object.

Four lemmas yield key conclusions:

1. **Lemma 1**: Models that do not rely on background features exhibit lower sensitivity to domain shifts (smaller Lipschitz constant).
2. **Lemma 2**: The MSE error gap is upper-bounded by $4|\mathbb{E}_{\mathcal{D}_T}[f(x)] - \mathbb{E}_{\mathcal{D}_S}[f(x)]|$.
3. **Lemma 3**: The cross-entropy gap satisfies $\Delta_{CE} \leq C \cdot \epsilon$, where $\epsilon$ is smaller when the model is less sensitive to the background.
4. **Lemma 4**: $f_2$ achieves better in-domain performance than $f_3$, confirming that retaining complete relevant features is superior to using a subset.

Theoretical insight: **High-quality masks simultaneously improve generalization (by reducing reliance on spurious features) and in-domain performance (by preserving complete relevant features).**

#### Masker Objective

The core idea is to maximize prediction confidence when the foreground is retained and minimize it when the foreground is removed:

$$dist(x) = f_y(x \odot M(x)) - f_y(x \odot (1-M(x)))$$

$$\mathcal{L}_{dist} = MSE(dist(x), 1)$$

Two regularization terms are added to ensure mask quality:
- **Sparsity loss** $\mathcal{L}_{sparsity} = \|M(x)\|_1$: prevents unnecessary activations.
- **Smoothness loss** $\mathcal{L}_{smooth}$: penalizes abrupt transitions between adjacent pixels to enforce spatial continuity.

$$\mathcal{L}_{mask} = \mathcal{L}_{dist} + \lambda_1 \mathcal{L}_{sparsity} + \lambda_2 \mathcal{L}_{smooth}$$

#### Classifier Objective

$$\mathcal{L}_{clf} = \mathcal{L}_{cls} + \lambda_3 \mathcal{L}_{egl} + \lambda_4 \mathcal{L}_{reg}$$

- **Classification loss** $\mathcal{L}_{cls} = CE(f(x), y)$
- **Explanation-guided loss** $\mathcal{L}_{egl} = BCE(\Phi_y(x), M(x))$: aligns Grad-CAM saliency maps with the learned masks.
- **Mixup regularization** $\mathcal{L}_{reg}$: applies Mixup to same-class samples to encourage consistency in attribution space and promote explanation sparsity.

### Loss & Training

- **Warm-up strategy**: The classifier is trained exclusively for the first 200 epochs using $\mathcal{L}_{cls} + \mathcal{L}_{reg}$, without explanation supervision, allowing the classifier to establish reliable initial decisions.
- Alternating optimization then begins: $M$ is updated with $f$ fixed, followed by updating $f$ with $M$ fixed, progressively aligning model reasoning with the learned masks.
- Grad-CAM is used as the explanation method throughout.

## Key Experimental Results

### Main Results

**VLCS Dataset (4 sub-domains, Accuracy/AUC)**:

| Method | VOC2007 Acc | VOC2007 AUC | LabelMe Acc | Caltech Acc | SUN09 Acc |
|--------|------------|------------|------------|------------|----------|
| ERM | 85.35 | 76.95 | 80.80 | 99.73 | 80.87 |
| SGT | 86.64 | 72.91 | 79.54 | 99.52 | 79.23 |
| DRE | 85.61 | 77.41 | 80.31 | 99.95 | 81.76 |
| **ALIGN** | **86.91** | **82.18** | 80.23 | **99.98** | **82.54** |

**Terra Incognita Dataset**:

| Method | Loc_38 Acc | Loc_43 Acc | Loc_46 Acc | Loc_100 Acc |
|--------|-----------|-----------|-----------|------------|
| ERM | 77.89 | 76.35 | 72.69 | 88.47 |
| DRE | 77.37 | 74.89 | 73.95 | 88.39 |
| **ALIGN** | **83.62** | 72.47 | **77.27** | **90.54** |

ALIGN achieves the best or highly competitive accuracy and AUC across most sub-domains, while also performing strongly on Sufficiency and Comprehensiveness metrics.

### Ablation Study

**Masker Ablation (VLCS VOC2007)**:

| Variant | Acc | AUC |
|---------|-----|-----|
| w/o EG (no explanation guidance) | 85.61 | 77.41 |
| m-SAM (SAM mask substitute) | 85.51 | 80.32 |
| m-Gray (grayscale mask) | 86.90 | 79.15 |
| **ALIGN** | **86.91** | **82.18** |

Key finding: Any external mask signal outperforms no EGL, but task-driven learned masks substantially outperform fixed masks.

### Key Findings

- **OOD generalization**: In settings where training is performed on VOC2007 and testing on other domains, ALIGN achieves the best performance in 5 out of 6 OOD configurations.
- Masks generated by general-purpose segmentation models such as SAM may focus on non-target regions (e.g., background environments), potentially misleading downstream tasks.
- Interpretability metrics (Sufficiency↓, Comprehensiveness↑) indicate that ALIGN produces more reliable attributions.

## Highlights & Insights

1. **Integration of theory and practice**: Generalization bound analysis under the PAC framework provides rigorous theoretical justification for the importance of mask quality, rather than relying solely on empirical evidence.
2. **Annotation-free EGL**: By replacing manual annotations or pretrained segmentation results with learned masks, ALIGN enables end-to-end explanation-guided learning.
3. **Warm-up strategy**: Training the classifier alone for the first 200 epochs prevents unstable early-stage masks from interfering with the training process.
4. **Dual mask regularization**: The combination of sparsity and smoothness constraints ensures that masks remain compact and spatially continuous rather than fragmented noise.

## Limitations & Future Work

- ALIGN does not achieve the best performance on certain sub-domains (e.g., LabelMe, Terra Loc_43), potentially because the learned masks miss a subset of relevant features (corresponding to Lemma 4).
- The masker is a lightweight convolutional network that may struggle to capture complex semantic structures; stronger architectures could be explored.
- Only Grad-CAM is employed as the explanation method; the effects of alternative attribution approaches (e.g., SHAP, Integrated Gradients) remain unexplored.
- Training and evaluation are conducted solely on classification tasks, leaving the effectiveness on dense prediction tasks such as detection and segmentation unverified.
- Alternating optimization may converge to local optima; joint end-to-end optimization has not been investigated.

## Related Work & Insights

- **EGL lineage**: The field has evolved from methods requiring manual annotations (CARE, GRADIA) → annotation-free methods relying on consistency (SGT, DRE) → ALIGN's self-learned masks, representing a clear progression.
- **Relationship to DRE**: ALIGN inherits the Mixup regularization idea from DRE but replaces DRE's fixed strategy with learned masks.
- **Domain generalization perspective**: Encouraging models to ignore spurious background features to improve OOD generalization resonates with the invariance principle in causal inference.
- The concept of learned mask guidance may inspire transfer to other tasks, such as domain generalization in object detection and semantic segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The EGL framework jointly learning masks and the classifier is novel, with solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two DG benchmarks, multiple ablations, and sufficient OOD experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theory, experiments, and methodology are logically coherent, with careful lemma derivations.
- Value: ⭐⭐⭐⭐ — Provides both a theoretical foundation and a practical solution for annotation-free EGL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion](jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)
- [\[CVPR 2026\] Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation](../../CVPR2026/segmentation/learning_and_aligning_click-aware_shape_prior_for_interactive_amodal_instance_se.md)
- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)
- [\[AAAI 2026\] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV](otter_mitigating_background_distractions_of_wide-angle_few-shot_action_recogniti.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](../../CVPR2025/segmentation/condensing_action_segmentation_datasets_via_generative_network_inversion.md)

</div>

<!-- RELATED:END -->
