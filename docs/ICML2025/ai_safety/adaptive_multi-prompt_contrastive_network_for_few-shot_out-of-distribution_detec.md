---
title: >-
  [Paper Note] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection
description: >-
  [ICML2025][AI Safety][OOD detection] An Adaptive Multi-prompt Contrastive Network (AMCN) is proposed to perform high-quality OOD detection under few-shot ID label conditions by generating three classes of adaptive textual prompts (learnable ID prompts, label-fixed OOD prompts, and label-adaptive OOD prompts) combined with class-adaptive thresholds, significantly outperforming existing few-shot OOD detection methods.
tags:
  - "ICML2025"
  - "AI Safety"
  - "OOD detection"
  - "few-shot learning"
  - "prompt learning"
  - "CLIP"
  - "contrastive learning"
  - "adaptive threshold"
date: 2026-05-08
content_hash: 4f0687fa432da0cb
---

# Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection

**Conference**: ICML2025  
**arXiv**: [2506.17633](https://arxiv.org/abs/2506.17633)  
**Code**: GitHub (authors claim open-source)  
**Area**: LLM/NLP  
**Keywords**: OOD detection, few-shot learning, prompt learning, CLIP, contrastive learning, adaptive threshold

## TL;DR

An Adaptive Multi-prompt Contrastive Network (AMCN) is proposed to perform high-quality OOD detection under few-shot ID label conditions by generating three classes of adaptive textual prompts (learnable ID prompts, label-fixed OOD prompts, and label-adaptive OOD prompts) combined with class-adaptive thresholds, significantly outperforming existing few-shot OOD detection methods.

## Background & Motivation

OOD detection aims to identify anomalous samples that deviate from the training distribution, preventing models from making incorrect predictions on unknown categories, which is crucial in safety-critical scenarios such as autonomous driving. However, most existing OOD detection methods rely on a large volume of in-distribution (ID) samples for training, severely limiting their practical application.

This paper targets a more challenging setting: **few-shot OOD detection**, where only a small number of labeled samples (e.g., 1 or 8) per ID class are available. This setting faces three core challenges:

**Background Sensitivity**: Few-shot training can easily bias the model towards image backgrounds. For instance, if cats are mostly photographed indoors and dogs outdoors, the model might rely on the background rather than the object itself to make decisions.

**Performance Degradation with Scaling Classes**: As the number of ID classes increases, the ID-OOD boundary becomes more complex, and overlaps in class features intensify, making it difficult for the model to learn fine-grained boundaries.

**Overfitting Risk**: With limited samples, models are highly susceptible to overfitting, severely restricting generalization capabilities.

Furthermore, the authors state that different classes exhibit **varying degrees of multi-diversity**. For example, the sample diversity of the "cat" class is much higher than that of the "ox" class. However, existing methods neglect this inter-class diversity variance and use a uniform threshold for OOD determination, leading to sub-optimal performance.

## Method

### Overall Architecture

AMCN is built upon the CLIP vision-language pre-trained model and consists of three core modules:

1. **Adaptive Prompt Generation**: Generates three types of textual prompts for ID classification.
2. **Prompt-based Multi-diversity Distribution Learning**: Learn the intra-class/inter-class distributions of each category to generate adaptive thresholds.
3. **Prompt-guided OOD Detection**: Achieves precise detection through ID-OOD separation loss and multi-prompt contrastive learning.

### Key Designs

#### Key Design 1: Three Types of Adaptive Prompts

To compensate for the lack of OOD samples and the scarcity of ID samples, three types of prompts are designed utilizing the text-image connection capabilities of CLIP:

* **Learnable ID Prompt (LIP)**: $f_{lip}^i = [W_1][W_2]\dots[W_{N_{IP}}][y_i]$, where the prefix tokens are learnable, and the label name is fixed to the ID class name.
* **Label-Fixed OOD Prompt (LFOP)**: $f_{lfop}^i = [M_1]\dots[M_{N_{lfop}}][o_i]$, where the prefix is learnable, and the label is fixed to OOD class names from external datasets (e.g., the class "chair" in CIFAR-100 that does not overlap with ID classes).
* **Label-Adaptive OOD Prompt (LAOP)**: $f_{laop}^i = [H_1]\dots[H_{N_{laop}}][o_i']$, where both the prefix and label are learnable, and the label is initialized by OOD classes and optimized freely.

The three types of prompts share the text encoder parameters. During training, all OOD prompts serve as negative samples for multi-prompt contrastive learning with ID image features. The ID classification loss $\mathcal{L}_C$ uses a weighted cross-entropy form, which pulls the ID prompts closer to the ID images and pushes OOD prompts away through temperature-scaled cosine similarity.

#### Key Design 2: Class-Adaptive Thresholds and Distribution Learning

Different categories exhibit diverse variations (e.g., the t-SNE distributions of "cat" and "ox" in ImageNet-1k show significant differences), making a uniform threshold sub-optimal for all categories. AMCN learns independent thresholds for each class:

- Compute class distribution score $\mathbb{S}_c(x_i) = \exp(o_c(x_i)) / (\tau_0 + \mathcal{M}_c^{pse})$
- Estimate the intra-class mean $\mu_c$ and standard deviation $\sigma_c$
- Define the **P-score** adaptive threshold: $P_c = \lambda \cdot \mu_c + (1-\lambda) \cdot \sigma_c$
- When $\mathbb{S}_c(x_i) > P_c$, it is classified as pseudo-OOD; if a sample is pseudo-OOD for all classes, it is deemed true OOD.

Simultaneously, a momentum update mechanism for the pseudo-OOD distribution $\mathcal{M}_c^{pse}$ is introduced, dynamically adjusting online during inference.

#### Key Design 3: Prompt-Guided ID-OOD Separation

To distinctly clear the boundary between ID and OOD prompt features, two key losses are designed:

- **ID-OOD Separation Loss** $\mathcal{L}_2$: Restricts the Euclidean distance between the OOD prototype and the ID image on the unit hypersphere to be greater than the distance between the ID prototype and the ID image, ensuring OOD features are pushed away.
- **OOD Alignment Loss** $\mathcal{L}_3$: Constrains the feature prototypes of LAOP (learnable labels) and LFOP (fixed labels) to align in the normalized space, preventing LAOP from drifting away from the OOD semantic space towards the ID semantics.

### Loss & Training

The total loss consists of four parts: $\mathcal{L} = \mathcal{L}_1 + \alpha_1 \mathcal{L}_2 + \alpha_2 \mathcal{L}_3 + \alpha_3 \mathcal{L}_4$

| Loss | Role | Description |
|------|------|------|
| $\mathcal{L}_C$ | ID Classification | Weighted multi-prompt contrastive learning |
| $\mathcal{L}_I^1$ | Intra-class Distribution Normalization | Balancing ID/OOD score ratios across classes |
| $\mathcal{L}_I^2$ | Inter-class Distribution Normalization | Balancing ID/OOD distributions across classes |
| $\mathcal{L}_2$ | ID-OOD Separation | Ensuring OOD prototype remains far from ID images |
| $\mathcal{L}_3$ | OOD Alignment | Constraining LAOP and LFOP semantic alignment away from ID |
| $\mathcal{L}_4$ | OOD Detection Contrastive Learning | Minimizing ID-OOD prompt similarity |

Where $\mathcal{L}_1 = \mathcal{L}_C + \mathcal{L}_I^1 + \mathcal{L}_I^2$. Hyperparameter settings: $\alpha_1=0.4, \alpha_2=0.2, \alpha_3=0.8$, utilizing AdamW optimizer with a learning rate of 0.003, batch size is 64, token length is 16, and training is conducted for 100 epochs. The counts of LFOP and LAOP are both set to 50.

## Key Experimental Results

### Main Results: Few-shot OOD Detection Performance Comparison (ImageNet-1k as ID)

| Method | Shot | Texture FPR95↓ | Texture AUROC↑ | Places FPR95↓ | Places AUROC↑ | SUN FPR95↓ | SUN AUROC↑ | iNat FPR95↓ | iNat AUROC↑ | Avg FPR95↓ | Avg AUROC↑ |
|------|------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|
| KNN | Full | 64.35 | 85.67 | 39.61 | 91.02 | 35.62 | 92.67 | 29.17 | 94.52 | 42.19 | 90.97 |
| NPOS | Full | 46.12 | 88.80 | 45.27 | 89.44 | 43.77 | 90.44 | 16.58 | 96.19 | 37.94 | 91.22 |
| GL-MCM | 0 | 57.93 | 83.63 | 38.85 | 89.90 | 30.42 | 93.09 | 15.16 | 96.71 | 35.59 | 90.83 |
| SCT | 1 | 48.87 | 86.66 | 32.81 | 91.23 | 23.52 | 94.58 | 19.16 | 95.70 | 31.09 | 92.04 |
| **AMCN** | **1** | **39.16** | **89.88** | **32.76** | **92.78** | **23.26** | **94.85** | **18.84** | **96.18** | **30.87** | **92.47** |
| SCT | 8 | 40.35 | 91.82 | 38.77 | 92.41 | 23.48 | 94.77 | 18.65 | 95.82 | 32.32 | 93.53 |
| **AMCN** | **8** | **38.31** | **93.43** | **32.45** | **93.96** | **23.17** | **95.89** | **18.17** | **96.89** | **30.56** | **94.29** |

- Under the 1-shot setting, AMCN's FPR95 on Texture is reduced by **9.71%** and AUROC is improved by **3.22%** compared to SCT.
- Under the 8-shot setting, the average AUROC of AMCN reaches 94.29%, outperforming all fully-supervised baselines.
- Remarkably, in the 1-shot setting, AMCN outperforms methods that use full training data, such as KNN, ViM, and ODIN.

### Ablation Study: Contribution of Each Module (Texture Dataset)

| M1 (Prompt Generation) | M2 (Distribution Learning) | M3 (OOD Detection) | 1-shot FPR95↓ | 1-shot AUROC↑ | 8-shot FPR95↓ | 8-shot AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✔ | ✔ | 41.36 | 82.59 | 40.82 | 86.24 |
| ✔ | ✗ | ✔ | 40.95 | 83.24 | 40.26 | 86.88 |
| ✔ | ✔ | ✗ | 40.35 | 83.19 | 39.72 | 87.92 |
| ✔ | ✔ | ✔ | **39.16** | **89.88** | **38.31** | **93.43** |

All three modules are indispensable; removing any module results in a 5-7 percentage point drop in AUROC, indicating that adaptive prompt, distribution learning, and ID-OOD separation form an effective synergy.

### Adaptive Threshold vs Fixed Threshold (iNaturalist)

| Threshold Type | 1-shot FPR95↓ | 1-shot AUROC↑ | 8-shot FPR95↓ | 8-shot AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|
| Fixed | 20.70 | 93.83 | 20.51 | 94.16 |
| **Adaptive** | **18.84** | **96.18** | **18.17** | **96.89** |

The adaptive threshold brings 2.35%/2.73% AUROC improvements, demonstrating the necessity of class-level decision boundaries.

## Highlights & Insights

1. **Ingenious complementary design of three prompt types**: LIP handles ID representation learning, LFOP introduces human prior OOD knowledge, and LAOP explores potential OOD semantic space. Together, they construct the ID-OOD separation boundary from different perspectives.
2. **Class-level adaptive threshold is a core contribution**: Recognizing the diversity variance across different classes and modeling it using P-score (incorporating both intra-class mean and standard deviation) yields substantial improvements over a global, fixed threshold.
3. **No OOD training data required**: Completely free from utilizing OOD image samples, constructing OOD proxies solely through text prompts, which fully exploits the cross-modal capability of CLIP.
4. **1-shot performance outperforms fully-supervised baselines**: The performance with just 1 labeled sample exceeds that of KNN/ViM/ODIN trained on complete training data, demonstrating the high efficacy of the proposed method.
5. **Hyperparameter insensitivity**: Sensitivity analysis of $\alpha_1, \alpha_2, \alpha_3$ shows that the model maintains stable performance across a wide range of values.

## Limitations & Future Work

1. **Dependency on CLIP pre-training quality**: The method builds on CLIP-ViT-B/16; if a target domain differs significantly from CLIP’s pre-training distribution (e.g., medical imaging), performance might be constrained.
2. **OOD label sources require prior knowledge**: LFOP requires obtaining OOD category names from other datasets. This step still relies on manual selection, lacking automated selection.
3. **Validated only on image classification**: The method has not been evaluated on more complex vision tasks like object detection or semantic segmentation, nor on non-vision modalities.
4. **Undiscussed computational overhead**: Jointly training three types of prompts and multiple loss functions incurs higher training costs compared to simpler methods (e.g., MCM, GL-MCM).
5. **Inference stability of pseudo-OOD online updates**: The momentum update mechanism might lack stability under extreme few-shot scenarios (e.g., 1-shot), and the paper does not thoroughly analyze its convergence.

## Related Work & Insights

- **OOD Detection**: ODIN (Liang et al., 2018), ViM (Wang et al., 2022), KNN (Sun et al., 2022), NPOS (Tao et al., 2023), MCM (Ming et al., 2022), progressing across classification, density, distance, and reconstruction methods.
- **Few-shot OOD Detection**: LoCoOp (Miyai et al., 2023), CoOp (Zhou et al., 2022), SCT (Yu et al., 2024) leverage prompt learning for few-shot OOD detection, but neglect differences in inter-class diversity.
- **Prompt Learning**: Extending from NLP prompts (GPT/BERT) to visual domains (CoOp/CoCoOp), but most assume a closed-set scenario, which is inherently inapplicable to OOD detection.
- **CLIP and OOD**: GL-MCM and SeTAR leverage CLIP for zero-shot OOD detection, whereas this work extends further to the few-shot paradigm and introduces OOD prompt construction.

## Rating

| Dimension | Score (1-5) | Description |
|------|:---:|------|
| Novelty | 4 | Novel combinatorial design of three prompt types and adaptive thresholds |
| Technical Depth | 4 | Complete multi-loss function design with a theoretically motivated distribution learning mechanism |
| Experimental Thoroughness | 4 | 4 OOD datasets + multi-shot settings + extensive ablations |
| Writing Quality | 3 | Generally clear but possesses heavy mathematical notations, with some inconsistencies in symbols |
| Value | 3 | Effective approach but relies on CLIP and OOD label priors |
| **Overall Score** | **3.6** | A solid piece of work on few-shot OOD detection with clear core contributions |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[ACL 2025\] Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](../../ACL2025/ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)
- [\[CVPR 2026\] RankOOD: Class Ranking-based Out-of-Distribution Detection](../../CVPR2026/ai_safety/rankood_-_class_ranking-based_out-of-distribution_detection.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](../../CVPR2025/ai_safety/h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](../../CVPR2025/ai_safety/leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
