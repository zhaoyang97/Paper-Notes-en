---
title: >-
  [Paper Note] Do Multiple Instance Learning Models Transfer?
description: >-
  [ICML2025 Spotlight][Medical Imaging][Multiple Instance Learning] This work presents the first systematic evaluation of the transfer learning capabilities of MIL models in computational pathology, finding that MIL models pre-trained on a pancancer dataset can generalize across organs and tasks, outperforming self-supervised slide foundation models (CHIEF, GigaPath) using less than 10% of the pre-training data.
tags:
  - "ICML2025 Spotlight"
  - "Medical Imaging"
  - "Multiple Instance Learning"
  - "Transfer Learning"
  - "Computational Pathology"
  - "Slide Foundation Model"
  - "Pancancer Pretraining"
date: 2026-05-08
content_hash: 54c036a67f5d320b
---

# Do Multiple Instance Learning Models Transfer?

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.09022](https://arxiv.org/abs/2506.09022)  
**Code**: [mahmoodlab/MIL-Lab](https://github.com/mahmoodlab/MIL-Lab)  
**Area**: Medical Image  
**Keywords**: Multiple Instance Learning, Transfer Learning, Computational Pathology, Slide Foundation Model, Pancancer Pretraining

## TL;DR

This work presents the first systematic evaluation of the transfer learning capabilities of MIL models in computational pathology, finding that MIL models pre-trained on a pancancer dataset can generalize across organs and tasks, outperforming self-supervised slide foundation models (CHIEF, GigaPath) using less than 10% of the pre-training data.

## Background & Motivation

**Core Problem**: Multiple Instance Learning (MIL) is the cornerstone paradigm for processing gigapixel whole-slide images (WSIs) in computational pathology. However, constrained by small-scale, weakly-supervised clinical datasets, model performance remains limited. Although transfer learning has been widely used to address data scarcity in NLP and traditional CV, the transferability of MIL models has rarely been studied—currently, random initialization remains the standard practice for MIL model development and evaluation.

**Research Motivation**:

- Although transfer learning for patch-level encoders (UNI, Virchow, etc.) has been widely adopted, transfer learning for slide-level aggregators is completely neglected.
- Self-supervised slide foundation models (CHIEF, GigaPath) require tens or hundreds of thousands of WSIs for pre-training, which introduces extremely high data and computational costs.
- The authors hypothesize that a MIL model supervisedly pre-trained on a large-scale multi-class pancancer classification task can serve as a simple yet effective alternative to slide foundation models.

**Review of MIL Workflow**: Given a WSI, it is first segmented into patches to extract features ($\sim$1000–10000 patches) using a pre-trained patch encoder, then all patch features are pooled into a slide-level representation through a trainable aggregator for downstream classification.

## Method

### Experimental Framework: Supervised MIL Transfer

For MIL architecture $f$, pre-training task $s$, and target task $t$, this research answers three core questions:

1. **$f_{s \to t}$ vs. $f_{\text{rand} \to t}$**: Does pre-training outperform training from scratch?
2. **$f_{s \to t}$ vs. $f_{s' \to t}$**: How do different pre-training tasks perform in transfer?
3. **$f_{s \to t}$ vs. $f'_{s \to t}$**: What are the differences in transfer capacity across different architectures?

### Evaluation Settings

- **11 MIL Architectures**: ABMIL, CLAM, DSMIL, DFTD, TransMIL, Transformer, ILRA, RRT, WIKG, MeanMIL, MaxMIL
- **21 Pre-training Tasks + 19 Target Tasks**: Spanning 4 organs (breast, lung, prostate, brain), including cancer classification, grading, molecular subtyping prediction, etc.
- **Pancancer Pre-training Tasks**: PC-43 (43 classes) and PC-108 (108-class OncoTree codes), consisting of 3,499 WSIs from 17 organs.
- **Two Evaluation Protocols**: End-to-end fine-tuning and frozen-feature KNN evaluation.

### Standardized Implementation

- Patch Partitioning: $256 \times 256$, size at 20× magnification (0.5 μm/pixel)
- Patch Encoder: UNI (DINOv2 pre-trained ViT-L/16)
- Optimizer: AdamW, learning rate $1 \times 10^{-4}$, cosine decay
- Maximum of 20 epochs, early stopping with patience = 5

## Key Experimental Results

### Quality Comparison of Pre-training Tasks (KNN Frozen Feature Evaluation)

| Pre-training Strategy | Average Gain Relative to Baseline |
|:---|:---|
| PC-108 pancancer | **+9.8%** |
| PC-43 pancancer | +8.6% |
| Single-organ task (in-domain) | +3–6% |
| Single-organ task (out-of-domain) | +1–4% |
| Randomly initialized baseline | 0% |

**Key Findings**: Even cross-organ pre-training (e.g., Lung $\to$ Breast) yields outstanding improvements.

### Fine-Tuning Transfer of 11 Architectures (PC-108 Pre-training vs. Random Initialization)

| Architecture | Random Initialization | PC-108 | Δ |
|:---|:---|:---|:---|
| ABMIL | 71.7 | 75.5 | **+3.8** |
| DFTD | 69.6 | 76.6 | **+7.0** |
| TransMIL | 68.1 | 73.9 | **+5.8** |
| Transformer | 68.5 | 74.3 | **+5.8** |
| DSMIL | 72.3 | 73.0 | +0.7 |
| CLAM | 69.0 | 70.5 | +1.5 |
| WIKG | 69.3 | 74.7 | +5.4 |
| Average of All Models | 70.1 | 73.4 | **+3.3** |

### Few-Shot Learning (K=4,16,32 samples/class)

- Under K=4, PC-108 pre-training yields a **171%** improvement for DFTD compared to random initialization.
- Across all fold/shot numbers, pancancer pre-training consistently outperforms random initialization across all 5 methods.
- PC-108 systematically excels over PC-43, indicating that fine-grained classification tasks lead to better data efficiency.

### Comparison with Slide Foundation Models

| Metric | PC-108 ABMIL | CHIEF | GigaPath |
|:---|:---|:---|:---|
| Pre-training Data Volume | **3,944 WSI** | 60,530 WSI | 171,189 WSI |
| Pre-training Protocol | Supervised Classification | Contrastive Learning + CLIP | Self-supervised MAE |
| Number of Wins (KNN) | **12/15** vs. CHIEF | 3/15 | 2/15 |
| Number of Wins (Fine-tuning) | **11/15** vs. CHIEF | 4/15 | 5/15 |
| KNN Average Gain | — | +5.9% over CHIEF | +9.7% over GigaPath |

PC-108 achieves better results on the majority of tasks using only 6.5% of the pre-training data of CHIEF and 2.3% of GigaPath.

### Model Scale and Transfer

- Random initialization leads to high performance fluctuation across different model scales.
- Under PC-108 pre-training, performance scales monotonically from 0.1M to 5M parameters, demonstrating a promising scaling trend.
- At 9M parameters, the performance drops slightly, yet still significantly outperforms random initialization.

### Analysis of Critical Components in Transfer

Through layer-wise reset experiments (ABMIL 4-layer structure):

| Reset Strategy | Performance Drop Relative to Full Transfer |
|:---|:---|
| Reset Attention Layer | **-5.0%** |
| Reset Attention + Linear Layer 3 | -5.2% |
| Reset Attention + Linear Layer 2+3 | -6.6% |
| Complete Reset (= Random Initialization) | -8.3% |

**The Attention aggregation layer is the core carrier of transferred knowledge**, which differs from the conclusion in CNN transfer where deep layers are less critical.

## Highlights & Insights

1. **"Pre-training matters more than architecture"**: The best randomly-initialized architecture (DSMIL, 72.3) performs worse than 9 out of 11 architectures after pre-training, demonstrating that a good initialization is much more important than a complex architecture.
2. **Simple architecture + Good initialization = Optimal**: ABMIL, being the simplest attention pooling method, performs the best after pre-training, validating the effectiveness of "strong patch encoder + simple aggregator".
3. **Supervised pancancer pre-training > Large-scale self-supervised learning**: Overperforming foundation models pre-trained on 60k–170k WSIs using extremely limited data (~4k WSIs) suggests that well-designed classification tasks are more effective than simply hoarding data.
4. **Attention heatmap visualization**: The pre-trained model focuses on tumor regions even before fine-tuning, whereas the attention of the randomly initialized model remains dispersed. Pre-training helps the model avoid shortcuts and spurious correlations.
5. **Consistent effectiveness across patch encoders**: Improvements from PC-108 pre-training are consistently observed across five different encoders: ResNet-50, CTransPath, GigaPath ViT, UNIv2, and CONCHv1.5.

## Limitations & Future Work

1. **Missing State-Space MIL models**: Architectures like the Mamba series were not included in the evaluation.
2. **Survival prediction tasks not evaluated**: The evaluation only covers classification/grading, leaving survival analysis (e.g., Cox regression) unexplored.
3. **Single-source pre-training data**: PC-108 is entirely derived from Brigham and Women's Hospital, which may introduce institutional bias.
4. **Advanced pre-training strategies unexplored**: Techniques such as data augmentation or hybrid self-supervised & supervised pre-training might yield further improvements.
5. **Frozen patch encoder**: The pre-trained patch encoder was kept frozen throughout, omitting explorations into end-to-end joint fine-tuning.

## Related Work & Insights

- **Patch Foundation Models**: UNI (Chen et al., 2024), Virchow (Vorontsov et al., 2024) — This work focuses on slide-level transfer, which is complementary to patch-level transfer.
- **Slide Foundation Models**: CHIEF (Wang et al., 2024), GigaPath (Xu et al., 2024) — This work demonstrates that supervised pre-training can serve as a more efficient alternative.
- **MIL Architectures**: ABMIL $\to$ CLAM $\to$ TransMIL $\to$ WIKG — This work discovers that architectural differences have limited impacts on performance after transfer.
- **Transfer Learning in NLP/CV**: The ImageNet pre-training paradigm — This work positions PC-108 as the "ImageNet" of computational pathology.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to systematically investigate transfer learning in pathological MIL, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely large-scale and rigorously designed, spanning 11 architectures $\times$ 21 tasks $\times$ multiple encoders.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, solid conclusions, and highly informative figures/tables.
- **Value**: ⭐⭐⭐⭐⭐ — Directly beneficial and highly practical for the pathology AI community, with open-sourced weights and code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](../../CVPR2025/medical_imaging/mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [\[ICLR 2026\] Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in Multiple Instance Learning](../../ICLR2026/medical_imaging/mixture_of_mini_experts_overcoming_the_linear_layer_bottleneck_in_multiple_insta.md)
- [\[ICLR 2026\] ASMIL: Attention-Stabilized Multiple Instance Learning for Whole-Slide Imaging](../../ICLR2026/medical_imaging/asmil_attention-stabilized_multiple_instance_learning_for_whole-slide_imaging.md)
- [\[CVPR 2026\] Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification](../../CVPR2026/medical_imaging/contrastive_cross-bag_augmentation_for_multiple_instance_learning-based_whole_sl.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](../../CVPR2026/medical_imaging/every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)

</div>

<!-- RELATED:END -->
