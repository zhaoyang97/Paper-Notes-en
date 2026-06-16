---
title: >-
  [Paper Note] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment
description: >-
  [AAAI 2026][Interpretability][Blind Image Quality Assessment] This paper proposes the DR.Experts framework, which leverages DA-CLIP to obtain distortion-type priors…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Blind Image Quality Assessment"
  - "Distortion Prior"
  - "Mixture of Experts"
  - "DA-CLIP"
  - "Differential Attention"
date: 2026-05-08
content_hash: 62b768c597bc2556
---

# DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment

**Conference**: AAAI 2026
**arXiv**: [2602.09531](https://arxiv.org/abs/2602.09531)  
**Code**: [https://github.com/FuBohan01/DR.Experts](https://github.com/FuBohan01/DR.Experts)  
**Area**: Interpretability
**Keywords**: Blind Image Quality Assessment, Distortion Prior, Mixture of Experts, DA-CLIP, Differential Attention

## TL;DR

This paper proposes the DR.Experts framework, which leverages DA-CLIP to obtain distortion-type priors, employs a Distortion Saliency Differential Module (DSDM) to disentangle distortion attention from semantic attention and thereby purify distortion features, and then applies a Dynamic Distortion Weighting Module (DDWM) to adaptively weight each distortion type's features according to its perceptual impact. The method achieves state-of-the-art performance on five BIQA benchmarks.

## Background & Motivation

Blind Image Quality Assessment (BIQA) aims to evaluate visual quality without reference images, serving as a quality control component in image processing pipelines. Existing methods share a fundamental limitation: **the absence of reliable distortion priors** — they directly learn a shallow mapping from unified image features to quality scores and are insensitive to diverse distortion types and degrees.

Specific limitations:

**Distortion insensitivity**: Models treat all image features uniformly, yet real-world images may simultaneously exhibit multiple distortions (underexposure, noise, motion blur, etc.), each affecting perceptual quality in different ways and to different extents.

**Dataset constraints**: BIQA datasets are limited in scale and lack distortion-type annotations, restricting models' ability to learn fine-grained distortion representations.

**Limitations of CLIP-based methods**: Although methods such as CLIP-IQA introduce vision-language priors, the distortion attention extracted by DA-CLIP is contaminated by semantic redundancy noise inherited from classification pretraining.

Core Idea: Introduce distortion-type priors → differentially refine the priors (remove semantic noise) → aggregate features weighted by the perceptual impact of each distortion type, forming a mixture-of-experts system. The key insight is that distortion attention and semantic attention coexist in the features but are separable — a differential mechanism can "subtract away" semantic attention to purify distortion signals.

## Method

### Overall Architecture

An RGB image is simultaneously fed into a ViT Image Encoder and DA-CLIP. DA-CLIP obtains distortion-specific visual attention as priors using text prompts for 10 distortion types. DSDM differentially refines these priors by subtracting semantic attention from distortion attention. The purified distortion features, together with semantic features and bridging features, are passed to DDWM for dynamic weighting to produce the final quality score.

### Key Designs

1. **Distortion-Specific Prior Acquisition (DA-CLIP)**:

    - Function: Extracts visual features corresponding to 10 distortion types (motion-blurry, hazy, jpeg-compressed, low-light, noisy, raindrop, rainy, shadowed, snowy, uncompleted) from pretrained DA-CLIP.
    - Mechanism: DA-CLIP's image controller $\mathcal{E}_D$ produces a distorted image representation $E_{dis}$, and the text encoder $\mathcal{E}_T$ encodes distortion-type text. Per-distortion features are obtained via Hadamard product: $F_D^i = \text{Linear}^i(E_{dis} \odot E_T^i)$.
    - Design Motivation: DA-CLIP achieves 99.2% accuracy on 10-class distortion classification, indicating that it has learned strong distortion-aware representations. However, directly using these features for quality assessment yields poor results (some distortion types such as raindrop are unrelated to quality), necessitating further refinement.

2. **Distortion Saliency Differential Module (DSDM)**:

    - Function: Removes ViT semantic attention noise from DA-CLIP's distortion attention.
    - Mechanism: Inspired by Differential Transformer, a heterogeneous differential attention is designed. For the $i$-th distortion type: $F_{distortion}^i = (\text{softmax}(Q_D^i {K_{dis}^i}^T) - \alpha \cdot \text{softmax}(Q^i {K^i}^T))V^i$, where the first term is self-attention over distortion features, the second term is attention over semantic features, and $\alpha$ is a learnable parameter controlling the degree of subtraction. The value $V^i$ is obtained by projecting the concatenation of distortion and semantic features.
    - Design Motivation: Both DA-CLIP and ViT are pretrained on ImageNet classification and thus encode substantial semantic information. Since distortion attention = distortion-specific attention + semantic attention, subtracting the semantic component purifies the distortion signal.

3. **Dynamic Distortion Weighting Module (DDWM)**:

    - Function: Adaptively weights each distortion type according to its perceptual impact on quality.
    - Mechanism: Three feature groups are constructed — distortion features $F_{Group}$ (refined by DSDM), semantic features $F$ (from ViT), and bridging features $F_{bridging} = F - F_{Group}$. After concatenation, an MLP with PReLU activation generates 10 distortion-type weights: $W_{distortion}^1, \ldots, W_{distortion}^{10} = \text{WG}(F_{com})$. The final quality score is: $\text{Score} = \sum_{i=1}^{10} W_{distortion}^i \cdot T_{score}$.
    - Design Motivation: Different images are affected differently by different distortions — low-light degradation strongly impacts indoor photography, whereas haze distortion may be irrelevant. Dynamic weighting simulates the multi-dimensional quality evaluation process of the human visual system. Bridging features compensate for the information gap between distortion and semantic representations.

### Loss & Training

Smooth L1 Loss is used for training. The DA-CLIP module is frozen throughout. DeiT-III Small (ImageNet pretrained) serves as the Image Encoder. The model is trained for 9 epochs with an initial learning rate of $2\times10^{-4}$, decayed by a factor of 10 every 3 epochs. An 80%/20% train/test split is repeated 10 times and median results are reported. Training is conducted on 4× RTX 4090 GPUs. Standard random crop augmentation is applied to inputs.

## Key Experimental Results

### Main Results

| Method | KonIQ SRCC | KonIQ PLCC | LIVEC SRCC | LIVEC PLCC | SPAQ SRCC |
|--------|-----------|-----------|-----------|-----------|----------|
| HyperIQA | 0.906 | 0.917 | 0.859 | 0.882 | 0.916 |
| MUSIQ | 0.916 | 0.928 | 0.702 | 0.746 | 0.917 |
| QPT⋆ | 0.927 | 0.941 | 0.895 | 0.914 | 0.925 |
| QCN | 0.934 | 0.945 | 0.875 | 0.893 | 0.923 |
| LODA | 0.932 | 0.944 | 0.876 | 0.899 | 0.925 |
| LQMamba | 0.928 | 0.943 | 0.863 | 0.903 | 0.927 |
| **DR.Experts** | **0.941** | **0.954** | **0.914** | **0.926** | **0.928** |

### Ablation Study

| Configuration | KonIQ SRCC | LIVEC SRCC | Note |
|---------------|-----------|-----------|------|
| Image Encoder only | 0.916 | 0.857 | ViT semantic features only |
| DA-CLIP only | 0.720 | 0.587 | Direct use of distortion features → poor performance |
| +DSDM | 0.930 | 0.885 | Differential refinement yields significant improvement |
| Full (DR.Experts) | **0.941** | **0.914** | DDWM weighting provides further gains |

### Key Findings

- Directly applying DA-CLIP features to BIQA performs poorly (KonIQ SRCC = 0.720), as distortion types such as raindrop and uncompleted are unrelated or detrimental to quality. DSDM refinement substantially raises performance to 0.930.
- Pronounced data efficiency: with only 20% training data, DR.Experts (LIVEC SRCC = 0.837) outperforms all competing methods trained on 40% of the data.
- Generalization is confirmed via cross-dataset training-testing (e.g., LIVEFB → LIVEC), where DR.Experts achieves the highest SRCC across all configurations.
- Feature group ablation shows that using all three groups (distortion / semantic / bridging) yields the best results; removing any single group leads to a performance drop.
- Attention visualizations confirm that DSDM effectively suppresses irrelevant semantic-region attention and eliminates spurious distortion attention noise.

## Highlights & Insights

- The pipeline of distortion prior → differential refinement → expert weighting is logically coherent, with each step having a clearly demonstrated necessity.
- The paper innovatively extends the core idea of Differential Transformer (homogeneous differential attention) to heterogeneous attention (distortion vs. semantic), representing an elegant technology transfer.
- The framework offers strong interpretability: the final quality score can be traced back to specific distortion factor weights, enhancing the credibility of the assessment.
- The data efficiency advantage is particularly relevant to the BIQA community, where dataset scales are generally small.

## Limitations & Future Work

- The fixed set of 10 distortion types may not cover all real-world distortions (e.g., lens calibration errors, sensor noise patterns, HDR tone-mapping artifacts).
- The DA-CLIP module is entirely frozen, which may limit adaptation to the distortion distribution of target BIQA datasets.
- The bridging feature design $F_{bridging} = F - F_{Group}$ is relatively simple (plain subtraction); more sophisticated interaction mechanisms may yield better results.
- The training requirement of 4× RTX 4090 GPUs is relatively demanding for academic settings.
- The SRCC of 0.585 on the large-scale LIVEFB dataset leaves room for further improvement.

## Related Work & Insights

- Key distinction from CLIP-IQA: CLIP-IQA directly uses CLIP's prompt responses for quality assessment, whereas DR.Experts adds differential refinement and dynamic weighting.
- The differential attention idea from Differential Transformer (Ye et al., ICLR 2025) has broad transfer potential across multiple domains.
- The paradigm of prior-driven feature extraction followed by refinement is generalizable to other quality assessment tasks (video quality assessment, 3D content quality assessment).

## Rating
- Novelty: ⭐⭐⭐⭐ (Heterogeneous extension of differential attention is innovative; the overall framework is cleverly composed)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 datasets, 14 competing methods, comprehensive evaluation of generalization, data efficiency, and ablation)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (Meaningful advancement for the BIQA field; data efficiency advantage is particularly notable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](../../ICML2026/interpretability/iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](../../CVPR2026/interpretability/draft_and_refine_with_visual_experts.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](../../CVPR2026/interpretability/ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[ICML 2026\] The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level](../../ICML2026/interpretability/the_expert_strikes_back_interpreting_mixture-of-experts_language_models_at_exper.md)

</div>

<!-- RELATED:END -->
