---
title: >-
  [Paper Note] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective
description: >-
  [AAAI 2026][Image Generation][Multimodal misinformation detection] This paper proposes RetSimd, which "replays the whole story" by segmenting text and generating a series of supplementary images via a text-to-image model…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Multimodal misinformation detection"
  - "text-to-image generation"
  - "graph neural network"
  - "modality contribution"
  - "information gain"
date: 2026-05-08
content_hash: 5ccca7c9a3544245
---

# Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective

**Conference**: AAAI 2026
**arXiv**: [2511.06284](https://arxiv.org/abs/2511.06284)
**Code**: [https://github.com/wangbing1416/RETSIMD](https://github.com/wangbing1416/RETSIMD)
**Area**: Image Generation / Multimodal
**Keywords**: Multimodal misinformation detection, text-to-image generation, graph neural network, modality contribution, information gain

## TL;DR
This paper proposes RetSimd, which "replays the whole story" by segmenting text and generating a series of supplementary images via a text-to-image model, combined with a graph neural network to fuse multi-image relationships. The approach significantly enhances the contribution of the image modality to misinformation detection, consistently improving the performance of five SOTA methods across three benchmark datasets.

## Background & Motivation

Multimodal misinformation detection (MMD) aims to determine whether information in social media posts — containing both text and images — constitutes misinformation. Prevailing methods typically assume that the text and image modalities are equally informative, relying primarily on fusing features from both modalities or learning cross-modal semantic inconsistencies.

However, the authors identify a critical observation: **the text modality carries substantially more information than the image modality**. Text typically describes the complete story of an event, whereas the accompanying image tends to depict only a partial scene (e.g., a news report on a hurricane may describe the entire disaster, while the image captures only a single collapsed building).

To substantiate this claim, the authors conduct systematic preliminary ablation experiments:
- Five variants — full / text-only / image-only / text-replaced / image-replaced — are evaluated on five SOTA methods.
- Results show that removing images (text-only) causes **far smaller** performance degradation than removing text (image-only); in some cases, image-only variants are nearly ineffective.
- The contribution gap between the two modalities is further quantified using information-theoretic measures (information gain).

**Key Challenge**: Insufficient information in the image modality degrades multimodal detection performance. **Key Insight**: Enriching the image modality by generating additional images that more completely represent the story described in the text.

## Method

### Overall Architecture
RetSimd comprises four core modules:
1. Feature encoders (BERT + ResNet34)
2. Text-to-image generator (based on Stable Diffusion)
3. Multimodal fusion network (graph neural network)
4. Veracity classifier

The generator and detector are optimized alternately during training.

### Key Designs

1. **Text Segmentation and Supplementary Image Generation**:

    - **Function**: The text $\mathbf{x}^t$ is divided into $K$ segments using a fixed-length sliding window; each segment is fed into a pretrained Stable Diffusion model to generate a corresponding image.
    - **Mechanism**: Each text segment describes a partial scene of the story, and the resulting image sequence "replays" the entire narrative.
    - **Design Motivation**: To compensate for the limitation of a single original image, which can depict only part of the story, thereby making the image modality as informationally rich as the text.

2. **Information-Theoretic Regularization for the Generator**:

    - **Function**: Two mutual information objectives are designed to fine-tune the generator.
    - **Text–Image Mutual Information $\mathcal{R}_{MTI}$**: Different segments of the same text serve as natural positive/negative pairs to constrain semantic consistency between generated images and their corresponding text segments; an adaptive weight $\xi_{jm}$ is introduced to account for temporal relationships among segments.
    - **Image–Label Mutual Information $\mathcal{R}_{MIL}$**: Maximizes the information gain $\mathscr{G}(y_i|\{\mathbf{x}_{ij}^g\})$ of the generated image sequence with respect to the veracity label.
    - The generator is additionally continually post-trained on the LAION-2B dataset to preserve image quality.
    - **Design Motivation**: To ensure that supplementary images are both semantically coherent and genuinely useful for misinformation detection.

3. **Graph-Structured Multimodal Fusion**:

    - **Function**: The original image and $K$ supplementary images are treated as nodes in a graph; a GNN learns fused representations over this structure.
    - Three heuristic edge types:
        - **Center relation**: All supplementary images are connected to the original image (which depicts the most central scene).
        - **Temporal relation**: Adjacent supplementary images are connected in the order of their corresponding text segments.
        - **Semantic relation**: Cross-attention is used to compute inter-image semantic similarity; semantically similar images are connected.
    - The fused image features are then combined with text features via cross-attention to produce the final representation.
    - **Design Motivation**: To capture latent relationships among supplementary images, yielding more effective fusion than simple concatenation.

### Loss & Training
- Generator objective: $\mathcal{L}_{GEN} = \mathcal{L}_{T2I} + \alpha_1 \mathcal{R}_{MTI} + \alpha_2 \mathcal{R}_{MIL}$
- Detector objective: $\mathcal{L}_{DET} = \mathcal{L}_{VC} + \beta \mathcal{R}_{CA}$ (cross-entropy + cross-attention regularization)
- Alternating optimization: the generator is trained with the detector fixed, then the detector is trained with the generator fixed.

## Key Experimental Results

### Main Results

On the GossipCop dataset, RetSimd consistently improves all five baseline methods:

| Baseline | Orig. Acc | +RetSimd Acc | Gain Δ | Image Info. Gain |
|----------|-----------|--------------|--------|-----------------|
| ResNet+BERT | 87.17 | **88.13*** | +1.21 | 0.0349→0.0301 |
| R&B+SAFE | 87.14 | **88.30*** | +1.29 | 0.0325→0.0287 |
| R&B+MCAN | 87.29 | **88.22*** | +1.12 | 0.0200→0.0129 |
| R&B+CAFE | 87.16 | **88.38*** | +1.09 | 0.0439→0.0402 |
| R&B+BMR | 87.32 | **88.42*** | +1.02 | 0.0458→0.0409 |
| R&B+GAMED | 87.03 | **88.30*** | +1.81 | 0.0350→(improved) |

*\* denotes statistical significance at p-value < 0.05*

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full RetSimd | 88.42 Acc | Best overall performance |
| w/o $\mathcal{R}_{MTI}$ | Performance drop | Text–image alignment is critical for generation quality |
| w/o $\mathcal{R}_{MIL}$ | Performance drop | Image–label information gain ensures detection utility |
| Fixed-length vs. semantic segmentation | Fixed-length superior | Simpler strategy yields more stable results |
| Varying $K$ (number of segments) | $K=4$ or $K=5$ optimal | Too few provides insufficient information; too many introduces noise |
| w/o GNN graph fusion | Performance drop | Graph-structured relationships are important for fusion |

### Key Findings
- RetSimd not only improves final detection accuracy but also **reduces the information gain gap of the image modality** (i.e., images become more informative).
- Consistent cross-dataset improvements are observed on the Weibo and Twitter datasets as well.
- There exists an optimal value for the number of supplementary images $K$; too many images may introduce noise.
- All improvements pass statistical significance tests (p < 0.05).

## Highlights & Insights
- **Novel problem framing**: The first systematic application of information theory to quantify modality contribution imbalance in MMD.
- **Generation for detection**: The text-to-image model is employed not to synthesize content for its own sake, but to enrich the input to the detection model.
- **Plug-and-play**: RetSimd is a framework-level method that can be applied on top of any MMD baseline.
- The three graph edge types (center / temporal / semantic) are well-motivated and capture the core relationships among images.
- The proposed contribution metric provides an analytical tool for understanding multimodal systems.

## Limitations & Future Work
- The generator relies on Stable Diffusion, incurring substantial inference overhead (generating $K$ images per sample).
- Image generation quality is bounded by SD's zero-shot capability and may be insufficient for news-specific scenes.
- The work addresses only binary classification (real/fake) without considering finer-grained misinformation types.
- The fixed-length segmentation strategy is simple but may fragment semantic units.
- Robustness under adversarial attacks is not discussed.

## Related Work & Insights
- The observation of modality contribution imbalance is likely generalizable and may apply to other multimodal tasks.
- The paradigm of using generative models to augment discriminative models is broadly applicable, especially when a particular modality is informationally deficient.
- The information-theoretic contribution analysis framework can be extended to other settings requiring an understanding of modality interaction.
- The graph-based multi-image fusion approach is transferable to tasks such as video understanding that require integrating information across multiple frames.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel perspective (image modality insufficiency + generative augmentation), though individual technical components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, five baselines, information-theoretic analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is well-argued; preliminary experiment design is elegant.
- Value: ⭐⭐⭐⭐ — The plug-and-play framework offers practical utility; the information-theoretic analysis is insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective](unseen_enhancing_dataset_pruning_from_a_generalization_perspective.md)
- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](../../ICLR2026/image_generation/when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](../../CVPR2026/image_generation/enhancing_image_aesthetics_with_dualconditioned_di.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

</div>

<!-- RELATED:END -->
