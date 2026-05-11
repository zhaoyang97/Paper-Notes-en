---
title: >-
  [Paper Note] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers
description: >-
  [AAAI 2026][Video Understanding][Knowledge Distillation] This paper proposes a "Distillation Dynamics" analytical framework (channel-wise FFT spectral analysis + Shannon entropy + activation magnitude tracking) to reveal…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Knowledge Distillation"
  - "Vision Transformer"
  - "Spectral Analysis"
  - "Information Bottleneck"
  - "Negative Transfer"
date: 2026-05-08
content_hash: 66155f8d8e8c9b95
---

# Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers

**Conference**: AAAI 2026
**arXiv**: [2511.06848](https://arxiv.org/abs/2511.06848)
**Code**: [GitHub](https://github.com/thy960112/Distillation-Dynamics)
**Area**: Video Understanding
**Keywords**: Knowledge Distillation, Vision Transformer, Spectral Analysis, Information Bottleneck, Negative Transfer

## TL;DR

This paper proposes a "Distillation Dynamics" analytical framework (channel-wise FFT spectral analysis + Shannon entropy + activation magnitude tracking) to reveal that ViTs exhibit a distinctive U-shaped information processing pattern (compression followed by expansion). The work demonstrates that the fundamental cause of feature-based distillation failure in ViTs is a representational paradigm mismatch between the teacher's distributed high-dimensional encoding in later layers and the student's limited channel capacity—rather than a simple capacity gap.

## Background & Motivation

Feature-based knowledge distillation methods (e.g., FitNet, AT) have proven highly successful for CNN compression—by training smaller models to mimic intermediate feature representations of larger ones, substantial performance gains can be achieved. Yet a puzzling phenomenon exists: these same methods not only fail on Vision Transformers but actually perform worse than simple logit-based distillation.

While works such as ViTKD have observed this phenomenon and proposed ViT-specific distillation strategies, none have explained **why** the success of CNN distillation does not transfer to ViTs. This theoretical gap severely constrains the design of ViT compression strategies, leaving researchers to rely on empirical trial and error without theoretical guidance.

The paper's starting point is: rather than rushing to propose a new distillation method, it first seeks a thorough understanding of ViT's internal information processing mechanisms to identify the root cause of feature distillation failure. Three complementary analytical tools are designed to "triangulate" the intrinsic properties of ViT representations from different perspectives.

## Method

### Overall Architecture

A three-dimensional analytical framework termed "Distillation Dynamics" is proposed: (1) channel-wise FFT spectral analysis to reveal feature encoding strategies; (2) Shannon entropy analysis to quantify per-layer information complexity; and (3) activation magnitude tracking to monitor signal propagation strength. Cross-validation across these three perspectives ensures that observed patterns are not artifacts of any single measurement. Building on this analysis, SpectralKD and ProjectorKD are designed as validation methods for the analytical conclusions.

### Key Designs

1. **Channel-Wise FFT Spectral Analysis**

    - Function: Reveals the encoding strategy of features at each ViT layer.
    - Mechanism: For the activation tensor $\mathbf{A} \in \mathbb{R}^{L \times B \times C \times H \times W}$ at each layer, a 1D FFT is applied **along the channel axis** (rather than the conventional spatial axis): $\mathbf{F}_{l,b,h,w}[k] = \frac{1}{C}\sum_{c=0}^{C-1}\mathbf{A}_{l,b,c,h,w} e^{-j2\pi kc/C}$. Low-frequency dominance indicates high inter-channel correlation (compact encoding), while uniform high-frequency content indicates channel decorrelation (distributed encoding). Averaging over batch and spatial dimensions yields a per-layer spectral signature $\mathbf{S} \in \mathbb{R}^{L \times C}$.
    - Design Motivation: Spatial FFT only captures spatial frequency characteristics of feature maps (already well-studied), whereas channel-wise FFT reveals **the structure of the feature space itself**—high inter-channel correlation implies feature redundancy, while decorrelation implies full utilization of representational capacity.

2. **Shannon Entropy + Activation Magnitude Joint Analysis**

    - Function: Quantifies per-layer information complexity and signal propagation strength.
    - Mechanism: For Shannon entropy, the channel activation vector at each spatial position is discretized into 100 bins and the distributional entropy is computed as $E_{l,b,h,w} = -\sum_{n:p_n>0} p_n \log_2 p_n$, averaged over batch and spatial dimensions. For activation magnitude, the mean absolute value per layer is computed as $M_l = \frac{1}{BCHW}\sum|\mathbf{A}_{l,b,c,h,w}|$.
    - Design Motivation: Low entropy indicates concentrated, structured representations (analogous to the bottleneck in Information Bottleneck theory); high entropy indicates uniform expansion. This cross-validates spectral analysis: the U-shaped entropy profile corresponds to a three-phase spectral evolution from uniform → low-pass → uniform.

3. **Distillation Evolution Analysis**

    - Function: Tracks the effect of different distillation strategies on student training dynamics.
    - Mechanism: The student's layer-wise entropy profile is recorded every 30 epochs during training to observe how the U-shaped pattern forms or is disrupted. The developmental trajectories of the student are compared across four configurations: SoftKD, SpectralKD-First, SpectralKD-Last, and SpectralKD-Both.
    - Design Motivation: Knowledge distillation is not static knowledge copying but a guidance of the student's "developmental trajectory." Incorrect guidance (late-layer alignment) disrupts the student's natural formation of the U-shaped pattern.

### Loss & Training

SpectralKD: $\mathcal{L}_{\text{Freq}} = \text{MSE}(\mathcal{F}_{\text{stack}}(\mathbf{A}_s), \mathcal{F}_{\text{stack}}(\mathbf{A}_t))$, where 2D RFFT is applied along spatial dimensions and real/imaginary parts are concatenated.

ProjectorKD: $\mathcal{L}_{\text{Proj}} = \text{MSE}(\text{Projector}(\mathbf{A}_s), \mathbf{A}_t)$, with a learnable projection layer for dimension matching.

Total loss: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{KD}} + \beta \mathcal{L}_{\text{Feature}}$, where $\beta$ controls the feature distillation weight. Teacher: CaiT-S24; Student: DeiT-Tiny; 300 epochs (500 epochs for select experiments).

## Key Experimental Results

### Main Results

| Method | Aligned Layers | β | Top-1 Acc (%) |
|------|--------|---|--------------|
| SoftKD (logit only) | - | - | 76.99 |
| SpectralKD | First1+Last1 | 0.2 | **77.08** |
| SpectralKD | First1 | 0.2 | 77.00 |
| SpectralKD | Last1 | 0.2 | 76.83 (−0.16) |
| SpectralKD | Last1 | 0.1 | 76.48 (−0.51) |
| SpectralKD | Last8 | 0.2 | 76.69 (−0.30) |
| ProjectorKD | First1 | 0.2 | 76.86 |
| ProjectorKD | Last1 | 0.2 | 76.72 (−0.27) |
| ProjectorKD | First1+Last1 | 0.2 | 76.80 |
| SoftKD (500ep) | - | - | 78.07 |
| SpectralKD (500ep) | Last1 | 0.2 | 77.59 (−0.48) |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Reduce β (0.2→0.1) | 76.83→76.48 | Weaker distillation signal performs worse—the issue is direction, not magnitude |
| Extend training (300→500ep) | Gap widens from 0.16 to 0.48 | Longer training cannot compensate for negative effects of late-layer distillation |
| CNN (ResNet) late-layer spectrum | Maintains low-pass characteristics | CNNs do not fully exploit channel capacity → students can imitate → CNN distillation succeeds |
| U-shape consistency across ViT/CaiT/MAE | Three architectures/training paradigms | U-shaped pattern is a universal ViT characteristic, not a model-specific artifact |

### Key Findings

- **U-shaped information processing is a fundamental characteristic of ViTs**: CaiT-S24, standard ViT, and MAE-pretrained ViT all exhibit consistent U-shaped entropy and activation magnitude curves. Layers 1–9 compress; layers 9–24 expand—corresponding to the two phases of the Information Bottleneck.
- **Late-layer spectra shift from low-pass to uniform distribution**: Phase 1 (early layers) is uniformly noisy → Phase 2 (middle layers) exhibits low-pass filtering → Phase 3 (late layers) shows uniform high-energy content. Phase 3 represents distributed high-dimensional encoding, where information is dispersed and entangled across the entire channel space.
- **Critical difference between CNNs and ViTs**: CNN (ResNet) late layers retain the low-pass characteristics of Phase 2 and **do not exploit full channel capacity** → small students can imitate → distillation succeeds. ViT late layers fully exploit channel capacity → students cannot replicate → distillation fails.
- **Counter-intuitive finding that reducing β leads to worse performance**: A weaker distillation signal disrupts the student's learning equilibrium, causing it to oscillate between the teacher's encoding paradigm and its own optimal solution.
- **Distillation as "developmental trajectory guidance" rather than "static knowledge copying"**: Under SoftKD, the student naturally develops a U-shaped pattern; late-layer alignment suppresses the natural formation of the expansion phase.

## Highlights & Insights

- **The discovery of the U-shaped information processing pattern** holds significant theoretical value—it reflects ViT's learned behavior rather than an architectural property, appears consistently under both supervised and self-supervised training, and provides a new perspective for understanding ViT's internal mechanisms.
- **Channel-wise FFT analysis** is a genuinely original tool—distinct from the commonly used spatial FFT, it reveals the encoding structure of the feature space itself.
- **The spectral difference between CNN and ViT late layers** precisely explains the divergence in distillation performance—this constitutes the most central insight of the paper.
- **The perspective of "distillation as guidance of developmental trajectory"** is profound—late-layer distillation does not provide a wrong *quantity* of signal, but rather a wrong *direction*.
- The finding and explanation that reducing $\beta$ leads to worse performance is highly instructive: it reveals a subtle dynamic equilibrium between the distillation loss and the classification loss.

## Limitations & Future Work

- **The paper is primarily analytical**: SpectralKD and ProjectorKD serve only as validation tools for the analysis, and actual performance improvements are marginal (best result: 77.08 vs. baseline 76.99).
- **No effective ViT distillation method is proposed**: "Phase-specific distillation" remains a recommendation and is not implemented.
- **Experiments are limited to ImageNet classification**: Whether the same U-shaped pattern and distillation failure generalize to downstream tasks such as detection and segmentation is not verified.
- **The formation mechanism of the U-shaped pattern is not deeply analyzed**: Why does ViT learn this pattern while CNNs do not? Is it an inherent property of self-attention, or an effect of training data and objectives?
- **Only one teacher–student pair is analyzed**: CaiT-S24 → DeiT-Tiny; other combinations (e.g., Swin → Swin-Tiny) are not evaluated.

## Related Work & Insights

Compared to ViTKD, which proposes a ViT-specific distillation method without explaining the failure mechanism, this paper provides the first mechanistic explanation. Compared to FitNet, this paper reveals the deeper reason for its success on CNNs—CNN late layers retain compact encodings that students can imitate. Compared to Information Bottleneck theory, this paper provides the first direct empirical evidence of IB theory in ViTs.

Core insight: ViT distillation should align only early-to-middle layers (the compression phase) and avoid late-layer alignment. A further idea is to design an "encoding translator" that converts the teacher's distributed representations into compact encodings digestible by the student, rather than having the student directly imitate them. The U-shaped pattern also has implications for VLM token pruning: token compression is most efficient at the entropy minimum (the bottleneck point).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First paper to explain ViT distillation failure from an information-theoretic and frequency-domain perspective; the U-shaped pattern and channel-wise FFT are genuinely original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analysis is very comprehensive (spectral + entropy + magnitude + distillation evolution), but validation of distillation methods is limited and restricted to ImageNet classification.
- Writing Quality: ⭐⭐⭐⭐⭐ Argumentation is rigorous and logically structured, progressing from phenomenon → analysis → explanation → validation, with clear and visually compelling figures.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental theoretical guidance for ViT compression and is expected to have lasting influence on the design of future distillation methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](../../CVPR2026/video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](../../ICCV2025/video_understanding/egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)
- [\[AAAI 2026\] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs](predicting_video_slot_attention_queries_from_random_slot-feature_pairs.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

</div>

<!-- RELATED:END -->
