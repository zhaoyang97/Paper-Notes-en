---
title: >-
  [Paper Note] ConsistCompose: Unified Multimodal Layout Control for Image Composition
description: >-
  [CVPR2026][Image Generation][Layout-controlled generation] Propounds the LELG (Language-Embedded Layout Guidance) paradigm, which encodes bounding box coordinates directly into text tokens within the language stream. This achieves layout-controllable multi-instance image generation in a unified multimodal Transformer without requiring any specialized layout encoders or branches.
tags:
  - CVPR2026
  - Image Generation
  - Layout-controlled generation
  - multi-instance composition
  - unified multimodal model
  - language-embedded layout
  - Classifier-Free Guidance
date: 2026-05-08
content_hash: 050b13791f944a28
---

<!-- Generated automatically from src/gen_stubs.py -->
# ConsistCompose: Unified Multimodal Layout Control for Image Composition

**Conference**: CVPR2026  
**arXiv**: [2511.18333](https://arxiv.org/abs/2511.18333)  
**Code**: TBD  
**Area**: Image Generation  
**Keywords**: Layout-controlled generation, multi-instance composition, unified multimodal model, language-embedded layout, Classifier-Free Guidance

## TL;DR

Propounds the LELG (Language-Embedded Layout Guidance) paradigm, which encodes bounding box coordinates directly into text tokens within the language stream. This achieves layout-controllable multi-instance image generation in a unified multimodal Transformer without requiring any specialized layout encoders or branches.

## Background & Motivation

1. **Existing unified multimodal models favor understanding**: Models like DALL-E and Stable Diffusion offer excellent generation quality, but unified models primarily focus on visual understanding (grounding), providing insufficient support for layout-controlled generation and limiting compositional scene synthesis.
2. **Diffusion architectures are heavily coupled**: Methods like GLIGEN and InstanceDiffusion rely on layout-image fusion modules or region-aware U-Net modifications, which are difficult to reconcile with Transformer-based generation frameworks.
3. **Autoregressive methods remain confined to layout tasks**: LayoutSAM, HiCo, and PlanGen only handle layout-related tasks and have not demonstrated compatibility with general multimodal capabilities like visual reasoning or image editing.
4. **Multi-reference identity-preserving generation is overlooked**: Most work focuses on text-conditioned layout control, rarely considering the more challenging setting of multi-reference, identity-preserving multi-instance composition.
5. **Lack of large-scale datasets**: There is a lack of large-scale datasets providing instance-level layouts, subject correspondence, and diverse multi-instance configurations, hindering the development of unified layout-aware systems.
6. **Core Insight — Encoding layout as language**: When layout semantics are represented as part of the text, the multimodal Transformer can learn spatial alignment through the same interleaved modeling used for perception and reasoning, without any architectural modifications.

## Method

### Overall Architecture

ConsistCompose is built on Bagel's MoT (Mixture-of-Transformers) architecture, containing two Transformer experts: one for multimodal understanding and one for multimodal generation. It uses two visual encoders: a ViT initialized with SigLIP for semantic perception and a VAE initialized with FLUX for image generation. All modalities (text, ViT features, VAE latents) are projected into a shared embedding space.

### LELG Paradigm and ICBP Mechanism

The core innovation is **Language-Embedded Layout Guidance (LELG)**: directly inserting normalized bounding boxes as text tokens. Coordinates $b_i=(x_1^i, y_1^i, x_2^i, y_2^i) \in [0,1]^4$ for each instance $i$ follow the corresponding subject phrase with three-decimal precision:

```
"a brown sofa <bbox>[0.123, 0.456, 0.789, 0.987]</bbox>"
```

This **Instance Coordinate Binding Prompt (ICBP)** mechanism binds the linguistic reference of an object to its spatial coordinates in a unified sequence. Without additional geometric losses, the model implicitly learns spatial alignment through self-attention over interleaved text and visual tokens.

### Coordinate-CFG for Enhanced Spatial Control

Introduces coordinate-aware Classifier-Free Guidance: the conditional branch uses prompts with coordinate tokens, while the unconditional branch omits them. The formula is:

$$\mathbf{v}_t^{\text{coord-cfg}} = \mathbf{v}_t^{\text{coord-uncond}} + s_{\text{coord}} (\mathbf{v}_t^{\text{coord}} - \mathbf{v}_t^{\text{coord-uncond}})$$

Additional velocity normalization is introduced to ensure stable guidance magnitude. Larger $s_{\text{coord}}$ results in stricter layout adherence, though excessive values degrade quality—optimum is 1.6 on COCO-Position and 0.4–0.8 on MS-Bench.

### Loss & Training

- **Flow Matching Loss**: $\mathcal{L}_{\text{FM}}$ is used for image generation, predicting the velocity field between latents.
- **Autoregressive LM Loss**: $\mathcal{L}_{\text{LM}}$ is used for next-token prediction in multimodal understanding.
- **Total Loss**: $\mathcal{L} = \lambda_{\text{FM}} \mathcal{L}_{\text{FM}} + \lambda_{\text{LM}} \mathcal{L}_{\text{LM}}$, introducing no extra coordinate regression loss.
- **Two-stage Training**: ① Alignment stage mixing general multimodal understanding data + ConsistCompose3M; ② Mixed SFT stage jointly training understanding, generation, editing, and multi-subject reference generation data.

### ConsistCompose3M Dataset

A large-scale dataset of 3.4 million samples is constructed, consisting of two parts:

- **T2I Component** (2.6M): Reprocessed LayoutSAM, injecting instance-level layout annotations into prompts via the ICBP mechanism.
- **Reference-conditioned Component** (0.8M): Reusing subject assets from Subjects200K and UNO, reorganized into multi-subject scenes under diverse layouts. Filtered by CLIP/DINO similarity to ensure identity consistency.

## Key Experimental Results

### COCO-Position Layout Control

| Method | Avg Instance SR(%) | Avg Image SR(%) | mIoU | AP | AP50 | AP75 |
|------|-------------------|-----------------|------|------|------|------|
| GLIGEN | 82.6 | 52.1 | 69.0 | 40.5 | 75.9 | 39.1 |
| InstanceDiffusion | 87.8 | 65.5 | 78.1 | 57.2 | 83.6 | 65.5 |
| MIGC++ | 86.8 | 63.4 | 74.9 | 48.3 | 79.2 | 52.6 |
| PlanGen | 82.5 | 50.3 | 66.2 | 31.9 | 74.0 | 21.5 |
| **Ours** | **92.6** | **76.1** | **85.3** | **70.9** | **89.1** | **76.9** |

ConsistCompose achieves the best performance across all metrics. mIoU increases by 7.2% (78.1→85.3), and AP increases by 13.7% (57.2→70.9). Advantages are particularly significant in high-instance count scenarios (L4-L6).

### MS-Bench Multi-reference Identity Preservation

| Method | CLIP-T | DINO | mIoU | AP |
|------|--------|------|------|-----|
| GLIGEN | 0.309 | 0.454 | 0.868 | 0.751 |
| MS-Diffusion | 0.336 | 0.555 | 0.466 | 0.108 |
| MUSE | 0.320 | 0.619 | 0.698 | 0.352 |
| **Ours** | 0.333 | **0.660** | **0.889** | **0.789** |

Achieves superior results in both identity preservation (DINO) and spatial accuracy (mIoU, AP), breaking the trade-off between these two dimensions found in previous methods.

### Ablation: General Capability Retention

Layout training does not significantly harm general capabilities: MMBench 81.4 (on par with Bagel Base), GenEval 0.88 (slightly better than Base's 0.86). DreamBench single/multi-target DINO reach 0.677/0.506 respectively, both being SOTA.

## Highlights & Insights

- **Minimalist Unified Design**: Layout information is fully encoded as text tokens without layout encoders, regional attention, or task branches, elegantly unifying spatial control with multimodal understanding.
- **Tunable Coordinate-CFG**: Flexible control over the balance between layout precision and visual quality via inference-time CFG scale; optimal values can be set independently for different tasks.
- **Large-scale Dataset Contribution**: ConsistCompose3M fills the gap in datasets jointly annotated with layout and identity.
- **Zero Loss in General Capabilities**: General metrics like MMBench/GenEval remain stable or slightly improve after layout training, proving that unified training does not sacrifice existing capabilities.

## Limitations & Future Work

- Currently only supports bounding box-level spatial control; does not support finer spatial descriptions like semantic segmentation or keypoints.
- Coordinate-CFG requires additional inference steps (dual-path inference with/without coordinates), increasing sampling overhead.
- MMMU score (42.3) decreased compared to Bagel Base (46.4), suggesting that intense layout training might slightly affect cross-domain reasoning.
- It remains unclear if three-decimal precision coordinate discretization is the optimal granularity.
- The Bagel-based MoT architecture is large, resulting in high deployment costs and training resource requirements.
- Richer application scenarios like interactive layout editing or progressive scene construction have not been explored.

## Related Work & Insights

- **vs GLIGEN/InstanceDiffusion**: These use U-Net based layout fusion modules with high architectural coupling; ConsistCompose implements spatial control via a pure text interface, making it more lightweight and general.
- **vs PlanGen/LayoutSAM**: These use structured spatial tokens or independent modalities to handle layouts, limited only to layout tasks; ConsistCompose unifies layout, understanding, and generation.
- **vs MS-Diffusion/MUSE**: Multi-reference generation methods often compromise on either spatial accuracy or identity preservation; ConsistCompose balances both.
- **vs OmniGen2/Bagel**: Unified multimodal models that lack explicit layout control; ConsistCompose completes spatial control capabilities on top of Bagel via LELG.

## Rating

- Novelty: ⭐⭐⭐⭐ — The LELG paradigm encoding layout as language tokens is simple yet effective; the Coordinate-CFG mechanism is well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple benchmarks including COCO-Position, MS-Bench, GenEval, and DreamBench, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, consistent notation, high-quality figures and tables.
- Value: ⭐⭐⭐⭐ — Provides a feasible path for adding layout control to unified multimodal models; the dataset contribution has lasting value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)
- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](../../AAAI2026/image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

</div>

<!-- RELATED:END -->
