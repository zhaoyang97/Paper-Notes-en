---
title: >-
  [Paper Note] ConsistCompose: Unified Multimodal Layout Control for Image Composition
description: >-
  [CVPR 2026][Image Generation][Layout-controlled generation] The paper proposes ConsistCompose, which achieves layout-controllable multi-instance image generation within a unified multimodal framework by embedding layout coordinates directly into language prompts (the LELG paradigm). It constructs the ConsistCompose3M dataset with 3.4 million samples providing layout and identity supervision. Coupled with a Coordinate-aware CFG mechanism, it achieves a 7.2% improvement in layout IoU and a 13.7% improvement in AP on COCO-Position while maintaining general understanding capabilities.
tags:
  - CVPR 2026
  - Image Generation
  - Layout-controlled generation
  - Multi-instance image synthesis
  - LELG
  - Coordinate-embedded prompt
  - Identity preservation
date: 2026-05-08
content_hash: 33cfece3bd9ac416
---

# ConsistCompose: Unified Multimodal Layout Control for Image Composition

**Conference**: CVPR 2026  
**arXiv**: [2511.18333](https://arxiv.org/abs/2511.18333)  
**Code**: None  
**Area**: Image Generation / Layout Control  
**Keywords**: Layout-controlled generation, Multi-instance image synthesis, LELG, Coordinate-embedded prompt, Identity preservation

## TL;DR
The paper proposes ConsistCompose, which achieves layout-controllable multi-instance image generation within a unified multimodal framework by embedding layout coordinates directly into language prompts (the LELG paradigm). It constructs the ConsistCompose3M dataset with 3.4 million samples providing layout and identity supervision. Coupled with a Coordinate-aware CFG mechanism, it achieves a 7.2% improvement in layout IoU and a 13.7% improvement in AP on COCO-Position while maintaining general understanding capabilities.

## Background & Motivation

**State of the Field**: Unified multimodal models (e.g., Bagel, OmniGen2) are already capable of both understanding and generation within a single architecture, but primarily focus on visual understanding (grounding). Precise layout control on the generation side remains weak.

**Limitations of Prior Work**: Existing methods for layout-controlled generation face fundamental obstacles: (a) Diffusion model approaches (GLIGEN, InstanceDiffusion) rely on specialized layout-image fusion modules or region-aware U-Nets, which are incompatible with Transformer generation frameworks; (b) Autoregressive models (LayoutSAM, PlanGen) treat layout as an independent modality, limiting them to layout tasks and preventing them from encompassing general capabilities like visual reasoning and understanding; (c) Most methods only support text-conditioned layout control and do not consider the more difficult scenario of multi-reference image identity preservation.

**Root Cause**: Layout control requires task-specific branches/encoders, which contradicts the philosophy of a "unified" framework. How can precise layout control be achieved without introducing additional architectural modules?

**Paper Goals**: To support layout-grounded text-to-image generation, identity-consistent multi-instance synthesis from multiple references, and general multimodal understanding simultaneously within a unified multimodal framework—using a single model for all three.

**Starting Point**: Layout is essentially information that can be expressed in language. Rather than designing specialized spatial encoders, coordinates can be encoded as text tokens, allowing the Transformer to naturally learn spatial grounding through language understanding.

**Core Idea**: Language as Layout Control—embed coordinates into the prompt to let the unified model learn spatial layouts via the text stream without any architectural changes.

## Method

### Overall Architecture
Based on Bagel's MoT (Mixture of Transformers) architecture, which includes two Transformer experts for understanding and generation. The input is a text prompt with coordinate annotations + optional reference images, and the output is a multi-instance image satisfying layout constraints. Three major components: (1) The LELG paradigm encodes layout semantics into text tokens; (2) Coordinate-CFG enhances spatial control during sampling; (3) ConsistCompose3M provides training data.

### Key Designs

1. **LELG Paradigm + Instance-Coordinate Binding Prompt (ICBP)**:

    - **Function**: Directly inserts the bounding box of each instance after its corresponding subject phrase to form a unified text sequence.
    - **Mechanism**: For the $i$-th instance, the normalized bbox $b_i = (x_1^i, y_1^i, x_2^i, y_2^i) \in [0,1]^4$ is represented with three decimal places and inserted into the text: "a brown sofa <bbox>[0.123, 0.456, 0.789, 0.901]</bbox>". Thus, coordinates become part of the language tokens, and the Transformer naturally learns instance-position binding via shared self-attention.
    - **Design Motivation**: (1) **Zero architectural changes**—no layout encoders, ControlNet, or extra attention modules required; (2) **Natural unification**—understanding and generation share the same token space, allowing spatial reasoning capabilities to transfer from understanding tasks to generation; (3) Discretization with three decimal places maps continuous space to approximately $1000^4$ discrete positions, which is sufficiently precise and compatible with the tokenizer.
    - **Novelty**: GLIGEN requires gated Transformer layers, InstanceDiffusion requires multimodal fusion modules, and CreatiLayout requires SiamLayout—all are architectural modifications. LELG solves the problem purely at the input level.

2. **Coordinate-aware Classifier-Free Guidance (Coordinate-CFG)**:

    - **Function**: Enhances spatial control during inference sampling by comparing the difference in prediction velocities with and without coordinate conditions.
    - **Mechanism**: $$\mathbf{v}_t^{\text{coord-cfg}} = \mathbf{v}_t^{\text{uncond}} + s_{\text{coord}}(\mathbf{v}_t^{\text{coord}} - \mathbf{v}_t^{\text{uncond}})$$, where $s_{\text{coord}}$ controls the spatial guidance strength. Velocity normalization $\alpha = \|\mathbf{v}_t^{\text{base}}\| / \|\mathbf{v}_t^{\text{coord-cfg}}\|$ is also introduced to prevent guidance magnitude explosion.
    - **Design Motivation**: ICBP provides spatial signals, but the model might not be sufficiently "obedient." Coordinate-CFG acts as a spatial version of text CFG, explicitly amplifying the difference between coordinate-conditioned and unconditional outputs, forcing the generation to follow the layout more precisely. Experiments show that increasing $s_{\text{coord}}$ progressively improves positional accuracy, with excessive values slightly affecting perceptual quality.

3. **ConsistCompose3M Dataset**:

    - **Function**: Provides 3.4 million samples of layout + identity supervised training data.
    - **Mechanism**: Two subsets: (a) **T2I subset** (2.6M): Reprocesses LayoutSAM data, appending bbox coordinates to captions for each instance using the ICBP mechanism; (b) **Reference-conditioned subset** (0.8M): Reuses subject assets from Subjects200K and UNO, recombining them into multi-subject scenes under various layouts, with identity consistency ensured by CLIP/DINO similarity filtering.
    - **Design Motivation**: Previously, there was no large-scale multi-instance generation dataset featuring both layout and identity annotations. The lack of data has been a significant reason for slow progress in layout-controlled generation.

### Loss & Training
- **Two-stage Training**: An alignment stage (mixing general understanding data + ConsistCompose3M to inject layout awareness), followed by a hybrid SFT stage (jointly training understanding/generation/editing/multi-subject reference generation + ConsistCompose3M).
- **Training Objective**: A weighted combination of Flow Matching loss $\mathcal{L}_{\text{FM}}$ + Language Model loss $\mathcal{L}_{\text{LM}}$, with no additional coordinate regression loss—spatial grounding is learned implicitly entirely from the language stream.
- **High-resolution Fine-tuning**: Further balances layout control and general image generation performance.

## Key Experimental Results

### Main Results (COCO-Position)

| Method | Instance Success Avg↑ | Image Success Avg↑ | mIoU↑ | AP↑ | AP50↑ | AP75↑ |
|------|---------------------|-------------------|-------|-----|-------|-------|
| GLIGEN | 82.6 | 52.1 | 69.0 | 40.5 | 75.9 | 39.1 |
| InstanceDiffusion | 87.8 | 65.5 | 78.1 | 57.2 | 83.6 | 65.5 |
| MIGC++ | 86.8 | 63.4 | 74.9 | 48.3 | 79.2 | 52.6 |
| CreatiLayout | 74.0 | 42.5 | 64.9 | 32.4 | 61.1 | 31.6 |
| PlanGen | 82.5 | 50.3 | 66.2 | 31.9 | 74.0 | 21.5 |
| **Ours** | **92.6** | **76.1** | **85.3** | **70.9** | **89.1** | **76.9** |

- Compared to the strongest baseline InstanceDiffusion: Layout mIoU +7.2%, AP +13.7%, Image Success Avg +10.6%.

### Ablation Study (Training Stages)

| Stage | Instance Success Avg | mIoU | AP |
|------|---------------------|------|-----|
| Alignment only | 88.4 | 79.1 | 58.3 |
| + Hybrid SFT | **92.6** | **85.3** | **70.9** |

### Key Findings
- **Effectiveness of LELG**: By embedding coordinates purely through language (no extra architecture), layout accuracy significantly outperforms all specially designed baselines.
- **Maintenance of General Capabilities**: Performance on MMMU and MMBench is on par with the Bagel backbone, indicating that layout control training does not harm general understanding.
- **Role of Coordinate-CFG**: $s_{\text{coord}}$ from 1 to 3 progressively improves positional precision; an optimal point exists (excessive values slightly degrade quality).
- **Necessity of Two-stage Training**: The Hybrid SFT stage further improves AP by 12.6% over the Alignment base.

## Highlights & Insights
- The **simplicity of the LELG paradigm** is impressive: it simplifies the "layout control" problem, which seemingly requires specialized modules, into "inserting coordinates into the prompt"—achieving SOTA layout precision with zero architectural changes. This design philosophy suggests a broader insight: many conditional controls that seem to require specialized modules (depth, edges, keypoints) could potentially be unified as part of a language interface.
- **Coordinate-CFG** cleverly extends CFG from "semantic guidance" to "spatial guidance" and works independently of text CFG, allowing them to be stacked. This design can be transferred to any generation model that supports CFG.
- The **dataset construction strategy** is worth emulating: building new-purpose datasets by reprocessing existing data (LayoutSAM → T2I, Subjects200K → reference-conditioned) efficiently utilizes existing resources.

## Limitations & Future Work
- Discretization of coordinates to three decimal places may lack precision in high-resolution scenarios (an error of about 0.1% of image width).
- Currently only supports bounding box-level layout control; it does not support finer-grained masks, keypoints, or depth conditions.
- Dependency on Bagel as a backbone limits it to Bagel's base generation quality and training scale.
- Requires the specialized construction of the ConsistCompose3M dataset, involving substantial data preparation costs.
- Performance may degrade in multi-instance scenarios with a high number of instances (e.g., >6); COCO-Position tests up to 6 instances.

## Related Work & Insights
- **vs GLIGEN [Li et al., 2023]**: GLIGEN introduces bbox constraints using gated Transformer layers, which is an architectural change. ConsistCompose's LELG paradigm is more lightweight and more effective (AP +30.4%).
- **vs InstanceDiffusion [Wang et al., 2024]**: InstanceDiffusion achieves instance-level control through multimodal input fusion but remains within the U-Net paradigm. ConsistCompose surpasses it within the Transformer generation paradigm.
- **vs PlanGen [Gong et al., 2024]**: PlanGen follows a two-step process of planning the layout then generating the image. ConsistCompose’s end-to-end approach is more unified and yields better results.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The LELG paradigm is a innovation in layout-controlled generation, unifying spatial control via a language interface.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across COCO-Position, MS-Bench, GenEval, MMMU, and MMBench.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and sufficient technical detail.
- Value: ⭐⭐⭐⭐⭐ Provides a simple and effective solution for layout control in unified multimodal models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideoomni_omnimotion_controlled_multisubject.md)

<!-- RELATED:END -->
