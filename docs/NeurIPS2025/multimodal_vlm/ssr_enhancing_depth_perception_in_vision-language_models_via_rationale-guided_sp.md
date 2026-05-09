---
title: >-
  [Paper Note] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial Reasoning] This paper proposes the SSR framework, which converts raw depth information into structured textual reasoning rationales and compresses them into compact latent embeddings via knowledge distillation, enhancing the spatial reasoning capabilities of existing VLMs in a plug-and-play manner.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Spatial Reasoning
  - Depth Perception
  - Vision-Language Models
  - Chain-of-Thought
  - Knowledge Distillation
date: 2026-05-08
content_hash: 8bceede0fb93cb1c
---

# SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.12448](https://arxiv.org/abs/2505.12448)
**Code**: [https://yliu-cs.github.io/SSR](https://yliu-cs.github.io/SSR)
**Area**: Multimodal VLM
**Keywords**: Spatial Reasoning, Depth Perception, Vision-Language Models, Chain-of-Thought, Knowledge Distillation

## TL;DR

This paper proposes the SSR framework, which converts raw depth information into structured textual reasoning rationales and compresses them into compact latent embeddings via knowledge distillation, enhancing the spatial reasoning capabilities of existing VLMs in a plug-and-play manner.

## Background & Motivation

VLMs have demonstrated strong performance across multimodal tasks; however, relying solely on RGB inputs makes it difficult to accurately capture spatial information such as relative positions and distances. Existing methods for incorporating spatial cues suffer from two categories of limitations: some depend on specialized sensors (e.g., LiDAR for point clouds) and are thus inapplicable in monocular RGB-only settings; others incorporate depth maps but exploit depth information only superficially—as a supplementary input—without leveraging its reasoning value.

The core insight is that humans, when reasoning about spatial relationships, do not merely "see" depth but treat it as part of the reasoning process—first analyzing spatial relationships between objects, then using that understanding to guide subsequent inference. Existing methods lack this implicit depth-reasoning capability, motivating a more sophisticated approach to depth integration.

## Method

### Overall Architecture

SSR introduces a core plug-and-play module, MIDI (Mamba-based Image-Depth Interpreter), which transforms RGB images and depth maps into latent token representations encoding spatial reasoning information. The overall pipeline is: input image → monocular depth estimation → MIDI module generates rationale latent tokens → fed together with the original image and text into the VLM for answer generation. Training proceeds in two stages: Stage 1 aligns the reasoning and semantic spaces; Stage 2 (optional) jointly trains MIDI and the VLM.

### Key Designs

1. **MIDI Module (Mamba-based Image-Depth Interpreter)**: RGB image features are encoded with CLIP ViT-L/14 and depth features with SigLIP, each projected into a shared semantic space via MLP. The image features, depth features, and text query are then fed into a Mamba-based language model to generate latent tokens representing intermediate reasoning rationales. A key innovation is the use of Mamba (rather than Transformer) as the reasoning backbone, offering higher computational efficiency. Special tokens are uniformly inserted into the rationale sequence to facilitate knowledge distillation compression.

2. **Rationale-to-Latent Knowledge Distillation**: Unlike conventional CoT approaches that rely on verbose textual reasoning, SSR compresses reasoning rationales into compact latent embeddings. In Stage 1, the LLM reconstructs textual rationales from MIDI-generated latent tokens, compelling the latent representations to encode complete reasoning information. After training, MIDI can be directly inserted into the VLM's input sequence without modifying VLM parameters.

3. **SSR-CoT Dataset and SSRBench**: Four data sources—LLaVA-CoT, Visual-CoT, VoCoT, and SpatialQA—are integrated; depth maps are extracted using Depth Pro, spatial attributes are mined via SpatialRGPT, and detailed reasoning chains are generated with GPT-4o, yielding approximately 1.2 million image–depth–question–rationale–answer pairs. SSRBench comprises 6 tasks (3 general + 3 spatial), sampled from SSR-CoT with rigorous deduplication.

### Loss & Training

- **Stage 1 (Reasoning and Alignment)**: Standard causal language modeling loss, training MIDI-generated latent tokens such that the downstream LLM can reconstruct the original rationale text. Only MIDI is trained; the VLM is frozen.
- **Stage 2 (Co-Training, optional)**: Intermediate rationales are discarded; the VLM is trained with standard causal loss to directly generate answers. MIDI and the VLM are jointly trained, with additional LLaVA-Instruct-150K data incorporated.
- Training uses LoRA + FSDP on a single node with 8×H800 GPUs; Stage 1 takes approximately 19 hours and Stage 2 approximately 48 hours.

## Key Experimental Results

### Main Results

| Benchmark | Metric | SSR (3B) | Qwen2.5-VL-3B | Gain |
|-----------|--------|----------|---------------|------|
| SpatialBench | Avg. | 64.8% | 59.3% | +5.4 |
| SSRBench General | Avg. | 79.3% | 62.8% | +16.5 |
| SSRBench Spatial | Avg. | 69.7% | 48.8% | +20.9 |
| CV-Bench | Avg. | 68.9% | 67.0% | +1.9 |
| VSR | Zero-shot | 82.9% | 76.4% | +6.5 |

SSR (7B) yields further improvements, surpassing baselines such as LLaVA-NeXT-13B and SpatialBot-3B on SpatialBench.

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| With rationale vs. without rationale | SSR-CoT data quality validation | Accuracy +11.62% (67.80→79.42) |
| Stage 1 only (plug-and-play) | No joint VLM training | Already yields significant gains |
| Stage 1 + Stage 2 | Joint training | Further improvements across benchmarks |
| MIDI module size 130M | Lightweight overhead | Substantial gains when paired with 3B VLM |

### Key Findings

- The MIDI module, with only 130M parameters, delivers an average spatial reasoning improvement of 20.9% for a 3B VLM, representing an exceptionally favorable cost–performance trade-off.
- The plug-and-play nature of Stage 1 training eliminates the need to retrain the VLM, enhancing practical deployability.
- The rationale quality assurance mechanism in SSR-CoT (cache pool + sampling validation + iterative re-annotation) proves effective.
- VLMs of varying scales (3B→7B) both benefit from SSR, with improvements observed not only on spatial tasks but also on general tasks.

## Highlights & Insights

- **The correct way to utilize depth information**: Rather than naively concatenating depth features, the framework converts depth into reasoning rationales and then compresses them into latent representations. This "reason first, then compress" paradigm allows depth information to genuinely participate in higher-order cognitive processes.
- **Plug-and-play design**: Once trained, the MIDI module can be directly inserted into the input sequence of any VLM without modifying the original model, enabling deployment-friendly integration.
- **Mamba as the reasoning engine**: Mamba is chosen over Transformer for the reasoning module, leveraging its selective state-space properties and linear complexity advantages.
- The data construction pipeline (Depth Pro + SpatialRGPT + GPT-4o) is systematic and reusable.
- The computational cost of Stage 1 (~19h) and Stage 2 (~48h) is acceptable, and Stage 2 remains optional.

## Limitations & Future Work

- The framework depends on the quality of the monocular depth estimation model (Depth Pro); failures in depth estimation will adversely affect reasoning.
- SSR-CoT dataset construction relies on GPT-4o, which incurs considerable cost and may introduce biases.
- Whether the lightweight Mamba reasoning module has sufficient expressive capacity for highly complex spatial scenes remains to be verified.
- SSRBench lacks spatial reasoning evaluations at the 3D scene or video level.
- The depth encoder (SigLIP) and image encoder (CLIP ViT-L/14) are heterogeneous models; the alignment quality between their feature spaces may affect downstream reasoning.
- Stage 2 joint training is validated only on Qwen2.5-VL; adaptability to other VLM backbones has yet to be confirmed.
- Rationale quality verification covers only 10% of samples, leaving the possibility of undetected low-quality instances.

## Related Work & Insights

- SSR is distinguished from works such as SpatialVLM and SpatialRGPT by introducing language-based reasoning capabilities.
- The approach of distilling rationales into latent representations is inspired by Coconut (Chain of Continuous Thought), but its application in the multimodal setting is novel.
- SSR has direct applicability to robot spatial reasoning in embodied intelligence.
- The multi-source data fusion pipeline of SSR-CoT (LLaVA-CoT + Visual-CoT + VoCoT + SpatialQA) can serve as a reference template for large-scale CoT data construction.
- The design philosophy of the MIDI module is generalizable to other modality-enhancement scenarios (e.g., tactile, thermal, optical flow).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paradigm of converting depth into reasoning rationales and distilling them into latent representations is novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple benchmarks, with a self-constructed SSRBench and comprehensive data quality evaluation
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and the pipeline is described in thorough detail
- **Value**: ⭐⭐⭐⭐ The plug-and-play spatial enhancement approach is highly practical and meaningful for embodied AI

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)

</div>

<!-- RELATED:END -->
