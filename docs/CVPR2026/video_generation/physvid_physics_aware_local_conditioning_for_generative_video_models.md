---
title: >-
  [Paper Note] PhysVid: Physics Aware Local Conditioning for Generative Video Models
description: >-
  [CVPR 2026][Video Generation][physics consistency] PhysVid is a physics-aware local conditioning scheme that segments videos into temporal chunks, annotates each chunk with physics phenomenon descriptions via a VLM…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "physics consistency"
  - "local conditioning"
  - "cross-attention"
  - "counterfactual guidance"
date: 2026-05-08
content_hash: 57e9b94522ec2b9d
---

# PhysVid: Physics Aware Local Conditioning for Generative Video Models

**Conference**: CVPR 2026  
**arXiv**: [2603.26285](https://arxiv.org/abs/2603.26285)  
**Code**: [Project Page](https://5aurabhpathak.github.io/PhysVid)  
**Area**: LLM / NLP (Other)  
**Keywords**: video generation, physics consistency, local conditioning, cross-attention, counterfactual guidance

## TL;DR

PhysVid is a physics-aware local conditioning scheme that segments videos into temporal chunks, annotates each chunk with physics phenomenon descriptions via a VLM, and injects them through chunk-level cross-attention. At inference, "negative physics prompts" (counterfactual guidance) steer generation away from physics violations, improving physics commonsense scores by approximately 33% on VideoPhy.

## Background & Motivation

Generative video models (e.g., Sora, Wan2.1) have made remarkable progress in visual fidelity, yet they still exhibit fundamental shortcomings in physics consistency — generated videos frequently violate basic physical laws (e.g., object penetration, gravitational anomalies, implausible deformations). Existing improvement approaches have the following limitations:

**Global text prompts are too coarse**: Standard T2V models condition all frames with the same text, failing to capture local physics detail changes within specific temporal segments.

**Frame-level conditioning is too myopic**: Per-frame control methods are domain-specific and lack cross-frame physical continuity.

**Global enhanced prompts are still insufficient**: Methods like DiffPhy and PhyT2V use LLMs to enhance global prompts with physics information, but cannot guarantee the model attends to the correct physics cues at the correct time segments.

**Fundamental flaw of global cross-attention**: Research shows that global cross-attention produces nearly static attention maps, causing temporal alignment failure for action-related tokens.

PhysVid's core insight: **physical phenomena are temporally local** — motion, collisions, and lighting changes occur within short time intervals and require locally aligned conditioning.

## Method

### Overall Architecture

PhysVid inserts chunk-level cross-attention layers into a pre-trained T2V model (Wan2.1-1.3B), achieving dual-pathway global + local conditioning. The pipeline is:

**Training data preparation** → Segment video into chunks → VLM annotates chunk-level physics descriptions  
**Model training** → Freeze base model → Train chunk cross-attention → Unfreeze for joint training  
**Inference** → LLM generates local + counterfactual prompts from global prompt → Dual-pathway CFG-guided generation

### Key Designs

1. **Chunk-Level Physics Annotation (Data Preparation)**

    - Each 5-second training video (81 frames @16fps) is divided into 7 temporal chunks (~0.7 seconds/chunk)
    - **VideoLLama3-7B** analyzes the visible physical phenomena in each chunk
    - The VLM is guided to focus on three categories of physics information: **dynamics** (motion, collisions, acceleration), **shape** (deformation, bending), and **optics** (reflection, shadow, refraction)
    - The global prompt is also provided to the VLM to ensure local annotations align with the global description
    - Constrained generation techniques strictly enforce structured output format
    - Design Motivation: The WISA dataset's built-in global physics annotations are not used, as they may not align with the segmented 5-second clips

2. **Chunk-Aware Cross-Attention (Core Architectural Innovation)**

    - A new chunk-level cross-attention layer is inserted into each Transformer block of pre-trained Wan2.1
    - Video query tokens are modulated via 3D spatiotemporal RoPE (frame, height, width)
    - **Key innovation**: RoPE is also applied to text keys, with a defined text grid that includes a **chunk axis**
    - Video and text use the same RoPE frequency base, giving attention logits **cross-modal positional awareness** — video tokens can distinguish text information from different chunks
    - Unlike standard T2V cross-attention (where text keys have only 1D positional encoding and lack frame alignment coupling)
    - Design Motivation: Each video temporal segment should attend only to physics descriptions temporally aligned with it

3. **Training Strategy (Two-Stage)**

    - **Stage 1** (1000 steps): Freeze base model, train only newly added chunk cross-attention modules — stabilizes new modules
    - **Stage 2** (2000 steps): Unfreeze base model, jointly train all parameters
    - 4 GPUs, effective batch size 64
    - Trained using flow matching loss

4. **Counterfactual Physics Guidance (Inference Innovation)**

    - **Local prompt generation**: At inference, only a global T2V prompt is available (no video to reference); an LLM "imagines" each chunk's physics descriptions from the global prompt
    - **Counterfactual prompt generation**: The LLM identifies key visual and physical elements in each local physics prompt and generates **counterfactual descriptions that violate those physical phenomena**
    - **Dual-pathway CFG guidance**:
     $$x_{T-1} = (1+w) \cdot \mathcal{G}(x_T, c_g, C, T) - w \cdot \mathcal{G}(x_T, c_n, C', T)$$
     where $C'$ is the counterfactual prompt set and $c_n$ is the global negative prompt
    - Design Motivation: Positive guidance reinforces correct physics + negative guidance pushes away from physics violations = dual safeguard

### Loss & Training

- Trained using the same **flow matching** objective as Wan2.1
- Training data: ~53K video samples processed from the WISA-80K dataset (832×480, 81 frames @16fps)
- Built-in physics annotations from WISA are not used; chunk-level annotations are entirely re-extracted from videos by the VLM

## Key Experimental Results

### Main Results

**VideoPhy Benchmark**

| Method | Params | SA (Semantic Alignment) | PC (Physics Commonsense) ↑ |
|--------|--------|------------------------|-----------------------------|
| Wan-1.3B | 1.3B | 0.46 | 0.24 |
| Wan-14B | **14B** | **0.52** | 0.24 |
| **PhysVid** | **1.7B** | 0.43 | **0.32** |

PhysVid surpasses the 14B model in physics commonsense with only 1.7B parameters, a **relative improvement of ~33%**.

**VideoPhy2 Benchmark**

| Method | Params | SA | PC ↑ |
|--------|--------|----|------|
| Wan-1.3B | 1.3B | 0.28 | 0.61 |
| Wan-14B | 14B | **0.29** | 0.59 |
| **PhysVid** | 1.7B | 0.28 | **0.64** |

~8% relative improvement over Wan-14B on VideoPhy2.

**Comparison with Existing Physics-Aware Methods (VideoPhy)**

| Method | Base Model | PC ↑ | Relative Gain |
|--------|-----------|------|---------------|
| WISA | CogVideoX-5B | 0.38 | +15% |
| VideoREPA-5B | CogVideoX-5B | 0.40 | +29% |
| Hao et al. | Wan-14B | 0.40 | +14% |
| PhyT2V | CogVideoX-5B | **0.42** | +62% |
| **PhysVid-1.7B** | Wan-1.3B | 0.32 | +33% |

### Ablation Study

| Method | VideoPhy PC ↑ | VideoPhy2 PC ↑ | Note |
|--------|--------------|----------------|------|
| Wan-1.3B baseline | 0.2401 | 0.6144 | No improvements |
| Direct fine-tuning | 0.2866 | 0.6261 | WISA data fine-tuning without chunk architecture |
| PhysVid (no counterfactual guidance) | 0.2924 | 0.6334 | Positive local conditioning only |
| **PhysVid (full)** | **0.3169** | **0.6411** | Positive + counterfactual guidance |

### Key Findings

1. **Local conditioning > Global conditioning**: PhysVid significantly outperforms direct fine-tuning (same data but no chunk architecture), proving that temporal alignment of physics information is critical
2. **Counterfactual guidance is effective**: Adding counterfactual negative prompts improves PC from 0.2924 to 0.3169
3. **Model scale ≠ physics capability**: Despite having 8× more parameters than PhysVid, Wan-14B has lower physics commonsense scores (0.24 vs 0.32)
4. **Cost of semantic alignment**: PhysVid's SA score is slightly lower than baseline (0.43 vs 0.46), suggesting physics-oriented conditioning may slightly sacrifice visual aesthetics
5. **Consistent improvement across categories**: Improvements are observed across all subcategories including solid-solid, solid-fluid, fluid-fluid, object interaction, and sports

## Highlights & Insights

1. **Local temporal granularity for physics conditioning is the right direction**: Global physics prompts cannot align to specific time segments; chunk-level design elegantly solves this problem
2. **VLM as automatic physics annotator**: No dependency on manual physics annotations; physics information is entirely extracted from videos by the VLM, making the method applicable to arbitrary datasets
3. **Elegant counterfactual guidance design**: Borrows the positive/negative guidance concept from CFG but extends it to the physics dimension — generating "what if physics were violated" descriptions as negative conditions
4. **Cross-modal RoPE alignment**: Applying video-aligned RoPE to text keys gives chunk boundaries explicit positional signals in attention computation
5. **Beating scale with design**: A 1.7B parameter model surpasses a 14B model in the physics dimension, demonstrating that architectural design matters more than simply scaling parameters

## Limitations & Future Work

1. **Semantic alignment degradation**: SA score drops from 0.46 to 0.43, suggesting local physics conditioning may interfere with global semantic generation
2. **Dependence on VLM annotation quality**: The accuracy of chunk-level physics descriptions depends on VideoLLama3-7B's capabilities; a stronger VLM may yield further improvements
3. **LLM overhead at inference**: Inference requires an LLM to generate local and counterfactual prompts, adding extra latency and complexity
4. **Narrow training data (WISA)**: Focused on physics phenomena data; generalization to general T2V scenarios is unverified
5. **Evaluation metric limitations**: VideoPhy scores are based on automatic evaluators mimicking human judgment, which are inherently subjective and potentially noisy
6. **Fixed chunk size**: The fixed division into 7 equal-length chunks may not match the actual temporal scales of all physical events

## Related Work & Insights

- **vs Hao et al.**: Hao et al. also use counterfactual guidance but only at the global level; PhysVid advances it to chunk-level
- **vs WISA**: WISA uses physics expert mixture modules and physics classifiers to inject global physics information; PhysVid uses VLM for automatic extraction and local injection
- **vs DiffPhy**: DiffPhy uses LLM-enhanced global prompts + MLLM for physics supervision; PhysVid directly learns local physics from videos
- **World model inspiration**: PhysVid can be seen as a step toward physics-aware world simulators — enabling models to understand how physics unfolds over time through local conditioning

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of chunk-level physics conditioning + counterfactual guidance is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multi-subcategory analysis, complete ablation
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, comprehensive related work review
- Value: ⭐⭐⭐⭐ — Architecture-compatible with existing T2V models, good scalability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](../../NeurIPS2025/video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[ACL 2026\] Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity](../../ACL2026/video_generation/accelerating_training_of_autoregressive_video_generation_models_via_local_optimi.md)

</div>

<!-- RELATED:END -->
