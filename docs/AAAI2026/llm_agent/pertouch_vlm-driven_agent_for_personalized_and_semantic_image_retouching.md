---
title: >-
  [Paper Note] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching
description: >-
  [AAAI 2026][LLM Agent][VLM Agent] This paper proposes PerTouch, a framework that integrates a semantic region-level retouching model based on Stable Diffusion + ControlNet with a VLM-driven Agent (incorporating feedback-…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "VLM Agent"
  - "Image Retouching"
  - "Personalized Editing"
  - "Semantic Awareness"
  - "Diffusion Model"
  - "Scene Memory"
date: 2026-05-08
content_hash: 5fb6e4ed971407fb
---

# PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching

**Conference**: AAAI 2026
**arXiv**: [2511.12998](https://arxiv.org/abs/2511.12998)  
**Code**: [https://github.com/Auroral703/PerTouch](https://github.com/Auroral703/PerTouch)  
**Area**: LLM Agent
**Keywords**: VLM Agent, Image Retouching, Personalized Editing, Semantic Awareness, Diffusion Model, Scene Memory

## TL;DR

This paper proposes PerTouch, a framework that integrates a semantic region-level retouching model based on Stable Diffusion + ControlNet with a VLM-driven Agent (incorporating feedback-driven rethinking and scene-aware memory) to achieve fine-grained, personalized image retouching.

## Background & Motivation

**Background**: Deep learning-based image retouching has evolved from end-to-end FCN approaches to controllable methods such as 3D-LUT and curve manipulation, and further to diffusion-prior-based methods like DiffRetouch. More recently, VLM-based Agent systems (e.g., RestoreAgent, PhotoArtAgent) have been applied to low-level vision tasks.

**Limitations of Prior Work**: Existing methods suffer from three fundamental limitations: (1) *Lack of subjectivity modeling*: deterministic architectures produce fixed outputs for given inputs, failing to capture the diversity of user preferences; (2) *Lack of region-level control*: approaches that incorporate external segmentation maps are sensitive to segmentation quality and prone to unnatural artifacts; (3) *Lack of user interaction and personalization*: these methods cannot interpret ambiguous instructions (e.g., "make it a bit brighter") and do not retain long-term editing preferences.

**Key Challenge**: Semantic region-level retouching demands precise spatial control, yet over-reliance on segmentation information sacrifices global aesthetic coherence; personalization requires understanding user intent, but user instructions are typically vague and subjective.

**Key Insight**: The paper leverages diffusion priors to learn high-quality retouching distributions and injects semantic region control via parameter maps; it introduces two complementary training mechanisms—semantic replacement and parameter perturbation—to balance region awareness with global aesthetics; and it designs a VLM Agent with scene memory and feedback-driven rethinking to enable personalization.

## Method

### Overall Architecture

PerTouch consists of two main components: (1) a semantic region retouching model based on Stable Diffusion + ControlNet, using multi-channel parameter maps as control signals; and (2) a VLM-driven personalized Agent responsible for translating natural language instructions into parameter map editing operations.

### Key Designs

1. **Semantic Region Retouching Model**

    - SAM performs panoptic segmentation to obtain semantic regions; four attribute scores are computed (colorfulness, contrast, color temperature, brightness).
    - Attribute scores are fused with the segmentation map to form a multi-channel parameter map, which is injected into Stable Diffusion via ControlNet.
    - The control range is $[-1, 1]$; adjusting parameter values for specific regions produces the corresponding retouching style while preserving global aesthetics.
    - The framework is extensible: any new attribute with a computable region-level score can be incorporated into the control pipeline.

2. **Semantic Replacement Module**

    - During training, a sample is selected at random; a region is chosen with probability proportional to its semantic area.
    - The selected region is replaced by the most attribute-dissimilar region from another sample, creating artificial variation.
    - Purpose: forces the model to learn region boundary awareness and fine-grained retouching capability.
    - Addresses the degeneration of the model into global retouching when parameter maps are injected directly.

3. **Parameter Perturbation Mechanism**

    - Multi-dimensional perturbations (channel shifts, Gaussian blur, etc.) are applied to the parameter maps.
    - This weakens the model's over-sensitivity to segmentation boundaries, allowing the diffusion prior to exert greater influence on global aesthetics.
    - Complementary to semantic replacement: the latter enhances region awareness, while the former prevents over-reliance on segmentation information.

4. **Strong and Weak Instruction Handling in the VLM Agent**

    - *Weak instructions* (e.g., "enhance this image"): median attribute values serve as defaults; historical preferences from scene memory are combined to automatically generate the parameter map.
    - *Strong instructions* (e.g., "significantly increase the brightness of the eagle"): VLM-based object detection localizes the region, SAM performs segmentation, and feedback-driven rethinking precisely adjusts the parameters.
    - Both modes can be applied to the same image simultaneously: weak instructions handle the global adjustment while strong instructions override specific regions.

5. **Feedback-driven Rethinking**

    - An initial control value $c_0$ generates the first-round result, which is sent back to the Agent along with the original image and instruction to evaluate whether the semantic intent is satisfied.
    - If not satisfied, the control value is revised to form a closed loop; convergence to a user-satisfactory result typically occurs within 2–3 rounds.
    - This establishes a learned mapping among language-level adjustment cues, control values, and perceived visual outcomes.

6. **Scene-aware Memory**

    - After each editing session, scene semantics and confirmed parameters are extracted and stored in a memory repository.
    - When editing a new image, a conditional preference distribution is estimated from the memory repository, enabling scene-conditioned personalization.

### Loss & Training

The base model is trained using the standard denoising loss of Stable Diffusion. The dataset is MIT-Adobe FiveK (5,000 RAW images with 5 expert-retouched versions: A/B/C/D/E). Semantic replacement and parameter perturbation are applied during training. The Agent component operates as an inference-time framework and requires no additional training.

## Key Experimental Results

### Main Results (MIT-Adobe FiveK)

| Method | Expert A PSNR | Expert B PSNR | Expert C PSNR | Expert D PSNR | Expert E PSNR |
|--------|-------------|-------------|-------------|-------------|-------------|
| PIENet | 21.52 | 25.91 | 25.19 | 22.90 | 24.12 |
| TSFlow | 20.61 | 25.25 | 25.62 | 22.37 | 23.54 |
| StarEnhancer | 20.71 | 25.73 | 25.52 | 23.39 | 24.46 |
| DiffRetouch | 24.51 | 26.15 | 25.91 | 24.51 | 24.74 |
| **PerTouch** | **25.14** | **27.47** | **26.75** | **25.97** | **25.66** |

### Ablation Study

| Component Variation | Effect |
|---------------------|--------|
| w/o Semantic Replacement | Model degenerates to global retouching; region awareness is lost |
| w/o Parameter Perturbation | Segmentation boundary artifacts appear; global aesthetic inconsistency |
| Both removed | Performance degrades to baseline DiffRetouch level |
| w/o Scene Memory | Same ambiguous instruction cannot differentiate between different user preferences |
| w/o Feedback-driven Rethinking | First-round parameter estimates frequently mismatch user intent |

### Key Findings

- PerTouch achieves the best PSNR on 4 out of 5 expert versions; Expert A PSNR improves by 0.63 dB over DiffRetouch.
- The complementary effect of semantic replacement and parameter perturbation is critical: each alone is insufficient, but their combination simultaneously achieves region control and global aesthetics.
- Feedback-driven rethinking typically converges to user-satisfactory results within 2–3 rounds, substantially outperforming single-round estimation.
- Scene memory demonstrates noticeably improved preference estimation after 5–10 user interactions, reflecting a "better with use" property.

## Highlights & Insights

- The **opposing-yet-unified design of semantic replacement and parameter perturbation** is elegant: the former reinforces region awareness while the latter mitigates over-reliance, and the tension between the two produces an ideal equilibrium.
- **Unified handling of strong and weak instructions** lowers the barrier for non-expert users, who can achieve quick edits via weak instructions, while professional users retain fine-grained control.
- **Scene-aware memory** realizes genuine personalization—not a one-size-fits-all style preference, but adaptive selection of preference parameters according to different scenes.

## Limitations & Future Work

- Currently only four controllable attributes are supported (colorfulness, contrast, color temperature, brightness); incorporating new attributes requires a region-level scoring function.
- The quality of SAM segmentation directly affects results; segmentation errors in complex scenes propagate into the retouching output.
- Feedback-driven rethinking requires multiple rounds of diffusion inference, incurring high computational cost and making it unsuitable for real-time editing scenarios.
- The cold-start problem for scene memory: new users initially lack historical data, limiting the effectiveness of personalization.

## Related Work & Insights

| Aspect | DiffRetouch | PerTouch |
|--------|------------|---------|
| Control Granularity | Global attribute control | Semantic region-level control |
| Interaction Paradigm | Manual parameter adjustment | VLM Agent natural language |
| Personalization | None | Scene memory + historical preferences |
| Boundary Handling | Relies on external segmentation | Softened via semantic replacement + parameter perturbation |

vs. **PhotoArtAgent/MonetGPT and similar Agent retouching systems**: these rely on fixed tool-invocation pipelines and lack personalized adaptation; PerTouch achieves continuous learning of user preferences through scene memory.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | The combined design of semantic replacement + perturbation + scene memory is original; VLM Agent retouching is a frontier direction |
| Technical Depth | ⭐⭐⭐⭐ | The training strategy for diffusion-based region control is carefully designed; the feedback-driven rethinking mechanism is formally well-defined |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive evaluation across 5 expert versions + component ablation + qualitative comparison |
| Practical Value | ⭐⭐⭐⭐⭐ | Targets general-purpose image editing needs; code is open-sourced; scene memory yields improving performance with continued use |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](reflection-driven_control_for_trustworthy_code_agents.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](../../ICCV2025/llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)
- [\[ICCV 2025\] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions](../../ICCV2025/llm_agent/embodied_image_captioning_self-supervised_learning_agents_for_spatially_coherent.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

</div>

<!-- RELATED:END -->
