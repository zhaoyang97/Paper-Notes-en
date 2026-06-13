---
title: >-
  [Paper Note] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration
description: >-
  [CVPR 2026][Image Restoration][Image Quality Assessment] RAR deeply integrates image quality assessment (IQA) with image restoration (IR) into a unified end-to-end model…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Image Quality Assessment"
  - "Iterative Restoration"
  - "Composite Degradation"
  - "Flow Matching"
date: 2026-05-08
content_hash: dd687c493ea858e1
---

# RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2603.26385](https://arxiv.org/abs/2603.26385)  
**Code**: [https://restore-assess-repeat.github.io/](https://restore-assess-repeat.github.io/)  
**Area**: Image Restoration
**Keywords**: Image Restoration, Image Quality Assessment, Iterative Restoration, Composite Degradation, Flow Matching

## TL;DR

RAR deeply integrates image quality assessment (IQA) with image restoration (IR) into a unified end-to-end model, iteratively executing an "assess–restore–verify" loop in the latent space. It achieves a +2.71 dB PSNR gain under composite degradation scenarios while running 11.27× faster than AgenticIR.

## Background & Motivation

Real-world image degradation is complex and unknown, potentially involving simultaneous blur, noise, rain, haze, and other artifacts. Existing approaches fall into two categories: all-in-one models (unified models handling multiple degradation types, but with limited performance) and agentic models (agents that iteratively select specialized tools, achieving better results but at extremely high latency).

**Key Challenge**: All-in-one models lack precise degradation recognition, while agentic models maintain a complete separation between IQA and IR modules—requiring repeated image encoding/decoding and LLM-based planning, resulting in bloated pipelines with significant information loss.

**Key Insight**: RAR combines the strengths of both paradigms—employing VLM-based free-text IQA (not restricted to predefined degradation categories) while deeply integrating IQA and IR within a shared latent space, enabling end-to-end trainable iterative restoration.

## Method

### Overall Architecture

RAR builds upon DepictQA (IQA module) and SD3.5 (IR module), aligning their latent spaces via adapters. A degraded image is encoded into the latent space → LQA assesses current quality → its output logits are directly used as conditions for IR (without text decoding) → the restored image is reassessed → the loop repeats until a stopping criterion is met.

### Key Designs

1. **Latent Quality Assessment (LQA)**:

    - **Function**: Integrates the IQA module into the restoration model's latent space for end-to-end training.
    - **Mechanism**: An input adapter $\mathcal{A}_I$ bridges the IR latent encoding to the IQA input space (avoiding image decoding), while an output adapter $\mathcal{A}_Q$ directly aligns the IQA output logits to the IR conditioning embedding space (bypassing text decoding and the text-conditioning branch). Training proceeds in two stages: adapter-only fine-tuning followed by full-parameter fine-tuning.
    - **Design Motivation**: Eliminates the IQA → text → IR information bottleneck, and enables removal of the IR text-conditioning branch to reduce parameters and latency.

2. **Flow Matching-Based Direct Mapping**:

    - **Function**: Directly maps from the degraded distribution to the high-quality distribution, supporting iterative updates.
    - **Mechanism**: Conventional diffusion models start from noise, and the noisy intermediate states prevent LQA from performing accurate assessment. RAR instead adopts Flow Matching to learn a direct mapping from the degraded distribution $\rho_{deg}$ to the high-quality distribution $\rho_{hq}$: $\mathcal{L}_v = \mathbb{E}\|v_\theta(\mathbf{z}_t^n, Q_{deg}^n, t) - (\mathbf{z}_{hq} - \mathbf{z}_{deg}^n)\|^2$. Intermediate states are linear interpolations between the degraded input and the target, always constituting "semantically meaningful images" that LQA can accurately evaluate.
    - **Design Motivation**: For the iterative assess–restore loop to function, intermediate representations must be interpretable by the IQA module, which fundamentally rules out noise-based diffusion schemes.

3. **RAR Stopping Criterion**:

    - **Function**: Automatically determines when to terminate iterative restoration.
    - **Mechanism**: Every $T$ steps, LQA compares image quality before and after restoration. Leveraging DepictQA's native support for pairwise image comparison, it outputs a binary decision: CONTINUE (the restored image has higher quality) or STOP (quality no longer improves). Upon STOP, the result from the previous iteration is retained.
    - **Design Motivation**: Adaptive stopping prevents over-restoration and unnecessary computation, allowing images with different degradation levels to naturally receive different numbers of iterations.

### Loss & Training

A flow matching velocity field loss is combined with the IQA training loss from LQA. During training, RAR's iterative process is seamlessly integrated into the standard flow matching training procedure—LQA can be invoked at arbitrary time steps to update the conditioning signal.

## Key Experimental Results

### Main Results

| Method | Composite Degradation PSNR↑ | MANIQA↑ | CLIP-IQA↑ | Speed |
|------|-------------|---------|-----------|------|
| AgenticIR | 21.04 | 0.3071 | 0.4474 | 1× |
| AutoDIR | 19.64 | 0.2500 | 0.3767 | — |
| MiOIR | 20.84 | 0.2451 | 0.3933 | — |
| **RAR** | **20.46** | **0.4659** | **0.6566** | **11.27×** |

RAR achieves substantial gains on perceptual metrics (MANIQA, CLIP-IQA) while running 11.27× faster than AgenticIR.

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| w/o LQA (fixed text condition) | Significant drop | Dynamic assessment conditioning is critical |
| Noise diffusion replacing flow matching | IQA failure | Noisy intermediate states cannot be assessed |
| w/o iteration (single-pass restoration) | Incomplete composite degradation handling | Iteration is necessary for composite degradation |
| Full RAR | Best | All components contribute synergistically |

### Key Findings

- Flow matching's direct mapping is a critical design choice for iterative restoration—the noise-injection process of diffusion models is fundamentally incompatible with iterative IQA evaluation.
- The iterative process naturally follows an order such as "denoise first, then dehaze," consistent with a degradation-difficulty ordering from easy to hard.
- End-to-end integration offers substantial advantages over pipeline-based approaches in both latency and performance.

## Highlights & Insights

- **Deep IQA–IR Integration**: The paradigm shifts from "two independent models collaborating" to "two capabilities within one model." This design philosophy is transferable to other tasks requiring assess–execute loops.
- **An Overlooked Advantage of Flow Matching**: Compared to diffusion models, a previously underappreciated advantage of flow matching is that its intermediate representations carry physical meaning, making iterative evaluation feasible.
- **Stopping Criterion Design**: Repurposing IQA's pairwise comparison capability as a stopping signal elegantly resolves the question of "how many iterations are sufficient."

## Limitations & Future Work

- PSNR falls short of AgenticIR in certain settings, reflecting a trade-off between fidelity and perceptual quality.
- The method relies on DepictQA's evaluation capacity and may fail for degradation types it does not handle well.
- The stopping criterion may lack robustness in certain edge cases.
- Future work could explore integration with larger generative models.

## Related Work & Insights

- **vs. AgenticIR**: AgenticIR uses an LLM for planning with specialized tools—powerful but extremely slow. RAR unifies assessment and restoration into a single model, achieving 11× speedup with end-to-end trainability.
- **vs. AutoDIR**: AutoDIR employs CLIP for closed-set degradation classification, whereas RAR uses DepictQA for open-set free-text assessment, offering stronger generalization.
- **vs. PromptIR/MiOIR**: All-in-one methods lack dynamic assessment capability and are insufficient for handling composite degradations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of IQA–IR latent space integration with flow matching-based iteration is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage of composite, single, and unknown degradations, with both fidelity and perceptual metrics reported.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear architecture presentation, rigorous logic, and polished illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Represents a paradigm-level contribution to the image restoration field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](../../ICCV2025/image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)

</div>

<!-- RELATED:END -->
