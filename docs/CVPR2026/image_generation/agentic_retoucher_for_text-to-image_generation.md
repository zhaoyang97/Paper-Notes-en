---
title: >-
  [Paper Note] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][Image Generation][Text-to-Image Generation] The problem of correcting local distortions (deformed fingers, facial abnormalities, text errors, etc.) in T2I diffusion model outputs is modeled as a Perception-Reasoning-Action multi-agent cyclic system named Agentic Retoucher. It utilizes a Perception Agent to locate defects via context-aware distortion saliency maps, a Reasoning Agent to diagnose distortion types through structured reasoning, and an Action Agent to execute repairs via tool selection. Combined with the GenBlemish-27K dataset, it achieves end-to-end iterative automatic correction.
tags:
  - CVPR 2026
  - Image Generation
  - Text-to-Image Generation
  - Post-processing Correction
  - Multi-Agent Systems
  - Distortion Detection
  - Image Inpainting
date: 2026-05-08
content_hash: b709633bd34da1bc
---

# Agentic Retoucher for Text-To-Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**Code**: To be confirmed  
**Area**: Image Generation  
**Keywords**: Text-to-Image Generation, Post-processing Correction, Multi-Agent Systems, Distortion Detection, Image Inpainting

## TL;DR

The problem of correcting local distortions (deformed fingers, facial abnormalities, text errors, etc.) in T2I diffusion model outputs is modeled as a Perception-Reasoning-Action multi-agent cyclic system named Agentic Retoucher. It utilizes a Perception Agent to locate defects via context-aware distortion saliency maps, a Reasoning Agent to diagnose distortion types through structured reasoning, and an Action Agent to execute repairs via tool selection. Combined with the GenBlemish-27K dataset, it achieves end-to-end iterative automatic correction.

## Background & Motivation

While the generation quality of current T2I diffusion models (e.g., SDXL, DALL-E 3) continues to improve, structural distortion problems in local details remain prevalent:

- **Hand Distortion**: Extra/missing fingers, misaligned joints, accounting for up to 46.8%.
- **Facial Abnormalities**: Disproportionate or asymmetrical features.
- **Text Rendering Errors**: Missing letters, distorted strokes.
- **Physical Illogicality**: Perspective errors, contradictory occlusion relationships.

Existing post-processing solutions mainly rely on VLMs (e.g., GPT-4V) as critics, but face two key bottlenecks:

**Weak Spatial Localization**: VLMs excel at global semantic judgment but struggle to precisely locate pixel-level distortion areas; their descriptions are often vague natural language rather than actionable spatial coordinates.

**Hallucination Issues**: VLMs may misidentify normal areas as distorted or overlook real defects, leading to unnecessary modifications or missing critical issues.

The core insight of Agentic Retoucher is that instead of relying on a single general-purpose VLM for the full localization+judgment+repair pipeline, the problem should be decomposed into three specialized agents—Perception, Reasoning, and Action—each performing its role through an iterative loop to achieve progressive correction.

## Method

### Overall Architecture

Agentic Retoucher adopts a three-stage cyclic architecture:

1. **Perception Agent**: Detects and localizes distorted regions in the image, outputting a set of binary mask candidates $\{M_i\}$.
2. **Reasoning Agent**: Performs type diagnosis and natural language description for each distorted region, outputting $\{D_i\}$.
3. **Action Agent**: Selects appropriate repair tools to perform inpainting based on masks and descriptions.

The three agents collaborate via an iterative loop: the repaired image is sent back to the Perception Agent for verification; if the saliency score $S > \tau$, the cycle continues, otherwise it terminates.

### Key Designs

**1. Perception Agent — Context-Aware Distortion Saliency Detection**

- **Encoder**: ViT extracts visual features + T5 encodes semantic features of the text prompt, using dual-stream fusion.
- **Attention Refinement**: Cross-modal attention injects text semantics into visual features, allowing the model to understand "what a place should look like" to accurately judge deviations.
- **Output**: A pixel-wise context-aware distortion saliency map $S \in [0, 1]^{H \times W}$.
- **Loss**:
$$\mathcal{L}_{\text{percept}} = \alpha \cdot \text{MSE}(S, S_{\text{gt}}) + (1 - \alpha) \cdot \text{KLD}(S \| S_{\text{human}})$$
  Where $S_{\text{human}}$ is the human fixation distribution (eye-tracking data). The KLD term aligns the model's distortion judgment with human visual attention.
- **Post-processing**: Binarization of $S$ (threshold $\theta$) + morphological dilation to generate the mask candidate set $\{M_i\}$, ensuring the repair area covers distortion boundaries.

**2. Reasoning Agent — Structured Distortion Diagnosis**

- **Base**: SFT based on VLM, using efficient LoRA fine-tuning.
- **Structured Initialization**: Encodes the distortion taxonomy (12 artifact categories) into structured prompts to guide the model to output standardized diagnostic results.
- **GRPO for Human Preference Alignment**: Group Relative Policy Optimization uses human-annotated preference contrast data for further alignment—making diagnostic descriptions more consistent with human judgments of distortion severity and type.
- **Output**: Distortion type labels and natural language descriptions for each mask region $\{D_i\} = \{(\text{type}_i, \text{desc}_i)\}$.

**3. Action Agent — Tool Selection and Repair Execution**

- **Tool Library**:
    - **Mask-guided inpainting**: Local repainting based on masks, suitable for distortions with clear structures (e.g., extra fingers).
    - **Instruction-driven inpainting**: Repair based on natural language instructions, suitable for distortions requiring semantic understanding (e.g., unnatural expressions).
- **Selection Strategy**: Automatically routes to the most appropriate tool based on distortion type $\text{type}_i$.
- **Iterative Verification**: The repaired image is re-evaluated by the Perception Agent; if significant distortion remains, it enters the next round of the cycle.

### Loss & Training

**Perception Agent Training**:
- Hybrid loss: MSE ensures pixel-level accuracy, KLD ensures consistency with human fixation distribution.
- $\alpha = 0.7$ (determined by ablation study).

**Reasoning Agent Training**:
- Stage 1: SFT + LoRA fine-tuning on GenBlemish-27K annotated data.
- Stage 2: GRPO preference alignment, optimizing with human A/B comparison data.

**GenBlemish-27K Dataset Construction**:
- 6K T2I generated images with 27K manually annotated distortion regions.
- Distribution of 12 artifact classes: hand (46.8%), face (15.7%), text (8.3%), body (7.2%), etc.
- Each annotation includes: bounding box, pixel-level mask, distortion type, severity, and natural language description.

## Key Experimental Results

### Main Results

| Method | Plausibility↑ | Aesthetics↑ | Human Pref. (%)↑ |
|------|-------------|------------|-------------------|
| Original T2I Output | 44.21 | 5.32 | — |
| VLM-Critic (GPT-4V) | 45.03 | 5.41 | 61.5 |
| HiveMind | 45.67 | 5.48 | 68.3 |
| **Ours** | **47.10** | **5.63** | **83.2** |

Agentic Retoucher improves Plausibility from 44.21 to 47.10 (+2.89), with 83.2% of human reviewers preferring the repaired results.

### Ablation Study

| Configuration | Plausibility↑ | Human Pref.↑ |
|------|-------------|-------------|
| Perception only (No Reasoning) | 45.38 | 69.1% |
| Perception + Reasoning only (No Iteration) | 46.22 | 76.4% |
| No KLD alignment | 46.01 | 73.8% |
| No GRPO preference alignment | 46.45 | 78.1% |
| **Full Agentic Retoucher** | **47.10** | **83.2%** |

### Key Findings

- **Iterative Loop is Critical**: Human preference increased from 76.4% to 83.2% when comparing single vs. iterative repair, indicating some distortions require multi-round progressive correction.
- **KLD Human Fixation Alignment is Effective**: Removal leads to a 1.09 drop in Plausibility, showing that aligning the perception model with human visual attention significantly improves localization accuracy.
- **GRPO Preference Alignment contributes stably**: Without it, the preference rate drops from 83.2% to 78.1%, validating the improvement of reasoning quality via human preference signals.
- **Category Analysis**: Improvements are most significant for hand distortions (+3.8 Plausibility), followed by faces (+2.1), while text remains the most difficult (+0.9).

## Highlights & Insights

1. **Elegant Problem Decoupling**: Splitting "problem discovery, diagnosis, and repair" into three specialized agents is more reliable than using a general VLM—aligning with the principle of separation of concerns in software engineering.
2. **Two-layer Injection of Human Priors**: The Perception Agent aligns with human fixation (low-level perception) via KLD, while the Reasoning Agent aligns with human preferences (high-level semantics) via GRPO, forming a complementary pair.
3. **Value of GenBlemish-27K Dataset**: The first large-scale annotated dataset for T2I distortions, providing a benchmark with fine-grained annotations of 12 artifact categories.
4. **Practicality of the Iterative Loop**: Analogous to the human retouching process of "finding → modifying → checking → re-modifying," the system design fits real-world application needs.

## Limitations & Future Work

- Repair quality is capped by the pre-trained inpainting models; if the underlying tools are weak, agent decisions cannot produce superior results.
- Termination conditions for the iterative loop (threshold $\tau$) require manual setting and may need adaptive adjustment for different distortion types.
- Dataset imbalance exists (hands at 46.8%); repair effectiveness for minority classes (e.g., perspective errors) may be insufficient.
- Textual distortion repair shows limited improvement (+0.9), potentially requiring specialized text rendering models.
- Inference overhead: Sequential agent processing and multiple iterations result in high latency, limiting real-time applications.

## Related Work & Insights

- **Instruct-Pix2Pix / MagicBrush**: Instruction-based image editing methods lacking automatic distortion detection.
- **VLM-as-Judge Paradigm**: Paradigms like GPT-4V as a critic; this paper points out its weak spatial localization and hallucination issues.
- **Saliency Detection**: Borrowing human fixation alignment ideas from eye-tracking prediction.
- **RLHF / GRPO**: Transferring LLM alignment techniques to visual reasoning tasks.
- **Insight**: The multi-agent + iterative loop paradigm can be extended to other visual correction tasks requiring "perception-reasoning-action," such as video stabilization, 3D model repair, or medical image quality control.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of a multi-agent cyclic framework to T2I post-processing; the three-agent decoupled design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablations, human preference evaluations, and categorical analyses, though more comparisons with baselines are needed.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture description, transparent dataset construction, and persuasive motivation.
- Value: ⭐⭐⭐⭐⭐ The GenBlemish-27K dataset + plug-and-play post-processing framework provides direct help for practical T2I applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

</div>

<!-- RELATED:END -->
