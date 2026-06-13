---
title: >-
  [Paper Note] [Paper Insight] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][Image Generation][T2I Post-processing] Agentic Retoucher reframes the defect restoration after T2I generation into a human-like closed-loop decision process of "Perception $\to$ Reasoning $\to$ Action…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "T2I Post-processing"
  - "Perception-Reasoning-Action Loop"
  - "Distortion Detection"
  - "Local Inpainting"
  - "GenBlemish-27K"
date: 2026-05-08
content_hash: 80136555723e4f43
---

# [Paper Insight] Agentic Retoucher for Text-To-Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**Code**: None  
**Area**: Image Generation / Agent / Image Quality Assessment  
**Keywords**: T2I Post-processing, Perception-Reasoning-Action Loop, Distortion Detection, Local Inpainting, GenBlemish-27K  

## TL;DR
Agentic Retoucher reframes the defect restoration after T2I generation into a human-like closed-loop decision process of "Perception $\to$ Reasoning $\to$ Action," using three collaborative agents for context-aware distortion detection, human-aligned diagnostic reasoning, and adaptive local restoration. It improves plausibility by 2.89 points on GenBlemish-27K, with 83.2% of results rated better than the original by humans.

## Background & Motivation
T2I diffusion models (SDXL, FLUX, etc.) generate high-quality images but often produce local distortions—malformed fingers, facial asymmetry, unreadable text, limb misplacement. Existing solutions either require expensive full-image regeneration or rely on VLMs for automated assessment, yet VLMs exhibit weak spatial localization (VLMs judge six-fingered images as normal). There is a lack of an automated system capable of **Autonomous Discovery $\to$ Diagnosis $\to$ Repair** of local defects.

## Core Problem
How to equip T2I models with the ability to autonomously perceive and fix generative defects? How to resolve the unreliability (misjudgment due to hallucination) of VLMs in fine-grained defect detection?

## Method

### Overall Architecture
Agentic Retoucher consists of three collaborative agents in a closed loop: (1) Perception Agent generates distortion saliency maps to locate problematic regions; (2) Reasoning Agent performs diagnostic reasoning (classification + text description) on located regions; (3) Action Agent selects tools for local restoration based on reasoning results. Restored images are re-sent to the Perception Agent for checking, iterating 2-3 rounds until no significant distortions remain.

### Key Designs
1.  **Context-Aware Perception Agent (Context-aware distortion detector)**: Uses a dual-encoder architecture (ViT for image + T5 for prompt) to fuse visual and textual information via self-attention, generating distortion saliency maps $S \in [0,1]^{H \times W}$. Trained with a hybrid loss: $\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE} + (1-\alpha) \mathcal{L}_{KLD}$, where the KLD term aligns with human gaze distribution. Outperforms traditional saliency models and general VLMs by over 10 percentage points on AUC-Judd.

2.  **Human-Aligned Reasoning Agent (Human-aligned reasoning agent)**: Based on Qwen2.5-VL-7B + LoRA fine-tuning. Two-stage training: (a) SFT stage establishes structured output formats and distortion classification (LoRA rank=64); (b) GRPO stage uses preference alignment to reduce hallucinations. Ultimately achieves 80.10% categorical accuracy (vs GPT-5 Zero-Shot 61.31%) and 0.8517 SimCSE for semantic descriptions.

3.  **Adaptive Action Agent (Adaptive restoration agent)**: Selects restoration methods from a modular tool library—VLM-based (Qwen-Edit, Gemini 2.5 Flash Image) or Mask-based (Flux-Fill, SD-inpainting). Determines spatial bounds, tool selection, and instructions based on reasoning, validating again after the loop.

### Loss & Training
- Perception Agent: MSE + KLD hybrid loss
- Reasoning Agent: SFT (Cross-entropy) + GRPO (Preference optimization, rewards based on classification accuracy and text alignment)

## Key Experimental Results

| Dataset | Condition | Plausibility | Aesthetics | Alignment | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GenBlemish-27K | Original | 44.21 | 53.69 | 57.89 | 47.15 |
| GenBlemish-27K | Ours w/ Qwen-Edit | **47.10** | **55.75** | **59.54** | **49.27** |
| SynArtifacts-1K | Ours w/ Gemini Flash | 65.96 | 65.27 | 62.94 | **58.43** |

Human evaluation: 83.2% of restored results were judged better than the original (48.8% significantly better + 34.4% slightly better).

### Ablation Study
- **Perception Agent**: Removing attention reduces SIM and CC; removing KLD loss reduces NSS and AUC-Judd.
- **Reasoning Agent**: GRPO only (no SFT) yields poor results (58.97% accuracy); SFT+GRPO is optimal (80.10%).
- **Tool Selection**: All tools (Qwen-Edit, Gemini, Flux-Fill, SD-inpainting) improve with Agentic Retoucher, showing framework tool-agnosticism.
- **GPT-5 and Gemini 2.5 Pro Zero-Shot**: Perform poorly on distortion reasoning (61.31%/60.28%), showing general VLMs are not adept at this task.

## Highlights
- First to model T2I post-processing restoration as a "Perception-Reasoning-Action" closed-loop agent system rather than simple one-time repair.
- GenBlemish-27K dataset provides 27K pixel-level annotated distortion regions across 12 defect categories, the first large-scale T2I defect annotation dataset.
- Experiments prove VLMs (including GPT-5) cannot reliably detect AI-generated image distortions in zero-shot settings—a significant finding.
- Framework is decoupled from specific restoration tools, allowing plug-and-play with different editing models.

## Limitations & Future Work
- Iterative restoration introduces extra computational overhead (2-3 rounds of inference).
- Current restoration tools are predefined and cannot learn new strategies.
- Primarily targets local geometric distortions (fingers, face); weaker coverage for style inconsistency or global semantic errors.
- GenBlemish-27K hand distortions account for 46.8%, indicating skewed data distribution.

## Related Work & Insights
- **vs RichHF**: RichHF evaluates but doesn't repair, focusing heavily on face/limb regions. Agentic Retoucher both evaluates and repairs in a closed loop.
- **vs AgenticIR/JarvisArt**: General image restoration/editing agents. Agentic Retoucher is specifically designed for AI-generated distortion types.
- **vs Imagic/Step1x-Edit**: These require manual masks or instructions. Agentic Retoucher automates localization and repair.

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing an agent decision system to T2I restoration is a new perspective, though components aren't individualy new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + multiple tools + ablation + human evaluation; lacks direct comparison with end-to-end methods.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, high-quality figures.
- Value: ⭐⭐⭐⭐ Fills a gap in automated T2I quality restoration; GenBlemish-27K has independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](../../ICML2026/image_generation/ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

</div>

<!-- RELATED:END -->
