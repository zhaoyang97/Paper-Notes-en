---
title: >-
  [Paper Note] [Paper Insight] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][Image Generation][T2I Post-processing] Agentic Retoucher reframes the defect restoration after T2I generation into a human-like closed-loop decision process of "Perception $\to$ Reasoning $\to$ Action," using three collaborative agents for context-aware distortion detection, human-aligned diagnostic reasoning, and adaptive local restoration. It improves plausibility by 2.89 points on GenBlemish-27K, with 83.2% of results rated better than the original by humans.
tags:
  - CVPR 2026
  - Image Generation
  - T2I Post-processing
  - Perception-Reasoning-Action Loop
  - Distortion Detection
  - Local Inpainting
  - GenBlemish-27K
date: 2026-05-08
content_hash: 80136555723e4f43
---

# Agentic Retoucher for Text-To-Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**Code**: None  
**Area**: Image Generation / Agent / Image Quality Assessment  
**Keywords**: T2I Post-processing, Perception-Reasoning-Action Loop, Distortion Detection, Local Inpainting, GenBlemish-27K

## TL;DR

Agentic Retoucher reframes the local defect restoration of T2I generated images into a multi-agent closed-loop decision process of Perception $\to$ Reasoning $\to$ Action. Through context-aware saliency detection, human-preference-aligned diagnostic reasoning, and adaptive tool selection, it achieves autonomous restoration, improving plausibility by 2.89 points on GenBlemish-27K, with 83.2% of results rated better than the original by humans.

## Background & Motivation

**Background**: T2I diffusion models (SDXL, FLUX, Qwen-Image, etc.) can generate highly realistic images for design, film, and entertainment industries. However, even state-of-the-art models frequently produce local small-scale distortions—malformed fingers, facial asymmetry, unreadable text, limb misplacement, and irrational object interactions. These defects typically occur within otherwise high-quality images, making them difficult to detect and expensive to fix.

**Limitations of Prior Work**: Current routes for improving generation quality mainly include prompt enhancement, RL-based optimization, and latent space alignment. They improve global realism but lack explicit spatial reasoning capability, failing to explain or repair local failures. Post-processing editing schemes (e.g., Imagic, Step1x-Edit) support local restoration but rely on manual masks or heuristic text instructions and cannot autonomously identify regions requiring repair.

**Key Challenge**: VLMs seemingly serve as automated critics, but experiments show even GPT-5 cannot reliably locate distortions in AI-generated images—six-fingered portraits are judged normal, and obvious facial distortions are ignored. The fundamental reason is that VLMs are optimized for high-level semantic alignment rather than pixel-level validation; their strong knowledge priors override visual evidence, leading to hallucinatory judgments.

**Goal**: How to equip T2I systems with the ability to autonomously perceive, diagnose, and repair local generation defects while avoiding the unreliability of VLMs in fine-grained spatial localization.

**Key Insight**: Modeling post-processing restoration as a human retoucher's perception-reasoning-action decision loop rather than a one-time feed-forward edit. Three specialized agents perform their duties to form an iteratively converging closed-loop system.

**Core Idea**: Reconstructing T2I post-processing into a self-correcting loop of Perceiving Distortions $\to$ Reasoning Diagnosis $\to$ Precise Restoration using a hierarchical multi-agent decision framework.

## Method

### Overall Architecture

Agentic Retoucher takes a T2I generated image $I_t$ and its corresponding prompt $P$ as input, outputting a restored distortion-free image. The overall process is a three-stage iterative loop: (1) Perception Agent generates a distortion saliency map $S_t$ to locate abnormal regions; (2) If $S_t$ exceeds a threshold $\tau_s$, the Reasoning Agent performs categorical diagnosis and generates text descriptions $\{D_i\}$ and masks $\{M_i\}$ for detected regions; (3) Action Agent selects appropriate restoration methods from a modular tool library based on reasoning results to execute local inpainting, obtaining an updated image $I_{t+1}$. The restored image is re-fed into the Perception Agent for validation, iterating 2-3 rounds until all significant distortions are eliminated.

### Key Designs

1.  **Context-Aware Perception Agent (Context-aware distortion detector)**:
    *   **Function**: Detects context-dependent local distortions from generated images to generate a distortion saliency map $S \in [0,1]^{H \times W}$.
    *   **Mechanism**: Employs a dual-encoder architecture (ViT for image + T5 for prompt), fusing visual and textual representations via self-attention to capture intrinsic correspondences between visual structures and textual semantics. A lightweight attention refinement module aggregates multi-scale contextual cues. Saliency maps are binarized and morphologically dilated to generate mask candidates $\{M_i\}$.
    *   **Design Motivation**: T2I distortions are often context-dependent (e.g., finger count requires global body structure judgment). Conventional pixel-level detection is unreliable. Dual-encoder fusion of prompt semantic information leverages text-image consistency cues to assist localization, while the KLD loss term aligns with human gaze distribution to prevent oversmoothing of saliency maps.

2.  **Human-Aligned Reasoning Agent (Human-aligned reasoning agent)**:
    *   **Function**: Performs diagnostic reasoning on detected distortion regions, outputting distortion category classification and natural language descriptions $\{D_i\}$.
    *   **Mechanism**: Based on VLM (e.g., Qwen2.5-VL-7B) + LoRA fine-tuning, using a two-stage progressive preference alignment training: (a) SFT stage uses cross-entropy loss to establish structured output formats and distortion classification systems (LoRA rank=64, $\alpha$=32); (b) GRPO stage reduces hallucinations via preference alignment reinforcement learning, with rewards based on classification accuracy and alignment of text descriptions with human annotations.
    *   **Design Motivation**: Simple classification or captioning is insufficient to describe distortion types, local features, and contextual relationships. Direct GRPO (without SFT initialization) leads to unstable output formats and factual drift, necessitating progressive training. Experiments confirm GPT-5/Gemini 2.5 Pro zero-shot accuracy is only ~61%, showing general VLMs are unfit for this task.

3.  **Adaptive Action Agent (Adaptive restoration agent)**:
    *   **Function**: Transforms reasoning results $\{M_i, D_i\}$ into controllable local editing operations for precise restoration.
    *   **Mechanism**: Dynamically selects restoration methods from a modular tool library based on computational constraints and user preferences—VLM-based (Qwen-Edit, Gemini 2.5 Flash Image) or Mask-based (Flux-Fill, SD-inpainting). It determines spatial bounds, tool selection, and inpainting instructions for each region.
    *   **Design Motivation**: Different distortion types suit different tools (e.g., text rendering suits VLM-based, geometric distortion suits Mask-based). Tool decoupling allows the framework to plug-and-play new tools independently of specific editing models.

### Loss & Training

*   **Perception Agent Loss**: Hybrid loss $\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE}(S, \hat{S}) + (1-\alpha) \mathcal{L}_{KLD}(S, \hat{S})$. MSE ensures pixel-level precision, while KLD aligns with human gaze distribution to maintain discriminability in ambiguous regions. Learning rate is $2 \times 10^{-5}$.
*   **Reasoning Agent Training**: Phase 1 SFT (Cross-entropy + LoRA), Phase 2 GRPO preference optimization ($\mathcal{L}_{GRPO}$ includes normalized advantage function $\hat{A}_t$ and KL regularization term $\beta D_{KL}[\pi_\theta || \pi_{ref}]$).
*   **Action Agent**: Training-free; invokes pre-trained editing models via APIs.

### GenBlemish-27K Dataset

To support fine-grained supervision and quantitative evaluation, the authors constructed the GenBlemish-27K dataset: 27,507 pixel-level distortion regions annotated in 6,025 T2I images (from 20+ models), covering 12 fine-grained categories across 6 dimensions (limb deformity, facial distortion, text anomaly, etc.). Annotation was completed via a four-stage human-machine collaboration with >95% consistency. Hand distortions account for 46.8%, facial defects 15.7%, with an average of 4.6 annotated regions per image.

## Key Experimental Results

### Main Results

Comparison against multiple inpainting baselines on GenBlemish-27K and SynArtifacts-1K (four perception metrics: plausibility/aesthetics/alignment/overall):

| Dataset | Method | Plausibility↑ | Aesthetics↑ | Alignment↑ | Overall↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| GenBlemish-27K | Original | 44.21 | 53.69 | 57.89 | 47.15 |
| GenBlemish-27K | Qwen-Edit (Direct) | 44.44 | 53.71 | 57.69 | 47.15 |
| GenBlemish-27K | **Ours w/ Qwen-Edit** | **47.10** | **55.75** | **59.54** | **49.27** |
| GenBlemish-27K | Gemini Flash (Direct) | 44.41 | 53.80 | 57.93 | 47.27 |
| GenBlemish-27K | **Ours w/ Gemini Flash** | 46.81 | 55.47 | 59.22 | 48.97 |
| SynArtifacts-1K | Original | 61.53 | 61.63 | 60.65 | 55.35 |
| SynArtifacts-1K | **Ours w/ Gemini Flash** | **65.96** | **65.27** | **62.94** | **58.43** |
| SynArtifacts-1K | **Ours w/ SD-inpainting** | **66.66** | 64.67 | 62.33 | 58.27 |

Human Evaluation (blind review by 5 evaluators):

| Method | Sig. Better(≫) | Better(>) | Tie(≈) | Worse(<) | Sig. Worse(≪) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 4.2% | 22.8% | 60.8% | 9.2% | 3.0% |
| **Ours** | **48.8%** | **34.4%** | 10.2% | 5.8% | 0.8% |

### Ablation Study

**Perception Agent Ablation** (Impact of attention and KLD loss):

| Config | AUC-Judd↑ | NSS↑ | CC↑ | SIM↑ | KLD↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| w/o attn & KLD | 0.9335 | 1.1957 | 0.5518 | 0.3766 | 1.4436 |
| w/o attn | 0.9335 | 1.2153 | 0.5544 | 0.3731 | 1.4412 |
| w/o KLD | 0.9313 | 1.1892 | 0.5546 | 0.3525 | 1.5008 |
| **Full model** | **0.9336** | **1.2087** | **0.5568** | **0.3822** | **1.4313** |

**Reasoning Agent Ablation** (Using Qwen2.5-VL-7B):

| Strategy | Accuracy↑ | SimCSE↑ | Word2Vec↑ | METEOR↑ | ROUGE↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot | 57.76% | 0.6658 | 0.6110 | 0.1678 | 0.0733 |
| +GRPO only | 58.97% | 0.7020 | 0.6592 | 0.1741 | 0.1003 |
| +SFT only | 78.34% | 0.8405 | 0.7768 | 0.4011 | 0.3515 |
| **+SFT+GRPO** | **80.10%** | **0.8426** | **0.7785** | **0.4037** | **0.3530** |

### Key Findings

*   All restoration tools (VLM-based and Mask-based) achieve consistent gains when integrated into Agentic Retoucher, proving the framework is tool-agnostic.
*   Zero-shot distortion classification accuracy of GPT-5 and Gemini 2.5 Pro is only 61.31%/60.28%, significantly lower than Ours (80.10%), proving general VLMs are unsuitable for fine-grained detection.
*   Using only GRPO without the SFT phase performs poorly (58.97%), showing progressive training is necessary.
*   The Perception Agent's AUC-Judd reaches 0.9336, significantly exceeding SALICON (0.9230), RichHF (0.9211), and all general VLMs.

## Highlights & Insights

*   **First T2I post-processing agent system**: Upgrades restoration from a one-time feed-forward operation to an iterative Perception-Reasoning-Action loop, enabling T2I systems to autonomously diagnose and repair. This paradigm naturally supports multi-round convergence, making it more robust than single-pass restoration.
*   **Revealing fine-grained localization blindness in VLMs**: Experiments quantitatively demonstrate serious deficiencies in even the strongest VLMs (including GPT-5) regarding AI-generated image distortion detection, warning against over-reliance on VLMs for AIGC quality assessment.
*   **Tool-decoupled architecture design**: The Action Agent organizes restoration methods as a modular tool library, allowing the framework to plug-and-play any new editing model, providing excellent extensibility.

## Limitations & Future Work

*   Iterative restoration introduces extra computational overhead (2-3 rounds of inference per image), making real-time applications difficult.
*   The restoration tool library is a predefined static set and cannot learn new strategies or self-evolve from restoration experience.
*   Focuses primarily on local geometric distortions (fingers, faces, text), with weaker coverage of style inconsistency or global semantic errors.
*   GenBlemish-27K data distribution is skewed (hand distortions 46.8%), potentially leading to unbalanced perception across other distortion types.

## Related Work & Insights

*   **vs RichHF**: RichHF performs evaluation but not restoration, focusing heavily on face and limb regions. This paper unifies evaluation and restoration in a closed loop.
*   **vs AgenticIR / JarvisArt**: These are general image restoration/retouching agent systems, whereas this paper designs perception and reasoning modules specifically for unique distortion types in AI-generated images.
*   **vs Imagic / Step1x-Edit**: These post-processing methods require manual masks or instructions, whereas this paper achieves full automation from detection to restoration.
*   The perception-reasoning-action closed-loop paradigm is transferable to other generative tasks requiring automated quality control, such as video and 3D generation.

## Rating

*   Novelty: ⭐⭐⭐⭐ Introducing an agent decision system to T2I post-processing is a new perspective, though individual components (saliency detection, VLM reasoning, inpainting) are not entirely new.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + four restoration tools + multi-model ablation + human blind review; lacks direct comparison with end-to-end restoration methods.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, high-quality figures, and concise formalization of the closed-loop paradigm.
*   Value: ⭐⭐⭐⭐ Fills a systematic gap in autonomous T2I quality restoration; the GenBlemish-27K dataset has independent academic value.

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
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)
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
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

</div>

<!-- RELATED:END -->
