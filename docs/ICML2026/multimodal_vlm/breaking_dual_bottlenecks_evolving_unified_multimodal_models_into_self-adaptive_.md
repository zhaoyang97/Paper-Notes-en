---
title: >-
  [Paper Note] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners
description: >-
  [ICML 2026][Multimodal VLM][Unified Multimodal Models] To address the "understanding–generation gap" (capable of understanding but failing to generate) in unified multimodal models for anything-to-image (X2I) tasks…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Unified Multimodal Models"
  - "X2I"
  - "Interleaved Reasoning"
  - "GRPO"
  - "Adaptive Planning"
date: 2026-05-08
content_hash: 8bd5dd3655976add
---

# Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners

**Conference**: ICML 2026  
**arXiv**: [2605.14709](https://arxiv.org/abs/2605.14709)  
**Code**: GitHub (Released at GitHub as noted in the paper, but no specific URL provided)  
**Area**: Multimodal VLM / Unified Models / Reinforcement Learning  
**Keywords**: Unified Multimodal Models, X2I, Interleaved Reasoning, GRPO, Adaptive Planning

## TL;DR
To address the "understanding–generation gap" (capable of understanding but failing to generate) in unified multimodal models for anything-to-image (X2I) tasks, this paper proposes the Self-Adaptive Interleaved Reasoner. Utilizing a hierarchical data synthesis pipeline, 50,000 samples are diverted among direct generation, self-reflection, and multi-step planning modes. The model is trained using SFT + GRPO equipped with step-wise reasoning rewards and intra-group complexity penalties, enabling Emu3.5 to surpass closed-source models such as GPT-4o and Gemini 2.5 Flash on KRIS-Bench and OmniContext.

## Background & Motivation

**Background**: Unified multimodal models (e.g., Emu3.5, BAGEL, OmniGen) can perform both understanding and generation within a single framework and have begun introducing CoT-style interleaved reasoning to tackle X2I (arbitrary conditions → image) tasks.

**Limitations of Prior Work**: The authors attribute the failure of unified models in complex X2I tasks to the "understanding–generation gap," decomposed into two specific bottlenecks: (i) **the attention entanglement bottleneck**—direct one-step generation for complex prompts almost inevitably fails, necessitating step-by-step processes; however, current Plan-then-Generate methods perform "blind planning" where the planner is unaware of the generator's actual execution capabilities, leading to impractical plans. (ii) **the visual refinement bottleneck**—single-pass pixel synthesis inherently contains flaws requiring further reflection and repair; however, current Generate-then-Reflect approaches mix "what is wrong" and "how to fix it" in unstructured text, which is inefficient for composite errors and often relies on switching between multiple models, causing inference costs to soar.

**Key Challenge**: Both strategies (Plan-then-Generate and Generate-then-Reflect) only solve one bottleneck individually and follow fixed workflows. Since instruction complexity varies significantly, applying a single rigid mode either leads to over-reasoning for simple prompts or insufficient reasoning for complex ones. No existing method can "adaptively select a mode based on prompt complexity."

**Goal**: To train a unified model capable of autonomously switching between "Direct Generation," "Reflection/Correction," and "Multi-step Planning" based on instruction complexity and its own capabilities, maintaining generation efficiency without relying on external models.

**Key Insight**: A hierarchical escalation data pipeline is first used to automatically categorize prompts of different complexities into three modes. SFT is then used to teach the model the syntax, followed by Reinforcement Learning (RL) to teach strategy (determining which mode is most cost-effective).

**Core Idea**: The decision of "when to reason more" is formulated as a reinforcement learning objective. Step-wise rewards ensure logical reasoning processes, while intra-group complexity penalties suppress over-reasoning where "more steps are used for marginal gains."

## Method

### Overall Architecture
A two-stage pipeline: **(A) Data Construction**—Given a raw X2I input, a baseline unified model first performs direct generation; Qwen3-VL-235B (Analyzer) scores the output across four dimensions: "Instruction, Consistency, Quality, and Commonsense." If it passes, it is categorized as *Direct*; otherwise, it enters a self-reflection loop of up to 3 rounds (where the Analyzer writes reflection prompts and Gemini-3-Pro-Image acts as the Generator for redrawing). If it still fails after 3 rounds, the Analyzer diagnoses the failure; if the cause is "excessive prompt complexity," it escalates to *Multi-step* mode (decomposing sub-tasks for step-by-step execution + intermediate evaluation); otherwise (e.g., missing domain knowledge), it is discarded. All samples are verified by two human annotators, resulting in 50,000 high-quality interleaved data points. **(B) Training**—SFT adapts the model to interleaved reasoning syntax with selective loss masking to skip failed intermediate images; GRPO reinforces strategy selection with a reward weighted by Outcome, Format, and Step-wise Reasoning, plus an intra-group complexity penalty to encourage "winning with fewer steps."

### Key Designs

1.  **Hierarchical Escalation Data Pipeline (Analyzer ⇋ Generator)**:
    *   **Function**: Automatically diverts X2I data into Direct, Self-Reflection, or Multi-step execution paths corresponding to different complexities.
    *   **Mechanism**: Qwen3-VL-235B serves as the "Reviewer + Diagnostician + Planner," while Gemini-3-Pro-Image acts as the "Generator." Each data point undergoes direct generation and four-dimensional scoring. If it fails, reflection is performed (up to 3 rounds). If it still fails and is diagnosed as "overly complex," it escalates to multi-step planning. Upon final success, trajectory pruning is applied to remove failed reflection attempts, leaving a clean trajectory: "initial direct failure → sub-task decomposition → step-by-step image generation." Final human auditing is performed.
    *   **Design Motivation**: To make the training samples demonstrate "mode selection by complexity"—simple prompts learn direct generation, complex prompts learn explicit decomposition, and intermediate ones learn reflection and error correction.

2.  **SFT with Selective Loss Masking**:
    *   **Function**: Avoids the model learning visual artifacts from "failed intermediate images" during the SFT stage while retaining semantic signals on "how to correct errors."
    *   **Mechanism**: The loss is only calculated on the selected subsequence $\mathcal{O}$. For *Direct* mode, $\mathcal{O}=\{G_1, E_1\}$. For *Self-Reflection* mode, it includes only the final diagnosis $E_{K-1}$, reflection prompt $R_{K-1}$, and the successful final image $G_K, E_K$, while all preceding failed intermediate images are masked. For *Multi-step* mode, it includes $E_1$ plus the complete planning sequence $\{S_i, G_i, E_i\}$.
    *   **Design Motivation**: If autoregressive NLL is applied to failed images, it effectively teaches the model "how to generate low-quality images," harming fidelity. Masking them ensures the model treats failure information as "context for reflection" rather than an "imitation target."

3.  **GRPO + Step-wise Reasoning Reward + Intra-group Complexity Penalty**:
    *   **Function**: Enables the model to autonomously select the most efficient execution path.
    *   **Mechanism**: The total reward is $\mathcal{R}_{\text{total}}=\alpha_1\mathcal{R}_o+\alpha_2\mathcal{R}_f+\alpha_3\mathcal{R}_s$, where $\mathcal{R}_o$ is the weighted average of four-dimensional outcome scores from an LMM, $\mathcal{R}_f$ is a binary structural validity flag, and $\mathcal{R}_s=\frac{1}{T}\sum_t \text{Analyzer}(\text{text}_t)$ is a dense reasoning reward scoring each segment of intermediate text (failure analysis, reflection prompts, sub-step decomposition). Crucially, the intra-group complexity penalty identifies a subset of trajectories "close to the maximum reward" (within a threshold $\epsilon$) and scales them by the image count $N_{\text{img}}^i$. Specifically, a term $N_{\text{img}}^*/N_{\text{img}}^i$ is added to the reward, further rewarding trajectories that achieve equivalent results with fewer images.
    *   **Design Motivation**: Simple outcome rewards could lead to over-reasoning because the model might assume more steps always yield higher scores. The intra-group penalty sets "achieving the same score with the fewest steps" as an implicit optimization goal, naturally assigning simple prompts to Direct mode and complex ones to Multi-step.

### Loss & Training
SFT: Standard AR-NLL on subset $\mathcal{O}$ (Eq. 1). RL: GRPO policy with the combined reward mentioned above (Eq. 2–5). Backbone = Emu3.5. RL data consists of 50,000 samples from UnicEdit-10M, X2Edit, AnyEdit, Pick-a-Pic, and UltraEdit.

## Key Experimental Results

### Main Results

| Benchmark | GPT-4o | Gemini 2.5 Flash | Emu3.5 (vanilla) | Ours |
|---|---|---|---|---|
| KRIS-Bench Overall | 80.09 | 77.29 | 73.75 | **80.18** |
| KRIS Procedural | 78.32 | 75.93 | 71.14 | **85.53** |
| KRIS Factual | 79.80 | 77.03 | 78.59 | **84.24** |
| OmniContext Avg. | 8.80 | 7.84 | 8.82 | **9.35** |
| GenEval | – | – | 0.86 | **0.89** |

### Ablation Study

| Configuration | GenEval | KRIS | Omni | Avg. Imgs |
|---|---|---|---|---|
| Direct Only | 0.86 | 75.16 | 8.89 | – |
| w/o Reflection | 0.86 | 75.21 | 9.03 | – |
| w/o Multi-step | 0.87 | 77.24 | 8.95 | – |
| Full Mix (SFT) | 0.88 | 78.24 | 9.15 | – |
| SFT Only (50k) | 0.86 | 79.16 | 9.12 | 2.45 |
| w/o Step-wise Reward | 0.88 | 79.65 | 9.25 | 1.62 |
| w/o Complexity Penalty | 0.89 | 80.25 | 9.38 | 2.73 |
| SFT + RL (Full) | **0.89** | **80.18** | **9.35** | **1.56** |

### Key Findings
- Removing Reflection drops KRIS by 3 points (78.24 → 75.21), and removing Multi-step drops Omni by 0.2 (9.15 → 8.95): These two modes handle "quality repair" and "complex multi-subject scenarios" respectively and cannot substitute for each other.
- Without the intra-group complexity penalty, the average number of generated images surges from 1.56 to 2.73 (+75%), while Omni only marginally increases to 9.38—confirming that the penalty effectively suppresses over-reasoning.
- Moving from SFT to SFT+RL reduces the average image count from 2.45 to 1.56 while simultaneously increasing quality, suggesting RL truly learns to "win with fewer steps."
- The greatest improvements are seen in multi-subject complex scenarios such as "Multiple / Scene" in OmniContext (9.56 / 9.44 vs Emu3.5's 8.65 / 8.78), validating that the planning mode specifically targets "attention entanglement."

## Highlights & Insights
- Framing "when to reason more" as an optimizable strategy and incorporating efficiency into the RL signal via intra-group complexity penalty is a rare instance of explicit modeling for "both quality and efficiency" in the reasoning-in-generation field.
- The data pipeline uses an Analyzer ⇋ Generator dual-LLM automatic escalation, turning "complexity-based diversion" into an automated workflow. It does not rely on fixed "plan-then-generate" or "generate-then-reflect" templates and can be directly applied to other multimodal tasks requiring adaptive reasoning depth.
- Selective loss masking is an underrated trick: in multi-step tasks involving "intermediate failures," whether or not failure steps are included in the NLL directly determines if the final model is contaminated by failed examples.

## Limitations & Future Work
- Strong dependency on Qwen3-VL-235B and Gemini-3-Pro-Image for data construction and step-wise rewards makes reproduction difficult and expensive, and it may propagate the Analyzer's biases to the training objective.
- The paper focuses on X2I editing/synthesis; whether this can be extended to longer-horizon tasks like video or 3D generation remains unverified.
- Escalating to multi-step planning after at most 3 rounds of "failure → reflection → redrawing" is a hard threshold that might miss moderately complex samples that could have been fixed in 4-5 rounds. Using learned confidence instead of a fixed iteration limit could be considered.

## Related Work & Insights
- **vs Plan-then-Generate (Uni-CoT / Echo-4o)**: These perform static text planning followed by execution. Ours performs reflection and planning simultaneously, with RL selecting the mode, gaining +1.1–1.5 points on OmniContext.
- **vs Generate-then-Reflect (VACoT)**: These perform iterative reflection without explicit planning. Ours explicitly separates "analysis" and "improvement" and adds multi-step planning for complex prompts.
- **vs Emu3.5 (Backbone)**: As a unified model, the vanilla backbone achieves only 0.86 / 73.75 / 8.82. Interleaved reasoning + RL raised KRIS to 80.18 and Omni to 9.35, proving that "adaptive strategy" is the next dimension of gain for unified models.

## Rating
- Novelty: ⭐⭐⭐⭐ First to formulate "adaptive mode selection" as an explicit RL optimization objective; complexity penalty is cleverly designed. However, individual components (Plan-then-Generate / Generate-then-Reflect / GRPO) are not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three major benchmarks: GenEval / KRIS-Bench / OmniContext. Ablations separately analyze data modes and RL components, and average image counts are reported to demonstrate efficiency.
- Writing Quality: ⭐⭐⭐⭐ The narrative (gap → two bottlenecks → adaptive solution) is clear. Fig. 1 / Fig. 2 / Fig. 3 diagrams effectively explain comparisons, data, and RL structures.
- Value: ⭐⭐⭐⭐⭐ Surpassing GPT-4o with the open-source Emu3.5 on KRIS-Bench points to a practical and effective route of "using RL to learn strategies" for the unified model community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement](diva_harnessing_the_representation_divergence_in_unified_multimodal_models_for_m.md)
- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](../../CVPR2026/multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)
- [\[ICCV 2025\] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](../../ICCV2025/multimodal_vlm/iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](../../ICLR2026/multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)

</div>

<!-- RELATED:END -->
