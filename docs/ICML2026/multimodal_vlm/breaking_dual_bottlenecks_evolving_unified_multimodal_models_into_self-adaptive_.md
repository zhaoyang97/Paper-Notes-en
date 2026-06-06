---
title: >-
  [Paper Note] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners
description: >-
  [ICML 2026][Multimodal VLM][Unified Multimodal Model] Addressing the "understanding–generation gap" in unified multimodal models on anything-to-image (X2I) tasks (can understand but cannot generate)…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Unified Multimodal Model"
  - "X2I"
  - "Interleaved Reasoning"
  - "GRPO"
  - "Self-Adaptive Planning"
date: 2026-05-08
content_hash: b7a7bb9fd85375ba
---

# Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners

**Conference**: ICML 2026  
**arXiv**: [2605.14709](https://arxiv.org/abs/2605.14709)  
**Code**: GitHub (available, marked as "released at GitHub" in the paper but no specific URL provided)  
**Area**: Multimodal VLM / Unified Models / Reinforcement Learning  
**Keywords**: Unified Multimodal Model, X2I, Interleaved Reasoning, GRPO, Self-Adaptive Planning

## TL;DR
Addressing the "understanding–generation gap" in unified multimodal models on anything-to-image (X2I) tasks (can understand but cannot generate), this paper proposes the Self-Adaptive Interleaved Reasoner: a hierarchical data synthesis pipeline routes 50,000 samples among direct generation, self-reflection, and multi-step planning modes; SFT + GRPO training is applied with step-wise reasoning rewards and intra-group complexity penalties, enabling Emu3.5 to surpass GPT-4o, Gemini 2.5 Flash, and other closed-source models on KRIS-Bench and OmniContext.

## Background & Motivation

**Background**: Unified multimodal models (e.g., Emu3.5, BAGEL, OmniGen) can now perform both understanding and generation within a single framework, and have begun to introduce CoT-style interleaved reasoning to tackle X2I (arbitrary condition → image).

**Limitations of Prior Work**: The authors attribute unified models' failures on complex X2I tasks to the "understanding–generation gap," decomposed into two specific bottlenecks: (i) **attention entanglement bottleneck**—direct one-shot generation from complex prompts almost always fails, necessitating stepwise approaches; however, existing Plan-then-Generate methods perform "blind planning," where the planner is unaware of the generator's actual capabilities, often producing unexecutable plans. (ii) **visual refinement bottleneck**—single-pass pixel synthesis inevitably contains flaws, requiring further reflection and correction; existing Generate-then-Reflect approaches mix "what went wrong" and "how to fix" in unstructured text, leading to inefficiency in handling compound errors, and often rely on switching between multiple models, greatly increasing inference cost.

**Key Challenge**: Both strategies (Plan-then-Generate and Generate-then-Reflect) address only one bottleneck each and are fixed in process; instruction complexity varies greatly, so applying a single mode uniformly leads to over-reasoning for simple prompts or insufficient reasoning for complex ones. No existing method can "adaptively select modes based on prompt complexity."

**Goal**: Train a unified model capable of autonomously switching among "direct generation / reflection correction / multi-step planning" based on instruction complexity and its own capabilities, while maintaining generation efficiency without relying on external models.

**Key Insight**: First, use a hierarchical escalation data pipeline to automatically assign prompts of varying complexity to three modes; then use SFT to teach the model syntax, and finally use RL to teach the model strategy (when to use which mode for optimal results).

**Core Idea**: Make "when to reason more" a reinforcement learning objective for autonomous model decision-making—step-wise rewards ensure logical reasoning processes, and intra-group complexity penalties suppress over-reasoning by discouraging marginal gains from extra steps.

## Method

### Overall Architecture
Two-stage pipeline: **(A) Data Construction**—given raw X2I input, first let a baseline unified model generate directly; Qwen3-VL-235B (Analyzer) scores on four dimensions: instruction, consistency, quality, and commonsense; if passed, assign to *Direct*; otherwise, enter up to 3 rounds of self-reflection (Analyzer writes reflection prompt, Gemini-3-Pro-Image as Generator redraws); if still unsuccessful after 3 rounds, Analyzer diagnoses the failure—if due to "prompt too complex," escalate to *Multi-step* mode (decompose into sub-tasks with intermediate evaluation), otherwise (e.g., lacking domain knowledge) discard. All samples are double-checked by two human annotators, resulting in 50,000 high-quality interleaved data points. **(B) Training**—SFT adapts to interleaved reasoning syntax + selective loss masking skips failed intermediate images; GRPO reinforces strategy selection, with rewards weighted by Outcome / Format / Step-wise reasoning, plus an intra-group complexity penalty to encourage "winning with fewer steps."

### Key Designs

1. **Hierarchical Escalation Data Pipeline (Analyzer ⇋ Generator)**:

    - **Function**: Automatically routes X2I data into Direct / Self-Reflection / Multi-step execution paths, corresponding to different complexities.
    - **Mechanism**: Qwen3-VL-235B acts as "reviewer + diagnostician + planner," Gemini-3-Pro-Image as "generator." Each data point first undergoes direct generation + four-dimensional scoring; if not passed, reflection is performed (up to 3 rounds); if still not passed and diagnosed as "overly complex," escalate to multi-step planning, and after final success, perform trajectory pruning to remove failed reflections, leaving a clean "direct attempt fails → sub-task decomposition → stepwise images" trajectory. Final manual verification.
    - **Design Motivation**: Training samples themselves demonstrate "mode selection by complexity"—simple prompts learn direct generation, complex prompts learn explicit decomposition, and intermediate ones learn reflection and correction.

2. **Selective Loss Masking in SFT**:

    - **Function**: During SFT, prevents the model from learning visual artifacts from "failed intermediate images," while retaining semantic signals on "how to correct errors."
    - **Mechanism**: Loss is computed only on the selected subsequence $\mathcal{O}$. For Direct mode, $\mathcal{O}=\{G_1, E_1\}$; for Self-Reflection, only the last diagnosis $E_{K-1}$, reflection prompt $R_{K-1}$, and final successful image $G_K, E_K$ are included, with all previous failed intermediate images masked; for Multi-step, $E_1$ plus the full planning sequence $\{S_i, G_i, E_i\}$ are included.
    - **Design Motivation**: If autoregressive NLL includes failed images, it teaches the model "how to generate low-quality images," which harms fidelity; masking them ensures the model treats failure information as "context for reflection" rather than "imitation targets."

3. **GRPO + Step-wise Reasoning Reward + Intra-group Complexity Penalty**:

    - **Function**: Enables the model to autonomously select the most efficient execution path.
    - **Mechanism**: Combined reward $\mathcal{R}_{\text{total}}=\alpha_1\mathcal{R}_o+\alpha_2\mathcal{R}_f+\alpha_3\mathcal{R}_s$, where $\mathcal{R}_o$ is the LMM's weighted average of four outcome scores, $\mathcal{R}_f$ is a binary structural validity, and $\mathcal{R}_s=\frac{1}{T}\sum_t \text{Analyzer}(\text{text}_t)$ is a dense reasoning reward for each intermediate text (failure analysis, reflection prompt, sub-step decomposition). Most crucial is the intra-group complexity penalty: within the same group of sampled trajectories, identify the subset "close to the highest reward" (within $\epsilon$ threshold), and scale by image count $N_{\text{img}}^i$—add $N_{\text{img}}^*/N_{\text{img}}^i$ to the reward, so trajectories achieving equivalent results with fewer images are further rewarded.
    - **Design Motivation**: Pure outcome rewards incentivize "more steps for higher scores," leading to over-reasoning; the intra-group penalty implicitly optimizes for "winning with the fewest steps," naturally assigning simple prompts to Direct and complex prompts to Multi-step.

### Loss & Training
SFT: standard AR-NLL on subset $\mathcal{O}$ (Eq. 1). RL: GRPO policy + combined reward as above (Eq. 2–5). Backbone = Emu3.5; RL data: 50,000 samples from UnicEdit-10M / X2Edit / AnyEdit / Pick-a-Pic / UltraEdit.

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
- Removing Reflection drops KRIS by 3 points (78.24 → 75.21), removing Multi-step drops Omni by 0.2 (9.15 → 8.95): the two modes address "quality refinement" and "complex multi-agent" respectively, and are not interchangeable.
- Removing intra-group complexity penalty causes average generated images to surge from 1.56 to 2.73 (+75%), but Omni only slightly increases to 9.38—confirming its role in suppressing over-reasoning.
- SFT→SFT+RL reduces average images from 2.45 to 1.56 while improving quality, indicating RL is indeed learning "winning with fewer steps."
- Largest gains on OmniContext's Multiple / Scene complex multi-agent scenarios (9.56 / 9.44 vs Emu3.5's 8.65 / 8.78), confirming the planning mode targets "attention entanglement."

## Highlights & Insights
- Elevates "when to reason more" to an optimizable strategy, and incorporates efficiency into the RL signal via intra-group complexity penalty—an explicit modeling of "quality and efficiency" rarely seen in current reasoning-in-generation research.
- The data pipeline uses dual LLMs (Analyzer ⇋ Generator) for automatic escalation, making "complexity-based routing" an automated process, not reliant on fixed "plan-then-generate" or "generate-then-reflect" templates, and directly transferable to other multimodal tasks requiring adaptive reasoning depth.
- Selective loss masking is an underrated trick: in multi-step tasks involving "intermediate failures," whether failed steps are included in NLL directly determines if the final model is contaminated by failure samples.

## Limitations & Future Work
- Strong reliance on Qwen3-VL-235B and Gemini-3-Pro-Image, both closed-source large models, for data construction and step-wise reward calculation, making reproduction difficult and costly, and potentially transferring Analyzer biases to training targets.
- The paper focuses on X2I editing/synthesis tasks; extension to video, 3D, or longer-horizon generation tasks remains unverified.
- The "failure → reflection → redraw" loop escalates to multi-step after at most 3 rounds, a hard threshold that may miss medium-complexity cases fixable with 4–5 rounds of reflection; learned confidence could replace the fixed iteration cap.

## Related Work & Insights
- **vs Plan-then-Generate (Uni-CoT / Echo-4o)**: They perform static text planning then execution; this work combines reflection and planning, with RL selecting the mode; +1.1–1.5 points on OmniContext.
- **vs Generate-then-Reflect (VACoT)**: They perform iterative reflection without explicit planning; this work explicitly separates "analysis" and "improvement," and adds multi-step planning for complex prompts.
- **vs Emu3.5 (Backbone)**: Also a unified model, backbone only achieves 0.86 / 73.75 / 8.82; interleaved reasoning + RL boosts KRIS to 80.18, Omni to 9.35, proving "adaptive strategy" is the next gain dimension for unified models.

## Rating
- Novelty: ⭐⭐⭐⭐ First to make "adaptive mode selection" an explicit RL optimization objective, with a clever complexity penalty design; though individual components (Plan-then-Generate / Generate-then-Reflect / GRPO) are not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers GenEval / KRIS-Bench / OmniContext benchmarks, ablates data modes and RL components, and reports average generated images to reflect efficiency.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative (gap → two bottlenecks → adaptive solution), with Fig. 1 / Fig. 2 / Fig. 3 illustrating comparison, data, and RL, and clean structure.
- Value: ⭐⭐⭐⭐⭐ Enables open-source Emu3.5 to surpass GPT-4o on KRIS-Bench, providing the unified model community with a practical RL-based strategy learning route.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](../../CVPR2026/multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)
- [\[ICCV 2025\] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](../../ICCV2025/multimodal_vlm/iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](../../ICLR2026/multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)
- [\[ICML 2026\] DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction](dcer_dual-stage_compression_and_energy-based_reconstruction.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)

</div>

<!-- RELATED:END -->
