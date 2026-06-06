---
title: >-
  [Paper Note] Empowering Small VLMs to Think with Dynamic Memorization and Exploration
description: >-
  [ICLR 2026][Multimodal VLM][Small-scale vision-language models] This paper proposes DyME (Dynamic Memorize-Explore), which progressively and dynamically alternates between an SFT memorization mode and a GRPO exploration…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Small-scale vision-language models"
  - "reasoning capability"
  - "dynamic switching"
  - "SFT and RLVR integration"
  - "visual supervision"
date: 2026-05-08
content_hash: 3f3f401b5a659390
---

# Empowering Small VLMs to Think with Dynamic Memorization and Exploration

**Conference**: ICLR 2026
**arXiv**: [2506.23061](https://arxiv.org/abs/2506.23061)  
**Code**: [Available](https://github.com/HKUST-LongGroup/DyME)  
**Area**: Multimodal VLM
**Keywords**: Small-scale vision-language models, reasoning capability, dynamic switching, SFT and RLVR integration, visual supervision

## TL;DR
This paper proposes DyME (Dynamic Memorize-Explore), which progressively and dynamically alternates between an SFT memorization mode and a GRPO exploration mode, enabling—for the first time—reasoning capabilities in small-scale vision-language models (SVLMs, <1B parameters) on domain-specific tasks.

## Background & Motivation
- Large models (e.g., Qwen2.5-VL-32B) can acquire reasoning abilities via SFT or RLVR, but **both paradigms fail for small models (SVLMs, <1B)**:
    - SFT failure: CoT data is verbose and contains abundant content irrelevant to vision; SVLMs lack sufficient capacity to absorb it, resulting in "pseudo reasoning traces."
    - RLVR failure: SVLMs exhibit poor instruction-following ability, frequently producing unverifiable outputs, which triggers **advantage collapse**.
- The static balance window for two-stage training (SFT→RL) is extremely narrow, making success nearly impossible for SVLMs.
- Practical motivation: SVLMs are well-suited for edge device deployment, making it highly meaningful to endow them with reasoning capability.

## Method

### Overall Architecture
DyME consists of two core mechanisms:
1. **Dynamic switching mechanism (Pure DyME)**: At each training step, adaptively selects between SFT and GRPO based on the quality of the model's generated outputs.
2. **Visual supervision mechanism**: Injects image-grounded dynamic guidance via a visual checker and a visual refiner.

### Key Designs

**Dynamic switching rule**:

Given input $x$, the policy model samples $K$ responses, each verified for answer correctness via rule-based evaluation:
- If **at least one response is correct** → GRPO mode (exploration)
- If **all responses are incorrect** (including format errors) → SFT mode (memorization)

This binary switch appears simple but proves highly effective: when all responses are incorrect, advantages are dominated by noise, making GRPO unstable; when at least one is correct, GRPO can safely leverage relative advantages to drive exploration.

**Gradient compatibility between SFT and GRPO**:
The authors demonstrate that the two objectives are formally equivalent at the gradient level:
- SFT gradient = log-probability gradient under an external data distribution
- GRPO gradient = weighted log-probability gradient under an internally sampled distribution
- SFT is a special case of GRPO (ground-truth samples + unit advantage values)
- This renders a unified loss function theoretically well-founded.

**GRPO mode enhancements**:
- Introduces an auxiliary reward $r_t$ for reasoning traces (computed via token-level F1 against reference CoT)
- **Removes** KL penalty and clipping (the dynamic SFT switching already provides implicit stability), simplifying the gradient formulation.

**Visual Checker**:
- Extracts visual facts $I_c$ from images (fine-grained visual components such as objects, attributes, and states)
- Evaluates responses along two dimensions: (1) whether they contain sufficient correct visual elements, and (2) whether they conform to style exemplars
- Scores serve as the reasoning reward signal for GRPO.

**Visual Refiner**:
- High-scoring exploration trajectories are stored in a dynamic exemplar pool.
- An LLM samples from the pool and, combined with structural templates and $I_c$, generates visually grounded reference responses for the SFT mode.
- This creates a positive feedback loop: "successful exploration → improved memorization targets."
- Visual fact extraction, checking, and refinement are all implemented via structured prompts to Qwen2.5-14B.

### Loss & Training
The full DyME objective dynamically selects between the GRPO loss and the SFT loss at each step via an indicator function:
- No additional hyperparameter tuning is required (no thresholds, annealing coefficients, or budgets).
- Visual facts are extracted automatically using domain-specific tools (e.g., BiomedGPT for medicine, DePlot for charts).
- Training requires only a few thousand samples.

## Key Experimental Results

### Main Results (across three domains)

| Model | Method | Medical | Chart | Geometry | Avg |
|-------|--------|---------|-------|----------|-----|
| SmolVLM (0.5B) | Baseline | 72.1 | 63.2 | 14.6 | 49.9 |
| SmolVLM | + SFT | 60.1 | 57.7 | 14.5 | 44.1 (↓) |
| SmolVLM | + GRPO | 61.1 | 53.8 | 17.1 | 44.0 (↓) |
| SmolVLM | + Two-stage | 59.4 | 60.1 | 16.7 | 45.4 (↓) |
| SmolVLM | **+ DyME** | **78.1** | **69.7** | **18.9** | **55.6 (+5.7)** |
| LLaVA-OV-S (0.5B) | Baseline | 74.9 | 61.4 | 15.9 | 50.7 |
| LLaVA-OV-S | **+ DyME** | **78.3** | **67.5** | **20.4** | **55.4 (+4.7)** |
| InternVL2-S (0.5B) | Baseline | 78.3 | 71.9 | 18.7 | 56.3 |
| InternVL2-S | **+ DyME** | **80.0** | **74.5** | **19.8** | **58.1 (+1.8)** |

DyME-trained SVLMs outperform MoVA (54.2), a 7B-parameter model.

| Switching Strategy Comparison (Medium Data) | Acc | Extra Cost |
|---------------------------------------------|-----|------------|
| Reward Thresholding ($t=0.5$) | 52.4 | None |
| SFT Annealing (Cosine) | 64.0 | +25% |
| SFT Budget (Hard Mining) | 59.6 | Budget-dependent |
| **Binary Switch (DyME)** | **64.9** | Baseline |

### Ablation Study

| DyME Variant | Medical | Chart | Geometry | Avg |
|--------------|---------|-------|----------|-----|
| DyME (Full) | 78.3 | 67.5 | 20.4 | 55.4 |
| w/o memorization mode | 63.2 | 53.4 | 15.0 | 43.9 (↓20.6%) |
| w/o exploration mode | 75.5 | 61.3 | 14.5 | 50.4 (↓9.0%) |
| w/o Visual Refiner | 75.6 | 62.3 | 16.8 | 51.6 (↓6.9%) |
| w/o Visual Checker | 76.9 | 64.3 | 17.1 | 52.8 (↓4.7%) |

### Key Findings
1. The memorization mode is the cornerstone (removal causes ↓20.6%); the exploration mode serves as the performance engine (removal causes ↓9.0%).
2. Open-source model (Qwen2.5-14B) + Full DyME achieves performance on par with GPT-4o data + Pure DyME.
3. DyME generalizes across modalities: text-only Qwen2.5-0.5B gains +5.8% on GSM8K; 7B models also benefit (+2.3%).
4. Training efficiency: Pure DyME runs at a speed comparable to standard GRPO (~14s/step); Full DyME incurs approximately 1.6× overhead.
5. Replacing the external auxiliary model from 14B to 7B results in negligible performance loss (67.5% vs. 66.8%).

## Highlights & Insights
- **Parameter-free switching rule**: No thresholds, annealing coefficients, or other hyperparameters are required; the binary switch itself constitutes the optimal strategy.
- The **proof of gradient compatibility** between SFT and GRPO provides a theoretical foundation for the unified loss function.
- The Visual Refiner establishes a positive feedback loop—"successful exploration → improved memorization targets"—enabling self-improving data quality throughout training.
- The work validates the intuition that "weaker models require smarter training paradigms."
- Significant gains are achievable even with low-quality (undesigned) CoT data combined with DyME, lowering the data quality barrier.

## Limitations & Future Work
- Visual supervision relies on the extractability of visual facts $I_c$, and may fail in scenarios involving abstract semantics (e.g., meme sarcasm) or unstructured perception.
- Validation is limited to models with ≤7B parameters; whether larger models benefit from DyME remains an open question.
- The current pipeline depends on an external LLM (Qwen2.5-14B) for visual refinement; a fully closed-loop, autonomous self-improvement system would be more desirable.
- Only a few thousand training samples per domain are used; behavior under larger-scale data warrants further investigation.

## Related Work & Insights
- The direct motivation derives from DeepSeek-R1: pure RL can stimulate reasoning, but requires strong base model capabilities.
- DyME is complementary to visual RLVR works such as R1-V and LMM-R1, which focus on large models, whereas DyME targets small models.
- The failure analysis of two-stage training (SFT→RL) provides empirical evidence for understanding the capability boundaries of SVLMs.
- Multimodal-CoT represents an early attempt but is constrained by data scale; G-LLaVA and LLaVA-CoT depend on large-scale CoT data.

## Rating
- Novelty: 5/5 (first work to address reasoning capability in SVLMs; the dynamic switching design is elegant)
- Experimental Thoroughness: 5/5 (dual-track experiments covering algorithm and system validation; three domains, three model architectures, comprehensive ablations)
- Writing Quality: 5/5 (motivation is clearly articulated; the unified gradient analysis of SFT and GRPO is particularly insightful)
- Value: 5/5 (highly practical; directly advances the deployment of SVLMs on edge devices)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](../../CVPR2026/multimodal_vlm/downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)
- [\[CVPR 2026\] Recursive Think-Answer Process for LLMs and VLMs](../../CVPR2026/multimodal_vlm/recursive_think-answer_process_for_llms_and_vlms.md)
- [\[ICLR 2026\] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)
- [\[ICML 2026\] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](../../ICML2026/multimodal_vlm/what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)

</div>

<!-- RELATED:END -->
