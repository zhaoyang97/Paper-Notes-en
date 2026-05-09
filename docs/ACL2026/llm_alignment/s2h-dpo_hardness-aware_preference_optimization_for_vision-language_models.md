---
title: >-
  [Paper Note] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models
description: >-
  [ACL 2026][LLM Alignment][To be supplemented] To be supplemented after a thorough reading of the paper.
tags:
  - ACL 2026
  - LLM Alignment
  - To be supplemented
date: 2026-05-08
content_hash: dcd408cf9a166cfa
---

# S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.18512](https://arxiv.org/abs/2604.18512)
**Code**: None
**Area**: Multimodal VLM / Preference Alignment
**Keywords**: Multi-image reasoning, DPO preference optimization, visual search, difficulty grading, VLM alignment

## TL;DR

This paper proposes a Simple-to-Hard (S2H) DPO framework that constructs multi-image preference data across three progressively harder levels (anchored reasoning → cross-image comparison → global visual search), systematically improving VLM multi-image reasoning while preserving single-image performance.

## Background & Motivation

**State of the Field**: VLMs have achieved remarkable progress in single-image understanding, yet effective reasoning across multiple images remains challenging. Multi-image reasoning requires localizing relevant images, comparing, and integrating information from multiple visual sources.

**Limitations of Prior Work**: Existing multi-image alignment methods (e.g., MIA-DPO) primarily focus on "anchored reasoning" — where the question pre-specifies which image to attend to (e.g., "In image 3, ...") — thereby bypassing global visual search and autonomous cross-image comparison, two critical capabilities. This leaves models underperforming in more complex multi-image scenarios.

**Root Cause**: MIA-DPO trains exclusively on Level 1 data (single-image anchored questions), neglecting the higher-order reasoning capabilities required at Level 2 (multi-image anchored comparison) and Level 3 (global visual search). Different levels induce qualitatively distinct reasoning patterns, and low-level training fails to generalize to higher levels.

**Paper Goals**: To explicitly define the capability hierarchy required for multi-image reasoning and construct preference data covering all levels to comprehensively improve VLM multi-image reasoning.

**Starting Point**: The paper defines a three-level capability hierarchy — Level 1 (reasoning over a pre-specified single image), Level 2 (comparing pre-specified multiple images), and Level 3 (autonomously searching all images to locate those satisfying a given condition) — and constructs corresponding chosen/rejected pairs for DPO training.

**Core Idea**: Chosen/rejected pairs are created via prompt-driven complexity rather than model-specific hallucinations, making the dataset model-agnostic and covering the full reasoning spectrum from simple to hard.

## Method

### Overall Architecture

S2H-DPO converts existing single-image data into multi-image preference data at three levels, with 20K samples per level. Level 1 constructs preference pairs using distractor images and model hallucinations; Level 2 designs kinship recognition and visual arithmetic tasks to test cross-image comparison; Level 3 designs global visual search tasks requiring the model to search all images before localizing the target. All levels are trained jointly.

### Key Designs

1. **Three-Level Reasoning Capability Hierarchy**:

    - Function: Systematically defines the complete capability spectrum for multi-image reasoning.
    - Mechanism: Level 1 (single-image anchored) — "What color is the car in image 2?" requires attending only to the specified image; Level 2 (multi-image anchored comparison) — "Are the cars in images 1 and 3 the same color?" requires cross-image relational comparison; Level 3 (global search) — "Which image contains a white car?" requires inspecting all images to locate the target. Each level strictly demands more capabilities than the previous.
    - Design Motivation: Training solely on Level 1, as in MIA-DPO, is insufficient — different levels induce qualitatively distinct reasoning patterns, and low-level training does not generalize to higher levels.

2. **Model-Agnostic Chosen/Rejected Construction**:

    - Function: Eliminates the need to regenerate data for each new model.
    - Mechanism: Level 1 uses distractor images to trigger hallucinations (identical to MIA-DPO); Level 2 leverages pre-labeled datasets (kinship datasets, synthetic visual arithmetic) to deterministically generate correct/incorrect pairs; Level 3 samples target concept images from ImageNet paired with random distractor images, where chosen responses accurately describe the target image and rejected responses provide generic, non-targeted descriptions. Semantic similarity filtering via CLIP/MPNet removes low-quality pairs.
    - Design Motivation: MIA-DPO relies on model-specific hallucinations to generate rejected samples, necessitating regeneration for each new model. The prompt-driven approach produces contrastive pairs through task design itself, making it universally applicable across models.

3. **Joint Multi-Level Training**:

    - Function: Simultaneously learns reasoning capabilities across all levels.
    - Mechanism: Data from all three levels are mixed and trained with the standard DPO loss $L_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$. Evaluation is conducted on LLaVA-v1.5-7B, Qwen2.5-VL-7B, and Qwen3-VL-2B.
    - Design Motivation: Ablation experiments demonstrate that joint training outperforms training on any single level, as different reasoning levels mutually reinforce each other.

### Loss & Training

Standard DPO loss with temperature $\beta=0.1$, learning rate $5 \times 10^{-5}$, trained for 3 epochs. 20K samples per level.

## Key Experimental Results

### Main Results

| Method | BLINK | MANTIS | NLVR2 | Multi-image Avg. |
|--------|-------|--------|-------|-----------------|
| LLaVA-v1.5 Baseline | 37.1 | 41.9 | 52.1 | 43.7 |
| MIA-DPO | 42.9 | 44.2 | 54.2 | 47.1 |
| S2H-DPO | **43.4** | **47.9** | **55.6** | **49.0** |
| Gain vs. Baseline | +6.3 | +6.0 | +3.5 | +5.3 |

### Ablation Study

| Configuration | Multi-image Avg. | Single-image Avg. | Notes |
|---------------|-----------------|-------------------|-------|
| Level 1 only | 47.1 | Maintained | Equivalent to MIA-DPO |
| Level 2 only | Improved | Maintained | Cross-image comparison is beneficial |
| Level 3 only | Improved | Maintained | Global search is most challenging |
| Level 1+2+3 | **49.0** | **Maintained** | Joint training is optimal |

### Key Findings

- S2H-DPO outperforms MIA-DPO on all multi-image benchmarks, with a more pronounced advantage on harder Level 3 tasks.
- Joint training across all three levels surpasses training on any single level; different reasoning levels mutually reinforce each other.
- A key advantage is that multi-image reasoning gains are achieved without any degradation in single-image performance (no decline on MMStar or POPE).
- Unlike MIA-DPO, S2H-DPO's data construction does not depend on model-specific hallucinations and is universally applicable across models.

## Highlights & Insights

- **Clear and compelling definition of capability levels**: The progressive hierarchy from anchored reasoning → comparison → search, where each level strictly requires more capabilities than the previous, provides a systematic task analysis framework transferable to other multimodal reasoning scenarios.
- **Prompt-driven vs. hallucination-driven contrastive design**: The former generates natural contrast through task difficulty, while the latter relies on model-specific deficiencies. The former is more general and does not become obsolete as models improve.
- **Practical importance of preserving single-image performance**: Multi-image gains should not come at the cost of single-image degradation; S2H-DPO successfully achieves both simultaneously.

## Limitations & Future Work

- The specific task designs for each level (kinship recognition, visual arithmetic) may lack sufficient diversity.
- Level 3 rejected samples are generated by "omitting the target specification," which may result in inconsistent quality.
- Validation is limited to 7B and 2B models; effectiveness on larger models remains unknown.
- Scenarios involving more than four images are not considered.

## Related Work & Insights

- **vs. MIA-DPO**: MIA-DPO relies solely on Level 1 data and model hallucinations; S2H-DPO covers all three levels with model-agnostic data construction.
- **vs. LLaVA-RLHF/HA-DPO**: These methods focus on single-image preference alignment, whereas S2H-DPO targets hierarchical improvement of multi-image reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-level capability hierarchy is insightful, though the underlying methodology (DPO + synthetic data) is not novel in itself.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three multi-image and two single-image benchmarks, three models, and sufficient ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the capability hierarchy visualization is effective, though some descriptions are verbose.
**Code**: To be confirmed
**Area**: llm_alignment
**Keywords**: To be supplemented

## TL;DR
To be supplemented after a thorough reading of the paper.

## Background & Motivation
To be supplemented after a thorough reading of the paper.

## Method
To be supplemented after a thorough reading of the paper.

## Key Experimental Results
To be supplemented after a thorough reading of the paper.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/llm_alignment/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)

<!-- RELATED:END -->
