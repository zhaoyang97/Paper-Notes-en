---
title: >-
  [Paper Note] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models
description: >-
  [ACL 2026][LLM Alignment][Multi-image Reasoning] Ours proposes the Simple-to-Hard (S2H) DPO framework, which systematically enhances multi-image reasoning capabilities of VLMs by constructing preference data across three…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Multi-image Reasoning"
  - "DPO Preference Optimization"
  - "Visual Search"
  - "Hardness Grading"
  - "VLM Alignment"
date: 2026-05-08
content_hash: 003ae0667808eee1
---

# S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.18512](https://arxiv.org/abs/2604.18512)  
**Code**: None  
**Area**: Multimodal VLM / Preference Alignment  
**Keywords**: Multi-image Reasoning, DPO Preference Optimization, Visual Search, Hardness Grading, VLM Alignment

## TL;DR

Ours proposes the Simple-to-Hard (S2H) DPO framework, which systematically enhances multi-image reasoning capabilities of VLMs by constructing preference data across three progressive difficulty levels (point-based reasoning → cross-image comparison → global visual search) while maintaining single-image performance.

## Background & Motivation

**Background**: VLMs have achieved significant progress in single-image understanding, but effective reasoning across multiple images remains challenging. Multi-image reasoning requires locating relevant images, comparing, and integrating information from multiple visual sources.

**Limitations of Prior Work**: Existing multi-image alignment methods (e.g., MIA-DPO) mainly focus on "point-based reasoning"—where the question pre-specifies which image to look at (e.g., "Look at Image 3..."), bypassing the key capabilities of global visual search and autonomous cross-image comparison. This leads to poor performance in more complex multi-image scenarios.

**Key Challenge**: MIA-DPO only utilizes Level 1 data (single-image point-based questions), ignoring higher-order reasoning required for Level 2 (multi-image point-based comparison) and Level 3 (global visual search). Different levels of questions induce qualitatively different reasoning patterns, and low-level training does not generalize to high-level reasoning.

**Goal**: To clearly define the hierarchy of capabilities required for multi-image reasoning and construct preference data covering all levels to comprehensively enhance VLM multi-image reasoning.

**Key Insight**: Define a three-level capability hierarchy—Level 1 (reasoning on a pre-specified single image), Level 2 (comparing pre-specified multiple images), and Level 3 (autonomously searching all images to locate those meeting specific conditions)—and construct corresponding chosen/rejected pairs for DPO training.

**Core Idea**: Create chosen/rejected pairs through prompt-driven complexity (rather than model-specific hallucinations), making the dataset model-agnostic and covering the full reasoning spectrum from simple to hard.

## Method

### Overall Architecture

S2H-DPO transforms existing single-image data into three levels of multi-image preference data, with 20K samples per level. Level 1 utilizes distractor images and model hallucinations to construct preference pairs; Level 2 designs kinship recognition and visual arithmetic tasks to test cross-image comparison; Level 3 designs global visual search tasks that require the model to search all images before locating the target. All levels are jointly trained.

### Key Designs

1. **Definition of Three-Level Reasoning Hierarchy**:

    - **Function**: Systematically defines the complete capability spectrum of multi-image reasoning.
    - **Mechanism**: Level 1 (Single-image point-based)—"What color is the car in Image 2?", requires only the specified image; Level 2 (Multi-image point-based comparison)—"Are the cars in Image 1 and Image 3 the same color?", requires cross-image association and comparison; Level 3 (Global search)—"Which image contains a white car?", requires checking all images to find the target. Each level strictly requires more capabilities than the previous one.
    - **Design Motivation**: Training solely on Level 1 as in MIA-DPO is insufficient; different levels induce distinct reasoning patterns, and low-level training cannot generalize to high-level tasks.

2. **Universal Chosen/Rejected Construction Method**:

    - **Function**: Eliminates the need to regenerate data for each specific model.
    - **Mechanism**: Level 1 uses distractor images to trigger hallucinations (consistent with MIA-DPO); Level 2 utilizes pre-labeled datasets (kinship datasets, synthetic visual arithmetic) to deterministically generate correct/incorrect pairs; Level 3 selects target images from ImageNet combined with random distractors, where the chosen response is an accurate target description and the rejected response is a generalized description without a specific target. Low-quality pairs are filtered using CLIP/MPNet semantic similarity.
    - **Design Motivation**: MIA-DPO relies on model-specific hallucinations for rejected samples, necessitating regeneration for different models. Prompt-driven methods generate contrast through task design itself, ensuring model-agnosticism.

3. **Joint Multi-Level Training**:

    - **Function**: Simultaneously learns reasoning capabilities across all levels.
    - **Mechanism**: Mixes data from all three levels and trains using the standard DPO loss $$L_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$$. Evaluations are conducted on LLaVA-v1.5-7B, Qwen2.5-VL-7B, and Qwen3-VL-2B.
    - **Design Motivation**: Ablation studies indicate that joint training outperforms training on a single level, suggesting that different reasoning levels facilitate each other.

### Loss & Training

Standard DPO loss is used with temperature $\beta=0.1$, a learning rate of $5 \times 10^{-5}$, and training for 3 epochs. Each level consists of 20K samples.

## Key Experimental Results

### Main Results

| Method | BLINK | MANTIS | NLVR2 | Multi-image Avg |
|------|-------|--------|-------|---------|
| LLaVA-v1.5 Baseline | 37.1 | 41.9 | 52.1 | 43.7 |
| MIA-DPO | 42.9 | 44.2 | 54.2 | 47.1 |
| S2H-DPO (Ours) | **43.4** | **47.9** | **55.6** | **49.0** |
| Gain vs Baseline | +6.3 | +6.0 | +3.5 | +5.3 |

### Ablation Study

| Configuration | Multi-image Avg | Single-image Avg | Description |
|------|---------|---------|------|
| Level 1 only | 47.1 | Maintained | Equivalent to MIA-DPO |
| Level 2 only | Gain | Maintained | Cross-image comparison is beneficial |
| Level 3 only | Gain | Maintained | Global search is the most challenging |
| Level 1+2+3 | **49.0** | **Maintained** | Joint training is optimal |

### Key Findings

- S2H-DPO outperforms MIA-DPO across all multi-image benchmarks, with a more pronounced advantage in difficult Level 3 tasks.
- Joint training across three levels is superior to training on any single level, indicating mutual promotion between reasoning tiers.
- Key Advantage: Enhances multi-image reasoning while completely maintaining single-image performance (no degradation on MMStar and POPE).
- Unlike MIA-DPO, the data construction in S2H-DPO does not depend on specific model hallucinations, making it model-agnostic.

## Highlights & Insights

- **Clear and Persuasive Capability Hierarchy**: The progression from point-based → comparison → search provides a systematic framework for task analysis that is transferable to other multimodal reasoning scenarios.
- **Prompt-driven vs. Hallucination-driven Design**: The former creates natural contrast through task difficulty, while the latter depends on specific model flaws. The former is more universal and does not lose effectiveness as models improve.
- **Practical Importance of Maintaining Single-image Performance**: Multi-image improvements should not come at the cost of single-image degradation; S2H-DPO successfully balances both.

## Limitations & Future Work

- Task-specific designs for each level (e.g., kinship recognition, visual arithmetic) may lack sufficient diversity.
- Level 3 rejected samples are generated by "not specifying a target," which may lead to unstable quality.
- Validation is limited to 7B and 2B models; performance on larger models remains unknown.
- Scenarios involving more images (>4) were not considered.

## Related Work & Insights

- **vs. MIA-DPO**: MIA-DPO only uses Level 1 data and relies on model hallucinations, whereas S2H-DPO covers all three levels and uses model-agnostic data construction.
- **vs. LLaVA-RLHF/HA-DPO**: These methods focus on single-image preference alignment, while S2H-DPO focuses on hierarchical enhancement for multi-image reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The definition of the three-level capability hierarchy is insightful, though the methodology (DPO + synthetic data) is standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 3 multi-image and 2 single-image benchmarks across 3 models with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and excellent visualization of capability levels, though some descriptions are verbose.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](../../ICLR2026/llm_alignment/token-importance_guided_direct_preference_optimization.md)
- [\[ACL 2026\] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](topology-enhanced_alignment_for_large_language_models_trajectory_topology_loss_a.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)

</div>

<!-- RELATED:END -->
