---
title: >-
  [Paper Note] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models
description: >-
  [ICLR 2026][Image Generation][backdoor attack] This paper is the first to reveal and systematically study "backdoor modality collapse" in multimodal diffusion models — a phenomenon where the backdoor effect degenerates t…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "backdoor attack"
  - "modality collapse"
  - "multimodal diffusion"
  - "Shapley value"
  - "trigger interaction"
date: 2026-05-08
content_hash: 1cd8cc87ff25994e
---

# When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2603.06508](https://arxiv.org/abs/2603.06508)
**Code**: N/A
**Area**: Image Generation
**Keywords**: backdoor attack, modality collapse, multimodal diffusion, Shapley value, trigger interaction

## TL;DR
This paper is the first to reveal and systematically study "backdoor modality collapse" in multimodal diffusion models — a phenomenon where the backdoor effect degenerates to rely on a single modality (typically text) during multimodal backdoor attacks. Two novel Shapley-value-based metrics, TMA and CTI, are proposed to quantify modality contribution and cross-modal interaction, uncovering a "winner-takes-all" dynamic and negative interaction.

## Background & Motivation

**Background**: Backdoor attacks on diffusion models have become an important research direction. Prior work (BadDiffusion, VillanDiffusion, etc.) has demonstrated the feasibility of injecting backdoors in both unimodal and multimodal settings.

**Limitations of Prior Work**: (1) The intuitive assumption that attacking multiple modalities simultaneously yields a stronger backdoor lacks empirical validation; (2) Inspired by modality collapse in multimodal learning, an analogous phenomenon may exist in backdoor attacks; (3) Existing evaluations focus solely on overall attack success rate without decomposing per-modality contributions.

**Key Challenge**: High attack success rates may obscure a critical fact — the backdoor effectively relies on only a subset of modalities. This implies defenders may underestimate the simplicity of the attack (i.e., manipulating the text prompt alone suffices to trigger it).

**Goal**: Does modality collapse exist in multimodal diffusion backdoors? How can the backdoor contribution of each modality and the cross-modal interaction be quantified?

**Key Insight**: The Shapley value framework from cooperative game theory is adopted, treating modalities as "players" and the backdoor shift as the "payoff" for precise contribution decomposition.

**Core Idea**: Shapley values are used to decompose each modality's marginal contribution to backdoor activation (TMA), and superadditivity tests are applied to quantify cross-modal interaction (CTI), thereby revealing the winner-takes-all collapse characteristic.

## Method

### Overall Architecture
Under a bimodal (image + text) setting with modality set $\mathcal{M} = \{I, T\}$, the value function $v(\emptyset), v(\{I\}), v(\{T\}), v(\{I,T\})$ is computed for four coalitions per sample, enabling exact Shapley value computation.

### Key Designs

1. **Per-example Value Function**:

    - For each sample and coalition $S$, the difference between the trigger score and the clean score is computed.
    - $v(S) = s_{\text{tr}}(S) - s_{\text{nr}}(S) = \cos(\mathbf{z}_S, \mathbf{z}_{\text{tr}}) - \cos(\mathbf{z}_S, \mathbf{z}_{\text{cl}})$
    - Cosine similarity in CLIP embedding space is used.
    - This measures whether the output is closer to the backdoor target or the clean reference.

2. **Trigger Modality Attribution (TMA)**:

    - $\phi_I = \frac{1}{2}(v(\{I\}) - v(\emptyset)) + \frac{1}{2}(v(\{I,T\}) - v(\{T\}))$
    - $\phi_T = \frac{1}{2}(v(\{T\}) - v(\emptyset)) + \frac{1}{2}(v(\{I,T\}) - v(\{I\}))$
    - Exact computation in the bimodal case (4 evaluations, no Monte Carlo approximation required).
    - Efficiency axiom: $\phi_I + \phi_T = v(\{I,T\}) - v(\emptyset)$

3. **Cross-Trigger Interaction (CTI)**:

    - $\mathcal{I} = v(\{I,T\}) - v(\{I\}) - v(\{T\}) + v(\emptyset)$
    - $\mathcal{I} > 0$: superadditive cooperation (genuine synergy)
    - $\mathcal{I} < 0$: interference / redundancy
    - Dataset-level aggregation: $\bar{\mathcal{I}} = \frac{1}{|\mathcal{D}_{\text{val}}|} \sum \mathcal{I}(x)$

4. **Experimental Setup**:

    - Three trigger pairs: White-box + mignneko, Eyeglasses + anonymous, Stop-sign + latte coffee
    - Two poisoning protocols: OR (three equal-sized subsets poisoned separately) and AND (joint poisoning only)
    - Three poisoning ratios: 1%, 5%, 10%
    - Model: InstructPix2Pix; Dataset: CelebA

## Key Experimental Results

### Key Results for TMA and CTI

| Trigger Pair | Protocol | Ratio | $\bar{\phi}_I$ (TMA-I) | $\bar{\phi}_T$ (TMA-T) | $\bar{\mathcal{I}}$ (CTI) |
|---|---|---|---|---|---|
| White-box + mignneko | OR | 5% | 0.0060 | 0.9743 | -0.0089 |
| White-box + mignneko | AND | 5% | 0.0045 | 0.9532 | -0.0086 |
| Eyeglasses + anonymous | OR | 5% | 0.1200 | 0.7376 | -0.2174 |
| Eyeglasses + anonymous | AND | 5% | 0.1063 | 0.8907 | -0.2185 |
| Stop-sign + latte coffee | OR | 5% | 0.0043 | 0.9280 | -0.0094 |
| Stop-sign + latte coffee | AND | 5% | 0.0048 | 1.0033 | -0.0101 |

### Key Findings

1. **Modality Dominance**:
    - The text modality TMA is almost always > 0.7, while the image modality TMA is almost always < 0.15.
    - White-box + mignneko at 5% AND: $\bar{\phi}_T = 0.9532$ vs. $\bar{\phi}_I = 0.0045$ — text almost entirely dominates.
    - The backdoor essentially degenerates into a unimodal text backdoor.

2. **Negative Interaction**:
    - CTI is negative or near zero across all configurations.
    - The most severe case: Eyeglasses + anonymous reaches a CTI of −0.22.
    - Joint triggers produce no synergistic effect; instead, interference is observed.

3. **Effect of Poisoning Protocol**:
    - AND poisoning should theoretically place greater emphasis on joint triggering, yet TMA still reveals text dominance.
    - OR poisoning yields a more balanced modality distribution (due to unimodal poisoning subsets), but text still dominates.

4. **Effect of Poisoning Ratio**:
    - The collapse pattern is consistent across the 1%–10% range.
    - Text dominance does not vary with the poisoning ratio.

5. **Security Implications**:
    - Attackers need only manipulate the text prompt (e.g., by appending a rare token) to trigger the backdoor.
    - Image-side triggers are largely redundant, lowering the barrier for attack deployment.
    - Defenses should prioritize anomaly detection on the text modality.

## Highlights & Insights
- **First revelation of an important security phenomenon**: Backdoor modality collapse reshapes the understanding of the threat model for multimodal backdoors.
- **Appropriate theoretical tool**: Shapley values, as the unique attribution method satisfying efficiency, symmetry, dummy player, and additivity axioms, are the optimal choice for decomposing modality contributions.
- **Counterintuitive finding on negative interaction**: This challenges the assumption that "multimodal = stronger attack," providing important guidance for defense design.
- **Practical security implication**: Defenders should recognize that a high-ASR multimodal backdoor may be detectable through text-side inspection alone.

## Limitations & Future Work
- Validation is limited to InstructPix2Pix; larger and more recent multimodal diffusion models remain to be tested.
- The bimodal setting is assumed; Shapley computation in scenarios with three or more modalities requires Monte Carlo approximation.
- No defense mechanism is proposed — the work is purely analytical (though the analysis itself provides significant value for defense design).
- The formation dynamics of modality collapse (i.e., when it emerges during training) could be further explored.

## Related Work & Insights
- **vs. VillanDiffusion**: That work evaluates overall ASR without decomposing modality contributions, potentially leading to the erroneous conclusion that multimodal attacks are stronger.
- **vs. modality collapse in multimodal learning**: This paper is the first to transfer the concept from training optimization to the backdoor attack domain.
- **vs. modality-balancing methods**: This suggests that analogous modality-balancing strategies may be needed in backdoor scenarios as well.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Backdoor modality collapse is an entirely new concept, and the TMA/CTI metrics are well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across three trigger pairs × two protocols × three poisoning ratios.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition and formalization are clear; the game-theoretic framework is applied naturally.
- Value: ⭐⭐⭐⭐ Provides important directional guidance for multimodal backdoor research in the AI security community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [\[ICLR 2026\] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)
- [\[CVPR 2026\] All-in-One Slider for Attribute Manipulation in Diffusion Models](../../CVPR2026/image_generation/all-in-one_slider_for_attribute_manipulation_in_diffusion_models.md)
- [\[AAAI 2026\] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective](../../AAAI2026/image_generation/enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)
- [\[CVPR 2026\] When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](../../CVPR2026/image_generation/when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)

</div>

<!-- RELATED:END -->
