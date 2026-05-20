---
title: >-
  [Paper Note] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation
description: >-
  [ICLR 2026][Reinforcement Learning][Sound Separation] MARS-Sep reformulates query-conditioned sound separation as a reinforcement learning problem…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Sound Separation"
  - "Multimodal Alignment"
  - "Beta Policy"
  - "Preference Reward"
date: 2026-05-08
content_hash: b0c5830b8e215f70
---

# MARS-Sep: Multimodal-Aligned Reinforced Sound Separation

**Conference**: ICLR 2026
**arXiv**: [2510.10509](https://arxiv.org/abs/2510.10509)  
**Code**: [https://github.com/mars-sep/MARS-Sep](https://github.com/mars-sep/MARS-Sep)  
**Area**: Audio Processing / Reinforcement Learning
**Keywords**: Sound Separation, Reinforcement Learning, Multimodal Alignment, Beta Policy, Preference Reward

## TL;DR

MARS-Sep reformulates query-conditioned sound separation as a reinforcement learning problem, performing stochastic decisions over time-frequency bins via a factorized Beta mask policy, and leverages a progressively aligned multimodal encoder to provide semantic reward signals, achieving simultaneous improvements in signal fidelity and semantic consistency.

## Background & Motivation

**Background**: Universal Sound Separation aims to isolate individual sound sources from arbitrary audio mixtures. Query-conditioned sound separation further allows users to specify target sources via audio, text, or image queries. Current mainstream methods (e.g., AudioSep, OmniSep) primarily optimize signal-level loss functions (e.g., SDR, SI-SDR), reconstructing target waveforms by predicting time-frequency masks.

**Limitations of Prior Work**: Existing methods face a fundamental "metric dilemma"—models optimized for waveform reconstruction may score high on signal metrics while still producing outputs with perceptually salient interference components that violate the semantic correspondence implied by the query. For example, models optimizing SDR may fail to distinguish between acoustically similar but semantically distinct sources (e.g., violin vs. viola), since signal-level losses encode no semantic information.

**Key Challenge**: There exists a fundamental misalignment between signal-level optimization objectives (low-level feature matching) and semantic-level separation requirements (high-level semantic alignment). Conventional regression-based mask prediction directly supervises against ground-truth masks, offering no mechanism to incorporate the semantic intent of the query into the optimization process.

**Goal**: (1) How can the separation model's optimization objective account for both signal fidelity and semantic consistency? (2) How can mask prediction be transformed from deterministic regression into an explorable stochastic decision? (3) How can a stable and semantically rich reward signal be obtained?

**Key Insight**: Inspired by RLHF, the authors draw an analogy between query-conditioned sound separation and preference alignment—the user query represents a preference, and the goal is to produce outputs that maximize semantic alignment with that query. The separation model is treated as a base policy and optimized via reinforcement learning.

**Core Idea**: A factorized Beta distribution policy performs stochastic mask sampling over time-frequency bins; a progressively aligned multimodal encoder provides semantic rewards; and a trust-region surrogate objective stabilizes training.

## Method

### Overall Architecture

MARS-Sep builds upon the OmniSep architecture. The input consists of a mixed audio spectrogram $X$ and a multimodal query $Q$ (audio/text/image). The separator predicts a deterministic mask proposal $P_\theta(X,Q) \in [0,1]^{H \times W \times K}$, which is then parameterized as a factorized Beta distribution policy $\pi_\theta(M|X,Q)$ from which a stochastic mask $M$ is sampled. The sampled mask is applied to the spectrogram, and the waveform $\hat{y}$ is reconstructed via iSTFT. A progressively aligned multimodal encoder (based on ImageBind) computes a semantic consistency reward $R$ between the separated audio and the query. The policy is updated via a clipped trust-region surrogate objective.

### Key Designs

1. **Factorized Beta Mask Policy**:

    - Function: Transforms deterministic mask prediction into an explorable stochastic policy.
    - Mechanism: The separator output $P_\theta$ is converted into Beta distribution parameters: $\pi_\theta(M|X,Q) = \prod_{h,w,k} \text{Beta}(M_{h,w,k}; \alpha_{h,w,k}, \beta_{h,w,k})$, where $\alpha = 1 + \kappa P_\theta$ and $\beta = 1 + \kappa(1-P_\theta)$. The concentration scale $\kappa > 0$ controls the exploration–exploitation trade-off. The factorized structure allows each time-frequency bin to be sampled independently, with the log-probability factorizing across bins.
    - Design Motivation: The $[0,1]$ support of the Beta distribution naturally matches the mask value range. Annealing $\kappa$ prevents degenerate near-binary masks in early training. Compared to Gaussian policies or discretization, Beta distribution is more natural and avoids truncation issues.

2. **Clipped Trust-Region Surrogate**:

    - Function: Stabilizes policy updates, avoiding the high variance and collapse associated with plain policy gradients.
    - Mechanism: The importance ratio is defined as $r_\theta(M) = \pi_\theta(M|X,Q) / \pi_{\theta_{\text{old}}}(M|X,Q)$, with group-relative advantage $\tilde{A} = (A - \mu(A))/(\sigma(A) + \varepsilon)$. The clipped surrogate objective is: $\mathcal{J}_{\text{clip}}(\theta) = \mathbb{E}[\min(r_\theta \tilde{A}, \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon)\tilde{A}) + \lambda_H \mathcal{H}(\pi_\theta) - \lambda_{\text{KL}} \text{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}})]$, incorporating entropy regularization and KL penalty.
    - Design Motivation: Single-step PPO updates keep the training loop simple, requiring neither an auxiliary value network nor complex advantage estimators. GRPO-style normalized advantages eliminate the effect of reward scale.

3. **Progressive Multimodal Encoder Alignment**:

    - Function: Trains a reliable multimodal reward model to prevent reward hacking.
    - Mechanism: The ImageBind encoder is fine-tuned in three stages. Stage 1: audio–text alignment, unfreezing only the projection head and temperature parameter, using a symmetric InfoNCE loss $\mathcal{L}_{S1}$ to establish semantic anchors. Stage 2: audio–audio discrimination, adding triplet loss and consistency loss $\mathcal{L}_{S2}$ to enhance intra-class discriminability, with partial Stage 1 data mixed in to prevent forgetting. Stage 3: audio–video grounding, jointly applying InfoNCE and triplet loss $\mathcal{L}_{S3}$ while retaining capabilities from prior stages.
    - Design Motivation: Using a pretrained ImageBind directly as a reward model leads to reward hacking—the policy learns to "fool" the reward rather than genuinely improve separation quality. Progressive training equips the encoder with incrementally stronger source discriminability, yielding a more stable and informative reward signal.

4. **Multimodal Reward Aggregation (Query-Pooling Reward)**:

    - Function: Fuses audio, text, and visual query modalities into a unified reward signal.
    - Mechanism: Multimodal low-rank bilinear pooling (MLBP) fuses target-side embeddings: $z^* = \text{MLBP}(\phi_a(y^*), \phi_t(t^*), \phi_v(v^*))$, and the scalar reward is $R = \text{sim}(\phi_a(\hat{y}), z^*)$. The separated audio retains its native representation while target modalities are fused into a semantic anchor.
    - Design Motivation: Computing rewards per modality separately risks over-weighting a single modality. Bilinear pooling explicitly models cross-modal interactions (e.g., an instrument specified by text should also appear visually), encouraging the separated audio to align simultaneously with all modalities.

### Loss & Training

The total training loss is $\mathcal{L}_{\text{RL}}(\theta) = -\mathcal{J}_{\text{clip}}(\theta)$, comprising the clipped surrogate objective, entropy regularization, and KL penalty. At each training step, masks are sampled from the frozen old policy $\pi_{\theta_{\text{old}}}$; rewards are computed; the current policy is updated; and the current policy snapshot is taken as the old policy for the next step.

## Key Experimental Results

### Main Results on VGGSOUND-clean+

| Method | Query | SDR↑ | SIR↑ | SAR↑ | SI-SDRi↑ | CLAP↑ |
|--------|-------|------|------|------|----------|-------|
| AudioSep | Text | 6.26 | 8.69 | 12.85 | 4.01 | 8.21 |
| OmniSep | Text | 6.70 | 9.04 | 13.61 | 4.38 | 8.98 |
| **MARS-Sep** | **Text** | **6.91** | **9.14** | **13.73** | **4.55** | **9.03** |
| OmniSep | Image | 6.66 | 10.00 | 13.73 | 4.43 | 8.79 |
| **MARS-Sep** | **Image** | **6.93** | **10.18** | **13.41** | **4.57** | **9.19** |
| OmniSep | Omni | 7.79 | 10.76 | 14.53 | 5.16 | 8.85 |
| **MARS-Sep** | **Omni** | **7.93** | **10.65** | **14.49** | **5.20** | **9.22** |

### Cross-Domain Validation on MUSIC-clean+

| Method | Query | SDR↑ | SIR↑ | SAR↑ | SI-SDRi↑ | CLAP↑ |
|--------|-------|------|------|------|----------|-------|
| CLIPSEP-NIT | Text | 11.03 | 16.40 | 17.37 | 7.53 | 5.29 |
| OmniSep | Text | 12.37 | 17.51 | 17.96 | 9.18 | 5.41 |
| **MARS-Sep** | **Text** | **12.91** | **17.61** | **18.28** | **9.85** | **6.18** |
| OmniSep | Image | 13.03 | 18.97 | 17.88 | 10.21 | 6.53 |
| **MARS-Sep** | **Image** | **13.64** | **19.24** | **18.05** | **10.70** | **6.94** |

### Key Findings

- **Simultaneous improvement on signal and semantic metrics**: MARS-Sep consistently leads on CLAP score (demonstrating improved semantic alignment) while also achieving comprehensive gains in SDR/SIR/SI-SDRi, indicating that the RL reward does not sacrifice signal quality.
- **Strong cross-domain generalization**: From VGGSound (300+ sound categories) to the instrument-focused MUSIC dataset, MARS-Sep's gains are maintained or even amplified, with a CLAP improvement of +0.77 (14.2% relative) on MUSIC.
- **Comparison with generative methods**: Generative approaches such as FlowSep and ZeroSep exhibit extremely high CLAP score variance (e.g., ZeroSep on MUSIC: $20.02 \pm 15.14$), whereas MARS-Sep achieves $6.18 \pm 0.93$, demonstrating far superior stability.
- **Necessity of progressive alignment**: Using a pretrained ImageBind directly as the reward model without three-stage fine-tuning leads to reward hacking, causing separation quality to degrade.

## Highlights & Insights

- **A refined analogy between sound separation and RLHF**: The authors successfully instantiate the "user query = preference" analogy into a complete RL framework; the natural correspondence between the Beta distribution and the mask value range represents an elegant engineering choice.
- **Robustness of the progressive alignment strategy**: The three-stage curriculum (semantic anchoring → intra-class discrimination → cross-modal grounding) avoids the instability of single-step alignment, and mixing data from prior stages at each stage prevents catastrophic forgetting.
- **Simplicity of the actor-only design**: By forgoing a value network and complex advantage estimation, the method achieves stable training with single-step PPO and a moving-average baseline, demonstrating that complex RL infrastructure is unnecessary for this single-step MDP formulation of mask prediction.

## Limitations & Future Work

- **Limitations of the single-step MDP**: The current formulation treats mask prediction as a one-step decision, ignoring temporal structure—sequential decision-making may be more effective for separating long audio.
- **Generality of the reward model**: Progressive alignment relies on ImageBind as the backbone, which may limit effectiveness for sound categories not well covered by ImageBind.
- **Computational overhead**: RL training requires multiple mask samples and reward computations per step; the additional training cost relative to direct supervised learning is not reported.
- **Absence of human evaluation**: Only objective metrics are used; no subjective listening evaluation is conducted to validate the perceptual quality of the semantic alignment improvements.

## Related Work & Insights

- **vs. OmniSep (Cheng et al., 2025)**: OmniSep provides a unified multimodal-query separation backbone but is trained with a weighted BCE loss. MARS-Sep retains the OmniSep architecture and layers RL training on top to inject semantic supervision.
- **vs. AudioSep (Liu et al., 2024)**: AudioSep achieves zero-shot separation using a CLAP encoder and 14k hours of training data, but training remains regression-based. MARS-Sep demonstrates that RL with semantic rewards can surpass purely supervised methods even with less training data.
- **vs. RLHF in LLMs**: This work represents an innovative application of the RLHF paradigm to audio generation and processing; the progressive reward model training strategy is potentially transferable to other cross-modal generative tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing RL preference alignment into sound separation is a novel cross-domain transfer; the Beta policy design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, four query modalities, and comprehensive baseline comparisons; however, human evaluation and computational cost analysis are absent.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly presented and the RLHF analogy is effective, though the notation is dense.
- Value: ⭐⭐⭐⭐ Introduces an RL alignment paradigm to audio processing, with potential to advance methodological development in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [\[AAAI 2026\] TextShield-R1: Reinforced Reasoning for Tampered Text Detection](../../AAAI2026/reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [\[ICLR 2026\] LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection](laddersym_a_multimodal_interleaved_transformer_for_music_practice_error_detectio.md)
- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)

</div>

<!-- RELATED:END -->
