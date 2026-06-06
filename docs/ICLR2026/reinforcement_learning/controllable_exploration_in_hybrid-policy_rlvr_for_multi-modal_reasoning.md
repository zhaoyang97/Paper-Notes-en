---
title: >-
  [Paper Note] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] CalibRL reframes expert data as a distribution calibration baseline (rather than a strict imitation target)…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "hybrid-policy optimization"
  - "multimodal reasoning"
  - "entropy collapse"
  - "controllable exploration"
date: 2026-05-08
content_hash: b7a841bc9bce4748
---

# Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning

**Conference**: ICLR 2026
**arXiv**: [2602.20197](https://arxiv.org/abs/2602.20197)  
**Code**: [https://github.com/zhh6425/CalibRL](https://github.com/zhh6425/CalibRL)  
**Area**: Multimodal VLM / Reinforcement Learning
**Keywords**: RLVR, hybrid-policy optimization, multimodal reasoning, entropy collapse, controllable exploration

## TL;DR
CalibRL reframes expert data as a distribution calibration baseline (rather than a strict imitation target), and achieves fine-grained control over the exploration–exploitation trade-off in MLLM reasoning training via asymmetric LeakyReLU activation combined with advantage weighting. This addresses entropy collapse in RLVR and substantially outperforms GRPO/DAPO on tasks such as geometric reasoning.

## Background & Motivation

**Background**: RLVR has become the dominant paradigm for enhancing MLLM reasoning capabilities (e.g., DeepSeek-R1), yet performance gains are often accompanied by a significant drop in policy entropy—entropy exhaustion has become a bottleneck for further improvement.

**Limitations of Prior Work**:
   - Conventional entropy regularization encourages randomness but provides no directional guidance → low exploration efficiency
   - In the SFT-then-RL paradigm, SFT anchors the policy to a static demonstration distribution → undermines subsequent RL exploration
   - Directly injecting SFT supervision into hybrid-policy frameworks → distributional mismatch between the current policy and expert trajectories → high bias-variance → accelerated entropy collapse

**Key Challenge**: Expert data is a double-edged sword—it provides useful guidance but also compresses the policy distribution. Maximizing $\pi_\theta(\tau^{expert})$ inevitably reduces the probability of other trajectories → overall entropy decreases.

**Key Insight**: Expert data should not be "imitated" as an absolute target, but rather used as a reference baseline for relative calibration—under-represented correct reasoning paths are reinforced, while overconfident incorrect predictions are suppressed.

**Core Idea**: Transform expert supervision from a rigid imitation signal into a fine-grained calibration mechanism, achieving directed and regulated exploration via a log-probability gap combined with asymmetric LeakyReLU gating.

## Method

### Overall Architecture
An controllable exploration loss term $\mathcal{L}_{exploration}$ is introduced on top of GRPO. The key innovation lies in using the log-probability gap $\Delta\ell_i = \log\pi_\theta(\tau_i^{policy}) - \log\pi_\theta(\tau_i^{expert})$ to measure the model's relative preference for its own responses versus expert responses, with asymmetric LeakyReLU activation controlling the magnitude of reinforcement and suppression.

### Key Designs

1. **Log-Probability Gap**:

    - Function: Measures the model's relative confidence in its own responses versus expert responses.
    - Mechanism: $\Delta\ell_i = \log\frac{\pi_\theta(\tau_i^{policy})}{\pi_\theta(\tau_i^{expert})}$. A positive value indicates the model favors its own answer; a negative value indicates lower confidence relative to the expert. This signal determines whether reinforcement or suppression is applied.

2. **Asymmetric LeakyReLU Activation**:

    - Function: Asymmetrically controls the gradient for reinforcement and suppression.
    - Mechanism: $\mathcal{L}_{exploration} = |\hat{A}_i| \cdot \text{LeakyReLU}(-s_i \cdot \Delta\ell_i, \alpha)$, where $s_i = +1$ (correct) or $-1$ (incorrect). When the input is negative, the gradient is scaled by $\alpha < 1$ → once the response probability crosses the expert baseline, the magnitude of further reinforcement or suppression is attenuated.
    - Design Motivation: Pure ReLU completely cuts off gradients → useful signals are discarded; linear activation → cannot distinguish regions that do or do not require reinforcement; LeakyReLU strikes a balance between the two.

3. **Advantage Weighting**:

    - Function: Scales updates by group-wise rarity.
    - Mechanism: $|\hat{A}_i|$ (absolute advantage value) serves as the weight. A rare correct response when most responses are incorrect receives a large weight → reinforced as an exploration signal. By modulating the update magnitude, rare but informative deviations are emphasized.

### Loss & Training
- Total objective = GRPO clipped surrogate + $\lambda \cdot \mathcal{L}_{exploration}$
- Key hyperparameters: $\alpha=0.5$ (LeakyReLU slope), $\lambda=0.1$ (exploration weight)
- Expert baseline > reference policy baseline (confirmed by ablation)

## Key Experimental Results

### Main Results (Geometric Reasoning, In-Domain Benchmarks)

| Method | GeoEval↑ | Geo3K↑ | GeoQA↑ | Avg.↑ |
|--------|---------|--------|--------|-------|
| GRPO | 26.15 | 39.77 | 52.52 | 39.48 |
| SFT+GRPO | 6.00 | 18.64 | 40.98 | 21.87 |
| DAPO | 25.19 | 40.93 | 52.52 | 39.55 |
| **CalibRL** | **33.44** | **40.60** | **60.74** | **44.93** |

### Ablation Study

| LeakyReLU $\alpha$ | Effect |
|-------------------|--------|
| 0.3 | Aggressive early exploration but unstable, entropy oscillation |
| **0.5** | **Balanced entropy growth, no oscillation** |
| 0.8 | Over-constrained, rapid entropy decay |

### Key Findings
- **SFT+GRPO performs worst**: Directly combining SFT and RL leads to severe entropy collapse—supporting the necessity of the "imitation → calibration" paradigm shift.
- **CalibRL achieves best performance both in-domain and out-of-domain**: Outperforms GRPO by an average of +5.45% (in-domain), with better generalization as well.
- **Stable entropy curve**: Policy entropy grows steadily throughout CalibRL training, whereas other methods exhibit continuous decline.
- **$\alpha=0.5$ is the sweet spot**: Too small → instability; too large → exploration is constrained.

## Highlights & Insights
- **"Calibration over imitation" represents a profound reconceptualization of hybrid-policy RL**—expert data should not be treated as a mandatory target to reach, but rather as a reference coordinate for measuring the current policy's deviation.
- **LeakyReLU provides an elegant gradient gating mechanism**—a simple choice of activation function realizes adaptive control that reinforces when necessary and attenuates when sufficient.
- **The finding that "SFT+GRPO performs worst" carries important practical implications**—naively stacking SFT and RL may cause them to cancel each other out.

## Limitations & Future Work
- Validation is limited to geometric reasoning—scenarios such as mathematical and code reasoning remain to be explored.
- The LeakyReLU slope $\alpha$ requires tuning—a more adaptive activation function design may be preferable.
- The quality of expert data affects the reliability of the calibration baseline.
- Integration with other RL variants (Dr.GRPO, CPPO) has not been explored.

## Related Work & Insights
- **vs. GRPO/DAPO**: Standard policy optimization does not address entropy collapse; CalibRL maintains exploration through the calibration mechanism.
- **vs. LUFFY**: Also a hybrid-policy framework but still relies on imitation-style supervision → distributional mismatch persists.
- **vs. SFT+GRPO**: Direct sequential combination leads to catastrophic interference; CalibRL's calibration paradigm avoids this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ The "calibration over imitation" paradigm is insightful, and the application of LeakyReLU is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are thorough, but the task scope is narrow (primarily geometric reasoning).
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is in-depth, and theoretical motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ A practical solution to entropy collapse in RLVR, with broader implications for hybrid-policy training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](../../ACL2026/reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](../../ACL2026/reinforcement_learning/healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Learning from Synthetic Data Improves Multi-hop Reasoning](learning_from_synthetic_data_improves_multi-hop_reasoning.md)

</div>

<!-- RELATED:END -->
