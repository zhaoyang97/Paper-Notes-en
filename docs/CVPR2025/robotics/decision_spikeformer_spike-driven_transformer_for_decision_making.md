---
title: >-
  [Paper Note] Decision SpikeFormer: Spike-Driven Transformer for Decision Making
description: >-
  [CVPR 2025][Robotics][Spiking Neural Networks] This work proposes DSFormer, the first spike-driven Transformer for offline reinforcement learning. It designs Temporal Spike Self-Attention (TSSA) and Position Spike Self-Attention (PSSA) to capture temporal/positional dependencies in RL, and introduces Progressive Threshold-dependent Batch Normalization (PTBN) to resolve the conflict between normalization and spiking properties. DSFormer outperforms ANN counterparts on the D4RL…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Spiking Neural Networks"
  - "Offline Reinforcement Learning"
  - "Spike-driven Transformer"
  - "Energy-efficient AI"
  - "Sequential Decision Making"
date: 2026-05-08
content_hash: f93d4c9111b280e5
---

# Decision SpikeFormer: Spike-Driven Transformer for Decision Making

**Conference**: CVPR 2025  
**arXiv**: [2504.03800](https://arxiv.org/abs/2504.03800)  
**Code**: Provided on the project homepage (see paper)  
**Area**: Reinforcement Learning / Spiking Neural Networks  
**Keywords**: Spiking Neural Networks, Offline Reinforcement Learning, Spike-driven Transformer, Energy-efficient AI, Sequential Decision Making

## TL;DR
This work proposes DSFormer, the first spike-driven Transformer for offline reinforcement learning. It designs Temporal Spike Self-Attention (TSSA) and Position Spike Self-Attention (PSSA) to capture temporal/positional dependencies in RL, and introduces Progressive Threshold-dependent Batch Normalization (PTBN) to resolve the conflict between normalization and spiking properties. DSFormer outperforms ANN counterparts on the D4RL benchmark while saving 78.4% of energy consumption.

## Background & Motivation

**Background**: Offline reinforcement learning models policy learning as a sequence prediction task through conditional sequence modeling (CSM), with Decision Transformer (DT) being a representative work. However, ANN-based Transformers face high energy consumption challenges in energy-constrained embodied AI scenarios. Spiking Neural Networks (SNNs) emerge as an alternative due to their event-driven, low-power characteristics, but existing SNN Transformers are primarily designed for visual classification tasks.

**Limitations of Prior Work**: (1) Existing SNN Transformer attention mechanisms are designed for spatial dimensions (images) and are unsuitable for modeling temporal dependencies in RL; (2) SNNs require BatchNorm (which can be fused with linear layers to achieve pure spike inference), but BatchNorm disrupts temporal dependencies within the sequence. Conversely, LayerNorm preserves temporal dependencies but introduces floating-point operations, violating spiking characteristics.

**Key Challenge**: The fundamental conflict between the discrete spike processing of SNNs and the precise continuous-value estimation required in RL, alongside the dilemma of normalization layers in balancing "preserving temporal dependency" and "maintaining spiking characteristics."

**Goal**: How to design an SNN self-attention mechanism suitable for sequence modeling in offline RL, and how to maintain the pure spiking inference capability of SNNs while preserving faithful temporal dependencies.

**Key Insight**: Incorporating the temporal dimension of SNNs into attention computation—concatenating inputs along the temporal dimension before computing attention to capture global temporal dependencies, and using positional bias to model local Markovian dependencies. The normalization dilemma is addressed using a progressive transition scheme.

**Core Idea**: Utilizing temporal concatenated attention + positional bias-based attention + progressive normalization to enable SNN Transformers to perform sequential decision-making effectively while maintaining low energy consumption.

## Method

### Overall Architecture
DSFormer follows the DT architecture. The input sequence $I_l = (a_{l-N}, \hat{R}_{l-N+1}, s_{l-N+1}, ..., a_{l-1}, \hat{R}_l, s_l)$ is passed through embedding layers, repeated T times along the time dimension (SNN time-steps), and fed into M stacked Decoder Blocks. Each block contains a spiking self-attention layer and a spiking MLP layer, ultimately outputting the next action via a prediction head.

### Key Designs

1. **Temporal Spike Self-Attention (TSSA)**:

    - **Function**: Captures global temporal dependencies across SNN time-steps, enhancing long-sequence credit assignment.
    - **Mechanism**: Instead of performing self-attention independently at each time-step like traditional SSSA does, TSSA concatenates the inputs of all time-steps along the temporal dimension before computing attention. The Q, K, and V matrices are derived from this temporal concatenation, integrated with a causal mask to prevent future information leakage. It is proven from an information theory perspective that the joint entropy of the concatenated representations satisfies $H(X^1,...,X^T) < \sum_t H(X^t)$ because LIF dynamics make adjacent time-steps non-independent. Concatenation enables more effective pattern learning. The time complexity is $O(TDN^2)$, which is identical to SSSA.
    - **Design Motivation**: The core characteristic of SNNs lies in the membrane potential dynamics across time-steps. Step-by-step self-attention loses these cross-time correlations.

2. **Position Spike Self-Attention (PSSA)**:

    - **Function**: Captures local position dependencies with linear complexity.
    - **Mechanism**: Introduces a learnable pairwise position bias matrix $P \in \mathbb{R}^{N \times N}$. The computation is formulated as $\text{Attn}(Q,K,V)_i = Q_i \odot \sum_j P_{ij} \odot K_j \odot V_j$, replacing matrix multiplication with element-wise multiplication. The position bias defines a local window $S$, keeping only positional relations where $|i-j| < S$. The time complexity is reduced to $O(TDN)$, which is linear with respect to the number of tokens.
    - **Design Motivation**: The Markovian property of RL trajectories makes local dependencies (adjacent state-action pairs) most critical, while global attention wastes computation. Element-wise operations are fully compatible with the additive nature of SNNs.

3. **Progressive Threshold-dependent Batch Normalization (PTBN)**:

    - **Function**: Preserves temporal dependencies during training and maintains pure spiking computation during inference.
    - **Mechanism**: Linearly combines tdLN (normalization along the channel dimension, preserving temporal dependencies) and tdBN (normalization along the batch dimension, which can be fused with linear layers) via a weight $\theta$: $\text{PTBN}(x) = \theta \cdot \text{tdLN}(x) + (1-\theta) \cdot \text{tdBN}(x)$. $\theta$ decays linearly from 1 to 0. In early training, tdLN is deployed to establish temporal dependencies, gradually transitioning to tdBN later for spike-driven inference. During inference, it degrades to pure tdBN, which can be merged with linear layers to eliminate floating-point operations.
    - **Design Motivation**: Directly using BatchNorm disrupts sequential dependencies, leading to poor performance, while using LayerNorm prohibits pure spiking inference. The progressive transition leverages the best of both worlds.

### Loss & Training
Following the Decision Transformer framework, an MSE loss is used to supervise action prediction. During training, $\theta$ decays linearly from 1 to 0, where a portion of training steps is allocated for the PTBN transition, and the rest is used for pure tdBN fine-tuning.

## Key Experimental Results

### Main Results

**MuJoCo Tasks (D4RL)**:

| Task | DT (ANN) | FCNet (ANN) | PSSA (SNN) | Energy |
|------|----------|-------------|------------|------|
| halfcheetah-m-e | 86.8 | 91.2 | **91.5** | - |
| walker2d-m-e | 108.1 | 108.8 | **108.9** | - |
| hopper-m-r | 82.7 | 65.3 | **96.3** | - |
| Average | 74.7 | 72.8 | **78.8** | **88.8μJ** |
| DT Energy | - | - | - | 410.5μJ |

**Adroit Tasks**:

| Task | DT | PSSA |
|------|-----|------|
| pen-e | 110.4 | **122.0** |
| relocate-e | 15.3 | **108.4** |
| Average | 27.8 | **48.6** (+74%) |

### Ablation Study

| Attention Type | MuJoCo Average | Time Complexity |
|-----------|------------|----------|
| SSSA | 73.8 | $O(TDN^2)$ |
| TSSA | 75.7 | $O(TDN^2)$ |
| **PSSA** | **78.8** | $O(TDN)$ |

| Normalization Method | Average Score |
|-----------|--------|
| tdBN | 62.3 |
| tdLN | 70.1 |
| **PTBN** | **78.8** |

### Key Findings
- PSSA achieves the best performance among all SNN variants and outperforms the ANN-based DT (78.8 vs 74.7) while saving 78.4% of energy.
- On the Adroit manipulation tasks, the improvement of SNN over DT is even more significant (+74%), indicating the great potential of spiking models in fine control tasks.
- PTBN is critical for performance; directly using tdBN leads to severe performance degradation (62.3 vs 78.8), demonstrating the supreme importance of modeling intra-sequence temporal dependencies.
- In long-sequence experiments (AntMaze), TSSA consistently outperforms DT at sequence lengths of 100-200, verifying the long-range modeling capability of temporal concatenated attention.

## Highlights & Insights
- **A Rare Case of SNN Outperforming ANN**: In offline RL, SNN not only matches but surpasses the ANN-based DT while saving 78% of energy. This challenges the stereotype that "SNN performance is inevitably inferior to ANN."
- **The Progressive Transition Idea of PTBN**: Smoothly transitioning from LayerNorm to BatchNorm is a versatile engineering technique that can be applied to any scenario requiring trade-offs between training flexibility and inference efficiency.
- **Positional Bias Replacing Attention Matrix Multiplication**: PSSA replaces traditional $QK$ matrix multiplication with element-wise multiplication + positional bias, reducing complexity from quadratic to linear, and remaining completely compatible with the additive property of SNNs.

## Limitations & Future Work
- Validated only on D4RL benchmarks (MuJoCo/Adroit/AntMaze); more complex visual RL tasks have not been explored.
- The SNN energy consumption estimation is based on theoretical operations rather than real neuromorphic hardware measurements.
- The DT framework itself has recognized limitations in stitching capabilities (unable to stitch optimal trajectories from sub-optimal data), and DSFormer inherits this limitation.
- The selection of the hyperparameter $T_p$ in PTBN lacks theoretical guidance.

## Related Work & Insights
- **vs Decision Transformer**: DSFormer replaces ANN with SNN in the DT framework, achieving a MuJoCo average of 78.8 vs 74.7, while saving 78.4% of energy.
- **vs SpikeGPT/SpikeBERT**: Existing SNN sequence models perform very poorly on RL tasks (SpikeGPT average is only 23.2), indicating that SNN designs designed for NLP cannot be directly transferred to RL.
- **vs Spikformer**: Spikformer's vision-oriented SSA lacks causal masking and temporal modeling. DSFormer's TSSA/PSSA are specifically designed for sequential decision-making.
- Holds significant reference value for embodied AI applications on low-power edge devices.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first SNN offline RL model, with all three designs (TSSA/PSSA/PTBN) being highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation on D4RL and detailed ablation studies, though lacking physical hardware energy measurements.
- Writing Quality: ⭐⭐⭐⭐ Well-described methodology with rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐ Opens up new directions for the application of SNNs in RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](../../NeurIPS2025/robotics/spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[ICML 2026\] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](../../ICML2026/robotics/psg-nav_probabilistic_scene_graph_navigation_via_multiverse_decision_making.md)
- [\[ICML 2025\] Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](../../ICML2025/robotics/beyond_cvar_leveraging_static_spectral_risk_measures_for_enhanced_decision-makin.md)
- [\[ICML 2026\] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation](../../ICML2026/robotics/dive_into_the_scene_breaking_the_perceptual_bottleneck_in_vision-language_decisi.md)

</div>

<!-- RELATED:END -->
