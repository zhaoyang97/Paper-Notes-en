---
title: >-
  [Paper Note] Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator
description: >-
  [ICLR 2026][Reinforcement Learning][Hippocampus sequence generator] Inspired by the intrinsic recurrent circuitry of hippocampal region CA3, this paper proposes a minimal sequence generator (shift register) integrated with an actor-critic framework to achieve maze navigation under sparse visual input, while giving rise to neurobiologically observed phenomena including place fields, DG orthogonalization, distance-dependent spatial kernels, and task-dependent remapping.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Hippocampus sequence generator
  - spatial representation
  - Actor-Critic
  - sparse coding
  - place cells
date: 2026-05-08
content_hash: 1ced1875af02c9cf
---

# Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator

**Conference**: ICLR 2026
**arXiv**: [2510.09951](https://arxiv.org/abs/2510.09951)
**Code**: [Available](https://github.com/xiaoxionglin/SF_hipposlam)
**Area**: Reinforcement Learning
**Keywords**: Hippocampus sequence generator, spatial representation, Actor-Critic, sparse coding, place cells

## TL;DR
Inspired by the intrinsic recurrent circuitry of hippocampal region CA3, this paper proposes a minimal sequence generator (shift register) integrated with an actor-critic framework to achieve maze navigation under sparse visual input, while giving rise to neurobiologically observed phenomena including place fields, DG orthogonalization, distance-dependent spatial kernels, and task-dependent remapping.

## Background & Motivation
- Hippocampal place cells fire in an ordered manner as theta sequences; the prevailing view attributes this to sensory input driven along the trajectory
- The authors propose a more parsimonious explanation: sequences originate from **intrinsic recurrent circuits in CA3**, which can propagate activity over extended periods without external input, functioning as a temporal memory buffer
- This intrinsic sequence generation mechanism is particularly critical when reliable sensory evidence is sparse (e.g., only a few landmarks during navigation)
- The dentate gyrus (DG) has only approximately 2–5% of granule cells active in any given environment, providing sparse coding at extremely low activity rates
- This mechanism resonates with SSM/structured linear RNN ideas in machine learning: expanding input into a high-dimensional temporal feature space and then compressing via a shallow nonlinear readout
- Existing successor representation, reservoir, and probabilistic approaches can reproduce place-field activity but rarely explicitly address the origin of the sequences

## Method

### Overall Architecture
The system comprises four modules:
1. **Visual encoder**: A pretrained ResNet (3 convolutional blocks, matching the IMPALA architecture) that extracts features from a first-person perspective; weights are frozen during training
2. **Dentate gyrus (DG) sparsification module**: Linearly projects ResNet features to $F=16$ features, followed by batch normalization (retention rate 0.95) and high-threshold ($\tau=2.43$) filtering to maintain approximately 2.5% activation rate
3. **CA3 sequence generator (shift register)**: A fixed-weight linear RNN where each DG feature corresponds to a dedicated prewired sequence of length $\ell = L + (R-1)$
4. **Actor-Critic decoder**: Two-layer FC+ReLU with linear readout of action probabilities and value estimates

### CA3 Shift Register Dynamics

For a single feature, the CA3 state update is given by $x_{t+1} = S x_t + J u_t$, where $S \in \mathbb{R}^{\ell \times \ell}$ is the shift operator (a matrix with ones on the subdiagonal of the lower triangle) and $J \in \mathbb{R}^{\ell \times 1}$ injects input into the first $R$ slots. Instantaneous input creates activity in the first $R$ positions, which then propagates through the register of length $\ell$ at one position per step.

Multiple features evolve independently and are extended into a block-diagonal update structure via Kronecker products $A = I_F \otimes S$ and $B = I_F \otimes J$, yielding the full CA3 state $X_t \in \mathbb{R}^{F\ell}$.

### Design Philosophy
- CA3 weights are **entirely fixed** (not trained), isolating the pure effect of sequence generation — a critical control variable
- **Only** the DG-to-CA3 input mapping and the CA3-to-decoder output weights are trained
- Parameter $L$ controls the number of theta cycles spanned by the sequence; $R$ controls the number of simultaneously active units per cycle and also serves as a temporal smoothing prior
- Sparsification is not incidental but central: it filters out noisy, non-informative cues and allows suprathreshold inputs to reliably localize to spatial regions

### Environment & Training
- **Environment**: A 19×19 DeepMind Lab continuous maze with walls randomly covering 15% of cells and uniform visual textures (spatial relationships cannot be inferred from visual similarity), supporting multiple paths to the goal
- Standard advantage actor-critic objective: policy gradient + value baseline + entropy regularization
- Distributed training using the Sample Factory framework (IMPALA architecture)
- Maximum 900 steps per episode; agent is randomly placed at least 5 units from the goal; 6 random seeds

## Key Experimental Results

### Architecture Comparison

| Input Type | CA3 (L=64, R=8) | Random RNN | HiPPO-LegS | LSTM |
|-----------|-----------------|------------|-------------|------|
| Sparse (steps to 80%) | **173.6±77.6M** | ✗ | ✗ | ✗ |
| Sparse (final success rate) | **0.86±0.10** | 0.51±0.12 | 0.52±0.11 | 0.56±0.06 |
| Dense (steps to 80%) | ✗ | ✗ | ✗ | **135.9±27.6M** |
| Dense (final success rate) | 0.71±0.07 | 0.78±0.15 | 0.64±0.21 | **0.93±0.09** |

Key finding: Under sparse input, CA3 is the **only** architecture capable of reaching an 80% success rate; under dense input, LSTM performs better — revealing a strong interaction effect between representational sparsity and memory architecture.

### Transfer Learning

| Transfer Scenario | Training Frames Required |
|------------------|--------------------------|
| New reward location | ~50M (existing map representation reusable) |
| New map | ~150M |
| Path blocked | Rapid adaptation |

This indicates that the agent acquires a generalizable representation of spatial layout rather than merely memorizing specific paths.

### Ablation Study
- **Sequence length**: $L=1, R=1$ (purely feedforward, bypassing CA3) fails completely; $L=16, R=8$ achieves some success rate but is unstable; $L=64, R=8$ is optimal
- **Causal role of spatial information**: Permuting the 32 decoder units with highest SI reduces success rate by 4.9% and increases trajectory length from 1065 to 2794 frames; permuting the lowest SI units has no effect
- **Noise robustness**: Adding pixel-level Gaussian noise to suppress weak signals affects DG+CA3 far less than LSTM
- **Parameter $R$**: Performance is stable over a broad range of $R$ values; sensitivity to $R$ increases at lower movement speeds

### Behavioral & Representational Analysis
- **Occupancy map evolution**: The agent gradually develops stable trajectories, tending to visit salient input/landmark locations before converging to the goal — analogous to habitual navigation strategies in familiar environments
- **Place field emergence**: DG and CA3 units naturally develop localized place fields; LSTM hidden units do not exhibit place-cell characteristics
- **DG orthogonalization**: During learning, correlations between DG population activity maps across locations progressively decrease, forming unique encodings for each position
- **Spatial broadening within sequences**: CA3 units farther from the DG input within a sequence exhibit broader spatial tuning (higher entropy), consistent with experimental observations
- **Distance-dependent spatial kernels**: Population vector correlations across all layers exhibit smooth distance dependence; the CA3 kernel is smoother than the DG kernel; Layer 1 of the decoder shows the most prominent spatial tuning; LSTM exhibits only a weak, non-isotropic spatial kernel at the output layer
- **Task-dependent remapping**: Place field centroids shift following reward location changes, indicating representational remapping; similarity between the trained and new-reward conditions is higher than between the initial and trained conditions, demonstrating generalizability of spatial layout knowledge

## Highlights & Insights
- A **minimal yet effective** biologically inspired design: fixed-weight shift registers require no learning of recurrent matrices, yielding a transparent and interpretable structure
- Clearly reveals a principle that different recurrent architectures are suited to different sensory regimes: sparse coding + sequence expansion vs. dense input + mixed recurrence (LSTM)
- The CA3 module expands sparse DG encodings into a temporally smoothed canonical basis set, providing long-horizon history without indiscriminately mixing features as in fully connected RNNs
- Behavioral patterns developed by the agent (habitual trajectories, landmark-oriented convergence) are consistent with animal navigation strategies; the LSTM agent's behavior is more akin to visual search
- A middle ground on the Bitter Lesson: structural priors (sparsity, sequences) do not prescribe representational schemes but rather constrain the hypothesis space without sacrificing scalability

## Biological Predictions
- Larger environments or sparser inputs require longer sequences for successful navigation
- Hippocampal spatial representations may rely primarily on intrinsic sequence generation circuits, with experience mainly shaping feedforward and readout connections
- Provides an explanation for the persistence of place cells following entorhinal cortex lesions
- The mechanism generalizes to species without prominent theta oscillations (e.g., hippocampal sequences locked to wingbeat cycles in bats)

## Limitations & Future Work
- CA3 weights are entirely fixed without incorporating local plasticity rules; introducing plasticity in the DG–CA3 pathway may better approximate biological reality
- Evaluated only in a single maze environment (DeepMind Lab); validation in more complex 3D environments and multi-task settings remains to be done
- Path integration and interactions with entorhinal grid cells are not modeled
- Training requires approximately 350M frames; sample efficiency warrants improvement
- Hierarchical cross-region theta sequence coordination is not explored

## Related Work & Insights
- **Successor Representation (SR)**: CA3 activity shares structural similarities with SR (policy-dependent, temporally ordered, prospective), but the predictive structure of CA3 arises from fixed topology rather than TD learning
- **HiPPO/S4/SSM**: The CA3 shift register generates a finite-length temporal basis, contrasting with the rotational modes of Legendre SSMs and the decaying modes of Laguerre SSMs; resonates with shift-diagonal architectures but is tailored to sparse sensory input
- **Reservoir Computing**: CA3 is essentially a biologically constrained reservoir network, but the shift structure endows it with interpretable sequential semantics
- Complements prior hippocampal RL studies that use allocentric inputs: this work demonstrates the emergence of Gaussian-like place fields from egocentric observations

## Rating
- Novelty: 4/5 (minimal biologically inspired design; the sparse–sequence synergy hypothesis is novel)
- Experimental Thoroughness: 5/5 (multi-dimensional validation across behavior, place fields, spatial information, population kernels, causal intervention, and noise robustness)
- Writing Quality: 4/5 (clear structure, accurate interdisciplinary exposition, balanced neuroscience and RL perspectives)
- Value: 4/5 (provides a unified mechanistic account of hippocampal theta sequences and RL-based navigation; opens a new direction for biologically inspired sparse architectures)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](../../AAAI2026/reinforcement_learning/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)

<!-- RELATED:END -->
