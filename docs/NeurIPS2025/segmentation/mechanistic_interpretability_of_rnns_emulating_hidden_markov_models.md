---
title: >-
  [Paper Note] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models
description: >-
  [NeurIPS 2025][Segmentation][RNN Interpretability] A vanilla RNN is trained to reproduce the emission statistics of a Hidden Markov Model (HMM)…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "RNN Interpretability"
  - "Hidden Markov Models"
  - "Noise-Driven Dynamics"
  - "Stochastic Resonance"
  - "Compositional Dynamical Primitives"
date: 2026-05-08
content_hash: e8fdc7f22869cafa
---

# Mechanistic Interpretability of RNNs Emulating Hidden Markov Models

**Conference**: NeurIPS 2025  
**arXiv**: [2510.25674](https://arxiv.org/abs/2510.25674)  
**Code**: [GitHub](https://github.com/EliaTorre/hmmrnn)  
**Area**: Human Understanding  
**Keywords**: RNN Interpretability, Hidden Markov Models, Noise-Driven Dynamics, Stochastic Resonance, Compositional Dynamical Primitives

## TL;DR

A vanilla RNN is trained to reproduce the emission statistics of a Hidden Markov Model (HMM), and its internal mechanisms are reverse-engineered to reveal that the network implements discrete stochastic state transitions via noise-sustained orbital dynamics, "kick neuron" circuits, and self-induced stochastic resonance.

## Background & Motivation

**Background**: RNNs are widely used in computational neuroscience to model neural population dynamics and generate hypotheses about neural computation. Prior work has focused primarily on deterministic, input-driven tasks such as motion discrimination and reaching movements.

**Limitations of Prior Work**: Natural behavior often manifests as stochastic transitions between discrete latent states—as described by HMMs—whereas RNNs operate over continuous state spaces. These two frameworks appear fundamentally incompatible, and little is known about how RNNs can generate spontaneous, stochastic, discrete-like behavior.

**Key Challenge**: HMMs model behavior with discrete states and stochastic transitions, while RNNs model dynamics with continuous trajectories—bridging these two paradigms remains an open challenge.

**Goal**: Can an RNN implement discrete stochastic state transitions using continuous dynamics? If so, what are the underlying internal mechanisms?

**Key Insight**: Train an RNN directly to fit the output distribution of an HMM, then perform multi-level reverse engineering.

**Core Idea**: The RNN implements discrete states via noise-sustained closed-orbit dynamics; slow noise accumulation combined with fast kick-neuron triggering forms "dynamical primitives," and multiple primitives can be composed to emulate complex HMM structures.

## Method

### Overall Architecture

1. **Training Pipeline**: The RNN receives i.i.d. Gaussian noise inputs $x_t \sim \mathcal{N}(0, I_d)$, updates its hidden state via ReLU recurrent dynamics, projects linearly to a 3-dimensional output, and converts to categorical samples via Gumbel-Softmax. The network is trained by minimizing the Sinkhorn divergence (an optimal-transport distance) between generated and target sequences.
2. **Reverse Engineering**: Analysis proceeds across four levels: global dynamics → local dynamics → individual neurons and connections → computational principles.

### Key Designs

1. **Training Paradigm (Noise-Driven RNN + Sinkhorn Loss)**:

    - **Function**: Train the RNN to learn the probabilistic output behavior of an HMM.
    - **Design Motivation**: The target sequences are probabilistic; step-wise supervised losses are ill-suited, and a distribution-level comparison is required rather than sample-level matching.
    - **Mechanism**: Sinkhorn divergence—a regularized optimal-transport distance with a softened coupling matrix enabling differentiable optimization—is used:
    $h_t = \text{ReLU}(h_{t-1}W_{hh}^T + x_t W_{ih}^T), \quad y_t = h_t A^T$
      Parameters $\Theta = \{W_{hh}, W_{ih}, A\}$ are optimized by minimizing the Sinkhorn divergence between predicted outputs and HMM target sequences.
    - **Novelty**: Standard RNN training employs deterministic losses; this work is the first to introduce an optimal-transport loss for training RNNs to reproduce probabilistic behavior.

2. **HMM Architecture Family**:

    - **Linear-Chain HMM**: 2–5 states with banded diagonal transition matrices, systematically spanning the spectrum from maximally discrete to nearly continuous.
    - **Fully-Connected HMM**: 3 states, with transitions permitted between any pair of states.
    - **Cyclic HMM**: 4 states with a bidirectional closed-loop transition structure.

3. **Global Dynamics Finding — Noise-Sustained Orbital Dynamics**:

    - Without input, RNN activity converges to a single fixed point, precluding discrete state switching.
    - With stochastic input, trajectories traverse closed orbits exhibiting slow regions that correspond to distinct output classes.
    - Orbital radius scales linearly with input variance.
    - Training undergoes a marked phase transition: from stable fixed point → instability → emergence of orbital dynamics.

4. **Local Dynamics Finding — Three Functional Regions**:

    - **Clusters** (dwell time > 8 steps): Stable slow regions corresponding to distinct output probability distributions; noise-sensitive.
    - **Kick-zones** (dwell time 2–8 steps): Located downstream of clusters, possessing a small number of unstable directions; serve as triggers for state transitions.
    - **Transitions** (dwell time < 2 steps): Brief, fast-passage corridors that lead quasi-deterministically to the next cluster; noise-insensitive.

5. **"Kick Neuron" Circuit**:

    - Two triplets of kick neurons are identified whose pre-activation values are strongly negative within clusters, near zero in kick-zones, and positive during transitions.
    - Connectivity: within-group mutual excitation; between-group mutual inhibition.
    - A larger "noise-accumulation population" (~70 neurons) modulates the gating of kick neurons via structured connectivity.
    - Causal intervention validates the mechanism: ablating kick neurons ($\mu=0$) traps trajectories in clusters; amplifying them ($\mu=2$) causes trajectories to overshoot.

### Loss & Training

- Loss function: Sinkhorn divergence (regularized optimal-transport distance)
- Network: vanilla RNN with hidden state dimension $|h| \in \{50, 150, 200\}$ and input dimension $d \in \{1, 10, 100, 200\}$
- Gumbel-Softmax temperature $\tau = 1$
- Primary analysis focuses on the $|h|=150, d=100$ configuration

## Key Experimental Results

### Main Results

**RNN Reproduction of HMM Emission Statistics**:

| Metric | Description | Result |
|--------|-------------|--------|
| Sinkhorn Distance | Global reconstruction error | Converges to near zero |
| Transition Matrix Error | 3×3 emission transition matrix | Closely matches target HMM |
| Marginal Frequency | Stationary distribution | Precisely reproduced |
| Output Lability | Proportion of output changes | Consistent with target |

All four metrics are successfully validated across all HMM architectures (linear-chain / fully-connected / cyclic).

### Ablation Study

**Causal Intervention Experiments (Kick Neuron Validation)**:

| Intervention | Modulation Factor $\mu$ | Effect | Key Eigenvalue Change |
|--------------|------------------------|--------|----------------------|
| Ablate kick neurons | $\mu=0$ | Trajectories trapped in current cluster; no switching | Critical eigenvalue pair vanishes |
| Ablate noise-accumulation population | $\mu=0$ | Same as above; causal consistency confirmed | Same as above |
| Amplify kick neurons | $\mu=2$ | Trajectories overshoot beyond target cluster | Critical eigenvalue pair preserved |
| Control group (non-accumulation population) | $\mu=0$ | Cluster switching normal; kick neuron noise drive maintained | No significant change |

### Key Findings

- The RNN implements discrete state representations via a single fixed point and noise-driven closed orbits rather than the expected multi-stable attractor landscape.
- A marked phase transition (secondary loss drop) occurs during training, corresponding to the emergence of unstable eigenvalues and orbital dynamics.
- **Compositional dynamical primitives**: The same "slow noise accumulation + fast kick trigger" primitive can be composed to produce complex discrete latent structure (fully-connected and cyclic HMMs decompose into combinations of multiple linear-chain primitives).
- The mechanism constitutes self-induced stochastic resonance (SISR): without an external periodic signal, noise alone drives quasi-periodic oscillations.

## Highlights & Insights

- **Bridging Discrete and Continuous**: The work elegantly demonstrates how a continuous dynamical system can implement discrete stochastic transitions, closing the conceptual gap between RNNs and HMMs.
- **Multi-Level Reverse Engineering**: A complete causal chain from population dynamics to individual neuron connectivity sets a benchmark for mechanistic interpretability.
- **Compositional Primitive Paradigm**: A single dynamical primitive is modularly reused to construct complex structures, analogous to function reuse in programming.
- **Biological Implications**: Cortical circuits inherently exhibit intrinsic noise (stochastic ion channels, probabilistic synaptic transmission); the proposed mechanism resonates strongly with the biological phenomenon of stochastic resonance.
- **Training Methodology Innovation**: The use of Sinkhorn divergence to train probabilistic RNN behavior is generalizable to modeling other stochastic processes.

## Limitations & Future Work

- Only vanilla RNNs with ReLU activations are studied; it remains unclear whether more complex architectures such as GRUs or LSTMs develop the same mechanisms.
- The HMM scale is limited to 2–5 states; whether the mechanism holds for HMMs with tens or hundreds of states is an open question.
- Validation on real neural or behavioral data is absent—the current analysis is conducted exclusively on synthetic HMM outputs.
- The Gumbel-Softmax temperature is fixed at 1; the effect of varying this parameter on learning outcomes is unexplored.
- High-dimensional inputs facilitate convergence, but their physical interpretation within neural circuits remains unclear.

## Related Work & Insights

- **Fixed-Point Analysis of RNNs** (Sussillo, Barak et al.): Understanding RNN computation via linearization around fixed-point topology. The present work extends this framework to noise-driven, non-equilibrium analysis.
- **Low-Rank RNNs** (Mastrogiuseppe, Barak et al.): Low-rank connectivity can reveal the relationship between connectivity and dynamics; kick neurons in this work naturally form low-rank structured connections.
- **Multi-Task RNNs** (Driscoll et al.): RNNs reuse shared dynamical primitives across tasks, echoing the compositional primitive finding reported here.
- **Inspiration**: The overall framework—train to emulate → reverse engineer → causal intervention—is transferable to mechanistic interpretation of other computational models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneering training of RNNs to emulate HMMs and discovery of noise-driven orbital dynamics and compositional primitives
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-level analysis (global / local / single neuron / computational principles) with causal intervention validation
- Writing Quality: ⭐⭐⭐⭐⭐ Logically layered argumentation, visually clear figures, and cohesive narrative
- Value: ⭐⭐⭐⭐⭐ Profound implications for both computational neuroscience and explainable AI

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

</div>

<!-- RELATED:END -->
