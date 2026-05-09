---
title: >-
  [Paper Note] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models
description: >-
  [NeurIPS 2025][Segmentation][Mechanistic Interpretability] By training RNNs to emulate the emission statistics of HMMs, then reverse-engineering the learned solutions, this work reveals how RNNs exploit noise-driven orbital dynamics, structured connectivity (noise-integrating populations + kick neurons), and self-induced stochastic resonance to implement discrete stochastic state transitions.
tags:
  - NeurIPS 2025
  - Segmentation
  - Mechanistic Interpretability
  - Recurrent Neural Networks
  - Hidden Markov Models
  - Stochastic Resonance
  - Dynamical Systems
date: 2026-05-08
content_hash: ac09a638898ec298
---

# Mechanistic Interpretability of RNNs Emulating Hidden Markov Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.25674](https://arxiv.org/abs/2510.25674)
**Code**: [GitHub](https://github.com/EliaTorre/hmmrnn)
**Area**: Segmentation / Interpretability
**Keywords**: Mechanistic Interpretability, Recurrent Neural Networks, Hidden Markov Models, Stochastic Resonance, Dynamical Systems

## TL;DR

By training RNNs to emulate the emission statistics of HMMs, then reverse-engineering the learned solutions, this work reveals how RNNs exploit noise-driven orbital dynamics, structured connectivity (noise-integrating populations + kick neurons), and self-induced stochastic resonance to implement discrete stochastic state transitions.

## Background & Motivation

1. **Background**: RNNs are powerful tools in neuroscience for inferring latent dynamics of neural populations, but prior work has focused primarily on relatively simple, input-driven, deterministic behaviors. HMMs, by contrast, can segment naturalistic behavior into discrete latent states with stochastic transitions.

2. **Limitations of Prior Work**: RNNs operate in continuous state spaces, whereas HMMs rely on discrete states and stochastic transitions — an apparent incompatibility. It remains unclear whether, and how, RNNs can produce stochastic transitions between discrete states through continuous dynamics.

3. **Key Challenge**: How can a continuous state space give rise to discrete stochastic behavior? Intuitively, RNNs should learn one fixed point per HMM state (a multi-well landscape), yet the actual solution turns out to be considerably more subtle.

4. **Goal**: To determine how RNNs emulate the discrete probabilistic behavior of HMMs using continuous internal dynamics, and to uncover the underlying computational mechanism.

5. **Key Insight**: Develop a training methodology (noise-driven RNN + Sinkhorn divergence) that fits RNNs to HMM emission statistics, followed by multi-level reverse engineering: global dynamics → local dynamics → connectivity structure → computational principles.

6. **Core Idea**: RNNs implement stochastic state transitions via a self-induced stochastic resonance (SISR) mechanism — slow noise integration combined with fast kick triggering — realizing composable dynamical primitives that emulate HMM behavior.

## Method

### Overall Architecture

The training pipeline consists of three steps: (A) noise input $x_t \sim \mathcal{N}(0, I_d)$ → (B) Vanilla RNN + Gumbel-Softmax → (C) Sinkhorn divergence loss. Three HMM architectures are considered: linear chain, fully connected, and ring.

### Key Designs

**1. Noise-Driven RNN Training Paradigm**

- **Function**: Enables RNNs to learn the stochastic transition dynamics of HMMs.
- **Mechanism**: A standard Vanilla RNN ($h_t = \text{ReLU}(h_{t-1}W_{hh}^T + x_tW_{ih}^T)$) receives i.i.d. Gaussian inputs; outputs are converted to categorical samples via Gumbel-Softmax. Sinkhorn divergence (an optimal transport distance) serves as the loss function to compare output and target distributions.
- **Design Motivation**: HMM target sequences are probabilistic, necessitating a loss function suited to distributional comparison. Sinkhorn divergence enables differentiable optimization through smoothed coupling matrices.

**2. Multi-Level Reverse Engineering Analysis**

- **Function**: Reveals the complete mechanistic chain by which RNNs implement HMMs.
- **Mechanism**:
    - **Global dynamics**: Without input, the RNN converges to a single fixed point; under noise input, it exhibits orbital dynamics along closed trajectories, with orbital radius scaling linearly with input variance.
    - **Local dynamics**: The state space is partitioned into three functional regions — **clusters** (long dwell times, locally stable), **kick zones** (intermediate dwell times, with unstable directions), and **transition corridors** (rapid, deterministic passages).
    - **Connectivity structure**: Structured connectivity is identified comprising kick-neuron triplets and noise-integrating populations.
- **Design Motivation**: Standard fixed-point linearization methods cannot account for the rich dynamics observed under a single-fixed-point regime.

**3. Causal Intervention Validation**

- **Function**: Validates the causal role of the kick mechanism.
- **Mechanism**: Ablating kick neurons or their noise inputs ($\mu=0$) traps trajectories within the current cluster, preventing transitions; amplification ($\mu=2$) causes overshooting beyond the target cluster. Control experiments on non-noise-integrating neurons show no effect on inter-cluster switching, confirming causal specificity.
- **Design Motivation**: Beyond discovering the mechanism, it is essential to verify its causal sufficiency and necessity.

### Loss & Training

- Loss function: Sinkhorn divergence, comparing the distribution of RNN output sequence $Y$ against HMM target sequence $Y^*$
- Evaluation metrics: Euclidean distance (global reconstruction error), transition matrix, marginal frequencies, output volatility
- Hyperparameters: hidden size $|h| \in \{50, 150, 200\}$, input dimension $d \in \{1, 10, 100, 200\}$

## Key Experimental Results

### Main Results

| HMM Architecture | No. of States | Emission Statistics | Transition Matrix | Stationary Distribution |
|------------------|---------------|--------------------|--------------------|------------------------|
| Linear chain | 2–5 | ✓ Exact match | ✓ | ✓ |
| Fully connected | 3 | ✓ Exact match | ✓ | ✓ |
| Ring | 4 | ✓ Exact match | ✓ | ✓ |

**Characteristics of the Training Transition Phase**:

| Training Phase | Dynamical Characteristics | Unstable Eigenvalues | Loss Behavior |
|----------------|--------------------------|----------------------|---------------|
| Early | Single fixed point | None | Normal descent |
| **Transition** | **Unstable** | **Complex eigenvalues emerge** | **Double descent** |
| Stable | Orbital dynamics | Stable oscillations | Convergence |

### Ablation Study

| Intervention | μ=0 (Ablation) | μ=2 (Enhancement) |
|--------------|---------------|-------------------|
| Kick neurons | Trapped in current cluster | Overshoots target cluster |
| Noise-integrating → kick pathway | Trapped in current cluster | Overshoots target cluster |
| Control neurons | Normal inter-cluster switching | Normal inter-cluster switching |

### Key Findings

- Trained RNNs possess only a **single fixed point** (rather than $n$ wells); noise is a necessary condition for sustaining dynamics.
- Orbital radius scales **linearly** with input variance, explainable via second-order perturbation analysis.
- RNNs trained on different HMM architectures reuse the same **composable dynamical primitives** — multiple instances of the same basic mechanism combine to produce more complex discrete structures.
- The mechanism resembles self-induced stochastic resonance (SISR): a synergy of slow noise integration and fast kick-driven resetting.

## Highlights & Insights

- A concrete mechanism by which a continuous system implements discrete stochastic behavior is identified, bridging the RNN and HMM paradigms.
- The notion of "composable dynamical primitives" is highly illuminating — complex discrete structures emerge from modular combinations of simple elementary units.
- Methodologically, the multi-level reverse engineering paradigm (global → local → single-neuron) merits broader adoption.
- Noise functions not as interference but as a computational resource — consistent with theories of stochastic resonance facilitating signal processing in the brain.

## Limitations & Future Work

- The study uses Vanilla RNNs with simple 3-output HMMs; scalability to larger models and more complex HMM structures remains unvalidated.
- Only HMMs with output dimension 3 are examined; higher-dimensional or continuous-emission cases are unexplored.
- No direct connection is drawn to experimental data from biological neural circuits.
- It remains open whether the single-fixed-point + noise-driven orbital mechanism is the unique solution for RNN emulation of HMMs, or merely one among multiple possible solutions.

## Related Work & Insights

- This work extends the tradition of RNN reverse engineering (fixed-point analysis, low-rank connectivity) to internally driven probabilistic behavior.
- It resonates with Driscoll et al. (2024)'s notion of "shared dynamical motifs," demonstrating a homogenizing effect of the training environment.
- Future directions: applying this framework to understand the emergence of discrete states in Transformers or SSMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Uncovers the complete mechanism by which RNNs emulate HMMs via self-induced stochastic resonance — highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-level analysis is highly systematic, and causal interventions are compelling, though the experimental scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative logic, proceeding from global to local to connectivity to principle, is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ Makes an important theoretical contribution to computational neuroscience; the concept of composable dynamical primitives has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

<!-- RELATED:END -->
