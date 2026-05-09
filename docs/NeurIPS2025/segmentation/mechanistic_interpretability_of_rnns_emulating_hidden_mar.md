---
title: >-
  [Paper Note] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models
description: >-
  [NeurIPS 2025][Segmentation][RNN] A vanilla RNN is trained to reproduce the emission statistics of an HMM; reverse engineering then reveals the mechanism by which the RNN implements discrete stochastic state transitions: noise-driven orbital dynamics combined with rapid transitions triggered by "kick neurons." The underlying principle is self-induced stochastic resonance (SISR), and this dynamical motif can be composed and reused to emulate more complex discrete latent structures.
tags:
  - NeurIPS 2025
  - Segmentation
  - RNN
  - HMM
  - mechanistic interpretability
  - stochastic resonance
  - kick neurons
  - orbital dynamics
date: 2026-05-08
content_hash: 7e60949be2ddea2e
---

# Mechanistic Interpretability of RNNs Emulating Hidden Markov Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.25674](https://arxiv.org/abs/2510.25674)
**Authors**: Elia Torre, Michele Viscione, Lucas Pompe, Benjamin F. Grewe, Valerio Mante (ETH Zurich / University of Zurich)
**Code**: [https://github.com/EliaTorre/hmmrnn](https://github.com/EliaTorre/hmmrnn)
**Area**: Image Segmentation
**Keywords**: RNN, HMM, mechanistic interpretability, stochastic resonance, kick neurons, orbital dynamics

## TL;DR

A vanilla RNN is trained to reproduce the emission statistics of an HMM; reverse engineering then reveals the mechanism by which the RNN implements discrete stochastic state transitions: noise-driven orbital dynamics combined with rapid transitions triggered by "kick neurons." The underlying principle is self-induced stochastic resonance (SISR), and this dynamical motif can be composed and reused to emulate more complex discrete latent structures.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **State of the Field**: Recurrent neural networks (RNNs) are widely used in computational neuroscience to infer latent dynamics of neural populations and to generate computational hypotheses about behavior. However, prior work has focused predominantly on relatively simple, input-driven, largely deterministic tasks — far less is known about how RNNs implement the richer, spontaneous, and potentially stochastic behaviors observed in natural environments.

Hidden Markov Models (HMMs) can segment natural behavior into discrete latent states with stochastic transitions between them. Such discrete-stochastic dynamics appear fundamentally at odds with the continuous state space of RNNs. The core question is: **Can RNNs exploit continuous dynamics to generate stochastic transitions between discrete states, and if so, by what mechanism?**

Existing reverse-engineering approaches for RNNs rely primarily on fixed-point topology analysis and local linearization, which are well-suited to deterministic tasks (e.g., motion discrimination, sentiment classification) but are not applicable to noise-driven stochastic behavioral regimes. This paper fills that gap.

## Method

### 1. Training Framework: Approximating an HMM with an RNN

**Network Architecture**: A standard vanilla RNN with hidden state dimensionality $|h| \in \{50, 150, 200\}$. At each time step the network receives Gaussian noise input $x_t \sim \mathcal{N}(0, I_d)$, $d \in \{1, 10, 100, 200\}$.

$$h_t = \text{ReLU}(h_{t-1} W_{hh}^\top + x_t W_{ih}^\top), \quad y_t = h_t A^\top$$

Output logits are converted to categorical samples via Gumbel-Softmax, simulating the discrete emissions of an HMM.

**Loss & Training**: The Sinkhorn divergence — an optimal-transport-based distance measure — is used to compare the distribution of RNN output sequences against the target HMM sequences. Normalized coupling matrices make the comparison differentiable and computationally efficient.

**HMM Family**: Three classes of target HMMs are designed to test generalization:
- **Linear-chain HMMs** ($M \in \{2,3,4,5\}$ states): spanning a spectrum from maximally discrete to near-continuous
- **Fully connected HMMs** (3 states): transitions permitted between any pair of states
- **Cyclic HMMs** (4 states): bidirectional closed-loop transitions

**Performance Metrics**: (i) Sinkhorn-aligned Euclidean distance; (ii) emission transition matrices; (iii) marginal observation frequencies; (iv) observation volatility. After training, the RNN reproduces the emission statistics of the target HMM across all metrics.

### 2. Global Latent Dynamics: Noise-Sustained Orbital Dynamics

Projecting RNN hidden states onto the first two principal components (PCA) reveals a key finding:

- **Without input**: starting from random initializations, activity converges to a **single fixed point** — no multi-attractor structure is present.
- **With noise input**: trajectories shift to **orbital dynamics** — unidirectional evolution along a closed orbit.

Noise pushes activity away from the fixed point while recurrent connectivity pulls it back, together producing a stable closed orbit. Along the orbit, the RNN exhibits slow regions (clusters), each corresponding to one HMM output class, with rapid transitions between them.

**Orbital radius scales linearly with noise variance**: A second-order perturbation analysis shows that under unbiased Gaussian inputs the first-order perturbation averages out, and the second-order term (linear in variance) dominates the post-transition dynamics.

**Emergence of orbital dynamics during training**: Early in training the RNN learns a single fixed point; subsequently the fixed point destabilizes (unstable eigenvalues appear) and the system transitions to orbital dynamics. This transition coincides with a double-descent phenomenon in the loss curve.

### 3. Local Latent Dynamics: Clusters, Transitions, and Kick-Zones

Going beyond fixed-point analysis, the paper identifies three functional regions in state space via short rollouts:

**Clusters (dwell time >> 8)**:
- Regions where trajectories linger longest
- Frequent sign reversals in logit gradients (5–20 times)
- Almost exclusively contracting eigenvalues; locally stable
- Each cluster corresponds to a distinct output probability distribution

**Kick-zones (2 ≤ dwell time ≤ 8)**:
- Located downstream of clusters
- Moderate logit gradient changes (2–4 times)
- A small number of unstable directions; locally stretching flow field
- The critical region that triggers state transitions

**Transitions (dwell time < 2)**:
- Brief passages entered after crossing a kick-zone
- Trajectories move nearly deterministically toward the next cluster
- Very few logit gradient changes (<1); stable, directional flow field

**Noise-sensitivity validation**: Transition regions are nearly insensitive to noise conditions — once a trajectory crosses the kick-zone it advances quasi-deterministically. Cluster regions are highly noise-sensitive — trajectories diverge substantially under different noise conditions.

### 4. Single-Neuron Computation and Connectivity Structure

**Discovery of "Kick Neurons"**: Two triplets of neurons display a distinctive spatiotemporal activation profile — strongly negative pre-activation in clusters, near zero (at the ReLU threshold) in kick-zones, and positive in transitions. Small input perturbations determine whether the ReLU gate opens or closes, making these neurons the triggers for state transitions.

**Connectivity Structure**: Analysis of the recurrent weight matrix $W_{hh}$ reveals:
- Kick neurons within the same triplet mutually excite each other
- The two triplets mutually inhibit each other
- Two larger neural populations (each ~70 neurons) form self-excitatory, mutually inhibitory circuits
- These "noise-integrating populations" modulate the kick neurons via structured connectivity

**Causal Intervention Validation**:
- **Ablation ($\mu=0$)**: silencing kick neurons or severing input from noise-integrating populations → trajectories become trapped in the current cluster, unable to transition; critical eigenvalue pairs vanish and orbital dynamics collapse to a fixed point
- **Enhancement ($\mu=2$)**: doubling kick-neuron activity → trajectories overshoot the target cluster; critical eigenvalue pairs are unchanged and orbital dynamics are preserved

### 5. Self-Induced Stochastic Resonance

All of the foregoing analyses converge on a unified computational principle — **self-induced stochastic resonance (SISR)**:

Unlike classical stochastic resonance (which requires an external periodic signal), SISR arises intrinsically in systems with time-scale separation. In the RNN:
- **Slow subsystem**: noise-integrating populations accumulate stochastic inputs in the cluster region
- **Fast subsystem**: kick neurons fire once the noise-modulated threshold is reached, triggering rapid transitions
- The two subsystems cooperate to produce stable quasi-periodic oscillations, whose period is determined by the interplay between noise variance and slow integration dynamics

The network effectively converts internal noise into a computational signal, implementing structured probabilistic inference through SISR-like dynamics.

## Key Contributions and the Compositionality Principle

The most important finding of this paper is the identification of **composable dynamical primitives**: the same basic unit (slow noise integration + fast kick-triggered reset) can be modularly reused and composed to generate more complex discrete latent structures.

For simple linear-chain HMMs a single orbit suffices; as the number of states increases, the RNN adjusts the alignment between the readout axis and the orbital plane to capture finer-grained emission discretization. For fully connected and cyclic HMMs, the RNN develops multiple orbits connecting different pairs of slow regions — each orbit being an instance of the same basic motif.

## Key Experimental Results

### Main Results

| Aspect | Core Finding |
|--------|-------------|
| Noise-free dynamics | All architectures → single fixed point; no multi-attractor structure |
| Noise-driven dynamics | Noise-sustained orbital dynamics; radius ∝ noise variance |
| State-space decomposition | Clusters (slow/stable) → Kick-zones (trigger) → Transitions (fast/deterministic) |
| Connectivity structure | Noise-integrating populations ↔ kick-neuron triplets; mutual excitation–inhibition |
| Causal validation | Ablation → trapped in cluster; enhancement → overshoot |
| Training dynamics | Fixed point → destabilization → orbital emergence, with double descent |
| Cross-architecture generalization | Linear-chain / fully connected / cyclic HMMs all use the same dynamical primitive |

## Limitations & Future Work

1. **Scale limitation**: Validation is restricted to HMMs with at most 5 states; whether larger discrete structures employ the same primitives remains to be explored.
2. **Architecture limitation**: Only vanilla RNNs (ReLU) are studied; whether gated architectures such as GRUs or LSTMs produce the same mechanism is unknown.
3. **Biological validation**: Although biological plausibility is hypothesized, no direct comparison with real neural circuits is provided.
4. **Deterministic output**: Training uses the Gumbel-Softmax approximation, which still differs from the exact discrete sampling of a true HMM.

## Personal Reflections

This work is methodologically elegant: rather than having RNNs perform behavioral tasks directly, it uses HMMs as "proxies for known computation," enabling reverse engineering of the RNN mechanism in a setting where the ground truth is available. This "synthetic neuroscience" paradigm is highly insightful.

The discovery of SISR is particularly compelling — it provides a unified account of global orbital dynamics, local state-space structure, single-neuron function, and connectivity topology, forming a complete causal chain from macro to micro scales. The concept of composable primitives implies a kind of RNN "programming language" in which complex behaviors can be assembled from simple modules.

Implications for computational neuroscience: the brain may implement discrete states not through multistable attractors (as traditionally hypothesized) but by functionally "sculpting" discrete structure from noise-driven continuous orbital dynamics. This has important consequences for understanding the neural mechanisms underlying natural behavior.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

<!-- RELATED:END -->
