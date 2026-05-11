---
title: >-
  [Paper Note] Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces
description: >-
  [ICLR2026][LLM Evaluation][SNN] This paper proposes an online SNN decoder that eliminates BPTT by combining three-factor Hebbian local learning rules with dual-timescale eligibility traces and adaptive learning rate cont…
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "SNN"
  - "BCI"
  - "Hebbian learning"
  - "online adaptation"
  - "spiking neural networks"
date: 2026-05-08
content_hash: 1c3f173a27ce2c91
---

# Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces

**Conference**: ICLR2026
**arXiv**: [2509.14447](https://arxiv.org/abs/2509.14447)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: SNN, BCI, Hebbian learning, online adaptation, spiking neural networks

## TL;DR
This paper proposes an online SNN decoder that eliminates BPTT by combining three-factor Hebbian local learning rules with dual-timescale eligibility traces and adaptive learning rate control. The approach achieves neural decoding accuracy comparable to offline-trained methods (Pearson R ≥ 0.63/0.81) under O(1) memory complexity, and demonstrates continuous adaptation to non-stationary neural signals in closed-loop simulations.

## Background & Motivation

### State of the Field
Brain-computer interfaces (BCIs) translate neural activity into control signals, bypassing conventional neuromuscular pathways. Invasive methods provide high-fidelity recordings but face challenges including signal instability, high noise, and resource constraints. Decoders have evolved from classical Kalman filters to deep learning approaches such as LSTMs; however, traditional methods struggle with non-stationarity, while deep models require frequent recalibration.

### Limitations of Prior Work

**Signal non-stationarity**: Neural recordings drift due to electrode encapsulation, neural plasticity, and other factors, requiring frequent recalibration that disrupts user experience.

**High-dimensional noise**: Electrophysiological data are high-dimensional and noisy, making low-latency decoding difficult.

**Poor cross-session/subject generalization**: Generalizing models across sessions or individuals requires retraining.

**Computational constraints**: BPTT requires O(T) memory, making it unsuitable for implantable systems with strict power and memory budgets; backpropagation also lacks biological plausibility in neural systems (the weight transport problem).

### Root Cause
A fundamental tension exists between online adaptability and computational efficiency: continuous online adaptation demands sufficiently expressive learning algorithms, yet implantable BCI hardware is severely resource-constrained and cannot afford the O(T) memory and computational overhead of BPTT. Furthermore, existing methods address individual problems in isolation, lacking a unified mechanism.

### Paper Goals
To design a unified framework that integrates multi-factor plasticity, dual-timescale consolidation, and online meta-learning within SNNs, such that: (1) BPTT is avoided to reduce memory and computational overhead; (2) sample-by-sample online adaptation is supported; and (3) the framework is compatible with neuromorphic hardware.

### Starting Point
Eligibility traces are reframed as Hebbian accumulators (rather than gradient surrogates approximating BPTT), modulated by reinforcement signals, and combined with fast–slow timescale memory consolidation to balance plasticity and stability.

### Core Idea
Construct an O(1)-memory online SNN-BCI decoder using local three-factor Hebbian rules, dual-timescale eligibility traces, and meta-learning adaptive learning rate control.

## Method

### Overall Architecture
The input is a raw spike-count vector $\mathbf{x}_t \in \mathbb{R}^N$ and the output is a 2D velocity prediction $\hat{\mathbf{y}}_t \in \mathbb{R}^2$. The network is a three-layer LIF neuron architecture (with recurrent connections in the first hidden layer), trained online at each time step to minimize the per-step squared error $\mathcal{L}_t = \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|_2^2$. The entire learning process requires neither unrolling the computation graph nor a replay buffer.

### Key Designs

1. **Three-Factor Hebbian Plasticity**:

    - *Function*: Computes local weight updates; forms the foundation of the entire learning algorithm.
    - *Mechanism*: Error-driving signals at each layer are propagated through the current weight space (not through time), and updates are computed by combining presynaptic activity, postsynaptic sensitivity (LIF surrogate gradient $d_{\text{LIF}}$), and the error signal: $\Delta W^{(\ell)}_{\text{hebb}}(t) = (\tilde{\mathbf{e}}^{(\ell)}_t \odot d^{(\ell)}_t)(\text{pre}^{(\ell)}_t)^\top$
    - *Design Motivation*: The three-factor rule preserves computational locality (only current time-step information is required), balancing biological plausibility with task supervision. The surrogate gradient acts as a "sensitivity gate," concentrating plasticity on neurons near their firing threshold.

2. **Dual-Timescale Eligibility Traces**:

    - *Function*: Accumulates instantaneous Hebbian updates into fast and slow traces, enabling information integration across timescales.
    - *Mechanism*: The fast trace $E^{\text{fast}}$ decays rapidly ($\tau_{\text{fast}}=120$ ms) and the slow trace $E^{\text{slow}}$ decays slowly ($\tau_{\text{slow}}=700$ ms), both updated via exponential decay: $E^{\text{fast}}(t) = \lambda_{\text{fast}} E^{\text{fast}}(t-1) + \Delta W_{\text{hebb}}(t)$. The combined trace is $E_{\text{comb}} = \alpha_{\text{mix}} E^{\text{fast}} + (1 - \alpha_{\text{mix}}) E^{\text{slow}}$.
    - *Design Motivation*: This mimics early and late long-term potentiation (LTP) in biological synaptic plasticity. The fast trace captures immediate changes for rapid correction, while the slow trace accumulates persistent evidence to maintain stability.

3. **Dual-Channel Weight Updates**:

    - *Function*: Applies eligibility traces via two separate pathways—fast and slow updates.
    - *Mechanism*: The fast update applies the combined trace at every time step: $W^{(\ell)} \leftarrow W^{(\ell)} + \eta_{\text{fast}} E^{(\ell)}_{\text{comb}}(t)$. The slow update applies every $K$ steps using a momentum-smoothed accumulator $G^{(\ell)}$ after RMS normalization: $W^{(\ell)} \leftarrow W^{(\ell)} + \eta_{\text{slow}} \mathcal{R}(\bar{G}^{(\ell)}_K)$.
    - *Design Motivation*: The fast pathway ensures immediate responsiveness to sudden non-stationarities, while the slow pathway ensures long-term stable learning. This design directly addresses the stability–plasticity dilemma in online learning.

4. **Stability Control Mechanisms**:

    - *Function*: Prevents numerical instability during continuous online adaptation.
    - *Mechanism*: Three safeguards are employed: (1) RMS normalization of error and spike signals using exponential moving averages; (2) weight projection constraining per-row weight norms as $\|W^{(\ell)}_{i:}\|_2 \leq c_\ell = 6$; and (3) adaptive learning rate control adjusting a learning rate multiplier every $K$ steps based on windowed loss change: $p_{t+1} = \text{clip}(p_t[1 + \eta_{\text{meta}} z_t])$, increasing plasticity when loss decreases and contracting it when learning stagnates.
    - *Design Motivation*: Sample-by-sample online updates are highly susceptible to divergence, necessitating hardware-friendly normalization and constraint mechanisms.

5. **Error-Modulated Lookup Table (LUT)**:

    - *Function*: Discretizes output errors into 16 bins to rescale the fast learning rate according to error magnitude.
    - *Design Motivation*: Provides a hardware-friendly coarse-grained neuromodulatory signal without additional computational complexity.

### Loss & Training
- **Loss function**: Per-step squared error $\mathcal{L}_t = \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|_2^2$
- **Training strategy**: Purely online, sample-by-sample updates (batch size = 1); convergence achieved in only 5 epochs.
- **Memory complexity**: O(P) in parameter dimensions; O(1) in sequence length $T$.

## Key Experimental Results

### Main Results
Evaluation is conducted on two primate intracortical datasets: MC Maze (resampled at 10 ms, 80 ms kinematic lag) and Zenodo Indy (50 ms bins, zero lag).

| Dataset | Method | Pearson R (X) | Pearson R (Y) | Notes |
|--------|------|---------------|---------------|------|
| MC Maze | Online SNN (Batched) | ~0.81 | ~0.81 | Comparable to BPTT-SNN |
| MC Maze | BPTT-SNN | ~0.85 | ~0.85 | 50 epochs + Adam |
| MC Maze | LSTM | ~0.80 | ~0.80 | Offline training |
| MC Maze | Kalman Filter | ~0.65 | ~0.65 | Online sequential |
| Zenodo Indy | Online SNN (Batched) | ~0.63 | ~0.63 | Comparable to offline methods |
| Zenodo Indy | BPTT-SNN | ~0.65 | ~0.65 | 50 epochs |

### Memory Overhead Comparison

| Architecture | Online (MB) | BPTT (MB) | Reduction |
|------|-------------|-----------|----------|
| 96-256-128-2 | 1.41 | 2.17 | 35% |
| 96-1024-512-2 | 19.15 | 26.67 | 28% |

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| Three-factor vs. Delta Rule | Dataset-dependent | Three-factor significantly better on Zenodo; marginal difference on MC Maze |
| Recurrent vs. feedforward | Recurrent superior | Recurrent connections contribute on both datasets; larger contribution on Zenodo |
| Full RMS vs. no RMS | Full RMS critical | RMS normalization essential on Zenodo; partial RMS should be avoided |
| Dual-timescale vs. single trace | Optimal choice is dataset-dependent | MC Maze favors slow/dual; Zenodo favors fast |
| Dual-channel vs. single-channel update | Dual-channel safest | Slow-only or frozen updates are harmful across all datasets |
| Meta-adaptive vs. fixed learning rate | Small gain | Worth retaining when resources allow, but not the primary driver |

### Key Findings from Closed-Loop Simulation
- **90% remapping perturbation**: Online SNN recovers to pre-perturbation reach time (~0.30 s) within ~20 reaches; the fixed model exceeds 1.5 s.
- **90% drift perturbation**: Online SNN adapts from 1.5 s to ~0.75 s within 20 reaches.
- **90% dropout perturbation**: Online SNN recovers within 15–20 reaches.
- **Learning from scratch**: Online SNN without pre-training starts at 0.75 s and stabilizes at 0.6 s through online learning; the offline fixed-weight method is nearly non-functional prior to calibration.

### Key Findings
- The Online SNN reaches performance approaching BPTT-SNN trained for 50 epochs using only 5 epochs of sample-by-sample updates, demonstrating superior sample efficiency.
- Ablation results exhibit strong dataset dependency: MC Maze has high SNR, so simple rules suffice; the continuous mixed recordings in Zenodo require the noise robustness provided by three-factor gating.
- Closed-loop adaptation is the most prominent advantage of the Online SNN—fixed-parameter methods are entirely unable to handle non-stationarity.

## Highlights & Insights
- The decomposition **three-factor = Hebbian × surrogate gradient × error** is particularly elegant: it preserves biological plausibility through local computation while introducing task-relevant credit assignment via surrogate gradient gating, representing a well-crafted compromise.
- The **fast/slow dual-timescale design permeates the entire method** (traces, weight updates, and learning rate control), with nested structures addressing adaptation needs at different timescales. This design philosophy transfers naturally to other continual learning scenarios.
- **RMS normalization and weight projection** serve as hardware-friendly stability tools that replace methods requiring global statistics (e.g., BatchNorm), offering valuable insights for neuromorphic chip deployment.
- The closed-loop "learning from scratch" experiment demonstrates the possibility of using BCIs without offline calibration, which carries significant implications for clinical applications.

## Limitations & Future Work
- Closed-loop experiments are based on synthetic neural populations and have not been validated on real chronic human recordings.
- The consolidation window $K$ and reset thresholds are tuned manually; fully automated scheduling mechanisms remain to be developed.
- Actual deployment and scalability on neuromorphic hardware have not been verified.
- The strong dataset dependency of ablation results suggests that the method may require hyperparameter tuning for different BCI scenarios, raising questions about generalizability.
- Only 2D velocity decoding is evaluated; more complex high-degree-of-freedom motor control tasks (e.g., finger movements) remain unexplored.

## Related Work & Insights
- **vs. e-prop (Bellec et al., 2020)**: e-prop also uses eligibility traces for BPTT-free SNN learning, but its traces originate from approximate gradients of BPTT. This work reframes traces as Hebbian accumulators, placing greater emphasis on biological plausibility and hardware friendliness.
- **vs. SuperSpike (Zenke & Ganguli, 2018)**: SuperSpike employs broadcast error signals with local traces, but still relies on gradient flow in trace derivation. The three-factor rule proposed here achieves more purely local computation.
- **vs. conventional R-STDP**: R-STDP uses sparse, delayed dopaminergic-like signals for modulation, whereas this work uses dense per-frame kinematic errors for credit assignment—richer in information but with slightly reduced biological plausibility.
- The dual-timescale consolidation concept invites interesting comparison with elastic weight consolidation (EWC) and related methods in continual/incremental learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — The unified framework organically integrates multiple existing ideas (three-factor rules, dual timescales, meta-learning), though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, comprehensive ablations, and closed-loop simulations are provided, but validation on real hardware and human data is absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)
- [\[ICLR 2026\] LCA: Local Classifier Alignment for Continual Learning](lca_local_classifier_alignment_for_continual_learning.md)
- [\[AAAI 2026\] SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](../../AAAI2026/llm_evaluation/spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)
- [\[AAAI 2026\] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version](../../AAAI2026/llm_evaluation/streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)

</div>

<!-- RELATED:END -->
