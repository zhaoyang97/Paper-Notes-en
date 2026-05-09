---
title: >-
  [Paper Note] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines
description: >-
  [NeurIPS 2025][Optimization][equilibrium propagation] This work presents the first complete mapping of Equilibrium Propagation (EP) onto Oscillator Ising Machine (OIM) hardware, leveraging GHz-scale physical dynamics to enable backpropagation-free local learning. The approach achieves 97.2%/88.0% accuracy on MNIST/Fashion-MNIST and demonstrates robustness under parameter quantization and noise.
tags:
  - NeurIPS 2025
  - Optimization
  - equilibrium propagation
  - oscillator Ising machine
  - neuromorphic computing
  - energy-based model
  - local learning rule
date: 2026-05-08
content_hash: dc0d58282e63a362
---

# Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines

**Conference**: NeurIPS 2025
**arXiv**: [2510.12934](https://arxiv.org/abs/2510.12934)
**Code**: [alexgower/OIM-Equilibrium-Propagation](https://github.com/alexgower/OIM-Equilibrium-Propagation)
**Area**: Optimization
**Keywords**: equilibrium propagation, oscillator Ising machine, neuromorphic computing, energy-based model, local learning rule

## TL;DR
This work presents the first complete mapping of Equilibrium Propagation (EP) onto Oscillator Ising Machine (OIM) hardware, leveraging GHz-scale physical dynamics to enable backpropagation-free local learning. The approach achieves 97.2%/88.0% accuracy on MNIST/Fashion-MNIST and demonstrates robustness under parameter quantization and noise.

## Background & Motivation
**State of the Field**: Physical systems naturally perform energy descent, which can directly accelerate optimization in energy-based models (EBMs). OIMs consist of coupled nonlinear oscillator networks whose GHz-frequency dynamics inherently correspond to gradient descent.

**Limitations of Prior Work**: (a) EP on conventional processors is constrained by long relaxation and sampling times; (b) prior attempts to implement EP on oscillator systems suffer from initialization or synchronization issues; (c) discrete Ising solvers do not support the continuous phase dynamics required by EP.

**Root Cause**: Although OIMs were originally designed for combinatorial optimization, their continuous phase dynamics and energy descent properties naturally satisfy the requirements of EP—raising the question of whether OIMs can be repurposed as neuromorphic learning processors without any hardware modification.

**Starting Point**: The paper demonstrates that the OIM energy function can exactly encode the total energy of an MLP (including MSE loss), and that EP update rules reduce to local phase measurements on the OIM.

## Method

### OIM Dynamics

A network of $n$ coupled oscillators, each parameterized by phase $\phi_i \in [0, 2\pi]$, evolves via gradient descent on an energy function $V$:

$$\frac{d\phi_i}{dt'} = -\frac{\partial V}{\partial\phi_i}$$

$$V = -\frac{1}{2}\sum_{i,j\ne i} J_{ij}\cos(\phi_i - \phi_j) - \sum_i h_i\cos(\phi_i) - \sum_i \frac{S_i}{2}\cos(2\phi_i)$$

where $J_{ij}$ are coupling strengths, $h_i$ are biases, and $S_i$ are synchronization fields. Physical time $t = t'/\bar{\omega}$ is inversely proportional to oscillator frequency.

### Mapping EP onto OIM

**Network Architecture**: The MLP has $n_x$ input, $n_h$ hidden, and $n_y$ output neurons. All non-input neurons correspond to oscillators, with activations $s_i = \cos(\phi_i) \in [-1, 1]$.

**Energy Decomposition**: $F = E + \beta\ell$, where $E$ is the free energy, $\ell$ is the MSE loss, and $\beta$ is the nudge factor.

**OIM Parameter Correspondence**:

| MLP Component | OIM Parameter |
|---------|----------|
| Hidden bias $b_i^{(h)}$ + input weights | $h_i^{(h)} = b_i^{(h)} + \sum_j w_{ji}^{(x,h)} x_j$ |
| Hidden-output weights $w_{ij}^{(h,y)}$ | $J_{ij}^{(h,y)} = w_{ij}^{(h,y)}$ |
| Output bias + target | $h_i^{(y)} = b_i^{(y)} + \beta\hat{y}_i$ |
| MSE loss term | $S_i^{(y)} = -\beta/2$ |

Crucially, expanding the MSE loss $\frac{1}{2}(\cos\phi_i - \hat{y}_i)^2$ yields $\cos(2\phi_i)$ and $\cos(\phi_i)$ terms that naturally align with the $S_i$ and $h_i$ terms already present in the OIM energy function.

### Three-Phase EP Training

For each training sample $x$:
1. **Free phase** ($\beta=0$): Evolve from reference state $\phi_0 = \{\pi/2\}$ (corresponding to activation $\cos(\pi/2)=0$) to steady state $\phi_*$
2. **Positive nudge phase** ($\beta>0$): Evolve from $\phi_*$ to $\phi_*^{+\beta}$
3. **Negative nudge phase** ($\beta<0$): Evolve from $\phi_*$ to $\phi_*^{-\beta}$

**Parameter Updates (fully local)**:

$$\Delta w_{ij}^{(h,y)} \propto -\frac{1}{2\beta}[\cos(\phi_i^{(h),-\beta} - \phi_j^{(y),-\beta}) - \cos(\phi_i^{(h),+\beta} - \phi_j^{(y),+\beta})]$$

Updates depend only on the phase differences of connected oscillators—no global backpropagation circuit is required. Updates are averaged over mini-batches and are compatible with standard optimizers.

### EP–BPTT Equivalence

In the limit $\beta \to 0$, EP updates are strictly equivalent to backpropagation through time (BPTT):

$$\lim_{\beta\to 0} \hat{\nabla}^{\rm EP}(\beta) = -\frac{\partial\ell}{\partial\theta}(y_*, \hat{y})$$

The paper experimentally verifies that this correspondence holds under OIM's nonlinear sinusoidal coupling (Fig. 1 inset).

## Key Experimental Results

### Classification Accuracy

| Architecture | Dataset | EP Accuracy | BPTT Accuracy | Comparison |
|------|--------|:---:|:---:|------|
| 784-500-10 | MNIST | **97.2±0.1%** | 96.8±0.1% | — |
| 784-500-10 | Fashion-MNIST | **88.0±0.1%** | — | p-bit Ising: 87.0% |
| 784-120-10 | MNIST/100 | 90.6±1.7% | — | D-Wave: ~85% |
| 784-120-10 (with noise) | MNIST/100 | **92.0±0.3%** | — | noise $\xi=0.2$ |

EP accuracy matches or slightly exceeds same-architecture BPTT (97.2 vs. 96.8), indicating that performance is limited by architecture rather than training method.

### Hardware Robustness

| Test Condition | Accuracy | Notes |
|---------|:---:|------|
| Phase quantization 4-bit | 89.8±1.5% | Feasible |
| Phase quantization 2-bit | Significant drop | Infeasible |
| Weight quantization 10-bit | 89.4±1.5% | Feasible |
| Weight quantization 8-bit | Degraded | Borderline |
| Gaussian phase noise $\xi=0.2$ | **92.0±0.3%** | Moderate noise acts as regularization |
| Gaussian phase noise $\xi=0.3$ | Maintained ($\beta \gtrsim \xi/2$) | OIM noise ≈ Langevin dynamics |

**Key Findings**: The system is noise-robust when $\beta \ge \xi/2$; moderate noise ($\xi=0.2$) improves accuracy through a regularization effect.

### Projected Hardware Speedup

| Condition | EP Simulation Time | BPTT Simulation Time | Physical OIM Estimate |
|------|:---------:|:---------:|:-----------:|
| MNIST 50 epochs | ~40 hours | ~60 hours | **seconds to minutes** |

GHz oscillator frequencies enable phase updates at the microsecond scale, yielding theoretically orders-of-magnitude speedup.

## Highlights & Insights
1. An OIM designed for combinatorial optimization is repurposed as a neuromorphic learning processor **with zero hardware modification**.
2. Local update rules require no global backpropagation circuit—each synapse/neuron measures only local phases.
3. Noise robustness relaxes hardware precision requirements; noise at $\xi=0.2$ is even beneficial.
4. The EP–BPTT equivalence is experimentally validated under nonlinear sinusoidal coupling.
5. The natural correspondence between MSE loss and the OIM synchronization field $S_i$ is particularly elegant.

## Limitations & Future Work
1. Only single-hidden-layer MLPs are validated; deeper networks require more oscillators and more complex coupling topologies.
2. All experiments are conducted via PyTorch simulation; validation on physical OIM hardware remains to be demonstrated.
3. Only MSE loss is supported (due to the $\cos(2\phi)$ correspondence); losses such as cross-entropy do not directly map onto the OIM energy function.
4. Accuracy on MNIST/Fashion-MNIST lags behind CNNs/Transformers; this work is primarily a proof of concept.
5. Training requires $T=4000$ free-phase steps and $K=400$ nudge-phase steps, which is relatively costly.

## Related Work & Insights
- **vs. D-Wave annealers** (Laydevant et al.): D-Wave is a discrete Ising solver that does not support continuous EP; OIM's continuous phases are naturally compatible.
- **vs. p-bit Ising machines** (Niazi et al.): p-bit machines train Deep Boltzmann Machines; OIM-EP achieves higher accuracy on Fashion-MNIST.
- **vs. conventional EP** (Scellier & Bengio): This is the first complete implementation within a physical system framework rather than digital simulation.
- **vs. Wang et al. / Rageau et al.** (oscillator-based EP): Prior works suffer from initialization/synchronization issues; this work requires no additional hardware modifications.

The paradigm of "physical dynamics as ML optimization" is broadly applicable to other physical platforms such as optical neural networks and spintronics. OIM noise corresponds to Langevin dynamics, suggesting native support for sampling-based generative models. The locality of EP makes it particularly well-suited for large-scale distributed or neuromorphic computing.

## Rating
- ⭐ **Novelty**: 4/5 — The mapping of EP onto OIM is elegant; the correspondence between MSE loss and the synchronization field is a notable highlight.
- ⭐ **Experimental Thoroughness**: 3/5 — MNIST-scale validation is solid, but real hardware experiments and more complex tasks are absent.
- ⭐ **Writing Quality**: 4/5 — Concepts are clearly presented; the physics–ML mapping is well explained.
- ⭐ **Value**: 3/5 — Currently a proof of concept; practical impact depends on hardware maturity.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Wasserstein Transfer Learning](wasserstein_transfer_learning.md)
- [\[NeurIPS 2025\] Learning from Interval Targets](learning_from_interval_targets.md)
- [\[NeurIPS 2025\] Learning Parameterized Skills from Demonstrations](learning_parameterized_skills_from_demonstrations.md)

<!-- RELATED:END -->
