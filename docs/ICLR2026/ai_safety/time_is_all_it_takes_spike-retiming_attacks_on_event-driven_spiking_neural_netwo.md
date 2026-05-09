---
title: >-
  [Paper Note] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks
description: >-
  [ICLR 2026][AI Safety][Spiking Neural Networks] This paper proposes the Spike-Retiming Attack — a temporal attack that perturbs only spike timestamps without adding or removing spikes. It formalizes a unified tri-norm budget ($\mathcal{B}_\infty$ local jitter / $\mathcal{B}_1$ total delay / $\mathcal{B}_0$ tamper count) under a capacity-1 constraint, and employs Projected-in-the-Loop (PIL) optimization to decouple strict forward projection from soft backward differentiation. The method achieves >90% ASR with <2% spike perturbation on CIFAR10-DVS/DVS-Gesture/N-MNIST, revealing a critical temporal vulnerability in event-driven SNNs.
tags:
  - ICLR 2026
  - AI Safety
  - Spiking Neural Networks
  - Adversarial Attacks
  - Spike Retiming
  - Event-Driven
  - Temporal Robustness
  - LIF Neuron
date: 2026-05-08
content_hash: 735120ba3b08330d
---

# Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2602.03284](https://arxiv.org/abs/2602.03284)
**Code**: [github.com/yuyi-sd/Spike-Retiming-Attacks](https://github.com/yuyi-sd/Spike-Retiming-Attacks)
**Area**: Adversarial Security of Spiking Neural Networks
**Keywords**: Spiking Neural Networks, Adversarial Attacks, Spike Retiming, Event-Driven, Temporal Robustness, LIF Neuron

## TL;DR
This paper proposes the Spike-Retiming Attack — a temporal attack that perturbs only spike timestamps without adding or removing spikes. It formalizes a unified tri-norm budget ($\mathcal{B}_\infty$ local jitter / $\mathcal{B}_1$ total delay / $\mathcal{B}_0$ tamper count) under a capacity-1 constraint, and employs Projected-in-the-Loop (PIL) optimization to decouple strict forward projection from soft backward differentiation. The method achieves >90% ASR with <2% spike perturbation on CIFAR10-DVS/DVS-Gesture/N-MNIST, revealing a critical temporal vulnerability in event-driven SNNs.

## Background & Motivation

**Temporal computation in SNNs**: Spiking Neural Networks (SNNs) rely on discrete spikes and temporal coding for computation, offering low energy consumption and low latency on neuromorphic processors. Directly trained SNNs using spatio-temporal backpropagation (STBP) with surrogate gradients have approached ANN-level accuracy, making temporal information central to their computation.

**Blind spots in existing attacks**: Existing SNN adversarial attacks (PGD, RGA, HART, SpikeFool, PDSG-SDA, etc.) largely inherit image-domain strategies — modifying intensity values or adding/removing events. These attacks alter energy/firing-rate statistics and are susceptible to detection by intensity- or rate-based methods.

**Realism of temporal attacks**: Event cameras inherently exhibit timestamp noise (jitter) and readout delays, and SNN pipelines typically quantize events into discrete time bins. Attacks that modify only timestamps while preserving spike count and amplitude fall entirely within the sensor's temporal uncertainty range, leaving frame-level intensity and rate statistics unchanged and rendering them extremely difficult for existing defenses to detect.

**Temporal gap in defenses**: Existing defenses — including certified robustness, adversarial training, and biologically inspired mechanisms — primarily regularize against intensity, rate, or membrane potential perturbations. Virtually no defense addresses input temporal perturbations, leaving a clear gap in temporal robustness evaluation.

**Paradigm shift from insertion/deletion to redistribution**: Conventional attacks search the 0-norm space of "adding/removing spikes." This work reframes the attack as a temporal-axis *redistribution* problem — reallocating spike timestamps while preserving the capacity-1 constraint per event line, applicable to both binary and integer event grids.

## Method

### Threat Model Definition

The temporal attack is formalized as a **Spike Timing Attack**: given an input event tensor $\bm{x} \in \mathbb{Z}_{\geq 0}^{T \times C \times H \times W}$, for each active spike $(s, j) \in \mathcal{A}(\bm{x})$, an integer offset $\delta_{s,j}$ is selected to shift the spike from time $s$ to $t = s + \delta_{s,j}$. Constraints include: (1) temporal boundary $0 \leq t < T$; (2) capacity-1 non-overlap constraint — at most one spike per event line per time bin. The placement function $P(\bm{x}; \delta)$ replays the same spikes at new times, preserving amplitude and count.

### Unified Tri-Norm Budget

- **$\mathcal{B}_\infty(\varepsilon)$**: bounds the maximum per-spike jitter $|\delta_{s,j}| \leq \varepsilon$, corresponding to sensor timestamp uncertainty, favoring local retiming.
- **$\mathcal{B}_1(\varepsilon)$**: bounds the total temporal displacement $\sum |\delta_{s,j}| \leq \varepsilon$, providing a global knob that scales with event density.
- **$\mathcal{B}_0(\varepsilon)$**: bounds the number of perturbed spikes $\sum \mathbb{I}\{\delta_{s,j} \neq 0\} \leq \varepsilon$, capturing covert minimal-footprint attacks.

Integer grids are accommodated under the capacity-1 constraint by decomposing each count into unit "packets."

### Projected-in-the-Loop (PIL) Optimization

The core challenge is the conflict between a discrete feasible space and gradient requirements. PIL resolves this via a straight-through estimator:

1. **Offset logits**: For each active spike, offset probabilities $\pi[s,j,u] = \text{softmax}(\phi[s,j,u]/\kappa)$ are introduced over the feasible offset set $\mathcal{U}_p$, modeling a distribution over offsets.
2. **Soft offset operator**: $\tilde{\bm{x}} = S_\pi(\bm{x})$ computes the expected retimed result, is fully differentiable, and provides temporally aligned gradients for backpropagation.
3. **Strict projection**: $\hat{\bm{x}} = P^*(\bm{x}; \pi, \mathcal{B}_p(\varepsilon))$ greedily places spikes in probability-sorted order during the forward pass, strictly satisfying capacity-1 and budget constraints.
4. **PIL coupling**: $\bm{x}_{\text{PIL}} = \hat{\bm{x}} + (\tilde{\bm{x}} - \text{stopgrad}(\tilde{\bm{x}}))$ — the forward pass uses the strictly projected result for evaluation, while the backward pass differentiates through the soft offset.

### Budget-Aware Objective

$$\mathcal{J} = \mathcal{L}(f(\bm{x}_{\text{PIL}}), y) - \lambda_{\text{cap}} \cdot \text{Cap}(\pi; \bm{x}) - \lambda_{\text{budget}} \cdot \mathcal{R}_p(\pi; \varepsilon)$$

- Task loss $\mathcal{L}$: maximizes cross-entropy for untargeted attacks.
- Capacity regularization $\text{Cap}$: penalizes time bins whose expected occupancy exceeds 1, $\text{Cap} = \frac{1}{|\mathcal{A}|} \sum_{j,t} [\text{occ}[t,j] - 1]_+^2$.
- Budget penalty $\mathcal{R}_p$: a normalized hinge loss guiding logits toward the feasible region; $\mathcal{B}_\infty$ requires no additional penalty (the support set already encodes the constraint), while $\mathcal{B}_1/\mathcal{B}_0$ use soft total displacement / soft moved-spike count respectively.

Logits are updated via clipped sign-PGD: $\phi \leftarrow \text{clip}_{[-\phi_{\max}, \phi_{\max}]}(\phi + \alpha \cdot \text{sign}(\nabla_\phi \mathcal{J}))$.

## Key Experimental Results

### Experimental Setup
- **Datasets**: CIFAR10-DVS (10 classes), DVS-Gesture (11 gesture classes), N-MNIST (handwritten digits)
- **Models**: ConvNet, Spiking ResNet18, VGGSNN, SpikingResformer — all directly trained SNNs
- **Time bins**: $T=10$; evaluation metric is Attack Success Rate (ASR)
- **Default hyperparameters**: $\kappa=1, \alpha=1, \phi_{\max}=10, \lambda_{\text{cap}}=20, \lambda_{\text{budget}}=10$

### Main Results (Binary Grid)

| Dataset | Model | Clean Acc. | $\mathcal{B}_\infty(1)$ | $\mathcal{B}_\infty(3)$ | $\mathcal{B}_0$ Max |
|--------|------|---------|------------------------|------------------------|---------------------|
| N-MNIST | ConvNet | 99.06% | **100%** | 100% | 98.5% (400) |
| N-MNIST | ResNet18 | 99.62% | **100%** | 100% | 100% (300) |
| DVS-Gesture | VGGSNN | 95.14% | 96.4% | 100% | 98.9% (4k) |
| DVS-Gesture | SpResF | 91.67% | 92.1% | 100% | 99.2% (4k) |
| CIFAR10-DVS | SpResF | 81.30% | **100%** | 100% | 100% (4k) |

Key finding: under $\mathcal{B}_\infty$, a jitter of only 1 bin nearly saturates ASR; $\mathcal{B}_0(4\text{k})$ achieves >98% ASR on DVS-Gesture by perturbing only 2.45% of spikes.

### Main Results (Integer Grid)

| Dataset | Model | Clean Acc. | $\mathcal{B}_\infty(1)$ | $\mathcal{B}_\infty(3)$ | $\mathcal{B}_0$ Max |
|--------|------|---------|------------------------|------------------------|---------------------|
| N-MNIST | VGGSNN | 99.71% | 46.3% | 100% | 49.8% (600) |
| DVS-Gesture | ResNet18 | 94.40% | 71.0% | 93.3% | 98.1% (8k) |
| DVS-Gesture | SpResF | 92.71% | 70.7% | 84.0% | 80.6% (8k) |
| CIFAR10-DVS | SpResF | 82.90% | **100%** | 100% | 100% (8k) |

Key finding: the integer grid is consistently more robust under $\mathcal{B}_1$ and $\mathcal{B}_0$, requiring a larger budget to achieve equivalent ASR. Contributing factors include: (1) integer multiplicity smooths the pre-activation distribution; (2) temporal convolution and normalization integrate over accumulated counts more stably; (3) surrogate gradients and normalization statistics exhibit smaller variance on integer inputs.

### Ablation Study (DVS-Gesture + VGGSNN)

| Variant | Binary $\mathcal{B}_\infty(1)$ | Binary $\mathcal{B}_1(8k)$ | Binary $\mathcal{B}_0(4k)$ | Integer $\mathcal{B}_\infty(3)$ | Integer $\mathcal{B}_0(8k)$ |
|------|------------|------------|------------|------------|------------|
| Full method | 96.4% | 98.5% | 98.9% | 85.0% | 95.9% |
| w/o PIL | 92.7% | 84.3% | 88.6% | 63.0% | 83.1% |
| w/o Cap | 95.6% | 98.5% | 98.5% | 77.6% | 89.6% |
| w/o $\mathcal{R}_p$ | — | 76.6% | 93.0% | — | 84.9% |

PIL contributes most (binary $\mathcal{B}_1$: 98.5%→84.3%); the budget penalty $\mathcal{R}_p$ has the greatest impact under $\mathcal{B}_1$.

## Highlights & Insights

1. **First purely temporal threat model**: Formalizes a timing attack that preserves spike count and amplitude, unifying three norm budgets ($\mathcal{B}_\infty/\mathcal{B}_1/\mathcal{B}_0$) and two event grids (binary/integer), filling a gap in SNN temporal robustness evaluation.

2. **Elegant PIL optimization framework**: Strict forward projection enforces feasibility while soft backward differentiation preserves gradient information; combined with capacity regularization and budget-aware penalties, PIL enables efficient gradient-guided search under discrete constraints.

3. **High stealthiness**: The attack leaves frame-level intensity and rate statistics entirely unchanged, operating within sensor temporal uncertainty. On DVS-Gesture, perturbing fewer than 2.5% of spikes suffices to break the model, rendering existing defenses (filtering, adversarial training) largely ineffective.

4. **Discovery of polarity-dependent temporal shift patterns**: Positive-polarity channels tend to delay (red-shift) while negative-polarity channels tend to advance (blue-shift), revealing an asymmetric temporal dependence of SNNs on positive and negative events.

## Limitations & Future Work

1. **White-box assumption**: The attack requires full model access (parameters and gradients); attack capability in black-box settings is limited, and cross-architecture transferability remains to be improved.

2. **Low targeted attack success rate**: Compared to >90% ASR for untargeted attacks, targeted ASR drops significantly (approximately 25% under $\mathcal{B}_\infty(1)$), necessitating stronger targeted optimization strategies.

3. **High cost of adversarial training**: Adversarial training with spike retiming severely degrades clean accuracy (to 22–48% on binary grids) with limited robustness gain, indicating that current defense frameworks are ill-suited to purely temporal threats.

4. **Attack efficacy decreases with larger time bin counts**: When $T$ increases from 10 to 40, ASR on binary grids under a fixed $\mathcal{B}_0$ budget drops from 98.9% to 13.3%, indicating a negative correlation between attack efficiency and temporal resolution.

5. **Computational overhead**: The strict projection step requires sorting and greedy scanning; scalability to large-scale event streams is not thoroughly discussed.

## Related Work & Insights

- **SNN training**: STBP spatio-temporal backpropagation (Wu et al., 2018), tdBN normalization (Zheng et al., 2021), TET temporally efficient training (Deng et al., 2022), surrogate gradient improvements (Li et al., 2021)
- **SNN adversarial attacks**: RGA rate gradient approximation (Bu et al., 2023), HART hybrid rate-timing attack (Hao et al., 2024), SpikeFool sparse rounding (Büchel et al., 2022), GSAttack Gumbel-Softmax (Yao et al., 2024), PDSG-SDA membrane-potential-correlated surrogate gradient + sparse dynamic attack (Lun et al., 2025)
- **SNN defenses**: Certified robustness via IBP/randomized smoothing (Mukhoty et al., 2024), gradient sparsity regularization (Liu et al., 2024d), membrane potential perturbation minimization (Ding et al., 2024a), DVS noise filtering (Marchisio et al., 2021b)

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Overall Recommendation**: ⭐⭐⭐⭐⭐ — A pioneering work that systematically establishes SNN temporal robustness evaluation for the first time. The threat model is rigorously formalized, the experiments are comprehensive (3 datasets × 4 models × 3 norms × 2 grids), the PIL optimization framework balances discrete feasibility with gradient-based optimization, and the work reveals a fundamental temporal vulnerability in event-driven SNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](../../AAAI2026/ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)

</div>

<!-- RELATED:END -->
