---
title: >-
  [Paper Note] Frequency Matching in Spiking Neural Networks for mmWave Sensing
description: >-
  [ICML 2026][AI Safety][LIF Neurons] This paper demonstrates from a "mechanism-data alignment" perspective that the LIF spiking neuron is equivalent to a first-order IIR low-pass filter. It proposes setting the membrane d…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "LIF Neurons"
  - "IIR Low-pass Filter"
  - "mmWave Sensing"
  - "Discriminative Spectrum"
  - "Neural Dynamics-Data Alignment"
date: 2026-05-08
content_hash: 2fc5cdc931c038d3
---

# Frequency Matching in Spiking Neural Networks for mmWave Sensing

**Conference**: ICML 2026  
**arXiv**: [2605.09983](https://arxiv.org/abs/2605.09983)  
**Code**: [GitHub](https://github.com/yudi-mars/Soul)  
**Area**: Edge Sensing / Spiking Neural Networks (SNN) / Wireless Sensing  
**Keywords**: LIF Neurons, IIR Low-pass Filter, mmWave Sensing, Discriminative Spectrum, Neural Dynamics-Data Alignment

## TL;DR
This paper demonstrates from a "mechanism-data alignment" perspective that the LIF spiking neuron is equivalent to a first-order IIR low-pass filter. It proposes setting the membrane decay coefficient $\beta$ based on the discriminative spectrum of mmWave signals. On four common mmWave datasets, the proposed SNN improves accuracy by an average of 6.22% compared to ANNs while reducing theoretical energy consumption by 3.64×.

## Background & Motivation
**Background**: mmWave radar is a critical sensor for edge-side posture, gesture, and activity recognition due to its privacy-friendliness, light immunity, and penetration capabilities. Mainstream solutions use ANNs like CNNs or Transformers, relying on depth and manual preprocessing for robustness, which incurs significant energy and latency costs.

**Limitations of Prior Work**: mmWave signals are inherently sparse, irregular, and heavily contaminated by high-frequency noise from multipath effects and phase jitter. ANNs lack a built-in temporal filtering bias; they either require manual low-pass preprocessing (which may remove useful high-frequency discriminative information) or rely on deeper networks for brute-force fitting, leading to unsustainable energy and latency.

**Key Challenge**: Discriminative information is often distributed in the "low-to-medium frequency bands," while noise is concentrated in high frequencies. Existing ANNs and low-pass preprocessing methods fail to distinguish between "useful high-frequency discriminative components" and "true high-frequency noise." While some SNN works show energy efficiency advantages, they rely on empirical hyperparameter tuning, leaving it unclear "when and why SNNs outperform ANNs."

**Goal**: To answer two questions from a signal processing perspective: (1) What is the mechanistic advantage of SNNs in mmWave sensing? (2) How should the key hyperparameter, membrane decay $\beta$, be selected based on the data spectrum?

**Key Insight**: Linearize the discrete dynamics of the LIF neuron into a first-order IIR low-pass filter and quantify the overlap between its cutoff frequency and the discriminative spectrum of the dataset. This transforms the task of "setting $\beta$" into a frequency-domain alignment problem.

**Core Idea**: Match the effective bandwidth $B_{\text{eff}}(\beta)$ of the LIF neuron to the discriminative spectrum $\Omega^\star$ of the mmWave data. "Frequency Matching" is the fundamental mechanism behind SNN superiority in such tasks and serves as the physical criterion for selecting $\beta$.

## Method

### Overall Architecture
The paper does not introduce a new network architecture but provides a frequency-domain mechanistic analysis and hyperparameter selection method for a "LIF neuron + LeNet-style SNN." The process consists of three steps: (1) Analyze each mmWave dataset along the time dimension using DFT and define a Fisher-style discriminative index $\mathrm{DI}(\omega_k)$, normalized as a probability distribution $\mathrm{DI}_{\text{norm}}$; (2) Formulate the LIF dynamics as $u_{t+1}=\beta u_t+(1-\beta)I_t-v_{\text{th}}O_t$, ignore the reset term to obtain the equivalent first-order IIR filter $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$, and define a DC-normalized power template $\tilde H(\omega_k;\beta)$ with a half-power cutoff $B_{\text{eff}}(\beta)$; (3) Measure the mechanism-data alignment using the dot product $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)$, and identify the over-low-pass boundary $\beta^\dagger$ using a "maximum deviation from reference diagonal" rule to partition $\beta$ into "under-filter," "stability window," and "over-low-pass" regions.

### Key Designs

1. **Data side: Discriminative Spectrum $\mathrm{DI}_{\text{norm}}(\omega_k)$**:

    - **Function**: Objectively quantifies the density of category-discriminative information at each frequency bin as the "data ground truth" for mechanism matching.
    - **Mechanism**: For each sample $\mathbf{X}_i\in\mathbb{R}^{L\times C\times H\times W}$, non-temporal dimensions are averaged to obtain a 1D sequence $\mathbf{s}_i\in\mathbb{R}^L$. After sample-wise de-meaning and DFT, the magnitude spectrum $A_i[k]$ is obtained. Inter-class variance $S_B[k]=\sum_c\pi_c(\mu_c[k]-\bar\mu[k])^2$ and intra-class variance $S_W[k]=\sum_c\pi_c\,\mathrm{Var}_c[k]$ are estimated per class to define $\mathrm{DI}(\omega_k)=S_B[k]/(S_W[k]+\varepsilon)$, which is then normalized in the frequency domain.
    - **Design Motivation**: Direct Fisher-style statistics of linear separability reflect both "signal energy distribution" and "category separability," serving as the intermediary between "data" and "mechanism."

2. **Mechanism side: LIF Low-pass Template and Monotonic Bandwidth Control (Lemma 3.2)**:

    - **Function**: Translates the temporal integration behavior of spiking neurons into a "low-pass filter with bandwidth monotonically controlled by $\beta$," allowing direct alignment with the data spectrum.
    - **Mechanism**: The LIF neuron ignoring the reset term is a first-order IIR with frequency response $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$. To eliminate global magnitude differences, a DC-normalized power template is defined as $\tilde H(\omega_k;\beta)=(1-\beta)^2/[(1-\beta)^2+2\beta(1-\cos\omega_k)]$. Lemma 3.2 proves: $\tilde H\in(0,1]$, $\tilde H(0;\beta)=1$, it is non-increasing w.r.t. $\omega_k$, and non-increasing w.r.t. $\beta$. The half-power point $\tilde H(\omega_c;\beta)=1/2$ defines the effective bandwidth $B_{\text{eff}}(\beta)=\omega_c$, making $\beta$ a clean "inverse bandwidth" knob.
    - **Design Motivation**: Giving the hyperparameter $\beta$ a clear physical meaning (bandwidth control) turns "tuning" into "bandwidth-spectrum alignment" rather than empirical trial-and-error.

3. **Alignment side: FMS Score and $\beta^\dagger$ Max Deviation Rule**:

    - **Function**: Provides a quantitative boundary $\beta^\dagger$ for the onset of over-low-passing, determined purely by the data spectrum and neural dynamics without relying on label accuracy.
    - **Mechanism**: The dot product of the template and data spectrum yields $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)\in[0,1]$, interpreted as the "quality of discriminative spectrum preserved by LIF under current $\beta$." Let $\tau=(1-\beta)^{-1}$; min-max normalization is applied to both $\log\tau$ and $\mathrm{FMS}_{\text{avg}}$ to get $(\phi_r,\psi_r)$. A reference diagonal $\hat L$ connects the endpoints. the point of maximum deviation $\beta^\dagger=\arg\max_r|\hat L(\phi_r)-\psi_r|$ is identified. Proposition 3.5 then defines three segments: under-filter ($\beta\to 0$, noise not suppressed), stability window ($0<\beta<\beta^\dagger$, where peak accuracy usually occurs), and over-low-pass ($\beta\geq\beta^\dagger$, where discriminative information is filtered out).
    - **Design Motivation**: Traditional $\beta$ tuning requires dataset-specific accuracy sweeps, which are expensive and lack mechanistic explanation. Defining $\beta^\dagger$ via frequency-domain geometric features makes "tuning" as simple as "drawing a line based on the spectrum," which is highly valuable for edge SNN deployment.

### Loss & Training
Standard SNN training with surrogate gradients is employed. A simple LeNet-style SpikingLeNet (~4.19M parameters) is used. The only additional step is pre-selecting $\beta$ for each dataset using the aforementioned method.

## Key Experimental Results

### Main Results: Accuracy on 4 mmWave Datasets (%, Mean of 3 seeds)

| Model | AOPHand | mmFiT | Pantomime | MMActivity | #Params (M) |
|------|---------|-------|-----------|------------|-------------|
| LeNet | 60.86 | 62.36 | 61.83 | 59.17 | 4.19 |
| VGG9 | 74.39 | 69.36 | 72.63 | 70.00 | 31.6 |
| ResNet50 | 72.54 | 71.84 | 73.90 | 61.67 | 23.5 |
| GRU | 67.52 | 14.11 | 75.45 | 47.50 | 0.075 |
| CNN-GRU | 61.98 | 67.80 | 72.77 | 65.00 | 0.46 |
| ViT | 21.39 | 36.40 | 42.16 | 65.83 | 2.18 |
| **SpikingLeNet** | **83.70** | **73.67** | **78.31** | **75.00** | 4.19 |

### Main Results: Theoretical Energy Consumption per Sample (μJ)

| Model | AOPHand | mmFiT | Pantomime | MMActivity |
|------|---------|-------|-----------|------------|
| LeNet | 251.08 | 251.08 | 251.10 | 251.08 |
| VGG16 | 6017.25 | 6017.26 | 6017.34 | 6017.24 |
| RNN | 7.35 | 7.35 | 7.36 | 7.35 |
| **SpikingLeNet** | **2.53** | **2.04** | **2.44** | **1.45** |

### Ablation Study

| Setting | Key Observation |
|------|---------|
| Explicit Low-pass + LeNet vs. SpikingLeNet | Adding filters improves LeNet but it still lags behind SpikingLeNet. Hard frequency truncation suppresses noise but also cuts high-freq discriminative info; LIF's "soft low-pass" is superior. |
| $\beta$ sweep (Fig. 4) | Accuracy increases then decreases with $\beta$; the peak $\beta^\ast < \beta^\dagger$. This validates the stability window prediction of Proposition 3.5. |
| $T$ sweep | Slight increases in $T$ lead to accuracy gains followed by saturation. Temporal steps primarily stabilize predictions; major improvements are driven by $\beta$. |
| t-SNE (Fig. 3) | SNN feature separation is significantly better than ANN. The suppression of high-frequency noise through frequency matching makes the feature space more discriminative. |
| Multi-platform Latency | ~4× slower than LeNet on Jetson GPUs, nearly matching on Darwin3. GPUs treat spikes as dense kernels; neuromorphic hardware is needed to realize sparsity benefits. |

### Key Findings
- SpikingLeNet, using the same LeNet backbone, outperforms the strongest ANN by an average of 6.22% across four datasets with identical parameter counts. This indicates that the performance difference stems from the temporal frequency bias provided by LIF, not model capacity.
- In terms of energy, SpikingLeNet is ~3.64× more efficient than the next best model (RNN) and two to three orders of magnitude lower than VGG/ResNet. This method is highly suitable for always-on edge sensing devices given proper hardware support.
- The optimal $\beta^\ast$ consistently appears before the theoretically determined $\beta^\dagger$, and $\mathrm{FMS}_{\text{avg}}$ correlates strongly with accuracy, proving the validity of the "frequency matching" hypothesis across all datasets.
- Current SNN latency bottlenecks on GPUs are primarily system-level artifacts. Moving workflows to neuromorphic chips like Darwin3 can realize the hardware advantages of event-driven and sparse computation.

## Highlights & Insights
- The study elevates the explanation of why SNNs excel at mmWave sensing from empirical observation to the level of frequency-domain mechanisms, supplemented by provable lemmas and propositions.
- By translating $\beta$ into "inverse bandwidth" and providing a graphical selection rule for $\beta^\dagger$, practitioners can achieve near-optimal $\beta$ without expensive sweeps. This mechanism-based tuning can be extended to other tasks with distinct frequency structures (EEG, IMU, radar tracking).
- The introduction of the discriminative spectrum $\mathrm{DI}_{\text{norm}}$ provides a lightweight and universal tool for "data spectrum profiling," which can be used to inspect whether the "frequency bias" of various networks matches the target data—a new perspective for model design and selection.

## Limitations & Future Work
- The framework is built entirely on the IIR linearization of "LIF + ignoring reset." Frequency analysis would need to be re-derived for neurons with hard resets, adaptive thresholds, or multi-state spiking dynamics.
- Experiments were conducted on small LeNet architectures; deep or multi-branch SNNs were not tested. It remains to be verified if "frequency matching" remains the primary bottleneck or if it is diluted by other hierarchical interactions in larger models.
- $\beta^\dagger$ is a geometric selection on a discrete candidate set and depends on sweep density; the optimal $\beta^\ast$ still requires training to confirm. The paper does not provide an analytical solution for the "best $\beta$ without any training samples."
- While latency issues are attributed to "system-level artifacts," practical deployment requires quantifiable hardware-algorithm co-design paths; a single case on Darwin3 may be insufficient.

## Related Work & Insights
- **vs. Fang et al. (2025)**: The authors advance the "LIF ≈ IIR low-pass" conclusion from a formulaic level to a "spectrum-data alignment" framework, providing the first criterion directly applicable to tuning.
- **vs. Arsalan et al. (2022/2023), Hu et al. (2025), etc.**: While previous works emphasized energy efficiency or engineering improvements, this paper explains *why* SNNs are suited for mmWave and offers reusable design principles.
- **vs. Classic Low-pass Preprocessing**: Traditional hard frequency truncation removes high-frequency discriminative info; the LIF's soft low-pass suppression of noise while retaining discriminative components provides experimental evidence that "frequency matching > hard truncation."

## Rating
- Novelty: ⭐⭐⭐⭐ Explains SNN advantages via frequency mechanisms and provides computable $\beta$ selection rules.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 common datasets and multi-platform testing, though limited to LeNet backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear lemmas and propositions with a complete mechanistic narrative.
- Value: ⭐⭐⭐⭐ Directly guides parameter tuning for edge SNN deployment and provides a template for "mechanism-data alignment" research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](../../ICLR2026/ai_safety/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](../../AAAI2026/ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)

</div>

<!-- RELATED:END -->
