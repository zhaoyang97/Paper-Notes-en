---
title: >-
  [Paper Note] Frequency Matching in Spiking Neural Networks for mmWave Sensing
description: >-
  [ICML 2026][AI Safety][LIF neuron] From a "mechanism-data alignment" perspective, this work proves that LIF spiking neurons are equivalent to a first-order IIR low-pass filter…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "LIF neuron"
  - "IIR low-pass filter"
  - "mmWave sensing"
  - "discriminative spectrum"
  - "neural dynamics-data alignment"
date: 2026-05-08
content_hash: 1b2dc3ac9e47f629
---

# Frequency Matching in Spiking Neural Networks for mmWave Sensing

**Conference**: ICML 2026  
**arXiv**: [2605.09983](https://arxiv.org/abs/2605.09983)  
**Code**: [GitHub](https://github.com/yudi-mars/Soul)  
**Area**: Edge Sensing / Spiking Neural Networks (SNN) / Wireless Sensing  
**Keywords**: LIF neuron, IIR low-pass filter, mmWave sensing, discriminative spectrum, neural dynamics-data alignment

## TL;DR
From a "mechanism-data alignment" perspective, this work proves that LIF spiking neurons are equivalent to a first-order IIR low-pass filter, and proposes setting the membrane decay coefficient $\beta$ according to the discriminative spectrum of mmWave signals. This enables SNNs to achieve an average of 6.22% higher accuracy and 3.64× lower theoretical energy consumption than ANNs on four standard mmWave datasets.

## Background & Motivation
**Background**: mmWave radar is an important sensor for edge-side posture, gesture, and activity recognition due to its privacy-friendliness, robustness to lighting, and penetration capability. Mainstream solutions use CNNs/Transformers and other ANNs, relying on deep stacking and manual preprocessing for robustness, but at the cost of high energy and latency.

**Limitations of Prior Work**: mmWave signals are inherently sparse, irregular, and heavily contaminated by high-frequency noise from multipath and phase jitter. ANNs lack built-in temporal filtering bias: either manual low-pass preprocessing is applied (which also removes useful high-frequency discriminative information), or deeper networks are used to fit the data, resulting in high energy and latency.

**Key Challenge**: Discriminative information is often distributed in the "low-to-mid frequency band," while noise is concentrated in the high-frequency band. Existing ANN/low-pass preprocessing cannot distinguish "useful high-frequency discriminative components" from "true high-frequency noise." Prior SNN work has shown energy efficiency advantages, but hyperparameter tuning is empirical, and there is no clear explanation of "when and why SNNs outperform ANNs."

**Goal**: From a signal processing perspective, answer two questions—(1) What is the mechanism behind SNNs' advantage on mmWave tasks? (2) How should the key hyperparameter, membrane decay coefficient $\beta$, be selected based on the data spectrum?

**Key Insight**: Linearize the discrete dynamics of the LIF neuron as a first-order IIR low-pass filter, directly quantify the overlap between its cutoff frequency and the dataset's discriminative spectrum, and thus turn "setting $\beta$" into a frequency-domain alignment problem.

**Core Idea**: Match the effective bandwidth $B_{\text{eff}}(\beta)$ of the LIF neuron to the discriminative spectrum $\Omega^\star$ of mmWave data—"frequency matching" is the fundamental mechanism for SNNs' superiority over ANNs in such tasks, and provides a physical criterion for selecting $\beta$.

## Method

### Overall Architecture
No new network structure is introduced; instead, a frequency-domain mechanism analysis and hyperparameter selection method is provided for "LIF neuron + LeNet-style SNN." The approach consists of three steps: (1) Use DFT along the temporal dimension to analyze each mmWave dataset, define a Fisher-style discriminative index $\mathrm{DI}(\omega_k)$ and normalize it to a probability distribution $\mathrm{DI}_{\text{norm}}$; (2) Express LIF as $u_{t+1}=\beta u_t+(1-\beta)I_t-v_{\text{th}}O_t$, ignore the reset term, yielding an equivalent first-order IIR filter $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$, define the DC-normalized power template $\tilde H(\omega_k;\beta)$ and half-power cutoff $B_{\text{eff}}(\beta)$; (3) Use the dot product $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)$ to measure mechanism-data alignment, then apply the "maximum deviation from reference diagonal" rule to identify the over-low-pass threshold $\beta^\dagger$, partitioning $\beta$ into "under-filter / stability window / over-low-pass" regions.

### Key Designs

1. **Data Side: Discriminative Spectrum $\mathrm{DI}_{\text{norm}}(\omega_k)$**:

    - **Function**: Objectively quantifies the "density of class-discriminative information at each frequency," serving as the "ground truth" for subsequent mechanism matching.
    - **Mechanism**: For each sample $\mathbf{X}_i\in\mathbb{R}^{L\times C\times H\times W}$, average over non-temporal dimensions to obtain a 1D sequence $\mathbf{s}_i\in\mathbb{R}^L$, perform sample-wise mean removal and one-sided DFT to get amplitude spectrum $A_i[k]$; estimate inter-class scatter $S_B[k]=\sum_c\pi_c(\mu_c[k]-\bar\mu[k])^2$ and intra-class scatter $S_W[k]=\sum_c\pi_c\,\mathrm{Var}_c[k]$ per frequency, define $\mathrm{DI}(\omega_k)=S_B[k]/(S_W[k]+\varepsilon)$ and normalize in the frequency domain.
    - **Design Motivation**: Directly applies Fisher-style statistics for linear separability, reflecting both "signal energy distribution" and "class separability," serving as a bridge between "data" and "mechanism."

2. **Mechanism Side: LIF Low-Pass Template and Monotonic Bandwidth Control (Lemma 3.2)**:

    - **Function**: Translates the temporal integration behavior of spiking neurons into "a low-pass filter with bandwidth monotonically controlled by $\beta$," enabling direct frequency-domain alignment with the data spectrum.
    - **Mechanism**: LIF without reset is a first-order IIR with frequency response $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$; to eliminate overall amplitude differences, define the DC-normalized power template $\tilde H(\omega_k;\beta)=(1-\beta)^2/[(1-\beta)^2+2\beta(1-\cos\omega_k)]$. Lemma 3.2 proves: $\tilde H\in(0,1]$, $\tilde H(0;\beta)=1$, non-increasing in both $\omega_k$ and $\beta$. The half-power point $\tilde H(\omega_c;\beta)=1/2$ defines the effective bandwidth $B_{\text{eff}}(\beta)=\omega_c$, making $\beta$ a clean "inverse bandwidth" knob.
    - **Design Motivation**: Assigns a clear physical meaning (bandwidth control) to the hyperparameter $\beta$, turning "tuning" into "bandwidth-data spectrum alignment" rather than empirical guesswork.

3. **Alignment Side: FMS Score and $\beta^\dagger$ Maximum Deviation Rule**:

    - **Function**: Provides a "starting point for over-low-pass" $\beta^\dagger$ that depends only on data spectrum and neural dynamics, not label accuracy, thus quantitatively partitioning the tunable range of $\beta$.
    - **Mechanism**: Take the inner product of the template and data spectrum to obtain $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)\in[0,1]$, interpreted as "the quality of discriminative spectrum retained by LIF at current $\beta$." Let $\tau=(1-\beta)^{-1}$, min-max normalize both $\log\tau$ and $\mathrm{FMS}_{\text{avg}}$ to $(\phi_r,\psi_r)$, connect endpoints to form reference diagonal $\hat L$, and select the point of maximum deviation $\beta^\dagger=\arg\max_r|\hat L(\phi_r)-\psi_r|$. Proposition 3.5 thus defines three regions: under-filter ($\beta\to 0$, noise not suppressed), stability window ($0<\beta<\beta^\dagger$, accuracy typically peaks here), over-low-pass ($\beta\geq\beta^\dagger$, discriminative information also suppressed).
    - **Design Motivation**: Traditional $\beta$ tuning requires dataset-specific accuracy sweeps, which are costly and lack mechanistic explanation; defining $\beta^\dagger$ via frequency-domain geometry turns "tuning" into "drawing a line on the spectrum," highly meaningful for edge SNN deployment.

### Loss & Training
Standard SNN training with surrogate gradients is used (details in the appendix), with a simple LeNet-style SpikingLeNet (≈4.19M parameters); the only extra step is to pre-select $\beta$ for each dataset as described above.

## Key Experimental Results

### Main Results: Accuracy (%) on 4 mmWave Datasets (mean of 3 seeds)

| Model | AOPHand | mmFiT | Pantomime | MMActivity | #Params (M) |
|-------|---------|-------|-----------|------------|-------------|
| LeNet | 60.86 | 62.36 | 61.83 | 59.17 | 4.19 |
| VGG9 | 74.39 | 69.36 | 72.63 | 70.00 | 31.6 |
| ResNet50 | 72.54 | 71.84 | 73.90 | 61.67 | 23.5 |
| GRU | 67.52 | 14.11 | 75.45 | 47.50 | 0.075 |
| CNN-GRU | 61.98 | 67.80 | 72.77 | 65.00 | 0.46 |
| ViT | 21.39 | 36.40 | 42.16 | 65.83 | 2.18 |
| **SpikingLeNet** | **83.70** | **73.67** | **78.31** | **75.00** | 4.19 |

### Main Results: Theoretical Energy per Sample (μJ)

| Model | AOPHand | mmFiT | Pantomime | MMActivity |
|-------|---------|-------|-----------|------------|
| LeNet | 251.08 | 251.08 | 251.10 | 251.08 |
| VGG16 | 6017.25 | 6017.26 | 6017.34 | 6017.24 |
| RNN | 7.35 | 7.35 | 7.36 | 7.35 |
| **SpikingLeNet** | **2.53** | **2.04** | **2.44** | **1.45** |

### Ablation Study & Diagnostics

| Setting | Key Observation | Note |
|---------|----------------|------|
| Explicit low-pass preprocessing + LeNet vs SpikingLeNet | Adding filter improves LeNet but still lags behind SpikingLeNet | Hard frequency truncation suppresses noise but also removes high-frequency discriminative information; LIF provides superior "soft low-pass" |
| $\beta$ sweep (Fig. 4) | Accuracy rises then falls with $\beta$, peak at $\beta^\ast<\beta^\dagger$ | Directly validates the stability window prediction of Proposition 3.5 |
| $T$ sweep | Slight increase in $T$ → accuracy improves then saturates | Moderate time steps mainly stabilize prediction; main driver is $\beta$ |
| t-SNE (Fig. 3) | SNN features show much clearer inter-class separation than ANN | Frequency matching suppresses high-frequency noise, making feature space more discriminative |
| Multi-platform latency | ~4× slower than LeNet on Jetson GPU, nearly matches on Darwin3 | Current GPUs run spikes as dense kernels; neuromorphic hardware is needed to realize sparsity advantage |

### Key Findings
- SpikingLeNet with the same LeNet backbone outperforms the strongest ANN by an average of 6.22% across four datasets, with identical parameter count—indicating the performance gain comes from the temporal-frequency bias of LIF, not model capacity.
- In terms of energy, SpikingLeNet is ~3.64× more efficient than the next best (RNN), and two to three orders of magnitude lower than VGG/ResNet; with hardware support, this approach is highly suitable for always-on edge sensing devices.
- The optimal $\beta^\ast$ always appears before the theoretically predicted $\beta^\dagger$, and $\mathrm{FMS}_{\text{avg}}$ is highly correlated with accuracy, confirming the "frequency matching" hypothesis on all four datasets.
- The apparent "slowness" of SNNs on GPUs is mainly a system-level artifact; deploying on neuromorphic chips like Darwin3 realizes the "event-driven + sparsity" hardware advantage.

## Highlights & Insights
- Elevates the explanation for "why SNNs outperform ANNs on mmWave" from empirical observation to frequency-domain mechanism, with provable lemmas and propositions, filling the explanatory gap in existing SNN-mmWave work without algorithmic novelty.
- Translates $\beta$ into "inverse bandwidth" and provides a graphical selection rule for $\beta^\dagger$, enabling practitioners to obtain near-optimal $\beta$ without expensive sweeps; this mechanism-based tuning method can be extended to other "LIF + frequency-structured" tasks (EEG, inertial sensing, radar tracking).
- Introducing the discriminative spectrum $\mathrm{DI}_{\text{norm}}$ as a "data spectrum profile" is a lightweight yet general tool, allowing inspection of whether a network's "frequency bias" matches the target data—a new perspective for model design/selection.

## Limitations & Future Work
- The framework is entirely based on the IIR linearization of "LIF + ignoring reset"; for hard reset, adaptive threshold, or multi-state spiking neurons, frequency-domain analysis needs to be redone.
- All experiments are conducted on small LeNet models; it remains to be verified on deeper/multi-branch SNNs whether "frequency matching" is still the key bottleneck or diluted by higher-level interactions.
- $\beta^\dagger$ is a geometric selection from a discrete candidate set, dependent on sweep density; the optimal $\beta^\ast$ still requires post-training determination, and the paper does not provide an analytic solution for "best $\beta$ without any training samples."
- The paper attributes latency issues to "system-level artifacts," but real-world deployment often requires quantifiable hardware-algorithm co-design paths; providing only a Darwin3 case is insufficient.

## Related Work & Insights
- **vs Fang et al. (2025)**: This work advances Fang et al.'s "LIF ≈ IIR low-pass" result from formulaic to a "spectrum-data alignment" framework, providing the first directly applicable criterion for hyperparameter tuning.
- **vs Arsalan et al. 2022/2023, Hu et al. 2025 and other SNN-mmWave works**: Previous studies emphasized energy efficiency or engineering improvements; this work explains "why SNNs suit mmWave" from a frequency-domain mechanism and offers reusable design principles.
- **vs Classical Low-Pass Preprocessing**: Traditional hard frequency truncation also removes high-frequency discriminative information; LIF's soft low-pass suppresses noise while retaining discriminative components, providing experimental evidence that "frequency matching > hard truncation."

## Rating
- Novelty: ⭐⭐⭐⭐ Explains SNN's advantage on mmWave from a frequency-domain mechanism and provides a computable $\beta$ selection rule; novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 standard mmWave datasets + multi-platform latency tests, but limited to LeNet backbone.
- Writing Quality: ⭐⭐⭐⭐ Lemmas and propositions are clear, mechanism narrative is complete.
- Value: ⭐⭐⭐⭐ Directly guides SNN deployment on edge devices, and provides a template for "mechanism-data alignment" research paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)

</div>

<!-- RELATED:END -->
