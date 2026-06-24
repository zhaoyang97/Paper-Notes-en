---
title: >-
  [Paper Note] TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation
description: >-
  [ICML 2025][Time Series][Unsupervised Domain Adaptation] This paper proposes TransPL, which discretizes time series patches into VQ codes and constructs class-channel-level transition matrices, leveraging Bayes' theorem to generate interpretable pseudo-labels in the target domain, achieving average improvements of 6.1% in accuracy and 4.9% in F1 score for time series unsupervised domain adaptation.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Unsupervised Domain Adaptation"
  - "Pseudo-Labeling"
  - "Vector Quantization"
  - "Transition Matrices"
  - "Optimal Transport"
date: 2026-05-08
content_hash: a8e600d0d7b0aeff
---

# TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation

**Conference**: ICML 2025  
**arXiv**: [2505.09955](https://arxiv.org/abs/2505.09955)  
**Code**: [Available](https://github.com/eai-lab/TransPL)  
**Area**: Time Series  
**Keywords**: Unsupervised Domain Adaptation, Pseudo-Labeling, Vector Quantization, Transition Matrices, Optimal Transport

## TL;DR

This paper proposes TransPL, which discretizes time series patches into VQ codes and constructs class-channel-level transition matrices, leveraging Bayes' theorem to generate interpretable pseudo-labels in the target domain, achieving average improvements of 6.1% in accuracy and 4.9% in F1 score for time series unsupervised domain adaptation.

## Background & Motivation

Time-series data often exhibits strong distribution discrepancies across different domains (e.g., different users, different devices), which primarily manifest at two levels: **temporal dynamic transitions** and **channel-level (sensor-level) shifts**. For instance, in human activity recognition, the gravity component of an accelerometer might remain stable across users, but gyroscope readings could shift drastically due to varying placement tightness.

Existing pseudo-labeling strategies (such as Softmax, NCP, SHOT, T2PL, etc.) are mostly derived from the image domain and suffer from two core problems:

**Ignoring temporal dynamics**: Treating time series as static data without modeling state transition patterns across the temporal dimension.

**Ignoring channel-level shifts**: Different sensors in multi-channel time series are affected by domain shifts to varying degrees, but existing methods fail to distinguish between them.

**Key Challenge**: Time series domain adaptation requires simultaneous modeling of temporal transitions and selective channel shifts, but existing methods treat this as a black-box operation, neither explicitly modeling it nor providing interpretability.

**Core Idea**: TransPL segments time series into patches, maps them to discrete codes via vector quantization (VQ), and constructs **class-level transition matrices** (modeling temporal patterns of each class in the source domain) and **channel-level transition matrices** (quantifying domain shifts of each channel). It then utilizes Bayes' theorem to generate weighted pseudo-labels. This not only achieves explicit joint temporal-channel modeling but also renders the pseudo-label generation process interpretable.

## Method

### Overall Architecture

TransPL consists of three stages:

**Stage A — Source Domain Training**: Utilizes labeled source domain data to train the encoder (Transformer), decoder, coarse-fine dual codebooks, and classifier. The time series $\mathbf{X} \in \mathbb{R}^{D \times T}$ is segmented into $N = \lfloor T/m \rfloor$ patches, encoded, and mapped to discrete codes via VQ.

**Stage B — Transition Matrix Construction**: Freezes model parameters, infers coarse code sequences from both source and target domains, and constructs three sets of transition matrices:
- Source domain class-level TM: $\mathbf{P}_{\text{cl}}^{\mathcal{S}} \in \mathbb{R}^{K \times D \times n_c \times n_c}$
- Source domain channel-level TM: $\mathbf{P}_{\text{ch}}^{\mathcal{S}} \in \mathbb{R}^{D \times n_c \times n_c}$
- Target domain channel-level TM: $\mathbf{P}_{\text{ch}}^{\mathcal{T}} \in \mathbb{R}^{D \times n_c \times n_c}$

**Stage C — Pseudo-Label Generation**: Computes channel-level class-conditional likelihood based on class-level TMs, combines it with channel alignment weights and prior distributions, and generates final pseudo-labels via Bayes' theorem.

### Key Designs

1. **Coarse-Fine Codebook**: Inspired by classic additive decomposition of time series (trend + residual), a two-level codebook structure is designed. The coarse codebook $\mathcal{C}_c$ ($n_c = 8$ codes) captures short-term trends of patches, while the fine codebook $\mathcal{C}_f$ ($n_f = 64$ codes) captures residual details. The quantization process is:
    $\tilde{c} = \arg\min_c \|\ell_2(\mathbf{z}) - \ell_2(\mathbf{e}_c)\|_2^2, \quad \mathbf{e}_c \in \mathcal{C}_c$
    $\tilde{f} = \arg\min_f \|\ell_2(\mathbf{z}) - \ell_2(\mathbf{e}_{\tilde{c}}) - \ell_2(\mathbf{e}_f)\|_2^2, \quad \mathbf{e}_f \in \mathcal{C}_f$
   Permutation Entropy (PE) analysis validates that coarse codes indeed capture simpler global trends (low PE), whereas fine codes encode more complex residual patterns (high PE). The key advantage is that $n_c \ll n_f$ makes the transition matrix computation feasible and avoids dead codes.

2. **VQ Code Transition Matrices**: Treating the coarse code sequence as a discrete Markov chain, the transition probability is:
    $p(s_{t+1} = \mathbf{e}_j | s_t = \mathbf{e}_i) = \frac{\text{count}(\mathbf{e}_i, \mathbf{e}_j)}{\text{count}(\mathbf{e}_i)}$
   **Class-level TM** is calculated from labeled source domain data, grouped by class and channel, and is used to compute the class-conditional likelihood of target sequences given class $k$ (analogous to maximum likelihood estimation in HMMs). **Channel-level TM** is constructed from source and target domains separately without class distinction, used for subsequent channel alignment scoring.

3. **Channel Alignment and Bayesian Pseudo-Labeling**: The core formula for pseudo-labeling is the aggregation of weighted channel-level class posteriors:
    $\hat{y}_k = \frac{1}{D} \sum_{d=1}^{D} w_d \frac{p(\mathbf{X}^d | y=k) \, p(k)}{\sum_{c=1}^{K} p(\mathbf{X}^d | y=c) \, p(c)}$
   The channel alignment score $w_d$ is computed via optimal transport: first, calculate the Earth Mover's Distance between the rows of the source and target channel TMs (with the cost matrix being the cosine distance between codes), then convert it to an alignment score via an RBF kernel:
    $w_d = \exp\left(-\left(\frac{1}{n_c}\sum_{i=1}^{n_c}\langle\gamma_i^*, \mathbf{M}\rangle\right)^2 / \sigma^2\right)$
   The intuition is: channels with smaller shifts receive higher weights. Class-conditional log-likelihood:
    $\log p(\mathbf{X}^d | y=k) = \frac{1}{N}\sum_{t=1}^{N-1}\log p(s_{t+1} | s_t, y=k)$

### Loss & Training

**Source Domain Training Loss**:
$$\mathcal{L}_{\text{src}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{VQ}}, \quad \mathcal{L}_{\text{VQ}} = \mathcal{L}_{\text{code}} + \mathcal{L}_{\text{rec}}$$

- $\mathcal{L}_{\text{code}}$: Standard VQ loss with stop-gradient and commitment loss ($\beta=0.25$), optimizing both coarse and fine codes simultaneously.
- $\mathcal{L}_{\text{rec}}$: MSE of the decoder reconstructing the original time series from $\mathbf{e}_c + \mathbf{e}_f$.
- $\mathcal{L}_{\text{ce}}$: Classification cross-entropy of the [CLS] token.

**Target Domain Fine-tuning Loss**:
$$\mathcal{L}_{\text{trg}} = \lambda_1 \mathcal{L}_{\text{ce}} + \lambda_2 \mathcal{L}_{\text{VQ}}$$

During fine-tuning, each mini-batch selects pseudo-labeled samples within the top $r_{\text{top}}$ confidence ratio. $\lambda_1, \lambda_2$ can be automatically adjusted using learnable weights from multi-task learning.

**Weakly Supervised Extension**: When the target domain label distribution is known, it can be directly incorporated as a prior $p(k)$ into the Bayesian formula, using the log-prior $\log p(k) / \tau$ and adjusted by temperature $\tau$.

## Key Experimental Results

### Main Results

| Dataset | Metric | TransPL | Prev. SOTA (SHOT) | Gain |
|--------|------|---------|-----------------|------|
| UCIHAR | Acc | **69.0** | 67.8 | +1.2 |
| UCIHAR | MF1 | **64.9** | 64.3 | +0.6 |
| WISDM | Acc | **64.0** | 62.2 | +1.8 |
| WISDM | MF1 | **56.2** | 54.6 | +1.6 |
| HHAR | Acc | **68.4** | 64.8 | +3.6 |
| HHAR | MF1 | **65.3** | 63.2 | +2.1 |
| PTB | Acc | **67.2** | 61.6 | +5.6 |
| PTB | MF1 | **74.0** | 66.9 | +7.1 |

In terms of pseudo-label accuracy, TransPL achieves average improvements of **6.1% accuracy** and **4.9% F1** compared to all baselines. With weak supervision, the gains are further expanded to 10.7% and 5.2%.

### Ablation Study

| Configuration | UCIHAR Acc | HHAR Acc | PTB Acc | Note |
|------|-----------|---------|---------|------|
| W/o CA W/o WS | 68.0 | 63.2 | 68.3 | Baseline |
| +CA | 69.0 | 68.4 | 67.2 | Channel alignment is effective |
| +WS | 68.6 | 62.3 | 72.2 | Weak supervision is effective |
| +CA +WS | **71.2** | **70.4** | **72.4** | Joint usage is optimal |

Codebook configuration ablation (HHAR): Coarse 8 + Fine 64 is optimal (PL Acc 68.4%, 0% dead codes), while a single large codebook (128) suffers from a high dead code rate up to 66.8%.

### Key Findings

- TransPL consistently outperforms all DA methods and pseudo-labeling methods across 4 datasets, achieving average improvements of 11.4% Acc and 12.2% MF1 compared to the no-adaptation baseline.
- Compared to directly modeling coarse code sequences with discriminative models (1D-CNN, LSTM, GRU), TransPL's generative transition matrix approach performs significantly better, demonstrating that TMs capture cross-domain invariant temporal dynamics more effectively.
- Weak supervision in CoDATS degrades performance on several datasets (e.g., dropping by 5.3% on PTB), whereas TransPL achieves consistent improvements across all datasets through Bayesian priors.
- Channel alignment analysis shows that prototype-distance-based methods fail to accurately measure channel shift, whereas the optimal transport-based method provides well-calibrated distances.

## Highlights & Insights

1. **A new paradigm of discrete modeling for temporal joint distribution**: Cleverly converting continuous time series $\to$ VQ codes $\to$ Markov transition matrices, which transforms the high-dimensional continuous density estimation problem into a computable discrete transition probability calculation.
2. **Interpretability**: Visualization of class-conditional likelihoods directly displays differences in temporal patterns for various classes, removing the black-box nature of pseudo-label generation, which is of great value for physical deployment.
3. **Elegant integration of weak supervision**: TransPL naturally incorporates label distribution information as Bayesian priors, which is mathematically better grounded than the ad-hoc KL divergence minimization in CoDATS.
4. **Optimal transport for measuring channel shift**: Utilizing semantic distance between codes as the cost matrix for OT captures relationship similarity among semantic transitions much better than simple Euclidean distance.

## Limitations & Future Work

1. **Equal collection of channel importance**: Present methods downweight channels with large shifts; however, if such a channel happens to contain critical classification information, this might be counterproductive. Future work can combine channel importance metrics.
2. **Lack of mathematical constraints on coarse-fine division**: Although experiments verify that coarse codes capture trends while fine codes capture residuals, there is no explicit regularization to enforce this hierarchical relationship.
3. **Limitations of the Markov assumption**: Relying on first-order Markov properties may fail to capture longer-range temporal dependencies.
4. **Fixed codebook size**: The same $n_c=8, n_f=64$ is used for all datasets, without exploring the feasibility of adaptive codebook sizes.
5. **Label space assumption**: Assumes identical label spaces for source and target domains, without considering scenarios with partially overlapping labels.

## Related Work & Insights

- **New Application of VQ-VAE in Time Series**: While VQ was previously used mainly for generation and self-supervised learning, TransPL is the first to employ it for UDA, inspiring the potential of discrete representations in more downstream tasks.
- **Connection with HMMs**: The class-conditional likelihood computation is essentially a simplified version of HMM forward probability computation (first-order Markov). Introducing a more complete HMM mechanism could be considered.
- **SSSS-TSA**, concurrent work, also focuses on channel-level shifts but employs self-attention mechanisms for channel selection, complementing TransPL's OT method.
- **Scalability to other modalities**: The concept of transition matrices can be extended to domain adaptation in other sequential data (e.g., NLP, audio).

## Rating

- Novelty: ⭐⭐⭐⭐ Using VQ code transition matrices for time-series UDA pseudo-labeling is a completely fresh perspective. The coarse-fine codebook and OT channel-alignment designs are creative, though the overall design leans heavily on combining existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, wide comparison with multiple baselines, thorough ablation studies, and intuitive interpretability analysis. However, the size and diversity of datasets are relatively small (mostly sensor data).
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured paper with rigorous methodology derivation, intuitive charts, standardized notation, and convincing interpretability visualizations.
- Value: ⭐⭐⭐⭐ Establishes a new interpretable pseudo-labeling paradigm for time-series domain adaptation. The weak supervision extension holds high practical value, though generic scalability merits validation on larger-scale datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport](../../CVPR2026/time_series/towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICML 2025\] Customizing the Inductive Biases of Softmax Attention using Structured Matrices](customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)
- [\[ICML 2025\] Causal Discovery from Conditionally Stationary Time Series](causal_discovery_from_conditionally_stationary_time_series.md)

</div>

<!-- RELATED:END -->
