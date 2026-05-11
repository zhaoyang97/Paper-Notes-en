---
title: >-
  [Paper Note] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation
description: >-
  [AAAI2026][Recommender Systems][sequential recommendation] This paper proposes WEARec, a model that employs Dynamic Frequency Filtering (DFF) to adaptively generate personalized frequency-domain filters conditioned on us…
tags:
  - "AAAI2026"
  - "Recommender Systems"
  - "sequential recommendation"
  - "frequency-domain filtering"
  - "wavelet transform"
  - "dynamic filter"
  - "personalized recommendation"
date: 2026-05-08
content_hash: 3d62fc2b52dfdc59
---

# Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation

**Conference**: AAAI2026
**arXiv**: [2511.07028](https://arxiv.org/abs/2511.07028)
**Code**: [GitHub](https://github.com/xhy963319431/WEARec)
**Area**: Sequential Recommendation / Frequency-Domain Signal Processing
**Keywords**: sequential recommendation, frequency-domain filtering, wavelet transform, dynamic filter, personalized recommendation

## TL;DR

This paper proposes WEARec, a model that employs Dynamic Frequency Filtering (DFF) to adaptively generate personalized frequency-domain filters conditioned on user context for capturing global preferences, and Wavelet Feature Enhancement (WFE) to compensate for the inability of global DFT to resolve short-term fluctuations. WEARec outperforms all 9 baselines on four datasets, achieving up to 11.4% improvement on long-sequence scenarios with 39–45% faster training speed.

## Background & Motivation

**Background**: Sequential recommendation captures dynamic user preferences by analyzing historical interaction sequences. Frequency-domain methods (FMLPRec, BSARec, SLIME4Rec, etc.) leverage the Fourier transform to decompose user behavior sequences into frequency components, effectively capturing periodic patterns that are difficult to identify in the time domain, and have emerged as an efficient alternative to self-attention.

**Limitation 1 — Static Filters Ignore Personalization**: Existing frequency-domain methods apply fixed static filters uniformly across all user sequences. However, experiments reveal that different users are driven by entirely different frequency components — some users' behaviors are dominated by low-frequency long-term preferences, while others are driven by high-frequency short-term interests (confirmed by counting the number of users independently driven by each frequency component).

**Limitation 2 — Global DFT Blurs Short-Term Dynamics**: DFT is a global frequency analysis tool that excels at capturing long-range dependencies but blurs non-stationary signals and short-term interest shifts. FMLPRec is essentially a low-pass filter, and while SLIME4Rec attempts hierarchical learning across frequency bands, it still favors low frequencies.

**Core Idea**: An MLP dynamically generates personalized filter parameters conditioned on the user sequence context (addressing Limitation 1), while a wavelet transform enhances high-frequency local features that DFT tends to obscure (addressing Limitation 2). Their combination achieves full-spectrum coverage of both global and local patterns.

## Method

### Overall Architecture

A Transformer-style encoder: Embedding layer (item embedding + positional embedding + LayerNorm + Dropout) → $L=2$ WEARec blocks (each comprising a DFF module + WFE module + Feature Integration + FFN + residual connections) → Prediction layer (softmax over candidate items). DFF and WFE together replace the conventional self-attention module.

### Key Designs

1. **Dynamic Frequency Filtering (DFF)**:
    - **Function**: Adaptively generates personalized frequency-domain filter parameters conditioned on each user's behavioral sequence context, enabling user-level frequency selection.
    - **Mechanism**: The embedding is first split into $k$ subspaces (multi-head projection). A 1D FFT is applied to each subspace to obtain the frequency-domain representation $\mathbb{F}_i^l \in \mathbb{C}^{M \times d/k}$. Simultaneously, a user context vector is computed as the temporal mean $\mathbf{c}^l = \frac{1}{N}\sum_{i=1}^{N}\mathbb{H}_i^l$, which is passed through two MLPs to produce a scaling factor $\Delta\mathbf{s}^l$ and a bias $\Delta\mathbf{b}^l$. These modulate a base filter: $\hat{\mathbb{W}}^l = \mathbb{W}^l \odot (1 + \Delta\mathbf{s}^l)$ and $\hat{\mathbf{b}}^l = \mathbf{b}^l + \Delta\mathbf{b}^l$. A frequency-domain linear transformation $\tilde{\mathbb{F}}_i^l = \mathbb{F}_i^l \odot \hat{\mathbb{W}}^l + \hat{\mathbf{b}}^l$ is then applied, followed by IFFT to return to the time domain.
    - **Design Motivation**: The scaling factor $\Delta\mathbf{s}$ controls the overall frequency response shape of the filter, while the bias $\Delta\mathbf{b}$ provides fine-grained adjustment at specific frequency bands. Together they implement context-conditioned filtering. Spectral visualizations confirm that WEARec covers the full frequency spectrum, whereas FMLPRec and SLIME4Rec are biased toward low frequencies.

2. **Wavelet Feature Enhancement (WFE)**:
    - **Function**: Decomposes the sequence into high- and low-frequency components via the Discrete Wavelet Transform (DWT), then adaptively enhances high-frequency detail signals using learnable matrices.
    - **Mechanism**: Haar wavelet decomposition is applied to each subspace along the item dimension: $\mathbb{A}_i^l, \mathbb{D}_i^l = \mathcal{W}(\mathbb{B}_i^l)$, where $\mathbb{A}$ denotes low-frequency approximation coefficients and $\mathbb{D}$ denotes high-frequency detail coefficients. The low-frequency component is left unmodified (preserving the primary sequence trend), while the high-frequency component is enhanced via a learnable matrix: $\tilde{\mathbb{D}}_i^l = \mathbb{D}_i^l \odot \mathbb{T}^l$. The sequence is then reconstructed via IDWT: $\mathbb{Y}_i^l = \mathcal{W}^{-1}(\mathbb{A}_i^l, \tilde{\mathbb{D}}_i^l)$.
    - **Design Motivation**: The wavelet transform possesses time-frequency localization (which DFT lacks), enabling precise identification of short-term non-stationary events. The Haar wavelet is selected for its simplicity, computational efficiency, and perfect reconstruction property. Only high-frequency components are enhanced while low-frequency components remain intact, avoiding disruption to the dominant trend information in the sequence.

3. **Feature Integration & Prediction**:
    - **Function**: Combines the global frequency-domain features from DFF with the local time-frequency features from WFE via a weighted fusion, yielding the final recommendation output.
    - **Mechanism**: Weighted fusion is performed as $\hat{\mathbb{H}}^l = \alpha \odot \mathbb{X}^l + (1-\alpha) \odot \mathbb{Y}^l$ (optimal at $\alpha \approx 0.3$), followed by residual connections, LayerNorm, and an FFN (with GELU activation). The prediction layer computes $\hat{\mathbf{y}} = \text{softmax}(\mathbf{h}^L (\mathbb{M})^\top)$.
    - **Design Motivation**: DFF captures global personalized frequency distributions while WFE enhances local non-stationary details; the two modules are complementary. The fact that $\alpha < 0.5$ indicates that local high-frequency enhancement should carry greater weight, as local feature modeling is precisely where frequency-domain methods fall short.

### Loss & Training

Standard cross-entropy loss $\mathcal{L}_{Rec} = -\sum_{i=1}^{|\mathcal{V}|} y_i \log(\hat{y}_i)$. Adam optimizer with learning rate $\in \{0.0005, 0.001\}$, embedding dimension 64, maximum sequence length $N=50$ (standard) / $N=200$ (long-sequence), batch size 256. Wavelet decomposition depth 1, $k \in \{1,2,4,8\}$. The absence of contrastive learning and self-attention substantially reduces computational overhead.

## Key Experimental Results

### Main Results

| Dataset | Metric | WEARec | BSARec | SLIME4Rec | DuoRec | Gain (vs. Best Baseline) |
|--------|------|--------|--------|-----------|--------|--------------|
| Beauty | HR@10 | **0.1041** | 0.1008 | 0.1006 | 0.0965 | +3.27% |
| Sports | HR@10 | **0.0631** | 0.0612 | 0.0611 | 0.0569 | +3.10% |
| LastFM | HR@10 | **0.0899** | 0.0807 | 0.0633 | 0.0624 | +11.40% |
| ML-1M | HR@10 | **0.2952** | 0.2757 | 0.2891 | 0.2704 | +2.10% |
| ML-1M | NDCG@10 | **0.1696** | 0.1568 | 0.1673 | 0.1530 | +1.37% |

### Ablation Study

| Configuration | Beauty HR@20 | Sports HR@20 | LastFM HR@20 | ML-1M HR@20 | Notes |
|------|-------------|-------------|-------------|-------------|------|
| WEARec (full) | **0.1391** | **0.0895** | **0.1202** | **0.4031** | Best |
| w/o WFE | Drop | Drop | Drop | Drop | Removing wavelet enhancement loses local information |
| w/o DFF | Largest drop | Largest drop | Largest drop | Largest drop | DFF is the core module |
| w/o multi-head projection | Drop | Drop | Drop | Drop | Removing subspace partitioning hurts performance |

Long-sequence efficiency ($N=200$): WEARec 66.46s/epoch vs. BSARec 109.26s vs. SLIME4Rec 120.43s, 39–45% faster.

### Key Findings

- DFF is the core module; its removal causes the largest performance drop, demonstrating that personalized frequency-domain filtering is more important than static filtering.
- The largest gain is observed on LastFM (+11.4%), whose average sequence length (48.2) is substantially longer than Beauty/Sports (~8), confirming that WEARec's advantage is more pronounced in long-sequence scenarios.
- The optimal fusion weight $\alpha \approx 0.3$ assigns 70% of the weight to WFE (local enhancement) and 30% to DFF (global filtering).
- Spectral visualizations show that WEARec covers the full frequency spectrum, while FMLPRec and SLIME4Rec are both biased toward low frequencies.
- WEARec's parameter count (426K on ML-1M) is higher than FMLPRec (324K) but its training time is lower than BSARec and SLIME4Rec.

## Highlights & Insights

- The dynamic filter design is elegant and concise: the mean of the user's sequence serves as context, and two small MLPs generate scaling/bias terms to modulate a base filter, achieving genuinely personalized frequency-domain processing at minimal parameter cost.
- The complementary combination of Fourier analysis (global frequency) and wavelets (time-frequency localization) has a solid theoretical foundation in signal processing; this paper represents the first successful introduction of this classical paradigm into recommender systems.
- Achieving state-of-the-art performance without self-attention or contrastive learning — relying purely on frequency-domain and time-frequency analysis — while training faster demonstrates the substantial potential of frequency-domain methods for sequential recommendation.

## Limitations & Future Work

- Only the Haar wavelet (the simplest wavelet basis) is employed; more sophisticated bases (Daubechies, Symlet, etc.) may capture finer time-frequency features.
- The fusion weight $\alpha$ is a globally fixed hyperparameter; it could be replaced by a sample-adaptive or layer-adaptive learnable weight.
- Item side information (text, images, etc.) is not incorporated; modeling relies solely on interaction sequences.
- Performance gains on Beauty and Sports are relatively modest (~3%), as the short sequences in these datasets offer less opportunity to benefit from personalization and local enhancement.

## Related Work & Insights

- **vs. FMLPRec**: FMLPRec is essentially a low-pass filter, whereas WEARec's DFF covers the full frequency spectrum; FMLPRec applies static filtering, while WEARec performs dynamic personalized filtering.
- **vs. BSARec**: BSARec uses frequency-domain components as an inductive bias for self-attention; WEARec fully replaces self-attention, achieving faster training.
- **vs. SLIME4Rec**: SLIME4Rec processes different frequency bands in a hierarchical manner but remains biased toward low frequencies; WEARec's WFE module explicitly enhances high-frequency signals.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of dynamic frequency-domain filtering and wavelet enhancement is novel, with a clear signal-processing motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four datasets, long-sequence analysis, spectral visualization, and hyperparameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ The signal processing background is particularly clearly presented, with complete methodological derivations.
- Value: ⭐⭐⭐⭐ Introduces an efficient and effective new paradigm for frequency-domain sequential recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)

</div>

<!-- RELATED:END -->
