---
title: >-
  [Paper Note] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings
description: >-
  [NeurIPS 2025 (ML4PS Workshop)][Physics][Self-supervised learning] This paper proposes AstroCo, a self-supervised encoder that introduces the Conformer architecture (attention + depthwise separable convolution + gating)…
tags:
  - "NeurIPS 2025 (ML4PS Workshop)"
  - "Physics"
  - "Self-supervised learning"
  - "Conformer"
  - "light curves"
  - "astronomical time series"
  - "masked reconstruction"
  - "few-shot classification"
date: 2026-05-08
content_hash: a7a7ec98fae2b3f8
---

# AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings

**Conference**: NeurIPS 2025 (ML4PS Workshop)  
**arXiv**: [2509.24134](https://arxiv.org/abs/2509.24134)  
**Code**: To be released  
**Area**: Physics  
**Keywords**: Self-supervised learning, Conformer, light curves, astronomical time series, masked reconstruction, few-shot classification  

---

## TL;DR

This paper proposes AstroCo, a self-supervised encoder that introduces the Conformer architecture (attention + depthwise separable convolution + gating) for irregular astronomical light curves. On the MACHO dataset, AstroCo reduces reconstruction error by 61–70% compared to Astromer v1/v2 and improves few-shot classification macro-F1 by approximately 7%.

---

## Background & Motivation

**Large-scale astronomical surveys**: Surveys such as MACHO and LSST generate massive volumes of unlabeled stellar light curves; manual annotation is prohibitively expensive, making label-efficient representation learning methods urgently needed.

**Self-supervised pioneer Astromer**: Donoso-Oliva et al. proposed Astromer v1/v2, representative works in the field that employ pure Transformer encoders with masked reconstruction pre-training.

**Limitations of pure attention**: Standard Transformers treat each time step uniformly, making it difficult to capture short-duration local phenomena in light curves (dips, flares, bursts) and lacking explicit control over noisy or temporally distant observations.

**Inspiration from Conformer**: The Conformer architecture (Gulati et al., 2020) from speech recognition demonstrated that a complementary "attention + convolution" design can simultaneously model global dependencies and local patterns.

**Gating mechanisms**: GLU (Dauphin et al., 2017) enables networks to adaptively select which local features to retain, which is particularly important for noisy astronomical observations.

**Core motivation**: Transfer the Conformer-style design to irregular astronomical time series to achieve better reconstruction accuracy and downstream classification performance with fewer training resources.

---

## Method

### Overall Architecture

```
Light curve {(t_i, m_i, σ_i)} 
    → Input embedding (concatenation fusion instead of additive fusion)
    → M Conformer-style encoder blocks
    → Inter-layer learnable scalar mixing
    → Masked mean pooling → sequence embedding
```

### Three Sub-layer Designs

#### (1) Multi-Head Self-Attention (MHSA)
- Standard multi-head self-attention + Dropout + residual connection + LayerNorm (post-norm).
- Responsible for modeling long-range dependencies between any two time steps in the light curve.

#### (2) Depthwise Separable Convolution Sub-layer (Depthwise Conv + GLU)
- LayerNorm → 1×1 pointwise projection expanding the dimension from $D$ to $2D$ → GLU gating (split into val and gate, $\text{val} \odot \sigma(\text{gate})$).
- Followed by depthwise separable Conv1D (kernel size $K=32$, selected after hyperparameter search over $K=5$–$128$), performing per-channel convolution along the time dimension to capture local temporal patterns.
- BatchNorm1d → SiLU activation → 1×1 pointwise projection back to $D$ → residual connection + LN.

#### (3) Gated Feed-Forward Network (Gated FFN)
- Expansion ratio $r=4$; the input is projected separately into two $rD$-dimensional vectors:
    - $\text{val} = \text{GeLU}(W_{\text{val}} X)$
    - $\text{gate} = \sigma(W_{\text{gate}} X)$
- Output: $Y = X + \text{Dropout}(W_{\text{out}}(\text{val} \odot \text{gate}))$, followed by LN.
- The gating mechanism adaptively determines which global features are retained.

### Input Embedding

- Photometric values $(m_i, \sigma_i)$ are projected to $d/2$ dimensions; timestamps $t_i$ are mapped to $d/2$ dimensions via sinusoidal embeddings.
- The two representations are **concatenated** and fused into $d$ dimensions through a linear layer + GeLU + LN, avoiding the scale mismatch problem of additive fusion.
- Masked/padding tokens replace original values with zeros to prevent information leakage.

### Inter-layer Scalar Mixing

- Inspired by BERT layer analysis (Tenney et al., 2019), learnable scalar weights $\{w_\ell\}$ (including the input layer) are normalized via softmax: $\alpha_\ell = \frac{\exp(w_\ell)}{\sum_j \exp(w_j)}$.
- The final representation is $\tilde{x}_i = \sum_{\ell=0}^{M} \alpha_\ell x_i^{(\ell)}$, allowing the model to adaptively fuse shallow and deep features.

### Loss & Training

- **Pre-training**: BERT-style masked reconstruction; 50% of positions are probe targets (30% masked, 10% randomly replaced, 10% unchanged), with RMSE computed over probe positions as the loss.
- **Downstream classification**: The encoder is frozen; only a linear head is trained using cross-entropy loss.

---

## Key Experimental Results

### Dataset

- **Pre-training**: MACHO survey R band, approximately 1.5 million single-band light curves, window length 200.
- **Classification**: MACHO LMC variable star catalog, 20,894 light curves, 6 classes (Cepheid I/II, eclipsing binary, long-period variable, RR Lyrae ab/c).

### Masked Reconstruction Results (Table 1)

| Model | RMSE ↓ | R² ↑ |
|-------|--------|------|
| Astromer v1 | 0.148 | — |
| Astromer v2 | 0.113 | 0.73 |
| **AstroCo-S (5.9M)** | **0.060** | **0.922** |
| **AstroCo-L (15.2M)** | **0.044** | **0.956** |

- AstroCo-S reduces RMSE by **59%** over v1 and **47%** over v2.
- AstroCo-L reduces RMSE by **70%** over v1 and **61%** over v2.

### Few-Shot Classification Results

- Under few-shot settings of 20/100/500 labels per class, AstroCo-S/L with a frozen encoder and linear head outperforms Astromer v1/v2 across all settings.
- Relative macro-F1 improvement of approximately **7%** (Figure 3).
- Results averaged over 3 folds × 3 seeds with stable variance.

### Key Findings

1. **Adding local convolution and gating** significantly improves the representation quality of pure attention-based encoders.
2. **Scalar mixing** outperforms fixed pooling strategies (e.g., using only the last layer).
3. **AstroCo-S (5.9M parameters, 11.6h, 4×A100)** already surpasses Astromer v1/v2 (5.4M parameters, 3 days, 4×A5000), with higher resource efficiency.
4. AstroCo-L (15.2M, 1.2 days, 4×H200) further advances the state-of-the-art.

---

## Highlights & Insights

- **Successful cross-domain transfer**: Applying the Conformer design from speech to irregular astronomical time series validates the generality of the "attention + local convolution + gating" combination.
- **High resource efficiency**: The smaller AstroCo-S surpasses prior baselines with substantially less compute.
- **Concatenation fusion** replaces additive fusion to avoid dimension/scale mismatches—a practically useful engineering improvement.
- **Scalar mixing** allows the model to adaptively leverage features from different depths, offering more flexibility than always using the last layer.
- The self-supervised → freeze → linear probe paradigm validates foundation model few-shot transfer capability in the astronomical domain.

---

## Limitations & Future Work

- **Workshop paper only**: Experimental scale and analytical depth are limited; only the MACHO single-survey dataset is used.
- **Single-band**: Multi-band fusion is not explored, whereas modern surveys (e.g., LSST) are inherently multi-band.
- **Limited downstream tasks**: Only variable star classification is evaluated; other important astronomical tasks such as anomaly detection and period estimation are not tested.
- **Insufficient ablation**: No isolated ablation experiments are provided for the gating, convolution, and scalar mixing components individually.
- **Limited interpretability**: The distribution of scalar mixing weights is not analyzed, and the behavior of the gating mechanism lacks visualization.
- **Not open-sourced**: Code and pre-trained weights are not yet publicly available, raising reproducibility concerns.

---

## Related Work & Insights

| Work | Core Idea | Relation to This Paper |
|------|-----------|------------------------|
| Astromer v1 (2023) | Transformer masked reconstruction pre-training for light curves | Direct baseline; AstroCo adds convolution and gating on top |
| Astromer v2 (2025) | Improved Transformer encoder | Stronger baseline; AstroCo still outperforms it significantly |
| Conformer (Gulati 2020) | Attention + convolution for speech recognition | Architectural inspiration |
| GLU (Dauphin 2017) | Gated linear units for language modeling | Core gating mechanism in the convolution sub-layer |
| BERT (Devlin 2019) | Masked language model pre-training | Source of the pre-training masking strategy |
| Scalar Mixing (Tenney 2019) | Inter-layer feature weighting | Inspiration for inter-layer aggregation |

---

## Rating

- **Novelty**: ⭐⭐⭐ — Transferring Conformer to astronomical time series is moderately novel, but the architectural components themselves (MHSA, GLU, depthwise conv, scalar mixing) are all combinations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ — Evaluation across masked reconstruction and few-shot classification is clear, but limited to a single dataset, lacks ablations, and covers few downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Within the workshop paper format, the structure is clear, the presentation is accurate, and architecture diagrams and equations are complete.
- **Value**: ⭐⭐⭐⭐ — Provides good reference value for the astronomical self-supervised learning community, validates the effectiveness of Conformer on irregular time series, and the resource efficiency advantage is noteworthy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)
- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)

</div>

<!-- RELATED:END -->
