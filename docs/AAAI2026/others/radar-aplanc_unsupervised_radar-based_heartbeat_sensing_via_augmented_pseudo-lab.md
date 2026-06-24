---
title: >-
  [Paper Note] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast
description: >-
  [AAAI 2026][radar heartbeat sensing] This paper proposes Radar-APLANC, the first unsupervised learning framework for radar-based heartbeat sensing. Through a noise contrastive triplet (NCT) loss and an augmented pseudo-label generator, it achieves two-stage unsupervised training without requiring expensive physiological signal annotations, attaining performance approaching supervised methods.
tags:
  - "AAAI 2026"
  - "radar heartbeat sensing"
  - "unsupervised learning"
  - "pseudo-label"
  - "noise contrastive learning"
  - "FMCW radar"
date: 2026-05-08
content_hash: 80fd8be72e8e4252
---

# Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast

**Conference**: AAAI 2026
**arXiv**: [2511.08071](https://arxiv.org/abs/2511.08071)  
**Code**: [https://github.com/RadarHRSensing/Radar-APLANC](https://github.com/RadarHRSensing/Radar-APLANC)  
**Area**: Other
**Keywords**: radar heartbeat sensing, unsupervised learning, pseudo-label, noise contrastive learning, FMCW radar

## TL;DR

This paper proposes Radar-APLANC, the first unsupervised learning framework for radar-based heartbeat sensing. Through a noise contrastive triplet (NCT) loss and an augmented pseudo-label generator, it achieves two-stage unsupervised training without requiring expensive physiological signal annotations, attaining performance approaching supervised methods.

## Background & Motivation

### Value and Challenges of Radar-based Heartbeat Sensing

FMCW radar enables contactless heartbeat sensing by detecting sub-millimeter (0.1–0.5 mm) chest wall displacements, offering unique advantages in privacy preservation, environmental robustness, and continuous monitoring. However:

**Noise sensitivity of traditional methods**: Phase extraction and unwrapping-based approaches suffer severe performance degradation under motion artifacts, multipath interference, and low SNR conditions, with phase wrapping ambiguity and noise sensitivity being fundamental limitations.

**Annotation bottleneck for supervised methods**: Deep learning methods (e.g., Equipleth RF, VitaNet, CardiacMamba) offer better noise robustness but require large-scale, high-quality physiological signal annotations (e.g., synchronized PPG signals), whose collection cost is prohibitive and limits training data scalability.

**Non-transferability of video-domain methods**: Unsupervised physiological monitoring methods from the video domain (e.g., contrastive learning paradigms) exist, but radar heartbeat signals have lower SNR, a different representation (chest wall motion vs. facial color changes), and conventional positive/negative sample construction strategies fail under strong noise.

### Core Insight

Radar range matrices inherently contain a contrastive structure between "heartbeat range bins" and "noise range bins"—this intrinsic signal-noise separation can be exploited to construct positive and negative samples without external annotations.

## Method

### Overall Architecture

Radar-APLANC is a two-stage unsupervised framework:
- **Stage 1**: Pseudo-labels are generated using conventional radar methods; pre-training is performed with the noise contrastive triplet (NCT) loss.
- **Stage 2**: An augmented pseudo-label generator is introduced to improve pseudo-label quality via quality assessment and adaptive noise-aware label selection, enabling further fine-tuning.

### Preliminaries

**Range matrix acquisition**: An FMCW radar transmits a linearly frequency-modulated signal $s(t)$; the received reflected signal $u(t)$ is IQ-demodulated to obtain the intermediate frequency signal $m(t)$. FFT is applied to each chirp's IF signal to yield the range profile $M_n[f]$, and $N$ chirps are concatenated to form the range matrix $M \in \mathbb{R}^{N \times D}$.

**Basic heartbeat sensing**: (1) Select the range bin $d^*$ with the maximum power ratio (human body location); (2) compute the phase signal of that bin; (3) apply phase unwrapping; (4) apply 0.8–3.0 Hz bandpass filtering to obtain the heartbeat signal $\Phi(\cdot) \in \mathbb{R}^N$.

### Key Designs

#### 1. **Noise Contrastive Triplet Loss (NCT Loss) — Stage 1**

**Mechanism**: The natural contrast between heartbeat bins and noise bins in radar range matrices is exploited to construct self-supervised learning signals.

- **Pseudo-label generation**: Conventional radar methods extract heartbeat signals from the central range bin $d^*$; random temporal sampling and power spectral density (PSD) transformation yield the pseudo-label set $S_{PL}$.
- **Positive sample construction**: Heartbeat matrices $M(\cdot, d^* \pm \Delta d)$ from bins within a window around the central bin are fed into a heartbeat extractor, producing predicted heartbeat signals $p(\cdot)$; PSD transformation yields positive sample set $S_P$.
- **Negative sample construction**: Windows from randomly selected non-central bins $d'$ are treated as noise matrices and fed into a noise extractor to produce noise signals $q(\cdot)$; PSD transformation yields negative sample set $S_N$.

**NCT Loss**:
$$\mathcal{L}_{NCT} = \underbrace{\frac{1}{K^2}\sum_{i,j}\|S_{PL}[i] - S_P[j]\|^2}_{\text{positive term: pull heartbeat toward pseudo-label}} + \underbrace{(-\frac{1}{K^2}\sum_{i,j}\|S_P[i] - S_N[j]\|^2)}_{\text{negative term: push heartbeat away from noise}}$$

**Design Motivation**: All traditional signal processing methods can be regarded as a form of pseudo-label generator; despite being noisy, they still carry heartbeat information. Range bins at non-body positions predominantly contain background noise, naturally serving as negative samples. PSD transformation enables more stable frequency-domain comparison.

#### 2. **Augmented Pseudo-Label Generator — Stage 2**

**Mechanism**: The Stage 1 pre-trained model is used to assess and select higher-quality pseudo-labels.

The module consists of two sub-components:

**Quality assessment module**:
- Conventional heartbeat signals $\{\Phi_1, \ldots, \Phi_{2\Delta d+1}\}$ are extracted from each of the $2\Delta d + 1$ range bins within the heartbeat window.
- Noise distance $X_i = D(\Phi_i, q)$: distance from the pre-trained noise signal; larger values indicate better signal quality.
- Heartbeat distance $Y_i = D(\Phi_i, p)$: distance from the pre-trained heartbeat signal; smaller values indicate better signal quality.
- The distance metric is the mean absolute error between the estimated heart rates of two signals.

**Decision module** (adaptive noise-aware label selection):
- Ideal case: $\arg\max_i X_i = \arg\min_i Y_i$ (maximum noise distance coincides with minimum heartbeat distance); the corresponding signal is directly selected.
- Conflict case: $\arg\max_i X_i \neq \arg\min_i Y_i$; the noise distance of the signal $\Phi_{\arg\min Y_i}$ (minimum heartbeat distance) is compared against the noise distance of the pre-trained heartbeat signal $D(p, q)$.
    - If greater: select $\Phi_{\arg\min Y_i}$ (conventional method is preferred).
    - Otherwise: select the pre-trained heartbeat signal $p$ (deep learning method is preferred).

### Loss & Training

- Both Stage 1 and Stage 2 use the NCT Loss; the only difference is the source of pseudo-labels.
- Both stages train the heartbeat extractor and the noise extractor.
- Optimizer: AdamW, learning rate 1e-4, trained for 200 epochs.
- Evaluation uses 10-second windows.

## Key Experimental Results

### Main Results

Two datasets are used: Equipleth (public, 91 subjects, 550 recordings) and RHB (self-collected, 80 subjects, 240 recordings).

| Method | Type | Equipleth MAE↓ | Equipleth r↑ | RHB MAE↓ | RHB r↑ |
|------|------|---------------|-------------|---------|--------|
| FFT-based RF | Traditional | 13.51 | 0.24 | 12.25 | 0.26 |
| Equipleth RF | Supervised | **2.18** | **0.89** | 3.19 | 0.82 |
| VitaNet | Supervised | 3.14 | 0.77 | 5.28 | 0.66 |
| mmFormer | Supervised | 6.50 | 0.52 | 8.89 | 0.28 |
| **Radar-APLANC** | **Unsupervised** | 3.95 | 0.64 | 3.92 | 0.77 |

Cross-dataset evaluation (generalization):

| Method | RHB→Equipleth MAE | Equipleth→RHB MAE |
|------|-------------------|-------------------|
| Equipleth RF | 4.53 (+107.8%) | 2.68 |
| VitaNet | 7.43 (+136.6%) | 2.38 |
| **Radar-APLANC** | **4.10 (+3.8%)** | **3.52** |

### Ablation Study

| Stage 1 Config | Stage 2 Config | MAE↓ | RMSE↓ | r↑ |
|-------------|-------------|------|-------|-----|
| Noise matrix only | — | 34.48 | 38.34 | 0.01 |
| Pseudo-label only | — | 8.94 | 15.88 | 0.30 |
| Noise + pseudo-label | — | 4.40 | 9.89 | 0.63 |
| Noise + pseudo-label | Augmented pseudo-label only | 7.42 | 13.61 | 0.38 |
| Noise + pseudo-label | Augmented pseudo-label + noise | **3.95** | **9.72** | **0.64** |

Augmented pseudo-label generator ablation:

| Pre-trained heartbeat | Conventional heartbeat | Noise signal | MAE↓ |
|-----------|---------|---------|------|
| ✓ | — | — | 4.56 |
| ✓ | ✓ | — | 8.75 |
| — | ✓ | ✓ | 14.48 |
| ✓ | ✓ | ✓ | **3.95** |

### Key Findings

1. **Noise matrix is critical**: Using pseudo-labels alone yields MAE = 8.94; adding noise contrast reduces it to 4.40, a reduction of more than half. This validates the effectiveness of leveraging the intrinsic noise structure of radar range matrices.
2. **Two stages are complementary**: Stage 1 reduces MAE from 8.94 to 4.40; Stage 2 further reduces it to 3.95. Augmented pseudo-labels require all three signal types to function optimally.
3. **Cross-dataset stability**: The unsupervised method exhibits an MAE variation of only 0.4 bpm across datasets, whereas supervised methods vary by over 100%.
4. **Skin-tone fairness**: Performance disparity between light and dark skin tones (fairness metric) is substantially smaller for radar methods than for RGB methods; the unsupervised radar method achieves fairness on par with supervised radar methods.

## Highlights & Insights

1. **First unsupervised radar heartbeat sensing framework**: Fills an important gap by simultaneously addressing the annotation bottleneck and preserving radar's privacy and robustness advantages.
2. **Paradigm shift: noise as a resource**: Conventionally regarded as something to be eliminated, noise is repurposed here as a source of negative samples for contrastive learning, turning a liability into an asset.
3. **Two-stage progressive refinement**: Pre-training with coarse pseudo-labels followed by fine-tuning with refined pseudo-labels constitutes a generalizable bootstrapping strategy.
4. **Practical skin-tone fairness**: Against the backdrop of growing attention to AI fairness, the unsupervised radar approach demonstrates more equitable sensing performance than RGB-based methods.
5. **New dataset contribution**: The RHB dataset (80 subjects) will be open-sourced, facilitating community research.

## Limitations & Future Work

1. An MAE gap of approximately 1.8 bpm remains compared to the best supervised method; pseudo-label quality is the bottleneck.
2. Evaluation is limited to seated scenarios at 0.5–1 m distance; motion interference and longer-range settings are untested.
3. Only heart rate estimation is validated; extension to more complex physiological indicators such as respiratory rate or heart rate variability has not been explored.
4. The decision rules in Stage 2's augmented pseudo-label generator are heuristic-based and could be replaced by a more end-to-end approach.
5. No cross-modal comparison experiments with unsupervised video-domain methods are conducted.

## Related Work & Insights

- **Unsupervised rPPG from video** (Gideon 2021, Sun 2022) provides the conceptual basis for applying contrastive learning to physiological signals, but their spatial-temporal positive/negative sample construction is not applicable to radar data.
- **Equipleth** (Vilesov 2022) provides a radar-video multimodal dataset and supervised baselines.
- **Self-supervised radar methods** (Song 2022, Zhang 2025) explore self-supervised learning for radar but still require annotated fine-tuning and thus do not constitute truly unsupervised approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first unsupervised radar heartbeat framework; noise contrastive idea is original)
- Experimental Thoroughness: ⭐⭐⭐⭐ (two datasets, cross-dataset evaluation, comprehensive fairness analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, well-motivated)
- Value: ⭐⭐⭐⭐⭐ (addresses a practical pain point; new dataset and code are open-sourced)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RaUF: Learning the Spatial Uncertainty Field of Radar](../../CVPR2026/others/rauf_learning_the_spatial_uncertainty_field_of_radar.md)
- [\[ICLR 2026\] RADAR: Learning to Route with Asymmetry-aware Distance Representations](../../ICLR2026/others/radar_learning_to_route_with_asymmetry-aware_distance_representations.md)
- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](../../NeurIPS2025/others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)

</div>

<!-- RELATED:END -->
