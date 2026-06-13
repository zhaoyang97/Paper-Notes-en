---
title: >-
  [Paper Note] Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling
description: >-
  [NeurIPS 2025][Medical Imaging][EEG] This paper extends CMMN (Convolutional Monge Mapping Normalization) by proposing two strategies — channel-averaged PSD with $\ell_1$-normalized barycenter and subject-to-subject match…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "EEG"
  - "Domain Adaptation"
  - "Optimal Transport"
  - "Convolutional Monge Mapping"
  - "Independent Component Classification"
date: 2026-05-08
content_hash: d4e31ee2b24c2c09
---

# Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling

**Conference**: NeurIPS 2025
**arXiv**: [2509.01721](https://arxiv.org/abs/2509.01721)  
**Code**: [https://github.com/cniel-ud/ICWaves](https://github.com/cniel-ud/ICWaves)  
**Area**: EEG Signal Processing / Domain Adaptation
**Keywords**: EEG, Domain Adaptation, Optimal Transport, Convolutional Monge Mapping, Independent Component Classification

## TL;DR
This paper extends CMMN (Convolutional Monge Mapping Normalization) by proposing two strategies — channel-averaged PSD with $\ell_1$-normalized barycenter and subject-to-subject matching — to generate a single time-domain filter for domain adaptation across EEG datasets with differing channel counts. On independent component (IC) brain/non-brain classification, the F1 score improves from 0.77 to 0.84, surpassing ICLabel (0.88→0.91).

## Background & Motivation

**Background**: EEG recordings capture rich neural activity information and are widely used in the diagnosis of epilepsy and psychiatric disorders. ICA combined with automatic IC labeling is the mainstream approach for artifact removal, with ICLabel being the most popular IC classifier.

**Limitations of Prior Work**: Different EEG acquisition systems (electrodes, amplifiers, analog/digital filters, power line interference) introduce substantial spectral variability — e.g., US recordings contain 60 Hz line noise while European recordings contain 50 Hz noise. Such domain shift severely degrades cross-dataset IC classification performance. Furthermore, datasets differ in channel count (134–235 vs. 64), and the original CMMN computes an independent filter per channel, making it incompatible across different channel configurations.

**Key Challenge**: The original CMMN designs a separate filter for each channel; however, ICs are linear mixtures of channels — applying different filters to different channels alters IC characteristics. More fundamentally, when the source and target domains have different numbers of channels, the original CMMN cannot be applied at all.

**Goal**: Design a single-filter CMMN variant that (a) accommodates EEG datasets with different channel counts and (b) preserves IC characteristics (since all channels share the same filter).

**Key Insight**: Replace per-channel PSD with channel-averaged PSD to produce one common filter per subject. Combine with $\ell_1$ normalization to eliminate signal amplitude discrepancies caused by impedance and electrode differences, enabling spectral shape alignment.

**Core Idea**: A CMMN filter based on channel-averaged PSD and an $\ell_1$-normalized barycenter achieves EEG domain adaptation across datasets with different channel counts and acquisition systems.

## Method

### Overall Architecture
**Input**: Source-domain EEG (training set, $I$ subjects, each with $C^S$ channels) + target-domain EEG (test set, $C^T$ channels, $C^S \neq C^T$). **Processing**: A single CMMN normalization filter $h[n]$ is computed for each target subject and applied uniformly across all channels. **Output**: Spectrally aligned target EEG fed directly into a classifier trained on the source domain.

### Key Designs

1. **Channel-Averaged PSD Computation (Step 1)**:

    - Function: Compress multi-channel EEG spectral information into a single PSD.
    - Mechanism: For each subject, the PSD $\mathbf{p}_c$ is computed per channel using Welch's method, then averaged across channels: $\bar{\mathbf{p}} = \frac{1}{C} \sum_{c=1}^{C} \mathbf{p}_c$. This yields a PSD vector of identical dimensionality regardless of the channel count in the source or target domain.
    - Design Motivation: The original CMMN designs an independent filter per channel and cannot be applied when channel counts differ. Channel averaging enables a single filter applicable to any number of channels without differentially affecting individual ICs.

2. **$\ell_1$-Normalized Barycenter (Step 2a)**:

    - Function: Compute a reference spectrum from source-domain subjects to eliminate amplitude bias.
    - Mechanism: The channel-averaged PSD of each source subject is $\ell_1$-normalized as $\tilde{\mathbf{p}}_i^S = \bar{\mathbf{p}}_i^S / \|\bar{\mathbf{p}}_i^S\|_1$, and the barycenter is taken as $\tilde{\mathbf{p}}_S = \frac{1}{I} \sum_{i=1}^I \tilde{\mathbf{p}}_i^S$. After $\ell_1$ normalization, the PSD becomes a probability mass function, giving each subject equal weight.
    - Design Motivation: PSD values are squared amplitudes; without normalization, high-impedance subjects dominate the mean. $\ell_1$ normalization ensures subjects with similar spectral shapes contribute equally.

3. **Subject-to-Subject Matching (Step 2b)**:

    - Function: Find the spectrally closest source subject for each target subject as the mapping reference.
    - Mechanism: The Hellinger distance between the $\ell_1$-normalized PSD of the target subject and each source subject is computed as $d_{\text{He}}(\tilde{\mathbf{p}}_i^S, \tilde{\mathbf{p}}^T) = \frac{1}{\sqrt{2}} \|\sqrt{\tilde{\mathbf{p}}_i^S} - \sqrt{\tilde{\mathbf{p}}^T}\|_2$, and the nearest neighbor is selected as $i^* = \arg\min_i d_{\text{He}}$. The Hellinger distance is equivalent to the Wasserstein-2 distance between variance-normalized Gaussian processes.
    - Design Motivation: The barycenter approach maps all subjects to the same reference, potentially losing individual specificity. Subject-to-subject mapping retains finer source–target matching details.

4. **Normalization Filter Construction (Step 3)**:

    - Function: Compute a linear filter that maps the target spectrum to the source reference spectrum.
    - Mechanism: The frequency response is defined as the square root of the PSD ratio, $H[n] = \sqrt{\bar{p}^S[n] / \bar{p}^T[n]}$, and the time-domain impulse response is obtained via IRFFT: $\mathbf{h} = \text{IRFFT}_M(\mathbf{H})$. This zero-phase linear filter solves the optimal transport problem between the source and target Gaussian distributions.
    - Design Motivation: The filter directly equalizes the channel-averaged PSD so that the filtered target signal's spectrum aligns with the source reference. Since the filter is a shared time-domain filter applied to all channels, it can be applied either before or after ICA decomposition.

### Loss & Training
- The method requires no training — the filter is computed in closed form directly from PSD statistics.
- The downstream classifier (random forest) is trained using PSD and autocorrelation features, with hyperparameters selected via leave-one-subject-out cross-validation.
- Segment length $l_{\text{train}}$ is treated as a hyperparameter; validation and testing use segments of 5 minutes and 50 minutes.

## Key Experimental Results

### Main Results — Cross-Dataset IC Classification (Brain Class F1)

| Classifier | Seg. Length | No Filter | Barycenter | $\ell_1$-norm Bary. | Subj-to-subj | p-value |
|---|---|---|---|---|---|---|
| PSD/Autocorr | 5 min | 0.77±0.09 | 0.78±0.12 | **0.84±0.07** | 0.79±0.17 | 0.0046 |
| ICLabel | 5 min | 0.88±0.06 | — | — | — | — |
| PSD/Autocorr | 50 min | 0.83±0.09 | 0.86±0.09 | **0.86±0.08** | 0.85±0.17 | 0.1696 |
| ICLabel | 50 min | 0.89±0.05 | — | — | — | — |

### Ablation Study — In-Domain Performance

| Classifier | 5 min | 50 min | Note |
|---|---|---|---|
| PSD/Autocorr | **0.93±0.05** | **0.96±0.05** | Best in-domain |
| ICLabel | 0.88±0.05 | 0.89±0.07 | General baseline |

### Comparison of CMMN Variants

| Variant | 5 min F1 | 50 min F1 | Note |
|---|---|---|---|
| No filter (baseline) | 0.77 | 0.83 | Severe domain shift |
| Standard Barycenter | 0.78 | 0.86 | Unnormalized barycenter; amplitude bias |
| $\ell_1$-norm Barycenter | **0.84** | **0.86** | Best variant; statistically significant |
| Subj-to-subj | 0.79 | 0.85 | Individual matching; high variance |

### Key Findings
- The $\ell_1$-normalized barycenter is the most stable variant: F1 improves from 0.77 to 0.84 on 5-minute segments (p=0.0046, Wilcoxon test, statistically significant).
- The learned filters are intuitively interpretable: they attenuate 50 Hz noise and amplify 60 Hz components when mapping from European to US data.
- The channel-averaged CMMN with PSD/Autocorr classifier (F1=0.91 in-domain, 0.84–0.86 cross-domain) outperforms ICLabel (0.88–0.89) in limited-data settings.
- The subject-to-subject variant exhibits high variance (±0.17), indicating unstable matching quality; the barycenter approach is more robust.
- Improvement is not statistically significant on 50-minute segments (p=0.17), suggesting that sufficient data length can partially mitigate domain shift.

## Highlights & Insights
- **The critical role of channel averaging**: A remarkably simple modification — averaging across channels rather than computing per-channel filters — resolves the fundamental obstacle of cross-channel-count domain adaptation while guaranteeing that IC characteristics remain unchanged. The elegance of this solution is noteworthy.
- **$\ell_1$ normalization eliminates amplitude bias**: Since PSD values are squared amplitudes, outliers dominate the barycenter without normalization. $\ell_1$ normalization converts the PSD into a PMF, making the Hellinger distance equivalent to the Wasserstein-2 distance — a theoretically principled formulation.
- **Training-free domain adaptation**: No training is required; the filter is computed in closed form from PSD statistics, making the approach well-suited for rapid clinical deployment.
- **Generalizability**: The channel-averaged CMMN framework is applicable to cross-device domain adaptation for any multi-channel physiological signal, including EMG and MEG.

## Limitations & Future Work
- Validation is limited to binary classification (brain vs. non-brain IC); fine-grained multi-class IC classification has not been evaluated.
- The experimental scale is small: 27+7 source subjects and only 12 target subjects.
- Channel averaging assumes comparable spectral profiles across all channels, which may not hold for sparse montages with highly heterogeneous spatial distributions.
- Only a simple random forest classifier is used; integration with deep learning models remains unexplored.
- The subject-to-subject variant exhibits high variance; top-K weighted matching or cluster-based matching could be explored to improve stability.

## Related Work & Insights
- **vs. Original CMMN (Gnassounou 2023)**: The original method constructs an independent filter per channel and targets sleep staging; this work uses channel averaging and $\ell_1$ normalization to extend applicability to cross-channel-count settings and IC classification.
- **vs. ICLabel**: ICLabel is trained on large-scale data using spatial and spectral features; this work relies solely on temporal features combined with CMMN domain adaptation, yet outperforms ICLabel in low-data regimes.
- **Relation to Optimal Transport Domain Adaptation**: CMMN is fundamentally an optimal transport mapping in the spectral domain using the Wasserstein-2 barycenter, providing an elegant theoretical framework for domain adaptation of 1D time-series signals.

## Rating
- Novelty: ⭐⭐⭐ An incremental extension of existing CMMN; channel averaging and $\ell_1$ normalization are conceptually straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Small experimental scale (2 datasets, 44 subjects); statistical significance is only established for 5-minute segments.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and mathematical derivations are complete; length is constrained by the workshop paper format.
- Value: ⭐⭐⭐⭐ Addresses a practical problem of cross-channel-count EEG domain adaptation with high potential for clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](../../ICCV2025/medical_imaging/radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)
- [\[ICML 2026\] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](../../ICML2026/medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)

</div>

<!-- RELATED:END -->
