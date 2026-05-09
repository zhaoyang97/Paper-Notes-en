---
title: >-
  [Paper Note] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals
description: >-
  [AAAI 2026][Cross-subject brain decoding] This paper proposes MindCross, a cross-subject brain decoding framework that learns subject-independent information via a shared encoder and subject-specific information via $N$ individual encoders. Combined with a fast calibration stage and a Top-K collaborative decoding module, a single unified model achieves performance comparable to per-subject models on fMRI/EEG-to-video benchmarks, with new subject adaptation requiring only minimal data and time (~1s vs. 5–17s for baselines).
tags:
  - AAAI 2026
  - Cross-subject brain decoding
  - video reconstruction
  - fMRI
  - EEG
  - fast adaptation
  - shared-specific encoder
  - Top-K collaboration
date: 2026-05-08
content_hash: 90ceade3771a15f1
---

# MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals

**Conference**: AAAI 2026
**arXiv**: [2511.14196](https://arxiv.org/abs/2511.14196)
**Code**: [GitHub](https://github.com/XuanhaoLiu/MindCross)
**Area**: Brain Signal Decoding / Brain-Computer Interface
**Keywords**: Cross-subject brain decoding, video reconstruction, fMRI, EEG, fast adaptation, shared-specific encoder, Top-K collaboration

## TL;DR
This paper proposes MindCross, a cross-subject brain decoding framework that learns subject-independent information via a shared encoder and subject-specific information via $N$ individual encoders. Combined with a fast calibration stage and a Top-K collaborative decoding module, a single unified model achieves performance comparable to per-subject models on fMRI/EEG-to-video benchmarks, with new subject adaptation requiring only minimal data and time (~1s vs. 5–17s for baselines).

## Background & Motivation

**Background**: Reconstructing videos from brain signals (fMRI/EEG) is an important brain decoding task. Existing methods (MinD-Video, NeuroClips, Mind-Animator, etc.) predominantly follow a per-subject paradigm—training a separate model for each subject—which demands large amounts of data and incurs significant training overhead.

**Limitations of Prior Work**:
   - **High cost of per-subject paradigm**: Every new subject requires training from scratch, severely limiting scalability in practical BCI applications.
   - **Data scarcity**: Brain-video experiments are expensive to conduct; EEG-video datasets contain only ~1,400 trials per subject.
   - **Insufficient cross-subject methods**: Cross-subject approaches such as MindBridge and GLFA over-emphasize subject-independent information while neglecting subject-specific information; new subject adaptation relies on fine-tuning strategies that are time-consuming and degrade performance on existing subjects.

**Key Challenge**: How can a single unified model learn cross-subject commonalities while preserving individual differences, and rapidly adapt to new subjects with limited data?

**Key Insight**: Inspired by ShaSpec, the paper designs a shared-specific encoder architecture that decouples the learning of subject-independent and subject-specific information. During new subject adaptation, only the subject-specific encoder parameters are updated while all other modules remain frozen.

## Method

### Overall Architecture
The framework consists of three stages: **Training** (joint training of $N$ specific encoders + 1 shared encoder) → **Calibration** (only the new subject's specific encoder and reconstructor are updated) → **Testing** (specific encoder + Top-K collaboration module for joint prediction).

Core data flow: brain signal $\mathbf{x}^i$ → specific encoder $\mathbf{E}_s$ producing $\mathbf{s}^i$ + shared encoder $\mathbf{E}_r$ producing $\mathbf{r}^i$ → ResFuse residual fusion → universal decoder $\mathbf{D}$ → predicted text CLIP embedding $\hat{\mathbf{e}}^i$ → T2V model (PyramidFlow) → video generation.

### Key Designs

1. **Shared-Specific Encoder Architecture**:

    - **Specific encoder $\mathbf{E}_s$**: One per subject, learning subject-dependent brain signal patterns (e.g., individual functional differences across brain regions).
    - **Shared encoder $\mathbf{E}_r$**: Shared across all subjects, learning subject-invariant representations.
    - **ResFuse residual fusion**: Concatenates both features and projects them, then adds the result as a residual to the shared feature: $\text{ResFuse}(\mathbf{s}^i, \mathbf{r}^i) = f(\text{concat}(\mathbf{s}^i, \mathbf{r}^i)) + \mathbf{r}^i$
    - **Design Motivation**: Decoupled learning avoids information redundancy; the residual connection ensures the shared feature remains the primary representation.

2. **Domain Classification + Domain Alignment + Discrepancy Constraint**:

    - **Domain classification loss $\mathcal{L}_c$**: Trains a domain classifier $\mathbf{C}_{dc}$ with cross-entropy to identify the subject from specific features $\mathbf{s}^i$, forcing the specific encoder to extract subject-discriminative information.
    - **Domain alignment loss $\mathcal{L}_{da}$**: A gradient reversal layer (GRL) confuses a second domain classifier so that shared features $\mathbf{r}^i$ cannot distinguish source subjects, achieving domain invariance.
    - **Discrepancy loss $\mathcal{L}_{diff}$**: Enforces orthogonality between the two feature types via $\|\mathbf{s}^i \odot \mathbf{r}^i\|_2^2$, preventing redundant encoding.

3. **Fast Calibration Stage**:

    - Upon arrival of a new subject, all existing modules (shared encoder, decoder, existing specific encoders) are frozen.
    - Only the new subject's specific encoder and reconstructor are trained, involving very few parameters (EEG: 9.77M vs. 126–247M for baselines).
    - Calibration loss: $\mathcal{L}_{calib} = \mathcal{L}_{align} + \alpha'\mathcal{L}_{rec}^t + \beta'\mathcal{L}_{diff}$
    - **Advantage**: Existing subjects' decoding performance is unaffected; training is fast (~1s vs. 5s for MindBridge and 10s for GLFA).

4. **Top-K Collaborative Decoding Module**:

    - The domain classifier $\mathbf{C}_{dc}$ computes the similarity $\mathbf{p}$ between the new subject's specific features $\mathbf{s}^t$ and those of existing subjects.
    - The Top-K most similar existing subjects are selected; their specific encoders independently predict text embeddings, which are combined via similarity-weighted summation.
    - Final prediction: $\hat{\mathbf{e}} = \hat{\mathbf{e}}^t + \lambda \hat{\mathbf{e}}^c$, where $\lambda = 0.01$
    - **Analogy**: Analogous to a memory retrieval mechanism, leveraging existing subjects whose brain patterns resemble those of the new subject to augment decoding.

### Loss & Training

Total training loss:
$$\mathcal{L}_{train} = \mathcal{L}_{align} + \alpha\mathcal{L}_{rec} + \beta\mathcal{L}_c + \gamma\mathcal{L}_{da} + \zeta\mathcal{L}_{diff}$$

The alignment loss combines SoftCLIP contrastive loss and MSE loss:
$$\mathcal{L}_{align} = \mathcal{L}_{SoftCLIP}(\mathbf{e}, \hat{\mathbf{e}}) + \frac{1}{N}\sum_i^N \|\mathbf{e}^i - \hat{\mathbf{e}}^i\|_2^2$$

## Key Experimental Results

### Datasets
- **SEED-DV** (EEG-to-video): 20 subjects, 1,400 trials per subject (40 concepts), 1,200 train / 200 test.
- **CC2017** (fMRI-to-video): 3 subjects, fMRI (3T MRI), 8,640 train / 1,200 test samples.

### Evaluation Metrics
- Semantic level: 2-way / 40(50)-way top-1 classification accuracy (VideoMAE for video-level, CLIP for frame-level).
- Spatiotemporal level: CLIP-pcc (cosine similarity of CLIP embeddings between adjacent frames).
- Pixel level: SSIM, PSNR.

### Main Results

**Cross-subject video reconstruction** (single model vs. multiple models):

| Method | # Models | 2-way-V | CLIP-pcc | SSIM |
|--------|----------|---------|----------|------|
| NeuroClips (per-subject) | 20 | 0.809 | 0.756 | 0.238 |
| Mind-Animator (per-subject) | 20 | 0.799 | 0.421 | 0.253 |
| GLFA (cross-subject) | 1 | 0.778 | 0.751 | 0.192 |
| MindBridge (cross-subject) | 1 | 0.782 | 0.753 | 0.185 |
| **MindCross** | **1** | **0.786** | **0.758** | **0.197** |

- MindCross achieves semantic-level performance comparable to 20 independent models using a single model, and substantially outperforms other cross-subject methods.
- Qualitative results show more accurate semantic decoding (e.g., MindBridge incorrectly decodes a bird as an airplane).

**New subject adaptation**:

| Method | 40-way-V | 40-way-F | PSNR | Adapt. Time | # Params |
|--------|----------|----------|------|-------------|----------|
| MindBridge | 0.142 | 0.104 | 8.514 | 5.104s | 126.81M |
| GLFA | 0.135 | 0.121 | 8.522 | 10.651s | 247.27M |
| **MindCross** | **0.137** | **0.117** | **8.587** | **1.090s** | **9.77M** |

- Adaptation time reduced by **5–10×**; parameter count reduced by **13–25×**.
- Competitive performance achieved with limited data (200 EEG samples / 500 fMRI samples).

### Ablation Study

1. **Training loss ablation**:
    - $\mathcal{L}_{align}$ only: 2-way-V = 0.756
    - +$\mathcal{L}_{rec}+\mathcal{L}_{da}+\mathcal{L}_{dc}$: 2-way-V = 0.789 (significant improvement, validating the shared-specific architecture)
    - +$\mathcal{L}_{diff}$: 2-way-V = 0.786 (marginal difference during training, but beneficial for new subject adaptation during calibration)

2. **Top-K module ablation**: No significant difference between $K=1$ and $K=2$; $K=1$ is used by default.

3. **Subject selection visualization**: Heatmaps show that certain subject pairs (e.g., sub10 and sub14) frequently collaborate with high probability, validating the memory retrieval mechanism.

4. **Feature visualization (t-SNE)**: Raw brain data shows large inter-subject variance; after MindCross, specific features cluster by subject while shared features merge across subjects.

## Highlights & Insights

1. **High practical value**: In BCI applications, training separate models for each new patient is impractical. MindCross's fast calibration (~1s + minimal data) has strong clinical potential.
2. **Elegant design**: Shared-specific decoupling + GRL domain alignment + discrepancy loss collectively achieve clean information separation through a coherent, well-motivated design.
3. **Top-K collaborative decoding**: Cleverly reuses the domain classifier to compute subject similarity without additional training; leverages existing subjects to assist decoding for new subjects.
4. **Freezing strategy preserves prior knowledge**: New subject adaptation does not cause forgetting of existing subjects' decoding capabilities, meeting practical requirements.
5. **Good scalability**: Adding a new subject requires only one additional lightweight specific encoder.

## Limitations & Future Work

1. **Video generation quality bounded by the T2V model**: PyramidFlow is used as the video generation module without any optimization; the upper bound on video quality is determined by the T2V model itself.
2. **Only text CLIP embeddings are predicted**: Frame-level latent variables are not predicted simultaneously, potentially losing low-level visual details (reflected in lower pixel-level metrics).
3. **Limited dataset scale**: SEED-DV contains only 1,400 trials per subject; CC2017 includes only 3 subjects.
4. **Linear growth of specific encoders**: With a very large number of subjects (e.g., 100+), storing and managing numerous specific encoders may become a practical concern.
5. **Fixed $\lambda = 0.01$**: The Top-K collaboration weight is a fixed constant; the possibility of adaptive weighting is not discussed.

## Related Work & Insights

- **Per-subject methods**: MinD-Video (NeurIPS'23), NeuroClips (NeurIPS'24), Mind-Animator (ICLR'25)
- **Cross-subject methods**: MindBridge (CVPR'24), GLFA (ECCV'24), MindEye2, MindTuner (AAAI'25), Wills Aligner (AAAI'25)
- **Shared-specific framework**: ShaSpec (NeurIPS Workshop)
- **Video generation**: Tune-A-Video, AnimateDiff, PyramidFlow

## Rating ⭐⭐⭐⭐

An innovative shared-specific cross-subject framework with comprehensive experiments across dual benchmarks (EEG + fMRI) and a significant efficiency gain for new subject adaptation. However, pixel-level video reconstruction quality still lags behind, and the T2V module is left unoptimized.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](../../NeurIPS2025/others/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)

<!-- RELATED:END -->
