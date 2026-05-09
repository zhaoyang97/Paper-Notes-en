---
title: >-
  [Paper Note] PACE: Pretrained Audio Continual Learning
description: >-
  [ICLR 2026][Audio & Speech][Audio continual learning] This paper presents the first systematic benchmark for audio continual learning (CL), identifies a fundamental upstream–downstream mismatch in pretrained audio models caused by the dominance of low-level spectral features, and proposes PACE—comprising improved first-session adaptation, adaptive subspace-orthogonal PEFT, and boundary-aware perturbation regularization—achieving substantial improvements over SOTA across 6 audio CL benchmarks.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - Audio continual learning
  - pretrained models
  - parameter-efficient fine-tuning
  - analytic classifier
  - catastrophic forgetting
date: 2026-05-08
content_hash: ed96af71bc431e0e
---

# PACE: Pretrained Audio Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2602.03355](https://arxiv.org/abs/2602.03355)
**Code**: Available (to be released with the paper)
**Area**: Audio & Speech
**Keywords**: Audio continual learning, pretrained models, parameter-efficient fine-tuning, analytic classifier, catastrophic forgetting

## TL;DR

This paper presents the first systematic benchmark for audio continual learning (CL), identifies a fundamental upstream–downstream mismatch in pretrained audio models caused by the dominance of low-level spectral features, and proposes PACE—comprising improved first-session adaptation, adaptive subspace-orthogonal PEFT, and boundary-aware perturbation regularization—achieving substantial improvements over SOTA across 6 audio CL benchmarks.

## Background & Motivation

Pretrained audio models excel on static tasks but suffer catastrophic forgetting under continuously evolving data distributions. Directly transferring visual-domain CL methods to the audio domain faces fundamental obstacles:

**Severe upstream–downstream mismatch**: Audio backbones (e.g., EAT) are pretrained via spectrogram reconstruction, emphasizing low-level time-frequency patterns rather than structured semantics, whereas downstream CL requires high-level discriminative representations.

**More severe representation drift**: The representation shift between consecutive sessions in the audio domain far exceeds that in the visual domain (quantified via t-SNE and CKA analysis), leading to more pronounced forgetting.

**Failure of PEFT-based methods**: Methods such as L2P and DualPrompt degrade approximately 3× more on audio than on vision.

Three key findings motivate the method design:

| Finding | Content | Implication |
|---------|---------|-------------|
| Finding 1 | Statistical methods (FSA + analytic classifier) outperform PEFT methods | Establishes the technical direction |
| Finding 2 | Coarse-grained representation saturation: the first session already captures most information | FSA requires improvement |
| Finding 3 | Larger fine-grained gap: the first session alone cannot bridge the semantic gap | Multi-session adaptation is needed |

## Method

### Overall Architecture

PACE is a staged framework consisting of three stages:

- **Stage 1 (FSA)**: Freeze the output head; adapt deep backbone layers with LoRA; replace the head with an analytic classifier.
- **Stage 2 (MSA)**: For sessions $t \in (1, T_3]$, introduce subspace-orthogonal PEFT to progressively align representations.
- **Stage 3 (Frozen)**: For $t > T_3$, freeze the backbone and update only the analytic classifier.

### Key Designs

#### 1. Improved First-Session Adaptation (Improved FSA)

**Constrained head learning**:
- Conventional FSA jointly trains the head and backbone, leading to head overfitting and insufficient backbone adaptation.
- PACE employs asymmetric optimization: $\eta_{head} \ll \eta_{bb}$.
- A staged procedure is used: first freeze the backbone and train the head for $E_{head}$ epochs, then freeze the head and fine-tune the backbone for $E_0$ epochs.
- This is **opposite** to the LAE/SLCA strategy in visual CL—audio backbones require encouraged adaptation rather than suppression.

**Later-layer LoRA**:
- CKA analysis shows that shallow layers encode domain-general time-frequency patterns while deep layers encode task-specific semantics.
- The first $L_{tune}-1$ layers are frozen; LoRA is applied only to layers $l \geq L_{tune}$:
  $$W_1^l = W_0^l + A_1^l B_1^l, \quad L_{tune} \leq l \leq L$$
- The boundary layer $L_{tune}$ is determined automatically via a CKA deviation threshold $\rho_{layer}$.

**Analytic classifier** (replacing the trainable head):
- A random projection $W_{proj}$ enhances feature discriminability.
- The autocorrelation matrix is updated recursively via the Woodbury identity:
  $$R_t = R_{t-1} - R_{t-1}\hat{Z}_t^\top(I + \hat{Z}_t R_{t-1} \hat{Z}_t^\top)^{-1}\hat{Z}_t R_{t-1}$$
- Classifier weights are updated in closed form—no sample storage and no destructive updates.

#### 2. Adaptive Multi-Session Subspace-Orthogonal PEFT

**Multi-session adaptation (MSA)**: Each session introduces an independent LoRA module; parameters from previous sessions are frozen:
$$W_t = W_0 + \sum_{\tau=0}^{t-1} B_\tau A_\tau + B_t A_t$$

**Gradient projection constraint**—ensures updates do not disrupt old-task representations:
$$g_{update} = P_{\mathcal{U}_t} \nabla_\theta \mathcal{L}_{ce}(g_t(f_t(\mathcal{X}_t)), \mathcal{Y}_t)$$

**Efficient null-space computation** (via LoRA subtraction):
- Construct an "unlearned model": $W_t^{unlearn} = W_0 - \sum_{\tau=0}^{t-1} A_\tau B_\tau$
- Compute the uncentered covariance matrix $X_t^{ucov}$ of current-session features.
- SVD decomposition identifies the projection subspace, retaining principal components with energy ratio $> \rho_{svd}$.
- No historical features need to be stored, significantly reducing memory overhead.

**Adaptive freezing**: When $\sum_{i=0}^{T_3} N_t > N_{stop}$, the framework transitions to Stage 3 and freezes the backbone.

#### 3. Boundary-Aware Regularization

Addresses decision boundary confusion caused by entanglement between old and new class representations.

**Boundary sample detection**:
- For each input, $N_p$ time-frequency masked perturbations are generated: $\tilde{x}_{i,t}^k = \mathcal{Q}(x_{i,t}, r_T, r_F)$
- If the misclassification rate of a temporary model $\theta_{temp}$ on the perturbations exceeds threshold $\rho_p$, the sample is added to the boundary set $\mathcal{B}_t$.

**Regularization loss**:
$$\mathcal{L}_{reg}(i) = \max(0, \delta + \frac{1}{|\mathcal{S}_i|}\sum_{u \in \mathcal{S}_i}\|f_t(u) - \mu(x_c)\|_2^2 - \min_{b \in \mathcal{B}_t}\|f_t(x_{i,t}) - b\|_2^2)$$

Effect: pulls representations toward class centers while pushing them away from boundary points, enlarging inter-class margins.

### Loss & Training

- FSA stage: cross-entropy $\mathcal{L}_{ce}$ + boundary regularization $\mathcal{L}_{reg}$
- MSA stage: cross-entropy + regularization + subspace-orthogonal gradient projection
- Stage 3: analytic classifier updated only (closed-form solution, no gradient-based training)
- Pretrained backbone: EAT (12-layer ViT, self-supervised pretraining on AudioSet-2M, ~5,000 hours)
- Data augmentation: SpecAugment-style time-frequency masking

## Key Experimental Results

### Main Results

**Table 1: Average Top-1 Accuracy (%) on 6 Audio CL Benchmarks**

| Method | ESC-50 | US8K | SC2 | TIMIT-2 | TIMIT-3 | VocalSet |
|--------|--------|------|-----|---------|---------|----------|
| Joint Training (upper bound) | 96.50 | 98.07 | 95.91 | 95.22 | 95.22 | 76.65 |
| L2P | 39.50 | 38.75 | 14.70 | 1.50 | 2.53 | 20.39 |
| RanPAC (w/ FSA) | 92.25 | 97.08 | 90.53 | 85.63 | 89.92 | 62.82 |
| HiDe-Prompt | 83.75 | 79.89 | 40.10 | 47.78 | 49.60 | 48.36 |
| **PACE** | **95.75** | **97.49** | **91.87** | **90.95** | **94.05** | **69.08** |

**Gap to joint training upper bound**: only 0.75% on ESC-50, 0.58% on US8K, and 1.17% on TIMIT-3.

**Table 2: Ablation—Improved FSA Components (Coarse-Grained)**

| Strategy | ESC-50 | US8K | SC2 |
|----------|--------|------|-----|
| w/o FSA | 92.50 | 96.49 | 81.22 |
| Naive FSA | 92.25 | 97.08 | 90.53 |
| + Low LR | 93.75 | 97.35 | 90.95 |
| + Later Layer LoRA | **95.75** | **97.49** | **91.87** |

### Ablation Study

PACE maintains its advantage on the SSLAM backbone, confirming backbone-agnostic generalizability.

Contribution of MSA on fine-grained benchmarks:
- FSA only → +MSA: +3.2% (TIMIT-2)
- +Subspace orthogonality: +1.5%
- +Boundary-aware regularization: +0.6%

### Key Findings

1. **Fundamental difference between audio and visual CL**: Audio backbones' emphasis on low-level spectral features leads to representation drift 3× greater than in vision.
2. **Counter-intuitive FSA finding**: Audio CL requires encouraging backbone adaptation (opposite to visual CL); freezing shallow layers while tuning deep layers is critical.
3. **Stability of analytic classifiers**: Prevents cumulative bias and propagation of representation drift.
4. **Novel use of LoRA subtraction**: Approximates the representation subspace of old tasks without storing historical features.

## Highlights & Insights

- **First systematic audio CL benchmark**: 6 benchmarks covering coarse/fine-grained, speech/music/environmental sound tasks.
- **"Adaptation over freezing"**: Stands in sharp contrast to the visual-domain finding that "freezing the backbone suffices," revealing unique properties of pretrained audio models.
- **Three-stage progressive framework**: FSA → MSA → frozen naturally balances plasticity and stability.
- **LoRA subtraction to construct an "unlearned model"**: Elegantly and efficiently approximates the historical representation subspace via parameter arithmetic.

## Limitations & Future Work

1. Approximation assumption in LoRA subtraction: subtracting LoRA $\neq$ exact unlearning; bias may be non-trivial under high rank or strong adaptation regimes.
2. Boundary detection relies on the quality of the temporary model $\theta_{temp}$.
3. The adaptive freezing threshold $N_{stop}$ requires manual specification; the optimal value may vary across scenarios.
4. Validation is limited to the class-incremental setting; task-aware, domain-incremental, and other CL protocols are not explored.
5. The gap on VocalSet remains 7.57%, indicating the most severe mismatch on fine-grained music tasks.

## Related Work & Insights

- The analytic classifier from **RanPAC** serves as a cornerstone of the technical approach.
- **LoRA subtraction**'s parameter arithmetic is innovatively repurposed for null-space projection.
- The mismatch between **EAT**'s spectrogram reconstruction pretraining objective and downstream classification is the root cause of the core challenge.
- Insight: the degree of alignment between pretraining objectives and downstream tasks determines the difficulty of continual learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — First audio CL benchmark + three-stage framework
- Technical Depth: ⭐⭐⭐⭐⭐ — Theoretically complete subspace-orthogonal PEFT and boundary-aware regularization
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 benchmarks, multi-backbone validation, comprehensive ablation
- Value: ⭐⭐⭐⭐ — Addresses genuine needs in audio CL; deployment scenarios warrant further clarification

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/audio_speech/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

<!-- RELATED:END -->
