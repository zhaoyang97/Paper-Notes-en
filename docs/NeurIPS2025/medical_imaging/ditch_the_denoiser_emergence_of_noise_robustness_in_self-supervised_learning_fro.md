---
title: >-
  [Paper Note] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum
description: >-
  [NeurIPS 2025][Medical Imaging][Self-Supervised Learning] A fully self-supervised noise-robust representation learning framework is proposed…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Self-Supervised Learning"
  - "Noise Robustness"
  - "Curriculum Learning"
  - "DINOv2"
  - "Denoiser-Free"
date: 2026-05-08
content_hash: 324d7c24945dbd8a
---

# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum

**Conference**: NeurIPS 2025
**arXiv**: [2505.12191](https://arxiv.org/abs/2505.12191)  
**Code**: [https://github.com/wenquanlu/noisy_dinov2](https://github.com/wenquanlu/noisy_dinov2)  
**Area**: Medical Imaging
**Keywords**: Self-Supervised Learning, Noise Robustness, Curriculum Learning, DINOv2, Denoiser-Free

## TL;DR
A fully self-supervised noise-robust representation learning framework is proposed, leveraging a "denoised→noisy" data curriculum strategy combined with denoised-teacher regularization. This enables SSL models such as DINOv2 to directly process noisy inputs at inference time without any denoiser, achieving a 4.8% improvement in linear probing accuracy under extreme Gaussian noise on ImageNet-1k.

## Background & Motivation

**Background**: SSL methods such as DINOv2 achieve outstanding visual representations on clean data, yet virtually all prior work assumes that training data is clean and high quality.

**Limitations of Prior Work**: In real-world domains such as medical imaging, astronomy, and remote sensing, data is inherently noisy (e.g., sensor noise, speckle noise), and clean reference images for supervised denoising are typically unavailable. Training DINOv2 directly on noisy data leads to severe degradation in representation quality.

**Key Challenge**: A naive solution is a "denoiser preprocessing pipeline"—first applying a self-supervised denoiser, then training the SSL model on denoised images. However, this requires retaining the denoiser at both inference and downstream fine-tuning, introducing significant inference latency, deployment complexity, and potential propagation of denoising bias.

**Goal**: Can the SSL backbone itself internalize noise robustness, enabling complete elimination of the denoiser in downstream tasks?

**Key Insight**: Inspired by curriculum learning—training first on "easy" denoised data, then switching to "hard" noisy data—the model progressively adapts to noise from a stable initialization. A frozen denoised teacher is additionally introduced as an anchor for regularization.

**Core Idea**: Through a denoised→noisy curriculum training strategy combined with denoised-teacher regularization, the SSL model internalizes noise robustness during pre-training, requiring no denoiser whatsoever at inference time.

## Method

### Overall Architecture
The pipeline consists of three steps: (1) train a self-supervised denoiser (e.g., Neighbor2Neighbor) to denoise the noisy dataset; (2) train DINOv2 via a curriculum strategy—first on denoised data for $k$ epochs to establish a stable initialization, then restart training on the original noisy data until convergence; (3) perform downstream fine-tuning and inference directly on noisy data, with no denoiser involved.

### Key Designs

1. **DINOv2 w/ NC (Noise Curriculum)**:

    - **Function**: Designs an "easy-to-hard" data curriculum, training first on denoised data then switching to noisy data.
    - **Mechanism**: Defines a distribution sequence $\langle Q_1, Q_2 \rangle$, where $Q_1$ corresponds to the low-entropy denoised image distribution and $Q_2$ to the high-entropy noisy image distribution, satisfying $H(Q_1) < H(Q_2)$. After training for $k$ epochs on denoised data, all training dynamics (learning rate, weight decay schedule, etc.) are **restarted** and training continues on noisy data.
    - **Design Motivation**: Validated via an MNIST toy experiment—under Gaussian noise $\sigma = 0.4 \times 255$, training SSL directly on noisy data yields only 64.55% accuracy, whereas curriculum learning (30 epochs clean + 20 epochs noisy) recovers 83.05%, demonstrating that the model retains noise-robust features learned during the clean phase.
    - **Downstream inference formula**: $\hat{y} = h_\theta(g_\theta(z))$, where $g_\theta$ is the backbone, $h_\theta$ is the prediction head, and $z$ is the noisy input—no denoising preprocessing is required.

2. **DINOv2 w/ NCT (Noise Curriculum Teacher)**:

    - **Function**: In high-noise regimes, a frozen denoised teacher anchors training to prevent the noisy training phase from drifting.
    - **Mechanism**: Introduces a three-component architecture during the noisy training phase—a trainable teacher $T$, a student $S$, and a frozen denoised teacher $T_{dn}$. Identical data augmentations are applied to the noisy image $x$ and its denoised counterpart $x_{dn}$, and a regularization term is added to the original DINOv2 loss:
    $$L = L_{\text{dinov2}} + \lambda L_{\text{dino\&ibot}}(T_{dn}(\tau_t(x_{dn})), S(\tau_s(x)))$$
    - The original DINOv2 loss is composed of DINO and iBOT cross-entropy terms: $L_{\text{dino\&ibot}} = -\sum p_t^{\text{img}} \log p_s^{\text{img}} - \sum p_t^{\text{patch}} \log p_s^{\text{patch}}$
    - **Design Motivation**: Under extreme noise (e.g., Gaussian $\sigma = 255$, SNR = 0.31 dB), the initialization from curriculum learning alone is insufficient to resist strong noise interference, necessitating an additional anchor constraint. The frozen teacher and the trainable teacher share identical initial weights (both derived from the denoising phase), ensuring alignment of output embeddings.

3. **Flexibility in Denoiser Selection**:

    - **Function**: Neighbor2Neighbor (N2N) is used as the self-supervised denoiser, but the framework imposes no restriction on the choice of denoiser.
    - **Mechanism**: N2N samples two sub-images from a single noisy image to construct noisy–noisy pairs for training a U-Net, requiring no clean reference images.
    - **Design Motivation**: In practice, the denoiser should be selected according to the noise characteristics of the target domain. Even with a very weak denoiser (trained for only 1 epoch), the NC method still yields substantial improvements.

### Loss & Training
- The NC method employs the standard DINOv2 loss (DINO + iBOT + Koleo regularization) in a two-stage training procedure.
- The NCT method additionally incorporates a $\lambda$-weighted denoising regularization term during the noisy training stage.
- Experiments on ImageNet-100 use ViT-S/16 for 200 epochs with batch size 40; experiments on ImageNet-1k use ViT-B/16 for 100 epochs with batch size 512.

## Key Experimental Results

### Main Results

| Dataset | Noise | Method | Linear Probing Acc. | Gain vs. DINOv2 |
|--------|------|------|------------|---------------|
| ImageNet-1k | Gaussian σ=100 (4.36dB) | DINOv2 w/ NCT | 72.1% | +1.4% |
| ImageNet-1k | Gaussian σ=100 (4.36dB) | N2N + DINOv2 | 73.1% | +2.4% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 w/ NCT | 55.8% | **+4.8%** |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | N2N + DINOv2 | 57.2% | +6.2% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 w/ NC | 53.5% | +2.5% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 baseline | 51.0% | — |

### Ablation Study (Generalization Across SSL Models, ImageNet-100 Gaussian σ=100)

| SSL Model | Architecture | Baseline Acc. | w/ NC Acc. | N2N + Model |
|---------|------|---------|-----------|-----------|
| DINOv2 | ViT-S | 55.4% | 68.1% | 69.0% |
| DINO | ViT-S | 57.9% | 62.1% | 62.5% |
| iBOT | ViT-S | 56.9% | **62.7%** | 61.9% |
| SimCLR | ResNet50 | 59.0% | 61.1% | 64.3% |
| MoCo v3 | ViT-S | 52.2% | 55.3% | 60.4% |
| SimSiam | ResNet50 | 64.8% | 65.7% | 68.4% |

### Key Findings
- NCT regularization yields the largest gains under extreme noise (NC→NCT improvement of 6.7% at Gaussian σ=255), but offers minimal benefit under moderate noise.
- **Surprising finding**: When evaluated on a clean validation set, DINOv2 w/ NC/NCT **outperforms** N2N + DINOv2 in most settings, suggesting that the denoiser-free approach learns more generalizable representations (as explicit denoising discards some useful information).
- The NCT method converges to or surpasses the accuracy of its anchor (the denoised teacher), confirming that regularization effectively guides the learning trajectory.
- Extended training narrows the performance gap between methods, but the noise curriculum strategy achieves comparable performance in approximately half the training time.

## Highlights & Insights
- **Complete elimination of the denoiser at inference time**: This is the most critical practical contribution—simpler deployment, faster inference, and no propagation of denoising bias, which is especially important for resource-constrained settings such as medical imaging.
- **"Restart" strategy at curriculum transition**: Rather than simply continuing training on noisy data after the denoised phase, all training dynamics (learning rate schedules, etc.) are reset, providing the model with sufficient capacity to adapt during the noisy phase.
- **Generalization across SSL frameworks**: The NC strategy is effective across 6 different SSL methods; iBOT w/ NC even surpasses N2N + iBOT, demonstrating broad applicability.
- **Alignment between the denoised teacher and the trainable teacher** is key to NCT's effectiveness—arbitrarily using a frozen teacher with a different initialization does not produce meaningful regularization.

## Limitations & Future Work
- Validation is currently limited to synthetic noise; evaluation on real-world noisy datasets (e.g., equipment noise in medical imaging) is lacking.
- The curriculum switching point (i.e., when to transition from denoised to noisy data) requires manual tuning; an adaptive strategy is absent.
- The framework assumes that the self-supervised denoiser can produce reasonable denoising—if the denoiser itself performs poorly under extreme noise, the overall framework may be adversely affected.
- The approach could be extended to other modalities involving sequential data, such as audio, EEG, and financial time series.

## Related Work & Insights
- **vs. N2N + DINOv2 (denoiser preprocessing pipeline)**: This baseline always requires the denoiser at inference time, whereas the proposed method eliminates the denoiser entirely at inference and achieves superior representation quality on clean test sets.
- **vs. Noisy sequential SSL [48]**: Prior noise-robust SSL work has focused on sequential data (EEG) and also uses denoisers to construct contrastive learning pairs; the underlying motivation is similar—leveraging denoisers to facilitate learning of noise-robust representations.
- The approach offers direct inspiration for SSL pre-training on medical images: medical data is typically noisy and lacks clean references, and the proposed method reduces reliance on denoising preprocessing.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of noise curriculum learning and denoised-teacher regularization is explored for the first time in the SSL context, though the individual components are not novel in themselves.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple noise types/intensities, multiple dataset scales, multiple SSL frameworks, and two downstream tasks (classification and retrieval).
- Writing Quality: ⭐⭐⭐⭐ The exposition is logically clear, progressing from toy examples to large-scale experiments in a coherent manner.
- Value: ⭐⭐⭐⭐ The approach has practical value for noisy-data domains such as medical imaging and remote sensing; denoiser-free inference is a strong practical advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[NeurIPS 2025\] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation](unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)

</div>

<!-- RELATED:END -->
