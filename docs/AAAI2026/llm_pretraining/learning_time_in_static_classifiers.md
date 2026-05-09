---
title: >-
  [Paper Note] Learning Time in Static Classifiers
description: >-
  [AAAI 2026][LLM Pretraining][temporal reasoning] This paper proposes the Support-Exemplar-Query (SEQ) learning framework, which injects temporal reasoning capabilities into standard feed-forward classifiers through loss function design rather than architectural modification. By aligning predicted sequences with class-level temporal prototypes via soft DTW, the method achieves consistent improvements on both fine-grained image classification and video anomaly detection.
tags:
  - AAAI 2026
  - LLM Pretraining
  - temporal reasoning
  - fine-grained classification
  - soft DTW
  - temporal prototype alignment
  - video anomaly detection
date: 2026-05-08
content_hash: 65f5a3de642ae8ee
---

# Learning Time in Static Classifiers

**Conference**: AAAI 2026
**arXiv**: [2511.12321](https://arxiv.org/abs/2511.12321)
**Code**: [https://github.com/Darcyddx/time-seq](https://github.com/Darcyddx/time-seq)
**Area**: Classification / Temporal Reasoning
**Keywords**: temporal reasoning, fine-grained classification, soft DTW, temporal prototype alignment, video anomaly detection

## TL;DR

This paper proposes the Support-Exemplar-Query (SEQ) learning framework, which injects temporal reasoning capabilities into standard feed-forward classifiers through loss function design rather than architectural modification. By aligning predicted sequences with class-level temporal prototypes via soft DTW, the method achieves consistent improvements on both fine-grained image classification and video anomaly detection.

## Background & Motivation

**Background**: Existing classifiers typically assume i.i.d. data and disregard temporal structure. However, in real-world scenarios such as surveillance, medical imaging, and robotics, visual data evolves gradually over time—through changes in pose, illumination, object state, and so on.

**Limitations of Prior Work**: While sequence models such as RNNs, LSTMs, and Transformers can model temporal dependencies, they introduce architectural complexity, require dense temporal annotations, and degrade in performance under label-scarce settings.

**Key Challenge**: Temporal reasoning ability is commonly assumed to require sequence-based architectures, yet many scenarios feature data with inherent temporal structure that static classifiers cannot exploit.

**Key Insight**: The authors raise a key question—**can standard feed-forward classifiers acquire temporal reasoning ability by changing the supervision signal (i.e., the loss function) alone, without any architectural modification?**

**Core Idea**: Temporal reasoning is achieved purely through loss function design by constructing temporally augmented sequences, class-level temporal prototypes, and soft DTW alignment.

## Method

### Overall Architecture

The input consists of static images or video frame sequences. Features are extracted by a frozen pretrained visual encoder (e.g., CLIP-ViT), and a simple FC + softmax classifier generates a prediction sequence $\bm{\Phi} \in \mathbb{R}^{\tau \times C}$. The training objective comprises three loss terms: temporal prototype alignment loss, cross-entropy semantic loss, and smoothness regularization loss.

### Key Designs

1. **Temporally Augmented Sequence Construction**:

    - Function: Generates a virtual temporal sequence from a single static image.
    - Mechanism: Augmentation parameters (rotation angle, brightness, scale, etc.) are linearly interpolated as $p_t = p_{\text{start}} + \frac{t-1}{\tau-1}(p_{\text{end}} - p_{\text{start}})$, producing a smoothly varying image sequence $\mathcal{X}_t = \mathcal{A}_t(\mathcal{X})$.
    - Design Motivation: This simulates gradual real-world variations in pose, illumination, and similar factors, enabling static images to produce temporal training signals.

2. **Support-Exemplar-Query (SEQ) Learning**:

    - Function: Organizes training in an episodic manner and constructs temporal prototypes for each class.
    - Mechanism: Within each episode, (1) $N$ support sequences $\mathcal{S}^\bullet$ are sampled per class; (2) a class exemplar is computed via the soft-DTW Fréchet mean: $\bm{M}^\bullet = \arg\min \sum_{n=1}^N \frac{w_n}{\tau_n} d_{\text{DTW}}^2(\bm{\Phi}_n^\bullet, \bm{M}^\bullet)$; (3) each query sequence is aligned to its corresponding class exemplar.
    - Design Motivation: Episodic training promotes abstraction of intra-class temporal dynamics; the exemplar serves as a "dynamic prototype" encoding the typical temporal evolution of prediction scores for a given class.

3. **Soft DTW Alignment**:

    - Function: Measures temporal distance between two sequences in a differentiable manner.
    - Mechanism: $d_{\text{DTW}}^2(\bm{\Phi}, \bm{\Phi}') = \text{SoftMin}_\gamma(\{\langle \bm{\Pi}, \bm{D} \rangle | \bm{\Pi} \in \mathcal{P}_{\tau,\tau'}\})$, where $\gamma$ controls the degree of alignment softening.
    - Design Motivation: Unlike hard DTW, which is non-differentiable, soft DTW provides smooth gradients for end-to-end training while accommodating variable-length sequence alignment.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{align}} + \alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{smooth}}$$

- **Alignment loss** $\mathcal{L}_{\text{align}}$: Soft-DTW distance between the query sequence and its class exemplar.
- **Cross-entropy** $\mathcal{L}_{\text{CE}}$: Applied frame-wise for sequence tasks, and after temporal averaging for classification tasks.
- **Smoothness regularization** $\mathcal{L}_{\text{smooth}} = \frac{1}{(\tau-1)} \sum_{t=2}^{\tau} \|\phi_t - \phi_{t-1}\|_2^2$: Penalizes abrupt changes between consecutive frame predictions.

## Key Experimental Results

### Main Results: Fine-Grained Image Classification

| Dataset | Baseline | + feat. traj. | + feat. traj. & SEQ (Ours) | Prev. SOTA |
|--------|----------|---------------|--------------------------|-----------|
| Stanford Cars | 94.7 | 95.6 | **96.1** | 95.4 (MPSA) |
| Stanford Dogs | 93.5 | 96.0 | **96.3** | 95.4 (MPSA) |
| Flowers-102 | 97.6 | 98.4 | **98.4** | 97.9 (SR-GNN) |
| SoyAging (ultra-fine-grained) | 79.6 | 79.8 | **80.0** | 79.0 (CLE-ViT) |

### Main Results: Video Anomaly Detection (MSAD)

| Method | AUC | AP |
|------|-----|-----|
| RTFM (I3D) | 86.6 | 68.4 |
| UR-DMU | 85.0 | 68.3 |
| EGO | 87.3 | 64.4 |
| Baseline (FC) | 86.7 | 72.2 |
| + feat. traj. | 92.1 | 77.3 |
| + feat. traj. & SEQ (Ours) | **93.5** | **78.9** |

### Ablation Study

| Configuration | Cars Acc | Dogs Acc | Flowers Acc |
|------|----------|----------|-------------|
| Full model (SEQ + all losses) | **96.1** | **96.3** | **98.4** |
| w/o $\mathcal{L}_{\text{align}}$ | 95.6 | 96.0 | 98.4 |
| w/o $\mathcal{L}_{\text{smooth}}$ | 95.8 | 96.1 | 98.3 |
| w/o SEQ (temporal augmentation only) | 95.6 | 96.0 | 98.4 |

### Key Findings
- Temporally augmented sequences (feat. traj.) alone yield significant improvements, particularly on video anomaly detection where AUC rises from 86.7 to 92.1.
- SEQ contributes more substantially on datasets with rich intra-class variation, such as Cars and Dogs.
- On video anomaly detection, frozen features combined with temporal augmentation substantially outperform complex architectures such as I3D (92.1 vs. 86.6 AUC).
- The hyperparameters $\alpha$, $\beta$, and $\gamma$ are relatively robust; the support set size $N$ performs best in the range of 3–5.

## Highlights & Insights
- **Temporal capability via loss design alone**: Without modifying any architecture, the framework endows feed-forward classifiers with temporal reasoning purely through training strategy—a lightweight and generalizable approach that can serve as a plug-and-play module.
- **Soft DTW alignment in prediction space**: Unlike conventional prototype matching in feature space, this work constructs temporal prototypes in the softmax output space and performs DTW alignment therein, capturing the temporal evolution of prediction patterns.
- **Bridge from static to temporal**: Virtual temporal sequences are constructed via linear interpolation of augmentation parameters, enabling the training of temporally aware models without requiring real video data.

## Limitations & Future Work
- The linear interpolation assumption for temporal augmentation is relatively strong; real-world temporal variations are often nonlinear.
- SEQ yields only marginal gains on ultra-fine-grained data (SoyAging: 79.8→80.0), suggesting that temporal prototypes offer limited discriminability when inter-class differences are minimal.
- The method is validated only with FC classifiers; whether gains persist with more complex classification heads (e.g., MLPs) remains unexplored.
- The alignment loss depends on the soft-DTW parameter $\gamma$; although the paper reports robustness, cross-task tuning may still be necessary.

## Related Work & Insights
- **vs. Prototypical Networks**: Prototypical Networks perform distance comparisons in embedding space, whereas this work aligns temporal prototypes in prediction space—lower-dimensional yet capturing temporal dynamics.
- **vs. Few-shot temporal methods**: Conventional approaches require temporal encoders; this work demonstrates that loss design alone suffices, yielding a simpler architecture.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of injecting temporal reasoning purely via loss design is novel, though individual components (soft DTW, episodic learning) are established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two distinct domains—fine-grained classification and video anomaly detection—with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Logically clear with rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ A lightweight, general-purpose plug-and-play solution with high practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[AAAI 2026\] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](../../NeurIPS2025/llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[CVPR 2026\] Watch and Learn: Learning to Use Computers from Online Videos](../../CVPR2026/llm_pretraining/watch_and_learn_computer_use_from_videos.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](../../ICLR2026/llm_pretraining/moma_a_modular_deep_learning_framework_for_material_property_prediction.md)

<!-- RELATED:END -->
