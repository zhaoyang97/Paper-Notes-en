---
title: >-
  [Paper Note] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging
description: >-
  [CVPR 2026][Medical Imaging][Federated Learning] This paper proposes OmniFM, a modality-robust and task-agnostic federated learning framework that integrates three complementary components—Global Spectral Knowledge Retri…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Federated Learning"
  - "Modality Heterogeneity"
  - "Frequency Domain Analysis"
  - "Task-Agnostic"
date: 2026-05-08
content_hash: f837218d30570b00
---

# OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.21660](https://arxiv.org/abs/2603.21660)
**Code**: N/A
**Area**: Medical Imaging / Federated Learning
**Keywords**: Federated Learning, Modality Heterogeneity, Frequency Domain Analysis, Medical Imaging, Task-Agnostic

## TL;DR

This paper proposes OmniFM, a modality-robust and task-agnostic federated learning framework that integrates three complementary components—Global Spectral Knowledge Retrieval, Embedding-wise Cross-Attention Fusion, and Prefix–Suffix Spectral Prompting—to support five medical imaging tasks (classification, segmentation, super-resolution, VQA, and multimodal fusion) within a unified FL pipeline, achieving substantial improvements over existing baselines under cross-modal heterogeneous settings.

## Background & Motivation

1. **Background**: Federated learning (FL) has emerged as the dominant paradigm for cross-institutional collaborative training on medical images, enabling joint model training without sharing raw data. Existing FL methods are primarily designed for specific tasks (e.g., CNNs for classification, U-Nets for segmentation) and assume homogeneous imaging modalities.

2. **Limitations of Prior Work**: (1) **Task binding**: Tasks such as classification, segmentation, and VQA each require customized FL pipelines, making task switching costly to re-engineer; (2) **Modality fragility**: Hospitals employ different imaging modalities (MRI, CT, PET, pathology, etc.), causing fundamentally divergent loss landscapes across local models. Aggregation pulls the global model toward conflicting local minima, resulting in slow and oscillatory convergence.

3. **Key Challenge**: Existing FL frameworks tightly couple federated optimization design with model architecture and task type, yielding a "one task, one pipeline" paradigm that incurs high engineering overhead and is difficult to deploy in real-world multimodal scenarios.

4. **Goal**: (1) Can a unified FL pipeline be constructed that is reusable across tasks? (2) When tasks are fixed but modalities are heterogeneous, can stable optimization behavior be maintained?

5. **Key Insight**: A frequency-domain insight—low-frequency spectral components across different modalities exhibit strong cross-modal consistency, encoding modality-invariant anatomical structural information. This property can bridge representational gaps between modalities.

6. **Core Idea**: Frequency-domain spectral embeddings serve as cross-modal anchors. Through a global retrieval–fusion–prompting mechanism, modality-invariant knowledge is injected into local representations, enabling a "single pipeline across tasks and modalities" federated learning framework.

## Method

### Overall Architecture

Each client in OmniFM extracts two types of representations: spatial-domain representations $\mathbf{r} \in \mathbb{R}^{L \times d}$ obtained via a backbone, and frequency-domain spectral embeddings $\mathbf{s}$ obtained via FFT. The spectral embeddings are uploaded to a server-side global knowledge bank for top-k retrieval. The retrieved global spectral prototypes are injected into the backbone representations via cross-attention and prefix–suffix prompting, and task heads produce final predictions. A spectral-proximal alignment loss suppresses modality-induced drift at the optimization level.

### Key Designs

1. **Global Spectral Knowledge Retrieval (GSKR)**:

    - **Function**: Retrieves modality-invariant spectral priors from a global knowledge bank.
    - **Mechanism**: FFT is applied to the input image to obtain the amplitude spectrum; low-pass filtering retains coarse-grained anatomical structures, which are then encoded into a normalized spectral token $\mathbf{s}$ via a spectral tokenization module (FreqMix + projection + pooling). The server maintains a knowledge bank $\mathcal{K}^{(r)}$; after the client uploads $\mathbf{s}$, the server retrieves the top-k global spectral prototypes $\mathbf{S}_g$ by cosine similarity. The knowledge bank is kept compact and modality-balanced through frequency-based pruning.
    - **Design Motivation**: Low-frequency components exhibit strong cross-modal consistency, encoding anatomical structure rather than modality-specific texture, thereby serving as stable cross-client anchors.

2. **Embedding-wise Cross-Attention Fusion (ECA)**:

    - **Function**: Injects retrieved global spectral context into backbone representations.
    - **Mechanism**: The backbone representation $\mathbf{r}$ serves as the query and the global spectral prototypes $\mathbf{S}_g$ serve as keys/values in standard cross-attention: $\mathbf{Z} = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}})\mathbf{V}$. This modulates local tokens with global low-frequency priors, biasing them toward modality-invariant anatomical features.
    - **Design Motivation**: Injecting global knowledge directly in the embedding space requires no modification to the backbone architecture, preserving task-agnosticism.

3. **Prefix–Suffix Spectral Prompting (PSP)**:

    - **Function**: Injects global and personalized priors into the token sequence.
    - **Mechanism**: The ECA-fused spectral token $\mathbf{Z}$ is prepended as a prefix to the backbone features $\mathbf{r}$, while a client-specific learnable CLS token $\mathbf{c}$ is appended as a suffix: $\mathbf{r}' = [\mathbf{Z} \| \mathbf{r} \| \mathbf{c}]$. The prefix biases representations toward modality-invariant structures, while the suffix captures institution-specific distributional adaptation.
    - **Design Motivation**: The dual-prompt design balances cross-client consistency (prefix) and local specialization (suffix), addressing the global–personalized trade-off inherent in federated learning.

### Loss & Training

The total loss is:

$$\min_{\phi,\psi} \mathcal{L}_\text{task}(h_\psi([\mathbf{Z}\|\mathbf{r}\|\mathbf{c}]), y) + \lambda \mathcal{L}_\text{align}$$

The **Spectral-Proximal Alignment (SPAlign)** loss $\mathcal{L}_\text{align} = \|\mathbf{s} - \bar{\mathbf{s}}_g\|_2^2$ enforces local spectral embeddings to stay close to the centroid of retrieved global prototypes, suppressing modality-induced optimization drift in the frequency domain while preserving flexibility in the spatial domain.

## Key Experimental Results

### Main Results

Cross-modal classification (MedMNIST-v2, Scenario 1 Hard, ResNet-18 backbone):

| Method | Acc@20% | Acc@100% | F1@100% |
|--------|---------|----------|---------|
| FedAvg | 56.31 | 84.21 | 0.474 |
| FedPer | 84.47 | 92.57 | 0.665 |
| **OmniFM** | **96.85** | **97.82** | **0.668** |

Super-resolution (BreaKHis, ×2 scale, average PSNR):

| Method | Scenario 1 | Scenario 2 |
|--------|-----------|-----------|
| FedAvg | 35.95 | 41.50 |
| FedPer | 39.52 | 40.87 |
| **OmniFM** | **42.21** | **42.30** |

### Ablation Study

VQA Task 1 performance under different fine-tuning strategies:

| Configuration | F-C (IID) | F-CL (Non-IID) |
|---------------|-----------|----------------|
| FedAvg | 0.783 | 0.827 |
| FedPer | 0.799 | 0.833 |
| OmniFM | **0.812** | **0.831** |

### Key Findings

- OmniFM achieves the best or second-best performance across all tasks (classification/segmentation/super-resolution/VQA) and all heterogeneous scenarios.
- In cross-modal classification, OmniFM achieves 96.85% accuracy at a 20% participation rate, compared to only 56.31% for FedAvg—a gap exceeding 40 percentage points.
- In super-resolution, OmniFM maintains its advantage at the challenging ×8 scale (30.02 vs. 29.36 PSNR).
- In cross-modal VQA Task 2 with 8 fully heterogeneous modality clients, OmniFM achieves an average accuracy of 79.27%, surpassing FedPer's 78.40%.

## Highlights & Insights

- **Strong frequency-domain invariance insight**: The observation that low-frequency spectra exhibit cross-modal consistency is concise and compelling, providing an elegant theoretical foundation for handling modality heterogeneity in federated learning. This insight is transferable to other cross-domain and cross-modal learning scenarios.
- **Truly task-agnostic pipeline**: Supporting five distinct tasks—classification, segmentation, super-resolution, VQA, and multimodal fusion—within a single federated optimization pipeline is highly uncommon in the FL literature.
- **Retrieval-augmented federated learning**: The global spectral knowledge bank combined with top-k retrieval resembles retrieval-augmented generation (RAG), but applied in the frequency domain of federated learning—a creative cross-domain innovation.

## Limitations & Future Work

- Although uploading spectral embeddings is lightweight, potential privacy leakage risks remain unexplored (spectra may contain recoverable patient information).
- Experiments involve a small number of clients (3–8), and performance in large-scale federated scenarios (50+ clients) has not been validated.
- The knowledge bank pruning strategy (based on retrieval frequency) may cause spectral prototypes of rare modalities to be discarded prematurely.
- Comparisons with recent foundation model federated fine-tuning methods (e.g., FedPETuning) are absent.

## Related Work & Insights

- **vs. FedPer**: FedPer achieves partial adaptation by separating personalized layers but does not address modality heterogeneity; OmniFM fundamentally mitigates modality discrepancy through frequency-domain alignment.
- **vs. FedProx**: FedProx applies proximal constraints in the spatial domain, offering limited effectiveness under modality heterogeneity; OmniFM's SPAlign operates in the frequency domain, more precisely decoupling modality-specific from modality-invariant components.
- The combination of frequency-domain priors and federated learning may inspire cross-institutional medical image pre-training approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The frequency-domain perspective for addressing modality heterogeneity in FL is a novel angle, though individual sub-modules (cross-attention, prefix/suffix prompting) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers five tasks and multiple heterogeneous scenarios, but lacks large-scale client experiments and privacy analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly with informative figures, though some formulations could be further streamlined.
- **Value**: ⭐⭐⭐⭐ Offers significant practical value for federated medical image analysis; the unified pipeline concept carries meaningful engineering implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](../../ICLR2026/medical_imaging/incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/medical_imaging/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)

</div>

<!-- RELATED:END -->
