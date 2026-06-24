---
title: >-
  [Paper Note] Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection
description: >-
  [CVPR 2025][AI Safety][Incremental Learning] Introduces the SUR-LID method to address the catastrophic forgetting problem in Incremental Face Forgery Detection (IFFD). It retains the global feature distribution of old tasks through Sparse Uniform Replay (SUR), and "stacks" the distributions of new and old tasks "brick by brick" in the latent space through feature isolation and decision alignment strategies in the Latent Incremental Detector (LID)…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Incremental Learning"
  - "Face Forgery Detection"
  - "Catastrophic Forgetting"
  - "Feature Isolation"
  - "Replay Strategy"
date: 2026-05-08
content_hash: 328d17955bdbb85e
---

# Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection

**Conference**: CVPR 2025  
**arXiv**: [2411.11396](https://arxiv.org/abs/2411.11396)  
**Code**: [github](https://github.com/beautyremain/SUR-LID)  
**Area**: AI Security  
**Keywords**: Incremental Learning, Face Forgery Detection, Catastrophic Forgetting, Feature Isolation, Replay Strategy

## TL;DR

Introduces the SUR-LID method to address the catastrophic forgetting problem in Incremental Face Forgery Detection (IFFD). It retains the global feature distribution of old tasks through Sparse Uniform Replay (SUR), and "stacks" the distributions of new and old tasks "brick by brick" in the latent space through feature isolation and decision alignment strategies in the Latent Incremental Detector (LID), rather than overwriting each other.

## Background & Motivation

With the rapid development of face forgery technology, an increasing number of forgery types (face swapping, facial reenactment, entire face synthesis, etc.) have emerged. Incremental Face Forgery Detection (IFFD) addresses evolving forgery methods by step-wise fine-tuning trained models with new forgery data, but faces a severe **catastrophic forgetting** problem.

The key challenge is that: IFFD is inherently a simple binary classification task (real/fake). When all forgery types are grouped into a single "Fake" class, the feature distributions of different forgery types overwrite each other, causing the model to forget the unique features of earlier tasks. Existing methods (e.g., DFIL, HDP) only retain a few representative samples (e.g., center or hard samples), failing to maintain the global feature distribution of old tasks.

The key insight of this paper is: **the feature distributions of new tasks should not overwrite old tasks; instead, the distributions of each task should be isolated in the latent space while aligning their decision boundaries**—akin to "stacking bricks" in the latent space. This requires two prerequisites: (1) the replay set must represent the global distribution of old tasks rather than just local points; (2) effective isolation and alignment mechanisms are required.

## Method

### Overall Architecture

SUR-LID consists of two core components: (1) **Sparse Uniform Replay (SUR)** strategy: selects a replay subset capable of representing the global distribution after each task completes training; (2) **Latent Incremental Detector (LID)**: utilizes SUR data for feature isolation (via isolation loss and distribution re-filling) and incremental decision alignment. EfficientNetB4 is used as the backbone, and 500 replay samples are retained for each task.

### Key Designs

**1. Sparse Uniform Replay (SUR)**

- **Function**: Selects a replay subset representing the global feature distribution of old tasks (distinct from traditional methods that only keep center/hard samples).
- **Mechanism**: Selects samples by simultaneously considering three factors: (a) **Magnitude Uniformity**: sorted by the distance from feature to centroid $M^t = \|F^t - c^t\|_2$, then sampled uniformly in intervals; (b) **Angular Uniformity**: selects the sample $f_a^t$ with the lowest cosine similarity to the most stable sample in each interval; (c) **Stability**: measures feature stability via consistency under grid shuffle $s_i^t = \frac{\tilde{f}_i^t \cdot (f_i^t)^T}{\|\tilde{f}_i^t\|_2 \cdot \|f_i^t\|_2}$, selecting the most stable feature $f_s^t$ per interval. Ultimately, 2 samples (the most stable + the most dissimilar) are selected from each of the $n_r/2$ intervals.
- **Design Motivation**: Traditional replay strategies (center, hard samples) cannot maintain the global distribution. Experiments verify that the MMD distance of SUR is significantly lower than that of existing methods.

**2. Feature Isolation with Distribution Re-filling**

- **Function**: Isolates the feature distributions of each task/domain (real/fake $\times$ new/old) to prevent mutual overwriting.
- **Mechanism**: (a) **Distribution Re-filling**: leverages the sparse uniformity of SUR, generating and filling new points between replay points and the centroid via interpolation: $f_{\text{filled}} = \beta(\alpha f_1 + (1-\alpha) f_2) + (1-\beta) c$; (b) **Isolation Loss**: based on supervised contrastive loss $\mathcal{L}_{iso}$, independent labels are assigned to the real/fake domains of each task to pull within-domain features closer and push across-domain features apart.
- **Design Motivation**: Although the SUR subset can represent the global distribution, it is still sparse. Distribution re-filling interpolates within triangular regions in the feature space to recover a more complete distribution, enhancing the isolation effect.

**3. Incremental Decision Alignment (IDA)**

- **Function**: Aligns the decision boundaries of independent classifiers for each task, allowing all real/fake domains to be separated by a unified boundary.
- **Mechanism**: Maintains an independent linear classifier $\mathcal{C}^t$ for each task. During training, the new classifier is aligned with the old one via angular interpolation: $\theta^{t+1} \leftarrow \|\theta^{t+1}\|_2 \cdot \frac{(1-\gamma)\tilde{\theta}^{t+1} + \gamma\tilde{\theta}^t}{\|(1-\gamma)\tilde{\theta}^{t+1} + \gamma\tilde{\theta}^t\|_2}$. During inference, the average prediction of all classifiers is taken: $y_{\text{infer}} = \frac{1}{t+1}\sum_{i=1}^{t+1}\mathcal{C}^i(f)$.
- **Design Motivation**: Feature isolation separates different domains, but binary classification is still ultimately required. Recursively aligning decision boundaries ensures that incrementally accumulated forgery information can be unified and utilized.

### Loss & Training

- **Total Loss**: $\mathcal{L}_{\text{overall}} = \mathcal{L}_{iso} + \mu_1 \mathcal{L}_{dis} + \mu_2 \mathcal{L}_{det}$
    - $\mathcal{L}_{iso}$: Supervised contrastive loss to achieve feature isolation.
    - $\mathcal{L}_{dis}$: Knowledge distillation loss to maintain consistency of old features, $\mathcal{L}_{dis} = \sum_{i=1}^{t}(\hat{F}^i - \mathcal{E}^t(\hat{X}^i))^2$.
    - $\mathcal{L}_{det}$: Cross-entropy loss for independent classifiers of each task.
- **Hyperparameters**: $\mu_1=1$, $\mu_2=0.1$, $\gamma=0.001$, learning rate 0.0002, epoch 20.
- After optimizing $\mathcal{L}_{\text{overall}}$ via backpropagation, IDA is used to align decision boundaries.

## Key Experimental Results

### Main Results

Protocol 1 (Dataset increment, average AUC after 4 tasks):

| Method | Replay Size | SDv21 | FF++ | DFDCP | CDF | Avg. |
|------|--------|-------|------|-------|-----|------|
| Lower Bound | 0 | 0.528 | 0.636 | 0.764 | 0.982 | 0.726 |
| LwF | 0 | 0.615 | 0.813 | 0.834 | 0.926 | 0.797 |
| DFIL | 500 | 0.933 | 0.740 | 0.791 | 0.988 | 0.863 |
| HDP | 500 | 0.906 | 0.804 | 0.841 | 0.950 | 0.875 |
| **SUR-LID** | **500** | **0.997** | **0.848** | **0.907** | **0.974** | **0.932** |

### Ablation Study

Ablation of each component (Protocol 1, AUC after T4):

| Variant | SDv21 | FF++ | DFDCP | CDF | Avg. |
|------|-------|------|-------|-----|------|
| w/o All | 0.873 | 0.674 | 0.769 | 0.972 | 0.822 |
| w/o IDA | 0.858 | 0.757 | 0.744 | 0.982 | 0.835 |
| w/o $\mathcal{L}_{iso}$ | 0.960 | 0.801 | 0.824 | 0.952 | 0.884 |
| w/o DR | 0.976 | 0.832 | 0.881 | 0.975 | 0.916 |
| **Ours** | **0.997** | **0.848** | **0.907** | **0.974** | **0.932** |

### Key Findings

1. **IDA is the most critical component**: removing it drops the average AUC from 0.932 to 0.835, showing that decision alignment is vital for leveraging accumulated forgery information.
2. **SUR significantly outperforms traditional replay strategies**: Avg AUC of SUR is 0.932 vs. 0.878 for Center+Hard vs. 0.783 for Random.
3. **Distribution Re-filling brings ~1.6% improvement**: enhances the isolation effect by filling gaps within sparse distributions.
4. **The advantage is more pronounced in Protocol 2**: when forgery types are diverse and real images originate from the same domain, existing methods degrade severely, whereas SUR-LID still maintains an average AUC of 0.943.

## Highlights & Insights

1. **The "brick-by-brick" metaphor is highly intuitive**: visualizing distribution management in incremental learning as stacking bricks provides clear intuition.
2. **The uniform sparse sampling idea of SUR possesses strong generalizability**: not limited to face forgery detection, it can be borrowed by any incremental learning scenario requiring distribution-level replay.
3. **UMAP visualization experiments** intuitively demonstrate the effect of distribution isolation: compared to the distribution overwriting phenomenon in DFIL, SUR-LID indeed achieves clear domain separation.

## Limitations & Future Work

1. Maintaining an independent classifier for each task leads to a linear increase in inference overhead as the number of tasks grows.
2. The replay set size is fixed at 500; the impact of different replay sizes on distribution preservation is not fully explored.
3. The isolation strategy assumes that the distributions of different forgery types should be completely separated, but in practice, some forgery methods may share underlying features.
4. Combining SUR with generative replay or introducing prompt-based incremental learning paradigms can be explored.

## Related Work & Insights

- **DFIL**: Uses center and hard sample replay, serving as the main baseline. The advantage of SUR in maintaining global distribution demonstrates the importance of uniform sampling.
- **HDP**: Uses Universal Adversarial Perturbation (UAP) as a replay mechanism, which is a novel idea but underperforms SUR in distribution preservation.
- **DER (Dark Experience Replay)**: A general incremental learning method, which has limited adaptability in the IFFD scenario.
- **SupCon Loss**: The supervised contrastive loss is adapted into an isolation loss, demonstrating a new application of contrastive learning in incremental learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The framework design of "aligned feature isolation" is novel, and the SUR strategy features clear innovation in distribution preservation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 protocols, detailed ablation studies, UMAP visualizations, and MMD analyses, which are highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear narrative logic; the "brick-by-brick stacking" metaphor is consistently applied throughout the text.
- **Value**: ⭐⭐⭐⭐ — Significantly advances the IFFD field, with the SUR strategy also offering insights for general incremental learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)
- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](towards_general_visual-linguistic_face_forgery_detection.md)
- [\[CVPR 2026\] A Sanity Check for Multi-In-Domain Face Forgery Detection in the Real World](../../CVPR2026/ai_safety/a_sanity_check_for_multi-in-domain_face_forgery_detection_in_the_real_world.md)
- [\[CVPR 2026\] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization](../../CVPR2026/ai_safety/diffusionff_a_diffusion-based_framework_for_joint_face_forgery_detection_and_fin.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)

</div>

<!-- RELATED:END -->
