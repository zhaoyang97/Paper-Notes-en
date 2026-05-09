---
title: >-
  [Paper Note] Representation Learning for Spatiotemporal Physical Systems
description: >-
  [CVPR 2026][Self-Supervised Learning][JEPA] This paper systematically benchmarks four learning paradigms — JEPA, VideoMAE, an autoregressive foundation model (MPP), and an operator learning method (DISCO) — across three PDE-based physical systems. It finds that latent-space predictive objectives (JEPA) consistently outperform pixel-level prediction methods on the downstream task of physical parameter estimation, achieving 28–51% relative MSE reduction with greater data efficiency.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - JEPA
  - Physical Systems
  - Representation Learning
  - Parameter Estimation
  - Spatiotemporal PDE
date: 2026-05-08
content_hash: 6119b7d46bf29502
---

# Representation Learning for Spatiotemporal Physical Systems

**Conference**: CVPR 2026
**arXiv**: [2603.13227](https://arxiv.org/abs/2603.13227)
**Code**: [https://github.com/helenqu/physical-representation-learning](https://github.com/helenqu/physical-representation-learning)
**Area**: Self-Supervised Learning
**Keywords**: JEPA, Physical Systems, Representation Learning, Parameter Estimation, Spatiotemporal PDE

## TL;DR
This paper systematically benchmarks four learning paradigms — JEPA, VideoMAE, an autoregressive foundation model (MPP), and an operator learning method (DISCO) — across three PDE-based physical systems. It finds that latent-space predictive objectives (JEPA) consistently outperform pixel-level prediction methods on the downstream task of physical parameter estimation, achieving 28–51% relative MSE reduction with greater data efficiency.

## Background & Motivation
Machine learning applied to spatiotemporal physical systems has largely focused on autoregressive surrogate modeling for next-frame prediction, aiming to learn efficient substitutes for numerical simulations. Such approaches are costly to train, suffer from compounding errors, and — more fundamentally — do not align with the core needs of scientific inquiry, which often center on higher-level downstream tasks such as estimating control parameters (Reynolds number, Prandtl number, etc.) or making qualitative predictions (e.g., laminar vs. turbulent flow).

The key question is: which learning paradigm best preserves physically meaningful information? Intuitively, methods specifically designed for physical modeling (e.g., autoregressive foundation models, neural operators) should outperform general-purpose self-supervised approaches. **But is this actually the case?** This question has previously lacked systematic investigation.

The paper addresses this by using **physical parameter estimation accuracy** as a quantifiable proxy for representation quality, systematically evaluating different learning paradigms — latent-space prediction (JEPA), pixel reconstruction (MAE), autoregressive foundation models (MPP), and operator learning (DISCO) — on physical systems. The core idea is: **predicting representations in latent space (rather than pixel values) may better capture the high-level dynamical information of physical systems**.

## Method

### Overall Architecture
Rather than proposing a new model architecture, this paper designs a systematic evaluation framework:
1. Four models are pretrained separately on three PDE-based physical systems.
2. Encoders are frozen, and attentive probes are trained for physical parameter estimation.
3. Representation quality is assessed via parameter estimation MSE.

### Key Designs
1. **JEPA (Joint Embedding Predictive Architecture) for Physical Dynamics**:

    - Function: Learns an encoder $f: \mathcal{X} \to \mathcal{Z}$ and predictor $g: \mathcal{Z} \to \mathcal{Z}$ to predict representations of future time segments in latent space.
    - Mechanism: Given $T$ timesteps $x_{0:T}$, each segment $x_{t:t+k}$ is encoded as $z_i = f(x_i)$; the latent prediction loss is minimized:
    $\mathcal{L}(f,g) = \mathbb{E}_{x_i, x_{i+1} \sim \mathcal{X}}[\ell_{\text{VICReg}}(g(f(x_i)), f(x_{i+1}))]$
    - VICReg loss is applied to prevent representational collapse:
    $\ell_{\text{VICReg}}(z_i, z_{i+1}) = \lambda s(z_i, z_{i+1}) + \mu[v(z_i) + v(z_{i+1})] + \nu[c(z_i) + c(z_{i+1})]$
      where $s$ is the invariance term (L2 distance), $v$ is variance regularization, and $c$ is covariance regularization.
    - Encoder: 3D ConvNeXt downsampling CNN; Predictor: inverted bottleneck CNN over the channel dimension.
    - Design Motivation: By minimizing error in representation space rather than pixel space, JEPA avoids learning low-level visual details (e.g., texture) and focuses on high-level dynamical features.

2. **VideoMAE Baseline (Pixel-Level Reconstruction)**:

    - Function: Trains an encoder–decoder pair to minimize pixel reconstruction error over masked regions.
    - Mechanism: Spatiotemporal tube masking with pixel-level MSE reconstruction.
    - Architecture: ViT-tiny/16, output shape $l/16 \times w/16 \times t/2 \times 384$.
    - Design Motivation: Serves as a representative of pixel-level predictive paradigms, providing a direct contrast to JEPA.

3. **Physics-Specific Baselines**:

    - **MPP (Multiple Physics Pretraining)**: An autoregressive foundation model that predicts pixel values frame by frame; uses publicly released pretrained weights (AViT-tiny).
    - **DISCO**: An operator meta-learning framework that infers trajectory-specific operator networks from short context windows; pretrained on The Well dataset.
    - Design Motivation: Tests whether methods specifically designed for physical modeling genuinely outperform general self-supervised methods on scientific downstream tasks.

### Loss & Training
- JEPA and VideoMAE are pretrained separately on each physical system for 6 epochs to learn system-specific dynamics.
- During fine-tuning, encoders are frozen and attentive probes are trained for 100 epochs (following the V-JEPA fine-tuning protocol).
- MPP, whose pretraining did not include the target datasets, undergoes end-to-end fine-tuning.
- AdamW optimizer with cosine learning rate scheduling.
- VICReg hyperparameters: $\lambda=2, \mu=40, \nu=2$.
- Input: $l \times w \times d \times 16$ (16-frame context).

## Key Experimental Results

### Main Results

| Method | Type | Active Matter MSE↓ | Shear Flow MSE↓ | Rayleigh-Bénard MSE↓ |
|--------|------|-------------------|-----------------|---------------------|
| **JEPA** | Latent-space prediction | **0.079** | **0.38** | **0.13** |
| VideoMAE | Pixel reconstruction | 0.160 | 0.67 | 0.18 |
| DISCO | Operator learning | 0.057 | 0.13 | 0.01 |
| MPP (full fine-tune) | Autoregressive foundation model | 0.230 | 0.59 | 0.08 |

JEPA vs. VideoMAE improvement: Active Matter **51%**, Shear Flow **43%**, Rayleigh-Bénard **28%**.

### Ablation Study

| Fine-tuning Data Fraction | JEPA MSE↓ | VideoMAE MSE↓ | Notes (Shear Flow) |
|--------------------------|-----------|--------------|-------------------|
| 10% | 0.57 | 0.98 | JEPA at 10% already surpasses VideoMAE at 100% |
| 50% | 0.40 | 0.75 | JEPA reaches 95% of its best performance |
| 100% | 0.38 | 0.67 | Baseline comparison |

### Key Findings
- **JEPA consistently outperforms VideoMAE across all three physical systems**, with a uniform margin of 28–51%.
- **Not all physics-specific methods outperform general self-supervised approaches**: MPP, despite end-to-end fine-tuning, underperforms frozen-encoder JEPA on Active Matter and Shear Flow. This mirrors findings in NLP where autoregressive models underperform encoder models on non-generative tasks (BERT vs. GPT).
- **DISCO and JEPA are top performers in their respective categories**: both operate via latent-space prediction mechanisms (DISCO through hypernetworks producing latent embeddings; JEPA through encoder-based latent prediction), whereas MPP and VideoMAE both perform pixel-level prediction — strongly suggesting that the latent-space mechanism is the decisive factor.
- **JEPA exhibits higher data efficiency**: it surpasses VideoMAE trained on 100% of the fine-tuning data using only 10%.
- The relative ranking of methods varies across systems: DISCO achieves MSE = 0.01 on Rayleigh-Bénard, far outpacing all other methods, possibly because the physical structure of that system aligns particularly well with operator learning.

## Highlights & Insights
- **Novel evaluation perspective**: The paper shifts self-supervised representation learning evaluation from ImageNet image classification to physical parameter estimation, providing a distinctive scientific lens.
- **Significant core finding**: Latent-space prediction outperforms pixel-level prediction — a conclusion that holds consistently across three distinct physical systems, suggesting broad generality.
- **Analogy to NLP**: The observation that autoregressive models underperform encoder models on non-generative downstream tasks — classically noted in the BERT vs. GPT debate — is validated here in the context of physical modeling (MPP vs. JEPA).
- **Elegant experimental design**: Using quantifiable physical parameters as a proxy for representation quality avoids the subjectivity inherent in conventional evaluation metric selection.
- **Concise and impactful**: The paper makes its contribution through experimental findings and insights rather than architectural novelty, communicating its message efficiently.

## Limitations & Future Work
- Evaluation is limited to three 2D PDE systems; generalizability to 3D systems, particle-based systems, or non-PDE systems remains unknown.
- Downstream tasks are restricted to parameter estimation (regression); classification tasks (e.g., laminar vs. turbulent flow) and other scientific tasks are not explored.
- The JEPA encoder is a simple 3D CNN; the effect of larger-scale models or more complex architectures is not investigated.
- The paper does not analyze what the learned representations physically encode — visualization or interpretability analysis is absent.
- DISCO substantially outperforms JEPA on certain systems (Rayleigh-Bénard MSE: 0.01 vs. 0.13), indicating that physics-informed inductive biases retain irreplaceable advantages in specific settings.

## Related Work & Insights
- **V-JEPA (Assran et al., 2025) and VICReg (Bardes et al., 2021)**: Provide the theoretical and practical foundation for applying JEPA to spatiotemporal data.
- **MPP (McCabe et al., 2024)**: A representative physics foundation model; its underperformance on non-generative tasks warrants attention from the community.
- **DISCO (Morel et al., 2025)**: A representative operator meta-learning method; its strong performance validates the value of physics-informed inductive biases.
- **The Well (Ohana et al., 2025)**: Provides standardized datasets for physical systems.
- Broader implication: Scientific machine learning may benefit from distinguishing between tasks requiring precise simulation and those requiring system understanding, favoring representation learning over autoregressive modeling for the latter.

## Rating
- Novelty: ⭐⭐⭐⭐ — The perspective is novel, though JEPA itself is not a new contribution; the value lies in the systematic experimental findings.
- Experimental Thoroughness: ⭐⭐⭐ — Three systems and data efficiency analysis are convincing, but diversity of systems and tasks could be further extended.
- Writing Quality: ⭐⭐⭐⭐ — Concise and clear, with key messages well-highlighted and accessible for quick reading.
- Value: ⭐⭐⭐⭐ — The finding that latent-space prediction outperforms pixel-level prediction provides actionable guidance for the scientific ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)

</div>

<!-- RELATED:END -->
