---
title: >-
  [Paper Note] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution
description: >-
  [CVPR 2025][Multimodal VLM][Image Quality Assessment] Proposes DeQA-Score, which discretizes the Gaussian distribution of quality scores into soft labels (replacing Q-Align's one-hot labels), significantly reducing discretization information loss (by 10-35 times). It introduces a fidelity loss based on the Thurstone model to achieve joint training on multiple IQA datasets, comprehensively outperforming baseline models on score regression tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Image Quality Assessment"
  - "MLLM"
  - "Score Regression"
  - "Distribution Discretization"
  - "Soft Label"
  - "Multi-dataset Joint Training"
date: 2026-05-08
content_hash: c20929ad2b929c5f
---

# Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution

**Conference**: CVPR 2025  
**arXiv**: [2501.11561](https://arxiv.org/abs/2501.11561)  
**Code**: [https://depictqa.github.io/deqa-score/](https://depictqa.github.io/deqa-score/)  
**Area**: Multimodal VLM  
**Keywords**: Image Quality Assessment, MLLM, Score Regression, Distribution Discretization, Soft Label, Multi-dataset Joint Training

## TL;DR
Proposes DeQA-Score, which discretizes the Gaussian distribution of quality scores into soft labels (replacing Q-Align's one-hot labels), significantly reducing discretization information loss (by 10-35 times). It introduces a fidelity loss based on the Thurstone model to achieve joint training on multiple IQA datasets, comprehensively outperforming baseline models on score regression tasks.

## Background & Motivation
MLLMs excel in describing image quality (linguistic evaluation) but still lag behind traditional IQA methods in precise quality score regression. The core obstacle is that quality scores are inherently continuous values (typically modeled as Gaussian distributions), whereas MLLMs generate discrete tokens—this distribution gap necessitates the discretization of scores.

**Limitations of Prior Work**: Methods like Q-Align discretize mean scores into one-hot labels (5 levels: bad/poor/fair/good/excellent), which suffer from three major issues:
1. **Severe Information Loss**: Discretization error is 10-35 times larger than that of soft labels.
2. **Disruption of Inter-image Relationships**: Images with similar quality but crossing boundary edges are assigned to different categories (e.g., MOS=3.38 -> fair, MOS=3.49 -> good).
3. **Independency Assumption of Levels**: One-hot labels assume the 5 levels are orthogonal, whereas "fair" is semantically closer to "good" than to "excellent".

**Core Idea**: Instead of discretizing the mean of the scores, discretize the entire distribution of the scores—integrate the Gaussian distribution around the center points of the 5 levels to obtain a soft label that preserves the distribution characteristics.

## Method

### Overall Architecture
DeQA-Score is based on the mPLUG-Owl2 architecture (CLIP ViT-L + Q-Former visual abstractor + LLaMA-2-7B). During training, a KL divergence loss is applied to the level token within the response template "The quality of this image is \<level\>" (aligned with the soft label), while other tokens use standard cross-entropy. During inference, the mean and variance of the continuous score distribution are reconstructed from the predicted probabilities of the 5 levels.

### Key Designs

1. **Distribution-based Soft Label Construction**
    - **Function**: Discretize the continuous quality score distribution into a probability distribution across 5 levels, minimizing information loss.
    - **Mechanism**: For a quality score $x \sim \mathcal{N}(\mu, \sigma^2)$, with 5 center points $c_i \in \{1,2,3,4,5\}$ serving as level centers and a width $d=1$, the probability of the score falling into each level is calculated as: $p_i^{raw} = \int_{c_i-d/2}^{c_i+d/2} f(x)dx$. A linear transformation $p_i = \alpha p_i^{raw} + \beta$ is then applied for post-adjustment under the constraints $\sum p_i = 1$ and $\sum p_i c_i = \mu$, ensuring the expectation of the discretized distribution is exactly equal to the original MOS.
    - **Design Motivation**: The discretization error (L1 Error) of one-hot labels is approximately 0.3, whereas that of soft labels is only about 0.01-0.02. Soft labels naturally preserve the relative relationships between images: images with similar quality yield similar distributions, while those with different qualities show distinct differences in their distributions.
    - **Loss & Training**: A KL divergence loss is applied at the level token: $\mathcal{L}_{kl} = -\sum_i p_i \log(p_i^{pred}/p_i)$.

2. **Fidelity Loss Based on Thurstone Model (Multi-dataset Joint Training)**
    - **Function**: Resolve the distribution discrepancies among different IQA datasets, enabling the model to be effectively trained jointly on multiple datasets.
    - **Mechanism**: Sample two images, A and B, from the same dataset. Compute the predicted probability that A is better than B using the predicted score distributions: $p^{pred}(A>B) = \Phi\left(\frac{\mu_A^{pred}-\mu_B^{pred}}{\sqrt{(\sigma_A^{pred})^2+(\sigma_B^{pred})^2}}\right)$. Train the model using the fidelity loss by comparing this with the ground-truth probability.
    - **Design Motivation**: Absolute scores are not comparable across different datasets (the same score represents different qualities in different datasets), but the **ordinal relationships within a dataset** are reliable. The fidelity loss learns only rankings rather than absolute values, which is naturally suited for joint training. Crucially, only models that can predict score *distributions* (rather than single scores) can employ this loss—this is precisely the unique advantage of soft labels.
    - **Final Loss**: $\mathcal{L} = \mathcal{L}_{fd} + \gamma(\mathcal{L}_{ce} + \mathcal{L}_{kl})$, where $\gamma=0.05$.

3. **Reconstructing Distribution from Discrete to Continuous**
    - **Function**: Reconstruct the continuous quality score distribution from the predicted probabilities of the 5 levels during inference.
    - **Mechanism**: $\mu^{pred} = \sum_i p_i^{pred} c_i$, $(\sigma^{pred})^2 = \sum_i p_i^{pred}(c_i - \mu^{pred})^2$, yielding the reconstructed distribution $\mathcal{N}(\mu^{pred}, (\sigma^{pred})^2)$.
    - **Design Motivation**: Output not only a single score but also the uncertainty of the score, matching the human-annotated distribution with high fidelity.

## Key Experimental Results

### Single-dataset Training for Score Regression (Tab. 3, Trained on KonIQ)

| Method | KonIQ (PLCC/SRCC) | LIVE-Wild | AGIQA-3K |
|------|-------------------|-----------|----------|
| Q-Align (one-hot) | 0.941/0.940 | 0.853/0.860 | 0.772/0.735 |
| **DeQA-Score (soft)** | **0.953/0.941** | **0.892/0.879** | **0.809/0.729** |
- In-domain gain of 1.3% PLCC, out-of-domain (LIVE-Wild) gain of 4.6% PLCC.

### Multi-dataset Joint Training for Score Regression (Tab. 4)

| Training Set | Method | KonIQ | SPAQ | KADID | PIPAL |
|-------|------|-------|------|-------|-------|
| KonIQ+SPAQ+KADID+PIPAL | Q-Align | 0.926/0.932 | 0.917/0.920 | 0.950/0.954 | 0.702/0.671 |
| KonIQ+SPAQ+KADID+PIPAL | **DeQA-Score** | **0.958/0.946** | **0.932/0.929** | **0.963/0.961** | **0.724/0.690** |
- The advantage is even more pronounced under joint training, validating the effectiveness of the fidelity loss.

### Comparison of Discretization Accuracy (Tab. 1)

| Metrics | One-hot (Q-Align) | Soft Label |
|------|-------------------|------------|
| L1 Error (KonIQ) | 0.302 | **0.008** (37.75× more precise) |
| PLCC/SRCC with MOS | 0.961/0.952 | **1.000/1.000** |

### Ablation Study (Tab. 6, Trained on KonIQ+SPAQ+KADID)

| Configuration | KonIQ PLCC | KADID PLCC | LIVE-Wild PLCC | Description |
|------|-----------|-----------|---------------|------|
| One-hot (Q-Align Baseline) | 0.945 | 0.935 | 0.887 | Original method |
| Soft label only | 0.954 | 0.951 | 0.896 | Significant improvement with +soft label |
| Soft label + Fidelity | **0.957** | **0.955** | **0.900** | Additional contribution of Fidelity loss under multiple datasets |

### Key Findings
- Soft label vs. one-hot: Across all 8 test datasets, soft labels consistently outshine one-hot labels.
- Fidelity loss contributes significantly in multi-dataset joint training, while having a limited effect in single-dataset training (as cross-dataset alignment is unnecessary).
- The average values of post-adjustment $\alpha$ and $\beta$ are close to 1 and 0, respectively, indicating minimal truncation error.
- The JS divergence between the predicted score distribution and human annotations is only 0.014 on KonIQ, compared to 0.415 for Q-Align, revealing a substantial gap.
- The semantic order of level texts is crucial: shuffling or reversing the level texts leads to a severe performance drop.

## Highlights & Insights
- **Elegant Formulation**: Replaces one-hot encoding with simple distribution discretization, significantly boosting accuracy with negligible computational overhead.
- **Distribution Prediction Capability**: For the first time, an MLLM can predict quality score distributions that are highly consistent with human annotations (JS divergence of only 0.001-0.022), rather than merely a point estimation.
- **Soft Labels Unlocking Fidelity Loss**: The two designs form a synergy—soft labels enable distribution prediction, distribution prediction makes fidelity loss applicable, and fidelity loss enables effective multi-dataset joint training.
- **Generality**: The core idea of soft labels can be extended to any task requiring continuous-value regression using MLLMs (e.g., depth estimation, age estimation).

## Limitations & Future Work
- Uses only 5 discrete levels; although the discretization precision is already high, utilizing more levels may further enhance accuracy.
- The backbone model mPLUG-Owl2 is relatively old, and the effectiveness on stronger base models (e.g., LLaVA-Next) has not been verified.
- Fidelity loss requires sampling image pairs from the same dataset; selection of sampling strategies for ultra-large datasets is not deeply explored.
- Score regression is a "low-level perception" task; the trade-off between this and the "high-level understanding" capability of MLLMs deserves further investigation.

## Related Work & Insights
- Q-Align (pioneer of one-hot discretization) -> This work directly improves upon its core discretization strategy.
- UNIQUE (creator of fidelity loss) -> This work migrates its concept to the MLLM framework for the first time.
- DepictQA series -> Focuses on linguistic descriptions rather than score regression; this work fills the gap of the latter.
- Insight: The discrete token output limitation of MLLMs is not necessarily fatal—a clever discretization strategy can bridge the continuous and discrete spaces with almost zero loss.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of discretizing distributions into soft labels is simple yet elegant, resolving a fundamental information loss issue.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with 9 IQA datasets, single/multi-dataset training, distribution prediction, ablations, and level text experiments.
- Writing Quality: ⭐⭐⭐⭐ Math derivations are clear, comparisons with Q-Align are intuitive, and diagrams/tables are well-designed.
- Value: ⭐⭐⭐⭐ The generic methodology can be generalized to any MLLM continuous-value regression task; the soft label + fidelity loss pair is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)
- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](../../AAAI2026/multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/multimodal_vlm/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)

</div>

<!-- RELATED:END -->
