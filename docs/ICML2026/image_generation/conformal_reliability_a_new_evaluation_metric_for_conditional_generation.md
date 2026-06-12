---
title: >-
  [Paper Note] Conformal Reliability: A New Evaluation Metric for Conditional Generation
description: >-
  [ICML2026][Image Generation][Reliability Evaluation] Ours proposes CReL, a reliability score based on Conformal Prediction. By constructing convex prediction sets in the latent space and optimizing for worst-case metric…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Reliability Evaluation"
  - "Conformal Prediction"
  - "Conditional Generation"
  - "Worst-case Analysis"
  - "Uncertainty Quantification"
date: 2026-05-08
content_hash: 5fb87d09811d43f3
---

# Conformal Reliability: A New Evaluation Metric for Conditional Generation

**Conference**: ICML2026  
**arXiv**: [2605.30807](https://arxiv.org/abs/2605.30807)  
**Code**: https://ggc29.github.io/CReL/ (Available)  
**Area**: Image Generation  
**Keywords**: Reliability Evaluation, Conformal Prediction, Conditional Generation, Worst-case Analysis, Uncertainty Quantification  

## TL;DR
Ours proposes CReL, a reliability score based on Conformal Prediction. By constructing convex prediction sets in the latent space and optimizing for worst-case metric performance, it achieves uncertainty-aware evaluation for conditional generative models. It reveals model reliability differences in image-text tasks that traditional single-output metrics fail to capture.

## Background & Motivation

**Background**: Conditional generative models (Text-to-Image, Image-to-Text, etc.) have made significant progress. Current mainstream evaluation metrics such as CLIP Score, BERT-SIM, and FID typically evaluate only the quality of a single generated output, reflecting the "average performance" of the model.

**Limitations of Prior Work**: Generative models possess intrinsic stochasticity—the same input can produce drastically different outputs under different sampling seeds. A model might achieve a high average score but still have a non-negligible probability of catastrophic failure. For example, in an Image-to-Text task, a model might usually generate "a person playing a guitar," but under certain seeds, it might produce "a person holding a gun." In safety-critical scenarios, single-output evaluation fails to quantify this tail risk.

**Key Challenge**: Existing metrics measure "how good a model can be," whereas reliability should measure "how bad a model can be at its worst." However, directly constructing prediction sets in high-dimensional output spaces and optimizing for worst-case metrics faces the dual difficulties of the curse of dimensionality and non-convex optimization.

**Goal**: Define a reliability score that accounts for uncertainty to quantify the worst-case performance of a model at a given confidence level $1-\alpha$, and provide an efficient computational framework.

**Key Insight**: Map high-dimensional outputs to a low-dimensional latent space, use Directional Quantile Regression (DQR) to construct convex prediction regions, and then ensure coverage guarantees through conformal calibration. Convexity allows the worst-case optimization to be solved using Projected Gradient Descent.

**Core Idea**: Construct convex prediction sets in the latent space that satisfy coverage guarantees, transforming the originally intractable high-dimensional non-convex reliability optimization problem into a solvable optimization problem over convex constraints.

## Method

### Overall Architecture
The input to the CReL framework is a conditional generative model $f$ to be evaluated and a user-specified similarity metric $\rho$. The output is the reliability score at the confidence level $1-\alpha$. The entire process is divided into four steps: (1) training a Latent Generative Model (LGM) to map high-dimensional outputs to a low-dimensional latent space; (2) using DQR in the latent space to construct convex quantile regions; (3) expanding the regions through conformal calibration to satisfy $1-\alpha$ coverage; (4) optimizing the worst-case metric using Projected Gradient Descent over the calibrated convex prediction set.

### Key Designs

1. **Latent Space Conformal Calibration**:

    - **Function**: Transforms the intractable problems of prediction set construction and calibration in the high-dimensional output space into solvable problems in the low-dimensional latent space.
    - **Mechanism**: The training data is split into three folds: $\mathcal{I}_{\text{lgm}}$ to train the VAE encoder/decoder, $\mathcal{I}_{\text{dqr}}$ to train the DQR model, and $\mathcal{I}_{\text{cal}}$ for conformal calibration. An encoder $\mathcal{E}$ maps outputs $\hat{Y}$ to latent variables $Z \in \mathbb{R}^r$. DQR estimates the $\alpha$-quantile halfspace $\mathbb{H}_u^+(x)$ for each direction $\mathbf{u} \in \mathbb{S}^{r-1}$. The intersection yields the convex region $R_\mathcal{Z}(x) = \bigcap_{\mathbf{u}} \mathbb{H}_u^+(x)$. Since the intersection of multiple directions leads to coverage below $1-\alpha$, the distance $E_i^+$ from each sample in the calibration set to $R_\mathcal{Z}$ is computed. The $\lceil(|\mathcal{D}_{\text{cal}}|+1)(1-\alpha)\rceil$ quantile is taken as $\gamma_{\text{cal}}$, expanding the region to $S^{\gamma_{\text{cal}}}(x)$.
    - **Design Motivation**: Calibration in the original output space requires grid discretization, with computational costs growing exponentially with dimension. Convex regions in the latent space allow efficient calculation of the projection distance $E_i^+$ via linear programming.

2. **Reliability Score Definition and Optimization**:

    - **Function**: Quantifies the worst-case performance of the model at the confidence level $1-\alpha$.
    - **Mechanism**: The reliability score is defined as $\min_{z \in C_\mathcal{Z}(X_{n+1})} \rho(\mathcal{D}ec(z; X_{n+1}), \text{GT}_{n+1})$, which involves finding the output within the calibrated prediction set that yields the worst metric $\rho$. Since $C_\mathcal{Z}$ is a compact convex set, the projection operator can be efficiently calculated via linear programming: first solving $y^* = \arg\min_{y_1 \in R_\mathcal{Z}(x)} \|y_1 - y\|_2$ (linear programming), then shifting along the direction by $\gamma_{\text{cal}}$. Projected Gradient Descent with multiple restarts (50) is used for the solution, with experiments showing a standard deviation of only 0.00027.
    - **Design Motivation**: In the original optimization, both $\rho$ and the constraint set $C_\mathcal{Y}$ are non-convex and intractable. By reformulating this in the latent space as a non-convex objective with convex constraints, global convergence guarantees of Projected Gradient Descent can be utilized.

3. **Coverage Theory Guarantee**:

    - **Function**: Proves that the calibrated prediction set satisfies $\mathbb{P}(\hat{Y}_{n+1} \in C_\mathcal{Y}(X_{n+1})) \geq 1-\alpha$.
    - **Mechanism**: Based on exchangeability, it is proven that the latent space coverage $\mathbb{P}(Z_{n+1} \in S^{\gamma_{\text{cal}}}) \geq 1-\alpha$. When the LGM accurately recovers the conditional distribution $\hat{Y}|X$, the decoder mapping preserves the coverage. The upper bound is $1-\alpha + 1/(1+|\mathcal{D}_{\text{cal}}|)$, which approaches the target as the calibration set grows.
    - **Design Motivation**: Compared to calibration in the output space by Feldman et al., latent space calibration may be slightly conservative due to decoder expansion, but it gains the critical advantage of optimization solvability.

## Key Experimental Results

### Synthetic Data Calibration Results

| Method | $\alpha$ | Coverage-$\mathcal{Z}$ | Coverage-$\mathcal{Y}$ | Region Area |
|------|----------|----------------------|----------------------|----------|
| CReL (Ours) | 0.10 | 0.8953 | 0.8915 | **232.7** |
| Feldman | 0.10 | — | 0.8940 | 234.5 |
| DQR | 0.10 | 0.8823 | 0.9145 | 287.4 |
| CReL (Ours) | 0.02 | 0.9770 | 0.9760 | **398.5** |
| DQR | 0.02 | 0.9818 | 0.9872 | 749.1 |

### Image-to-Text Reliability Evaluation ($\alpha=0.1$)

| Model | CLIP-SIM | CReL-CLIP | BERT-SIM | CReL-BERT |
|------|----------|-----------|----------|-----------|
| BLIP-base | 0.2330 (4th) | **0.0070 (1st)** | 0.8349 (3rd) | 0.6335 (3rd) |
| BLIP-large | 0.2453 (3rd) | −0.0074 (4th) | 0.8106 (4th) | 0.5631 (4th) |
| GIT-base | 0.2511 (2nd) | −0.0021 (2nd) | **0.8620 (2nd)** | **0.6474 (1st)** |
| GIT-large | 0.2550 (1st) | −0.0043 (3rd) | **0.8649 (1st)** | 0.6459 (2nd) |

### Key Findings
- **Ranking Inversion Phenomenon**: BLIP-base ranks lowest in average CLIP-SIM score (0.2330) but first in CReL-CLIP (0.0070) because its score distribution is more concentrated, resulting in better worst-case performance.
- **Region Area Advantage**: The prediction set area of CReL (232.7) is significantly smaller than DQR (287.4) and comparable to Feldman (234.5), indicating that joint calibration yields more compact information sets.
- **Scalability**: Unlike Feldman's grid method, which grows exponentially in high dimensions, the runtime of CReL’s latent space calibration grows linearly with dimensionality.
- **Text-to-Image Tasks**: Similar inversions were observed. SD3-M ranks third in CLIP-SIM but first in CReL-CLIP, while Kandinsky-2.2 has the highest average but ranks third in reliability.

## Highlights & Insights
- **Redefining Reliability as a Worst-case Problem**: Moving beyond traditional average metrics, the framework uses Conformal Prediction to quantify tail risks of generative models. This concept is simple and applicable to any user-specified metric $\rho$. This perspective has direct value for evaluating generative models in safety-critical scenarios (medicine, autonomous driving).
- **Latent Space Convexification Strategy**: Converting non-convex high-dimensional problems into low-dimensional optimization with convex constraints via the LGM+DQR combination is an elegant engineering-theoretical balance. Reducing the projection operator to linear programming makes the entire framework practically operational.
- **Discovering Model Ranking Inversions**: This provides practical guidance, showing that models with high average scores are not necessarily reliable. Distributional concentration is a key characteristic of reliability, which is transferable to any scenario requiring the evaluation of generative consistency.

## Limitations & Future Work
- LGM requires additional training (VAE encoder/decoder), increasing evaluation costs, and the coverage guarantee depends on the assumption of LGM reconstruction quality.
- Currently, evaluation is limited to image-text tasks on MS-COCO, without covering more complex conditional generation scenarios like video generation or 3D reconstruction.
- Conformal prediction provides marginal coverage guarantees rather than conditional coverage, which may not be strict enough for specific difficult inputs.
- Could be extended to many-to-many mapping scenarios (video, robot control), but requires designing new joint latent space representations and calibration strategies.

## Related Work & Insights
- Feldman et al. (2023) calibrate multi-output quantile regression in the output space, where non-convexity makes optimization difficult; CReL gains convexity by moving to the latent space.
- Directional Quantile Regression (DQR) (Kong & Mizera, 2012) provides the foundation for constructing convex prediction sets but is too conservative in high dimensions.
- PCP (Wang et al., 2022b) constructs prediction sets for conditional generative models, but its coordinate-wise calibration may be more conservative than joint latent space calibration (area 854.24 vs 232.70).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conf-Gen: Conformal Uncertainty Quantification for Generative Models](conf-gen_conformal_uncertainty_quantification_for_generative_models.md)
- [\[CVPR 2026\] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric](../../CVPR2026/image_generation/shoe_semantic_hoi_open-vocabulary_evaluation_metric.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)
- [\[ICML 2026\] HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing](holofair_unified_t2i_fairness_evaluation_and_fair-grpo_debiasing.md)

</div>

<!-- RELATED:END -->
