---
title: >-
  [Paper Note] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration
description: >-
  [ECCV 2024][Medical Imaging][Unsupervised registration] To address the issue of spurious reconstruction errors caused by confounding factors such as noise and occlusions in unsupervised medical image registration, this paper proposes an adaptive correspondence scoring framework (AdaCS). By learning pixel-wise correspondence confidence maps to re-weight error residuals, AdaCS consistently improves the performance of three mainstream registration architectures across three data…
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Unsupervised registration"
  - "correspondence scoring"
  - "adaptive weighting"
  - "deformation estimation"
  - "cardiac images"
date: 2026-05-08
content_hash: 7023d361b2f70863
---

# Adaptive Correspondence Scoring for Unsupervised Medical Image Registration

**Conference**: ECCV 2024  
**arXiv**: [2312.00837](https://arxiv.org/abs/2312.00837)  
**Code**: [https://voldemort108x.github.io/AdaCS/](https://voldemort108x.github.io/AdaCS/)  
**Area**: Medical Image Registration  
**Keywords**: Unsupervised registration, correspondence scoring, adaptive weighting, deformation estimation, cardiac images  

## TL;DR
To address the issue of spurious reconstruction errors caused by confounding factors such as noise and occlusions in unsupervised medical image registration, this paper proposes an adaptive correspondence scoring framework (AdaCS). By learning pixel-wise correspondence confidence maps to re-weight error residuals, AdaCS consistently improves the performance of three mainstream registration architectures across three datasets in a plug-and-play manner.

## Background & Motivation
Unsupervised medical image registration relies on image reconstruction as the supervisory signal: the source image is warped by an estimated displacement field and compared against the target image to calculate loss. The core assumption of this paradigm is homogeneous intensity constancy between the two images. In practice, however, this assumption is broken by three types of factors: (1) noise and non-overlapping regions (such as acoustic shadows in ultrasound); (2) violations of the Lambertian assumption during physical wave imaging (such as reflection characteristics in ultrasound imaging); and (3) inconsistent acquisition conditions. These factors result in non-zero reconstruction errors even when the displacement estimator has correctly established the correspondence, generating spurious gradient signals that continuously push the model away from local optima during training, leading to performance degradation.

## Core Problem
During training, current unsupervised registration methods calculate reconstruction errors by treating all pixels equally, whereas many pixels cannot establish valid intensity correspondence in reality. Large error residuals from these "non-corresponding" regions introduce noisy gradients that mislead the parameter update direction of the displacement estimator. Automatically identifying which regions can establish valid correspondences and which cannot during the training process, and adapting the weights of training signals accordingly, is the key to improving unsupervised registration performance.

## Method

### Overall Architecture
The inputs are a source image $I_s$ and a target image $I_t$. A displacement estimator $f_\theta$ (which can be any architecture, such as Voxelmorph/Transmorph/Diffusemorph) predicts a displacement field $\hat{u}$ to spatially warp the source image into a reconstructed image. Before calculating the reconstruction error, a scoring estimator $g_\phi$ (a U-Net architecture) is additionally introduced. It takes only the target image $I_t$ as input and predicts pixel-wise correspondence score maps $\hat{S} \in [0,1]$ to weight the error residuals. The two networks are optimized alternately. During inference, only the displacement estimator is required, resulting in zero extra computational overhead for the scoring network.

### Key Designs

1. **Adaptive Displacement Estimation Loss**: The scoring map weighting is introduced into the standard MSE reconstruction loss: $\mathcal{L}_{de} = \frac{1}{|\Omega|}\sum_{x} \lfloor\hat{S}(x)\rfloor [I_t(x) - I_s(x+\hat{u}(x))]^2 + \lambda\|\nabla\hat{u}\|^2$. Regions with high scores (good correspondence) retain the original error weights, while regions with low scores (unable to establish correspondence) have their error weights reduced. $\lfloor\cdot\rfloor$ denotes the stop-gradient operation, which prevents the gradients of the displacement estimator from backpropagating through the scoring map.

2. **Unsupervised Correspondence Scoring Objective**: The training loss of the scoring estimator itself is $\mathcal{L}_{ucs} = \frac{1}{|\Omega|}\sum_x \hat{S}(x)[I_t(x)-I_s(x+\lfloor\hat{u}(x)\rfloor)]^2$, which encourages assigning low scores to high-residual regions. However, minimizing this term alone leads to a trivial solution (all-zero scores). Therefore, regularization is required: (a) **Anti-degeneration regularization** $\mathcal{L}_{reg} = \frac{1}{|\Omega|}\sum_x [1-\hat{S}(x)]^2$, which penalizes scores deviating from 1; (b) **Smoothness regularization** $\mathcal{L}_{smooth} = m_T \frac{1}{|\Omega|}\sum_x \|\nabla\hat{S}(x)\|^2$, which encourages spatial smoothness of the scoring maps.

3. **Momentum-Guided Adaptive Smoothness Weight**: The smoothness regularization weight $m_T$ is not a fixed scalar but is dynamically adjusted during training. First, the average residual of the current batch $\mu_T$ is mapped to $[0,1]$ via a cosine activation $b_T = \cos(\frac{\pi}{2}\mu_T)$. Then, momentum is calculated using an exponential moving average $m_T = \gamma m_{T-1} + (1-\gamma)b_T$ ($\gamma=0.99$). In the early stages of training, the residuals are large, making $m_T$ small, which allows the scoring maps to explore freely. In the later stages of training, the residuals converge, and $m_T$ increases, enforcing smoothness constraints to help convergence.

### Loss & Training
- Total loss of the scoring estimator: $\mathcal{L}_{se} = \mathcal{L}_{ucs} + \alpha\mathcal{L}_{reg} + \beta\mathcal{L}_{smooth}$
- A warm-up + alternate optimization strategy is adopted: during the first $N_w$ epochs, only the displacement estimator is trained; in the next $N_w$ epochs, only the scoring estimator is trained; after that, the two are optimized alternately. This prevents the two networks from passing noisy gradients to each other in the early stages.
- Hyperparameters: $\alpha=0.02\sim0.04$, $\beta=1.5$, which are relatively insensitive (Dice fluctuates by approximately 1%).

## Key Experimental Results

| Dataset | Architecture | Metric | Baseline | AdaCS | Gain |
|--------|------|------|------|-------|------|
| ACDC (MRI) | Voxelmorph | DSC↑ | 79.48 | **80.50** | +1.02 |
| ACDC | Transmorph | DSC↑ | 76.94 | **78.39** | +1.45 |
| ACDC | Diffusemorph | DSC↑ | 67.38 | **72.09** | +4.71 |
| CAMUS (US) | Voxelmorph | DSC↑ | 81.50 | **81.74** | +0.24 |
| CAMUS | Transmorph | DSC↑ | 79.24 | **79.64** | +0.40 |
| CAMUS | Diffusemorph | DSC↑ | 75.23 | **77.65** | +2.42 |
| OASIS (3D Brain) | Voxelmorph | DSC↑ | 67.80 | **69.32** | +1.52 |
| OASIS | c-LapIRN | DSC↑ | 76.41 | **76.87** | +0.46 |

All improvements are verified to be statistically significant via paired t-tests (p-value far less than 0.05).

### Ablation Study
- **$\mathcal{L}_{reg}$ is indispensable**: Without it, the scoring map degenerates to all zeros, rendering AdaCS completely ineffective.
- **Contribution of $\mathcal{L}_{smooth}$**: Adding it consistently yields improvements across all architectures, preventing irregular jumps of the score map between adjacent pixels that lead to unstable training signals.
- **Effect of momentum weight $m_T$**: Its inclusion improves Voxelmorph/ACDC from 80.00 to 80.50, with improvements also observed in other architectures.
- **Comparison with other robust losses** (NCC/MI/Tukey): The improvements of AdaCS on top of MSE still hold, and it can be combined with different loss functions.
- **All baseline methods (NLL, $\beta$-NLL, AdaReg, AdaFrame) actually degraded performance**, indicating that simple uncertainty estimation or statistical hypothesis-driven adaptive weighting is not suitable for this problem.

## Highlights & Insights
- **Plug-and-play design**: The scoring network is only used during training, with zero additional overhead during inference (memory and inference time remain completely unchanged), which is highly attractive for practical deployment.
- **Precise problem definition**: The "intensity constancy violation" problem in unsupervised registration is formulated as a correspondence scoring problem, rather than simple uncertainty modeling.
- **Ingenious momentum-guided smoothness strategy**: By utilizing the dynamic variation of training residuals to automatically adjust regularization strength, it avoids over-constraint in the early stages while guaranteeing convergence in the later stages.
- **Bidirectional use of stop-gradients**: The displacement estimator and the scoring estimator detach from each other, preventing gradient entanglement and maintaining a clean design.
- **Largest improvement on Diffusemorph** (+4.71 DSC), suggesting that models more sensitive to noisy gradients benefit more from AdaCS.

## Limitations & Future Work
- Hyperparameters $\alpha$ and $\beta$ still require grid search for each architecture and dataset. Although they are relatively insensitive, this increases tuning efforts.
- Failure cases exist in patients with extremely thin myocardium, as these regions require highly precise displacement predictions.
- The experiments focus only on cardiac datasets (ACDC, CAMUS, 3D Echo), lacking validation on other anatomical regions (abdomen, lung) and large-scale public 3D datasets.
- The scoring network only receives the target image $I_t$ and does not utilize the information from the source and reconstructed images, which potentially limits its ability to determine correspondence.
- Training time increases by approximately 60-70% (e.g., Voxelmorph increases from 0.26h to 0.43h), and the warm-up phase requires extra $N_w$ epochs of training.
- → Amortized hyperparameter optimization can be explored to automate tuning (mentioned by the authors in the conclusion).
- → The scoring network could attempt to receive dual-image inputs or difference maps as additional information.

## Related Work & Insights
- **vs NLL / $\beta$-NLL (aleatoric uncertainty)**: These methods model the problem as heteroscedastic Gaussian noise and perform inverse weighting by predicting variance. However, experiments demonstrate that such simple joint optimization actually hurts registration performance. AdaCS models this as a binarized problem of "whether correspondence exists" rather than estimating a noise distribution, which, combined with the alternate optimization strategy, is more effective.
- **vs AdaReg / AdaFrame (adaptive weighting)**: These methods calculate adaptive weights based on residual statistics (e.g., local/global residual ratio, z-score) and are purely driven by mathematical formulas. AdaCS replaces manual statistical formulas with a learned network, demonstrating stronger representation ability and avoiding the issue of inaccurate early weights through the warm-up strategy.
- **vs Voxelmorph / Transmorph / Diffusemorph (base models)**: AdaCS does not alter the architectures of the base models themselves but improves them solely through training strategies, providing orthogonal and complementary improvements.

## Insights
- The idea of correspondence scoring can be extended to other unsupervised learning tasks that rely on reconstruction loss (such as optical flow estimation, depth estimation, and point cloud registration).
- The design pattern of using an auxiliary network during training and discarding it during inference is also valuable in other fields (similar to the teacher-student concept in knowledge distillation, but more lightweight).
- Momentum-guided weight scheduling is a general training trick that can be transferred to other scenarios requiring gradual regularization.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing correspondence scoring as an independent module into registration training represents a new perspective, though the essence remains adaptive weighting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three architectures and three datasets with extensive ablation studies and statistical testing, but lacks verification on non-cardiac data such as abdomen/lung.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem motivation is clear, mathematical derivations are complete, and figures visually demonstrate the effectiveness of the score maps.
- Value: ⭐⭐⭐⭐ The plug-and-play and zero-inference-overhead design offers strong practicality, although the absolute performance gain is modest (1-2% DSC on cardiac data).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)
- [\[ECCV 2024\] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)
- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](../../CVPR2025/medical_imaging/sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)

</div>

<!-- RELATED:END -->
