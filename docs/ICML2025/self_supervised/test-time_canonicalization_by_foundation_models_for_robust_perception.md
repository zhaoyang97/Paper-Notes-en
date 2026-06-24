---
title: >-
  [Paper Note] Test-Time Canonicalization by Foundation Models for Robust Perception
description: >-
  [ICML 2025][Self-Supervised Learning][Test-time optimization] This work proposes the FoCal framework, which leverages visual priors from CLIP and Stable Diffusion during the inference phase. Utilizing a "Vary-then-Rank" strategy, it transforms input images into their most visually canonical versions, enhancing downstream model robustness to variations in viewpoint, illumination, and rotation without any retraining.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Test-time optimization"
  - "canonicalization"
  - "foundation models"
  - "robust perception"
  - "energy function"
date: 2026-05-08
content_hash: 7255d7ef2fb2ee2e
---

# Test-Time Canonicalization by Foundation Models for Robust Perception

**Conference**: ICML 2025  
**arXiv**: [2507.10375](https://arxiv.org/abs/2507.10375)  
**Code**: [https://github.com/sutkarsh/focal](https://github.com/sutkarsh/focal)  
**Area**: Self-Supervised Learning  
**Keywords**: Test-time optimization, canonicalization, foundation models, robust perception, energy function

## TL;DR

This work proposes the FoCal framework, which leverages visual priors from CLIP and Stable Diffusion during the inference phase. Utilizing a "Vary-then-Rank" strategy, it transforms input images into their most visually canonical versions, enhancing downstream model robustness to variations in viewpoint, illumination, and rotation without any retraining.

## Background & Motivation

In real-world scenarios, robotics and autonomous driving systems require stable object perception under varying viewpoints, illuminations, and environmental conditions. However, even large-scale foundation models like CLIP and SAM exhibit vulnerabilities; for instance, CLIP misclassifies objects from unconventional viewpoints, while SAM fails to segment side-facing objects. This vulnerability stems from **photographer's bias** in the training data, where internet images are heavily concentrated on frontal/upright poses under ideal lighting conditions.

Existing solutions fall into two main categories, each with inherent limitations:

- **Data Augmentation (DA)**: Requires predefined transformation types, performs poorly on rare classes, and may over-regularize, thereby harming performance on certain categories.
- **Equivariant Networks**: Hardcode mathematical symmetries into the network architecture, but fail to scale to complex real-world transformations such as 3D viewpoint changes.

The fundamental issue with both approaches is that invariance is fixed during training, making them unable to adapt to novel transformations outside the training distribution. Inspired by human "mental rotation"—where humans mentally rotate unfamiliar objects to a canonical perspective to identify them—this work proposes dynamically achieving invariance at test time.

## Method

### Overall Architecture

FoCal (Foundation-model guided Canonicalization) adopts a two-stage **"Vary-then-Rank"** strategy:

1. **Vary Stage**: Generates a set of candidate transformed versions for the input image (e.g., images at different rotation angles, or 3D renderings from different viewpoints).
2. **Rank Stage**: Uses an energy function constructed from foundation models to score all candidates, selecting the version with the lowest energy (most "canonical") as the canonical form.
3. Feeds the canonicalized image into downstream models (e.g., CLIP classification, SAM segmentation) for inference.

The core optimization objective is:

$$t^* = \arg\min_{t \in \mathcal{T}} E_{\text{FoCal}}(t(\mathbf{x}))$$

$$\mathbf{y} = f(t^*(\mathbf{x}))$$

where $\mathcal{T}$ represents the set of transformations, $E_{\text{FoCal}}$ is the energy function, and $f$ is the downstream task model.

### Key Designs

**1. Theoretical Foundation of Canonicalization**

Based on the formulation by Kaba et al. (2022), the canonicalization function is defined as $h(\mathbf{x}) = \arg\min_{t \in \mathcal{T}} E(t(\mathbf{x}))$. Under mild conditions, this function can be proven to satisfy invariance/equivariance. The key insight is that all transformed versions of an image define a "slice" of the natural image distribution, where certain versions appear with a higher frequency in real-world data. Foundation models implicitly learn this distributional prior.

**2. CLIP Energy Function**

Viewing CLIP as an energy-based model, the unconditional energy is defined as a combination of mean and maximum logits:

$$E_{\text{CLIP}}(\mathbf{x}; \alpha, \beta) = (\alpha \cdot \text{mean} - \beta \cdot \text{max})_{c \in \{1,...,|C|\}} f_\theta(\mathbf{x})[c]$$

where $\alpha, \beta$ are hyperparameters. A CLIP ViT-H-14 is used, where the cosine similarity of the image-text embeddings serves as the logit. CLIP energy focuses on semantics, selecting the candidate image closest to a predefined category.

**3. Diffusion Model Energy Function**

Energy is extracted from Stable Diffusion 2 as follows:

$$E_{\text{diff}}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_{\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right]$$

In practice, 5-10 denoising steps are sufficient. The diffusion energy acts as a general appearance prior.

**4. Joint Energy Function**

The two energies are combined via weighted summation:

$$E_{\text{FoCal}}(t(\mathbf{x})) = \gamma_1 \cdot E_{\text{CLIP}}(t(\mathbf{x})) + \gamma_2 \cdot E_{\text{diff}}(t(\mathbf{x}))$$

**5. Candidate Generation under Different Transformations**

- **2D Rotation**: Directly enumerates $C_8$ (8 discrete rotation angles).
- **3D Viewpoint**: Uses the TRELLIS generative model to render multi-view images on a sphere at every 30° interval (yielding 60 candidates).
- **Color/Contrast**: Samples in the log-chrominance and gamma spaces.
- **Day/Night Transformation**: Interpolates in the Stable Diffusion latent space.

### Loss & Training

FoCal is a **completely training-free** framework, involving no gradient updates or parameter fine-tuning. Its "training strategy" is embodied in:

- **Bayesian Optimization (BO)**: For continuous/high-dimensional transformation spaces (such as 2D color space, 6D active vision), Gaussian Processes (GP) with an RBF kernel and the Expected Improvement acquisition function are utilized. Typically, 50-100 evaluations are sufficient to find a good solution, avoiding brute-force search.
- **Hyperparameter Selection**: $\alpha=1, \beta=0.5$ is suitable for most classification scenarios; $\gamma_1=0.54, \gamma_2=0.67$ is used for segmentation tasks, tuned via BO on a small validation set.
- **Assumptions**: (1) At least one in-distribution image exists within the set of transformations; (2) foundation models assign lower energy to in-distribution images; (3) downstream models perform optimally on in-distribution data.

## Key Experimental Results

### Main Results

**3D Viewpoint Robustness (Objaverse-LVIS & CO3D)**

| Dataset | Metric | Ours (FoCal) | Prior Methods | Gain |
|---|---|---|---|---|
| Objaverse-LVIS (Worst-case) | Classification Accuracy | 62.0% | 12.0% (OV-Seg) | +50.0% |
| Objaverse-LVIS (Overall Top-10) | Mean Accuracy | 84.5% | 76.4% (TTA-10) | +8.1% |
| CO3D (t=0.3) | Classification Accuracy | 49.5% | 45.9% (TRELLIS) | +3.6% |
| CO3D (t=0.5) | Classification Accuracy | 55.3% | 53.4% (TRELLIS) | +1.9% |

**2D Rotation (vs PRLC, under PRLC training configurations)**

| Dataset | Architecture | FoCal Rotation Accuracy | PRLC Rotation Accuracy | Gain |
|---|---|---|---|---|
| CIFAR10 | ResNet-50 | 95.6% | 95.1% | +0.5% |
| CIFAR10 | ViT | 96.0% | 94.8% | +1.2% |
| CIFAR100 | ResNet-50 | 82.2% | 81.8% | +0.4% |
| CIFAR100 | ViT | 84.4% | 82.2% | +2.2% |
| ImageNet (ViT) | ViT | 71.9% | 60.5% | +11.4% |

### Ablation Study

| Energy Configuration | Pose Accuracy | Pose Error | Description |
|---|---|---|---|
| CLIP Energy Only | 68.9% | 37.1° | Semantic priors are insufficient for precise localization |
| Diffusion Energy Only | 82.7% | 22.6° | Appearance priors are more effective |
| CLIP + Diffusion (Full FoCal) | 89.5% | 13.5° | Mutually complementary, reducing error by 64% |

| Method | CIFAR10 | CIFAR100 | STL10 | Description |
|---|---|---|---|---|
| No Correction | 65.4 | 50.6 | 93.4 | Baseline |
| Ours | 93.7 | 76.2 | 97.5 | Substantial improvement |
| TTA | 82.8 | 61.7 | 96.6 | FoCal outperforms TTA by 10-15% |

### Key Findings

1. **Zero-Shot Outperforming Supervised Canonicalizers**: FoCal matches or outperforms PRLC across all PRLC training settings (6 datasets × architecture combinations), despite requiring no training whatsoever.
2. **Strong Cross-Dataset Generalization**: While PRLC exhibits a 12-18% drop in pose accuracy during cross-dataset transfer, FoCal's performance varies by less than 3%.
3. **Efficacy on Segmentation Tasks**: Achieves comparable mAP (65.9) to PRLC on COCO, while improving pose accuracy by 2.1%.
4. **Day/Night Transformations**: Using only the "street" category CLIP energy, it selects daytime images with a 91% probability.
5. **Active Vision**: In 6-DoF virtual scenes, the camera naturally focuses on salient objects and maintains an upright perspective.

## Highlights & Insights

- **Paradigm Innovation**: Shifts invariance from training-time hardcoding to inference-time optimization, analogous to test-time compute scaling in LLMs.
- **Theoretical Elegance**: Leverages the energy minimization framework of Kaba et al., which theoretically guarantees invariance/equivariance without requiring the energy function itself to be equivariant.
- **"Slice" Intuition**: The family of transformations defines a "slice" of the natural image distribution, and the foundation model energy function can find the most probable point on this slice. This perspective unifies various transformations such as rotation, color, and viewpoint.
- **Practicality**: Fully plug-and-play, demanding no modifications to downstream model architectures or retraining.

## Limitations & Future Work

1. **High Computational Overhead**: Requires evaluating CLIP + SD energy for each candidate transformation. 2D rotation incurs approximately 56× inference overhead, while 3D viewpoint tasks involve TRELLIS generation taking about 13.3 seconds per sample. This issue can be mitigated using a System-1/2 strategy (first determining if canonicalization is necessary).
2. **Manual Selection of Transformations**: Currently, it requires manual decision-making on which transformation generator to use (rotation, viewpoint, or color). Future work should focus on automatic detection.
3. **Limitations of Non-invertible Transformations**: The theoretical framework assumes invertible transformations, whereas 3D viewpoint transformations are non-invertible, offering only approximate invariance.
4. **Inferior Color Correction compared to Dedicated Methods**: On the RCC dataset, it yields a median angular error of 6.4°, which falls short of Barron & Tsai's 1.3°. This is because FoCal optimizes for "visual typicality" rather than "color neutrality."
5. **Parallelization Demands**: Although all candidate evaluations can theoretically be parallelized, it requires substantial GPU memory in practice.

## Related Work & Insights

- **Kaba et al. (2022)** & **PRLC (Mondal et al., 2023)**: Lay the theoretical and practical foundations for learning-based canonicalization, but require training on specific datasets/transformations, restricting overall generalization.
- **Grathwohl et al. (2020)**: Provide the theoretical foundation for interpreting classifiers as energy-based models, which FoCal directly adopts.
- **Graikos et al. (2022)**: Establish diffusion models as plug-and-play priors, which FoCal leverages to construct the energy functions.
- **Test-time compute scaling (Snell et al., 2024; Zaremba et al., 2025)**: FoCal can be viewed as an application of test-time compute scaling in the vision domain—generating multiple candidates and utilizing a learned ranker to select the best option.
- **Insight**: This work pioneeringly introduces test-time search/optimization to the domain of visual robustness, which may inspire more "inference-time adaptive" vision methodologies.

## Rating

| Dimension | Rating (1-5) | Description |
|---|---|---|
| Novelty | ⭐⭐⭐⭐⭐ | Shifting invariance from training-time constraints to test-time optimization is a paradigm-level innovation. |
| Theoretical Depth | ⭐⭐⭐⭐ | The energy minimization framework is backed by solid theoretical guarantees, though the portion addressing approximate invariance is relatively weaker. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Covers 5 types of transformations, 4+ datasets, and 3 downstream models with thorough ablation studies. |
| Value | ⭐⭐⭐⭐ | Zero-training, plug-and-play utility; however, the computational cost represents a bottleneck for practical deployment. |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear motivation, excellent illustrations, and a strong integration of intuition and theory. |
| Overall Rating | ⭐⭐⭐⭐⭐ | Pioneering work in the field of visual robustness, with the potential to be a highly influential paper. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[ICML 2025\] Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions](beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h.md)

</div>

<!-- RELATED:END -->
