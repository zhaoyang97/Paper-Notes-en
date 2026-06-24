---
title: >-
  [Paper Note] Active Generation for Image Classification
description: >-
  [ECCV 2024][Active learning] This paper proposes ActGen, which integrates the concept of active learning into the image generation process of diffusion models. By identifying misclassified validation samples as guidance images and combining attentive guidance with gradient-based generation control, ActGen achieves a +2.26% accuracy improvement on ImageNet using only 10% generated images, outperforming previous methods that utilize 94% synthetic data.
tags:
  - "ECCV 2024"
  - "Active learning"
  - "diffusion models"
  - "data augmentation"
  - "image generation"
  - "hard sample mining"
date: 2026-05-08
content_hash: 466e97d9702dc03a
---

# Active Generation for Image Classification

**Conference**: ECCV 2024  
**arXiv**: [2403.06517](https://arxiv.org/abs/2403.06517)  
**Code**: [https://github.com/hunto/ActGen](https://github.com/hunto/ActGen)  
**Area**: Data Augmentation / Image Classification  
**Keywords**: Active learning, diffusion models, data augmentation, image generation, hard sample mining

## TL;DR
This paper proposes ActGen, which integrates the concept of active learning into the image generation process of diffusion models. By identifying misclassified validation samples as guidance images and combining attentive guidance with gradient-based generation control, ActGen achieves a +2.26% accuracy improvement on ImageNet using only 10% generated images, outperforming previous methods that utilize 94% synthetic data.

## Background & Motivation
Deep generative models, especially diffusion models, show significant potential in improving image classification accuracy. However, existing methods suffer from severe efficiency issues: they require generating synthetic images at a scale comparable to the original dataset size, but yield only marginal accuracy gains. For example, Azizi et al. used 1.2 million synthetic images (nearly the size of the original dataset) on ImageNet and obtained an accuracy increase of only 1.78%. This contradiction between massive computational overhead and trivial gains severely limits the practicality of synthetic data methods. The core problem lies in the fact that existing methods **generate images indiscriminately and randomly**, resulting in a high degree of redundancy in synthetic samples that contribute little to model training. The key insight of this paper is to borrow ideas from active learning—specifically, generating only the samples that the model truly needs, i.e., variants of the hard samples currently misclassified by the model. Core Idea: **generating a small number of precise hard samples is much more effective than generating a large volume of random samples**.

## Method

### Overall Architecture
ActGen adopts a training-aware online generation strategy: after each training epoch, a validation set is used to evaluate the model to identify misclassified samples. These samples serve as guidance images to generate similar hard samples via a diffusion model, which are then added to the training set for the next epoch of training. The generation process combines two guidance mechanisms: attentive image guidance (retaining foreground objects while diversifying backgrounds) and gradient-based generation guidance (controlling generation difficulty and diversity using contrastive loss and classification loss).

### Key Designs
1. **Active Hard Sample Generation Strategy**:
    - **Function**: Accurately identify and enhance the training samples most needed by the model.
    - **Mechanism**: A validation set is partitioned from the training set. After each training epoch, the model is evaluated on the validation set, and misclassified samples are extracted as prototypes of hard samples. These samples typically feature incomplete objects, unusual poses, or rare intra-class patterns.
    - **Design Motivation**: Draw from key findings in active learning and curriculum learning—training on batches of hard samples converges faster than on random batches. Therefore, generating variants of these hard samples maximizes performance gains with a minimal number of samples.

2. **Attentive Image Guidance**:
    - **Function**: Guide the generation of new images similar to the misclassified samples during the diffusion denoising process.
    - **Mechanism**: In each denoising step of DDPM, the latent of the generated image is interpolated with the latent of the guidance image to obtain the final latent feature $\tilde{x}_{t-1} = x_{t-1} + m_t \odot \gamma_t(x_{t-1}^{(g)} - x_{t-1})$. The attention map of the class prompt in the cross-attention layer serves as the foreground mask $m_t$, restricting guidance to the foreground area and allowing the background to be generated freely.
    - **Design Motivation**: Directly applying pixel-level guidance to the entire image leads to a lack of diversity in background scenes. Using an attention mask for selective guidance preserves foreground object characteristics while generating diverse backgrounds.

3. **Gradient-based Guidance**:
    - **Function**: Control the diversity and classification difficulty of the generated images using loss functions.
    - **Mechanism**: Two loss functions are designed to update the text embedding via gradients: $c_{t-1} = c_t - \nu \frac{\nabla_{c_t}\mathcal{L}}{||\nabla_{c_t}\mathcal{L}||_2}$. A contrastive loss $\mathcal{L}_{contra}$ utilizes a memory bank to store the latents of already generated images, penalizing current generations that are too similar to existing ones. An adversarial loss $\mathcal{L}_{adv} = -\text{CE}(\Omega(o_t), y)$ maximizes the classification loss to make the generated images more challenging for the current model.
    - **Design Motivation**: Relying solely on image guidance easily produces redundant images. Contrastive loss ensures diversity, while adversarial loss guarantees that the generated samples are truly challenging hard samples that benefit model learning.

### Loss & Training
The total loss for generation guidance is $\mathcal{L} = \mathcal{L}_{contra} + \lambda \mathcal{L}_{adv}$, where $\lambda$ is a balancing factor. The contrastive loss employs Euclidean distance with a margin $\rho=200$. Gradient updates are normalized and applied to the text embeddings. In terms of training strategy, generation is only performed during the first half of the epochs (since the learning rate is small in the latter half, newly generated samples have limited impact). Each GPU generates 64 images after each epoch. The validation set size is 10K.

## Key Experimental Results

### Main Results
ImageNet classification results:

| Model | Real Data | Generated Data | Gen/Real Ratio | Accuracy | Gain |
|------|---------|---------|-------------|--------|------|
| ResNet-50 (Real only) | 1.28M | 0 | 0% | 76.39% | - |
| Azizi et al. (Imagen) | 1.28M | 1.2M | 94% | 78.17% | +1.78% |
| **ActGen (Ours)** | **1.28M** | **0.13M** | **10%** | **78.65%** | **+2.26%** |
| ViT-S/16 (Real only) | 1.28M | 0 | 0% | 79.89% | - |
| Azizi et al. (Imagen) | 1.28M | 1.2M | 94% | 81.00% | +1.11% |
| **ActGen (Ours)** | **1.28M** | **0.08M** | **6%** | **81.12%** | **+1.23%** |

### Ablation Study

| Configuration | ImageNet Accuracy | Description |
|------|---------------|------|
| Baseline (Real only) | 76.39% | No synthetic data |
| Random SD Gen | 76.64% | +0.25%, limited effect from random generation |
| + Image Guidance (IG) | 77.93% | +1.54%, image guidance contributes the most |
| + Attentive IG (AIG) | 78.15% | +0.22%, foreground mask improves diversity |
| + Contrastive loss $\mathcal{L}_{contra}$ | 78.36% | +0.21%, reduces redundancy |
| + Adversarial loss $\mathcal{L}_{adv}$ | 78.65% | +0.29%, increases classification difficulty |

### Key Findings
- **Significant efficiency improvement**: ActGen outperforms prior methods while using only about 10% of their synthetic images (surpassing Azizi et al. on ImageNet by +0.48% while saving approximately 1 million synthetic images).
- Performance stabilizes when the validation set size is above 5K, indicating that a large validation set is not required to identify hard samples.
- In few-shot scenarios (EuroSAT), ActGen matches or even outperforms the Real guidance method using only 2K generated images compared to 8K.
- Additional generation computational overhead is manageable: compared to conventional training, it adds about 58% more GPU time (15.2 vs 9.6 GPU days), which is far superior to generating 10x the data (~40 GPU days).

## Highlights & Insights
- Bringing the concept of active learning into generative data augmentation is a clean and powerful methodological innovation—focusing on generating the *right* data rather than *more* data.
- The use of the attention mask is highly elegant: it leverages the foreground spatial information naturally encoded in the cross-attention maps of the diffusion model, without requiring any external segmentation models.
- Adversarial sample generation does not inject noise into the image; instead, it modifies the generated semantics by updating gradient steps on the text embedding, producing more natural hard samples (e.g., blur, occlusion, or style changes).
- Training-aware online generation avoids the domain gap issues typical of two-stage methods.

## Limitations & Future Work
- The computational overhead of the method is still non-negligible, requiring the diffusion model to run after each epoch, which remains a bottleneck for large-scale training.
- Currently evaluated only on classification tasks; whether downstream tasks like detection and segmentation also benefit remains to be explored.
- Semantic-level similarity measures are not considered, as the contrastive loss uses global latent distances.
- Partitioning the validation set from the training set reduces the available training data, which could negatively impact small datasets.
- The quality of generated images depends heavily on the base diffusion model's generation capacity for specific categories; performance may degrade on classes where generation is weak.

## Related Work & Insights
- The biggest difference from Azizi et al.: transitioning from "generating enough" to "generating well", greatly reducing computational costs.
- Similar to Real guidance in using real images to guide the diffusion process, but introducing a training-aware active selection mechanism.
- Insight: The key to data augmentation is targeted quality rather than sheer quantity; the concept of hard sample mining is equally crucial in generative augmentation.
- Future directions: Can combine more advanced diffusion models (such as SDXL) to further improve generation quality, or generalize the framework to other vision tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The integration of active learning and diffusion-based generation is novel, with clever designs in attentive guidance and gradient control.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three datasets (ImageNet, CIFAR, EuroSAT) and various model architectures, with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, with solid motivation explanations and intuitive diagram designs.
- **Value**: ⭐⭐⭐⭐ Significantly lowers the computational barrier of generative data augmentation, offering practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](../../CVPR2026/others/prototype-based_causal_intervention_for_multi-label_image_classification.md)
- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)
- [\[ECCV 2024\] COIN-Matting: Confounder Intervention for Image Matting](coin-matting_confounder_intervention_for_image_matting.md)

</div>

<!-- RELATED:END -->
