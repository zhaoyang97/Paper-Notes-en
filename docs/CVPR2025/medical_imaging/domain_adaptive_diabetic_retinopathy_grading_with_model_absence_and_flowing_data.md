---
title: >-
  [Paper Note] Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data
description: >-
  [CVPR 2025][Medical Imaging][Domain Adaptation] This paper proposes GUES (Generative Unadversarial Examples) to improve the performance of frozen source models on target domain diabetic retinopathy (DR) grading under the extreme scenario of Online Model-aGnostic Domain Adaptation (OMG-DA), where target data arrives in a streaming fashion without accessing source model parameters and labels. Specifically, the method generates personalized unadversarial perturbations via a VAE…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Domain Adaptation"
  - "Diabetic Retinopathy"
  - "Model Absence"
  - "Unadversarial Examples"
  - "Online Adaptation"
date: 2026-05-08
content_hash: b0a1ae9e366325bb
---

# Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data

**Conference**: CVPR 2025  
**arXiv**: [2412.01203](https://arxiv.org/abs/2412.01203)  
**Code**: None  
**Area**: Medical Image  
**Keywords**: Domain Adaptation, Diabetic Retinopathy, Model Absence, Unadversarial Examples, Online Adaptation

## TL;DR

This paper proposes GUES (Generative Unadversarial Examples) to improve the performance of frozen source models on target domain diabetic retinopathy (DR) grading under the extreme scenario of Online Model-aGnostic Domain Adaptation (OMG-DA), where target data arrives in a streaming fashion without accessing source model parameters and labels. Specifically, the method generates personalized unadversarial perturbations via a VAE and utilizes saliency maps as pseudo-supervision.

## Background & Motivation

**Background**: Diabetic Retinopathy (DR) is a major cause of blindness worldwide, and deep learning has achieved significant progress in automated DR grading. However, domain shifts caused by different devices, ethnicities, and testing times severely degrade the generalization ability of models in real-world clinical scenarios. Existing domain adaptation methods include UDA (requiring source domain data), DG (requiring annotated data from multiple source domains), and SFDA (requiring source model parameters).

**Limitations of Prior Work**: In practical clinical environments, three constraints co-occur: (1) **Model Absence**—the architecture and parameters of the source model are inaccessible to prevent model attacks (e.g., membership inference attacks); (2) **Streaming Data**—patient data arrives in batches rather than being pre-collected, preventing offline training; (3) **Source Data Privacy**—the source domain training data is inaccessible. Existing SFDA methods rely on full access to model parameters for adaptation, failing to meet the requirement under model absence.

**Key Challenge**: Under the triple constraints of no source model parameters, no annotated data, and streaming data arrival, how can the prediction capability of a model on new domains be improved? Traditional methods require at least accessing model parameters (e.g., SFDA/TTA), whereas the model is treated as a complete black box in the OMG-DA scenario.

**Goal**: (1) Define and address the new problem setting of OMG-DA (Online Model-aGnostic Domain Adaptation); (2) Adapt to the target domain by modifying the input data distribution without accessing any model information; (3) Verify the effectiveness of the method on DR grading tasks.

**Key Insight**: Since modifying the model is impossible, the focus shifts to modifying the data—moving from "model adaptation" to "data adaptation." The authors reformulate traditional unadversarial learning from an iterative optimization form into a generative one, proving that perturbations can be directly output by a generative function without requiring model gradients. This generative function is then instantiated using a VAE, with saliency maps serving as pseudo-supervision.

**Core Idea**: Utilizing a VAE to learn personalized unadversarial perturbations with saliency maps as pseudo-labels, achieving domain adaptation from the data side under strict model-agnostic conditions.

## Method

### Overall Architecture

The input to GUES is the unlabeled target-domain streaming data $x_t$. The VAE encoder maps $x_t$ to a latent variable $z$ (approximating the partial derivative of the initial noise with respect to the image, i.e., $\partial\delta_0/\partial x$), and the decoder generates a personalized perturbation $\delta_t$. The perturbation $\delta_t$ is combined with the original input via a shortcut connection to obtain the unadversarial example $\hat{x}_t = x_t + \delta_t$. Saliency maps $g_t$ of $x_t$ are used as reconstruction supervision during training. During inference, the generated $\hat{x}_t$ is fed into the frozen source model for prediction.

### Key Designs

1. **Generative Unadversarial Learning Theory (Theorem 1)**:

    - **Function**: Reformulate traditional iterative unadversarial perturbation optimization into a generative function.
    - **Mechanism**: Traditional unadversarial learning solves the optimal perturbation iteratively via $\delta_{k+1} = \delta_k + \alpha \cdot \text{sign}(\nabla_x L(f_\theta(x+\delta_k), y))$, which requires model parameters $\theta$, loss function $L$, and label $y$. The authors prove that this iterative process is equivalent to $\delta_k = \delta_0 + V \cdot F_\Phi(\partial\delta_0/\partial x)$, where $F_\Phi$ is a learnable generative function and $\partial\delta_0/\partial x$ is a latent input variable. This converts the problem into learning this generative function, eliminating the need for model parameters and labels.
    - **Design Motivation**: Since model gradients and labels are unavailable in OMG-DA scenarios, traditional iterative unadversarial learning cannot be applied. The generative reformulation allows this problem to be solved via standard generative model training.

2. **VAE Model Instantiation**:

    - **Function**: Learn the generative function $F_\Phi(\partial\delta_0/\partial x)$.
    - **Mechanism**: Two sub-problems need to be solved: (A) identifying the unknown latent input $\partial\delta_0/\partial x$, and (B) selecting a pseudo-supervision signal. For (A), since $\partial\delta_0/\partial x$ depends on both random noise $\delta_0$ and input $x$, it is modeled as a Gaussian distribution $\mathcal{N}(\mu(x), \sigma^2(x))$ conditioned on $x$ using the VAE's encoder and the reparameterization trick. The decoder then implements $F_\Phi$.
    - **Design Motivation**: VAEs possess the inherent capability to sample latent variables from inputs and generate outputs of the same dimension, perfectly matching the theoretical framework of generative unadversarial learning.

3. **Saliency Map as Pseudo-Supervision (Theorem 2)**:

    - **Function**: Provide training signals for perturbations under unlabeled conditions.
    - **Mechanism**: The fine-grained saliency map $s = G(x)$ is selected as the pseudo-perturbation label. It is theoretically proven that $\partial\delta_0/\partial x \leq U \cdot s$, indicating that the saliency map provides an upper bound for the latent input variable. In practice, the saliency map of a DR image effectively localizes lesion areas such as hemorrhages, hard exudates, and soft exudates (similar to Grad-CAM activations), providing guidance for the perturbation direction.
    - **Design Motivation**: Saliency maps are computed purely based on image pixels (center-surround difference) without needing models or labels. Moreover, since they (1) can localize DR-related lesions, and (2) provide a theoretical upper bound for the latent variables, they serve as an ideal choice under OMG-DA constraints.

### Loss & Training

The total loss is $L_{GUES} = \alpha L_{KL} + \beta L_{MSE}$. $L_{KL}$ represents the KL divergence regularization on the latent space to ensure $z$ is close to a standard normal distribution; $L_{MSE}$ is the reconstruction loss between the generated unadversarial example $\hat{x}_t$ and the saliency map $g_t$. Training is conducted online on streaming data, updating the VAE parameters as each batch of data arrives.

## Key Experimental Results

### Main Results

12 cross-domain tasks are constructed across 4 DR datasets (APTOS, DDR, DeepDR, MD2). Comparisons are conducted against Source (directly using the source model), SFDA methods (SHOT, NRC, CoWA), and TTA methods (TENT, DDA).

| Transfer Task | Evaluation | Source | GUES | SHOT (SFDA) | TENT (TTA) |
|---------|------|--------|------|-------------|------------|
| DDR→APTOS | ACC/QWK/AVG | 65.6/72.9/69.3 | **76.0/81.8/78.9** | 77.0/84.2/80.6 | 66.3/73.2/69.7 |
| DDR→DeepDR | AVG | 52.8 | **62.5** | 66.9 | 53.1 |
| APTOS→DDR | AVG | 59.9 | **60.8** | 67.9 | 59.2 |

As an OMG-DA method (without requiring access to model parameters), GUES shows competitive performance compared to SFDA methods that require full model access, approaching SHOT on some tasks. Its advantage is pronounced when compared to TTA methods under similar constraints.

### Combined Experiments (GUES + Existing Methods)

| Method | DDR→APTOS AVG | APTOS→DDR AVG | Average |
|------|--------------|--------------|------|
| SHOT | 80.6 | 67.9 | 74.1 |
| SHOT + GUES | **83.0** | **70.2** | 76.6 |
| TENT | 69.7 | 59.2 | 64.5 |
| TENT + GUES | **73.4** | **61.8** | 67.6 |

As a data preprocessing module, GUES can be layered onto existing adaptation methods to further improve performance.

### Key Findings
- Under the most stringent OMG-DA setting (no model, no labels, and streaming data), GUES still effectively improves the performance of the source model in more than 10 out of 12 transfer tasks.
- GUES is robust to batch size: it maintains effective adaptation even when batch size is 1.
- GUES can be used as a plug-and-play module combined with SFDA/TTA methods to further boost performance.
- Saliency maps as pseudo-supervision are particularly effective in DR tasks because they inherently highlight lesion areas.

## Highlights & Insights
- **A paradigm shift from "model adaptation" to "data adaptation"**: Under the constraint of model absence, shifting to modifying the input data distribution is highly valuable in medical privacy scenarios.
- The theoretical contribution of **Generative Reformulation of Unadversarial Learning** (Theorem 1) converts iterative optimization into a generative function, removing dependence on model gradients, which serves as the core theoretical support.
- **The dual role of saliency maps**—serving as both a lesion localization tool and a theoretical upper bound for latent variables—establishes a highly ingenious connection (Theorem 2).
- As a new problem setting, OMG-DA is closer to real clinical deployment scenarios (where models are typically provided as encrypted APIs) than SFDA, showing practical significance.

## Limitations & Future Work
- The performance of GUES is still notably weaker than SFDA methods (such as SHOT) that require full model access on some tasks, as data-side adaptation is naturally less capable than model-side adaptation.
- Saliency maps as pseudo-supervision have only been verified on tasks with obvious lesion areas like DR; extension to other medical tasks (such as tumor grading) requires further validation.
- The generative quality of the VAE may be limited by the diversity of streaming data, and the perturbation quality might be sub-optimal during early batches due to insufficient training.
- The method is only verified on classification tasks in this paper; whether it is applicable to detection and segmentation tasks remains to explore.

## Related Work & Insights
- **vs SHOT/NRC (SFDA)**: These methods require full access to model parameters for entropy minimization or contrastive learning. In contrast, GUES does not require model parameters at all and only modifies input data, which imposes stronger constraints but offers wider applicability.
- **vs TENT (TTA)**: TENT updates the BN layers of the model at test time, which still requires model parameters. GUES significantly outperforms TENT under equivalent model-absence conditions.
- **vs Traditional Unadversarial Learning [Salman et al.]**: Traditional methods require model gradients and labels to generate class-level perturbations, whereas GUES generates instance-level perturbations without any model information.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The OMG-DA problem setting closely aligns with clinical needs with no precedent, and the theoretical derivation of generative unadversarial learning is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ The 12 cross-domain tasks are comprehensively covered, but the ablation study could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the theoretical derivations are rigorous, though some symbolic notations are somewhat complex.
- Value: ⭐⭐⭐⭐ The adaptation scheme has practical value in medical privacy preservation scenarios, but needs validation across more scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[CVPR 2025\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[NeurIPS 2025\] Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading](../../NeurIPS2025/medical_imaging/ordinal_label-distribution_learning_with_constrained_asymmetric_priors_for_imbal.md)
- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[ICML 2025\] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation](../../ICML2025/medical_imaging/langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)

</div>

<!-- RELATED:END -->
