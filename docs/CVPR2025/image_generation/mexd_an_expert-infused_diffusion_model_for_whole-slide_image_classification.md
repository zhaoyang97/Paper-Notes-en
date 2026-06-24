---
title: >-
  [Paper Note] MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification
description: >-
  [CVPR 2025][Image Generation][Whole-Slide Images] MExD is the first to apply generative diffusion models to whole-slide image (WSI) classification. By utilizing a Dynamic Mixture-of-Experts (Dyn-MoE) aggregator to select key instances and provide conditional information, combined with a Diffusion Classifier (Diff-C) to iteratively reconstruct class labels from noise, it achieves state-of-the-art (SOTA) performance on three benchmarks: Camelyon16, TCGA-NSCLC, and BRACS.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Whole-Slide Images"
  - "Multi-Instance Learning"
  - "Diffusion Classifier"
  - "Mixture of Experts"
  - "Data Imbalance"
date: 2026-05-08
content_hash: 40b4811510ea2204
---

# MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification

**Conference**: CVPR 2025  
**arXiv**: [2503.12401](https://arxiv.org/abs/2503.12401)  
**Code**: [https://github.com/JWZhao-uestc/MExD](https://github.com/JWZhao-uestc/MExD)  
**Area**: Image Generation  
**Keywords**: Whole-Slide Images, Multi-Instance Learning, Diffusion Classifier, Mixture of Experts, Data Imbalance

## TL;DR
MExD is the first to apply generative diffusion models to whole-slide image (WSI) classification. By utilizing a Dynamic Mixture-of-Experts (Dyn-MoE) aggregator to select key instances and provide conditional information, combined with a Diffusion Classifier (Diff-C) to iteratively reconstruct class labels from noise, it achieves state-of-the-art (SOTA) performance on three benchmarks: Camelyon16, TCGA-NSCLC, and BRACS.

## Background & Motivation

1. **Background**: WSI classification typically adopts the "decompose-aggregate" strategy of Multi-Instance Learning (MIL) — segmenting a WSI into patches to extract features, then magnifying/aggregating them into a slide-level representation for classification. Mainstream methods include attention pooling (ABMIL), Transformer-based aggregation (TransMIL), and graph-based models (WiKG).
2. **Limitations of Prior Work**: (1) Severe data imbalance — positive patches (containing cancer cells) within a WSI are far fewer than negative patches, causing models to favor the majority class; (2) massive non-informative regions introduce noise; (3) simple aggregation methods struggle to capture the complex relationships between patches.
3. **Key Challenge**: Existing methods are entirely based on the discriminative paradigm, handling all patches in a unified model while lacking a dedicated attention mechanism for minority classes. Discriminative feature integration is inherently susceptible to interference from noise and irrelevant patches.
4. **Goal**: (1) How to effectively extract minority class information under extreme imbalance? (2) How to aggregate patch features in a noise-robust manner?
5. **Key Insight**: Reformulate WSI classification as a "conditional generation task" — instead of predicting labels directly from features, generate label distributions from noise conditioned on features. Meanwhile, a Mixture-of-Experts (MoE) mechanism allocates dedicated "experts" to each class for instance filtering.
6. **Core Idea**: Employ MoE routing to select key instances as conditions and utilize a diffusion model to iteratively generate one-hot class labels, achieving a paradigm shift in WSI classification from discriminative to generative.

## Method

### Overall Architecture
Input WSI is cropped into patches $\to$ Patch Feature Extractor (frozen pre-trained ViT/CTransPath) extracts $N$ patch embeddings $\to$ Dyn-MoE Aggregator filters a sparse instance set via $K+1$ expert routes and generates a prior prediction $\rho_\theta$ and an expert insight set $g_\alpha$ $\to$ Diffusion Classifier (Diff-C) uses $\rho_\theta$ and $g_\alpha$ as conditions to iteratively denoise from noise and generate class distributions $\to$ Output classification results.

### Key Designs

1. **Dynamic Mixture-of-Experts (Dyn-MoE) Aggregator**:

    - **Function**: Sparsify the instance set, allocate dedicated experts to each class, and generate prior predictions and expert insights.
    - **Mechanism**: First, global dependencies $\{l_i\}_{i=1}^N$ are establish via an Adapter (2-layer Transformer + PPEG convolution block). Then, $K+1$ routers are configured ($K$ positive experts + 1 negative expert), where each router is a two-class MLP+softmax producing two scores for each instance. The negative expert retains instances with max score index = 0, while positive experts retain those with index = 1. Each subset selects instances with the top-$k$ routing scores (controlled by a sampling ratio $\alpha$). Each expert applies average pooling to its retained instances to obtain class centroid features $e_r$, and generates a prediction $y^{ex}$ and confidence score $c_r$ through a $(K+1)$-class classifier. All sparsified instances are concatenated with a learnable class embedding $d$, and then processed through the Adapter and a linear classifier to generate the prior prediction $\rho_\theta$.
    - **Design Motivation**: Traditional MIL processes all patches in a single, unified model, causing minority class signals to be submerged. Dyn-MoE allocates a dedicated "channel" to each class; positive experts focus solely on candidate positive instances, effectively mitigating imbalance. Dynamic routing along with top-$k$ filtering reduces the number of instances by more than half, significantly curtailing noise.

2. **Diffusion Classifier (Diff-C)**:

    - **Function**: Translates the classification task into a conditional generation task, iteratively recovering class labels from noise.
    - **Mechanism**: Encodes ground-truth (GT) labels into a one-hot vector $f \in \mathbb{R}^{1 \times (K+1)}$ as the initial signal. Drawing from the CARD method, it treats the prior prediction $\rho_\theta$ as the conditional expectation at the end of forward diffusion (rather than standard Gaussian noise); the forward process is defined as $q(f_t|f_0, \rho_\theta) = \mathcal{N}(f_t; \sqrt{\bar\alpha_t}f_0 + (1-\sqrt{\bar\alpha_t})\rho_\theta, (1-\bar\alpha_t)I)$. For reverse denoising, a 3-layer MLP is employed as the denoising network $\mathcal{D}$, conditioned on the weighted expert insight features $Z = \sum_r c_r \cdot e_r$ and prior prediction $\rho_\theta$, to iteratively step backwards: $f_T \to f_{T-\Delta} \to \cdots \to f_0'$.
    - **Design Motivation**: Generative methods naturally outperform discriminative ones in handling noise and data distributions. The iterative denoising in the diffusion process acts as a multi-step refinement of class predictions, leveraging expert knowledge at each step for corrections. Employing the prior prediction as the diffusion target endpoint, rather than pure noise, substantially accelerates convergence.

3. **Two-Stage Training Strategy**:

    - **Function**: Ensures stable convergence for both Dyn-MoE and Diff-C.
    - **Mechanism**: Stage 1 trains the Dyn-MoE, where the joint loss $\mathcal{L}_a$ includes: (a) cross-entropy of the prior prediction $\Phi(f_0, \rho_\theta)$, (b) cross-entropy of the negative class expert $\Phi(\dot{y}_0, y_0^{ex})$, and (c) weighted cross-entropy of selective positive class experts $\sum_r \lambda_r \Phi(\dot{y}_r, y_r^{ex})$ ($\lambda_r=1$ only when the label corresponding to the positive expert matches the GT, and 0 otherwise). Stage 2 freezes the Dyn-MoE and trains the denoising network $\mathcal{D}$ using the standard noise-estimation loss $\mathcal{L}_e = \|\epsilon - \epsilon_\theta(Z, f_t, \rho_\theta, t)\|^2$.
    - **Design Motivation**: Dyn-MoE must first learn meaningful routing and prior predictions to provide high-quality conditioning for Diff-C. Joint training might lead to unstable routing. The selective expert loss ensures that each expert is only responsible for samples that "rightfully belong to it".

### Loss & Training
- Stage 1: $\mathcal{L}_a = \frac{1}{R}\sum(\Phi(f_0, \rho_\theta) + \Phi(\dot{y}_0, y_0^{ex}) + \sum_r \lambda_r \Phi(\dot{y}_r, y_r^{ex}))$
- Stage 2: $\mathcal{L}_e = \|\epsilon - \epsilon_\theta(Z, f_t, \rho_\theta, t)\|^2$
- Patch features are extracted offline, supporting plug-and-play deployment.

## Key Experimental Results

### Main Results (CTransPath Features)

| Method | Camelyon16 AUC | TCGA-NSCLC AUC | BRACS AUC |
|------|---------------|----------------|-----------|
| ABMIL | 92.41 | 95.87 | 80.44 |
| TransMIL | 95.03 | 95.89 | 85.18 |
| IBMIL | 96.41 | 97.45 | 85.84 |
| MHIM-MIL | 96.14 | 96.73 | 84.79 |
| **MExD** | **98.87** | **98.13** | **88.08** |

### Ablation Study

| Configuration | Camelyon16 AUC | BRACS AUC | Description |
|------|---------------|-----------|------|
| Full MExD | 98.87 | 88.08 | Full model |
| w/o Diff-C (Dyn-MoE only) | ~96.5 | ~86 | Removes diffusion classifier, degrades to discriminative model |
| w/o MoE Routing | Significant Drop | Significant Drop | Fails to handle imbalance effectively |
| w/o Prior Prediction Conditioning | Slower Convergence | Dropped Accuracy | Pure noise starting point is inferior to the prior starting point |

### Key Findings
- MExD achieves 98.87% AUC on Camelyon16, representing a 2.46% gain over the strongest baseline (IBMIL 96.41%).
- The performance improvement is most prominent on the three-class BRACS dataset (88.08% vs. 85.84%), demonstrating that the MoE mechanism is especially effective in multi-class imbalanced scenarios.
- MExD also performs optimally using ViT (MoCo V3) features, verifying the framework's architecture-agnostic compatibility with different feature extractors.
- The iterative refinement of the diffusion classifier consistently outperforms single-step discriminative predictions, especially on hard samples.
- It achieves a comprehensive lead with an F1-score of 97.29% and an accuracy (ACC) of 97.48% on Camelyon16.

## Highlights & Insights
- **Paradigm Shift from Discriminative to Generative**: Introduces a generative approach to WSI classification for the first time, formulating classification as a conditional generation process from noise to labels. This direction breaks away from the discriminative paradigm dominant in the MIL field, naturally imbuing the model with noise robustness through the iterative refinement of diffusion.
- **Deep Fusion of MoE and Diffusion**: The MoE module acts not only as an aggregator but also feeds two types of conditioning information into the diffusion model — prior predictions (defining the diffusion endpoint) and expert insights (guiding the denoising direction). This "conditioning on conditions" design is more organic than simple feature concatenation.
- **Selective Expert Loss**: Positive experts activate loss only when the label matches, shielding the expert routing learning from disruptive gradients of mismatched categories. This design can serve as a reference for other MoE applications.

## Limitations & Future Work
- Diffusion inference requires multi-step iterations ($T$ steps), resulting in lower inference efficiency than discriminative methods. Despite a lightweight MLP denoising network, latency remains a concern.
- The two-stage training scheme increases the complexity of hyperparameter tuning.
- The MoE routing utilizes hard routing (argmax), which might discard some information. Whether soft routing or invertible routing performs better remains unexplored.
- The diffusion operates on a 1D signal (one-hot vector), missing out on utilizing the generative advantages of diffusion models in high-dimensional spaces.
- Model interpretability is not extensively discussed — which patches are assigned to which experts? Does the routing decision align with the judgments of pathologists?

## Related Work & Insights
- **vs ABMIL/TransMIL**: These discriminative methods aggregate all patches using attention or Transformers, lacking specialized mechanisms for minority classes. Conversely, MExD's MoE routing explicitly allocates dedicated experts to each class.
- **vs DTFD-MIL**: DTFD alleviates imbalance via pseudo-bag partitioning but remains discriminative in nature. MExD fundamentally restructures the information flow using a generative framework.
- **vs IBMIL**: IBMIL is a current MIL SOTA, upon which MExD further improves via MoE sparsification and diffusion classification.
- **vs CARD**: Diff-C borrows the conditional diffusion classification framework from CARD, but introduces MoE expert insights as an extra condition to achieve WSI-specific adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first work to apply diffusion models to WSI classification, presenting a novel integration of MoE and diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on three benchmarks and validation with two feature extractors, though some ablation details could be further expanded.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and the mathematical formulation is thorough, of course, the abundance of formulas slightly adds to the reading load.
- Value: ⭐⭐⭐⭐ Opens up a new paradigm for WSI classification, with SOTA results convincingly proving the potential of generative approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)
- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](../../CVPR2026/image_generation/advancing_image_classification_with_discrete_diffusion_classification_modeling.md)
- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](../../NeurIPS2025/image_generation/camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)

</div>

<!-- RELATED:END -->
