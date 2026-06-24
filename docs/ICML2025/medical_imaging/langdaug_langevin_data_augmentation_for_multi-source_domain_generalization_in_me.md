---
title: >-
  [Paper Note] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation
description: >-
  [ICML 2025][Medical Imaging][Data Augmentation] LangDAug utilizes an energy-based model (EBM) trained via contrastive divergence to generate intermediate samples by traversing between source domains using Langevin dynamics, thereby achieving multi-source domain generalization for medical image segmentation, and theoretically proving its induced regularization effect while bounding the Rademacher complexity.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Data Augmentation"
  - "Domain Generalization"
  - "Langevin Dynamics"
  - "Energy-Based Models"
  - "Medical Image Segmentation"
date: 2026-05-08
content_hash: 13a29a6f911405fe
---

# LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation

**Conference**: ICML 2025  
**arXiv**: [2505.19659](https://arxiv.org/abs/2505.19659)  
**Code**: [https://github.com/backpropagator/LangDAug](https://github.com/backpropagator/LangDAug)  
**Area**: Medical Imaging  
**Keywords**: Data Augmentation, Domain Generalization, Langevin Dynamics, Energy-Based Models, Medical Image Segmentation

## TL;DR
LangDAug utilizes an energy-based model (EBM) trained via contrastive divergence to generate intermediate samples by traversing between source domains using Langevin dynamics, thereby achieving multi-source domain generalization for medical image segmentation, and theoretically proving its induced regularization effect while bounding the Rademacher complexity.

## Background & Motivation

1. **Background**: Medical image segmentation models suffer from a severe lack of generalization capability across different domains (e.g., different hospitals, devices, imaging parameters). Domain generalization (DG) methods address this issue through representation learning or data augmentation.

2. **Limitations of Prior Work**: Representation learning methods seek domain-invariant features but often rely on ad-hoc techniques and lack formal guarantees. Although data augmentation methods yield close or superior performance, existing augmentation strategies (e.g., random style transfer) lack principled designs, and it remains unclear "to what extent the augmentation should be performed."

3. **Key Challenge**: How can one design a theoretically guaranteed data augmentation strategy that systematically generates effective intermediate samples to bridge the gap between source domains?

4. **Goal**: Propose a principled data augmentation method based on energy-based models and Langevin dynamics.

5. **Key Insight**: Model different source domains as distinct valleys in an energy landscape, and use Langevin dynamics to "walk" between domains to generate intermediate domain samples.

6. **Core Idea**: Train an EBM to capture the joint energy landscape of the source domains, then traverse between domains using a Langevin sampler; the generated intermediate samples are then used to train the segmentation model.

## Method

### Overall Architecture

- **Input**: Medical images and their corresponding segmentation labels from multiple source domains
- **First Step**: Train an EBM to model the joint distribution of multiple source domains
- **Second Step**: Starting from any source domain, run a Langevin dynamics MCMC chain to generate intermediate domain samples
- **Third Step**: Jointly train the segmentation model using both the original and augmented data
- **Output**: A segmentation model with enhanced generalization capability

### Key Designs

1. **Domain Modeling with Energy-Based Models (EBM)**:
    - Train the EBM using Contrastive Divergence: $p_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$
    - The energy landscape of the EBM naturally encodes the distribution of different source domains.
    - Different domains correspond to distinct low-energy areas, while the regions between domains represent high-energy "hills".
    - **Design Motivation**: EBM provides a continuous energy landscape, ensuring smooth transitions between domains.

2. **Traversing Between Domains via Langevin Dynamics**:
    - Starting from a sample in domain A, run Langevin MCMC: $\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\eta}{2} \nabla_\mathbf{x} E_\theta(\mathbf{x}_k) + \sqrt{\eta} \epsilon_k$
    - As the number of steps increases, the sample transitions from domain A to the intermediate region between domain A and domain B.
    - The number of steps controls the distance of the augmented samples from the source domains.
    - **Design Motivation**: The stationary distribution of Langevin dynamics is precisely the distribution defined by the EBM, which guarantees the validity of the sampling.

3. **Theoretical Guarantees**:
    - Prove the implicit regularization effect induced by LangDAug.
    - For Generalized Linear Models (GLMs), LangDAug bounds the Rademacher complexity by the intrinsic dimension of the data manifold.
    - This implies that the effectiveness of the augmentation is related to the true complexity of the data rather than the number of model parameters.
    - **Design Motivation**: Provide formal generalization guarantees, rather than merely empirical improvements.

### Loss & Training

- EBM Training: Contrastive Divergence loss $\mathcal{L}_{CD} = \mathbb{E}_{p_\text{data}}[E_\theta(\mathbf{x})] - \mathbb{E}_{p_\theta}[E_\theta(\mathbf{x}')]$
- Segmentation Model Training: Standard segmentation loss (Cross-Entropy + Dice), trained on both original and augmented data.
- LangDAug can be combined and used alongside other domain randomization methods.

## Key Experimental Results

### Main Results

| Dataset | Metric | LangDAug | Prev. SOTA DG | Gain |
|--------|------|----------|-------------|------|
| Fundus Segmentation (Fundus) | Dice↑ | SOTA | Suboptimal | Significant |
| Prostate MRI | Dice↑ | SOTA | Suboptimal | Significant |
| Fundus + Domain Rand. | Dice↑ | Best | Domain Rand. alone | Complementary Gain |

### Ablation Study

| Configuration | Dice | Description |
|------|------|------|
| No Augmentation | Baseline | Trained only on source domain data |
| Random Augmentation | Small Gain | Traditional augmentation |
| Domain Randomization (DR) | Moderate Gain | Existing SOTA augmentation |
| LangDAug alone | Better | Outperforms DR |
| **LangDAug + DR** | **Best** | Complementary to each other |
| Varying Langevin Steps | Step-sensitive | Excess steps may deviate too far from the source domains |

### Key Findings

- LangDAug outperforms SOTA domain generalization methods on both the fundus segmentation and prostate MRI segmentation benchmarks.
- LangDAug is complementary to existing domain randomization methods—the combination yields the best results.
- The number of Langevin steps is a key hyperparameter: too few steps are ineffective, while too many may generate out-of-distribution samples.
- The theoretical regularization effect is empirically validated in the experiments.

## Highlights & Insights

1. **Solid Theory**: The Rademacher complexity bound provides a stronger guarantee than empirical-only methods.
2. **Physical Intuition**: The analogy of energy landscape + Langevin dynamics is intuitive and precise.
3. **Complementarity**: Orthogonal to existing augmentation methods, allowing cumulative usage.
4. **Cross-Modal Validation**: Effective across different modalities including fundus and MRI.

## Limitations & Future Work

1. EBM training itself is unstable and requires careful hyperparameter tuning.
2. Langevin sampling is relatively slow, increasing the overall training time.
3. Currently validated only on 2D segmentation; the extension to 3D volumetric segmentation remains to be explored.
4. Labels of the augmented samples require additional handling (how the segmentation labels of augmented samples are obtained in this paper warrants attention).

## Related Work & Insights

- Domain generalization methods such as DSU and CIRL are the primary baselines for comparison.
- Augmentation methods like AdvBias and FedDG provide baselines.
- Insight: The EBM + Langevin augmentation strategy can be extended to other medical imaging tasks (e.g., classification, detection).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of EBM + Langevin for domain generalization augmentation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two benchmarks along with complementarity experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations.
- Value: ⭐⭐⭐⭐ Highly practical value for medical image domain generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2025/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2025/medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](../../NeurIPS2025/medical_imaging/posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[ICCV 2025\] Controllable Latent Space Augmentation for Digital Pathology](../../ICCV2025/medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)

</div>

<!-- RELATED:END -->
