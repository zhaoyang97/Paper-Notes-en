---
title: >-
  [Paper Note] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection
description: >-
  [CVPR 2025][Object Detection][Domain Generalization] This paper introduces diffusion models to domain-generalized object detection for the first time. By extracting multi-timestep intermediate features from the diffusion process to build a domain-invariant detector, and designing a dual-level (feature-level and object-level) alignment knowledge transfer framework, the generalization capability is distilled into a lightweight common detector. It achieves an average improvement…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Domain Generalization"
  - "Diffusion Models"
  - "Knowledge Distillation"
  - "Domain-Invariant Features"
date: 2026-05-08
content_hash: 3be993ceac52a238
---

# Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.02101](https://arxiv.org/abs/2503.02101)  
**Code**: [https://github.com/heboyong/Generalized-Diffusion-Detector](https://github.com/heboyong/Generalized-Diffusion-Detector)  
**Area**: Object Detection  
**Keywords**: Domain Generalization, Diffusion Models, Knowledge Distillation, Domain-Invariant Features, Object Detection

## TL;DR
This paper introduces diffusion models to domain-generalized object detection for the first time. By extracting multi-timestep intermediate features from the diffusion process to build a domain-invariant detector, and designing a dual-level (feature-level and object-level) alignment knowledge transfer framework, the generalization capability is distilled into a lightweight common detector. It achieves an average improvement of 14.0% mAP across six DG benchmarks, even outperforming most domain adaptation methods.

## Background & Motivation
1. **Background**: Object detection performs exceptionally well under consistent distributions but suffers severe degradation in the presence of domain shifts (e.g., different cameras, weather conditions, styles). Domain adaptation (DA) methods require target-domain data, while domain generalization (DG) methods enhance robustness through data augmentation, adversarial training, and meta-learning on source domains.
2. **Limitations of Prior Work**: Practical deployment of DA methods is constrained due to the requirement for target-domain data. Existing DG methods yield limited improvements; state-of-the-art approaches like ClipGap leverage the generalization capability of CLIP, but it remains insufficient.
3. **Key Challenge**: Feature encoders of standard detectors (e.g., ResNet) tend to overfit the source domain, thereby learning domain-specific attributes rather than domain-invariant features.
4. **Goal**: To leverage the powerful representation capabilities of public diffusion models to extract domain-invariant features for object detection while bypassing the slow inference speed of diffusion models.
5. **Key Insight**: The denoising process of diffusion models is inherently robust to visual perturbations (such as noise, blur, and illumination changes). Furthermore, their intermediate features contain rich multi-scale semantic information, which may be intrinsically domain-invariant.
6. **Core Idea**: To utilize a frozen diffusion model as a domain-invariant feature extractor to construct a teacher detector, and then distill its generalization capability into a standard ResNet detector via a dual-level alignment framework.

## Method

### Overall Architecture
The proposed method consists of two stages: (1) extracting multi-timestep features using Stable Diffusion, aggregating them, and integrating them into Faster R-CNN to build the diffusion detector $\mathcal{F}_{diff}$; (2) freezing $\mathcal{F}_{diff}$ and training a standard ResNet detector $\mathcal{F}_{comm}$ through feature-level and object-level alignments, allowing it to inherit the generalization capability of the diffusion model without adding inference overhead.

### Key Designs

1. **Multi-Timestep Feature Extraction and Fusion**:
    - **Function**: Extracting a domain-invariant multi-scale feature pyramid from the frozen diffusion model.
    - **Mechanism**: Input images are progressively perturbed with noise up to different timesteps $t \in \{1,...,T\}$ ($T=5$). At each timestep, three intermediate features are extracted from each of the four upsampling blocks of the UNet. After aligning dimensions via a trainable bottleneck structure, a weighted aggregation module with learnable weights is used for cross-timestep fusion. This yields a 4-level feature pyramid with channels $C_l = 256 \times 2^{l-1}$ and spatial resolutions $H/2^{l+1} \times W/2^{l+1}$.
    - **Design Motivation**: Features at different timesteps capture semantic information at various levels (earlier timesteps preserve fine details, while later ones capture high-level semantics). The 4-level pyramid structure is compatible with the standard FPN design in common detectors.

2. **Feature-Level Imitation and Alignment**:
    - **Function**: Enabling the standard detector to learn the domain-invariant feature distribution of the diffusion detector.
    - **Mechanism**: It consists of two components: ① Feature alignment loss $\mathcal{L}_{align}$: Features from each FPN layer are normalized and aligned via L2 loss using the PKD method: $\sum_{l=1}^{L} \frac{1}{N_l} \|\hat{\mathcal{M}}_{comm}^l - \hat{\mathcal{M}}_{diff}^l\|_2^2$; ② Cross-feature adaptation $\mathcal{L}_{cross}$: Features from the diffusion detector are fed into the standard detector's RPN and ROI head to compute standard detection losses, enabling the detection heads to process heterogeneous features.
    - **Design Motivation**: Directly aligning heterogeneous model features can be unstable; PKD's correlation matching and cross-feature strategy offer stable pathways for cross-architecture transfer.

3. **Object-Level Knowledge Transfer (Shared RoI)**:
    - **Function**: Aligning detection predictions at the object level to transfer robust detection capabilities.
    - **Mechanism**: The RPN of $\mathcal{F}_{comm}$ is used to generate shared region proposals $\mathcal{R}_{roi}$. RoI pooling is applied to features from both detectors, which are then fed into the respective detection heads. Classification knowledge is transferred using KL divergence with temperature: $\mathcal{L}_{cls} = \frac{1}{N}\sum \tau^2 D_{KL}(\mathbf{Q}_{cat}^i \| \mathbf{P}_{cat}^i)$, and regression knowledge is transferred using L1 loss: $\mathcal{L}_{reg} = \frac{1}{N}\sum |\mathbf{Q}_{bbox}^i - \mathbf{P}_{bbox}^i|_1$.
    - **Design Motivation**: Inspired by CrossKD, using shared RoI proposals aligns the two detectors on identical candidate regions, mitigating mismatch issues of heterogeneous architectures in traditional knowledge distillation.

### Loss & Training
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{det} + \lambda_{feature}(\mathcal{L}_{align} + \mathcal{L}_{cross}) + \lambda_{object}(\mathcal{L}_{cls} + \mathcal{L}_{reg})$
- Hyperparameters: $\lambda_{feature}=0.5$, $\lambda_{object}=1$, diffusion steps $T=5$, max-timestep is 500 for artistic domains and 100 for other domains.
- Training configuration: Faster R-CNN + ResNet101, ImageNet pre-training, batch size 16, learning rate 0.02, SGD, 20K iterations, EMA update.
- Domain augmentation: Strong Augmentation (color + spatial transformations) + domain-level augmentation (FDA / histogram matching / pixel distribution matching).

## Key Experimental Results

### Main Results (Cross Camera: Cityscapes→BDD100K, mAP)

| Method | Type | mAP |
|------|------|-----|
| SHADE (ECCV'22) | DG | 24.0 |
| MAD (CVPR'23) | DG | 28.0 |
| HT (CVPR'23) | DA (w/ target domain) | 40.2 |
| **Diff. Detector (SD-1.5)** | DG | **46.6** |
| **Diff. Guided (SD-1.5)** | DG | **46.3** (+20.9 vs baseline) |

### Ablation Study

| Configuration | Description |
|------|------|
| Baseline (ResNet101) | mAP ~25.4 |
| + Domain Augmentation | Significant improvement |
| + Feature Alignment $\mathcal{L}_{align}$ | Further improvement |
| + Cross Feature $\mathcal{L}_{cross}$ | Stabilizes training |
| + Object-level Alignment $\mathcal{L}_{cls}+\mathcal{L}_{reg}$ | Final performance |
| SD-1.5 vs SD-2.1 | Close performance, SD-1.5 slightly superior |

### Key Findings
- The diffusion-guided detector yields an average improvement of 15.9% mAP over the baseline across six benchmarks, demonstrating the domain invariance of diffusion features.
- It achieves an mAP of 46.6% on the Cross Camera task, significantly outperforming the DA method HT (40.2%), which requires target-domain data.
- The highest improvement of up to 27.2% is observed in the Synthetic $\rightarrow$ Real setting (Sim10K $\rightarrow$ Cityscapes), suggesting that diffusion models are highly effective at bridging synthetic-to-real domain gaps.
- Performance is similar between SD-1.5 and SD-2.1, indicating that the method is insensitive to the specific version of the diffusion model.
- While the diffusion detector itself suffers from slow inference (due to multi-step feature extraction), the distilled standard detector incurs no additional inference overhead.

## Highlights & Insights
- **"Utilizing diffusion models for feature extraction rather than generation"** is highly novel: instead of employing diffusion models to generate new images, it leverages the inherent domain-invariant intermediate representations during the denoising process, effectively transforming a generative model into a perceptual one.
- The **dual-level alignment framework** resolves practical deployment challenges: while the diffusion model offers robust generalization but slow inference, the standard detector is lightweight but lacks generalization. The distillation strategy achieves the best of both worlds.
- Outperforming DA methods without using target-domain data is remarkable. This suggests that the domain-invariant representations extracted from diffusion models are more robust than traditional alignment techniques.
- The knowledge transfer framework can be generalized to other vision tasks, such as segmentation and keypoint detection.

## Limitations & Future Work
- The training overhead of the diffusion detector remains high due to multi-step feature extraction, limiting overall training speed.
- Currently, the approach is verified only on Faster R-CNN and has not been extended to transformer-based detectors like DETR.
- The choice of $T=5$ timesteps is manually determined, and the timestep selection strategy can be further optimized.
- The resolution constraints of the pre-trained diffusion model (e.g., $512 \times 512$) may affect the performance of small object detection.
- Generalization performance under extreme domain shifts (e.g., nighttime infrared) has not been evaluated.

## Related Work & Insights
- **vs ClipGap**: ClipGap utilizes CLIP for domain-generalized detection. This work leverages Stable Diffusion, whose intermediate features are more tailored for detection tasks (multi-scale, high-resolution feature maps versus CLIP's global features).
- **vs OADG**: OADG employs data augmentation strategies for DG. This work addresses domain invariance more fundamentally from the feature level.
- **vs PKD/CrossKD**: These are typical knowledge distillation methods; this work applies them to a heterogeneous, cross-architecture (diffusion-to-CNN) transfer scenario.
- Insight: The intermediate features of pre-trained foundation models (Diffusion / CLIP / DINO) may be inherently more valuable than their final outputs, warranting further exploration across diverse vision tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Innovatively introduces diffusion models to domain-generalized object detection with impressive empirical results.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across 6 DG benchmarks and 13 datasets, establishing solid comparisons with other DA/DG baselines along with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed methodological explanations, though somewhat mathematically dense.
- Value: ⭐⭐⭐⭐⭐ Establishes a promising direction for leveraging pre-trained diffusion models in domain generalization. Code is open-sourced, offering high follow-up potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](../../ICCV2025/object_detection/diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [\[CVPR 2025\] One-for-More: Continual Diffusion Model for Anomaly Detection](one-for-more_continual_diffusion_model_for_anomaly_detection.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/object_detection/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)

</div>

<!-- RELATED:END -->
