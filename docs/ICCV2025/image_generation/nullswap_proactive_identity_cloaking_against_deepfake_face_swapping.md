---
title: >-
  [Paper Note] NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping
description: >-
  [ICCV2025][Image Generation][Deepfake defense] This paper proposes NullSwap, which embeds identity-guided invisible perturbations into source images to cloak facial identity information, preventing Deepfake face-swapping models from extracting the correct identity, thereby enabling proactive defense against face-swapping attacks in a purely black-box setting.
tags:
  - "ICCV2025"
  - "Image Generation"
  - "Deepfake defense"
  - "face swapping"
  - "proactive perturbation"
  - "identity cloaking"
  - "adversarial perturbation"
  - "black-box defense"
date: 2026-05-08
content_hash: b06ed518a6c8c296
---

# NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping

**Conference**: ICCV2025
**arXiv**: [2503.18678](https://arxiv.org/abs/2503.18678)  
**Code**: Not released  
**Area**: Image Generation
**Keywords**: Deepfake defense, face swapping, proactive perturbation, identity cloaking, adversarial perturbation, black-box defense

## TL;DR

This paper proposes NullSwap, which embeds identity-guided invisible perturbations into source images to cloak facial identity information, preventing Deepfake face-swapping models from extracting the correct identity, thereby enabling proactive defense against face-swapping attacks in a purely black-box setting.

## Background & Motivation

### State of the Field
Deepfake face-swapping technology has become increasingly sophisticated, and passive detection methods are approaching a bottleneck due to the improving quality of generated content. Proactive defense, which pre-inserts invisible signals into benign images to disrupt Deepfake operations, represents a more promising direction.

### Limitations of Prior Work

**Visual degradation**: Existing methods insert perturbations via direct element-wise addition, resulting in visible visual artifacts (abnormal lighting, blurring, etc.).

**Weak face-swapping defense**: Prior methods primarily target facial attribute editing and facial reenactment, with limited defensive capability against face-swapping attacks.

**Reliance on generative models**: Most methods require white-box or gray-box settings during training, incorporating actual Deepfake generative models or surrogate models, incurring substantial computational overhead.

### Root Cause
This paper analyzes the essence of Deepfake face swapping: in face-swapping attacks, the true victim is the **source image** (the identity provider), not the target image. When a celebrity's face is swapped into inappropriate content, it is the person whose identity is being impersonated that requires protection. Therefore, the defensive focus should shift from protecting the target image to protecting the source identity information.

## Method

### Overall Architecture
The NullSwap framework consists of four core modules:

#### 1. Identity Extraction Module
- Takes source image $I_s$ as input and extracts facial identity features
- Uses ConvBlock (CNN + BatchNorm + ReLU) + MaxPooling
- Followed by $L=4$ SEResBlocks (ResNet bottleneck + SENet) to preserve identity features in matrix format
- The squeeze-and-excitation mechanism in SENet efficiently analyzes inter-channel correlations

#### 2. Perturbation Block
- Receives identity features and generates identity-guided perturbations
- Feature refinement via ConvBlock, followed by $M=3$ SEResBlocks for hierarchical feature aggregation
- Introduces adaptive random noise to prevent overfitting:

$$\text{RandNoise} = \beta \cdot (\alpha \cdot \text{RandNoise} + \eta)$$

where $\eta$ is learnable noise and $\alpha, \beta$ are learnable amplitude-scaling parameters.

#### 3. Feature Block
- Performs shallow feature extraction on the input image
- Three ConvBlocks for local feature analysis and dimension adjustment
- $N=5$ SEResBlocks to enhance feature adaptability and contextual awareness

#### 4. Cloaking Block
- **Feature-level reconstruction**: Assigns learnable weights $\gamma$ to the perturbation, concatenates it with image features, and performs fusion reconstruction via SEResBlock → DeConvBlock → ConvBlock → DeConvBlock
- **Image-level reconstruction**: Concatenates the feature-level result with the original input and outputs the final cloaked image $I_s'$ through three ConvBlocks
- The two-stage reconstruction ensures visual fidelity while successfully embedding identity perturbations

### Dynamic Loss Weighting (DLW)

To ensure generalization across different face-swapping algorithms (which use different identity extractors), the paper proposes a DLW mechanism to adaptively balance identity losses from multiple face recognition tools:

$$\mathcal{L}_{id}(t_e, t_b) = \sum_{i=1}^{c} \hat{w_i}(t_e, t_b) \cdot \mathcal{L}_i(t_b)$$

Weights are determined by two core components:
- **Loss variance** $\sigma_i^2$: measures loss stability over the most recent $k=30$ iterations; higher variance leads to lower weight
- **Relative progress** $\Delta_i$: evaluates the rate of loss improvement; slower-improving losses receive higher weights

In the weight computation, $\beta$ increases linearly with training epochs (from $\beta_{init}=0.5$ to $\beta^*=2$), progressively amplifying the influence of the progress factor.

### Total Loss Function

$$\mathcal{L}_{total} = \lambda_{id} \mathcal{L}_{id} + \lambda_{MSE} \mathcal{L}_{MSE} + \lambda_{LPIPS} \mathcal{L}_{LPIPS} + \lambda_D \mathcal{L}_D$$

- $\mathcal{L}_{MSE}$: pixel-level reconstruction quality ($\lambda_{MSE}=1.8$)
- $\mathcal{L}_{LPIPS}$: perceptual similarity ($\lambda_{LPIPS}=1.2$)
- $\mathcal{L}_D$: adversarial loss with discriminator trained from scratch ($\lambda_D=0.1$)
- $\mathcal{L}_{id}$: identity cloaking loss weighted by DLW ($\lambda_{id}=0.08$)

## Key Experimental Results

### Experimental Setup
- Datasets: CelebA-HQ (30K images, 6,217 identities) for training/testing; LFW (5,749 identities) for cross-dataset validation
- Face recognition tools: ArcFace and FaceNet (for training); VGGFace and SFace (for testing, to validate generalization)
- Face-swapping models: SimSwap, InfoSwap, UniFace, E4S, DiffSwap (all appear only at test time)
- Hardware: 8× Tesla A100 GPUs, batch size 256, 60 epochs

### Visual Quality (Perturbed Image vs. Original)

| Metric | Initiative | Anti-Forgery | CMUA | DF-RAP | **NullSwap** |
|--------|-----------|-------------|------|--------|-------------|
| PSNR↑ | 39.38 | 38.07 | 38.64 | 38.85 | **41.31** |
| SSIM↑ | 0.9544 | 0.9530 | 0.9504 | 0.9349 | **0.9864** |
| LPIPS↓ | 0.0200 | 0.0282 | 0.0333 | 0.0511 | **0.0049** |

NullSwap is the only method achieving PSNR > 40, SSIM > 0.98, and LPIPS < 0.005.

### Identity Cloaking (CelebA-HQ, Top-1 Accuracy↓)

| Recognizer | Clean | Initiative | Anti-Forgery | CMUA | DF-RAP | **NullSwap** |
|-----------|-------|-----------|-------------|------|--------|-------------|
| ArcFace | 0.976 | 0.968 | 0.975 | 0.976 | 0.974 | **0.628** |
| FaceNet | 0.918 | 0.925 | 0.862 | 0.865 | 0.920 | **0.590** |
| VGGFace | 0.853 | 0.856 | 0.858 | 0.864 | 0.847 | **0.529** |
| SFace | 0.791 | 0.720 | 0.732 | 0.720 | 0.680 | **0.513** |
| Average | 0.885 | 0.867 | 0.857 | 0.856 | 0.855 | **0.565** |

Competing methods maintain average accuracy above 0.85, while NullSwap reduces it to 0.565, achieving a substantial margin.

### Face-Swapping Identity Similarity (CelebA-HQ, ArcFace/VGGFace Cosine Similarity↓)

| Swap Model | Initiative | CMUA | DF-RAP | **NullSwap** |
|-----------|-----------|------|--------|-------------|
| SimSwap | 0.928/0.897 | 0.921/0.930 | 0.468/0.431 | **0.217/0.240** |
| InfoSwap | 0.941/0.919 | 0.947/0.888 | 0.913/0.849 | **0.375/0.359** |
| UniFace | 0.987/0.960 | 0.983/0.965 | 0.947/0.891 | **0.369/0.329** |
| E4S | 0.925/0.893 | 0.920/0.900 | 0.880/0.855 | **0.398/0.368** |
| DiffSwap | 0.660/0.657 | 0.658/0.648 | 0.636/0.631 | **0.310/0.352** |
| Average | 0.888/0.865 | 0.886/0.866 | 0.769/0.731 | **0.334/0.330** |

NullSwap reduces the identity similarity of face-swapping outputs from ~0.9 to ~0.33, effectively defending against all evaluated face-swapping models.

### Ablation Study on DLW

| Strategy | ArcFace↓ | FaceNet↓ | VGGFace↓ | SFace↓ | Average↓ |
|----------|---------|---------|---------|-------|---------|
| ArcFace only | 0.653 | 0.758 | 0.680 | 0.576 | 0.667 |
| FaceNet only | 0.843 | 0.546 | 0.504 | 0.495 | 0.597 |
| Simple average | 0.846 | 0.613 | 0.600 | 0.525 | 0.646 |
| **DLW** | **0.628** | **0.590** | **0.529** | **0.513** | **0.565** |

DLW achieves balanced performance across all recognizers and attains the best average.

## Highlights & Insights

1. **Perspective shift**: This paper is the first to identify that face-swapping defense should protect the source identity rather than the target image, a insight that reframes the problem definition.
2. **Purely black-box**: The training process involves no generative models whatsoever, relying solely on multiple face recognition tools, which substantially reduces computational cost.
3. **Feature-level embedding**: By abandoning direct element-wise addition and instead integrating perturbations naturally through shallow feature extraction and two-stage reconstruction, the method achieves visual quality that comprehensively surpasses existing approaches.
4. **DLW adaptability**: Dynamic weighting via dual factors—loss variance and relative progress—avoids the optimization imbalance caused by naive averaging.
5. **Cross-model generalization**: Trained with only ArcFace and FaceNet, the method remains effective on unseen VGGFace and SFace recognizers as well as five face-swapping models at test time.

## Limitations & Future Work

1. **Diversity of identity extractors**: Although DLW improves generalization, training uses only 2 recognition tools; a more diverse set of training signals may further improve robustness.
2. **High-resolution adaptation**: Experiments are conducted at 256×256 resolution; the effectiveness of perturbations on high-resolution images in practical applications remains to be verified.
3. **Social network compression**: Although DF-RAP has explored OSN compression, NullSwap does not explicitly discuss the robustness of perturbations after JPEG compression or social network transmission.
4. **Adaptive attacks**: If an adversary is aware of NullSwap's existence, they may design targeted purification methods to remove the perturbations.
5. **Video scenarios**: The current approach targets static images; perturbation strategies that maintain temporal consistency in video face swapping merit further exploration.
6. **Ethical concerns**: The technique itself could potentially be repurposed (e.g., to obstruct legitimate identity verification), necessitating careful consideration of deployment contexts.

## Related Work & Insights

- **SimSwap (ACM MM 2020)**: An AdaIN-based identity injection paradigm and a landmark work in face swapping.
- **Initiative (AAAI 2021)**: Gray-box proactive defense that constructs a surrogate model to mimic Deepfake operations and generate poisoning perturbations.
- **Anti-Forgery (ACM MM 2022)**: Embeds perturbations in the Lab color space to improve visual compatibility.
- **CMUA (AAAI 2022)**: A cross-model universal adversarial watermark that iteratively attacks multiple Deepfake models.
- **DF-RAP (TIFS 2024)**: Approximates GAN compression to enhance perturbation persistence in OSN scenarios.
- **ArcFace / FaceNet / SENet**: Core tools for identity feature extraction and channel attention.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](../../CVPR2026/image_generation/attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](../../CVPR2026/image_generation/preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[ICML 2025\] Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](../../ICML2025/image_generation/synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)

</div>

<!-- RELATED:END -->
