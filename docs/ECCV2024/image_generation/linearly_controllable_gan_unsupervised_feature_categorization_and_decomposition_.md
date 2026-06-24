---
title: >-
  [Paper Note] Linearly Controllable GAN: Unsupervised Feature Categorization and Decomposition for Image Generation and Manipulation
description: >-
  [ECCV 2024][Image Generation][Controllable GAN] This paper proposes LC-GAN, which achieves unsupervised geometry-appearance feature disentanglement in the GAN latent space through contrastive feature categorization and spectral regularization. This enables independent linear control of various attributes in generated images, achieving SOTA generation quality on FFHQ, CelebA-HQ, and AFHQ-V2.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Controllable GAN"
  - "Feature Disentanglement"
  - "Contrastive Learning"
  - "Spectral Regularization"
  - "Unsupervised Decomposition"
date: 2026-05-08
content_hash: aaa796112205f36e
---

# Linearly Controllable GAN: Unsupervised Feature Categorization and Decomposition for Image Generation and Manipulation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Controllable GAN, Feature Disentanglement, Contrastive Learning, Spectral Regularization, Unsupervised Decomposition

## TL;DR
This paper proposes LC-GAN, which achieves unsupervised geometry-appearance feature disentanglement in the GAN latent space through contrastive feature categorization and spectral regularization. This enables independent linear control of various attributes in generated images, achieving SOTA generation quality on FFHQ, CelebA-HQ, and AFHQ-V2.

## Background & Motivation

**Background**: Controllable latent space editing in GANs is one of the core problems in the field of image generation. Existing methods either rely on supervised signals (e.g., attribute annotations) to guide feature disentanglement, or employ post-processing approaches (e.g., GANSpace, SeFa, which perform PCA/SVD analysis on pre-trained GANs) to discover editable directions. However, these methods either require additional annotations or can only identify a limited number of editing dimensions.

**Limitations of Prior Work**: Supervised methods require large amounts of labeled data, limiting their generalizability and scalability. Although post-processing methods do not require labels, the discovered editing directions often suffer from attribute entanglement—modifying one attribute (such as pose) simultaneously affects others (such as skin color), leading to imprecise editing. Furthermore, existing methods rarely structurally distinguish between geometry-related and appearance-related features, resulting in incomplete disentanglement.

**Key Challenge**: To achieve true linearly controllable generation, different dimensions in the latent space must correspond to different image attributes, and these dimensions must be mutually orthogonal. However, traditional GAN training objectives focus solely on generation quality without imposing structural constraints on the latent space, which naturally leads to feature entanglement.

**Goal**: (1) How to automatically categorize latent codes into geometry-related and appearance-related groups under unsupervised conditions? (2) How to ensure that different feature dimensions within the same category can independently control different aspects of the image? (3) How to maintain or even improve generation quality while achieving disentanglement?

**Key Insight**: The authors observe that the geometric attributes (pose, shape) and appearance attributes (color, texture) of an image exhibit different invariances to image augmentations—geometric transformations (rotation, cropping) change geometry but preserve appearance, whereas color transformations change appearance but preserve geometry. Utilizing this prior, contrastive learning can be leveraged to let the discriminator automatically construct geometry and appearance feature spaces.

**Core Idea**: Utilizing the differences in augmentation invariance, contrastive learning is used to automatically categorize geometric/appearance features, and spectral regularization is then applied to project features of the same class into orthogonal subspaces, achieving a completely unsupervised linearly controllable GAN.

## Method

### Overall Architecture
LC-GAN introduces two core mechanisms into the standard GAN framework: (1) a contrastive feature categorization module on the discriminator side, which utilizes different types of data augmentation and contrastive learning to divide the feature space into geometric and appearance spaces; (2) a spectral regularization module on the generator side, which automatically categorizes input latent codes into geometry-related and appearance-related groups, and ensures that each component within each group controls a different independent attribute of the image via SVD decomposition. The input is a random noise vector $z$, and the output is the generated image, but the intermediate latent codes are structurally decomposed into independently manipulable subspaces.

### Key Designs

1. **Contrastive Feature Categorization**:

    - **Function**: To enable the discriminator to automatically construct geometric and appearance feature spaces.
    - **Mechanism**: Geometric augmentations (random cropping, rotation, etc.) and appearance augmentations (color jittering, grayscale conversion, etc.) are applied to the same image. The geometrically augmented image pair is physically similar in appearance but different in geometry; thus, contrastive learning can be used to pull their appearance features closer while pushing their geometric features apart, and vice versa. The intermediate features of the discriminator are mapped to the geometric and appearance spaces through two projection heads, respectively, and each is trained using the InfoNCE loss.
    - **Design Motivation**: Unlike supervised methods requiring annotations, this augmentation-invariance-based contrastive learning method is entirely unsupervised and directly exploits physical priors of geometry and appearance, rendering the categorization results more reasonable.

2. **Spectral Regularization**:

    - **Function**: To ensure that the components within the categorized feature subspaces are mutually orthogonal, with each component controlling an independent attribute.
    - **Mechanism**: After dividing the generator's input latent code $z$ into two groups: $z_g$ (geometry) and $z_a$ (appearance), the Jacobian matrix of the generator's intermediate layers is decomposed via SVD. By constraining the singular value distribution of the Jacobian (making it as uniform as possible), it is ensured that the sensitivity of the output to each input dimension is comparable and the directions are orthogonal. Specifically, nuclear norm regularization is applied to $\partial G / \partial z_g$ and $\partial G / \partial z_a$, respectively.
    - **Design Motivation**: Merely performing feature categorization can distinguish between the two major categories of geometry and appearance but cannot guarantee the independence of different dimensions within each category. Spectral regularization enforces orthogonality constraints, ensuring that each dimension corresponds to an independent, linearly interpolatable editing direction.

3. **Adaptive Feature Routing**:

    - **Function**: To guide the generator to automatically allocate input latent codes to the corresponding categories based on the geometric/appearance space learned by the discriminator.
    - **Mechanism**: After receiving the full latent code $z$, the generator splits it into $z_g$ and $z_a$ via a learnable routing module. The training signals for the routing module originate from the discriminator's contrastive feature space—changes in the generated image within the discriminator's geometric space should be primarily caused by $z_g$, while changes in the appearance space should be primarily caused by $z_a$.
    - **Design Motivation**: End-to-end training avoids manually designating which dimensions correspond to geometry and which to appearance, making the allocation more flexible and adaptable to the characteristics of different datasets.

### Loss & Training
The total loss consists of four components: (1) standard GAN adversarial loss to guarantee generation quality; (2) geometry contrastive loss $\mathcal{L}_{geo}$ and appearance contrastive loss $\mathcal{L}_{app}$ to train the discriminator's feature space; (3) spectral regularization loss $\mathcal{L}_{spec}$ to constrain the singular value distribution of the Jacobian matrix; (4) routing consistency loss to ensure that the generator's feature allocation is consistent with the discriminator's space division. A progressive training strategy is adopted, stabilizing adversarial training first, and then gradually increasing the weights of contrastive learning and spectral regularization.

## Key Experimental Results

### Main Results

| Dataset | Metric (FID↓) | LC-GAN | StyleGAN2 | EigenGAN | Gain |
|--------|-----------|--------|-----------|----------|------|
| FFHQ 256×256 | FID | 3.51 | 3.83 | 8.51 | +0.32 vs SG2 |
| CelebA-HQ 256×256 | FID | 4.12 | 4.39 | 7.86 | +0.27 vs SG2 |
| AFHQ-V2 512×512 | FID | 3.68 | 3.95 | 6.12 | +0.27 vs SG2 |

### Ablation Study

| Configuration | FID↓ | Disentanglement Score↑ | Description |
|------|------|----------|------|
| Full LC-GAN | 3.51 | 0.87 | Full model |
| w/o Contrastive Categorization | 3.72 | 0.61 | Disentanglement drops significantly without contrastive learning |
| w/o Spectral Regularization | 3.58 | 0.73 | Lack of orthogonality constraints leads to intra-class entanglement |
| w/o Adaptive Routing | 3.65 | 0.79 | Manual allocation is less flexible than adaptive routing |

### Key Findings
- Contrastive feature categorization is the largest contributor to disentanglement capability; removing it reduces the disentanglement score from 0.87 to 0.61.
- Spectral regularization significantly improves intra-class independence while maintaining FID.
- In user studies, the attribute editing precision of LC-GAN is significantly superior to post-processing methods such as GANSpace and SeFa.
- Linear interpolation experiments demonstrate that each dimension indeed controls an independent semantic attribute (e.g., one dimension controls pose, while another controls hair color).

## Highlights & Insights
- **Augmentation invariance as a disentanglement prior**: Utilizing the asymmetry between geometric and appearance augmentations to automatically distinguish feature categories is an ingenious design. This concept can be generalized to other tasks requiring unsupervised feature categorization (e.g., motion/appearance decomposition in video).
- **Spectral regularization guarantees linear controllability**: Enforcing orthogonality constraints via SVD ensures that linear interpolation can precisely control individual attributes without requiring complex non-linear mappings.
- **A win-win for generation quality and controllability**: Unlike most disentanglement methods that sacrifice generation quality, the FID of LC-GAN is actually superior to the original StyleGAN2, demonstrating that a structured latent space is also beneficial for generation.

## Limitations & Future Work
- Binary categorization into geometry/appearance might be too coarse; certain attributes (such as expression) involve simultaneous changes in both geometry and appearance, making them difficult to classify into a single category.
- Spectral regularization requires computing the SVD of the Jacobian matrix, which increases training overhead, especially at high resolutions.
- The method is based on the StyleGAN2 architecture, and its applicability to newer generators like diffusion models has not yet been verified.
- The method has currently been validated only on face and animal datasets; its performance on more complex scenes (e.g., multi-objective, indoor scenes) remains to be investigated.

## Related Work & Insights
- **vs GANSpace**: GANSpace discovers editing directions in the activation space of pre-trained GANs via PCA. It is a post-processing method, which does not guarantee direction orthogonality and can suffer from entanglement; LC-GAN achieves more thorough disentanglement through constraints during training.
- **vs EigenGAN**: EigenGAN also pursues orthogonal disentanglement but uses rotation matrix parameterization, leading to a decline in generation quality; LC-GAN's spectral regularization is more flexible and does not damage FID.
- **vs InfoGAN**: InfoGAN achieves disentanglement by maximizing the information bottleneck but lacks explicit separation of geometry and appearance, and its disentangled directions lack semantic interpretability.
- The combined idea of contrastive learning and GANs can also be applied to tasks such as conditional generation and style transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing the augmentation invariance of contrastive learning for GAN feature categorization is an ingenious design, and using spectral regularization to guarantee orthogonality is also natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets for both FID and disentanglement capability, with ablation study and user studies; both quantitative and qualitative results are relatively comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The methodology motivation is clear, with a complete logical chain from augmentation invariance to contrastive categorization and then to spectral regularization.
- Value: ⭐⭐⭐⭐ It holds practical value for controllable GAN generation, and the unsupervised disentanglement approach can be migrated to other generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)
- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)

</div>

<!-- RELATED:END -->
