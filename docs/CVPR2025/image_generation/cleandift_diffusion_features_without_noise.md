---
title: >-
  [Paper Note] CleanDIFT: Diffusion Features without Noise
description: >-
  [CVPR 2025][Image Generation][Diffusion Features] Proposes CleanDIFT, which enables diffusion models to directly extract high-quality semantic features on clean images through lightweight unsupervised fine-tuning (only 30 minutes on a single A100 GPU). This eliminates the limitations of traditional methods requiring noise addition and timestep tuning, significantly outperforming standard diffusion features on multi-tasks such as semantic correspondence, depth estimation…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Features"
  - "Noise-Free Feature Extraction"
  - "Semantic Correspondence"
  - "Knowledge Distillation"
  - "Timestep-Agnostic"
date: 2026-05-08
content_hash: 3f9175f1aebb9c43
---

# CleanDIFT: Diffusion Features without Noise

**Conference**: CVPR 2025  
**arXiv**: [2412.03439](https://arxiv.org/abs/2412.03439)  
**Code**: [https://compvis.github.io/cleandift](https://compvis.github.io/cleandift)  
**Area**: Diffusion Models / Self-Supervised Learning  
**Keywords**: Diffusion Features, Noise-Free Feature Extraction, Semantic Correspondence, Knowledge Distillation, Timestep-Agnostic

## TL;DR
Proposes CleanDIFT, which enables diffusion models to directly extract high-quality semantic features on clean images through lightweight unsupervised fine-tuning (only 30 minutes on a single A100 GPU). This eliminates the limitations of traditional methods requiring noise addition and timestep tuning, significantly outperforming standard diffusion features on multi-tasks such as semantic correspondence, depth estimation, and segmentation.

## Background & Motivation

**Background**: Internal representations of pre-trained diffusion models (diffusion features) have been proven to be powerful semantic descriptors, performing exceptionally well in tasks like semantic correspondence matching, segmentation, and detection. However, extracting these features requires adding noise to the image before feeding it into the model, as diffusion models are trained on noisy images.

**Limitations of Prior Work**: Adding noise brings three challenges. (1) **Information loss**: Noise destroys high-frequency details of the image; even at a low noise level of $t=261$, the reconstructed image has lost a significant amount of information. (2) **Feature encoded noise**: Experiments show that even at low $t$ values, a considerable portion of variance in diffusion features comes from noise rather than image information. (3) **Timestep dependency**: Different downstream tasks require different optimal $t$ values (e.g., $t=261$ for semantic correspondence, $t=100$ for segmentation), demanding per-task hyperparameter tuning. Some methods mitigate this by multi-noise ensembling (e.g., averaging 8 times in DIFT), which multiplies the computational overhead.

**Key Challenge**: Diffusion models learn different levels of semantic information at different noise levels (high noise $\rightarrow$ coarse structures, low noise $\rightarrow$ fine details), which provides multi-scale features but also causes timestep dependency. It is necessary to integrate information from all timesteps into a single noise-free forward pass.

**Goal**: How to extract noise-free, timestep-agnostic universal semantic features from diffusion models while preserving or exceeding the performance of noisy, multi-timestep features.

**Key Insight**: Treat the diffusion model as a family of $T$ different feature extractors (one for each timestep). Train a student model to output features on clean images, and align its features to all $T$ teacher feature extractors using timestep-conditioned projection heads, integrating multi-timestep information into a single feature space.

**Core Idea**: Use a feature extraction model operating on clean images and align its features with those of a frozen diffusion model at all noise levels via timestep-conditioned projection heads, achieving noise-free, timestep-agnostic unified feature extraction.

## Method

### Overall Architecture
Using a pre-trained diffusion model (SD 1.5/2.1) as the teacher, initialize a trainable copy as the student feature extractor. The student receives clean images (no noise, no timestep inputs), while the teacher receives the corresponding noisy images and timesteps. At $K=11$ feature extraction positions (U-Net intermediate blocks + decoder blocks), the student features are projected into the teacher feature space using timestep-conditioned projection heads to maximize cosine similarity. Training only requires 400 steps/30 minutes. During inference, the projection heads are discarded, and the internal features of the student model are directly used.

### Key Designs

1. **Timestep-Conditioned Projection Heads**

    - **Function**: Align the timestep-agnostic features of the student model with the timestep-specific features of the teacher at different noise levels.
    - **Mechanism**: Each feature extraction position $k$ has an independent projection head $\text{proj}^{(k)}(\text{feat}_c^{(k)}(\mathbf{x}_0); t)$, consisting of 3 layers of zero-initialized FFNs (initially acting as identity mapping due to residual connections). The projection head takes timestep $t$ as additional input and learns to map the unified feature to the feature space of the corresponding timestep. The loss is $\mathcal{L} = -\sum_{k=1}^{K} \text{sim}(\text{proj}^{(k)}(\text{feat}_c^{(k)}(\mathbf{x}_0); t), \text{feat}^{(k)}(\mathbf{x}_t; t))$.
    - **Design Motivation**: Since diffusion features at different timesteps possess different semantic attributes, the projection heads map the student's unified feature to each timestep space, forcing the student feature to contain information from all timesteps.

2. **Stratified Timestep Sampling**

    - **Function**: Efficiently cover the complete noise spectrum for each training image.
    - **Mechanism**: Divide $[0, T]$ into $I=3$ uniform intervals, and uniformly sample one timestep $t_i \sim \mathcal{U}(\frac{i}{I}T, \frac{i+1}{I}T)$ within each interval. Construct alignment losses for 3 different noise levels per image, sharing a single student forward pass.
    - **Design Motivation**: Ensure that training covers low, medium, and high noise ranges, avoiding bias toward any single noise region.

3. **Noise-Feature Relationship Analysis**

    - **Function**: Provide theoretical motivation by showing that a large amount of noise is encoded in standard diffusion features.
    - **Mechanism**: Analyze the proportion of variance in the full feature $\text{feat}(\mathbf{x}_t; t)$ explained by the pure noise $\text{feat}(\epsilon; T)$ via linear regression. Discover that even at $t=261$, noise explains a substantial portion of the feature variance. Furthermore, the remaining variance cannot be fully explained by the clean image feature at $t=0$, indicating that intermediate timesteps contain unique semantic information (not present in $t=0$ or $t=T$).
    - **Design Motivation**: Directly feeding clean images to the diffusion model at $t=0$ performs poorly, because the model was not trained on clean images. Dedicated fine-tuning is required to enable the model to output useful features under noise-free inputs.

### Loss & Training
Cosine similarity alignment loss, summed over $K=11$ feature locations. Full fine-tuning of the student model + projection heads (45M parameters). Trained using Adam optimizer with a learning rate of 2e-6, linear warmup, and a batch size of 8. Training is conducted on a random subset of COYO-700M (in-distribution with SD training data, ensuring gains come from the method rather than data). Convergence is achieved in just 400 steps.

## Key Experimental Results

### Main Results

| Task/Method | Standard Diffusion Features | CleanDIFT | Gain |
|----------|------------|-----------|------|
| Semantic Correspondence DIFT PCK_img | 66.53 | **68.32** | +1.79 |
| Semantic Correspondence TaleOfTwo PCK_img | 72.31 | **73.35** | +1.04 |
| Semantic Correspondence TellLeftRight PCK_img | 77.07 | **78.40** | +1.33 |
| Depth Estimation NYUv2 RMSE↓ | 0.469 | **0.444** | -0.025 |
| Semantic Segmentation VOC mIOU | Optimal $t$-value results | **Outperforms all $t$** | — |
| Classification ImageNet kNN | Optimal $t$-value results | **Slightly outperforms** | — |

### Ablation Study

| Configuration | PCK_img | Description |
|------|---------|------|
| DIFT (8x noise ensemble) | 66.53 | 8x computational overhead |
| DIFT (single noise) | 65.51 | Baseline |
| **CleanDIFT (noise-free single)** | **68.32** | Outperforms 8x ensemble with 1x overhead |
| Supervised fine-tuning DHF (50-step inversion) | 72.75 | 50x overhead |
| **CleanDIFT + Supervised fine-tuning** | **72.48** | Close to 50x performance with 1x overhead |

### Key Findings
- Noise-free features consistently outperform noisy features at their optimal timesteps across all tasks, proving that noise indeed degrades feature quality.
- Simply feeding clean images with a non-zero $t$ to diffusion models is insufficient (leads to worse performance); dedicated fine-tuning is required.
- Performance gains hold when integrated with advanced methods (+DINOv2 integration, +pose alignment, both yielding over 1+ point gains).
- 30-minute fine-tuning with 1x computational overhead (no ensemble needed) provides an 8-50x efficiency boost.
- Serves as a direct drop-in replacement for standard diffusion features.

## Highlights & Insights
- The insight that **noise is the bottleneck of feature quality**, while obvious in hindsight, was not systematically validated before. The variance decomposition experiment clearly demonstrates the noise proportion in features.
- The idea of **integrating $T$ feature extractors into one** is highly elegant — achieved via timestep-conditioned projection heads to realize "one single forward pass, knowledge of all timesteps".
- **Extremely low training cost** (400 steps / 30 minutes / single GPU) makes the method highly practical; any workflow using diffusion features can benefit directly.

## Limitations & Future Work
- Only validated on SD 1.5/2.1 (U-Net); less validation on DiT architectures (though preliminary results exist in Tab. 7).
- Applicability to larger models like SDXL and Flux has not been fully explored.
- Projection heads add 45M parameters (though discarded during inference).
- Performance improvement on classification tasks is modest, indicating that noise impact diminishes after pooling, and the primary advantages lie in dense prediction tasks.

## Related Work & Insights
- **vs DIFT**: DIFT requires noise addition + 8x ensembling. CleanDIFT outperforms it with a single noise-free forward pass, achieving 8x acceleration + better performance.
- **vs A Tale of Two Features / Telling Left Right**: These methods merge DINOv2 and diffusion features. Replacing them with CleanDIFT leads to consistent improvements, showing that gains stem from the inherent quality enhancement of the features.
- **vs DHF (Supervised fine-tuning)**: DHF requires a 50-step DDIM inversion. CleanDIFT achieves comparable performance with a single forward pass, yielding 50x speedup.

## Rating
- Novelty: ⭐⭐⭐⭐ Deep insights (noise degrades features), elegant method (multi-timestep integration), but technically standard distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering four downstream tasks + integration with various methods + detailed ablations + variance analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly persuasive motivation analysis (variance decomposition figures) and clear experimental organization.
- Value: ⭐⭐⭐⭐⭐ Highly deployable due to low training cost; any work leveraging diffusion features can directly benefit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)

</div>

<!-- RELATED:END -->
