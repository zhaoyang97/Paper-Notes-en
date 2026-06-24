---
title: >-
  [Paper Note] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection
description: >-
  [CVPR2025][Image Generation][Diffusion models] Proposes RAPTA (training-time region-aware prompt variant augmentation based on object detection) and ADMCD (inference-time training-free multimodal copy detection with three-stream attention fusion) to address the training data memorization issue in text-to-image diffusion models in an end-to-end manner from both mitigation and detection perspectives.
tags:
  - "CVPR2025"
  - "Image Generation"
  - "Diffusion models"
  - "Memorization mitigation"
  - "Prompt augmentation"
  - "Copy detection"
  - "Multimodal fusion"
date: 2026-05-08
content_hash: 8ff4e34265467074
---

# Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection

**Conference**: CVPR2025  
**arXiv**: [2603.13070](https://arxiv.org/abs/2603.13070)  
**Code**: To be confirmed  
**Area**: Object Detection  
**Keywords**: Diffusion models, Memorization mitigation, Prompt augmentation, Copy detection, Multimodal fusion  
**Institution**: University of Western Australia, University of Melbourne, Curtin University

## TL;DR
Proposes RAPTA (training-time region-aware prompt variant augmentation based on object detection) and ADMCD (inference-time training-free multimodal copy detection with three-stream attention fusion) to address the training data memorization issue in text-to-image diffusion models in an end-to-end manner from both mitigation and detection perspectives.

## Background & Motivation
**Memorization Phenomenon**: Text-to-image diffusion models (e.g., Stable Diffusion) are trained on large-scale weakly annotated web data. High model capacity, strong text-image alignment, and over-reliance on single caption-image pairs jointly cause the model to memorize and copy training data, raising copyright and privacy risks. Copying behaviors exist on a continuum—from pixel-level near-exact replication to style/semantic duplication.

**Limitations of Prior Mitigation Schemes**: Inference-time prompt perturbation methods (random token insertion, BLIP rewriting [10], adding Gaussian noise to CLIP embeddings [9]) reduce the copy rate but damage the prompt-image alignment quality, and do not resolve the root cause of memorization during training. Training-time schemes (such as generic template diversification or GLIGEN/ControlNet-style conditioning [11, 24]) may introduce semantic drift.

**Deficiencies in Detection**: Individual metrics have their own biases—LPIPS favors texture/color, ORB relies on rigid geometric keypoints, and SSIM is dominated by luminance/contrast—and exhibit low sensitivity to local or style copying, with thresholds varying across classes. The single-stream global fingerprint of SSCD [13] degrades significantly under flipping and occlusion, while DreamSim [5] optimizes for general perceptual similarity rather than copy detection.

**Key Challenge**: How to reduce training-time memorization without sacrificing generation quality? How to perform zero-shot copy detection without large-scale annotated data?

**Key Insight**: At training time, use detector proposals to generate semantically anchored prompt variants (rather than random perturbations). At detection time, use three-stream feature attention fusion to replace individual metrics and introduce two-level thresholds to distinguish retrieve/exact from style copying.

## Method

### Overall Architecture
Two complementary modules: (a) RAPTA operates during the training phase, diversifying prompts in a region-aware manner so the model sees multiple semantically anchored descriptions for the same image, reducing the memorization of specific caption-image pairs at the root; (b) ADMCD operates during the inference/evaluation phase, blending three complementary visual features for training-free copy detection and classification.

### RAPTA (Region-Aware Prompt Augmentation)
- **Region Detection**: For each training image, a pre-trained Faster R-CNN [16] is used to extract candidate proposals $\{(b_i, c_i, S_i)\}_{i=1}^N$. Overlapping boxes are merged via NMS (IoU threshold $\tau_{\text{nms}}$), and low-confidence detections $S_i \le \tau_b$ are discarded to keep the top-$M$ high-quality proposals.
- **Position Discretization**: The center point of each retained box is calculated, normalized by image dimensions, and mapped to a $3\times3$ grid $\mathcal{G} = \{\text{top-left}, \ldots, \text{bottom-right}\}$, yielding a coarse-grained position token $\text{pos}_i$.
- **Template Instantiation**: A few fill-in templates $\{T_j\}_{j=1}^J$ (e.g., "p, with a ⟨c⟩ in the ⟨pos⟩" or "p, featuring ⟨c⟩ and ⟨c'⟩") are used to combine (base prompt, category, position) into a set of region-aware variants: $V = \{p\} \cup \{T_j(p, c_i, \text{pos}_i) \mid i=1,\ldots,M;\ j=1,\ldots,J\}$.
- **CLIP Weighted Sampling**: For each variant $v \in V$, the CLIP image-text alignment score $S_v = \cos(f_I, f_v)$ is computed and converted to a non-negative weight $w_v = (S_v)_+^\gamma$ using a temperature parameter $\gamma > 0$, then normalized to a sampling distribution $\pi(v) = w_v / \sum_{u \in V} w_u$.
- **Training Pipeline**: In each iteration, a prompt $\tilde{p}$ is sampled from $\pi$ and encoded as the denoiser conditioning $e = \text{CLIP}_{\text{text}}(\tilde{p})$. The loss function remains the standard diffusion objective: $\mathcal{L}_{\text{diff}} = \mathbb{E}_{I,\epsilon,t,\tilde{p}}[\|\epsilon - \epsilon_\theta(x_t, t, e)\|_2^2]$.
- **Design Motivation**: Inference-time perturbation only exposes the model to a single text view and risks semantic drift. RAPTA exposes multiple semantically anchored descriptions across iterations during training, reducing the reliance on a single caption-image pairing at its root. $J$ is kept small to avoid combinatorial explosion; if no reliable detection box is found, $V = \{p\}$ degrades to original training, ensuring fallback safety.

### ADMCD (Attention-Driven Multimodal Copy Detection)
- **Three-Stream Feature Extraction**: (1) ViT [4] patch-level local visual descriptors $\mathbf{f}^{\text{vis}} \in \mathbb{R}^d$ capture spatial layout and geometric structure; (2) CLIP [15] global semantic descriptors $\mathbf{f}^{\text{clip}} \in \mathbb{R}^d$ provide color/illumination-invariant semantic representations; (3) ResNet [6] texture descriptors $\mathbf{f}^{\text{tex}} \in \mathbb{R}^d$ encode low-level texture and noise resilience.
- **Attention Fusion**: The three streams are projected linearly to a shared dimensional space, stacked, and fused via a lightweight Transformer encoder. The final embedding is L2-normalized: $\hat{\mathbf{f}}_{\text{fus}}(X) = \text{Attn}([\mathbf{f}^{\text{vis}}; \mathbf{f}^{\text{clip}}; \mathbf{f}^{\text{tex}}]) / \|\cdot\|_2$.
- **Stage 1—Copy Determination**: The fused cosine similarity is calculated as $S_{\text{fus}} = \cos(\hat{\mathbf{f}}_{\text{fus}}(G), \hat{\mathbf{f}}_{\text{fus}}(R))$. If $S_{\text{fus}} > \tau_1 = 0.938$, it is flagged as a copy.
- **Stage 2—Copy Type Classification**: For pairs flagged as copies, a stream-level similarity is calculated and weighted: $\bar{S} = 0.24 S_{\text{vis}} + 0.38 S_{\text{clip}} + 0.38 S_{\text{tex}}$. If $\bar{S} > \tau_2 = 0.970$, it is classified as a Retrieve/Exact copy; otherwise, it is classified as a Style copy.
- **Training-Free Deployment**: No task-specific training data is required. The thresholds $(\tau_1, \tau_2)$ and weights $(\omega_1, \omega_2, \omega_3)$ are determined once via grid search on a validation set and held constant during testing.

## Key Experimental Results

**Datasets**: LAION-10k; evaluation set of 1,200 pairs (approx. 25 retrieve/exact + approx. 200 style + approx. 1,000 non-copy); copy types manually annotated by 5 annotators.

### Copy Rate Comparison (Detected by ADMCD, lower is better)

| Method | Copy Rate | FID | CLIP Score | KID |
|------|----------|-----|-----------|-----|
| DCR [20] | 3.2 | 7.9 | 30.5 | 2.9 |
| LDM-T2I [2] | 5.3 | 10.4 | 33.2 | 3.1 |
| SD2.1-base [22] | 7.4 | 8.3 | 27.8 | 3.3 |
| **RAPTA (Ours)** | **2.6** | 8.1 | 23.1 | **1.6** |

- Compared to the three baselines, RAPTA reduces the copy rate by 18.8%, 50.9%, and 64.9%, respectively (an absolute reduction of 0.6, 2.7, and 4.8 percentage points).
- FID/KID remain comparable or better (KID 1.6 vs. 2.9–3.3), while the CLIP Score is slightly lower (23.1 vs. 27.8–33.2), reflecting the trade-off between copy suppression and precise text-image alignment.

### Noise Robustness (ADMCD vs. Single-Modal Metrics)

| Method | Original | Gaussian Noise | Gaussian Blur | Poisson | Salt & Pepper | Speckle |
|------|------|---------|---------|------|------|------|
| LPIPS [25] | 0.233 | 0.444 | 0.335 | 0.375 | 0.612 | 0.569 |
| SSIM [23] | 0.677 | 0.504 | 0.664 | 0.591 | 0.389 | 0.407 |
| SSCD [13] | 0.680 | 0.594 | 0.443 | 0.429 | 0.485 | 0.407 |
| DreamSim [5] | 0.857 | 0.781 | 0.714 | 0.691 | 0.689 | 0.707 |
| **ADMCD** | **0.974** | **0.923** | **0.940** | **0.929** | **0.871** | **0.894** |

### Geometric Robustness

| Method | Crop 20% | Horizontal Flip | Vertical Flip | Occlusion 10% | Rotation 30° |
|------|--------|---------|---------|--------|--------|
| SSIM [23] | 0.570 | 0.556 | 0.427 | 0.642 | 0.207 |
| SSCD [13] | 0.577 | 0.404 | 0.464 | 0.391 | 0.489 |
| DreamSim [5] | 0.617 | 0.524 | 0.564 | 0.691 | 0.689 |
| **ADMCD** | **0.970** | **0.886** | **0.857** | **0.748** | **0.939** |

- **Complementary Mechanism**: ViT provides spatial anchors, CLIP provides color/illumination invariance, and CNN provides noise/blur resilience. The attention fusion automatically downweights failing streams, ensuring no single vulnerability dominates the final score.

### Top-5 Retrieval Analysis
- ADMCD's fused similarity ranks the Top-5 candidates most clearly and stably (R1 score of 0.959 is significantly higher than R2–R5 scores of 0.850–0.878), whereas single-modal methods show small margins (e.g., in SSIM, R1=0.486 vs. R5=0.631, even in reverse order), making threshold decisions unreliable.
- This is the only method capable of simultaneously detecting copying and distinguishing between retrieve/exact vs. style copy types.

## Highlights & Insights
- **Joint Training-Time and Inference-Time Approach**: RAPTA reduces memorization at the source, while ADMCD detects copying at the output end. Together they form a complete pipeline that can be seamlessly integrated into existing diffusion workflows.
- **Semantically Anchored Prompt Variants**: Unlike random perturbations (which risk semantic drift), RAPTA utilizes detector proposals and CLIP consistency scoring to ensure each variant is a reasonable description of the image, where CLIP weighted sampling ensures high-quality variants are selected more frequently.
- **Training-Free Multimodal Copy Detection**: ADMCD requires no large-scale annotations and can differentiate between retrieve/exact and style copy types using two fixed thresholds, making it ready for deployment.
- **Three-Stream Attention Fusion as a Universal Similarity Metric**: It serves as a superior image similarity metric on its own, aligns better with human perception under 10 types of noise/geometric attacks than SSIM/LPIPS/DreamSim/SSCD, and works across conditions with a single pair of thresholds.
- **Interpretable Copy Classification**: Diagnosing the source of copy types (texture similarity vs. semantic similarity vs. structural similarity) is achieved via the individual similarity contributions of the three streams, offering actionable signals for downstream analysis.

## Limitations & Future Work
- The evaluation set consists of only 1,200 pairs (with only about 25 retrieve/exact pairs), which is relatively small and limits statistical robustness; the inherent rarity of near-exact duplicates makes collecting more positive samples challenging.
- RAPTA relies on the coverage of the pre-trained Faster R-CNN and may fail for rare objects (e.g., abstract art) that the detector cannot recognize.
- The drop in CLIP Score (23.1 vs. 27.8–33.2) reflects that the trade-off between prompt diversification and precise text-image alignment has not been fully resolved.
- The thresholds $\tau_1, \tau_2$ of ADMCD may require recalibration when transferring to other data domains, as they were only validated on LAION-10k in this paper.
- Lack of combinatorial experiments with other training-time mitigation strategies (such as data deduplication or differential privacy training).

## Related Work & Insights
- **Diffusion Model Foundations**: DDPM [8] proposes learning a reverse progressive corruption process to achieve high-fidelity generation, while Latent Diffusion [17] operates in a compressed latent space to improve efficiency.
- **Memorization Research**: [9, 20, 21] systematically document the behavior of diffusion models copying training data in small/large-scale retrieval, while [1] raises security concerns by extracting training data from diffusion models.
- **Inference-Time Mitigation**: Implemented via random token insertion, BLIP rewriting [10], and adding noise to CLIP embeddings [9]; training-time schemes include explicit target/position conditioning like GLIGEN [11]/ControlNet [24].
- **Copy Detection**: Traditional copy-move forensics (SIFT/SURF/ORB) [18], learned fingerprinting SSCD [13], and perceptual distances such as LPIPS [25]/DISTS [3]/DreamSim [5] are all single-stream and struggle to differentiate copying types.
- **Robustness Benchmarks**: ImageNet-C [7] reveals the vulnerability of perceptual metrics to noise, blur, and weather attacks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined scheme of training-time region-aware prompt enhancement and inference-time multimodal fusion detection is novel, and the two-stage threshold copy classification is practical.
- Experimental Thoroughness: ⭐⭐⭐ Comparisons across three backbones and robustness tests under 10 attacks are comprehensive, but the evaluation set size is small (especially with only 25 retrieve/exact pairs).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive pipeline diagrams, and complete algorithm pseudocode.
- Value: ⭐⭐⭐⭐ Highly practical for copyright protection in diffusion models, and ADMCD holds independent value as a universal similarity metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RAD: Region-Aware Diffusion Models for Image Inpainting](rad_region-aware_diffusion_models_for_image_inpainting.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [\[ICML 2025\] Localizing and Mitigating Memorization in Image Autoregressive Models](../../ICML2025/image_generation/localizing_and_mitigating_memorization_in_image_autoregressive_models.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](../../ICML2025/image_generation/understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)

</div>

<!-- RELATED:END -->
