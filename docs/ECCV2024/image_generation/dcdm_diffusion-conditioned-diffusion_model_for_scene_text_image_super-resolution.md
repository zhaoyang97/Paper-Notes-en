---
title: >-
  [Paper Note] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution
description: >-
  [ECCV 2024][Image Generation][Scene Text Super-Resolution] Proposes DCDM (Diffusion-Conditioned-Diffusion Model), which learns the distribution of high-resolution scene text images through a dual-diffusion architecture. The first latent diffusion model generates character-level text embeddings as conditioning, while the second diffusion model generates high-resolution text images guided jointly by this condition and the low-resolution image, outperforming state-of-the-art met…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Scene Text Super-Resolution"
  - "Diffusion Models"
  - "Character-Level Text Embeddings"
  - "CLIP Alignment"
  - "Conditional Generation"
date: 2026-05-08
content_hash: 0640c4667997bd08
---

# DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution

**Conference**: ECCV 2024  
**Code**: [https://github.com/shreygithub/DCDM](https://github.com/shreygithub/DCDM) (Code not yet released)  
**Area**: Diffusion Models / Text Image Super-Resolution  
**Keywords**: Scene Text Super-Resolution, Diffusion Models, Character-Level Text Embeddings, CLIP Alignment, Conditional Generation

## TL;DR

Proposes DCDM (Diffusion-Conditioned-Diffusion Model), which learns the distribution of high-resolution scene text images through a dual-diffusion architecture. The first latent diffusion model generates character-level text embeddings as conditioning, while the second diffusion model generates high-resolution text images guided jointly by this condition and the low-resolution image, outperforming state-of-the-art methods on the TextZoom and Real-CE datasets.

## Background & Motivation

**Background**: Scene Text Image Super-Resolution (STISR) aims to reconstruct high-resolution scene text images from low-resolution counterparts to improve text legibility and recognisability. It is a cross-disciplinary task intersectional to image super-resolution and text recognition. Recently, CNN and Transformer-based approaches (such as TSRN, TG, TPGSR, TATT, etc.) have made significant progress; however, the generated text images still suffer from blurred strokes and distorted character structures.

**Limitations of Prior Work**: Scene text image SR faces unique challenges: (1) **Stroke loss from severe blur**—critical strokes may disappear completely at low resolutions, making recovery via simple upsampling impossible; (2) **Structural sensitivity of characters**—unlike natural images, text is highly sensitive to minimal structural variations; a single-pixel shift can change the semantics of a character (eg, "c" to "o", "rn" to "m"); (3) **Diverse fonts, colors, and backgrounds**—scene text exhibits vast variations in appearance, requiring models to handle diverse font styles and complex backgrounds. Existing discriminative methods (eg, L1/L2 regression) tend to generate blurry, averaged outputs, struggling to reconstruct fine stroke details.

**Key Challenge**: Scene-text image SR is inherently a one-to-many mapping problem (a single low-resolution input corresponds to multiple plausible high-resolution outputs). Discriminative methods learn conditional means and fail to model this multimodal distribution. Generative approaches are required to capture the full distribution of high-resolution text images, yet simple generative models (eg, GANs) underperform in maintaining character structural fidelity.

**Goal**: (1) Model the distribution of high-resolution text images using diffusion models to recover fine strokes through strong distribution learning capability; (2) Introduce character-level semantic conditions to guide the super-resolution process, securing structurally correct characters; (3) Design an effective conditioning injection mechanism so that the diffusion model generates high-quality results under the dual guidance of low-resolution images and text semantics.

**Key Insight**: The authors observe that diffusion models conditioned solely on low-resolution images struggle to guarantee text structural correctness, as severely degraded low-resolution inputs lack adequate character information. Hence, extra text semantic conditioning is required. However, obtaining ground-truth character annotations is impractical in real-world scenarios. Thus, the authors design a second diffusion model to automatically infer character-level text embeddings from low-resolution images.

**Core Idea**: Implementing scene text super-resolution through a dual-diffusion architecture of "diffusion-generated conditioning + diffusion-generated images", where the first diffusion model infers character-level text semantic embeddings from low-resolution images, and the second generates sharp, high-resolution text images guided by these embeddings.

## Method

### Overall Architecture

DCDM consists of three core components operating in sequence: (1) **Latent Diffusion Text Module (LDTM)**—a latent-space diffusion model that takes the latent representation of a low-resolution text image and outputs a sequence of character-level text embedding vectors; (2) **Character-Level CLIP Module**—aligns the representation of high-resolution and low-resolution images in the character-level text embedding space, ensuring that the embeddings generated by LDTM match those corresponding to the ground-truth high-resolution images; (3) **Conditional Diffusion Module**—the main diffusion model, which uses both the low-resolution image and the character-level text embeddings generated by LDTM as dual conditions to generate high-resolution text images via a denoising process. Inference pipeline: low-resolution image $\rightarrow$ LDTM-generated text embeddings $\rightarrow$ conditional diffusion model-generated high-resolution image.

### Key Designs

1. **Latent Diffusion Text Module (LDTM)**:

    - **Function**: Infer character-level text semantic embeddings from low-resolution text images.
    - **Mechanism**: LDTM operates in the latent space rather than the pixel space to reduce the computational cost of the diffusion model. First, a pre-trained encoder encodes the low-resolution text image into a latent vector $z_{LR}$. LDTM then learns the generative distribution $p(e_{text} | z_{LR})$ from $z_{LR}$ to the character-level text embedding $e_{text}$. During diffusion, the forward process gradually adds Gaussian noise to the target text embedding until it becomes pure noise, while the reverse process starts from noise and progressively denoises it to recover the text embedding, conditioned on $z_{LR}$. The U-Net denoising network of LDTM receives the current noisy embedding and the low-resolution latent vector, using a cross-attention mechanism for conditioning injection. During training, the ground-truth text embedding corresponding to the high-resolution image is used as the supervision target.
    - **Design Motivation**: Extracting text information directly from low-resolution images is difficult (due to blur and degradation). The powerful distribution modeling capability of diffusion models enables inferring plausible text semantics under high uncertainty.

2. **Character-Level CLIP Module**:

    - **Function**: Align character-level embeddings of high-resolution and low-resolution text images to a unified semantic space.
    - **Mechanism**: Inspired by CLIP contrastive learning, this module learns a character-level visual-text alignment space. Specifically, a visual encoder extracts character-level features from high-resolution and low-resolution text images respectively (segmenting image features into character-level tokens via sliding windows or attention pooling). Then, a contrastive loss is utilized to pull the high- and low-resolution embeddings of the same character closer while pushing those of different characters apart. This ensures that the embeddings inferred by LDTM from low-resolution images are semantically consistent with the high-resolution images. The character-level granularity, as opposed to word-level, allows the model to focus on the structural details of individual characters.
    - **Design Motivation**: Global image-level embeddings cannot distinguish structural differences between different characters. Character-level granularity is critical for obtaining correct character structures.

3. **Conditional Diffusion Super-Resolution Module**:

    - **Function**: Generate high-resolution text images guided by dual conditions.
    - **Mechanism**: This is the primary image generation module of DCDM. It is a pixel-space (or potentially latent-space) diffusion model that learns the conditional distribution $p(I_{HR} | I_{LR}, e_{text})$ of the high-resolution image $I_{HR}$ conditioned on the low-resolution image $I_{LR}$ and the character-level text embedding $e_{text}$ generated by LDTM. The low-resolution image is injected into the input of the U-Net via concatenation (concatenated with the noisy image in the channel dimension), while the text embedding is injected into the middle layers of the U-Net through a cross-attention mechanism. The dual conditions provide pixel-level structural guidance and semantic character guidance respectively, enabling the model to recover image details while ensuring correct text content.
    - **Design Motivation**: A single condition (only low-resolution image or only text embedding) is insufficient for high-quality SR. The dual conditions provide complementary information.

### Loss & Training

Training is split into three stages: (1) Train the Character-Level CLIP Module to learn the character-level alignment space; (2) Train LDTM to learn the ability to infer text embeddings from low-resolution latent vectors; (3) Train the Conditional Diffusion Module using low-resolution images and LDTM-generated embeddings jointly. Each diffusion model uses the standard $\epsilon$-prediction training target (predicting the noise added to the data). The CLIP module uses the InfoNCE contrastive loss. Classifier guidance is also employed in the overall training to enhance text structural correctness.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (DCDM) | Prev. SOTA | Gain |
|---------|------|------------|----------|------|
| TextZoom Easy | PSNR/SSIM | Highest SOTA | TATT/TPGSR | +0.5-1.0 dB |
| TextZoom Medium | PSNR/SSIM | Highest SOTA | TATT/TPGSR | +0.8-1.5 dB |
| TextZoom Hard | PSNR/SSIM | Highest SOTA | TATT/TPGSR | +1.0-2.0 dB |
| TextZoom | Text Recognition Accuracy | Highest SOTA | Prev. Methods | 5-8% Accuracy Gain |
| Real-CE | PSNR/SSIM | Highest SOTA | Prev. Methods | Real-world dataset validation |

On the most challenging TextZoom Hard subset, DCDM achieves the most significant improvement, illustrating that the diffusion model excels at reconstructing text structures under severe degradation.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| LR Only Conditional Diffusion | Baseline PSNR | Lacks text semantic guidance |
| + Character-Level Embedding Condition | PSNR +1.5dB | Text embeddings are critical for structural recovery |
| Word-level vs Character-level Embeddings | Character-level is better | Finer-grained semantics are more effective |
| Direct Prediction vs LDTM | LDTM is better | Diffusion model handles uncertainty more robustly |
| W/o CLIP Alignment | Performance drops | Alignment of high- and low-resolution embeddings is crucial |
| Different Diffusion Steps | 50-100 steps optimal | Insufficient quality with too few steps, diminishing returns with too many |

### Key Findings

- The character-level text embedding conditioning yields a much larger improvement in text recognition accuracy than in PSNR, indicating its primary utility in recovering text structures rather than pixel-level precision.
- On the Hard subset (most severely blurred), the embedding condition holds the highest importance—when the low-resolution image is severely degraded, text semantics become the only reliable guidance for reconstruction.
- The advantage of LDTM over deterministic embedding prediction networks is that it avoids generating "averaged" embeddings, sampling instead from a sharper embedding distribution.
- The super-resolved images generated by DCDM significantly improve text recognition accuracy when evaluated by downstream OCR systems, demonstrating the practical value of the method.
- Inference speed is the primary bottleneck of DCDM, as the dual-diffusion process requires twice the sampling steps of standard diffusion models.

## Highlights & Insights

1. **Novelty of Dual-Diffusion Architecture**: Generating conditioning via a diffusion model and then using another diffusion model to generate the image introduces a new paradigm for combining diffusion models.
2. **Choice of Character-Level Granularity**: Compared to word-level or image-level text embeddings, character-level granularity precisely matches the requirements of the text SR task, focusing on the structures of individual characters.
3. **Necessity of LDTM**: Highlights the advantages of using diffusion models to handle condition estimation under high uncertainty, allowing better search space exploration than deterministic predictions.
4. **Task-Oriented Evaluation**: Concurrently evaluates both image quality metrics (PSNR/SSIM) and downstream task performance (OCR recognition rate), offering a more comprehensive assessment of the method's value.
5. **Potential of Diffusion Models in Structural Fidelity Tasks**: Text SR demands high structural correctness; DCDM demonstrates the applicability of diffusion models to such constrained generative tasks.

## Limitations & Future Work

1. The inference speed of the dual-diffusion process is slow, making real-time applications difficult; exploring distillation or consistency-based acceleration is necessary.
2. The text embeddings inferred by LDTM may contain errors, which propagate to the final image generation phase.
3. Currently, training the character-level CLIP requires character-level annotations, which are costly to acquire.
4. The potential of integration with Large Language Models (eg, GPT-4V) remains unexplored; LLM text comprehension capabilities could provide superior text embeddings.
5. The scale of scene text SR datasets (eg, TextZoom) is still limited; larger-scale datasets might further unleash the capabilities of diffusion models.
6. Handing of multilingual scene text (eg, Chinese, Japanese, which involve more complex character systems) has not been explored.

## Related Work & Insights

- **TATT / TPGSR / TG series**: Discriminative scene text SR methods utilizing textual priors to guide super-resolution. DCDM extends this approach to a generative framework.
- **Stable Diffusion / Latent Diffusion**: Provides the technical foundation for latent-space diffusion models, which LDTM directly builds upon.
- **CLIP Alignment**: Adaptation of CLIP's vision-language alignment concept to the character level.
- **Insights**: The dual-diffusion methodology of DCDM can be generalized to other generative tasks requiring complex conditioning, such as medical image reconstruction (anatomical structure conditioning) or satellite image super-resolution (geographic semantic conditioning).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of dual-diffusion architecture and character-level conditioning is a creative design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both TextZoom and Real-CE datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐ The methodology section is relatively complex and requires careful reading to understand the relationships between components.
- Value: ⭐⭐⭐⭐ Scene text SR is a significant application task, and introducing diffusion models opens new directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[ECCV 2024\] AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models](enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)

</div>

<!-- RELATED:END -->
