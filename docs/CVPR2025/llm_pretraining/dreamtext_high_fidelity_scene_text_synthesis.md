---
title: >-
  [Paper Note] DreamText: High Fidelity Scene Text Synthesis
description: >-
  [CVPR 2025][LLM Pretraining][Scene Text Synthesis] DreamText reconstructs the training pipeline of diffusion models, introducing character-level balanced supervision and a heuristic alternate optimization strategy to calibrate character attention. Combined with the joint training of the text encoder and generator to learn diverse font styles, it significantly outperforms state-of-the-art methods in scene text synthesis tasks (improving SeqAcc from 0.763 of UDiffText to 0.940)…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Scene Text Synthesis"
  - "Diffusion Models"
  - "Character Attention"
  - "Balanced Supervision"
  - "Alternate Optimization"
date: 2026-05-08
content_hash: d0d2fa33a77fb278
---

# DreamText: High Fidelity Scene Text Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2405.14701](https://arxiv.org/abs/2405.14701)  
**Code**: [https://github.com/OpenBMB/DreamText](https://github.com/OpenBMB/DreamText)  
**Area**: Diffusion Models / Text Generation  
**Keywords**: Scene Text Synthesis, Diffusion Models, Character Attention, Balanced Supervision, Alternate Optimization

## TL;DR

DreamText reconstructs the training pipeline of diffusion models, introducing character-level balanced supervision and a heuristic alternate optimization strategy to calibrate character attention. Combined with the joint training of the text encoder and generator to learn diverse font styles, it significantly outperforms state-of-the-art methods in scene text synthesis tasks (improving SeqAcc from 0.763 of UDiffText to 0.940).

## Background & Motivation

**Background**: Scene text synthesis aims to render specified texts into arbitrary images. Existing methods can be categorized into GAN-based (e.g., MOSTEL) and diffusion-based (e.g., TextDiffuser, AnyText, UDiffText) methods. Diffusion-based methods have gradually become the mainstream due to their superior generation capabilities.

**Limitations of Prior Work**: Current diffusion-based methods suffer from three core issues: (1) **character distortion**: incorrectly generated character shapes; (2) **character repetition**: a single character being generated multiple times; (3) **character absence**: some characters failing to be rendered. These issues are particularly severe in polystylistic scenarios. The root cause is the lack of effective **character-level guidance** during training in existing methods—the model does not know where each character should appear, leading to attention being scattered in incorrect regions.

**Key Challenge**: Text synthesis requires precise control over the position and shape of each character, yet the end-to-end training of diffusion models relies solely on image-level loss, lacking character-level spatial constraints. Additionally, the text encoders in most methods are pre-trained on a single font style, failing to adapt to the highly diverse font styles in real-world applications.

**Goal**: (1) To introduce fine-grained character-level guidance during diffusion training to expose and correct character-level attention in the model; (2) to jointly train the text encoder and image generator, enabling the learning of diverse fonts in the training set.

**Key Insight**: A "latent character mask" for each character can be extracted from the cross-attention maps, representing where the model perceives the character should appear. Aligning this mask with the ground-truth character position allows for the calibration of the model's attention.

**Core Idea**: To design a balanced supervision strategy where a latent character mask is first encoded from the cross-attention map at each training step, and then these masks are used to update the character embeddings, enabling the generator to correct character attention in the next step. The text encoder and generator learn cooperatively through this alternate optimization process.

## Method

### Overall Architecture

DreamText is based on the Stable Diffusion inpainting pipeline. The input consists of the original image, a mask region, and the target text to be rendered. A text encoder encodes the target text into a sequence of character embeddings, which are then passed to the UNet of the diffusion model for conditional denoising. The core innovation lies in the training process: at each training step, a latent character mask is extracted from UNet's cross-attention map, character embeddings are updated using balanced supervision signals, and then the UNet corrects its attention distribution guided by the updated embeddings. This constitutes a mixed optimization problem involving discrete variables (character masks) and continuous variables (embedding weights).

### Key Designs

1. **Latent Character Mask Extraction & Balanced Supervision**:

    - **Function**: To extract spatial position priors for each character and guide character attention calibration with moderate supervision intensity.
    - **Mechanism**: During training, attention maps corresponding to each character token are extracted from the cross-attention maps of various UNet layers. After aggregation, they yield the "latent character mask", representing where the model perceives the character should be located. Balanced supervision compares these masks with the ground-truth character segmentation masks. Instead of employing overly strong supervision (such as pure cross-entropy), it designs a "balanced" strategy that lies between unsupervised and strongly supervised schemes. Specifically, it grants the model a certain degree of freedom to estimate the optimal character locations while providing positional correction signals via the ground-truth masks.
    - **Design Motivation**: Unsupervised training (FID 62.36, SeqAcc 0.212) completely fails because the model lacks prior knowledge of character locations; strong supervision (FID 14.92, SeqAcc 0.862) overly constrains character locations, limiting the model's flexibility to adapt to complex scenarios (such as curved text and non-standard layouts); balanced supervision (FID 12.13, SeqAcc 0.940) achieves optimal performance, striking a balance between guidance and flexibility.

2. **Heuristic Alternate Optimization**:

    - **Function**: To solve the mixed optimization problem involving discrete variables (character masks) and continuous variables (character embeddings, generator parameters).
    - **Mechanism**: Each training step is divided into two alternately executed sub-steps. **E-step**: The generator parameters are fixed, and the latent mask of each character is "Estimated" from the current cross-attention map, which encodes position information from attention. **M-step**: Fixing the estimated mask, the character embedding representations are updated using balanced supervision signals, embedding the character position information. Then, the generator performs denoising under the guidance of the updated embeddings, thereby correcting the attention distribution in the next step. This alternate optimization establishes a positive feedback loop between "character position estimation" and "attention correction".
    - **Design Motivation**: Text synthesis is essentially a mixed optimization problem—character masks are discrete spatial allocations, whereas embedding weights are continuous. EM-style alternate optimization is a classic strategy to solve such problems.

3. **Joint Training of Text Encoder and Generator**:

    - **Function**: To enable the text encoder to learn diverse font styles present in the training set.
    - **Mechanism**: Discarding the common practice of freezing the pre-trained text encoder in existing methods, DreamText incorporates the text encoder into the training loop. In the M-step of the alternate optimization, updating the character embeddings not only corrects positional information but also enables the encoder to learn the visual features of various fonts. This joint training is naturally integrated into the alternate optimization framework—the encoder learns font embeddings and position estimation, while the generator learns to correctly render characters based on these embeddings.
    - **Design Motivation**: Pre-trained text encoders are typically trained on a single font style (such as Arial) and lack representational capacity when dealing with diverse styles like handwriting and artistic fonts. Joint training allows the encoder to "encounter" diverse fonts within the training set.

### Loss & Training

Standard diffusion denoising loss (MSE between predicted noise and actual noise) combined with the balanced supervision loss is used. The training data utilizes the LAION-OCR subset, which provides diverse scene text images and corresponding character-level segmentation masks (used for balanced supervision).

## Key Experimental Results

### Main Results (LAION-OCR Test Set)

| Method | SeqAcc (Recon)↑ | SeqAcc (Editing)↑ | FID↓ | Type |
|------|-----------------|-------------------|------|------|
| MOSTEL | Low | Low | High | GAN |
| SD-inpainting v2.0 | Low | Low | High | Diffusion |
| DiffSTE | Medium | Medium | Medium | Diffusion |
| TextDiffuser | Medium | Medium | Medium | Diffusion |
| AnyText | Medium | Medium | Medium | Diffusion |
| UDiffText | 0.763 | — | ~15 | Diffusion |
| **DreamText (Ours)** | **0.940** | **0.887** | **12.13** | Diffusion |

### Ablation Study on Balanced Supervision

| Supervision Strategy | SeqAcc (Recon) | SeqAcc (Editing) | FID | mIoU |
|---------|---------------|------------------|-----|------|
| Unsupervised | 0.212 | 0.157 | 62.36 | 0.203 |
| Strong Supervision (Cross-Entropy) | 0.862 | 0.813 | 14.92 | 0.617 |
| **Balanced Supervision (Ours)** | **0.940** | **0.887** | **12.13** | **0.722** |

### Key Findings

- **Significant SeqAcc Improvement**: DreamText's sequence accuracy (0.940) is 23.2% higher than the best baseline UDiffText (0.763), highlighting the high effectiveness of character-level guidance.
- **Necessity of Balanced Supervision**: Unsupervised learning completely fails (SeqAcc 0.212), indicating that diffusion models cannot reliably generate correct texts without character-level guidance. Strong supervision (0.862) is decent but still has a gap, as excessive constraints limit flexibility.
- **mIoU Validates Attention Calibration**: Under balanced supervision, the mIoU of latent character masks with ground-truth character positions reaches 0.722, which is significantly higher than unsupervised (0.203) and strong supervision (0.617). More accurate character position estimation leads to better generation quality.
- **Joint Training is Crucial for Multi-font Scenarios**: Models with frozen encoders exhibit higher character error rates in polystylistic scenarios.
- **Human Study**: In a 50-pair pairwise comparison with UDiffText, DreamText received significantly more human preference in terms of diversity and quality.

## Highlights & Insights

- **Balanced Supervision as an Exquisite Design**: It neither lets the model explore freely without constraints (leading to complete failure) nor over-constrains character positions (hurting flexibility), but rather identifies a "guiding but not forcing" equilibrium. This scheme can be transferred to any conditional generation tasks requiring fine-grained spatial control without losing generation flexibility.
- **Clever EM Framework for Mixed Optimization**: Modeling text synthesis as a mixed optimization of discrete (character masks) and continuous (embedding parameters) variables, the EM-style alternate optimization provides an elegant solution. This problem formulation can inspire other generative tasks that require joint optimization of structures and parameters.
- **Extracting Character Masks from Cross-Attention**: Leveraging the pre-existing attention structures in diffusion models to obtain character spatial information eliminates the need for extra localization networks, making the design highly lightweight.

## Limitations & Future Work

- **Inability to Generate Texts in Multiple Regions Simultaneously**: The current version can only render text inside one masked region at a time. Multi-region text editing requires sequential inference.
- **Dependency on Character-Level Segmentation Annotations**: Balanced supervision during training requires character-level segmentation masks, which incurs high data collection costs.
- **Limited Capability for Long Text Generation**: As the number of characters increases, the difficulty of attention calibration escalates, potentially raising the generation error rate.
- **Privacy and Security Risks**: High-fidelity text synthesis technology could be manipulated for malicious purposes, such as forging signatures.
- **Future Directions**: Explore simultaneous multi-region editing and reduce the dependency on character-level annotations (e.g., substituting part of the balanced supervision with weak or self-supervision).

## Related Work & Insights

- **vs UDiffText**: UDiffText is also a diffusion-based method for scene text synthesis but lacks character-level guidance. DreamText significantly outperforms it in SeqAcc via balanced supervision (0.940 vs. 0.763).
- **vs AnyText**: AnyText supports mixed Chinese and English generation and features glyph-guided controls, but its overall SeqAcc is inferior to DreamText. DreamText's advantage lies in more precise control over character-level attention.
- **vs TextDiffuser**: TextDiffuser uses standard-font rendered character masks as conditional inputs, which acts as a "hard" guidance. DreamText's attention calibration performs a "soft" guidance, allowing for higher flexibility.
- **vs MOSTEL (GAN-based)**: While GAN-based methods have been thoroughly outperformed by diffusion methods in text synthesis tasks, MOSTEL's local editing paradigm still holds reference value.

## Rating

- Novelty: ⭐⭐⭐⭐ The balanced supervision and alternate optimization framework are exquisite designs, offering deep insights into the text synthesis problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ It includes quantitative comparisons, ablation studies, and a human study, representing a comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ The problem modeling is clear, and the formulation of mixed optimization is rigorous.
- Value: ⭐⭐⭐⭐ It establishes a new state-of-the-art and methodological benchmark for diffusion-based text synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Scene Language: Representing Scenes with Programs, Words, and Embeddings](the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)
- [\[CVPR 2025\] AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](../../ACL2025/llm_pretraining/data-constrained_synthesis_of_training_data_for_de-identification.md)
- [\[CVPR 2025\] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)

</div>

<!-- RELATED:END -->
