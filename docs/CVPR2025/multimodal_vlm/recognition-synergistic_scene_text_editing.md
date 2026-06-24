---
title: >-
  [Paper Note] Recognition-Synergistic Scene Text Editing
description: >-
  [CVPR 2025][Multimodal VLM][Scene Text Editing] This work proposes RS-STE (Recognition-Synergistic Scene Text Editing), which unifies text recognition and text editing into a single multimodal parallel decoder. It leverages the recognition model's inherent ability to implicitly disentangle style and content to assist the editing process, and designs a cyclic self-supervised fine-tuning strategy to enable effective training on real-world data without paired annotations.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Scene Text Editing"
  - "Text Recognition Synergy"
  - "Cyclic Self-Supervised Fine-tuning"
  - "Style-Content Disentanglement"
  - "Multimodal Parallel Decoding"
date: 2026-05-08
content_hash: 4be8c2fac270fd3e
---

# Recognition-Synergistic Scene Text Editing

**Conference**: CVPR 2025  
**arXiv**: [2503.08387](https://arxiv.org/abs/2503.08387)  
**Code**: [https://github.com/ZhengyaoFang/RS-STE](https://github.com/ZhengyaoFang/RS-STE)  
**Area**: Multimodal VLM  
**Keywords**: Scene Text Editing, Text Recognition Synergy, Cyclic Self-Supervised Fine-tuning, Style-Content Disentanglement, Multimodal Parallel Decoding

## TL;DR

This work proposes RS-STE (Recognition-Synergistic Scene Text Editing), which unifies text recognition and text editing into a single multimodal parallel decoder. It leverages the recognition model's inherent ability to implicitly disentangle style and content to assist the editing process, and designs a cyclic self-supervised fine-tuning strategy to enable effective training on real-world data without paired annotations.

## Background & Motivation

Scene Text Editing (STE) requires modifying the textual content in an image while preserving the original style (such as background, font, and layout). Traditional methods (e.g., SRNet, MOSTEL, DARLING) follow a complex pipeline: "explicitly disentangling style and content $\rightarrow$ fusing target content $\rightarrow$ validating content consistency using an external recognition model." These methods suffer from two core limitations of prior work: (1) Explicitly separating style and content is inherently challenging, and imprecise separation leads to poor fusion results. (2) Joint optimization of multiple independent modules easily results in suboptimal outcomes.

The authors' key insight is that **text recognition models naturally possess the ability to implicitly disentangle style and content**. As shown in Figure 2, in the feature space encoded by a recognition model, images with the same textual content but different background styles tend to cluster together. Therefore, modeling recognition and editing synergistically can achieve both implicit style-content disentanglement and content consistency guarantees, thereby significantly simplifying the pipeline.

## Method

### Overall Architecture

RS-STE consists of three components: **(1) Input Tokenizer**: encodes the target text $T_B$ and the reference style image $I_A$ into embedding sequences, respectively; **(2) Multi-modal Parallel Decoder (MMPD)**: based on a Transformer decoder, it simultaneously predicts the original text content $T_A'$ (recognition) and the target image token features $\mathbf{D}_{I_B}^i$ (editing); **(3) Image Detokenizer**: uses a pretrained VAE decoder to reconstruct the target image from the features. The training consists of two stages: pre-training on synthetic data and cyclic self-supervised fine-tuning on real-world data.

### Key Designs

1. **Multi-modal Parallel Decoder (MMPD)**:
    - **Function**: Performs both text recognition and text editing within a unified framework.
    - **Mechanism**: Initializes text query embeddings $\mathbf{E}_{query}^t \in \mathbb{R}^{L \times C}$ and image query embeddings $\mathbf{E}_{query}^i \in \mathbb{R}^{N \times C}$. These are concatenated with the target text embeddings and style image embeddings before being fed into the Transformer decoder: $[\mathbf{D}_{NULL}^t, \mathbf{D}_{NULL}^i, \mathbf{D}_{T_A}^t, \mathbf{D}_{I_B}^i] = \mathcal{F}_{MMPD}([\mathbf{E}_{T_B}^t, \mathbf{E}_{I_A}^i, \mathbf{E}_{query}^t, \mathbf{E}_{query}^i])$. Here, $\mathbf{D}_{T_A}^t$ is utilized for text recognition, while $\mathbf{D}_{I_B}^i$ is used for image generation.
    - **Design Motivation**: The recognition branch forces the model to understand "what is currently written in the image" (implicitly learning style-content disentanglement), while the editing branch utilizes this understanding to generate the new content. The execution of both tasks shares feature representations, creating a natural synergy.

2. **Cyclic Self-Supervised Fine-tuning**:
    - **Function**: Enables effective training on real-world scene data without paired annotations.
    - **Mechanism**: Edits style image $I_A$ to generate $I_B'$, and then uses $I_B'$ as the new style image to edit back to the original text $T_A'$, obtaining the reconstructed image $I_A'$. Specifically: $(I_B', T_A') = \mathcal{F}_{RS-STE}(I_A, T_B)$, $(I_A', T_B') = \mathcal{F}_{RS-STE}(I_B', T_A')$. The difference between $I_A$ and $I_A'$ is used as the supervision signal.
    - **Design Motivation**: STE suffers from a lack of paired real-world data; training exclusively on synthetic data leads to severe domain shift. Cyclic editing allows the model to leverage unlabeled real-world data through self-supervised learning, while the recognition loss prevents the model from degenerating into an identity mapping.

3. **Input Tokenizer**:
    - **Function**: Encodes text and images uniformly into processable token sequences.
    - **Mechanism**: Text is encoded character-by-character using a character embedding matrix $\mathbf{E} \in \mathbb{R}^{(|\Sigma|+1) \times C}$. Images are processed using ViT-style patch embeddings, dividing $I_A$ into $N = HW/P^2$ patches using a $P \times P$ convolutional kernel.
    - **Design Motivation**: Unified tokenization allows text and images to be processed interactively within the same Transformer decoder.

### Loss & Training

**Pre-training Phase** (synthetic paired data):
$$\mathcal{L}^{pre} = \lambda_1 \mathcal{L}_{rec}^{pre} + \lambda_2 \mathcal{L}_{mse}^{pre} + \lambda_3 \mathcal{L}_{per}^{pre}$$
- $\mathcal{L}_{rec}$: Cross-entropy recognition loss, $\lambda_1=1$
- $\mathcal{L}_{mse}$: Pixel-level MSE loss, $\lambda_2=10$
- $\mathcal{L}_{per}$: VGG-16 perceptual loss (from layers relu1_2 to relu4_3), $\lambda_3=1$

**Cyclic Fine-tuning Phase** (unpaired real data):
$$\mathcal{L}^{cyc} = \lambda_4 \mathcal{L}_{mse}^{cyc} + \lambda_5 \mathcal{L}_{per}^{cyc} + \lambda_6 \mathcal{L}_{rec}^{cyc\text{-}1} + \lambda_7 \mathcal{L}_{rec}^{cyc\text{-}2}$$
- Recognition losses of the two editing passes ($\lambda_6=\lambda_7=50$) prevent identity mapping degeneration.
- Pixel loss ($\lambda_4=10$) and perceptual loss ($\lambda_5=1$) guarantee style consistency.
- Fine-tuning is conducted on real-world data from MLT-2017 or Union14M-L for only 1k iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | STEEM | MOSTEL | TextCtrl |
|--------|------|--------|-------|--------|----------|
| Tamper-Syn2k | MSE↓ | **0.0076** | 0.0122 | 0.0135 | 0.0130 |
| Tamper-Syn2k | PSNR↑ | **22.54** | 20.83 | 20.27 | 20.79 |
| Tamper-Syn2k | RecAcc↑ | **86.12** | 78.80 | 66.54 | 74.17 |
| Tamper-Scene | RecAcc↑ | **91.80** | - | 37.69 | 84.67 |
| ScenePair | MSE↓ | **0.0267** | - | 0.0519 | 0.0447 |
| ScenePair | RecAcc↑ | **91.80** | - | 37.69 | 84.67 |
| STR Bench Avg | RecAcc↑ | **82.9** | - | 36.8 | 66.2 |

| Downstream Recognition Augmentation | Model | No Aug Avg | +MOSTEL Avg | +Ours Avg |
|------------|------|----------|-----------|------------|
| Union14M-Bench | ABINet | 67.3 | 68.0(+0.7) | **69.5(+2.2)** |
| Union14M-Bench | MAERec-S | 78.6 | 78.9(+0.3) | **81.1(+2.5)** |

### Ablation Study

| Configuration | MSE↓ | PSNR↑ | SSIM↑ | FID↓ | Description |
|------|------|-------|-------|------|------|
| Ours (Full) | 0.0076 | 22.54 | 72.90 | 30.29 | - |
| w/o $\mathcal{L}_{rec}^{pre}$ | 0.0082 | 22.26 | 69.70 | 33.96 | Removing recognition loss decreases SSIM by 3.2 |
| w/ External Recognition Model | 0.0079 | 22.44 | 70.71 | 31.73 | Inferior to intrinsic recognition synergy |
| w/o $\mathcal{L}^{cyc}$ | - | - | - | - | RecAcc drops from 86.12% to 69.01% |
| w/o $\mathcal{L}_{rec}^{cyc}$ | - | - | - | - | Model degenerates into identity mapping (RecAcc=0%) |

### Key Findings

- **Intrinsic vs. External Recognition**: The jointly trained recognition branch (SSIM=72.90) performs better than supervision from an external pre-trained recognition model (SSIM=70.71). This is because intrinsic recognition achieves both style-content disentanglement and content consistency simultaneously.
- **Recognition Loss in Cyclic Fine-tuning is Key to Preventing Degeneration**: Discarding $\mathcal{L}_{rec}^{cyc-1}$ or $\mathcal{L}_{rec}^{cyc-2}$ causes the model to degenerate into an identity mapping (RecAcc=0%), as the model learns to simply copy the input image.
- **Performance on Standard STR Benchmarks**: After fine-tuning on MLT2017, RS-STE achieves a recognition accuracy of 81.8% on standard STR benchmarks. With Union14M-L, it reaches 82.9%, which is close to the upper bound of the original images (91.8%).
- **Downstream Data Augmentation Utility**: Hard samples generated by RS-STE as a data augmentation tool improve MAERec-S by 2.5% on Union14M-Benchmark, significantly outperforming MOSTEL's improvement of 0.3%.

## Highlights & Insights

- **Highly Convincing Core Insight**: Recognition models naturally and implicitly disentangle style and content—images with the same content but different styles cluster together in the feature space (Figure 2). This quantitative evidence strongly supports the design of the method.
- **Ultra-Minimalist Architecture**: A single Transformer decoder performs both recognition and editing simultaneously, without requiring foreground/background separation modules or external recognition validators. The parameter size is only 54.4M (compared to TextCtrl's 1216M).
- **Cyclic Fine-tuning Strategy**: Leverages the reversibility of editing to construct self-supervised signals, elegantly addressing the long-standing issue in the STE field regarding the lack of paired real-world data.
- **Downstream Application Value**: Edited results can be directly utilized as hard samples to augment text recognition models, establishing a positive feedback loop of "editing $\rightarrow$ recognition".

## Limitations & Future Work

- The image resolution is fixed at $32 \times 128$, which prevents the handling of large-scale or non-horizontal text.
- The Transformer decoder based on minGPT (22.5M parameters) has limited capacity, potentially bounding the editing quality in complex scenes.
- The cyclic fine-tuning strategy relies on the assumption that "editing twice can restore the original image"; when the first edit is of poor quality, it may cause training instability.
- Multi-line text editing and arbitrary-shaped text are not supported.

## Related Work & Insights

- **Core Distinction from MOSTEL** (explicit foreground/background separation + style augmentation): RS-STE implicitly achieves disentanglement using a recognition branch, avoiding the need for explicit separation.
- The **cyclic self-supervised idea** shares similarities with dual learning in CycleGAN, but incorporates key modifications tailored for the STE task: the addition of recognition losses to prevent degeneration.
- **Synergistic task modeling**: Treating the "byproduct" of recognition models (implicit style-content disentanglement) as the core capability for text editing serves as an outstanding case study of leveraging task synergies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Creative core observation of recognition-editing synergy, well-designed cyclic fine-tuning strategy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across synthetic, real-world, standard STR benchmarks, and downstream augmentation, with highly detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and rich experiments, though the method description is slightly wordy.
- **Value**: ⭐⭐⭐⭐ SOTA in the STE field, with doubled practical value thanks to its applicability to downstream data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](../../ICCV2025/multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[CVPR 2025\] MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures](markushgrapher_joint_visual_and_textual_recognition_of_markush_structures.md)
- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](../../CVPR2026/multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)

</div>

<!-- RELATED:END -->
