---
title: >-
  [Paper Note] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction
description: >-
  [ICCV 2025][Medical Imaging][fMRI-to-Video] This paper proposes NEURONS, a framework inspired by the hierarchical structure of the human visual cortex that decouples fMRI-to-video reconstruction into four sub-tasks (key…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "fMRI-to-Video"
  - "Brain Decoding"
  - "Visual Cortex"
  - "diffusion model"
  - "Neuroscience"
date: 2026-05-08
content_hash: cd196783ccd4527b
---

# NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2503.11167](https://arxiv.org/abs/2503.11167)  
**Code**: [https://github.com/xmed-lab/NEURONS](https://github.com/xmed-lab/NEURONS)  
**Area**: Medical Imaging
**Keywords**: fMRI-to-Video, Brain Decoding, Visual Cortex, diffusion model, Neuroscience

## TL;DR

This paper proposes NEURONS, a framework inspired by the hierarchical structure of the human visual cortex that decouples fMRI-to-video reconstruction into four sub-tasks (key object segmentation, concept recognition, scene description, and blurry video reconstruction), emulating the functional specialization of cortical regions V1/V2/V4/ITC. NEURONS substantially outperforms state-of-the-art methods in video consistency (+26.6%) and semantic accuracy (+19.1%).

## Background & Motivation

Decoding visual stimuli from brain activity is a key pathway toward understanding the human brain. While fMRI-to-image reconstruction has achieved considerable success (leveraging CLIP and Stable Diffusion), fMRI-to-video reconstruction remains highly challenging:

**Spatiotemporal complexity**: Video requires capturing object motion, scene transitions, and temporal consistency—far beyond what static image reconstruction demands.

**Implicit alignment issues in existing methods**:
   - MinD-Video: conditions a diffusion model on visual fMRI features but lacks low-level visual detail.
   - NeuroClips: introduces semantic and perceptual reconstructors but relies primarily on implicit alignment in the CLIP latent space.
   - The CLIP latent space is highly semantic, whereas fMRI voxels encode information at multiple granularities, making implicit alignment fragile.

Core insight: The human visual cortex has well-defined functional subdivisions (V1/V2 for edges/shapes; V4/ITC for object/face recognition, etc.). Directly emulating this hierarchy by decomposing learning into multiple explicit sub-tasks enables more robust decoding of visual information at different granularities.

## Method

### Overall Architecture

NEURONS consists of three components:
1. **Brain Model**: Maps fMRI representations to motion embeddings (pretraining stage).
2. **Decoupler**: Progressively learns four explicit sub-tasks from the motion embeddings.
3. **Aggregated Video Reconstruction**: At inference, integrates all sub-task outputs to guide a text-to-video (T2V) diffusion model.

### Key Designs

1. **Decoupled Task Construction (automatic label generation via off-the-shelf models)**:

    - **Scene description generation**: Qwen2.5-VL-72B generates descriptive captions for each frame.
    - **Concept name generation**: Qwen identifies the primary objects in each frame, categorized into 51 concept classes summarized from WordNet.
    - **Concept segmentation masks**: Grounded-SAM generates per-frame binary masks using object names as text prompts.
    - **Key object discovery**: A multi-criteria method based on motion dynamics (inter-frame displacement), object size, and semantic importance. Background categories are excluded; semantically important categories such as humans and animals are prioritized.

   Design motivation: Processing proceeds from coarse to fine—starting from segmentation (least semantic information) and progressively incorporating concepts, descriptions, and motion, mirroring the hierarchical processing of the visual cortex.

2. **Brain Model**:

    - Built upon the pretrained MindEyeV2 backbone.
    - Introduces a motion projection $\mathcal{P}_{vid}(\cdot)$ that maps image embeddings $e^i \in \mathbb{R}^{B \times N \times C}$ to the spatiotemporal space $e^v \in \mathbb{R}^{B \times F \times N \times C}$ (accounting for inter-frame temporal relationships, unlike NeuroClips).
    - Aligns $e^v$ with the target video embeddings from the CLIP visual encoder using a BiMixCo contrastive loss.
    - Additionally projects $e^v$ into text embeddings $e^t$ for contrastive learning against CLIP text embeddings.

   Design motivation: The pretraining stage produces suitable visual and text embeddings that serve as the foundation for subsequent decoupled sub-tasks.

3. **Decoupler — Four Sub-tasks**:

    - **Key Object Segmentation**: A text-driven VAE video decoder takes $e^v$ and the CLIP text embeddings of key object class names as input, activates corresponding patches via cross-attention, and produces binary masks through upsampling and a segmentation head. Loss: BCE $\mathcal{L}_{seg}$.
    - **Concept Recognition**: A multi-label classifier $\mathcal{D}_{cls}$ identifies concepts in frames, using the frame-dimension mean of visual embeddings as input. Loss: cross-entropy $\mathcal{L}_{cls}$.
    - **Scene Description**: A GPT-2 text decoder is fine-tuned with $e^t$ as a prefix to generate captions. Loss: prefix language modeling $\mathcal{L}_{txt}$.
    - **Blurry Video Reconstruction**: Reuses the segmentation VAE decoder with a reconstruction head to generate a blurry video $y_c^{rec}$, aligned to the latent embeddings of the SD VAE. Loss: MAE $\mathcal{L}_{rec}$.

4. **Progressive Learning Strategy**:
   A sinusoidal weight schedule $w = 1 + 9 \cdot |\sin(\frac{C}{T} \cdot \pi)|$ is applied, with the "ascending phases" of the four loss weights staggered to ensure smooth transitions: segmentation emphasis → concept recognition → scene description → video reconstruction.

### Loss & Training

$$\mathcal{L}_{total} = w_1 \mathcal{L}_{seg} + w_2 \mathcal{L}_{cls} + w_3 \mathcal{L}_{txt} + w_4 \mathcal{L}_{rec}$$

where $w_1$–$w_4$ vary in a staggered sinusoidal schedule (range [1, 10]), ensuring progressive learning from simple to complex tasks.

**Inference**: Key object masks are rescaled to [0.5, 1] and multiplied onto the control image and blurry video to emphasize key objects, which are then fed into the T2V diffusion model (AnimateDiff) to generate the final video.

**Dataset**: The cc2017 open-source fMRI-video dataset, 3T MRI, 2 s temporal resolution. 8,640 training samples and 1,200 test samples.

## Key Experimental Results

### Main Results (Tables)

**Video-level quantitative comparison:**

| Method | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|------|---------|----------|------------|
| Wen | - | 0.166 | - |
| Wang | 0.773 | - | 0.402 |
| Kupershmidt | 0.771 | - | 0.386 |
| MinD-Video | 0.839 | 0.197 | 0.408 |
| MindAnimator | 0.830 | - | 0.428 |
| NeuroClips | 0.834 | 0.220 | 0.738 |
| **NEURONS** | **0.863** | **0.262** | **0.934** |

**Per-subject results:**

| Subject | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|------|---------|----------|------------|
| Subject 1 | 0.862 | 0.254 | 0.932 |
| Subject 2 | 0.860 | 0.252 | 0.933 |
| Subject 3 | 0.868 | 0.278 | 0.937 |

### Ablation Study (Tables)

**Key component ablation (Subject 1):**

| Brain Model | $\mathcal{L}_{seg}$ | $\mathcal{L}_{cls}$ | $\mathcal{L}_{txt}$ | $\mathcal{L}_{rec}$ | PL | AVR | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|:-----------:|:---:|:---:|:---:|:---:|:--:|:---:|---------|----------|------------|
| ✓ |  |  |  | ✓ |  |  | 0.814 | 0.164 | 0.894 |
| ✓ | ✓ |  |  | ✓ |  |  | 0.834 | 0.225 | 0.926 |
| ✓ | ✓ | ✓ |  | ✓ |  |  | 0.836 | 0.234 | 0.911 |
| ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 0.847 | 0.213 | 0.923 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 0.856 | 0.235 | 0.937 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **0.862** | **0.254** | **0.932** |

**Concept recognition accuracy (Top-5 categories):**

| Rank | Category | Accuracy |
|------|------|--------|
| 1 | human | 0.735 |
| 2 | food | 0.600 |
| 3 | animal | 0.450 |
| 4 | water body | 0.310 |
| 5 | fish | 0.292 |

**Scene description comparison:**

| Method | BLEU-1 | BLEU-4 | CIDEr | Verb Acc |
|------|--------|--------|-------|----------|
| NeuroClips | 0.227 | 0.022 | 0.156 | 0.116 |
| **NEURONS** | **0.238** | **0.036** | **0.239** | **0.243** |

### Key Findings

1. **CLIP-pcc improves by 26.6%** (0.738 → 0.934): Spatiotemporal consistency improves substantially, demonstrating that decoupled learning effectively captures motion information.
2. **50-way accuracy improves by 19.1%** (0.220 → 0.262): Semantic precision is significantly enhanced.
3. **Segmentation task contributes the most**: Adding $\mathcal{L}_{seg}$ raises 50-way from 0.164 to 0.225 (+37%) and CLIP-pcc from 0.894 to 0.926.
4. **Progressive learning is essential**: Adding PL improves 2-way from 0.847 to 0.856 and 50-way from 0.213 to 0.235.
5. **Verb accuracy doubles** (0.116 → 0.243): Generating descriptions directly from fMRI is more reliable than captioning generated keyframes.
6. **Functional mapping validates the design**: Segmentation embeddings correspond to V1/V2/MT; concept recognition to V4/EBA/LOC; scene description to PPA/FFA—a perfect match with the intended design.

## Highlights & Insights

- **Neuroscience inspiration → engineering design → neuroscience validation**: Sub-tasks are designed from the hierarchical structure of the visual cortex, and the resulting brain functional mapping validates the alignment, forming an elegant closed loop.
- **Explicit decoupling vs. implicit alignment**: Explicitly separating multi-granularity information into distinct sub-tasks is more robust than implicit alignment in the CLIP latent space.
- **Progressive sinusoidal weight scheduling**: A simple and elegant training strategy that ensures smooth transitions from simple to complex tasks.
- **VLM-generated training labels**: Qwen2.5-VL-72B and Grounded-SAM automatically construct labels for all sub-tasks.
- **Key object mask enhancement**: At inference, multiplying masks onto conditioning signals to emphasize key objects is a straightforward yet effective technique.

## Limitations & Future Work

1. The Dice score for key object segmentation is relatively low (35.63%), primarily due to the prevalence of unseen categories in the test set.
2. The fMRI temporal resolution of 2 s makes it difficult to capture rapid visual changes.
3. The method depends on the capabilities of the pretrained MindEyeV2 backbone and the AnimateDiff T2V model.
4. The limited number of training samples per subject (8,640) means cross-subject generalization remains a challenge.
5. The generated blurry video has low resolution ($H/8 \times W/8$), and detail recovery relies on the T2V model.
6. The 51 concept categories may be insufficient to cover all visual content.

## Related Work & Insights

- **fMRI-to-image methods**: Mind-Reader, MindEye, BrainDiffuser, and others establish the foundation for static reconstruction.
- **NeuroClips**: The direct predecessor, proposing dual semantic and perceptual reconstructors, but with fragile implicit alignment.
- **Visual cortex functional parcellation**: Neuroscientific findings on the V1/V2 (edges/shapes) → V4/ITC (object recognition) → PPA/FFA (scenes/faces) hierarchy directly inspired the sub-task design.
- The work holds potential value for brain-computer interfaces and clinical applications, such as assisting patients with aphasia in expressing visual experiences.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first work to explicitly map the hierarchical structure of the visual cortex onto decoupled sub-tasks; the design concept is highly inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-angle validation through visual reconstruction, sub-task evaluation, and brain functional mapping; however, only a single dataset is used.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative logic—from neuroscience motivation to engineering design to validation—is excellent.
- **Value**: ⭐⭐⭐⭐ Achieves a significant breakthrough in fMRI-to-video reconstruction, with brain mapping results that are also valuable for neuroscience research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](../../NeurIPS2025/medical_imaging/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)

</div>

<!-- RELATED:END -->
