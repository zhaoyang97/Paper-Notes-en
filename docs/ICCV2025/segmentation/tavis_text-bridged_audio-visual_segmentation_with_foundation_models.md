---
title: >-
  [Paper Note] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models
description: >-
  [ICCV 2025][Segmentation][Audio-visual segmentation] TAViS is a text-bridged audio-visual segmentation framework that couples the cross-modal alignment capability of ImageBind with the precise segmentation capability of…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Audio-visual segmentation"
  - "foundation models"
  - "SAM2"
  - "ImageBind"
  - "text bridging"
  - "cross-modal alignment"
  - "zero-shot segmentation"
date: 2026-05-08
content_hash: 425f98b8249d1cbd
---

# TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models

**Conference**: ICCV 2025
**arXiv**: [2506.11436](https://arxiv.org/abs/2506.11436)
**Code**: Not released
**Area**: Image Segmentation
**Keywords**: Audio-visual segmentation, foundation models, SAM2, ImageBind, text bridging, cross-modal alignment, zero-shot segmentation

## TL;DR

TAViS is a text-bridged audio-visual segmentation framework that couples the cross-modal alignment capability of ImageBind with the precise segmentation capability of SAM2. By introducing a text-bridged hybrid prompting mechanism and alignment supervision strategy, TAViS achieves state-of-the-art performance across single-source, multi-source, semantic, and zero-shot segmentation scenarios.

## Background & Motivation

Audio-Visual Segmentation (AVS) aims to generate pixel-level segmentation maps of sounding objects in a scene based on audio signals. The core challenge lies in effectively aligning the audio and visual modalities.

Limitations of existing methods can be categorized into three types:

**Fusion-based methods** (AVSBench, CATR, AVSegFormer): These establish audio-visual relationships via cross-attention, but are constrained by small-scale training data and fail to adequately capture complex inter-modal relationships.

**Single-modality foundation model methods** (AV-SAM, SAMA-AVS, BAVS): These leverage visual foundation models such as SAM/Semantic-SAM or audio foundation models such as BEATs, but relying solely on single-modality priors cannot resolve the cross-modal alignment problem.

**Assembled foundation model combinations**: Although a few works combine SAM and ImageBind, the two models operate independently without interaction, constituting an "offline" combination.

**Core challenges**:
- **Feature space mismatch**: SAM2 and ImageBind reside in different feature spaces, making direct knowledge transfer difficult.
- **Insufficient supervision signal**: Supervising with segmentation loss alone implicitly suggests alignment needs but does not explicitly guide the model to learn meaningful audio-visual associations.
- **Intra-modal noise**: Audio contains non-semantic information (timbre, tone), while images contain diverse backgrounds and appearance information; both exhibit significant intra-class variation, making direct audio-visual alignment unreliable.

**Core insight**: Text, as a "bridge," can concisely express high-level prototype information, extracting and aligning shared semantic concepts between audio and visual modalities.

## Method

### Overall Architecture

TAViS integrates two foundation models:
- **SAM2**: Image encoder $F_E$ + memory attention module $F_M$ + mask decoder $F_D$, responsible for precise segmentation.
- **ImageBind**: Image encoder $E_I$ + text encoder $E_T$ + audio encoder $E_A$, responsible for cross-modal alignment.

All ImageBind parameters and the SAM2 image encoder are frozen; the SAM2 memory attention module and mask decoder are fine-tuned.

### Key Design 1: ImageBind-guided Query Decomposition (IBQD)

Prior SAM-based AVS methods treat the entire audio feature as a single prompt, but the mixed information from multiple sound sources conflicts with the target-level queries required by SAM2.

IBQD decomposes audio features into target-level queries while preserving ImageBind's aligned feature space.

Learnable queries $t_W \in \mathbb{R}^{N \times C}$ are introduced and decomposed from audio trunk features $f_a$ via multi-head cross-attention:

$$t_W = \text{Softmax}((W_q t_W)(W_k f_a^T))(W_v f_a) + t_W$$

Target-level biases are generated and added to the audio cls token $t_a$:

$$t'_a = t_a + \text{Linear}(t_W)$$

The updated $t'_a$ is then refined via another cross-attention operation. Crucially, the decomposed query takes the form of a cls token plus a bias, preserving $t_a$'s aligned feature space.

### Key Design 2: Text-bridged Hybrid Prompting

**Sparse prompts**: Dual audio-text prompts
- **Pseudo-text prompt** $p^t$: The decomposed audio query $t'_a$ and global cls token $t_a$ are concatenated via an MLP and fed into the ImageBind text encoder $E_T$ to generate high-level category prototype information.
- **Audio prompt** $p^a$: Another MLP concatenates $t_W$ and $t_a$ to retain audio-specific fine-grained features.
- Final sparse prompt: $p = \text{MLP}([p^a; p^t])$

Supervision for pseudo-text generation:

$$\mathcal{L}_{a2pt} = \text{MSE}(E_T(\text{MLP}(t'_a)), E_T(t^t))$$

**Dense prompt**: The image is passed through the ImageBind image encoder to obtain the cls token $t_v$, which is replicated and added to every spatial position of the SAM2 image embedding to provide aligned visual context.

### Key Design 3: Text-bridged Alignment Supervision (TbAS)

Text serves as an intermediate bridge to establish separate audio-text and image-text alignment relationships.

**Audio-text loss** $\mathcal{L}_{a2t}$: The audio query $t'_a$ is projected into the shared embedding space via ImageBind's projection layer, similarity is computed against text embeddings of all categories, and a cross-entropy loss is applied after Hungarian matching.

**Image-text loss** $\mathcal{L}_{i2t}$: The predicted segmentation mask is used to highlight foreground regions in the original image (with Gaussian-blurred background); visual tokens are extracted via the ImageBind image encoder and supervised with a cross-entropy loss against text embeddings.

Overall training loss:

$$\mathcal{L} = \mathcal{L}_{a2pt} + \mathcal{L}_{a2t} + \mathcal{L}_{i2t} + \sum_{i=0}^{N}\mathcal{L}_{sep} + \mathcal{L}_{binary}$$

### Why Not Align Audio and Visual Directly?

Experiments show that directly adding $\mathcal{L}_{a2i}$ degrades performance. Audio and visual tokens are derived from queries and segmentation masks, which lack accurate ground-truth supervision; aligning these noisy tokens introduces uncertainty. Text filters out noise from both modalities and distills them into concise prototype representations.

## Key Experimental Results

### Main Results

Comparison with 14 methods on three AVSBench subsets:

| Method | Backbone | Size | S4 $\mathcal{M_J}$ | S4 $\mathcal{M_F}$ | MS3 $\mathcal{M_J}$ | MS3 $\mathcal{M_F}$ | AVSS $\mathcal{M_J^I}$ |
|--------|----------|------|-----|-----|------|------|------|
| COMBO | PVT-v2 | 224 | 84.7 | **0.919** | 59.2 | 0.712 | 42.1 |
| SAMA-AVS | ViT-H | 1024 | 83.2 | 0.901 | 66.9 | 0.754 | - |
| **TAViS** | ViT-L | 224 | 84.8 | 0.912 | **68.2** | **0.759** | **44.2** |
| **TAViS** | ViT-L | 1024 | **87.0** | **0.926** | **71.2** | **0.796** | - |

TAViS@224 already surpasses SAMA-AVS at 1024 resolution (MS3: 68.2 vs. 66.9) while achieving lower model MACs (255G vs. 598G).

### Ablation Study

Core component ablation:

| Setting | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|---------|-----|------|
| w/o IBQD | 79.6 | 58.5 |
| w/o TbAS | 83.6 | 64.9 |
| w/o TbHP | 83.9 | 65.1 |
| TAViS | **84.8** | **68.2** |

Alignment supervision ablation:

| Setting | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|---------|-----|------|
| w/o $\mathcal{L}_{a2t}$ | 83.8 | 65.2 |
| w/o $\mathcal{L}_{i2t}$ | 84.0 | 64.4 |
| $\mathcal{L}_{a2i}$ (direct alignment) | 83.1 | 66.5 |
| $\mathcal{L}_{a2t} + \mathcal{L}_{i2t}$ (text-bridged) | **84.8** | **68.2** |

Key findings:
1. Removing IBQD causes a 9.7-point drop on MS3, demonstrating that audio query decomposition is critical for multi-source scenarios.
2. Both $\mathcal{L}_{a2t}$ and $\mathcal{L}_{i2t}$ are necessary; using $\mathcal{L}_{a2i}$ alone leads to a 1.7-point drop on S4.
3. Adding $\mathcal{L}_{a2i}$ on top of $\mathcal{L}_{a2t} + \mathcal{L}_{i2t}$ also degrades performance (68.2→66.5), confirming that text-bridged alignment is superior to direct alignment.
4. t-SNE visualizations show that text bridging produces more compact intra-class embeddings; removing it leads to substantially increased intra-class scatter.

### Zero-shot Generalization

| Method | Zero-shot $\mathcal{M_J^I}$ | Trainable Parameters |
|--------|---------------|-----------|
| OV-AVSS | 22.20 | 183.6M |
| **TAViS** | **28.21** | **54.9M** |

TAViS outperforms OV-AVSS in the zero-shot setting with significantly fewer parameters (54.9M vs. 183.6M), achieving a +6.01 mIoU gain and validating the importance of text bridging for generalization.

## Highlights & Insights

1. **Profound insight of text bridging**: Rather than directly aligning two noisy modalities, text acts as a denoising intermediary to extract shared semantics — a principle supported by both theoretical reasoning and experimental evidence.
2. **Elegant IBQD design**: Decomposed queries take the form of a cls token plus a bias, preserving ImageBind's alignment space without disruption.
3. **Unified architecture**: A single framework supports binary, semantic, and zero-shot segmentation modes.
4. **Efficiency advantage**: Performance at 224 resolution surpasses competing methods at 1024 resolution, with a 57% reduction in MACs.

## Limitations & Future Work

1. Due to computational constraints, the 1024-resolution model is not evaluated on AVSS (10 frames/video).
2. Category prediction in the zero-shot setting relies on a simple additive combination of audio-text and image-text similarities, lacking adaptive weighting.
3. The upper bound of zero-shot generalization is constrained by the scope of ImageBind's pre-trained knowledge.
4. Dependence on the AVSBench dataset scale remains (S4 contains only 4,932 videos).

## Related Work & Insights

- **AVS methods**: Fusion-based methods including AVSBench, CATR, and AVSegFormer; generation-based DiffusionAVS; SAM-based methods including AV-SAM and SAMA-AVS.
- **Joint multimodal representation**: Multimodal alignment frameworks including ImageBind, LanguageBind, and OmniBind.
- **Audio-text alignment**: Audio-text paired representation learning with WavCaps; CLIP extension to audio with AudioCLIP and WAV2CLIP.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The design philosophy of text bridging is original, with a complete and coherent loop from theory to experiment.
- **Technical Quality**: ⭐⭐⭐⭐⭐ — Ablations are thorough; every design choice is validated with comparative experiments.
- **Practicality**: ⭐⭐⭐⭐ — Unified framework with zero-shot capability, though simultaneous loading of two foundation models is required.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain is clear; motivation, challenges, solutions, and validation flow seamlessly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ICCV 2025\] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

</div>

<!-- RELATED:END -->
