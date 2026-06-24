---
title: >-
  [Paper Note] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows
description: >-
  [CVPR 2025][Image Generation][Multimodal Generation] OmniFlow extends the rectified flow framework of Stable Diffusion 3 to joint multimodal (text + image + audio) generation scenarios. Through a modular Omni-Transformer architecture and a novel multimodal guidance mechanism, it achieves superior generation quality compared to previous any-to-any models like CoDi and UniDiffuser without requiring training from scratch.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multimodal Generation"
  - "Rectified Flow"
  - "Any-to-Any"
  - "MMDiT"
  - "Multimodal Guidance"
date: 2026-05-08
content_hash: 2d923ec57a61048a
---

# OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows

**Conference**: CVPR 2025  
**arXiv**: [2412.01169](https://arxiv.org/abs/2412.01169)  
**Code**: [https://github.com/jacklishufan/OmniFlows](https://github.com/jacklishufan/OmniFlows)  
**Area**: Diffusion Models / Multimodal VLM  
**Keywords**: Multimodal Generation, Rectified Flow, Any-to-Any, MMDiT, Multimodal Guidance

## TL;DR
OmniFlow extends the rectified flow framework of Stable Diffusion 3 to joint multimodal (text + image + audio) generation scenarios. Through a modular Omni-Transformer architecture and a novel multimodal guidance mechanism, it achieves superior generation quality compared to previous any-to-any models like CoDi and UniDiffuser without requiring training from scratch.

## Background & Motivation

**Background**: Current generative models have achieved outstanding results on single tasks (e.g., text-to-image, text-to-audio). However, each of these models can only perform one task and incurs massive training costs. To achieve any-to-any generation, prior works like CoDi achieve this by concatenating multiple modality-specific encoders and decoders, but this design severely limits cross-modal information interaction.

**Limitations of Prior Work**: When executing tasks like audio+text-to-image (A+T→I), CoDi simply computes a weighted average of audio and text embeddings and feeds it into the image generator. The problem with this approach is that many different sets of multimodal embeddings might average out to the same vector, leading to severe information loss. While unified models like UniDiffuser and Chameleon can better integrate cross-modal information, they require training from scratch, which consumes immense computational resources.

**Key Challenge**: In any-to-any generation, there is an inherent contradiction between "modularity to reuse pretrained weights" and "early fusion for sufficient information interaction." CoDi chooses modularity at the expense of interaction, while UniDiffuser chooses fusion but requires training from scratch.

**Goal**: (1) Build a unified multimodal generation framework allowing features of different modalities to interact directly at each layer; (2) Maintain a modular design to reuse pretrained model weights, saving training costs; (3) Provide a flexible multimodal guidance mechanism to control alignment strength across different modalities.

**Key Insight**: The MMDiT architecture of SD3 is intrinsically a "dual-stream + joint attention" design—the text stream and the image stream possess independent parameters but interact through joint attention. The authors observe that this design naturally possesses modular advantages and can be seamlessly extended to more modalities.

**Core Idea**: Extend the dual-stream MMDiT of SD3 to a multi-stream Omni-Transformer, where each modality has independent projection and feed-forward layer parameters (which can be independently pretrained/initialized), but deep cross-modal interaction is achieved through joint attention.

## Method

### Overall Architecture
The input to OmniFlow consists of multimodal data (text, image, audio), which is first encoded into the latent space through modality-specific VAEs, and then injected with random noise according to the forward process of multimodal rectified flows. Three sinusoidal embeddings separately encode the time steps $t_1, t_2, t_3$ of each modality, which are fused via an MLP into a unified time step embedding $y$. Finally, the noised latent representations and the time step embedding are processed by $N$ Omni-Transformer blocks, and the final hidden state of each modality is passed through a linear layer to output the predicted velocity field $v$.

### Key Designs

1. **Multi-Modal Rectified Flow**:

    - **Function**: Extends rectified flow from single-modality to multimodal joint distribution modeling.
    - **Mechanism**: Considering the joint distribution of multimodal data $(x_1^0, x_2^0, ..., x_n^0) \sim \pi_{data}$ and corresponding independent Gaussian noise $(x_1^1, x_2^1, ..., x_n^1)$, the forward process for each modality is defined as $x_i^{t_i} = (1-t_i)x_i^0 + t_i x_i^1$. Different any-to-any tasks are encoded via paths in the $[0,1]^n$ space. For example, text-to-image (T→I) corresponds to the path from $(1,0,1)$ to $(0,0,1)$. Key insight: For data with missing modalities (e.g., text-image pairs only), the time step of the missing modality is set to 1 (i.e., pure noise).
    - **Design Motivation**: This path encoding allows the same network to learn all any-to-any tasks with a unified training objective, and degenerates to the standard single-modality rectified flow formulation when only two modalities are present.

2. **Multi-Modal Guidance**:

    - **Function**: Allows users to flexibly control the alignment strength between different modalities in the generated output.
    - **Mechanism**: Define $\delta_{ij} = v_\theta(x_i^t, x_j^0) - v_\theta(x_i^t)$ as the influence of input modality $j$ on output modality $i$. The final guidance formula is $\hat{v}_\theta = v_\theta + \sum_{j \neq i}(\alpha_{ij} - 1)\delta_{ij}$, where $\alpha_{ij}$ can be adjusted independently. For example, in the audio+image-to-text (A+I→T) task, adjusting $\alpha_{im}$ and $\alpha_{au}$ can control whether the generated text leans toward the image description or the audio description.
    - **Design Motivation**: Compared to the simple embedding averaging in CoDi, this fine-grained guidance mechanism allows users to precisely control the relationship strength between each pair of input-output modalities.

3. **Modular Omni-Transformer Architecture**:

    - **Function**: Achieves deep cross-modal interaction while maintaining modularity (support for independent pretraining).
    - **Mechanism**: Within each Omni-Transformer block, inputs of each modality pass through independent projection layers to obtain their respective $q, k, v$, which are then concatenated into $Q = Concat(q_1, q_2, q_3)$, $K = Concat(k_1, k_2, k_3)$, and $V = Concat(v_1, v_2, v_3)$ to execute joint attention. The outputs of joint attention then pass through modality-independent FFNs. Crucially, the joint attention mechanism itself does not contain any trainable parameters; only the projection layers and FFNs are modality-specific and independent.
    - **Design Motivation**: This "parameter independence + attention interaction" design allows parameters of each modality to be initialized from single-task expert models (e.g., initializing image and text modules from SD3), significantly reducing training costs.

### Loss & Training
Training proceeds in three stages: (1) Initialize text and image modules with SD3 (Model 1); (2) Train an independent audio module using text-audio pairs (Model 2, with the text branch initialized from SD3); (3) Merge Model 1 and Model 2 (averaging the text branch weights) to obtain Model 3, and jointly fine-tune it on all any-to-any tasks for 150k steps. Training utilizes 8 A6000 GPUs, the AdamW optimizer, and a learning rate of 5e-6. The text encoder uses Flan-T5-L instead of SD3's T5-XXL (4.7B→783M), coupled with QFormer and TinyLlama-1.1B decoders to construct the text VAE.

## Key Experimental Results

### Main Results

| Dataset/Benchmark | Metric | OmniFlow | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| MSCOCO-30K | FID↓ | 13.40 | 9.71 (UniDiff) | - |
| MSCOCO-30K | CLIP↑ | 31.54 | 30.93 (UniDiff) | +0.61 |
| GenEval | Score↑ | 0.62 | 0.43 (UniDiff) | +0.19 |
| AudioCaps | FAD↓ | 1.75 | 1.80 (CoDi) | -0.05 |
| AudioCaps | CLAP↑ | 0.183 | 0.053 (CoDi) | +0.130 |

### Ablation Study (Audio/Text Generation Recipe Selection)

| Configuration | FAD↓ (Audio) | CLAP↑ (Text) | Description |
|------|---------|---------|------|
| rf/lognorm | 1.79 | .254 | Best configuration, adopted in final model |
| rf/uniform | 1.82 | .227 | Uniform time step sampling |
| v/linear | 1.86 | .126 | v-prediction + linear schedule |
| eps/linear | 2.08 | .141 | $\epsilon$-prediction + linear schedule |
| SEDD (discrete) | - | .180 | Discrete text diffusion, inferior to continuous options |
| MDLM (discrete) | - | .163 | Discrete text diffusion, inferior to continuous options|

### Key Findings
- rf/lognorm is comprehensively optimal in audio and text generation, proving that the optimal configuration for image generation in SD3 is equally effective on other modalities.
- Discrete text diffusion (SEDD, MDLM) does not outperform continuous schemes in multimodal settings, which differs from conclusions in single-task text generation.
- Time step shifting (shift=3.0) significantly improves both audio and text generation, indicating that SD3's resolution-aware adaptive scheduling is also applicable to other modalities.
- Joint training provides mutual benefits: for instance, T2I data helps improve the Aesthetic score of audio-conditioned image generation (+1.22).

## Highlights & Insights
- **Modularity $\neq$ Weak Interaction**: OmniFlow proves that "independent parameters but shared attention" can simultaneously yield the training efficiency of modularity and the generation quality of early fusion. This design paradigm is worth popularizing in other multimodal systems.
- **Elegance of Path Encoding**: Using paths in $[0,1]^n$ space to uniquely encode all any-to-any tasks is mathematically concise and practically effective. It achieves performance competitive with the 7B Transfusion model using only 30M training images.
- **Multimodal Guidance**: The $\alpha_{ij}$ mechanism allows users fine-grained control over cross-modal alignment. Not only is it effective numerically, but qualitative experiments also reveal that it can capture subtle stylistic differences between audio descriptions and image descriptions in training data.

## Limitations & Future Work
- Performance on text generation tasks lags significantly behind dedicated models (e.g., BLIP-2), with a CIDEr of only 47.3 vs. BLIP-2's 145.8. This might be due to the mixed styles of text in the training data.
- Only text, image, and audio modalities are handled; video, 3D, and other modalities are not yet covered.
- The quality of synthetic triplets (text-image-audio triplets) may limit the performance of joint generation—theoretical analysis shows that triplet data is crucial for correctly modeling the joint distribution.
- The amount of training data (30M images) is still relatively small compared to large-scale models. Scaling up the data could further improve performance.

## Related Work & Insights
- **vs CoDi**: CoDi achieves any-to-any generation by concatenating independent encoders and decoders, with cross-modal interaction relying on embedding weighted averaging, causing severe information loss. OmniFlow achieves layer-wise interaction through joint attention, significantly outperforming CoDi on T2I (CLIP +0.85) and T2A (CLAP +0.13).
- **vs UniDiffuser**: UniDiffuser requires training from scratch and only supports text+image. OmniFlow's modular design allows reusing SD3 weights, achieving competitive performance with 1/60th of the training data.
- **vs Transfusion/Chameleon**: These 7B models require 3.5B training images. OmniFlow, with 3.4B parameters, achieves comparable performance on GenEval (0.62 vs. 0.63) using only 30M images.

## Rating
- Novelty: ⭐⭐⭐⭐ The path encoding of multimodal rectified flows and the modular design have highlights, but the overall formulation represents a natural extension of the SD3 architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple tasks such as T2I, T2A, A2T, I2T, with systematic explorations of multiple ablation studies and recipes.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with standard mathematical formulas, though some experimental setups are scattered in the appendix.
- Value: ⭐⭐⭐⭐ Provides an efficient, open-source solution for multimodal joint generation, and the recipe exploration is valuable for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](symbolic_representation_for_any-to-any_generative_tasks.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](../../ECCV2024/image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)

</div>

<!-- RELATED:END -->
