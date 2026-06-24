---
title: >-
  [Paper Note] CAMILA: Context-Aware Masking for Image Editing with Language Alignment
description: >-
  [NeurIPS 2025][Image Generation][context-aware editing] This paper proposes CAMILA, a context-aware image editing method that leverages a multimodal large language model (MLLM) to automatically determine whether a given instruction is executable on the input image. It introduces dedicated [MASK] and [NEG] tokens to distinguish editable regions from regions that should remain unchanged, enabling precise multi-instruction editing while effectively filtering out non-executable i…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "context-aware editing"
  - "multi-instruction image editing"
  - "instruction executability assessment"
  - "MLLM"
  - "diffusion model editing"
date: 2026-05-08
content_hash: 94bd4b6ed7b1c7d4
---

# CAMILA: Context-Aware Masking for Image Editing with Language Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2509.19731](https://arxiv.org/abs/2509.19731)  
**Code**: None  
**Area**: Image Editing
**Keywords**: context-aware editing, multi-instruction image editing, instruction executability assessment, MLLM, diffusion model editing

## TL;DR

This paper proposes CAMILA, a context-aware image editing method that leverages a multimodal large language model (MLLM) to automatically determine whether a given instruction is executable on the input image. It introduces dedicated [MASK] and [NEG] tokens to distinguish editable regions from regions that should remain unchanged, enabling precise multi-instruction editing while effectively filtering out non-executable instructions.

## Background & Motivation

Text-guided image editing has become a critical tool for content creation. However, existing models (e.g., InstructPix2Pix, MGIE, SmartEdit) suffer from a fundamental flaw: they attempt to execute all user instructions regardless of whether those instructions are feasible or contradictory given the current image. For instance, when a user requests "remove the pancakes from the plate" but no pancakes exist in the image, these models still attempt to perform the operation, producing unreasonable outputs.

The limitations of prior work can be summarized at three levels: (1) simple text encoders such as CLIP fail to capture fine-grained semantics of complex multi-step instructions; (2) cross-attention-based region localization (e.g., FoI) frequently produces misaligned attention maps, especially when edits involve spatial relationships or regions not directly associated with key tokens; (3) relying on GPT for instruction parsing or reformulation increases pipeline complexity and propagates intermediate errors. CAMILA is the first to explicitly incorporate instruction executability assessment into the editing pipeline, addressing this previously overlooked gap.

## Method

### Overall Architecture

CAMILA consists of three core modules in series: (1) an MLLM that jointly processes the image and instructions to produce a sequence of [MASK] or [NEG] tokens; (2) a Token Broadcaster that aligns these tokens with the text embeddings of the diffusion model; and (3) a Token Decoder that decodes [MASK] tokens into binary masks. The resulting masks modulate the cross-attention layers of the diffusion model to achieve precise region-specific editing.

### Key Designs

1. **[MASK]/[NEG] Dual-Token Mechanism**: The MLLM (LLaVA-7B) jointly processes the input image $x_{\text{img}}$ and text instruction $x_{\text{txt}}$, producing a token sequence $\mathcal{O} = \{\mathcal{O}_1, ..., \mathcal{O}_n\}$, where each token is classified as either [MASK] (marking regions to be edited) or [NEG] (marking regions to be preserved or instructions that are non-executable). This is the core of context-awareness—the model not only understands the instruction content but also assesses its feasibility given the input image. [NEG] tokens are directly mapped to all-black masks, fully suppressing edits in the corresponding regions.

2. **Token Broadcaster for Cross-Space Alignment**: The MLLM output tokens and the diffusion model's text embeddings $c_T$ reside in different latent spaces and must be aligned. Trainable projection matrices $W_O$ and $W_T$ map both into a shared space, and a cosine similarity matrix $S_{i,j}$ is computed. For each text embedding $j$, the best-matching MLLM token index is identified as $\alpha_j = \arg\max_i \text{softmax}(S_{i,j})$. This ensures precise correspondence between the editing masks and the text conditioning of the diffusion model, serving as the key bridge for transferring MLLM understanding to the diffusion editing process.

3. **Cross-Attention Mask Modulation**: The binary masks for all tokens are concatenated into a unified mask $\mathcal{M}$, which modulates the cross-attention layers at 16×16 resolution in the U-Net. Editing regions use the full text-plus-image conditional score $\mathcal{X}$, while non-editing regions use the image-only conditional score $\mathcal{Y}$:
$$\mathcal{A}' = \text{softmax}(\mathcal{X} \odot \mathcal{M} + \mathcal{Y} \odot (1 - \mathcal{M}) / \sqrt{d})$$
This ensures that non-target regions retain their original image features without modification.

### Loss & Training

The main training stage jointly optimizes four loss terms:

$$\mathcal{L}_{\text{main}} = \lambda_1 \mathcal{L}_{\text{CE}}^{\text{token}} + \lambda_2 \mathcal{L}_{\text{CE}}^{\text{broadcast}} + \lambda_3 \mathcal{L}_{\text{dice}} + \lambda_4 \mathcal{L}_{\text{BCE}}$$

- $\mathcal{L}_{\text{CE}}^{\text{token}}$: token classification loss ([MASK]/[NEG] classification)
- $\mathcal{L}_{\text{CE}}^{\text{broadcast}}$: broadcast alignment loss (token-embedding mapping)
- $\mathcal{L}_{\text{dice}}$: mask overlap loss (spatial accuracy)
- $\mathcal{L}_{\text{BCE}}$: pixel-level binary cross-entropy (mask fine-grainedness)

A Surrogate Module is additionally designed—a single-layer Transformer that approximates the CLIP-T score to indirectly optimize mask quality. The Surrogate is first trained to fit real CLIP-T scores, after which its predictions serve as a signal to fine-tune the MLLM, Broadcaster, and Decoder. Hyperparameters are set as $\lambda_1=\lambda_2=\lambda_3=\lambda_4=1$ and $\lambda_5=10$. LoRA is used to fine-tune the MLLM, with the visual backbone and text encoder frozen. Training takes approximately 3 days on 2×A100 80GB GPUs.

## Key Experimental Results

### Main Results

Multi-instruction editing and context-aware editing (extended MagicBrush dataset):

| Method | L1↓ | L2↓ | CLIP-I↑ | DINO↑ | CLIP-T↑ |
|--------|-----|-----|---------|-------|---------|
| IP2P | 0.1460 | 0.0514 | 0.7975 | 0.6429 | 0.2715 |
| MGIE | 0.1592 | 0.0750 | 0.8090 | 0.6519 | 0.2637 |
| SmartEdit | 0.1111 | 0.0495 | 0.8739 | 0.7726 | 0.2824 |
| FoI | 0.0891 | 0.0284 | 0.8895 | 0.8190 | 0.2888 |
| **CAMILA** | **0.0661** | **0.0222** | **0.9296** | **0.8932** | **0.3006** |

On PickScore (human preference), CAMILA outperforms FoI by 24% on context-aware tasks.

### Ablation Study

| Variant | L1↓ | CLIP-I↑ | DINO↑ | CLIP-T↑ |
|---------|-----|---------|-------|---------|
| w/o Surrogate (Multi) | 0.0957 | 0.8961 | 0.8329 | 0.2975 |
| w/ Surrogate (Multi) | 0.0945 | 0.8980 | 0.8392 | 0.2984 |
| w/o Surrogate (Context-Aware) | 0.0673 | 0.9284 | 0.8910 | 0.3002 |
| w/ Surrogate (Context-Aware) | 0.0661 | 0.9296 | 0.8932 | 0.3006 |

Experiments replacing the Token Decoder with SAM show that the trained Token Decoder achieves superior CLIP and DINO scores, as it incorporates editing instruction information that SAM lacks.

### Key Findings

- Token classification accuracy reaches 90.21%, demonstrating that the MLLM effectively distinguishes executable from non-executable instructions.
- Mask quality (IoU 0.3819, Dice 0.4986), while not pixel-perfect, is sufficient as high-level guidance—editing prioritizes semantic fidelity over strict spatial matching.
- On the EMU dataset, CAMILA achieves the highest CLIP-dir on context-aware tasks, confirming the most precise control over semantic editing direction.
- CAMILA's MLLM inference takes only 0.7s, with a total inference time of 9.2s, comparable to FoI (9.1s).

## Highlights & Insights

- This is the first work to explicitly model instruction executability assessment as a first-class task in image editing, establishing a new paradigm of context-aware image editing.
- The [MASK]/[NEG] dual-token design is concise and elegant—replacing continuous attention weights with discrete classification yields clearer decision boundaries.
- The Surrogate Module elegantly circumvents the non-differentiability of multi-step diffusion forward passes by using a learnable approximator to indirectly optimize mask quality.
- The method does not rely on GPT for instruction parsing (as required by FoI), reducing pipeline complexity and error propagation.

## Limitations & Future Work

- The Token Decoder occasionally produces imprecise mask localization—particularly when adding objects, the mask region is smaller than expected, as the underlying IP2P diffusion model is not optimized for masked editing.
- The editing capability of the base diffusion model (Stable Diffusion) imposes an upper bound on overall performance.
- Training data relies on ChatGPT-4V to generate non-executable instructions, which may introduce biases.
- Validation is limited to the IP2P framework; integration with more recent editing frameworks (e.g., FLUX-based editors) has not been explored.
- The context-aware evaluation dataset is relatively small in scale (approximately 2,600 samples).

## Related Work & Insights

- **FoI**: The strongest competitor, using cross-attention for multi-instruction editing, but dependent on GPT for keyword extraction and prone to attention map misalignment.
- **SmartEdit**: Uses an MLLM to enhance instruction understanding, but lacks context-awareness and treats all instructions equally.
- **LISA / GSVA**: The [SEG] token design in referring segmentation MLLMs inspired CAMILA's [MASK]/[NEG] scheme.
- Core insight: editing models should not blindly comply with all instructions—the ability to "refuse" unreasonable operations is itself a manifestation of intelligence.

## Rating

⭐⭐⭐⭐ — The newly defined task (context-aware editing) addresses a genuine practical need, the [MASK]/[NEG] design is concise and effective, and CAMILA substantially outperforms FoI in multi-instruction scenarios. Drawbacks include reliance on the relatively dated IP2P framework and room for improvement in mask precision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[NeurIPS 2025\] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [\[CVPR 2026\] Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing](../../CVPR2026/image_generation/re-align_structured_reasoning-guided_alignment_for_in-context_image_generation_a.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)

</div>

<!-- RELATED:END -->
