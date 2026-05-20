---
title: >-
  [Paper Note] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation
description: >-
  [NeurIPS 2025][Image Generation][Autoregressive models] By systematically analyzing three key properties that hinder visual semantic learning in autoregressive image generation — local conditional dependency…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Autoregressive models"
  - "visual understanding"
  - "contrastive learning"
  - "masked image modeling"
  - "LlamaGen"
date: 2026-05-08
content_hash: 158361c075c1f6dc
---

# Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.15185](https://arxiv.org/abs/2509.15185)  
**Code**: [https://github.com/yuexy/ST-AR](https://github.com/yuexy/ST-AR)  
**Area**: Autoregressive Image Generation / Self-Supervised Learning
**Keywords**: Autoregressive models, visual understanding, contrastive learning, masked image modeling, LlamaGen

## TL;DR

By systematically analyzing three key properties that hinder visual semantic learning in autoregressive image generation — local conditional dependency, inter-step semantic inconsistency, and the absence of spatial invariance — this paper proposes ST-AR, a training framework that incorporates masked image modeling and contrastive learning into the next-token prediction objective. Without relying on any pretrained representation model, ST-AR improves the FID of LlamaGen-XL by approximately 49% (from 19.42 to 9.81), achieving performance comparable to a 3B-parameter model trained for 300 epochs within only 50 epochs.

## Background & Motivation

Autoregressive (AR) models, as the core paradigm underlying large language models such as GPT and Llama, have also demonstrated strong potential for unified multimodal generation in the image domain. However, AR image generation models exhibit a clear deficit in visual understanding, which constrains their generation quality.

Prior works such as REPA and ImageFolder improve semantic quality by distilling pretrained representation models (e.g., DINOv2) into diffusion models or tokenizers. However, AR models face unique challenges: **next-token prediction is an effective pretraining objective in NLP, but in the visual domain, due to fundamental modality differences between images and text, it struggles to learn high-level visual semantics.**

Through a thorough analysis of the LlamaGen-B model, the authors systematically identify three key properties that obstruct visual understanding for the first time:

**Local and Conditional Dependency (Obs. 1)**: Attention maps reveal that the model predominantly attends to the conditioning token (class label) and spatially adjacent tokens, making little use of distant tokens despite their full visibility. This causes local errors to propagate and accumulate during generation.

**Inter-Step Semantic Inconsistency (Obs. 2)**: Linear probing experiments reveal that the semantic content of intermediate features varies dramatically across autoregressive steps — extremely low in early steps (due to few visible tokens), rising in the middle, yet declining after step 192 — indicating the model cannot sustain globally coherent semantics throughout generation.

**Absence of Spatial Invariance (Obs. 3)**: The VQ-GAN tokenizer may produce entirely different token sequences for slightly transformed versions of the same image (e.g., crops or rescalings). The same semantic concept is thus encoded into highly divergent tokens, substantially increasing the difficulty of autoregressive learning.

- **Core Idea**: Rather than relying on external pretrained representation models, ST-AR employs self-supervised learning objectives to self-guidedly enhance the visual understanding capability of the AR model, thereby indirectly improving generation quality.

## Method

### Overall Architecture

ST-AR augments standard next-token prediction training with three additional self-supervised losses: a masked image modeling loss ($L_{\text{MIM}}$), an inter-step contrastive loss ($L_{\text{step}}$), and an inter-view contrastive loss ($L_{\text{view}}$). An EMA teacher network (sharing the architecture with the student) provides alignment targets and positive-sample representations for contrastive learning. The overall framework resembles iBOT-style training but is adapted to respect the causal constraints of autoregressive models. At inference, only the student network is used for standard autoregressive sampling, leaving the inference pipeline unchanged.

### Key Designs

1. **Attention-Masked Masked Learning (addressing Obs. 1)**: Conventional MIM replaces input tokens with MASK tokens, which is infeasible for AR models that require ground-truth preceding tokens for next-token prediction. ST-AR instead applies random masks directly to the **attention maps** of Transformer layers — setting the attention weights of a fraction $r$ of token positions to $-\infty$. This forces the model to make accurate predictions even when certain local tokens are masked out, effectively enlarging the receptive field and reducing over-reliance on local neighbors. The MIM loss aligns the last-layer hidden states of the student (masked) and teacher (unmasked) at each position using cosine distance.

2. **Inter-Step Contrastive Learning $L_{\text{step}}$ (addressing Obs. 2)**: Enforces semantic consistency across features from different autoregressive steps of the same image. $K$ positions are randomly sampled from the token sequence; the student feature at position $i$ (after a projector) serves as the anchor, the teacher feature at a different position $j$ on the same image serves as the positive sample, and features from other images serve as negatives, optimized with an InfoNCE loss.

3. **Inter-View Contrastive Learning $L_{\text{view}}$ (addressing Obs. 3)**: Enforces semantic consistency across features from different augmented views of the same image. For each image, $M=2$ randomly augmented views are generated; the student feature at position $k$ in view $i$ is paired with the teacher feature at the same position $k$ in view $j$ as a positive sample, with other images as negatives. This encourages the model to produce consistent high-level semantic representations even when the two VQ-GAN-encoded token sequences differ substantially.

### Loss & Training

Total loss: $L_{\text{ST-AR}} = L_{\text{AR}} + \alpha \cdot L_{\text{MIM}} + \beta \cdot \frac{1}{2}(L_{\text{step}} + L_{\text{view}})$

Key hyperparameters:
- Attention mask ratio $r = 0.25$
- Number of sampled steps $K = 4$
- Reconstruction loss weight $\alpha = 1.0$, contrastive loss weight $\beta = 0.5$
- EMA decay $0.9999$
- Contrastive losses applied at an intermediate layer (layer 6 for LlamaGen-B; layer 18 for L/XL)
- Training resolution $256\times256$ (vs. LlamaGen's original $384\times384$)
- Student network uses a projector (multi-layer MLP) to prevent representation collapse

## Key Experimental Results

### Main Results

| Model | Params | Epochs | FID↓ | sFID↓ | IS↑ | Prec.↑ | Rec.↑ |
|-------|--------|--------|------|-------|-----|--------|-------|
| LlamaGen-B | 111M | 300 | 26.26 | 9.22 | 48.07 | 0.59 | 0.62 |
| + ST-AR | 111M | 300 | **18.44** | **6.71** | **66.18** | **0.64** | 0.62 |
| LlamaGen-L | 343M | 300 | 13.45 | 8.32 | 82.29 | 0.66 | 0.64 |
| + ST-AR | 343M | 300 | **9.38** | **6.64** | **112.71** | **0.70** | 0.65 |
| LlamaGen-XL | 775M | 50 | 19.42 | 8.91 | 66.20 | 0.61 | 0.67 |
| + ST-AR | 775M | 50 | **9.81** | **6.94** | **109.77** | **0.71** | 0.63 |
| LlamaGen-3B | 3.1B | 300 | 9.38 | 8.24 | 112.88 | 0.69 | 0.67 |
| + ST-AR (XL) | 775M | 300 | **6.20** | **6.47** | **147.47** | **0.73** | 0.65 |

ST-AR enables LlamaGen-XL (775M) trained for only 50 epochs to approach the performance of LlamaGen-3B (3.1B) trained for 300 epochs, effectively saving approximately $4\times$ in parameters and $6\times$ in training compute. With classifier-free guidance, ST-AR applied to LlamaGen-XL achieves a FID of 2.37, approaching DiT-XL's 2.27.

### Ablation Study

| $L_{\text{MIM}}$ | $L_{\text{step}}$ | $L_{\text{view}}$ | FID↓ | LP Acc. (%)↑ | Notes |
|---|---|---|------|-------------|-------|
| ✗ | ✗ | ✗ | 31.35 | 18.68 | Baseline |
| ✓ | ✗ | ✗ | 30.58 | 22.71 | MIM alone contributes little |
| ✓ | ✓ | ✗ | 28.02 | 27.73 | Adding inter-step contrastive |
| ✗ | ✗ | ✓ | 27.78 | 38.31 | Inter-view contrastive contributes most |
| ✓ | ✓ | ✓ | **26.58** | **45.27** | All three combined is optimal |

| Ablation Dimension | Optimal Value | FID↓ | Notes |
|---|---|---|---|
| Mask ratio $r$ | 0.35 | 26.36 | Too high damages low-level spatial structure |
| Contrastive loss depth | 1/2 depth | 26.58 | Mid-depth serves as encoder/decoder boundary |
| Step count $K$ | 16 | 25.78 | Diminishing returns beyond $K=4$ |

### Key Findings

- Linear probing accuracy increases from 18.68% to 45.27% (nearly $2.5\times$), confirming a genuine improvement in visual understanding
- Attention maps shift from attending predominantly to local neighbors to attending to semantically relevant regions, exhibiting clear semantic patterns
- The inter-step semantic inconsistency issue is resolved — LP accuracy no longer degrades after step 192
- The inter-view contrastive loss contributes most substantially (38.31% LP accuracy when used alone), identifying spatial invariance as the most critical bottleneck
- Applying contrastive loss at the mid-depth layer (1/2 of total depth) yields the best results, consistent with the view that the network's first half encodes and second half decodes

## Highlights & Insights

- **Systematic validation of the "understanding promotes generation" principle**: Rather than naively distilling an external model, the paper follows a complete logical chain of analysis → diagnosis → targeted remedy.
- **Clever design of MIM via attention masking**: This circumvents the constraint that AR models cannot apply masking to input tokens without disrupting next-token prediction.
- **Fully self-guided with no external models**: Unlike REPA (which depends on DINOv2) and ImageFolder (which relies on pretrained representations), ST-AR autonomously improves itself through an EMA teacher and self-supervised objectives.
- **Remarkable efficiency**: A 775M model trained for 50 epochs approaches a 3.1B model trained for 300 epochs, demonstrating that improved understanding can substantially compensate for limited model capacity.

## Limitations & Future Work

- Training overhead increases due to the EMA teacher, multi-view encoding, and contrastive loss computation; the paper does not quantify the exact additional cost.
- Validation is limited to LlamaGen; generalizability to other AR architectures such as MAR and VAR remains to be explored.
- Evaluation is restricted to class-conditional ImageNet; text-to-image generation scenarios have not been investigated.
- Mask ratio and contrastive loss depth may require re-tuning for different model scales.
- Integration with more efficient sampling strategies such as masked AR and parallelized AR has not been explored.

## Related Work & Insights

- REPA reveals semantic insufficiency in diffusion model intermediate representations; ST-AR demonstrates that AR models face analogous but fundamentally more complex challenges.
- Self-supervised methods including iBOT, DINO, and MoCo provide the technical tools for contrastive learning and MIM; ST-AR adapts these elegantly to the AR generation framework.
- MAE demonstrates that MIM can enlarge the effective receptive field; ST-AR transfers this insight to attention masking in AR models.
- ImageFolder injects semantic constraints at the tokenizer level, which is complementary to ST-AR's model-level injection; combining the two may yield further gains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)
- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[NeurIPS 2025\] Conditional Panoramic Image Generation via Masked Autoregressive Modeling](conditional_panoramic_image_generation_via_masked_autoregres.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](../../ICCV2025/image_generation/holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](../../CVPR2026/image_generation/rewardflow_generate_images_by_optimizing_what_you_reward.md)

</div>

<!-- RELATED:END -->
