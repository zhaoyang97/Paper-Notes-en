---
title: >-
  [Paper Note] InsightEdit: Towards Better Instruction Following for Image Editing
description: >-
  [CVPR 2025][Image Generation][image editing] This work proposes InsightEdit, constructs a high-quality editing dataset with 2.5 million pairs named AdvancedEdit, and designs a two-stream bridging mechanism to inject both the textual reasoning features and visual semantic features of an MLLM into a diffusion model, achieving SOTA performance in complex instruction following and background consistency.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "image editing"
  - "instruction following"
  - "MLLM"
  - "diffusion model"
  - "two-stream bridging"
  - "AdvancedEdit dataset"
date: 2026-05-08
content_hash: ff4507fb67ddb69e
---

# InsightEdit: Towards Better Instruction Following for Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2411.17323](https://arxiv.org/abs/2411.17323)  
**Code**: [Project Page](https://poppyxu.github.io/InsightEdit_web/)  
**Area**: Image Generation  
**Keywords**: image editing, instruction following, MLLM, diffusion model, two-stream bridging, AdvancedEdit dataset

## TL;DR

This work proposes InsightEdit, constructs a high-quality editing dataset with 2.5 million pairs named AdvancedEdit, and designs a two-stream bridging mechanism to inject both the textual reasoning features and visual semantic features of an MLLM into a diffusion model, achieving SOTA performance in complex instruction following and background consistency.

## Background & Motivation

**Background**: Instruction-based end-to-end image editing has made significant progress in recent years, with methods such as InstructPix2Pix, InstructDiffusion, and SmartEdit exploring various paradigms.

**Limitations of Prior Work**:
1. **Low dataset quality**: Existing datasets (such as those generated via the Prompt2Prompt method in InstructPix2Pix) suffer from low resolution ($512^2$), poor background consistency, and over-simplified or templated instructions.
2. **Underutilization of image conditions**: Existing methods majorly rely on CLIP text encoders or text-level understanding from MLLMs to provide conditions, neglecting the rich visual semantic information of the source image. This leads to weak capabilities in following complex instructions and poor background preservation.

**Key Challenge**: High-quality editing requires both understanding the semantics of complex instructions and maintaining the visual consistency of unedited regions, whereas existing methods relying solely on textual conditions fail to reconcile both.

**Goal**: Construct a high-quality dataset and design an editing framework that utilizes both textual and visual conditions to achieve complex instruction following and high background consistency.

**Key Insight**: Address the problem from both data and model perspectives: construct a high-quality editing pair dataset using an automated pipeline, and inject both textual and visual information from the MLLM into the diffusion model via two-stream bridging.

## Method

### Overall Architecture

InsightEdit consists of three modules:
1. **Comprehension Module**: Uses LLaVA-7B to receive the source image and editing instructions, compressing the multimodal understanding results through the "[MM]" special token.
2. **Bridging Module**: Employs a two-stream design to align textual features and visual features to the diffusion model space, respectively.
3. **Generation Module**: Uses decoupled cross-attention to inject textual and visual conditions into the UNet to generate the target image.

### Key Designs

**1. AdvancedEdit Dataset Construction Pipeline**
- **Function**: Automates the construction of over 2.5 million high-quality editing pairs, covering three categories of tasks: removal, addition, and replacement.
- **Mechanism**: A five-step pipeline: ① MLLM extracts global descriptions and target JSON lists $\to$ ② GroundedSAM generates masks $\to$ ③ Mask-based editing models (e.g., BrushNet/PowerPaint) generate target images $\to$ ④ MLLM rewrites instructions (simple and reasoning versions) $\to$ ⑤ Quality filtering via VIEScore.
- **Design Motivation**: Mask-based editing models yield much better generation quality than Prompt2Prompt; utilizing MLLMs to rewrite instructions introduces reasoning complexity; VIEScore filtering ensures high quality; source data utilizes Pexels high-resolution (~2K+) real photos.

**2. Two-Stream Bridging Mechanism**
- **Textual Branch (Q-Former + BIM)**: Uses a text-aligned Q-Former to extract textual reasoning information $q' = Q_\beta(q, h)$ from the [MM] token hidden states, followed by a BIM module to achieve bidirectional information exchange between source image features and textual features, outputting $f_{txt}$ (textual condition) and $v_{txt}$ (text-aware visual features, added to UNet input).
- **Visual Branch (IAA)**: Designs an Image Alignment Adapter (IAA) using an MLP Mapper to map the [MM] token hidden states $h \in \mathbb{R}^{r \times 4096}$ to $\mathbb{R}^{1 \times 768}$, aligned with target image CLIP features via supervised loss; it is then linearly expanded to a token sequence $f_{img}$ of size $\mathbb{R}^{N \times 768}$.
- **Design Motivation**: Textual features provide high-level editing semantics, while visual features contain richer detailed conditions (such as the target background); the two complement each other to guide the editing process more precisely.

**3. Decoupled Cross-Attention Generation**
- **Function**: Uses two independent cross-attention layers in each UNet block to handle textual and visual conditions separately.
- **Mechanism**: $\mathbf{Z} = \text{Attention}(\mathbf{Q}, \mathbf{K_{txt}}, \mathbf{V_{txt}}) + \lambda \cdot \text{Attention}(\mathbf{Q}, \mathbf{K_{img}}, \mathbf{V_{img}})$, where $\lambda$ can be adjusted during inference to control the weight of the visual condition.
- **Design Motivation**: Inspired by IP-Adapter, the decoupled design ensures the independent effectiveness of both paths while providing flexible control during inference.

### Loss & Training

- **LLM Loss**: Negative log-likelihood $L_{\text{LLM}}$ of predicting the [MM] token, with LLM parameters frozen and fine-tuned via LoRA.
- **IAA Alignment Loss**: $\mathcal{L}_{\text{IAA}} = \|\text{CLIP}(\mathbf{I}_{\text{tar}}) - \text{Mapper}(h)\|_2^2$
- **Diffusion Loss**: Standard $\epsilon$-prediction loss, with inputs being the concatenated noisy latent, source image latent, and $v_{txt}$.
- **Three-stage training**: Trained on 8 × H100 GPUs.
- Only 202,822 editing pairs from AdvancedEdit were used (due to resource constraints).

## Key Experimental Results

### Main Results (Comparison on AdvancedEdit-Eval)

| Method | VIEScore↑ | CLIPScore↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| InstructPix2Pix | 0.342 | 19.528 | 20.192 | 0.694 | 0.182 |
| SmartEdit-7B | 0.682 | 20.114 | 20.115 | 0.651 | 0.131 |
| InsightEdit | 0.738 | 20.395 | 21.267 | 0.675 | 0.112 |
| **InsightEdit + AdvancedEdit** | **0.831** | **21.002** | **22.871** | **0.716** | **0.071** |

### Reason-Edit Comparison

| Method | Understanding VIEScore↑ | Reasoning VIEScore↑ |
|---|---|---|
| SmartEdit-7B | 0.866 | 0.835 |
| InsightEdit | 0.901 | 0.893 |
| **InsightEdit + AdvancedEdit** | **0.934** | **0.947** |

### Ablation Study (IAA Module)

| IAA | PSNR↑ | SSIM↑ | LPIPS↓ | CLIPScore↑ | VIEScore↑ |
|---|---|---|---|---|---|
| ✗ | 22.348 | 0.692 | 0.095 | 20.652 | 7.307 |
| ✓ | **22.871** | **0.716** | **0.071** | **21.002** | **7.545** |

### Key Findings

1. **Data-driven improvement**: InsightEdit + AdvancedEdit shows a significant performance gain over the model-only version in VIEScore (0.738 $\to$ 0.831), demonstrating the value of high-quality data.
2. **Key role of the visual branch**: The IAA module reduces LPIPS from 0.095 to 0.071 and increases VIEScore from 7.307 to 7.545, effectively improving background consistency.
3. **Double superiority in comprehension and reasoning**: Outperforms SmartEdit in both understanding and reasoning scenarios on Reason-Edit, proving the comprehensive advantages of the two-stream mechanism.
4. **AdvancedEdit enhances generalization**: Training on complex instruction data consistently brings improvements across both understanding and reasoning scenarios.

## Highlights & Insights

- The pipeline combining mask-based editing models and MLLM instruction rewriting is clever, resolving the quality issues of the Prompt2Prompt method.
- The two-stream bridging mechanism simultaneously utilizes the textual reasoning and visual perception capabilities of the MLLM, presenting a natural extension of SmartEdit which only uses textual embeddings.
- The IAA module is elegantly designed—using target image CLIP features to supervise Mapper alignment, eliminating the need for target images during inference.
- Decoupled cross-attention + adjustable $\lambda$ provides practical flexibility during inference.

## Limitations & Future Work

- Only around 200k pairs from AdvancedEdit were utilized (out of 2.5 million+); due to resource constraints, training on a larger scale may bring further improvements.
- Data construction relies on commercial models like GPT-4o, leading to high costs.
- Validity has only been verified on removal, addition, and replacement tasks, leaving more complex tasks like style transfer or attribute editing unaddressed.
- Inference speed is not reported; the full pipeline inference overhead containing LLaVA-7B + UNet may be relatively large.

## Related Work & Insights

- SmartEdit first applied MLLMs to instruction understanding, but only utilized text embeddings. This work extends it to two streams (text and image).
- The decoupled cross-attention design from IP-Adapter is effectively transferred to editing tasks.
- The pipeline of mask-based editing $\to$ instruction rewriting $\to$ quality filtering in data construction can be generalized to building other image editing datasets.

## Rating

⭐⭐⭐⭐ — Dual-line innovation in both data and model with thorough evaluation. The automation level and quality of the data pipeline are impressive, and the two-stream bridging design is reasonable and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [\[CVPR 2025\] MagicQuill: An Intelligent Interactive Image Editing System](magicquill_an_intelligent_interactive_image_editing_system.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](../../ICCV2025/image_generation/superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](../../ICCV2025/image_generation/early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)

</div>

<!-- RELATED:END -->
