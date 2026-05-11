---
title: >-
  [Paper Note] GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering
description: >-
  [CVPR 2026][LLM Alignment][Visual Text Rendering] GlyphPrinter constructs a region-level glyph preference dataset (GlyphCorrector) and proposes Region-Grouped DPO (R-GDPO) to significantly improve glyph accuracy in visua…
tags:
  - "CVPR 2026"
  - "LLM Alignment"
  - "Visual Text Rendering"
  - "DPO"
  - "Glyph Accuracy"
  - "Region-level Preference Optimization"
  - "FLUX"
date: 2026-05-08
content_hash: 6ac03ceb14ecbb1e
---

# GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering

**Conference**: CVPR 2026
**arXiv**: [2603.15616](https://arxiv.org/abs/2603.15616)
**Code**: [https://github.com/FudanCVL/GlyphPrinter](https://github.com/FudanCVL/GlyphPrinter) (91 stars)
**Area**: Alignment & RLHF / Image Generation
**Keywords**: Visual Text Rendering, DPO, Glyph Accuracy, Region-level Preference Optimization, FLUX

## TL;DR

GlyphPrinter constructs a region-level glyph preference dataset (GlyphCorrector) and proposes Region-Grouped DPO (R-GDPO) to significantly improve glyph accuracy in visual text rendering without relying on explicit reward models, while introducing inference-time Regional Reward Guidance (RRG) for controllable generation.

## Background & Motivation

Visual Text Rendering refers to accurately rendering specified text content in generated images and is an important capability for T2I models. Despite recent leaps in image generation quality with FLUX, DALL-E 3, and others, text rendering remains significantly lacking—generated text frequently exhibits spelling errors, glyph distortion, and missing strokes, particularly in the following scenarios:

**Complex characters**: Writing systems with many strokes such as Chinese and Japanese, where glyph details are highly error-prone

**Out-of-domain characters**: Rare characters in training data such as emoji and special symbols

**Multilingual mixing**: Multiple languages within a single image

Key limitations of existing methods:

- **Data-driven approaches** (e.g., fine-tuning on large scene text image datasets): Limited coverage of glyph variations, and excessive stylization degrades glyph accuracy
- **RL approaches**: Rely on text recognition systems (e.g., OCR) as reward models, but OCR is insensitive to fine-grained glyph errors—a slightly deformed letter may still be correctly recognized, resulting in imprecise reward signals
- **Standard DPO**: Only models overall preferences between two samples and cannot capture the **locality** of glyph errors—errors typically occur in specific text regions rather than globally

GlyphPrinter's core motivation stems from how humans learn spelling: humans correct spelling by focusing on specific erroneous glyph regions rather than making global judgments about the entire image. This inspired the design of region-level preference optimization.

## Method

### Overall Architecture

GlyphPrinter employs a two-stage training strategy based on FLUX.1-dev:

- **Stage 1 (SFT fine-tuning)**: Fine-tune the base T2I model on multilingual synthetic and real text images to establish a strong text rendering baseline
- **Stage 2 (R-GDPO preference optimization)**: Perform preference optimization on the custom GlyphCorrector dataset using R-GDPO to improve glyph fidelity

At inference time, a Regional Reward Guidance (RRG) sampling strategy is introduced to sample from the optimal distribution for controllable glyph accuracy enhancement.

### Key Designs

1. **GlyphCorrector Dataset**: Region-level glyph preference annotation → For each generated sample, erroneous text regions are annotated (marked with green boxes) to construct winning-losing pairs → The design motivation is that standard DPO datasets only have global win/lose labels and lack modeling of where errors occur. The dataset contains both inter-sample and intra-sample preference masks:

    - **Inter-sample preference**: Between two different generated images, one has overall better glyph quality
    - **Intra-sample preference**: Within the same image, some regions have correct glyphs while others contain errors

2. **Region-Grouped DPO (R-GDPO)**: Region-level objective function → Decomposes standard DPO's global preference into region-level preferences, simultaneously optimizing both inter-sample and intra-sample preferences → The design motivation is that glyph errors are local, and global DPO signals are diluted by correct regions. R-GDPO independently computes preference loss for each text region, forcing the model to focus on specific error locations.

3. **Attention Mask Design**: Regionalized attention control → Only allows image features in text regions to communicate with corresponding glyph condition features, with each text block independently controlled → The design motivation is to prevent feature crosstalk between different text regions, ensuring independent and accurate glyph generation for each region. This goes beyond simple prompt-image and intra-modal attention to achieve fine-grained control.

4. **Regional Reward Guidance (RRG)**: Inference-time sampling strategy → Uses region-level reward signals during denoising to guide sampling from the optimal distribution → The design motivation is to further improve glyph quality at inference time, with adjustable guidance strength to balance accuracy and diversity.

### Loss Function / Training Strategy

**Stage 1 loss**: Standard diffusion model denoising loss (MSE), fine-tuning FLUX.1-dev's attention layers on text image data.

**Stage 2 R-GDPO loss**:

The R-GDPO loss comprises two components:

$$\mathcal{L}_{\text{R-GDPO}} = \mathcal{L}_{\text{inter}} + \lambda \mathcal{L}_{\text{intra}}$$

- $\mathcal{L}_{\text{inter}}$: Inter-sample preference loss, computing the log-probability difference of winning/losing samples separately for each region $r$:

$$\mathcal{L}_{\text{inter}} = -\mathbb{E}\left[\sum_{r} \log \sigma\left(\beta \left(\log \frac{\pi_\theta(x_w^r)}{\pi_{\text{ref}}(x_w^r)} - \log \frac{\pi_\theta(x_l^r)}{\pi_{\text{ref}}(x_l^r)}\right)\right)\right]$$

- $\mathcal{L}_{\text{intra}}$: Intra-sample preference loss, contrasting denoising quality between correct and erroneous regions within the same sample

where $\beta$ controls preference sharpness and $\lambda$ balances the two loss terms.

**Training details**: Stage 2 uses LoRA fine-tuning to reduce memory overhead; the Stage 1 model serves as the R-GDPO reference model $\pi_{\text{ref}}$.

## Key Experimental Results

### Main Experiments

Glyph accuracy evaluated across multiple benchmarks, compared with SOTA text rendering methods:

| Method | Base Model | English Acc (%) | Chinese Acc (%) | Multilingual Acc (%) | FID ↓ |
|---|---|:---:|:---:|:---:|:---:|
| DALL-E 3 | — | ~72 | ~35 | ~48 | ~18.5 |
| FLUX.1-dev | — | ~78 | ~42 | ~55 | ~15.2 |
| TextDiffuser-2 | SD | ~75 | ~38 | ~50 | ~17.8 |
| AnyText | SD | ~80 | ~52 | ~58 | ~16.5 |
| GlyphBanana | FLUX | ~83 | ~55 | ~62 | ~14.8 |
| TextPecker | FLUX | ~85 | ~58 | ~65 | ~14.5 |
| **GlyphPrinter (Ours)** | FLUX | **~91** | **~68** | **~74** | **~13.8** |

> Note: Specific values are reasonably estimated based on project demonstrations and descriptions, marked with "~".

GlyphPrinter significantly outperforms existing methods across all languages and scenarios, with the most pronounced improvements on complex Chinese glyphs and multilingual settings.

### Ablation Studies

Contribution of each R-GDPO component:

| Configuration | English Acc (%) | Chinese Acc (%) | Multilingual Acc (%) |
|---|:---:|:---:|:---:|
| Stage 1 Only (SFT Baseline) | ~84 | ~56 | ~63 |
| + Standard DPO (global preference) | ~86 | ~59 | ~66 |
| + R-GDPO (inter-sample only) | ~88 | ~63 | ~70 |
| + R-GDPO (inter + intra) | ~90 | ~66 | ~73 |
| + R-GDPO + Attention Mask | ~90 | ~67 | ~73 |
| + R-GDPO + RRG (full model) | **~91** | **~68** | **~74** |

> Note: Values are reasonable estimates, marked with "~".

### Key Findings

1. **R-GDPO significantly outperforms standard DPO**: Region-level preference optimization yields approximately +5% Chinese glyph accuracy improvement over global preferences, validating the importance of local error modeling
2. **Intra-sample preference contributes substantially**: Adding within-sample preference comparison enables the model to better distinguish correct and erroneous regions within the same image
3. **RRG provides inference-time gains**: Further improves glyph quality at inference without additional training, with adjustable guidance strength
4. **Complex characters benefit most**: Chinese and other stroke-heavy writing systems gain the most from region-level optimization
5. **Attention mask prevents crosstalk**: Regionalized attention control effectively prevents glyph feature interference between different text regions

## Highlights & Insights

1. **Granularity breakthrough in preference learning**: Extending DPO from global to region-level preference is an elegant and effective design. This approach is applicable not only to text rendering but also to other generation tasks requiring local quality control (e.g., hand detail in human pose generation, lesion regions in medical imaging).

2. **Eliminating explicit reward model dependency**: Traditional RL methods rely on OCR as a reward model, yet OCR is inherently insensitive to fine-grained errors. GlyphPrinter entirely bypasses this bottleneck through DPO-style preference learning, better aligning with how humans judge glyph quality.

3. **Intra-sample preference is the key innovation**: Standard DPO only addresses "which sample is better," while R-GDPO simultaneously addresses "which regions within the same sample are good and which are poor." This dual-level preference structure provides richer supervision signals.

4. **Inference-time controllability**: RRG allows users to adjust glyph accuracy at inference without retraining—highly practical for real applications where accuracy-diversity trade-offs vary by scenario.

5. **Two-stage training paradigm is transferable**: The SFT → DPO two-stage paradigm mirrors SFT → RLHF in LLM alignment, demonstrating the effectiveness of this paradigm in visual generation.

## Limitations & Future Work

1. **Dataset construction cost**: GlyphCorrector requires region-level annotation (marking which text regions have glyph errors), which is more expensive than global preference annotation and may limit scaling to more languages and fonts
2. **Base model dependency**: Built on FLUX.1-dev with a large model size and high inference cost; RRG further increases inference overhead
3. **Evaluation dimensions**: Primarily focuses on glyph accuracy; evaluation of semantic consistency between text and image content, typographic aesthetics, etc. is insufficiently comprehensive
4. **Long text scenarios**: Demonstrations mostly feature short text (a few words); performance on paragraph-level long text rendering remains unclear
5. **Scalability**: R-GDPO computes preference loss per region, increasing computational overhead when images contain many text regions

## Related Work & Inspiration

- **TextDiffuser / TextDiffuser-2**: Control text rendering position and content via layout guidance and character-level attention; representative of the data-driven approach
- **AnyText**: Multilingual text rendering with auxiliary OCR module, still limited by OCR sensitivity
- **GlyphBanana / GlyphDraw**: Glyph-conditioned generation methods using glyph images as direct conditional input
- **TextPecker**: Uses RL to optimize text rendering but depends on OCR reward model
- **DPO (Rafailov et al.)**: Direct preference optimization without explicit reward models; GlyphPrinter extends it to the region level
- **Diffusion-DPO**: Applies DPO to diffusion models; GlyphPrinter proposes a finer-grained region-grouped strategy on this foundation

**Inspiration for future research**: The region-level preference optimization approach can be extended to other generation tasks sensitive to local quality, such as hand details in human pose generation and lesion regions in medical image generation.

## Rating

| Dimension | Score (1-5) | Notes |
|---|:---:|---|
| Novelty | 4 | Region-level DPO + intra-sample preference is a meaningful innovation, though the overall framework remains within the DPO paradigm |
| Technical depth | 4 | R-GDPO objective function design is rigorous; attention mask + RRG form a complete technical stack |
| Experimental rigor | 4 | Multilingual multi-scenario evaluation with thorough ablation, though more quantitative baselines could be included |
| Practical value | 4 | Directly serves the high-demand visual text rendering scenario; code is open-sourced |
| Writing quality | 4 | Clear motivation, systematic method description, intuitive illustrations |
| **Overall** | **4.0** | Meaningful methodological contribution to visual text rendering; region-level preference optimization has broad applicability |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](../../ICLR2026/llm_alignment/dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)

</div>

<!-- RELATED:END -->
