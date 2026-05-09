---
title: >-
  [Paper Note] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance
description: >-
  [ICCV 2025][Image Generation][visual style prompt] This paper proposes **Negative Visual Query Guidance (NVQG)**, a training-free method that suppresses content leakage by injecting the reference image's queries as a negative guidance signal in self-attention layers. The approach achieves high-quality visual style prompting and outperforms existing methods in both style similarity and text alignment.
tags:
  - ICCV 2025
  - Image Generation
  - visual style prompt
  - content leakage
  - Classifier-Free Guidance
  - self-attention feature swapping
  - negative visual query guidance
  - training-free style transfer
  - diffusion models
date: 2026-05-08
content_hash: 242c74305f857379
---

# StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance

**Conference**: ICCV 2025
**arXiv**: [2510.06827](https://arxiv.org/abs/2510.06827)
**Code**: [GitHub](https://github.com/) (publicly released)
**Area**: Image Generation
**Keywords**: visual style prompt, content leakage, Classifier-Free Guidance, self-attention feature swapping, negative visual query guidance, training-free style transfer, diffusion models

## TL;DR

This paper proposes **Negative Visual Query Guidance (NVQG)**, a training-free method that suppresses content leakage by injecting the reference image's queries as a negative guidance signal in self-attention layers. The approach achieves high-quality visual style prompting and outperforms existing methods in both style similarity and text alignment.

## Background & Motivation

**Core Problem**: When using reference images as visual style prompts in text-to-image diffusion models, a critical "content leakage" problem arises — non-stylistic elements such as pose, layout, and objects from the reference image inevitably bleed into the generated output, reducing diversity and degrading text alignment.

**Limitations of Prior Work**:

- **Training-based methods** (DreamBooth, LoRA, Textual Inversion, IP-Adapter, etc.): Require additional training with substantial computational cost, and suffer from an inherent style–content trade-off.
- **Training-free methods** (StyleAligned, CrossAttn, StyleID, etc.): Transfer style by swapping self-attention keys/values, but cannot fully eliminate content leakage; some do not support real images as references or primarily target image-to-image (I2I) rather than text-to-image (T2I) scenarios.
- **CFG neglected**: Prior work ignores the role of Classifier-Free Guidance when manipulating features, leading to degraded image quality and text alignment.

**Motivation**: A method is needed that can **independently control** the style and content strength from a visual prompt, enabling generated results that faithfully reflect the reference style while remaining fully aligned with the text prompt — without any additional training.

## Method

### Overall Architecture

StyleKeeper takes a text prompt and a visual style prompt as input, and produces style-consistent images without content leakage through four core components:

1. **CFG with Swapping Self-Attention**
2. **Negative Visual Query Guidance (NVQG)**
3. **Stochastic Encoding** — for real reference images
4. **Color Calibration** — for real reference images

### Key Design 1: CFG with Swapping Self-Attention

The core idea is to maintain two denoising processes — the original process (driven by the text prompt) and the reference process (driven by the visual style prompt) — and inject the **keys and values** from the reference process into the original process at the self-attention layers:

$$\text{Attention}(Q_{\text{text}}, K_{\text{visual}}, V_{\text{visual}}) = \text{Softmax}\left(\frac{Q_{\text{text}} K_{\text{visual}}^\top}{\sqrt{d}}\right) V_{\text{visual}}$$

The authors are the first to integrate this operation with CFG, proposing a unified guidance formulation:

$$\tilde{\epsilon}_\theta = (1+w) \cdot \ddot{\epsilon}_\theta(x_t, Q_{\text{text}}, KV_{\text{visual}}) - w \cdot \epsilon_\theta(x_t, \emptyset)$$

where queries come from the original process (preserving content) and keys/values from the reference process (carrying style). Incorporating CFG substantially improves image quality and text alignment.

### Key Design 2: Negative Visual Query Guidance (NVQG)

**Core Insight**: Although KV injection tends to preserve content and transfer style, content information from the reference image still leaks through the KV features. NVQG applies Bayes' rule to decompose the visual prompt condition into separate style and content factors, then uses **negative guidance** to suppress the content factor.

Specifically, query injection (injecting the reference image's queries into the original process) is used to approximate the score corresponding to "only the reference image's content":

$$\ddot{\epsilon}_\theta(x_t, Q_\emptyset, KV_{\text{visual}}^{\text{content}}) \approx \ddot{\epsilon}_\theta(x_t, Q_{\text{visual}}, KV_\emptyset)$$

This term is then subtracted as a negative guidance signal from the final score, effectively suppressing content leakage from the reference image. The method essentially "deliberately simulates content leakage and then subtracts it away."

### Key Design 3: Self-Attention Layer Selection

Diffusion model architectures consist of three parts — downblock, bottleneck, and upblock:

- **Bottleneck**: Encodes the content elements of the image; swapping here directly transfers content and should be avoided.
- **Downblock**: Content layout in feature maps is not well-defined; swapping leads to incoherent generation.
- **Upblock**: Applying swapping self-attention only in the upblock effectively transfers style without leaking content.

Further experiments show that beginning the swap from layer 24 of SDXL achieves the optimal balance, and this optimal layer is consistent across different reference images.

### Key Design 4: Support for Real Images

- **Stochastic Encoding**: Intermediate latents are obtained directly via the forward diffusion process $x_t^{\text{visual}} = \sqrt{\alpha_t} \cdot x_0^{\text{visual}} + \sqrt{1-\alpha_t} \cdot \epsilon_t$, avoiding the accumulated errors and artifacts of DDIM inversion.
- **Color Calibration**: AdaIN is applied to the predicted $x_0$ during denoising to match the channel-wise mean and standard deviation of the reference image, enabling precise color alignment.

### Loss & Training

This is a **training-free** method with no loss function or training involved. The core lies in the guidance formulation at the sampling stage, achieving style–content disentanglement by combining three scores:

- Positive guidance: $\ddot{\epsilon}_\theta(x_t, Q_{\text{text}}, KV_{\text{visual}})$ — conditional score carrying the target style
- Negative guidance: $\ddot{\epsilon}_\theta(x_t, Q_{\text{visual}}, KV_\emptyset)$ — score carrying reference content (to be subtracted)
- Unconditional score: $\epsilon_\theta(x_t, \emptyset)$ — CFG baseline

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | Style Similarity (DINO↑) | Text Alignment (CLIP↑) | Diversity (LPIPS↑) | Gram Matrix↑ |
|--------|--------------------------|------------------------|-------------------|-------------|
| **StyleKeeper (Ours)** | **Best** | **Best** | **Best** | **0.791** |
| StyleAligned | Second | Medium | Poor (content leakage) | 0.759 |
| IP-Adapter | High (at cost of text) | Worst | Medium | 0.768 |
| DreamBooth-LoRA | Medium | Medium | Medium | 0.759 |
| StyleDrop | Worst | Medium | Medium | 0.659 |

Evaluation setting: 40 reference images × 120 text prompts × 6 initial noise seeds = **720** generated images.

### User Study

| Method | User Preference |
|--------|----------------|
| **StyleKeeper** | **58.15%** |
| IP-Adapter | 18.47% |
| StyleAligned | 13.15% |
| DreamBooth-LoRA | 7.66% |
| StyleDrop | 2.58% |

62 participants, 20 evaluation groups. More than half of the users rated StyleKeeper as best in both style alignment and text alignment.

### Ablation Study

| Configuration | Result |
|---------------|--------|
| No CFG + No NVQG | Severe artifacts, very poor image quality |
| CFG + No NVQG | Improved quality, but severe content leakage (layout, structure) |
| CFG + NVQG (full method) | Best results, clear style–content separation |
| DDIM inversion vs. Stochastic Encoding | Stochastic Encoding outperforms DDIM inversion on all metrics |
| Without Color Calibration | Reduced style similarity and worse color matching |

### Key Findings

1. **NVQG is critical for eliminating content leakage**: Without NVQG, the pose, layout, and objects of the reference image bleed into results; with NVQG enabled, diverse scenarios (painting styles, multi-instance, pose-specified) are generated correctly.
2. **Swapping self-attention only in the upblock is the optimal strategy**: A consistent "inflection layer" exists across reference images where all metrics shift sharply.
3. **Stochastic Encoding > DDIM Inversion**: K-S tests show that stochastic encoding produces latents closer to a standard Gaussian distribution, with p-values > 0.05.
4. **Generalizability**: The method is compatible with ControlNet (I2I style transfer), DreamBooth-LoRA, Stable Diffusion v1.5, and Pixart-α.

## Highlights & Insights

1. **The NVQG design is elegantly conceived**: Rather than directly "blocking" content leakage, it deliberately simulates the leakage as a negative sample and cancels it via CFG subtraction — a concise and effective "fight fire with fire" strategy.
2. **First to unify CFG with feature swapping**: Prior work overlooked the role of CFG in feature manipulation; this paper demonstrates that CFG is indispensable for quality and text alignment.
3. **Systematic analysis of layer selection**: Visualization of attention maps reveals that late upblock layers focus on style-relevant regions, while early layers exhibit overly broad attention that causes leakage.
4. **Stochastic Encoding as a replacement for DDIM Inversion**: A single forward operation yields statistically aligned intermediate latents, efficiently and without accumulated errors or storage of intermediate states.
5. **Fully training-free**: No model weights are modified, no additional datasets are required, and the method operates purely at sampling time as a plug-and-play module.

## Limitations & Future Work

1. **Bounded by the pretrained model's capability**: Concepts outside the training distribution cannot be generated (e.g., "stone golem" generation fails).
2. **Visual style dominates when conflicting with textual style**: When the text-described style contradicts the reference image's style, the visual style suppresses the textual specification.
3. **Additional computational overhead**: Running both the original and reference denoising processes, plus the extra forward pass for NVQG, results in inference time approximately 3× that of standard generation.
4. **Validated only for T2I and I2I scenarios**: Extension to other modalities such as video generation has not been explored.
5. **Optimal layer selection is architecture-dependent**: Layer 24 is optimal for SDXL; a new search is required for other architectures.

## Related Work & Insights

- **StyleAligned [Hertz et al., 2023]**: Achieves style alignment via shared self-attention but retains original features, causing content leakage — this paper addresses that core deficiency via NVQG.
- **CrossAttn [Alaluf et al., 2024]** / **StyleID [Chung et al., 2024]**: DDIM inversion-based KV injection primarily targeting I2I; insufficient style reflection — this paper's stochastic encoding and color calibration offer a superior approach for real image handling.
- **Composable Diffusion [Liu et al., 2022]**: Bayesian decomposition framework for concept composition — extended here to style/content decomposition in feature space.
- **MasaCtrl [Cao et al., 2023]**: Mutual self-attention control — this paper reveals the semantic differences across block types, providing layer-selection guidance for feature manipulation methods.
- **AdaIN [Gatys, 2015]**: Channel-wise statistics matching in classical style transfer — repurposed here for color calibration at intermediate diffusion steps, a clever reuse of a classical technique.

## Rating

| Dimension | Score | Note |
|-----------|-------|------|
| Novelty | ⭐⭐⭐⭐ | NVQG's negative guidance approach is novel; unifying CFG with feature swapping is insightful |
| Technical Depth | ⭐⭐⭐⭐ | Bayesian decomposition is rigorously derived; layer selection analysis is systematic |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 720-image quantitative evaluation + 62-participant user study + comprehensive ablations + multi-model validation |
| Practical Value | ⭐⭐⭐⭐ | Training-free plug-and-play, compatible with multiple models and ControlNet |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure with rich illustrations; some notation is dense |
| **Overall** | **⭐⭐⭐⭐** | Solid work that makes a meaningful contribution to training-free visual style prompting |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](aicomposer_any_style_and_content_image_composition_via_feature_integration.md)

</div>

<!-- RELATED:END -->
