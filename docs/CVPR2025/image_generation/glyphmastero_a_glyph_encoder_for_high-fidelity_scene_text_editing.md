---
title: >-
  [Paper Note] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing
description: >-
  [CVPR 2025][Image Generation][Scene Text Editing] Proposes GlyphMastero, a glyph encoder that leverages dual-stream (local character-level + global text line-level) feature extraction, cross-level attention interaction, and multi-scale FPN fusion to provide stroke-level precise glyph guidance for diffusion models, improving sentence accuracy by 18.02% and reducing FID by 53.28% in multilingual scene text editing.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Scene Text Editing"
  - "Glyph Encoder"
  - "Diffusion Models"
  - "Cross-Level Feature Fusion"
  - "Multilingual Text Generation"
date: 2026-05-08
content_hash: 5fbd3bf8dfcf1293
---

# GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing

**Conference**: CVPR 2025  
**arXiv**: [2505.04915](https://arxiv.org/abs/2505.04915)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Scene Text Editing, Glyph Encoder, Diffusion Models, Cross-Level Feature Fusion, Multilingual Text Generation

## TL;DR
Proposes GlyphMastero, a glyph encoder that leverages dual-stream (local character-level + global text line-level) feature extraction, cross-level attention interaction, and multi-scale FPN fusion to provide stroke-level precise glyph guidance for diffusion models, improving sentence accuracy by 18.02% and reducing FID by 53.28% in multilingual scene text editing.

## Background & Motivation

1. **Background**: Scene text editing requires replacing text content in images while maintaining the original style and visual consistency. Diffusion models have shown potential in this task, exemplified by DiffUTE (a conditional inpainting framework) and AnyText (which integrates OCR features).
2. **Limitations of Prior Work**: Existing methods extract text features using pre-trained OCR models as conditions for the diffusion model, but fail to capture the hierarchical nature of text structures—from individual strokes to inter-stroke relationships and up to the entire character/line structure. Consequently, generated characters often exhibit distortions or become unrecognizable, particularly in complex writing systems like Chinese.
3. **Key Challenge**: Global neck features of OCR models (such as those used in AnyText) compress an entire text line into a single vector, losing fine-grained character-level details. Conversely, isolated character-level features lack the contextual relationships between characters within a line. It is necessary to model both local (character-level) and global (line-level) features simultaneously and allow them to interact.
4. **Goal**: How to design a glyph encoder that generates more fine-grained guidance signals representing hierarchical text structures than existing OCR features, thereby improving the accuracy of scene text generation by diffusion models?
5. **Key Insight**: Feed the glyph images of individual characters and the full text line separately into an OCR model to obtain local and global dual-stream features, and use cross-level attention interaction to allow local features to "query" the global context, thereby encoding both character details and line-level structures simultaneously.
6. **Core Idea**: Design a trainable dual-stream glyph encoder that explicitly models cross-level interactions between character-level and text line-level features via a glyph attention module, providing stroke-level precise conditional guidance for the diffusion model.

## Method

### Overall Architecture
Input: Target text string + text region mask + original scene image. Output: Edited image with the target text substituted while preserving the original style. The pipeline is based on Stable Diffusion 2.1 inpainting, with GlyphMastero acting as the conditioning encoder to replace the CLIP text encoder. Specifically, the process is: (1) Render the target text into individual character glyph images (local stream) and a full text-line glyph image (global stream); (2) Extract backbone and neck features separately using PaddleOCR-v4; (3) Cross-fuse the local-global features using two sets of glyph attention modules; (4) Concatenate and project the features via an aggregator to obtain the final conditional embedding $c \in \mathbb{R}^{N \times D}$, which guides UNet denoising through cross-attention.

### Key Designs

1. **Dual-Stream Glyph Integration**:
    - **Function**: Separately capture character-level and text line-level glyph information to provide dual-stream features for subsequent cross-level fusion.
    - **Mechanism**: The local stream renders $N$ characters into independent glyph images $x_l \in \mathbb{R}^{N \times H_l \times W_l}$, extracting the last-layer backbone output $l_b$ and neck output $l_n$. The global stream renders the entire text line into a single glyph image $x_g \in \mathbb{R}^{H_g \times W_g}$, extracting the neck output $g_n$ and fusing 5 levels of hierarchical features $x_1,...,x_5$ from the backbone using an FPN to obtain enhanced global backbone features $g_b$. The FPN fuses multi-scale features top-down via $p_i = g_i(u(p_{i+1}) + c_i)$.
    - **Design Motivation**: The local stream preserves individual stroke structure details for each character, while the global stream captures spatial relationships and line-level context among characters. FPN fusion enables the global features to encompass both high-resolution details from shallow layers and semantic information from deep layers.

2. **Glyph Attention Module**:
    - **Function**: Enable interaction between character-level local features and text line-level global features through cross-level attention, generating enhanced glyph representations that incorporate contextual information.
    - **Mechanism**: Replicate the global feature $g$ $N$ times to match the sequence length of local feature $l$. Both are mapped to the attention space ($d'=512$) via linear projection, with Rotary Position Embedding (RoPE) added. Multi-head cross-attention is then executed: local features serve as the Query, while global features serve as the Key and Value, yielding the output $o = \psi_o(z) \in \mathbb{R}^{N \times d_o}$ after LayerNorm and linear projection. Two sets of glyph attention modules, $T_n$ and $T_b$, process the neck and backbone features respectively, and the final aggregator concatenates and projects them into $c = A(o_b, o_n)$.
    - **Design Motivation**: Cross-attention allows local features of each character to "refer to" the global context (such as character spacing and alignment) of the entire text line, thereby generating more accurate glyph representations under the line-level context. A 4-head attention is sufficient to capture diverse levels of interaction patterns.

3. **Inpainting-Based Generation Framework**:
    - **Function**: Generate the target text within the specified text area while keeping the content outside the area unchanged.
    - **Mechanism**: Adopting the SD 2.1 inpainting framework, the noisy latent $z_t$ is concatenated with the binary mask $m$ and masked image latent $\mathcal{E}(x_m)$ as $\hat{z}_t = [z_t; m; \mathcal{E}(x_m)]$. The conditional embedding $c$ generated by GlyphMastero is injected into each UNet layer via cross-attention, guiding the generation during the denoising process. A null condition with a probability of 0.1 is used during training to support classifier-free guidance (CFG scale = 3 in this work) during inference.
    - **Design Motivation**: The inpainting framework is naturally suited for text editing—it only edits the masked area while preserving unmasked regions. Compared to latent-space guidance like ControlNet, the cross-attention approach achieves better style preservation and is not strictly bound to the font style of the rendered glyphs.

### Loss & Training
The standard LDM training objective $L_{LDM} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, c, t)\|^2_2]$ is used, and GlyphMastero is jointly trained end-to-end with the UNet. The weights of the OCR feature extractor (from PaddleOCR-v4) are frozen, and only the glyph attention modules and the FPN are trainable. Training is performed for 15 epochs with a global batch size of 256 on 8×V100S-32G GPUs. Inference uses 20-step DDIM denoising.

## Key Experimental Results

### Main Results
Compared with multilingual SOTAs on AnyText-Eval (2000 images):

| Method | English Sen.Acc↑ | Chinese Sen.Acc↑ | English FID↓ | Chinese FID↓ |
|------|------------|------------|---------|---------|
| DiffUTE | 0.3319 | 0.2523 | 14.32 | 24.93 |
| AnyText | 0.6067 | 0.5801 | 10.43 | 24.90 |
| **Ours** | **0.8170** | **0.7301** | **4.61** | **11.89** |

The overall sentence accuracy (average of English + Chinese) is 18.02% higher than AnyText, and the FID is 53.28% lower than AnyText. It also outperforms methods like TextCtrl in the English-only comparison (Sen.Acc 0.8170 vs 0.7654).

### Ablation Study
Trained on a subset of 375K images for 15 epochs:

| Configuration | English Sen.Acc | Chinese Sen.Acc | Description |
|------|-----------|-----------|------|
| Full model | 0.5494 | 0.5120 | Full model |
| - FPN | 0.4536 | 0.3698 | Removing FPN, average drop of 22.42% |
| - $T_b$ (backbone attention) | 0.5065 | 0.4271 | Removing backbone attention |
| - $T_n$ w/ $l_n$ (local neck) | 0.3263 | 0.2735 | Removing neck attention, using local features |
| - $T_n$ w/ $g_n$ (global neck) | 0.1003 | 0.0719 | Removing neck attention, using global features |

### Key Findings
- **Glyph Attention is Key**: Removing the glyph attention module at the neck layer (replacing it with $g_n$) causes the accuracy to plummet to ~8% (vs. ~52% originally), because the global neck feature compresses $N$ characters into a single vector, losing character-level information.
- **FPN is Crucial for Chinese**: Removing the FPN leads to a 27.8% drop in Chinese accuracy, indicating that the complex stroke structures of Chinese characters demand multi-scale feature fusion.
- **Stable Style Metrics**: The variations in FID and LPIPS across different ablation configurations are minimal, indicating that the glyph encoder primarily impacts text accuracy, while style preservation is mostly handled by the inpainting model itself.
- **Interesting Finding**: Simultaneously removing both the FPN and $T_b$ yields better performance than removing only the FPN. This is because the FPN is a custom feature fusion module tailored for $T_b$; without the FPN, $T_b$ performs worse when handling raw backbone features.

## Highlights & Insights
- **"Local-to-Global-to-Local" Information Flow**: Extracts character-level and line-level features separately, then allows local features to query the global context via cross-attention. The final output consists of enhanced character features that are "aware of their positions in the line". This design paradigm can be transferred to any sequence generation tasks requiring local-global interactions.
- **Deep Utilization of OCR Features**: Instead of simply using the final OCR output, it leverages the multi-layer pyramid features from the backbone along with the neck features, fully mining the glyph knowledge learned by the OCR backbone through trainable modules.
- **High Practical Value**: Chinese scene text editing is a highly demanded feature in the industry (e.g., Meitu Xiu Xiu—the authors are from Meitu). The significant improvement of this method on Chinese text holds direct commercial value.

## Limitations & Future Work
- Performance on long text generation still has room for improvement, constrained by the 512×512 resolution training and the capabilities of the base LDM.
- Only English and Chinese have been verified; the effectiveness on other complex writing systems such as Arabic or Thai remains unknown.
- Using SD 2.1 as the base model; switching to stronger models like SD-XL or Flux could yield further improvements.
- The FPN and glyph attention are customized for PaddleOCR-v4; changing the OCR model would require a redesign.
- For text scenes with severe curves or perspective distortions, the current rectangular-mask-based inpainting framework might lack flexibility.

## Related Work & Insights
- **vs DiffUTE**: DiffUTE uses the final hidden state of TrOCR as a fixed-length conditional vector, losing character-level details. GlyphMastero preserves independent encodings for each of the $N$ characters, retaining richer information.
- **vs AnyText**: AnyText uses OCR neck features for ControlNet-like latent conditioning, which essentially relies on global feature guidance. GlyphMastero preserves character-level precision through local-global interactions.
- **vs TextCtrl**: TextCtrl is slightly superior in English FID/LPIPS, likely due to its specialized style alignment design, but its text accuracy falls short of GlyphMastero.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-stream + cross-level attention design of the glyph encoder is novel and effective, pushing the utilization of OCR features to a new height.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-method comparisons, systematic ablations, and complete qualitative and quantitative analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear architectural diagram and highly systematic method description.
- Value: ⭐⭐⭐⭐⭐ Achieved a massive breakthrough in the difficult and practical task of Chinese scene text editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [\[ACL 2025\] FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](../../ACL2025/image_generation/flashaudio_rectified_flow_tta.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](../../ICCV2025/image_generation/your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[CVPR 2025\] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)
- [\[NeurIPS 2025\] From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging](../../NeurIPS2025/image_generation/from_cradle_to_cane_a_two-pass_framework_for_high-fidelity_lifespan_face_aging.md)

</div>

<!-- RELATED:END -->
