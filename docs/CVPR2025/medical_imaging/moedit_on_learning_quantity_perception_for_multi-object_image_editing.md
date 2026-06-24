---
title: >-
  [Paper Note] MoEdit: On Learning Quantity Perception for Multi-Object Image Editing
description: >-
  [CVPR 2025][Medical Imaging][multi-object editing] Proposes MoEdit, an auxiliary-tool-free multi-object image editing framework. It compensates for the cross-attribute confusion in CLIP embeddings via the FeCom module, and injects quantity perception into the U-Net through the QTTN module, ensuring quantity consistency and attribute independence during editing.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "multi-object editing"
  - "quantity perception"
  - "feature compensation"
  - "Stable Diffusion"
  - "SDXL"
date: 2026-05-08
content_hash: 48df2f7f557567b4
---

# MoEdit: On Learning Quantity Perception for Multi-Object Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2503.10112](https://arxiv.org/abs/2503.10112)  
**Code**: [Tear-kitty/MoEdit](https://github.com/Tear-kitty/MoEdit)  
**Area**: Medical Images  
**Keywords**: multi-object editing, quantity perception, feature compensation, Stable Diffusion, SDXL

## TL;DR

Proposes MoEdit, an auxiliary-tool-free multi-object image editing framework. It compensates for the cross-attribute confusion in CLIP embeddings via the FeCom module, and injects quantity perception into the U-Net through the QTTN module, ensuring quantity consistency and attribute independence during editing.

## Background & Motivation

**Background**: Image editing methods based on Stable Diffusion perform well in single-object scenarios but face severe quantity inconsistency in multi-object settings—where objects may multiply, decrease, or suffer from attribute blending after editing. Existing approaches either focus solely on individual objects (ignoring overall quantity consistency) or rely on auxiliary tools such as masks or bounding boxes (leading to high training costs and requiring one-to-one mapping).

**Core Problem**: How to achieve quantity perception in multi-object image editing without any auxiliary tools (such as masks, LLMs, or bounding boxes)—ensuring consistent object counts between input and output while allowing individual object attributes to be edited independently?

**Key Observations**:
1. When encoding multi-object images with CLIP image encoders, attribute features of different objects interlace with each other, leading to chaotic attributes during editing (e.g., attributes of a fox shifting to those of a rabbit).
2. Even when Gaussian noise is added to the CLIP features, the structure and clarity of the image are preserved, indicating that the root cause of attribute interlacing is not informational loss, but CLIP's lack of ability to distinguish attribute boundaries between objects.
3. Existing methods that use auxiliary tools can extract individual object attributes but fail to model "global" information simultaneously.

## Method

### Overall Architecture

MoEdit is built upon SDXL and consists of two core modules:
1. **FeCom (Feature Compensation)**: Utilizes text prompts (containing quantity and object information) to compensate for the limitations of CLIP encoding, enhancing the distinguishability and separability of object attributes.
2. **QTTN (Quantity Attention)**: Perceives both individual and collective object information from the enhanced features and injects it into the 4th block of the U-Net to control the editing process.

During training, $c_e$ (the editing instruction) is set to null-text to isolate textual interference, utilizing only MSE loss. During inference, users can customize $c_e$ to guide the editing process.

### Key Design 1: Feature Compensation (FeCom) Module

- **Problem**: CLIP(I) cannot encode distinguishable object attributes for multi-object images.
- **Solution**: Maps the quantity and object information from the text prompt $c_q$ (e.g., "six doll bears") onto CLIP(I) using a Feature Attention mechanism to generate a compensation feature $I_c$.

$$I_g = CLIP(I) + \lambda \cdot I_c$$

Where $I_c = \text{Softmax}(\frac{Q_g K_t^T}{\sqrt{d_k}}) V_t$, $Q_g$ originates from CLIP(I), and $K_t, V_t$ originate from $c_q$.

- **Function**: Compensates the ambiguous CLIP features using semantic quantity and object name information from the text, turning them into an enhanced feature $I_g$ with distinct object attributes.

### Key Design 2: Quantity Attention (QTTN) Module

- **Structure**: Contains three components: the Extraction module $E_t$, Attention interaction, and U-Net injection.
- **Extraction**: Extracts both individual and global information for each object from the enhanced feature $I_g$.
- **Attention**: Interacts the extracted information with the noise $z_t^4$ of the 4th block in the U-Net.

$$V_{new} = \text{Softmax}(\frac{Q_z K_g^T}{\sqrt{d_k}}) V_g$$

- **Injection**: $z_t^5 = \text{Attn}(Q_z, K_i, V_i) + \beta \cdot V_{new}$, directly injecting quantity perception into the cross-attention output of the U-Net.

### Key Design 3: Selection of Injection Point

- The U-Net is decomposed into 11 Transformer blocks (4 downsampling + 1 middle + 6 upsampling).
- The 4th block $B_4$ is chosen as the injection point to yield the optimal balance between quantity perception utilization and editing flexibility.
- Experiments validate the impact of different injection points on performance.

### Loss & Training

Training utilizes only the MSE Loss (the standard reconstruction loss of the diffusion model) along with null-text input to avoid textual interference.

## Key Experimental Results

### Main Results

Comparison with 7 methods across 6 objective metrics + 2 subjective metrics (Table 2):

| Method | NIQE↓ | HyperIQA↑ | CLIP Score(Whole)↑ | MOS↑ | Numerical (3 Objects)↑ | Numerical (9+ Objects)↑ |
|---|---|---|---|---|---|---|
| SSR-Encoder | 3.106 | 62.16 | 0.2897 | 67.75 | 25.87 | 3.75 |
| IP-Adapter | 2.952 | 71.19 | 0.2938 | 70.55 | 15.87 | 0.54 |
| Emu2 | 3.059 | 68.51 | 0.3012 | 75.09 | 82.56 | 73.46 |
| TurboEdit | 2.872 | 71.72 | 0.3101 | 72.13 | 78.65 | 59.33 |
| **MoEdit** | **2.749** | **75.66** | **0.3254** | **77.05** | 84.31 | **70.34** |

- MoEdit achieves the best results across all objective metrics (lowest NIQE, highest HyperIQA and CLIP Score).
- MOS (User Satisfaction) of 77.05 is significantly leading.
- In terms of numerical accuracy for 3 objects, MoEdit (84.31) outperforms TurboEdit (78.65).

### Ablation Study

- Removal of FeCom: Severe attribute confusion and degraded editing quality.
- Removal of QTTN: Loss of quantity consistency.
- Different Injection Points: $B_4$ yields the best effect, whereas early injection ($B_1$-$B_3$) restricts editing flexibility, and late injection ($B_5$+) weakens quantity perception.

### Key Findings

1. The attribute interlacing issue of the CLIP encoder can be resolved through a simple text-guided attention compensation mechanism, without needing to replace it with a stronger encoder.
2. The auxiliary-tool-free scheme can match or even outperform methods that utilize masks/LLMs in terms of quantity consistency.
3. The larger the quantity (9+ objects), the more pronounced MoEdit's advantages become—thanks to not relying on one-to-one auxiliary tools.
4. Setting the editing instruction to null-text during training is crucial to prevent interference between the quantity perception module and the textual conditions.

## Highlights & Insights

1. **Precise Problem Definition**: "Quantity perception" reframes the key challenge of multi-object editing into a concrete technical objective, which is much more actionable than generic "multi-object consistency."
2. **Auxiliary-Tool-Free Design**: Completely independent of masks, bboxes, or LLMs, achieving quantity perception solely through internal feature interactions and significantly lowering the barrier to usage.
3. **Clever CLIP Compensation Idea**: Instead of replacing the encoder, text information is used to compensate for its limitations, preserving CLIP's advantages in structure and macro-semantics.
4. **Interesting Finding in Figure 2(b)**: Adding Gaussian noise to CLIP features still preserves clear visual structures, revealing the true origin of attribute interlacing.

## Limitations & Future Work

1. The paper classification under the medical_imaging folder seems to be an error, as it actually belongs to the field of image editing and generation.
2. The LPIPS metric (0.2555) is inferior to some methods (e.g., TurboEdit at 0.1684), suggesting there is still room for improvement in pixel-level fidelity.
3. In scenarios with absolutely no text descriptions ($c_q$ unavailable), the FeCom module may fail.
4. Based on SDXL, the inference speed is limited by diffusion sampling, making it non-real-time.

## Related Work & Insights

- **IP-Adapter/SSR-Encoder**: Extract attributes via independent object queries, but lack overall quantity perception.
- **TurboEdit/Emu2**: Achieve visual quantity consistency via LLM alignment, but are limited to single-object editing.
- **MS-diffusion**: Uses masks to extract object attributes, but requires one-to-one auxiliary tools.
- **Insights**: The weaknesses of encoders can be compensated for through cross-modal compensation (text compensating for vision). This paradigm of "compensation over replacement" is also applicable to other multimodal tasks.

## Rating

⭐⭐⭐⭐ — The problem definition is clear, the auxiliary-tool-free design is highly practical, and the CLIP compensation concept is novel. The shortcomings lie in the relatively low pixel-level fidelity metrics, and the experimental comparisons do not fully cover the latest diffusion editing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[NeurIPS 2025\] Exploring and Leveraging Class Vectors for Classifier Editing](../../NeurIPS2025/medical_imaging/exploring_and_leveraging_class_vectors_for_classifier_editing.md)
- [\[ECCV 2024\] RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](../../ECCV2024/medical_imaging/radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)

</div>

<!-- RELATED:END -->
