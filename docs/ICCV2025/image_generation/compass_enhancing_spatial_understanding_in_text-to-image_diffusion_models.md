---
title: >-
  [Paper Note] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models
description: >-
  [ICCV 2025][Image Generation][Text-to-image generation] CoMPaSS leverages the SCOP data engine to curate spatially unambiguous training data and introduces the parameter-free TENOR module to inject token ordering information into the attention mechanism, substantially improving spatial relationship generation accuracy in T2I diffusion models (VISOR +98%, GenEval Position +131%).
tags:
  - ICCV 2025
  - Image Generation
  - Text-to-image generation
  - spatial understanding
  - diffusion models
  - data engine
  - token positional encoding
date: 2026-05-08
content_hash: a2f08e127b32af4a
---

# CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models

**Conference**: ICCV 2025  
**arXiv**: [2412.13195](https://arxiv.org/abs/2412.13195)  
**Code**: [https://github.com/blurgyy/CoMPaSS](https://github.com/blurgyy/CoMPaSS)  
**Area**: Image Generation  
**Keywords**: Text-to-image generation, spatial understanding, diffusion models, data engine, token positional encoding

## TL;DR
CoMPaSS leverages the SCOP data engine to curate spatially unambiguous training data and introduces the parameter-free TENOR module to inject token ordering information into the attention mechanism, substantially improving spatial relationship generation accuracy in T2I diffusion models (VISOR +98%, GenEval Position +131%).

## Background & Motivation
Text-to-image diffusion models (e.g., SD, FLUX.1) excel at generating photorealistic images but frequently fail to render accurate spatial relationships (e.g., "to the left of," "above"). The authors identify two root causes:

**Data ambiguity**: Existing datasets (LAION, CC12M, COCO) contain severely ambiguous spatial descriptions — inconsistent viewpoint conventions (viewer-centric vs. object-intrinsic), non-spatial uses of spatial words (e.g., "the right choice"), and missing or incorrect reference objects.

**Spatial understanding deficiencies in text encoders**: Proxy task evaluations reveal that encoders such as CLIP and T5-XXL are nearly incapable of distinguishing semantically equivalent spatial descriptions. For instance, "A left of B" and "B right of A" should yield similar embeddings, yet T5-XXL achieves only 4.84% accuracy and CLIP variants approach 0%.

These two issues compound each other: even if a text encoder correctly encodes spatial relations, data ambiguity still impedes learning; conversely, even with unambiguous data, the encoder fails to transmit the correct signal. CoMPaSS addresses both problems simultaneously.

## Method

### Overall Architecture
CoMPaSS comprises two complementary components: the SCOP data engine (providing high-quality spatial training data) and the TENOR module (ensuring the model can distinguish prompts with different syntactic structures). The two components work synergistically to enhance spatial understanding in T2I models.

### Key Designs

1. **SCOP (Spatial Constraints-Oriented Pairing) Data Engine**:

    - Function: Extracts object pairs with unambiguous spatial relationships from images and pairs them with accurate textual descriptions.
    - Mechanism: A three-stage pipeline:
        - **Relation inference**: Enumerates all $\binom{n}{2}$ candidate object pairs in an image.
        - **Spatial constraint enforcement**: Filters ambiguous pairs via five constraints — visual salience ($\frac{\text{Area}(B_i \cup B_j)}{\text{Area}(I)} > \tau_v$), semantic distinctiveness (different categories), spatial clarity (centroid distance / minimum diagonal $< \tau_u$), minimal overlap (overlap ratio $< \tau_o$), and size balance (area ratio $> \tau_s$).
        - **Relation decoding**: Decodes structured descriptors into image-crop–text-prompt pairs.
    - Design Motivation: Filters the COCO training set to yield 28,000+ object pairs (only 0.004% of LAION-400M), achieving 85.2% human annotation agreement.

2. **TENOR (Token ENcoding ORdering) Module**:

    - Function: Injects token ordering information into the text-image attention layers of the diffusion model.
    - Mechanism:
        - For UNet architectures (SD series): absolute positional encodings are added to the key vectors $K$ in each text-image attention layer.
        - For MMDiT architectures (FLUX.1): positional encodings are added to the text query $Q_{\text{text}}$ and key $K_{\text{text}}$.
    - Design Motivation: Standard Transformer positional encodings are applied only once at the initial embedding stage; ordering information is largely lost after multiple layers of processing. TENOR injects ordering information at every attention operation, ensuring that "A left of B" and "B left of A" produce distinct conditioning signals. It introduces no additional parameters and incurs negligible inference overhead (~2.47% time, ~0.6% VRAM).

3. **Synergy between SCOP and TENOR**:

    - TENOR itself does not encode the semantic meaning of "left/right"; it provides a structural signal that ensures different prompts yield different conditioning.
    - SCOP supplies semantic ground truth (unambiguous spatial data), enabling the model to learn to map structural differences to correct visual outputs.
    - SCOP alone: GenEval Position for SD1.5 improves from 0.04 to 0.39.
    - Adding TENOR: further improves to 0.54.

### Loss & Training
Standard diffusion training loss is used for fine-tuning. Training overhead is minimal (+3.9% time / +0.7% VRAM); no new text encoder training is required. Significant improvements are obtained with as few as 500 images, demonstrating exceptional data efficiency.

## Key Experimental Results

### Main Results

| Model | Benchmark | Metric | Original | +CoMPaSS | Gain |
|-------|-----------|--------|----------|----------|------|
| FLUX.1 | VISOR uncond | Accuracy | 37.96% | 75.17% | +98% |
| FLUX.1 | T2I-CompBench Spatial | Score | 0.18 | 0.30 | +67% |
| FLUX.1 | GenEval Position | Score | 0.26 | 0.60 | +131% |
| FLUX.1 | DPG-Bench Relation | Score | 92.30 | 94.12 | +2% |
| SD1.5 | GenEval Position | Score | 0.04 | 0.54 | +1250% |
| SD2.1 | GenEval Position | Score | 0.07 | 0.51 | +629% |
| FLUX.1 | FID↓ | Image quality | 27.96 | 26.40 | Improved |
| FLUX.1 | CMMD↓ | Image quality | 0.8737 | 0.6859 | Improved |

### Ablation Study

| Configuration | T2I-CompBench Spatial | GenEval Position | Note |
|---------------|-----------------------|-----------------|------|
| SD1.5 baseline | 0.08 | 0.04 | Baseline |
| SD1.5 + SCOP | 0.32 | 0.39 | Large contribution from data engine |
| SD1.5 + SCOP + TENOR | 0.35 | 0.54 | TENOR further improves generalization |
| FLUX.1 baseline | 0.18 | 0.26 | Baseline |
| FLUX.1 + SCOP | 0.29 | 0.56 | Large contribution from data engine |
| FLUX.1 + SCOP + TENOR | 0.30 | 0.60 | Full method achieves best performance |

Data efficiency ablation: using only 500 images, FLUX.1's GenEval Position improves from 0.26 to 0.56, approaching the result of 0.60 obtained with the full 28k dataset.

### Key Findings
- Existing text encoders almost completely fail on spatial semantics: CLIP accuracy ranges from 0% to 0.03%, and T5-XXL reaches only 4.84%.
- Data ambiguity is the primary cause of spatial understanding failure; SCOP alone yields substantial improvements.
- TENOR is critical for generalization to unseen prompts, ensuring that prompts with different syntactic structures produce distinct conditioning signals.
- CoMPaSS not only improves spatial metrics but also enhances overall generation quality and image fidelity.
- Computational overhead is minimal: inference adds only 2.47% in time and 0.6% in VRAM.

## Highlights & Insights
- The paper features rigorous problem analysis: proxy tasks are constructed to quantify spatial understanding deficiencies in text encoders, clearly pinpointing the root causes.
- The SCOP data engine is elegantly designed; five principled constraints effectively filter ambiguous data, requiring only 28k samples.
- TENOR is remarkably simple (parameter-free, near-zero overhead) yet highly effective, exemplifying a targeted, problem-driven design philosophy.
- The method is highly generalizable: it is applicable to both UNet and MMDiT architectures and demonstrates consistent effectiveness across four different models.

## Limitations & Future Work
- The current approach supports spatial relationships between only two objects; extension to complex multi-object scenes remains to be explored.
- SCOP relies on bounding box annotations from COCO, limiting data scale to available annotated datasets.
- Spatial relation types are restricted to left/right/above/below; more complex relations (e.g., "between," "surrounding") are not addressed.
- The positional encoding strategy in TENOR is relatively simple (absolute positional encoding); superior encoding schemes may exist.

## Related Work & Insights
- CoMPaSS and SPRIGHT are both training-based methods, but CoMPaSS substantially outperforms SPRIGHT in both efficiency and effectiveness.
- Inference-only methods (R&B, Attention Refocusing, etc.) require bounding box inputs and incur high computational costs.
- Insight: many failures of T2I models may stem from training data quality rather than insufficient model capacity.

## Rating
- Novelty: ⭐⭐⭐⭐ The analysis of root causes is thorough, and the combined SCOP + TENOR design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 4 models, 4 benchmarks, detailed ablations, data efficiency, and computational efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical flow is clear and coherent, moving seamlessly from problem analysis to the proposed solution.
- Value: ⭐⭐⭐⭐ Presents a systematic solution to spatial understanding in T2I generation with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](../../CVPR2026/image_generation/enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](../../CVPR2026/image_generation/spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)

</div>

<!-- RELATED:END -->
