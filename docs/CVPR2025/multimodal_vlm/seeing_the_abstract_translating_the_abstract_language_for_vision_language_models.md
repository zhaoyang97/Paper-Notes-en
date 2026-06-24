---
title: >-
  [Paper Note] Seeing the Abstract: Translating the Abstract Language for Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Abstract language understanding] Proposes ACT (Abstract-to-Concrete Translator), which analyzes the representation discrepancy between abstract and concrete texts in the VLM latent space via PCA. During inference, ACT shifts the representation of abstract descriptions towards the concrete direction in a training-free manner, mitigating the VLM's insufficient understanding of abstract language and significantly outperforming fine-tuned models on tex…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Abstract language understanding"
  - "fashion retrieval"
  - "representation shift"
  - "training-free method"
  - "VLM alignment"
date: 2026-05-08
content_hash: e0baf91d3621605e
---

# Seeing the Abstract: Translating the Abstract Language for Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2505.03242](https://arxiv.org/abs/2505.03242)  
**Code**: [https://github.com/davidetalon/fashionact](https://github.com/davidetalon/fashionact)  
**Area**: Multimodal VLMs  
**Keywords**: Abstract language understanding, fashion retrieval, representation shift, training-free method, VLM alignment

## TL;DR

Proposes ACT (Abstract-to-Concrete Translator), which analyzes the representation discrepancy between abstract and concrete texts in the VLM latent space via PCA. During inference, ACT shifts the representation of abstract descriptions towards the concrete direction in a training-free manner, mitigating the VLM's insufficient understanding of abstract language and significantly outperforming fine-tuned models on text-to-image retrieval tasks in the fashion domain.

## Background & Motivation

Natural language contains a large number of abstract concepts (e.g., "sexy", "classic", "casual"). Especially in the fashion domain, the frequency of abstract adjectives in seller descriptions is comparable to or even higher than that of concrete adjectives. However, pre-training data (e.g., LAION-400M) for existing VLMs (such as CLIP, SigLIP) predominantly features concrete descriptions, leading to a severe under-representation of abstract language.

The authors discover four key facts through statistical analysis:
1. **Fashion descriptions are inherently abstract-oriented**: In DeepFashion, abstract adjectives appear 14,045 times vs. 10,905 concrete ones.
2. **Abstract attributes carry novel information**: MCC correlation analysis shows most abstract attributes have low correlation with concrete ones.
3. **Abstract attributes aid retrieval**: In an oracle retrieval system, the precision of abstract descriptions is consistently higher than that of concrete ones.
4. **Current VLMs under-represent abstract language**: Retrieval performance is significantly better when using concrete descriptions compared to abstract ones.

While fine-tuning is an intuitive solution, it is constrained by the limited scale of abstract-oriented fashion datasets (FACAD is only ~100K), making large-scale abstract data nearly impossible to acquire. Therefore, a training-free method is required to bridge the representation shift.

## Method

### Overall Architecture

ACT consists of two phases: a **preparation phase** (one-time offline) to extract the abstract-to-concrete representation shift direction, and an **inference phase** that corrects the abstract query toward the concrete direction using LLM rewriting combined with representation shift enhancement.

### Key Designs

1. **A-C Database Construction**:
    - Function: Create paired abstract-concrete text descriptions for each fashion image.
    - Mechanism: Utilize the abstract description $d_s^A$ from the original dataset, and generate a concrete description $d_s^C = \psi(x_s, p_v)$ for the same image using a frozen image captioning model (Qwen2-VL-7B) to construct a paired database $S^{\text{A-C}} = \{(d_s^A, d_s^C)\}$.
    - Design Motivation: Image captioning models naturally tend to generate concrete descriptions (validated in Tab.2), making them ideal proxies for concrete descriptions without requiring human annotation.

2. **A-C Representation Shift Analysis**:
    - Function: Extract the primary shift direction from abstract to concrete in the latent space of the VLM.
    - Mechanism: Encode the paired descriptions using the VLM text encoder $f_T$ to obtain $H^A$ and $H^C$, compute the discrepancy $\Delta^{\text{A-C}} = H^C - H^A$, project it using PCA after normalization $W = \text{PCA}(\Delta^{\text{A-C}}, k)$, and obtain the shift projector $W \in \mathbb{R}^{l \times k}$.
    - Design Motivation: The principal components extracted by PCA capture the primary directions of missing information in the abstract representations relative to the concrete ones. Keeping $k=600$ principal components is an empirical choice.

3. **Inference-time Dual Correction**:
    - Function: Shift user abstract queries toward the concrete direction in the latent space of the VLM.
    - Mechanism: First, rewrite the abstract description into a more concrete version $q' = g(q^A, p_r)$ using an LLM (Llama-3.1-8B); second, project the rewritten text representation $h_{q'}$ onto the shift direction and add it back: $h_{q'}^{\Delta} = N(h_{q'})WW^T \cdot \sigma_\Delta + \mu_\Delta$, resulting in $\hat{h}_q^C = h_{q'} + h_{q'}^{\Delta}$.
    - Design Motivation: LLM rewriting can only partially solve the issue of abstract words (the rewritten text may still contain abstract terms); hence, it is complemented by explicit shift in the latent space. Combining both yields the best performance (confirmed by the ablation study).

### Loss & Training

ACT is a **strictly training-free** method. The preparation phase only requires a one-time PCA calculation, and the inference phase only involves LLM rewriting and matrix multiplication, incurring extremely low computational cost. No gradient updates or backpropagation are required.

## Key Experimental Results

### Main Results

| Dataset | Metric | ACT-df | SigLIP-ft-df | Gain |
|--------|------|--------|-------------|------|
| DeepFashion (In-domain) | H@1 | .437 | .417 | +2.0% |
| DeepFashion (In-domain) | R@1 | .089 | .083 | +0.6% |
| DeepFashion (In-domain) | H@5 | .665 | .659 | +0.6% |

| Dataset | Metric | ACT-facad | SigLIP-ft-facad | Gain |
|--------|------|-----------|----------------|------|
| DeepFashion (Cross-domain) | H@1 | .428 | .352 | +7.6% |
| DeepFashion (Cross-domain) | R@5 | .302 | .259 | +4.3% |
| DeepFashion (Cross-domain) | H@5 | .661 | .568 | +9.3% |

### Ablation Study

| Configuration | R@5 | H@1 | Description |
|------|-----|-----|------|
| No enhancement | .228 | .311 | SigLIP baseline |
| Language Rewriting only | .294 | .411 | LLM rewriting yields +10% H@1 |
| Representation Shift only | .246 | .347 | Representation shift contributes +3.7% H@1 |
| ACT Full Version | .303 | .437 | Combining both yields best performance |
| CogVLM2 replacing Qwen2-VL | .302 | .433 | Robust to captioning model selection |
| Replacing text shift with image embeddings | .266 | .406 | Modality gap leads to performance degradation |

### Key Findings

- ACT improves up to +12.6% H@1 compared to zero-shot SigLIP, and consistently outperforms fine-tuned models.
- Extremely strong cross-dataset generalization: ACT-facad on DeepFashion performs nearly on par with ACT-df, whereas fine-tuned models suffer significant cross-domain degradation.
- ACT is consistently effective across different VLM families and model scales (SigLIP, CLIP, O-CLIP, EVA-CLIP, from ViT-B to ViT-H), with an average H@1 improvement of +4.9%.

## Highlights & Insights

- **Valuable problem identification**: Systematically reveals the "concrete bias" in VLM pre-training data, where abstract language is heavily neglected in existing models.
- **Elegant and simple approach**: The entire pipeline requires zero training, utilizing only PCA and LLM inference, making it plug-and-play.
- **Cross-domain generalization outperforms fine-tuning**: The training-free approach significantly outperforms fine-tuned models in cross-domain settings, showing that the representation shift is a fundamental problem rather than something easily addressed by fine-tuning.
- **Interpretability of PCA shift directions**: The shift directions capture systematic patterns of "missing concrete information" rather than random noise.

## Limitations & Future Work

- Only validated in the fashion domain; other domains dense with abstract language (e.g., art, movie reviews) remain unexplored.
- Abstract word identification is based only on adjectives, overlooking abstract concepts in verbs and nouns.
- The number of PCA components $k=600$ is empirically chosen, lacking sensitivity analysis.
- Reliance on external LLMs for rewriting increases inference latency.

## Related Work & Insights

- The representation shift concept is similar to alignment in concept bottleneck models, but is addressed here using PCA rather than contrastive learning.
- Provides a valuable analysis framework for the gap between "human natural language vs. model-friendly language" in multimodal retrieval.
- Inspiration: Professional terminology in other domains (e.g., medical, legal) may suffer from similar representation shift issues.

## Rating

- Novelty: ⭐⭐⭐⭐ Excellent problem identification; the solution is clever but technically conventional (PCA+LLM).
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive in-domain/cross-domain, multi-model, and ablation experiments, but limited to the fashion domain.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative in the first half data analysis, and concise description of the methodology.
- Value: ⭐⭐⭐⭐ Unveils an important blind spot of VLMs; the plug-and-play solution is highly valuable for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models](../../CVPR2026/multimodal_vlm/abstract_3d_perception_for_spatial_intelligence_in_vision-language_models.md)
- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](taxonomy-aware_evaluation_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
