---
title: >-
  [Paper Note] Trade-offs in Image Generation: How Do Different Dimensions Interact?
description: >-
  [ICCV 2025][Image Generation][Image generation evaluation] This paper proposes TRIG-Bench, a benchmark comprising 40,200 samples across 10 evaluation dimensions and 132 pairwise dimension subsets, along with a VLM-as-Judge metric termed TRIGScore. It is the first work to systematically reveal and analyze trade-offs among evaluation dimensions (e.g., realism, relation alignment, style) in image generation models, and leverages a Dimension Trade-off Map (DTM) to guide fine-tuni…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image generation evaluation"
  - "trade-off analysis"
  - "multi-dimensional benchmark"
  - "VLM-as-Judge"
  - "text-to-image"
date: 2026-05-08
content_hash: 6f067d04f809ad4e
---

# Trade-offs in Image Generation: How Do Different Dimensions Interact?

**Conference**: ICCV 2025
**arXiv**: [2507.22100](https://arxiv.org/abs/2507.22100)  
**Code**: [https://github.com/fesvhtr/TRIG](https://github.com/fesvhtr/TRIG)  
**Area**: Image Generation
**Keywords**: Image generation evaluation, trade-off analysis, multi-dimensional benchmark, VLM-as-Judge, text-to-image

## TL;DR

This paper proposes TRIG-Bench, a benchmark comprising 40,200 samples across 10 evaluation dimensions and 132 pairwise dimension subsets, along with a VLM-as-Judge metric termed TRIGScore. It is the first work to systematically reveal and analyze trade-offs among evaluation dimensions (e.g., realism, relation alignment, style) in image generation models, and leverages a Dimension Trade-off Map (DTM) to guide fine-tuning for performance improvement.

## Background & Motivation

The performance of text-to-image (T2I) and image-to-image (I2I) models is typically characterized along multiple axes—quality, alignment, diversity, and robustness. However, existing benchmarks and evaluation methodologies suffer from two fundamental shortcomings:

**Limitation 1: Lack of datasets designed to reveal inter-dimension trade-offs.** Existing T2I benchmarks (e.g., HEIM, T2I-CompBench) evaluate multiple dimensions but do not construct prompts that simultaneously target specific dimension pairs. For instance, prompts that jointly probe "style" and "spatial alignment" (e.g., "a cartoon-style painting with a castle to the left of a river") are absent, making it impossible to quantify the interaction between these two dimensions.

**Limitation 2: Single metrics applied across multiple dimensions.** Mainstream benchmarks employ CLIPScore to evaluate diverse dimensions such as alignment and reasoning, leading to metric conflation—improvements in one aspect may conceal degradation in another.

**Key Challenge**: Figure 1 illustrates a concrete example: Janus-Pro exhibits a clear trade-off between "relation alignment" and "realism"—images that correctly express spatial relationships tend to score lower on realism, and vice versa. This type of inter-dimension interaction is entirely overlooked by existing evaluation frameworks.

**Key Insight**: Construct a benchmark dataset specifically designed for pairwise dimension analysis, paired with dimension-specific evaluation metrics, to systematically reveal trade-off patterns across model dimensions.

## Method

### Overall Architecture

The TRIG framework consists of three core components:
1. **TRIG-Bench Dataset**: 40,200 prompts/editing instances covering 3 tasks (T2I, image editing, subject-driven generation), 10 dimensions, and 132 pairwise dimension subsets.
2. **TRIGScore Metric**: A dimension-specific evaluation metric based on a VLM (Qwen2.5-VL).
3. **Trade-off Relationship Identification System**: Classifies dimension pairs into 4 relationship types and constructs a Dimension Trade-off Map (DTM).

### Key Designs

1. **Ten-Dimension Evaluation Framework**

    - **Function**: Define a comprehensive and orthogonal set of evaluation dimensions for image generation.
    - **Core Design**: 10 dimensions organized into 4 categories:
        - Image Quality (IQ): Realism, Originality, Aesthetics
        - Task Alignment (TA): Content, Relation, Style
        - Diversity (D): Knowledge, Ambiguity
        - Robustness (R): Toxicity, Bias
    - **Design Motivation**: Extended from the HEIM benchmark to cover all critical aspects of image generation, with sufficient independence between dimensions to enable meaningful interaction analysis.

2. **Pairwise Dimension Subset Construction**

    - **Function**: Construct dedicated prompt sets for each pair of dimensions.
    - **Mechanism**: All combinations of 10 dimensions are enumerated ($C_{10}^2 = 45$ base pairs, ×3 tasks = 132 subsets). Prompts within each subset are carefully crafted to simultaneously address both target dimensions. For example, a prompt targeting "style + relation" may read: "a watercolor painting of a dog sitting beside a cat."
    - **Annotation Pipeline**: (1) Manual construction of sub-prompt lists per dimension; (2) semi-automatic annotation for T2I tasks (GPT-4o-assisted); (3) GPT-4o generates editing instructions for I2I tasks based on dimension definitions and images; (4) quality control by 10 annotators over two months.
    - **Design Motivation**: Trade-offs between two dimensions can only be observed effectively when prompts simultaneously activate both.

3. **TRIGScore Metric**

    - **Function**: Enable automated, dimension-specific evaluation.
    - **Mechanism**: Employs a VLM (Qwen2.5-VL) as the judge. Rather than relying on unstable numerical text outputs, the score is derived from the logit distribution over predefined rating tokens (Good/Medium/Bad):
      $$\tilde{p}(t) = \frac{\exp(z(t))}{\sum_{t' \in \mathcal{U}} \exp(z(t')) + \epsilon}$$
      A final score $S' = C \cdot S$ is obtained via linear mapping and confidence weighting, where $C = \max_i \tilde{p}(t_i)$.
    - **Design Motivation**: Textual numerical outputs from VLMs are unstable and coarse-grained, whereas logit probability distributions are more stable and informative. The confidence weight $C$ reduces the impact of uncertain model predictions on the final score.

4. **Trade-off Relationship Identification and DTM**

    - **Function**: Automatically identify trade-off types between dimension pairs and visualize them.
    - **Mechanism**: Four relationship types are defined—Synergy (both dimensions simultaneously high), Bottleneck (both simultaneously low), Tilt (one high, one low), and Dispersion (no clear pattern). Automatic classification is performed using Spearman correlation coefficients and thresholded density analysis.
    - **DTM Application**: Identified trade-off patterns guide targeted fine-tuning—weak dimension pairs revealed on the DTM can be addressed through directed data augmentation.

### Loss & Training

- TRIG-Bench is an evaluation benchmark and does not involve model training per se.
- The authors validate a DTM-guided fine-tuning strategy: after identifying Bottleneck dimension pairs, targeted fine-tuning with dimension-relevant data improves weak dimensions without significantly degrading strong ones.
- For example, after DTM-guided fine-tuning, Sana's Bias dimension improves from 0.48 to 0.66, Relation from 0.63 to 0.67, with other dimensions largely unchanged.

## Key Experimental Results

### Main Results (T2I Dimension Scores for 14 Models, TRIGScore)

| Model | Realism | Originality | Aesthetics | Content | Relation | Style | Knowledge | Ambiguity | Toxicity | Bias |
|-------|---------|-------------|------------|---------|----------|-------|-----------|-----------|----------|------|
| DALL·E 3 | 0.70 | **0.82** | **0.80** | **0.77** | **0.75** | **0.80** | **0.66** | **0.67** | 0.48 | **0.91** |
| FLUX | 0.66 | 0.66 | 0.72 | 0.68 | 0.69 | 0.57 | 0.49 | 0.50 | **0.46** | 0.54 |
| SD3.5 | 0.67 | 0.71 | 0.73 | 0.70 | 0.68 | 0.69 | 0.57 | 0.60 | 0.36 | 0.44 |
| Janus-Pro | 0.68 | 0.73 | 0.72 | 0.69 | 0.69 | 0.63 | 0.56 | 0.60 | 0.33 | 0.44 |
| Sana | 0.57 | 0.70 | 0.71 | 0.64 | 0.63 | 0.69 | 0.49 | 0.58 | 0.35 | 0.48 |
| Sana (w/ DTM) | 0.60 | 0.72 | 0.72 | 0.65 | 0.67 | 0.70 | 0.50 | 0.62 | 0.37 | **0.66** |

### Ablation Study (TRIGScore Agreement with Human Evaluation)

| Comparison Axis | TRIGScore–Human Ranking Agreement | CLIPScore Dimension Discriminability | Notes |
|-----------------|-----------------------------------|--------------------------------------|-------|
| Content alignment vs. Style | ✓ Directionally consistent | ✗ Cannot distinguish dimensions | TRIGScore exhibits strong dimension specificity |
| Realism vs. Originality | ✓ Directionally consistent | ✗ Cannot distinguish dimensions | CLIPScore assigns identical scores across dimensions |
| Intra-dimension ranking | ✓ Highly consistent | Partially consistent | Based on 300 samples × 10 annotators |
| Overall correlation | High | Moderate | Validates the advantage of logits-based evaluation |

### Key Findings
- DALL·E 3 leads comprehensively across nearly all dimensions, with Bias (0.91) far exceeding other models, reflecting OpenAI's substantial engineering investment in bias mitigation.
- FLUX achieves the best Toxicity score (0.46) but underperforms on Style (0.57) and Knowledge (0.49).
- DTM-guided fine-tuning is effective: Sana's Bias improves by 37.5% (0.48→0.66) with negligible degradation across other dimensions.
- Trade-off patterns differ by model type: Realism–Originality pairs in T2I models tend toward Synergy, while Relation–Style pairs tend toward Tilt.
- The distribution of the four trade-off relationship types varies with model architecture and training strategy, providing concrete directions for model improvement.

## Highlights & Insights
- This work is the first to systematically study inter-dimension trade-offs in image generation, addressing a notable gap in the evaluation literature.
- The logits-based design of TRIGScore circumvents the instability of VLM text outputs, offering a generalizable VLM-as-Judge paradigm.
- Although constructing 132 pairwise dimension subsets is labor-intensive, it provides the necessary foundation for fine-grained analysis.
- The DTM functions not only as an analytical tool but also directly guides model fine-tuning, forming a closed loop of "evaluation → diagnosis → improvement."
- The four-type trade-off taxonomy (Synergy/Bottleneck/Tilt/Dispersion) provides a clear analytical framework.

## Limitations & Future Work
- TRIGScore relies on a specific VLM (Qwen2.5-VL); substituting alternative VLMs may yield different results.
- Although the 10 dimensions are comprehensive, boundaries for certain dimensions (e.g., Ambiguity, Knowledge) may be ambiguous.
- GPT-4o-assisted dataset annotation may introduce labeling bias.
- The current DTM-guided fine-tuning strategy is relatively straightforward (targeted data augmentation); more sophisticated optimization approaches warrant further exploration.
- The threshold parameters ($\theta_s$, $\theta_b$) for trade-off type classification may influence the resulting relationship assignments.

## Related Work & Insights
- HEIM is the most comprehensive multi-dimensional T2I benchmark, but does not analyze inter-dimension interactions. TRIG-Bench takes a critical step forward in this regard.
- The VLM-as-Judge trend traces a clear trajectory: from CLIPScore (global matching) → human preference alignment (ImageReward) → per-dimension VLM evaluation (TRIGScore).
- The trade-off analysis paradigm introduced here can be generalized to multi-dimensional evaluation in other generation tasks (text generation, video generation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](../../CVPR2025/image_generation/conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)
- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](../../CVPR2025/image_generation/enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[ICCV 2025\] Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts](erasing_more_than_intended_how_concept_erasure_degrades_the_generation_of_non-ta.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
