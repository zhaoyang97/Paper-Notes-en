---
title: >-
  [Paper Note] Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] This paper proposes that VLM urban perception benchmarks should possess two key attributes: "reliability-aware" and "negotiated." By utilizing a benchmark comprising 100 Montreal street-view images, 12 community annotators, and 30 measurement dimensions, it reveals that model alignment is positively correlated with ann
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 340e2c615c237413
---
# Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated

**Conference**: ICML2026  
**arXiv**: [2606.00871](https://arxiv.org/abs/2606.00871)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: VLM Evaluation, Urban Perception, Annotator Reliability, Benchmark Negotiation, Street-view Imagery  

## TL;DR
This paper proposes that VLM urban perception benchmarks should possess two key attributes: "reliability-aware" and "negotiated." By utilizing a benchmark comprising 100 Montreal street-view images, 12 community annotators, and 30 measurement dimensions, it reveals that model alignment is positively correlated with annotator consistency and that models exhibit systematic distributional biases compared to humans in subjective evaluation dimensions.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) are increasingly used to generate structured descriptions of street-view images, supporting tasks such as urban auditing, mapping, and public consultation. Existing evaluation benchmarks typically treat annotator consensus as a stable "ground truth" and measure model performance using point-estimate accuracy.

**Limitations of Prior Work**: Urban perception tasks mix observable attributes (e.g., presence of a sidewalk) with evaluative categories (e.g., comfort, sense of safety). Annotations for the latter are inherently distributions rather than deterministic labels. Disagreements and explicit abstentions (Not applicable) exist among different annotators, but traditional benchmarks collapse these signals into a single label, obscuring systematic divergences.

**Key Challenge**: When a benchmark treats annotation disagreement in subjective evaluations as "label noise" rather than meaningful measurement outcomes, model scores reflect both model capability and attributes of the annotation process—confounding the two. in downstream applications like urban governance, this confusion can mislead decision-making.

**Goal**: (1) Report inter-annotator reliability and model alignment simultaneously in evaluations; (2) Treat disagreement and abstention as measurement outcomes rather than errors; (3) Treat label spaces and scoring strategies as artifacts revisable through community negotiation.

**Key Insight**: The authors observe that annotation consistency varies significantly across different dimensions—high for visible physical attributes and low for subjective evaluations. If model alignment co-varies with human reliability, macro-level scores are effectively mixing dimensions of differing measurement quality.

**Core Idea**: Urban perception VLM benchmarks should be "reliability-aware" (reporting annotation reliability to contextualize alignment scores) and "negotiated" (allowing label systems to be revised and versioned by stakeholders).

## Method

### Overall Architecture
The paper does not propose a new model but rather an evaluation framework: construct a small-scale community-annotated benchmark → evaluate 7 VLMs using a fixed zero-shot protocol → calculate both annotation reliability and model alignment → analyze the relationship and distributional biases between the two → propose a mechanism for benchmark negotiation and versioning.

### Key Designs

**1. Reliability-Aware Evaluation Protocol: Reporting annotator consistency alongside alignment scores**

Without distinguishing whether "annotator disagreement is due to the inherent ambiguity of the dimension" or "the model identified it incorrectly," alignment scores remain uninterpretable. This design reports reliability data alongside alignment scores to make these cases distinguishable. Specifically, Krippendorff's $\alpha$ (nominal distance) is used as the reliability measure for each dimension. Accuracy is calculated for single-choice dimensions, and the Jaccard index for multi-choice dimensions. Abstention labels (Not applicable / Cannot judge) are treated as "non-responses" rather than ordinary categories—samples where the consensus is abstention are excluded from accuracy calculations, and cases where both parties result in empty sets after removing abstentions are marked as missing. Consequently, if a dimension has low annotator $\alpha$, the model's accuracy is no longer interpreted as "poor model performance" but as "the consensus label itself is unstable."

**2. Community-Annotated Benchmark Dataset: Making disagreement and abstention "observable"**

Existing large-scale benchmarks sacrifice the observability of annotation reliability for coverage. Once disagreements are collapsed into a single label, they are lost. This paper takes the opposite approach by creating a small, high-quality empirical anchor: 100 Montreal street-view images (50 real, 50 SDXL-generated), annotated in French by 12 annotators from 7 community organizations across 30 dimensions. These cover scene settings, human presence, architectural form, and subjective impressions. Each image receives 1–3 independent annotations (230 total). Following deterministic French-to-English mapping, consensus is built using majority voting for single-choice and a $\geq 50\%$ threshold for multi-choice. Despite the small scale, preserving raw judgments and abstentions allows structural issues ignored by large benchmarks to be exposed.

**3. Benchmark Negotiation and Versioning: Treating benchmarks as revisable artifacts rather than fixed standards**

In urban governance, the choice of label systems directly influences policy. If these labels cannot be questioned or revised by stakeholders, the benchmark becomes an implicit power structure. Thus, the authors define 6 disclosure elements (label specifications, judgment collection, reliability reporting, aggregate scoring, model interface, revision history) to make all benchmark assumptions auditable. Every revision produces a new version of the label space and scoring strategy, tracked via version numbers. Evaluation results from different versions are reported in parallel rather than overwriting each other. This shifts the implicit assumption of "benchmark as a measurement instrument" to "benchmark as a socio-technical artifact carrying value judgments, subject to community negotiation."

### Evaluation Protocol
A deterministic zero-shot evaluation is used: temperature=0, top_p=1. The prompt includes enumerated definitions for all 30 dimensions, requiring a single-line CSV formatted response. Seven VLMs are evaluated: Claude-Sonnet, OpenAI-o4-mini, GPT-4.1, Gemini-2.5-Pro, Grok-2-Vision, Qwen2.5-VL, and LLaMA-4-Maverick.

## Key Experimental Results

### Main Results

| Metric | Minimum | Maximum | Description |
|------|--------|--------|------|
| Macro Alignment Score | 0.16 | 0.31 | Macro average across 30 dimensions (Accuracy + Jaccard) for 7 VLMs |
| Observable vs. Evaluative | — | — | All models scored higher on observable attribute subsets than evaluative ones |
| Photos vs. Synthetic | — | — | All models showed higher alignment on real photos than on synthetic images |

### Dimension-Level Analysis

| Dimension Type | Krippendorff's $\alpha$ Range | Model Alignment Trend | Key Finding |
|----------|------------------------------|----------------|----------|
| Observable Attributes (e.g., Sidewalks, Vegetation) | Higher ($\alpha > 0.4$) | High alignment, ranked top | Low alignment = Model recognition or prompt limits |
| Subjective Evaluations (e.g., Comfort, Impression) | Lower ($\alpha < 0.2$) | Low alignment, ranked bottom | Low alignment likely due to dimension ambiguity |
| Overall Impression Dimension | Low | Significant distributional bias | Models chose "Not applicable" much more often than humans; "Accessible" much less |

### Key Findings
- **Reliability-Alignment Correlation**: Dimension-level human reliability ($\alpha$) correlates positively with average model alignment—dimensions where annotators agree more are also where models perform better. This implies macro scores are essentially mixing dimensions of different "measurement quality."
- **Systematic Differences in Abstention**: In the Overall Impression dimension, several models used "Not applicable" to express uncertainty, whereas humans distinguished between "Cannot judge" and "Not applicable," causing distributional shifts. This indicates the mapping between prompt semantics and human semantics is non-trivial.
- **Domain Shift in Synthetic Imagery**: All models exhibited lower alignment on synthetic rendered scenes than on real photos, supporting stratified reporting when using synthetic data.
- **Structural Consistency Across Models**: The difficulty ranking across dimensions is highly consistent among all 7 models, suggesting that difficulty is primarily determined by the evaluation system (dimension definitions + scoring strategy) rather than model differences.

## Highlights & Insights
- **Benchmarks as "Artifacts" rather than "Facts"**: This is a significant shift in evaluation philosophy—benchmarks are not just tools but socio-technical artifacts carrying value judgments. This perspective is insightful for any ML benchmark involving subjective labels (toxicity detection, medical imaging, content moderation).
- **Reliability Reporting Transforms the Meaning of Scores**: Knowing a dimension has an annotator $\alpha = 0.1$ changes the interpretation of a 30% model accuracy from "the model is poor" to "the target label itself is unstable." This conditional interpretation prevents misleading model rankings.
- **Abstention as a First-Order Signal**: Reporting the frequency of "Not applicable" as an independent metric reveals systematic differences in how models and humans handle uncertainty, which is completely invisible in traditional accuracy metrics.

## Limitations & Future Work
- **Scale Constraints**: A small-scale benchmark with 100 images and 12 annotators cannot support a model ranking with high statistical power; the authors state this is an empirical anchor rather than a representative sample.
- **Geographic and Cultural Limitations**: Annotations come from a specific community in Montreal; perceptual judgments may not be cross-culturally universal.
- **Zero-shot Only**: The study does not cover fine-tuning, multi-turn interaction, or multi-image contexts, limiting a comprehensive assessment of model potential.
- **French-English Mapping Errors**: Deterministic normalization might underestimate scores in edge cases.
- **Future Directions**: Validating the reliability-alignment co-variation on larger benchmarks across more cities; exploring the integration of negotiated versioning into frameworks like VLMEvalKit.

## Related Work & Insights
- Place Pulse 2.0 (Dubey et al., 2016) used pairwise comparisons to quantify urban perception at scale but did not report annotation reliability.
- UrbanCLIP / UrbanVLP extended contrastive learning to urban tasks, increasing the need for evaluation practices.
- CheXpert (Irvin et al., 2019) introduced uncertainty labels and multi-expert references in medical imaging, aligning with this paper's reliability-aware concept.
- The spirit of Datasheets for Datasets (Gebru et al., 2018) continues in the 6 disclosure elements proposed here.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](../../ICLR2026/multimodal_vlm/citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](../../NeurIPS2025/multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[CVPR 2026\] Same or Not? Enhancing Visual Perception in Vision-Language Models](../../CVPR2026/multimodal_vlm/same_or_not_enhancing_visual_perception_in_vision-language_models.md)
- [\[CVPR 2026\] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models](../../CVPR2026/multimodal_vlm/abstract_3d_perception_for_spatial_intelligence_in_vision-language_models.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](../../CVPR2025/multimodal_vlm/taxonomy-aware_evaluation_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
