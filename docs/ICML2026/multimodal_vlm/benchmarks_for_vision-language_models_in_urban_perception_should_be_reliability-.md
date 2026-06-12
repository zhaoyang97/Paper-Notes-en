---
title: >-
  [Paper Note] Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated
description: >-
  [ICML2026][Multimodal VLM][VLM evaluation] This paper proposes that VLM urban perception evaluation should be "reliability-aware" and "negotiable." Through a benchmark featuring 100 Montreal street-view images…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "VLM evaluation"
  - "Urban perception"
  - "Annotator reliability"
  - "Benchmark negotiation"
  - "Street-view imagery"
date: 2026-05-08
content_hash: 30d76a2fe799f850
---

# Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated

**Conference**: ICML2026  
**arXiv**: [2606.00871](https://arxiv.org/abs/2606.00871)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: VLM evaluation, Urban perception, Annotator reliability, Benchmark negotiation, Street-view imagery  

## TL;DR
This paper proposes that VLM urban perception evaluation should be "reliability-aware" and "negotiable." Through a benchmark featuring 100 Montreal street-view images, 12 community annotators, and 30 dimensions, it reveals that model alignment is positively correlated with annotator consistency, and systematic distributional biases exist between models and humans in subjective evaluation dimensions.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) are increasingly used to generate structured descriptions of street-view imagery to support tasks such as urban auditing, mapping, and public consultation. Existing evaluation benchmarks typically treat annotation consensus as a stable "ground truth" and measure model performance using point-estimate accuracy.

**Limitations of Prior Work**: Urban perception tasks mix observable attributes (e.g., presence of a sidewalk) with evaluative categories (e.g., comfort, sense of safety). Annotations for the latter are inherently distributions rather than deterministic labels. While there are disagreements and explicit abstentions (Not applicable) among different annotators, traditional benchmarks flatten these signals into a single label, masking systematic divergence.

**Key Challenge**: When a benchmark treats annotation disagreement in subjective evaluations as "label noise" rather than a meaningful measurement result, model scores actually reflect both model capability and properties of the annotation process—confounding the two. In downstream applications like urban governance, this confusion can mislead decision-making.

**Goal**: (1) Report inter-annotator reliability and model alignment simultaneously during evaluation; (2) Treat disagreement and abstention as measurement outcomes rather than errors; (3) Frame the label space and scoring strategies as artifacts revisable through community negotiation.

**Key Insight**: The authors observe that annotation consistency varies significantly across different dimensions—high for visible physical attributes and low for subjective evaluations. If model alignment co-varies with human reliability, then macro scores are conflating dimensions of different measurement qualities.

**Core Idea**: Urban perception VLM benchmarks should be "reliability-aware" (reporting annotation reliability to qualify the interpretation of alignment scores) and "negotiable" (allowing the label system to be revised and versioned by stakeholders).

## Method

### Overall Architecture
Ours does not propose a new model but rather a comprehensive evaluation framework: building a small-scale benchmark with community labels $\rightarrow$ evaluating 7 VLMs using a fixed zero-shot protocol $\rightarrow$ calculating both annotation reliability and model alignment $\rightarrow$ analyzing the relationship and distributional biases between the two $\rightarrow$ proposing a benchmark negotiation and versioning mechanism.

### Key Designs

1. **Reliability-Aware Evaluation Protocol**:
    - **Function**: Reports inter-annotator reliability per dimension alongside model alignment scores.
    - **Mechanism**: Uses Krippendorff's $\alpha$ (nominal distance) as a dimension-level reliability metric. Accuracy is calculated for single-choice dimensions, and the Jaccard index for multi-choice dimensions. Abstention labels (Not applicable / Cannot judge) are treated as non-responses rather than ordinary labels—samples where the consensus is "abstain" are excluded from accuracy calculations, and cases where both parties are empty after removing abstentions in multi-label settings are recorded as missing.
    - **Design Motivation**: Evaluation scores are uninterpretable without distinguishing between "annotator disagreement due to dimensional ambiguity" and "annotator disagreement due to model misidentification." Reliability data makes these two scenarios distinguishable.

2. **Community-Labeled Benchmark Dataset**:
    - **Function**: Provides an empirical anchor to make disagreement and abstention "observable."
    - **Mechanism**: 100 Montreal street-view images (50 real photos + 50 SDXL-synthesized) annotated by 12 volunteers from 7 community organizations across 30 dimensions (covering scene settings, human presence, architectural form, and subjective impressions). Each image received 1-3 independent annotations (230 total), normalized via deterministic French-to-English mapping, with consensus built via majority vote (single-choice) or a $\geq 50\%$ selection threshold (multi-choice).
    - **Design Motivation**: Large-scale benchmarks often sacrifice the observability of annotation reliability for coverage. A small but meticulously designed benchmark can reveal structural issues overlooked in evaluation.

3. **Benchmark Negotiation and Versioning Mechanism**:
    - **Function**: Treats the benchmark as a revisable artifact rather than a fixed standard.
    - **Mechanism**: Defines 6 disclosure elements (label specifications, judgment collection, reliability reporting, aggregate scoring, model interface, revision logs) to make benchmark assumptions externally auditable. Revisions produce new versions of the label space and scoring strategies, tracked by version numbers. Evaluation results from different versions can be reported in parallel rather than replacing one another.
    - **Design Motivation**: In urban governance, label selection directly affects policy. If the label system cannot be questioned or revised by stakeholders, the benchmark becomes an implicit power structure.

### Evaluation Protocol
A deterministic zero-shot evaluation is employed: temperature=0, top_p=1, with a fixed prompt enumerating 30 dimensions and their definitions, requiring the model to output structured answers in single-line CSV format. Seven VLMs were evaluated: Claude-Sonnet, OpenAI-o4-mini, GPT-4.1, Gemini-2.5-Pro, Grok-2-Vision, Qwen2.5-VL, and LLaMA-4-Maverick.

## Key Experimental Results

### Main Results

| Metric | Minimum | Maximum | Description |
|------|--------|--------|------|
| Macro Alignment Score | 0.16 | 0.31 | Macro average across 30 dimensions for 7 VLMs (Single-choice Accuracy + Multi-choice Jaccard) |
| Observable vs. Evaluative | — | — | All models scored higher on observable attribute subsets than on evaluative subsets |
| Photos vs. Synthetic | — | — | All models showed higher alignment on real photos than on synthetic images |

### Dimension-level Analysis

| Dimension Type | Krippendorff's $\alpha$ Range | Model Alignment Trend | Key Findings |
|----------|------------------------------|----------------|----------|
| Observable Attributes (e.g., sidewalks, vegetation) | Higher ($\alpha > 0.4$) | High alignment, top rankings | Low alignment = model recognition or prompt limits |
| Subjective Evaluation (e.g., comfort, overall impression) | Lower ($\alpha < 0.2$) | Low alignment, bottom rankings | Low alignment may be due to the ambiguity of the dimension itself |
| Overall Impression Dimension | Low | Significant distributional bias | Models chose "Not applicable" much more frequently than annotators; chose "Accessible" much less frequently |

### Key Findings
- **Reliability-Alignment Correlation**: Dimension-level human reliability ($\alpha$) is positively correlated with average model alignment—models perform better in dimensions where annotators are more consistent. This implies macro scores mix dimensions of varying "measurement quality."
- **Systematic Differences in Abstention**: In the "Overall Impression" dimension, several models used "Not applicable" to express uncertainty, whereas annotators distinguished between "Cannot judge" and "Not applicable," causing distributional bias. This indicates the mapping between prompt semantics and human semantics is non-trivial.
- **Synthetic Image Domain Shift**: All models showed lower alignment on synthetically rendered scenes compared to real photos, supporting the need for stratified reporting when using synthetic data.
- **Consistent Structure Across Models**: The ranking of difficulty across dimensions is highly consistent among the 7 models, suggesting that difficulty is primarily determined by the evaluation system (dimension definitions + scoring strategy) rather than model differences.

## Highlights & Insights
- **Viewing Benchmarks as "Artifacts" rather than "Facts"**: This is a significant shift in evaluation philosophy—a benchmark is not a neutral measurement instrument but a socio-technical artifact carrying value judgments that can be questioned and revised. This perspective is insightful for any ML benchmark involving subjective labels (e.g., toxicity detection, medical imaging, content moderation).
- **Reliability Reporting Changes Score Interpretation**: When we know a dimension has an annotator $\alpha = 0.1$, a 30% model accuracy no longer simply means the "model is poor" but rather "the consensus label for this dimension is inherently unstable." This conditional interpretation prevents misleading model rankings.
- **Abstention as a First-Order Signal**: Reporting the frequency of "Not applicable" as an independent metric reveals systematic differences in how models and humans handle uncertainty, which is entirely invisible in traditional accuracy metrics.

## Limitations & Future Work
- **Scale Constraints**: The small-scale benchmark (100 images, 12 annotators) lacks the statistical power for robust model ranking; the authors state this is an empirical anchor rather than a representative sample.
- **Geographic and Cultural Limitations**: Annotations come from specific Montreal communities; perceptual judgments may not possess cross-cultural universality.
- **Zero-shot Only**: The study does not cover fine-tuning, multi-turn interaction, or multi-image context, limiting a comprehensive assessment of model potential.
- **French-English Mapping Errors**: Deterministic normalization may underestimate model scores in edge cases.
- **Future Directions**: Validating whether the reliability-alignment co-variation holds universally across larger, multi-city benchmarks, and exploring the integration of negotiated versioning into evaluation frameworks like VLMEvalKit.

## Related Work & Insights
- Place Pulse 2.0 (Dubey et al., 2016) used large-scale pairwise comparisons to quantify urban perception but did not report annotation reliability.
- UrbanCLIP / UrbanVLP extended contrastive learning to urban tasks, increasing the demand for evaluation practices.
- CheXpert (Irvin et al., 2019) introduced uncertainty labels and multi-expert reference standards in medical imaging, aligning with the reliability-aware philosophy of this work.
- The spirit of "Datasheets for Datasets" (Gebru et al., 2018) is continued in the 6 disclosure elements proposed here.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](../../ICLR2026/multimodal_vlm/citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](../../NeurIPS2025/multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)
- [\[ICML 2026\] Contextualized Visual Personalization in Vision-Language Models](contextualized_visual_personalization_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
