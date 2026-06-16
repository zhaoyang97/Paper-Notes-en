---
title: >-
  [Paper Note] Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] This paper proposes that VLM evaluations for urban perception should be "reliability-aware" and "negotiable." Through a benchmark involving 100 Montreal street-view images, 12 community annotators, and 30 measurement dimensions, it reveals that model alignment correlates positively with annotator consensus and that sys
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 3a1d9a732470b879
---
# Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated

**Conference**: ICML2026  
**arXiv**: [2606.00871](https://arxiv.org/abs/2606.00871)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: VLM Evaluation, Urban Perception, Annotator Reliability, Benchmark Negotiation, Street-view Imagery  

## TL;DR
This paper proposes that VLM evaluations for urban perception should be "reliability-aware" and "negotiable." Through a benchmark involving 100 Montreal street-view images, 12 community annotators, and 30 measurement dimensions, it reveals that model alignment correlates positively with annotator consensus and that systematic distributional biases exist between models and humans in subjective evaluative dimensions.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) are increasingly used to generate structured descriptions of street-view imagery, supporting tasks such as urban auditing, mapping, and public consultation. Existing evaluation benchmarks typically treat annotation consensus as a stable "ground truth" and measure performance using point-estimate accuracy.

**Limitations of Prior Work**: Urban perception tasks mix observable attributes (e.g., presence of a sidewalk) with evaluative categories (e.g., sense of comfort or safety). Annotations for the latter are inherently distributions rather than definitive labels. Disagreements and explicit abstentions (Not applicable) exist among annotators, but traditional benchmarks flatten these signals into a single label, masking systematic divergence.

**Key Challenge**: When a benchmark treats annotation disagreement in subjective evaluation as "label noise" rather than a meaningful measurement result, model scores actually reflect both model capability and the properties of the annotation process—the two become conflated. In downstream applications like urban governance, this confusion can lead to misleading decisions.

**Goal**: (1) Report both inter-annotator reliability and model alignment during evaluation; (2) Treat disagreement and abstention as results rather than errors; (3) Frame label spaces and scoring strategies as artifacts revisable through community negotiation.

**Key Insight**: The authors observe that annotation consistency varies drastically across dimensions—it is high for observable physical attributes and low for subjective evaluations. If model alignment co-varies with human reliability, macro-scores are effectively mixing dimensions of differing measurement quality.

**Core Idea**: Urban perception VLM benchmarks should be "reliability-aware" (reporting annotation reliability to contextualize alignment scores) and "negotiable" (allowing the label system to be revised and versioned by stakeholders).

## Method

### Overall Architecture
Instead of proposing a new model, this paper introduces an evaluation framework: construct a small-scale community-annotated benchmark → evaluate 7 VLMs using a fixed zero-shot protocol → simultaneously calculate annotation reliability and model alignment → analyze the relationship between the two and distributional biases → propose a benchmark negotiation and versioning mechanism.

### Key Designs

**1. Reliability-aware evaluation protocol: Reporting annotator consistency alongside alignment scores**

If one does not distinguish whether "annotator disagreement is due to dimension ambiguity" or "model failure," alignment scores become uninterpretable. This design reports reliability data alongside alignment scores to make these scenarios distinguishable. Specifically, Krippendorff's $\alpha$ (nominal distance) is used for per-dimension reliability, accuracy is calculated for single-choice dimensions, and the Jaccard index for multi-choice dimensions. Abstention labels (Not applicable / Cannot judge) are treated as "non-responses" rather than normal categories—samples where the consensus is "abstension" are excluded from accuracy calculations, and cases where both parties become empty sets after removing abstentions in multi-label settings are recorded as missing. Consequently, if a dimension has a low annotator $\alpha$, a low model accuracy is no longer interpreted as "poor model performance," but rather as "instability of the consensus label itself."

**2. Community-labeled benchmark dataset: Making disagreement and abstention "observable"**

Existing large-scale benchmarks sacrifice the observability of annotation reliability for coverage; once disagreements are flattened into single labels, they disappear. This paper does the opposite, creating a precise empirical anchor: 100 Montreal street-view images (50 real photos + 50 SDXL-synthesized), annotated in 30 dimensions by 12 annotators from 7 community organizations in French. Dimensions cover scene settings, human presence/activity, architectural form/aesthetics, and subjective impressions. Each image receives 1–3 independent annotations (230 total), normalized via deterministic French-to-English mapping. Consensus is built using majority voting for single-choice and a $\geq 50\%$ threshold for multi-choice. While small, the preservation of raw judgments and abstentions exposes structural issues flattened by larger benchmarks.

**3. Benchmark negotiation and versioning mechanism: Treating benchmarks as revisable artifacts rather than fixed standards**

In urban governance, the choice of label systems directly permeates policy decisions. If labels cannot be questioned or revised by stakeholders, the benchmark silently becomes an implicit power structure. Thus, the paper defines 6 disclosure elements (label specifications, judgment collection, reliability reporting, aggregate scoring, model interface, revision logs) to make benchmark assumptions auditable. Any revision produces a new version of the label space and scoring strategy, tracked by version numbers, with results reported in parallel rather than overwritten. This shifts the implicit premise from "the benchmark is a measurement instrument" to "the benchmark is a sociotechnical artifact carrying values, subject to community negotiation."

### Evaluation Protocol
A deterministic zero-shot evaluation is employed: temperature=0, top_p=1, with a fixed prompt enumerating 30 dimensions and their definitions. Models are required to output structured answers in single-line CSV format. 7 VLMs are evaluated: Claude-Sonnet, OpenAI-o4-mini, GPT-4.1, Gemini-2.5-Pro, Grok-2-Vision, Qwen2.5-VL, and LLaMA-4-Maverick.

## Key Experimental Results

### Main Results

| Metric | Min Value | Max Value | Description |
|------|--------|--------|------|
| Macro Alignment Score | 0.16 | 0.31 | Macro average across 30 dimensions for 7 VLMs (Accuracy + Jaccard) |
| Observable vs. Evaluative | — | — | All models scored higher on the observable attribute subset than the evaluative subset |
| Photos vs. Synthetic | — | — | All models showed higher alignment on real photos than on synthetic images |

### Dimensional Analysis

| Dimension Type | Krippendorff's $\alpha$ Range | Model Alignment Trend | Key Finding |
|----------|------------------------------|----------------|----------|
| Observable Attributes (e.g., sidewalk, vegetation) | High ($\alpha > 0.4$) | High alignment, top rankings | Low alignment indicates model recognition or prompt issues |
| Subjective Evaluation (e.g., comfort, impression) | Low ($\alpha < 0.2$) | Low alignment, bottom rankings | Low alignment likely due to intrinsic dimension ambiguity |
| Overall Impression | Low | Significant distributional bias | Models chose "Not applicable" far more often than humans; "Accessible" far less |

### Key Findings
- **Reliability-Alignment Correlation**: Per-dimension human reliability ($\alpha$) correlates positively with average model alignment—models perform better in dimensions where humans agree more. This implies macro-scores conflate dimensions of varying "measurement quality."
- **Systematic Differences in Abstention**: In the "Overall Impression" dimension, multiple models used "Not applicable" to express uncertainty, whereas human annotators distinguished between "Cannot judge" and "Not applicable," leading to distributional bias. This shows the mapping between prompt semantics and human semantics is non-trivial.
- **Synthetic Image Domain Shift**: All models showed lower alignment on synthetically rendered scenes compared to real photos, supporting the need for stratified reporting when using synthetic data.
- **Cross-Model Structural Consistency**: The difficulty ranking of dimensions was highly consistent across all 7 models, suggesting that difficulty is primarily determined by the evaluation system (definitions + scoring) rather than model differences.

## Highlights & Insights
- **Benchmarks as "Artifacts" rather than "Facts"**: This is a significant philosophical shift in evaluation—benchmarks are not neutral instruments but sociotechnical artifacts carrying value judgments. This perspective is applicable to any ML benchmark involving subjective labels (e.g., toxicity, medical imaging, moderation).
- **Reliability Reporting Changes the Meaning of Scores**: Knowing a dimension has an $\alpha = 0.1$ means a 30% model accuracy no longer implies "the model is bad" but rather "the consensus label itself is unstable." This conditional interpretation avoids misleading model rankings.
- **Abstention as a First-Order Signal**: Reporting the frequency of "Not applicable" as an independent metric reveals systematic differences in uncertainty handling between models and humans, which is invisible in traditional accuracy metrics.

## Limitations & Future Work
- **Scale Constraints**: The small-scale benchmark (100 images, 12 annotators) cannot support high-statistical-power model rankings; the authors state this is an empirical anchor rather than a representative sample.
- **Geographical and Cultural Bias**: Annotations come from a specific community in Montreal; perceptual judgments may not be cross-culturally universal.
- **Zero-shot Only**: The study does not cover fine-tuning, multi-turn interaction, or multi-image context, limiting a full assessment of model potential.
- **French-English Mapping Errors**: Deterministic normalization might underestimate scores in edge cases.
- **Future Directions**: Verifying if the reliability-alignment co-variation holds across larger, more diverse city benchmarks and exploring the integration of versioning into frameworks like VLMEvalKit.

## Related Work & Insights
- Place Pulse 2.0 (Dubey et al., 2016) used pairwise comparisons to quantify urban perception at scale but did not report reliability.
- UrbanCLIP / UrbanVLP extended contrastive learning to urban tasks, increasing the need for evaluation practices.
- CheXpert (Irvin et al., 2019) introduced uncertainty labels and multi-expert reference standards in medical imaging, aligning with the reliability-aware philosophy.
- The spirit of Datasheets for Datasets (Gebru et al., 2018) continues in the 6 disclosure elements proposed here.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](../../ICLR2026/multimodal_vlm/citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](../../NeurIPS2025/multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](../../CVPR2025/multimodal_vlm/taxonomy-aware_evaluation_of_vision-language_models.md)
- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)

</div>

<!-- RELATED:END -->
