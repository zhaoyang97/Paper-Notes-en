---
title: >-
  [Paper Note] Benchmarking Overton Pluralism in LLMs
description: >-
  [ICLR 2026][LLM Evaluation][Overton pluralism] This paper proposes the OvertonBench framework, which formalizes Overton pluralism as a set-coverage metric called OvertonScore through a large-scale human study (1,208 demographically representative U.S. participants, 60 subjective questions, 8 LLMs). All evaluated models score only 0.35–0.41 (theoretical maximum: 1.0), and an automated evaluation tool achieving high correlation with human judgments (ρ=0.88) is constructed.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Overton pluralism
  - LLM bias
  - benchmark
  - viewpoint coverage
  - automated evaluation
date: 2026-05-08
content_hash: ea95dbc22350ed2b
---

# Benchmarking Overton Pluralism in LLMs

**Conference**: ICLR 2026
**arXiv**: [2512.01351](https://arxiv.org/abs/2512.01351)
**Code**: [https://github.com/elinorpd/overtonbench](https://github.com/elinorpd/overtonbench)
**Area**: Human Understanding / LLM Alignment / Pluralistic Representation
**Keywords**: Overton pluralism, LLM bias, benchmark, viewpoint coverage, automated evaluation

## TL;DR
This paper proposes the OvertonBench framework, which formalizes Overton pluralism as a set-coverage metric called OvertonScore through a large-scale human study (1,208 demographically representative U.S. participants, 60 subjective questions, 8 LLMs). All evaluated models score only 0.35–0.41 (theoretical maximum: 1.0), and an automated evaluation tool achieving high correlation with human judgments (ρ=0.88) is constructed.

## Background & Motivation

**State of the Field**: LLMs have broadly influenced political discourse, education, and everyday interaction. Conventional alignment strategies typically aggregate diverse preferences, compressing genuine disagreement into a single normative position (value monism), thereby erasing minority viewpoints.

**Limitations of Prior Work**:
   - Existing political bias evaluations (e.g., Model Slant) measure only whether a model leans toward a particular side, and cannot quantify whether the model covers a plurality of viewpoints.
   - Ostensibly "neutral" responses may achieve neutrality by omitting minority perspectives, thereby exacerbating representational harm.
   - Pursuing political neutrality has been shown to be both impossible and not always desirable.

**Root Cause**: Rather than seeking consensus, LLMs should present the range of reasonable viewpoints within the "Overton window" of public discourse; yet no systematic metric exists to measure model performance in this regard.

**Paper Goals**:
   - How should Overton pluralism be defined and quantified?
   - How well do current LLMs represent a plurality of viewpoints?
   - How can scalable evaluation be conducted without repeated expensive human studies?

**Starting Point**: Building on Sorensen et al.'s three-tier taxonomy of pluralism (Overton, steerable, distributional), this work focuses on the most practically relevant tier—Overton pluralism—whereby a model should present multiple reasonable viewpoints within a single response.

**Core Idea**: Pluralistic alignment is reframed from a normative goal into a measurable set-coverage benchmark; opinion clusters are discovered via participant-based clustering, and model response coverage across clusters is then assessed.

## Method

### Overall Architecture
The input is a set of 60 subjective questions; the output is an OvertonScore for each LLM. Three stages are involved: (1) human data collection—participants write responses and rate LLM replies; (2) opinion clustering—distinct viewpoint clusters are discovered from pairwise agreement/disagreement voting patterns; (3) coverage computation—each opinion cluster is assessed for whether it feels represented in the model's response.

### Key Designs

1. **OvertonScore Metric**:

    - Function: Quantifies what fraction of the Overton window is covered by a model's response.
    - Mechanism: For question $x$, the Overton window $W(x)$ contains all reasonable viewpoints. Viewpoint $y$ is considered covered if a majority of participants in the corresponding cluster rate the model's representativeness ≥4 on a 5-point scale. Coverage$(\\mathcal{M}, x) = \\frac{1}{|W(x)|} \\sum_{y \\in W(x)} \\mathbb{1}\\{y \\in \\mathcal{M}(x)\\}$; OvertonScore is the average Coverage across all questions.
    - Design Motivation: Unlike pairwise comparisons that can only assert "A is more pluralistic than B," set coverage provides an absolute quantification with a well-defined theoretical maximum (1.0), making the direction of improvement measurable.
    - Weighted variant OvertonScore$_W$: Weights clusters by their proportion in the population, preventing rare long-tail viewpoints from disproportionately influencing the score.

2. **Vote-Based Opinion Clustering**:

    - Function: Automatically discovers distinct opinion clusters from pairwise participant voting data.
    - Mechanism: Participants vote Agree/Disagree/Neutral on one another's free-text responses; a k-means variant is applied to sparse voting data, with the optimal $k$ determined dynamically via Silhouette scores.
    - Design Motivation: More faithful than clustering based on semantic similarity or NLI—it directly reflects how people interpret and disagree with each other's views, rather than imposing externally derived categories—and avoids introducing model bias through NLP pipelines.

3. **Automated Benchmark (LLM-as-Judge)**:

    - Function: Replaces human raters with an LLM to predict participant representativeness ratings of model responses.
    - Mechanism: Gemini 2.5 Pro is used with a few-shot + free-response (FS+FR) prompting strategy to predict each participant's 1–5 Likert rating.
    - Design Motivation: Repeated large-scale human studies are costly and slow. Automated evaluation serves as a preliminary screening tool during model development, narrowing the candidate pool before comprehensive human evaluation.

### Data Collection Strategy
- Question sources: Model Slant (15 political topics) + PRISM alignment dataset (45 value-oriented questions).
- Participants: 1,208 U.S. English-speaking users recruited via Prolific, representative across political and demographic dimensions.
- Evaluated LLMs: GPT-4.1, o4-mini, Gemma 3-27B, DeepSeek R1/V3, Llama 4 Maverick/3.3-70B, Claude 3.7 Sonnet.
- Dataset scale: 28,992 data points.

## Key Experimental Results

### Main Results

| Model | Adj. OvertonScore | Adj. OvertonScore$_W$ | Significance |
|------|------------------|----------------------|--------|
| DeepSeek V3 | 0.41 (highest) | 0.52 (highest, p=0.035) | Weighted score significantly above mean |
| DeepSeek R1 | 0.40 | 0.49 | Not significant |
| Llama 3.3-70B | 0.40 | 0.49 | Not significant |
| GPT-4.1 | 0.40 | 0.49 | Not significant |
| o4-mini | 0.39 | 0.48 | Not significant |
| Claude 3.7 Sonnet | 0.38 | 0.47 | Not significant |
| Llama 4 Maverick | 0.38 | 0.47 | Not significant |
| Gemma 3-27B | 0.35 (lowest, p=0.016) | 0.44 (lowest, p=0.036) | Significantly below mean on both metrics |
| *Cross-model best* | *0.687* | *0.768* | *Best results combined across all eight models* |
| *Single-opinion baseline* | *0.169* | *0.524* | *Only one cluster covered per question* |

### Automated Evaluation Validation

| Evaluation Method | MAE (Likert) | Spearman ρ | Notes |
|---------|-------------|-----------|------|
| Gemini 2.5 Pro (FS+FR) | 0.66±0.01 | 0.66 | Best automated method |
| Mean-of-others baseline | 0.70±0.01 | 0.64 | Average score from other responses |
| Semantic similarity baseline | 0.72±0.02 | 0.59 | Cosine similarity matching |
| Leave-one-out OvertonScore | — | 0.88 (rank) | Model-level rank correlation |

### Key Findings
- All models score far below the theoretical maximum of 1.0 (mean: 0.39); even combining the best results across all models yields only 0.687.
- DeepSeek V3 performs best on the full benchmark but worst on the Model Slant subset—pluralism is not a single unified capability but is domain-dependent.
- **Political neutrality ≠ pluralistic representation**: o4-mini is rated the second most politically biased model by Model Slant, yet performs well on OvertonScore (r=−0.41 negative correlation).
- Llama 3.3 outperforms Llama 4 on both subsets, casting doubt on the practical effectiveness of political bias mitigation efforts for pluralistic representation.
- The automated benchmark shows no significant gender or racial fairness disparities, though small significant differences exist for political orientation and model identity (effect size η²<0.004).

## Highlights & Insights
- **The set-coverage formalization of OvertonScore** is the paper's most important contribution—it transforms the vague notion of "pluralism" into a quantifiable metric between 0 and 1 with a well-defined theoretical maximum. This is more informative than pairwise comparisons because it measures absolute coverage rather than relative superiority.
- **Vote-based participant clustering** cleverly sidesteps bias introduced by NLP pipelines—real human disagreement patterns define the opinion clusters, rather than having an algorithm presuppose what constitutes a "distinct viewpoint."
- **The negative correlation between political neutrality and pluralism** carries far-reaching implications, suggesting that the industry's current pursuit of "neutrality" may be counterproductive and actually reduce viewpoint coverage. This insight is transferable to any AI alignment research involving subjective values.

## Limitations & Future Work
- Coverage is limited to U.S. English-speaking participants, which cannot represent Overton windows across global cultural contexts.
- The 60 questions provide limited coverage and do not address emerging topics such as technology ethics or environmental justice.
- Opinion clustering relies on k-means, which may fail to capture nuanced differences along continuous spectrums of opinion.
- Claude 3.7 Sonnet is systematically overestimated in automated evaluation (Δ=+0.103), indicating that automated scores for certain models still require calibration.
- The paper does not explore how to actually improve OvertonScore—it provides a measurement tool but not an improvement methodology.
- **Potential direction**: An RLHF reward signal based on OvertonScore could be designed to guide models to proactively present diverse viewpoints in their responses.

## Related Work & Insights
- **vs. Model Slant (Westwood et al., 2025)**: Model Slant measures a model's political leaning (binary bias), whereas this paper measures pluralistic viewpoint coverage. The two dimensions are distinct; this paper finds a negative correlation between them—neutrality does not equal pluralism.
- **vs. Modular Pluralism (Feng et al., 2024)**: Modular Pluralism detects values via NLI and performs pairwise comparison but does not directly estimate the Overton window; this paper uses real human opinion clusters for set-coverage computation, making it more grounded in human judgment.
- **vs. GlobalOpinionQA (Durmus et al., 2024)**: That work evaluates whether LLMs reproduce the response distributions of specific populations; this paper evaluates whether a single response simultaneously covers multiple viewpoints—a different definition and measurement objective.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing pluralism as a quantifiable benchmark is a significant contribution, though the core techniques (clustering + coverage) are not themselves complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ A large-scale human study with 1,208 participants, 8 LLMs, automated validation, subgroup fairness analysis, and comparison across two dataset subsets—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured with rigorous definitions and informative figures (Figure 1 in particular intuitively illustrates the OvertonScore computation pipeline).
- Value: ⭐⭐⭐⭐ Provides the first quantifiable benchmark for pluralistic LLM alignment research; the discovered negative correlation has policy implications.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Benchmarking LLMs for Political Science: A United Nations Perspective](../../AAAI2026/llm_evaluation/benchmarking_llms_for_political_science_a_united_nations_perspective.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](../../ACL2026/llm_evaluation/bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](../../ACL2026/llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ICLR 2026\] RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](../../ACL2026/llm_evaluation/do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)

<!-- RELATED:END -->
