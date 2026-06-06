---
title: >-
  [Paper Note] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models
description: >-
  [ACL 2026][Social Computing][Spatial Gender Bias] This paper proposes SPAGBias, a framework that systematically evaluates gender bias in LLMs within urban micro-spatial contexts through three diagnostic layers—explicit…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Spatial Gender Bias"
  - "LLM Fairness"
  - "Urban Space"
  - "Bias Measurement Framework"
  - "Narrative Analysis"
date: 2025-05-08
content_hash: 2b38da9f9c7c68a2
---

# SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.14672](https://arxiv.org/abs/2604.14672)  
**Code**: N/A  
**Area**: Social Computing / AI Safety  
**Keywords**: Spatial Gender Bias, LLM Fairness, Urban Space, Bias Measurement Framework, Narrative Analysis

## TL;DR

This paper proposes SPAGBias, a framework that systematically evaluates gender bias in LLMs within urban micro-spatial contexts through three diagnostic layers—explicit, probabilistic, and constructive bias—revealing structured spatial-gender association patterns and tracing how bias is embedded and amplified throughout the model development pipeline.

## Background & Motivation

**Background**: LLMs are increasingly deployed in urban planning, navigation, and disaster response—domains that rely on spatial reasoning. Feminist geography has long established that spaces are not neutral physical constructs but projections of social power and gender norms: kitchens are feminized as caregiving spaces while workplaces and streets are masculinized as domains of authority.

**Limitations of Prior Work**: Extensive research has documented gender bias in LLMs for occupational prediction and text generation, yet the spatial dimension remains almost entirely unexplored. This gap is critical: spatial bias can distort key decisions—for example, healthcare services designed around male activity patterns may limit women's access to medical resources.

**Key Challenge**: No systematic framework exists for analyzing how LLMs encode gender in micro-geographic urban contexts. The traditional public–private binary is too coarse to capture fine-grained spatial-gender mappings.

**Goal**: Establish the first multi-level framework for measuring spatial gender bias in LLMs, answering three core questions: Do LLMs exhibit systematic spatial gender bias? What distributional patterns does this bias follow? How is bias constructed in generated narratives?

**Key Insight**: The authors draw on feminist geography to import the concept of "gendered spaces" into NLP bias research, designing a taxonomy covering 62 urban micro-spaces.

**Core Idea**: A three-layer diagnostic (explicit, probabilistic, constructive) comprehensively measures spatial gender bias in LLMs, revealing that bias is not a simple public/private dichotomy but fine-grained micro-spatial mapping that is embedded and amplified throughout the model development pipeline.

## Method

### Overall Architecture

The SPAGBias framework consists of three pillars: (1) a taxonomy of 62 urban micro-spaces (43 public + 19 private), (2) a structured prompt library with three prompt types, and (3) three diagnostic layers for quantifying and diagnosing bias. The input is LLM responses to spatial-gender prompts; the output is multi-dimensional bias measurements and analysis results.

### Key Designs

1. **Spatial Taxonomy**:

    - Function: Operationalizes "space" as an analytical unit covering the most representative micro-locations in urban environments
    - Mechanism: Defines 62 urban micro-spaces. Public spaces cover transportation (bus stops, private cars), leisure (cinemas, sports fields), commercial (malls, restaurants), and healthcare (hospitals, clinics); private spaces cover domestic labor (kitchens, laundry rooms) and leisure (patios, game rooms). The taxonomy is grounded in urban map legends, spatial planning literature, and LLM semantic understanding of spatial terms
    - Design Motivation: Existing bias research typically stays at the macro level (e.g., country/region), overlooking micro-spatial differences in everyday urban life

2. **Structured Prompt Library**:

    - Function: Elicits spatial-gender associations from LLMs through diverse linguistic perspectives
    - Mechanism: Three prompt types are designed—Forced-Choice Prompts (FCPrompt) requiring a binary male/female selection; Single-Gender Prompts (SGPrompt) generating short narratives of a single gender in a specific space; Co-Gender Prompts (CGPrompt) generating narratives of male-female interaction in the same space. Each prompt type is repeatedly sampled across all 62 spaces
    - Design Motivation: A single prompt type cannot fully capture bias—forced choice exposes explicit preferences while generation tasks reveal deeper lexical and semantic-role-level bias

3. **Multi-Level Diagnostic Pipeline**:

    - Function: Captures spatial gender bias comprehensively from surface to depth
    - Mechanism: The explicit bias layer uses repeated sampling and binomial tests to determine whether the model significantly favors one gender, quantified by the Entropy Deviation Index (EDI = $1 - H(p)$). The probabilistic bias layer analyzes model log-probabilities for gender tokens, distinguishing genuine neutrality from refusal strategies. The constructive bias layer analyzes lexical bias (odds ratio OR), semantic role bias (ARG0/ARG1 mapping), and narrative role bias (leader/supporter/observer/dependent role assignment) in generated narratives
    - Design Motivation: Surface-level answers may exhibit false neutrality due to alignment training; probing probability and narrative layers is necessary to reveal genuine bias

### Experimental Setup

Six representative models are evaluated (GPT-3.5-turbo, GPT-4, Llama3-8B-instruct, Qwen2-7B-instruct, Phi-3-mini, Deepseek-llm-7b-chat). Each space is sampled 30 times per model (temperature=1), yielding 1,860 explicit bias data points; probabilistic bias directly extracts log-probabilities; constructive bias yields 5,580 narrative texts.

## Key Experimental Results

### Main Results

| Model | Spaces w/ Significant Bias (/62) | Bias Ratio | EDI Variance |
|-------|----------------------------------|------------|--------------|
| Phi-3 | 62 | 100% | Highest mean, near-zero variance |
| GPT-3.5-turbo | >56 | >90% | Medium |
| Qwen2-7b | >56 | >90% | Medium |
| Llama3-8b | >56 | >90% | Medium |
| GPT-4 | ~47 | ~76% | Lowest (24.78% refusal) |
| Deepseek-7b | 32 | 51.6% | Most balanced |

| Diagnostic Layer | Key Finding |
|-----------------|-------------|
| Explicit Bias | All six models exhibit statistically significant spatial gender bias |
| Probabilistic Bias | Only Phi-3 shows the traditional "public–private" gender split |
| Constructive Bias – Lexical | Male narratives skew toward cold, negative tones ("gray", "lonely"); female narratives skew toward sensory-rich vocabulary |
| Constructive Bias – Semantic Role | GPT-4 systematically assigns higher agency to males across all spaces (>0.8 vs ~0.5) |
| Constructive Bias – Narrative Role | Private spaces: male=leader / female=supporter; public spaces: pattern reverses |

### Ablation Study

| Robustness Variable | Average MAE | Impact Level |
|--------------------|-------------|--------------|
| Prompt format variation | 0.15 (GPT-4 lowest) | Moderate |
| Option order variation | Highest MAE | Significant |
| Temperature variation (0/0.5/1) | Low | Minimal |
| Model scale variation | Low | Minimal |

### Key Findings

- **Gender bias is not a simple public–private dichotomy**: Only Phi-3 exhibits the classic "public=male, private=female" pattern. Most models display fine-grained micro-spatial mappings—males are associated with leisure and autonomous spaces (garages, game rooms) while females are associated with domestic labor and caregiving spaces (kitchens, children's rooms)
- **Bias is embedded throughout the model development pipeline**: Reward models already encode strong stereotypes; instruction tuning only partially corrects them; pre-training data itself contains corpus-level gender-space co-occurrence imbalances
- **Model bias far exceeds real-world distributions**: While directionally consistent, the magnitude is significantly amplified
- **Dual failure in downstream tasks**: In urban planning (normative) tasks, bias distorts decisions (GPT-4's OR as low as 0.00); in user profiling (descriptive) tasks, models fail to reflect real distributions (accuracy only 5%–20%)

## Highlights & Insights

- **Pioneering spatial-dimension bias research**: Combines feminist geography theory with computational analysis, opening a new dimension in bias research. The 62-micro-space taxonomy serves as reusable infrastructure
- **Elegant three-layer diagnostic design**: Distinguishes "genuine neutrality" from "strategic refusal"—GPT-4 refuses to answer in 24.78% of cases, yet its internal probability distribution still encodes asymmetric gender associations
- **Narrative role analysis reveals space-dependent gender dynamics**: Private spaces reinforce traditional hierarchies (male dominance), while public spaces reverse the pattern (female narrative prominence). This spatially conditional role assignment is a novel finding
- **The "recognize but restrain" ideal model standard** is transferable to other bias domains: models should remain neutral in normative tasks and reflect real distributions in descriptive tasks

## Limitations & Future Work

- Spatial vocabulary covers only urban areas, excluding suburban and rural spaces; sub-space granularity is not explored (e.g., CEO office vs. employee office)
- Only English text is evaluated; spatial gender bias patterns may differ across languages and cultures
- Designed on a binary gender paradigm; non-binary gender groups are not covered
- Bias tracing uses the C4 corpus as a proxy, not the actual training data of all models, so findings indicate trends rather than causal relationships

## Related Work & Insights

- **vs Occupational Gender Bias Research (Bolukbasi et al., 2016)**: Traditional bias research focuses on occupation–gender associations; this work extends to space–gender associations. Spatial bias is more covert yet has greater impact on applications such as urban planning
- **vs Macro-Geographic Bias (Manvi et al., 2024)**: Prior work examines country/region-level spatial bias; this paper drills down to urban micro-space level, uncovering finer-grained patterns
- **vs Alignment / Debiasing Research**: This work shows that RLHF and instruction tuning only partially mitigate bias; core association patterns are already embedded in pre-training data

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of spatial gender bias in LLMs with solid theoretical grounding
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six models, three diagnostic layers, robustness analysis, tracing experiments, and downstream validation—very comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure, though some sections are slightly verbose
- Value: ⭐⭐⭐⭐ Opens a new research direction, though concrete debiasing solutions are not yet proposed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)

</div>

<!-- RELATED:END -->
