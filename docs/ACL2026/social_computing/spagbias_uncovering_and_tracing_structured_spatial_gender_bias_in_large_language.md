---
title: >-
  [Paper Note] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models
description: >-
  [ACL 2026][Social Computing][Spatial Gender Bias] This paper proposes the SPAGBias framework to systematically evaluate gender bias in LLMs within micro-spatial urban contexts for the first time. Through three diagnostic…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Spatial Gender Bias"
  - "LLM Fairness"
  - "Urban Space"
  - "Bias Measurement Framework"
  - "Narrative Analysis"
date: 2026-05-08
content_hash: 6589ae8614694715
---

# SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.14672](https://arxiv.org/abs/2604.14672)  
**Code**: None  
**Area**: Social Computing / AI Safety  
**Keywords**: Spatial Gender Bias, LLM Fairness, Urban Space, Bias Measurement Framework, Narrative Analysis

## TL;DR

This paper proposes the SPAGBias framework to systematically evaluate gender bias in LLMs within micro-spatial urban contexts for the first time. Through three diagnostic layers—explicit bias, probabilistic bias, and constructive bias—it reveals structured spatial-gender association patterns in LLMs and traces the embedding and amplification of these biases throughout the model development lifecycle.

## Background & Motivation

**Background**: LLMs are increasingly applied in fields relying on spatial reasoning, such as urban planning, navigation, and disaster response. Feminist geography has long revealed that space is not a neutral physical construct but a projection of social power and gender norms—kitchens are feminized as places of care, while workplaces and streets are masculinized as domains of authority.

**Limitations of Prior Work**: Extensive research has documented gender bias in LLMs regarding occupational prediction and text generation, but the spatial dimension has been almost entirely overlooked. This gap is critical: spatial bias can distort key decisions, such as healthcare services designed based on male activity patterns restricting women's access to medical resources.

**Key Challenge**: There is no systematic framework to analyze how LLMs encode gender within micro-geographic urban contexts. The traditional public-private space dichotomy is too coarse to capture finer-grained spatial-gender mapping relationships.

**Goal**: To establish the first multi-level framework for measuring spatial gender bias in LLMs and address three core questions: Do LLMs exhibit systematic spatial gender bias? What are the distribution patterns of this bias? How is bias constructed within generated narratives?

**Key Insight**: Drawing from the theoretical foundations of feminist geography, the authors introduce the sociological concept of "gendered space" into NLP bias research, designing a taxonomy covering 62 types of urban micro-spaces.

**Core Idea**: Comprehensively measure spatial gender bias in LLMs through a three-layer diagnosis (explicit, probabilistic, and constructive). The study finds that bias is not a simple public/private binary but a fine-grained micro-spatial mapping, which is embedded and amplified throughout the entire model development process.

## Method

### Overall Architecture

The SPAGBias framework consists of three pillars: (1) A spatial taxonomy of 62 urban micro-spaces (43 public + 19 private), (2) A structured prompt library (containing three prompt types), and (3) A three-layer diagnostic pipeline for quantifying and diagnosing bias. The input resides in LLM responses to spatial-gender-related prompts, and the output consists of multi-dimensional bias measurements and analysis results.

### Key Designs

1.  **Spatial Taxonomy**:
    *   **Function**: Operationalizes "space" into units of analysis, covering the most representative micro-locations in cities.
    *   **Mechanism**: Constructs 62 urban micro-spaces; public spaces cover transportation (bus stops, private cars), leisure (cinemas, playgrounds), commerce (malls, restaurants), and medical (hospitals, clinics); private spaces cover domestic labor (kitchen, laundry room) and recreational spaces (terrace, game room). The classification is based on urban map legends, spatial planning literature, and LLM semantic understanding of spatial terms.
    *   **Design Motivation**: Existing bias research typically remains at a macro level (e.g., country/region), ignoring micro-spatial differences in everyday urban life.

2.  **Structured Prompt Library**:
    *   **Function**: Elicits spatial-gender associations from different linguistic perspectives.
    *   **Mechanism**: Designs three prompt types—Forced-Choice Prompts (FCPrompt) requiring a binary choice between male/female; Single-Gender Prompts (SGPrompt) generating short narratives of a single gender in specific spaces; and Co-existence Prompts (CGPrompt) generating narratives where both genders interact in the same space. Each prompt is sampled repeatedly across the 62 spaces.
    *   **Design Motivation**: A single prompt type cannot capture bias comprehensively—forced choice exposes explicit preferences, while generation tasks reveal deep-seated biases at lexical and semantic role levels.

3.  **Multi-Level Diagnosis**:
    *   **Function**: Captures spatial gender bias from surface to deep levels.
    *   **Mechanism**: The **explicit bias** layer uses repeated sampling and binomial tests to determine if a model significantly prefers a specific gender, quantifying bias intensity with an Entropy Deviation Index (EDI = $1 - H(p)$); the **probabilistic bias** layer analyzes log-probabilities of gender tokens to distinguish between true neutrality and refusal strategies; the **constructive bias** layer analyzes lexical bias (Odds Ratio, OR), semantic role bias (ARG0/ARG1 mapping), and narrative role bias (assignment of Leader, Supporter, Observer, or Dependent roles) in generated narratives.
    *   **Design Motivation**: Surface-level answers may appear falsely neutral due to alignment training; deep dives into probabilities and narratives are required to reveal true bias.

### Experimental Design

Six representative models were evaluated (GPT-3.5-turbo, GPT-4, Llama3-8B-instruct, Qwen2-7B-instruct, Phi-3-mini, Deepseek-llm-7b-chat). Each model was sampled 30 times per space (temperature=1), generating 1,860 data points for explicit bias; log-probabilities were directly extracted for probabilistic bias; and 5,580 narrative texts were generated for constructive bias.

## Key Experimental Results

### Main Results

| Model | No. of Spaces with Sig. Bias (/62) | Bias Ratio | EDI Variance |
| :--- | :--- | :--- | :--- |
| Phi-3 | 62 | 100% | Highest mean, near-zero variance |
| GPT-3.5-turbo | >56 | >90% | Medium |
| Qwen2-7b | >56 | >90% | Medium |
| Llama3-8b | >56 | >90% | Medium |
| GPT-4 | ~47 | ~76% | Lowest (24.78% refusal) |
| Deepseek-7b | 32 | 51.6% | Most balanced |

| Diagnostic Layer | Key Findings |
| :--- | :--- |
| Explicit Bias | All 6 models exhibit statistically significant spatial gender bias. |
| Probabilistic Bias | Only Phi-3 exhibits the traditional "public-private" gender divide. |
| Constructive Bias - Lexical | Male narratives favor cold-toned negative words ("gray", "lonely"), while female narratives favor sensory-rich words. |
| Constructive Bias - Semantic Role | GPT-4 systematically assigns higher agency (ARG0) to males across all spaces (>0.8 vs ~0.5). |
| Constructive Bias - Narrative Role | Private spaces: Male=Leader / Female=Supporter; Public spaces: Pattern reverses. |

### Ablation Study

| Robustness Variable | Average MAE | Level of Impact |
| :--- | :--- | :--- |
| Prompt Format Change | 0.15 (Lowest for GPT-4) | Moderate |
| Option Order Change | Highest MAE | Significant |
| Temperature Change (0/0.5/1) | Low | Low |
| Model Size Change | Low | Low |

### Key Findings

*   **Gender bias is not a simple public-private dichotomy**: Only Phi-3 follows the classic "public=male, private=female" pattern. Most models show fine-grained micro-spatial mapping—males are associated with leisure and autonomous spaces (garage, game room), while females are associated with domestic labor and care spaces (kitchen, nursery).
*   **Bias is embedded throughout the development lifecycle**: Reward models already encode strong stereotypes, instruction tuning only partially corrects them, and pre-training data itself contains spatial-gender co-occurrence imbalances at the corpus level.
*   **Model bias far exceeds real-world distribution**: While the direction is consistent with reality, the magnitude is significantly amplified.
*   **Double failure in downstream tasks**: Bias distorts decisions in urban planning (normative) tasks (GPT-4's OR as low as 0.00), and fails to reflect true distributions in user profiling (descriptive) tasks (accuracy only 5%-20%).

## Highlights & Insights

*   **Pioneering bias research in the spatial dimension**: By combining feminist geography theory with computational analysis, this work opens a new dimension in bias research. The taxonomy of 62 micro-spaces serves as reusable infrastructure.
*   **Exquisite three-layer diagnostic design**: The framework distinguishes between "true neutrality" and "strategic refusal"—while GPT-4 refuses to answer in 24.78% of cases, its internal probability distributions still encode asymmetrical gender associations.
*   **Narrative role analysis uncovers space-dependent gender dynamics**: Private spaces reinforce traditional hierarchies (male dominance), while public spaces show a reversal (females gain narrative prominence). This pattern of space-conditional role assignment is a novel finding.
*   **"Recognize but Restrain" as an ideal model standard**: This can be transferred to other bias domains—models should remain neutral in normative tasks while reflecting real-world distributions in descriptive tasks.

## Limitations & Future Work

*   Spatial vocabulary only covers urban areas, excluding suburban and rural spaces, and lacks finer-grained sub-space divisions (e.g., CEO office vs. employee office).
*   The study only evaluates English text; spatial gender bias patterns may vary across different languages and cultural backgrounds.
*   The design is based on a binary gender paradigm and does not include non-binary gender groups.
*   Bias tracing uses the C4 corpus as a proxy; since it is not the actual training data for all models, the findings reveal trends rather than direct causation.

## Related Work & Insights

*   **vs. Occupational Gender Bias (Bolukbasi et al., 2016)**: Traditional research focuses on occupation-gender associations; this work extends to space-gender associations. Spatial bias is more covert but has a greater impact on applications like urban planning.
*   **vs. Macro-Geographic Bias (Manvi et al., 2024)**: Previous work focused on country/region-level spatial bias; this paper drills down to the urban micro-spatial level, discovering more fine-grained patterns.
*   **vs. Alignment/Debiasing Research**: This paper demonstrates that RLHF and instruction tuning only partially mitigate bias, as core association patterns are already embedded in pre-training data.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic study of spatial gender bias in LLMs with a solid theoretical foundation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, covering six models, three diagnostic layers, robustness analysis, provenance experiments, and downstream verification.
*   Writing Quality: ⭐⭐⭐⭐ Structurally clear, though some sections are slightly verbose.
*   Value: ⭐⭐⭐⭐ Opens a new research direction, though actual debiasing solutions are yet to be proposed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)
- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)

</div>

<!-- RELATED:END -->
