---
title: >-
  [Paper Note] What Makes a Good Natural Language Prompt?
description: >-
  [ACL 2025][LLM (Other)][prompt quality evaluation] By conducting a meta-analysis of over 150 prompting papers, this work proposes an attribute-centric prompt quality evaluation framework consisting of 6 dimensions and 21 attributes. Empirical experiments on reasoning tasks show that single-attribute enhancement often outperforms multi-attribute combinations, and fine-tuning on attribute-enhanced data can further boost model reasoning capabilities.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "prompt quality evaluation"
  - "meta-analysis"
  - "attribute framework"
  - "cognitive load theory"
  - "instruction tuning"
date: 2026-05-08
content_hash: 75ed33121b765523
---

# What Makes a Good Natural Language Prompt?

**Conference**: ACL 2025  
**arXiv**: [2506.06950](https://arxiv.org/abs/2506.06950)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: prompt quality evaluation, meta-analysis, attribute framework, cognitive load theory, instruction tuning

## TL;DR

By conducting a meta-analysis of over 150 prompting papers, this work proposes an attribute-centric prompt quality evaluation framework consisting of 6 dimensions and 21 attributes. Empirical experiments on reasoning tasks show that single-attribute enhancement often outperforms multi-attribute combinations, and fine-tuning on attribute-enhanced data can further boost model reasoning capabilities.

## Background & Motivation

**Background**: Prompts have become the primary interface for human-LLM interactions, yet a systematic consensus on "what makes a good prompt" is still lacking. Existing studies mostly propose fragmented prompting techniques (e.g., CoT, few-shot), while companies like OpenAI and Anthropic have released practical guides, which are largely piecemeal empirical summaries.

**Limitations of Prior Work**:
1. Lack of a unified attribute-level framework to systematically understand and compare various prompting strategies.
2. Existing evaluations are outcome-centric, focusing solely on task performance metrics while disregarding the quality of the prompt itself.
3. The cross-model and cross-task generalizability of different attributes has not been validated.
4. The interaction effects and combined impacts of multiple attributes have rarely been explored.

**Key Challenge**: Outcome-oriented prompt optimization can produce machine-friendly but human-unintelligible prompts, leading to issues with alignment, transparency, and maintainability. Conversely, while attribute-oriented evaluation is interpretable, it lacks a systematic theoretical framework.

**Goal**: To establish a unified, attribute-centric prompt quality evaluation framework to answer three questions: (1) what attributes should a good prompt possess? (2) how do these attributes affect different models and tasks? (3) is multi-attribute combination or single-attribute enhancement more effective?

**Key Insight**: Drawing upon humanistic and cognitive science theories such as Grice's Maxims, Cognitive Load Theory, and Gagne's Nine Events of Instruction, this work systemizes prompt attributes into a framework and validates it through a meta-analysis and empirical experiments.

**Core Idea**: The quality of natural language prompts can be decomposed into 21 independently evaluable attributes across 6 dimensions. Precisely enhancing a single attribute is often more effective than stacking multiple attributes.

## Method

### Overall Architecture

The study consists of four stages: (1) Literature Meta-analysis: Surveyed 150+ papers from ACL/EMNLP/NAACL/ICLR/NeurIPS (2022-2025) and enterprise blogs to extract prompting recommendations and conceptualize them into attributes -> (2) Attribute Impact Analysis: Analyzed the research distribution and effectiveness of each attribute across different models and tasks -> (3) Attribute Correlation Analysis: Evaluated correlations among the 21 attributes on 969 high-quality prompts -> (4) Reasoning Task Case Study: Validated the effectiveness of single/multi-attribute enhancements in both prompting and fine-tuning settings.

### Key Designs

1. **6-Dimension 21-Attribute Taxonomy**:
    - **Function**: To provide a complete evaluation dimension and actionable attribute definitions for prompt quality.
    - **Mechanism**: The six dimensions are: **I. Communication & Language** (token count, presentation, interactive engagement, politeness), **II. Cognition** (managing intrinsic load, reducing extraneous load, stimulating germane load), **III. Instruction** (goal setting, external tools, metacognition, exemplars, reward mechanism), **IV. Logic & Structure** (structural logic, contextual logic), **V. Hallucination** (hallucination awareness, balance of factuality and creativity), and **VI. Responsibility** (bias, safety, privacy, reliability, social norms).
    - **Design Motivation**: Drawing from mature humanities and cognitive theories such as Grice's Maxims (communication dimension), Sweller's Cognitive Load Theory (cognitive dimension), and Gagne's Nine Events of Instruction (instruction dimension) to ensure the framework is both theoretically grounded and highly actionable.

2. **Attribute Correlation Analysis Method**:
    - **Function**: To reveal co-occurrence and correlation patterns among attributes in high-quality prompts, deriving practical recommendations.
    - **Mechanism**: Collected 969 high-quality prompts (from Prompt Engineering papers, Awesome ChatGPT Prompts, Alpaca, Natural Instructions, etc.), scored each prompt on the 21 attributes (1-10) using GPT-4o + Self-consistency, and calculated the correlation coefficients between attributes. For attribute pairs with an average score of $<5$, correlation analysis was omitted to avoid spurious correlations.
    - **Design Motivation**: To directly analyze the co-occurrence patterns of attributes in human-crafted prompts, providing empirical evidence on "which attributes should be jointly optimized" for prompt optimization.

3. **Comparison Experiments on Single/Multi-Attribute Enhancement**:
    - **Function**: To validate the effectiveness of different prompt attribute enhancement strategies in actual reasoning tasks.
    - **Mechanism**: Using zero-shot CoT as the baseline, four attributes were enhanced by appending simple statements: Politeness (adding "Please"), Germane load (requesting recall of prior knowledge), Metacognition (requesting self-verification), and Rewards (offering a \$100 reward), testing individual and combined effects. Beyond prompting, fine-tuning experiments were conducted: fine-tuning Qwen-2.5-7B-It on the Alpaca-GPT-4o dataset using polite versus original data, respectively.
    - **Design Motivation**: To answer the critical question of "whether more attributes are always better"; the fine-tuning experiments validate whether attribute-enhancement can be internalized by the model.

## Key Experimental Results

### Main Results

**Attribute-Enhanced Prompting Results** (Table 2, accuracy % per task):

| Configuration | MMLU | CommonsenseQA | ARC-C | GSM8K |
|------|:----:|:----:|:----:|:----:|
| **Llama-3.1-8B-It** | | | | |
| Zero-shot CoT | 65.00 | 76.00 | 81.50 | 82.0 |
| + Politeness | **68.00**↑ | **83.50**↑ | **84.50**↑ | **87.5**↑ |
| + Germane load | 66.00↑ | 75.50↓ | 82.00↑ | 82.0 |
| + Metacognition | 61.00↓ | 81.50↑ | 81.00↓ | 81.5↓ |
| + Rewards | 64.00↓ | 80.50↑ | 82.00↑ | 84.0↑ |
| + Pol.+Ger.+Met. | 69.50↑ | 75.00↓ | 82.50↑ | 81.5↓ |
| **Qwen-2.5-7B-It** | | | | |
| Zero-shot CoT | 45.50 | 55.00 | 59.50 | 76.5 |
| + Metacognition | **52.50**↑ | **56.50**↑ | **62.00**↑ | 83.5↑ |
| + Germane load | 44.50↓ | 56.50↑ | 53.50↓ | **90.0**↑ |
| + Politeness | 41.00↓ | 45.50↓ | 54.00↓ | 79.0↑ |
| + Rewards | 40.50↓ | 48.00↓ | 52.00↓ | 66.0↓ |
| **o3-mini** | | | | |
| Zero-shot CoT | 92.00 | 88.50 | 94.50 | 97.0 |
| + Politeness | 88.50↓ | 87.00↓ | 93.50↓ | 96.0↓ |
| + Germane load | 88.00↓ | 82.00↓ | 95.00↑ | 96.5↓ |

### Ablation Study

**Attribute-Enhanced Fine-Tuning Results** (Table 3, Qwen-2.5-7B-It after fine-tuning, politeness data / original data):

| Configuration | MMLU | CQA | ARC | GSM8K | Avg. |
|------|:----:|:----:|:----:|:----:|:----:|
| Zero-shot CoT | 60.0/67.0 | 67.5/69.0 | 73.5/68.5 | 85.0/85.0 | 71.50/72.38 |
| + Politeness | **69.5**/62.5 | **72.5**/70.0 | **85.0**/79.5 | 85.0/88.5 | **78.00**/75.13 |
| + Metacognition | 61.0/54.0 | 72.0/68.0 | 75.0/71.0 | 86.5/89.0 | 73.63/70.50 |
| + Pol.+Ger.+Met. | 69.0/66.5 | **77.5**/**79.5** | **86.5**/**83.5** | 82.5/81.5 | **78.88**/**77.75** |

### Key Findings

1. **Single-attribute enhancement often outperforms multi-attribute combinations**: Politeness is effective for Llama across all 4 tasks ($+3$ to $+7.5\%$), but adding Germane load actually decreases performance on CommonsenseQA from 83.50 to 79.50.
2. **Different models respond very differently to the same attribute**: Politeness is effective across the board for Llama but degrades performance on MMLU/CQA/ARC for Qwen; Metacognition is fully effective for Qwen but drops by $4\%$ on MMLU for Llama.
3. **Strong models are barely affected by attribute enhancement**: o3-mini's performance degrades under all attribute enhancements, likely because its extensive CoT training makes additional attributes drift the prompt away from the training distribution.
4. **Attribute enhancement can be internalized through fine-tuning**: Qwen fine-tuned on polite data improves performance on prompts with "Please" from 45.5 to 69.5 (MMLU), and average performance increases from 71.50 to 78.00, outperforming original data fine-tuning under almost all attribute-enhanced configurations.
5. **Attribute Correlation Analysis**: Out of 210 pairs, 17 strong correlations ($\ge 0.7$) were found across 969 high-quality prompts, such as token count ↔ presentation ↔ structural logic ↔ extraneous load, goal ↔ intrinsic load ↔ germane load, and hallucination awareness ↔ reliability.

## Highlights & Insights

- **Solid Theoretical Grounding**: The framework is built upon theoretical bases such as Grice's Maxims, Cognitive Load Theory, and Gagne's Nine Events of Instruction, rather than arbitrary categorization.
- **Counter-intuitive "Less is More" Finding**: Precisely targeting 1 attribute > stacking multiple attributes, which offers significant guidance for prompt engineering practices.
- **Attribute-to-Model Asymmetry**: Clearly demonstrates that "there is no silver bullet attribute"—different models require different attribute enhancements, echoing the "No Free Lunch" theorem.
- **Synergy of Fine-Tuning and Prompting**: Attribute-enhancement can be used not only at inference time but can also be internalized via fine-tuning, with the two showing superior synergy when combined.
- **Valuable Open Questions**: Proposes 8 open questions (OQ1-OQ8) spanning attribute transferability, causality, task specificity, etc., charting a path for future research.

## Limitations & Future Work

- Although the literature review covers 150+ papers, manual effort was inherently limited in covering all related works.
- Evaluation of the 21 attributes relies on GPT-4o as a judge. Open-source models (e.g., DeepSeek R1, Mistral) only showed a format-following rate of 65-71%, limiting evaluation reliability.
- Multi-attribute combination experiments only utilized the simplest prompt enhancement forms (e.g., adding "Please") without model-specific optimization.
- The responsibility dimension (bias, safety, privacy, social norms, etc.) is overly broad with minimal literature support.
- Correlation analysis was conducted on a single prompt set; correlations may vary across different task scenarios.
- Experimental tasks only covered reasoning categories (MMLU/CommonsenseQA/ARC-C/GSM8K), leaving generation and NLU tasks unvalidated.

## Related Work & Insights

- **vs. Automatic Prompt Optimization (APE/OPRO/RLPrompt)**: Automatic optimization concentrates on searching for the optimal prompt text, whereas this work offers an attribute-level design framework intelligible to humans, making the two approaches complementary.
- **vs. Prompt Analysis (e.g., LLMLingua)**: Existing analyses focus on prompt structural components or compression. This work offers a fresh perspective from the angle of attribute quality.
- **vs. Corporate Prompt Guidelines (OpenAI/Anthropic)**: Practical guides offer specific recommendations (such as "specify output length"), whereas this work abstracts and systematizes them into studyable attributes.
- **Insights**: Prompt design should shift from being empirically driven towards a systematic engineering paradigm of "attribute diagnosis -> precise enhancement"; the collaborative pathway of attribute enhancement combined with fine-tuning warrants further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic framework for prompt attribute classification, with solid theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐ The meta-analysis is broad, but the empirical experiments are restricted to reasoning tasks, and the attribute enhancement methods are overly simplistic.
- Writing Quality: ⭐⭐⭐⭐ The framework exhibits clear hierarchy, and the open questions have profound depth.
- Value: ⭐⭐⭐⭐ Offers direct guidance for both prompt engineering research and practice; the attribute framework can serve as a benchmark for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [\[ACL 2025\] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)

</div>

<!-- RELATED:END -->
