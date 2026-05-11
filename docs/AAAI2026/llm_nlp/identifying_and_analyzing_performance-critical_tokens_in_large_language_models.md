---
title: >-
  [Paper Note] Identifying and Analyzing Performance-Critical Tokens in Large Language Models
description: >-
  [AAAI 2026][LLM/NLP][In-context learning] Through representation-level and token-level ablation experiments, this paper identifies the "performance-critical tokens" that LLMs directly rely on during ICL as template and s…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "In-context learning"
  - "performance-critical tokens"
  - "attention ablation"
  - "template tokens"
  - "information aggregation"
date: 2026-05-08
content_hash: a623910ce7d1eeeb
---

# Identifying and Analyzing Performance-Critical Tokens in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2401.11323](https://arxiv.org/abs/2401.11323)
**Code**: [https://github.com/ybai-nlp/PCT_ICL](https://github.com/ybai-nlp/PCT_ICL)
**Area**: LLM/NLP
**Keywords**: In-context learning, performance-critical tokens, attention ablation, template tokens, information aggregation

## TL;DR
Through representation-level and token-level ablation experiments, this paper identifies the "performance-critical tokens" that LLMs directly rely on during ICL as template and stopword tokens (e.g., "Answer:"), rather than the content tokens that humans would attend to (e.g., actual text). It further reveals that LLMs indirectly exploit content by aggregating content information into the representations of these critical tokens.

## Background & Motivation
**Background**: ICL has become the dominant few-shot learning paradigm for LLMs, yet how LLMs learn and generalize from demonstrations remains poorly understood. Prior work has shown that ICL is highly sensitive to minor prompt variations such as demonstration order.

**Limitations of Prior Work**: Existing studies either focus solely on the last token (function vectors) or on label words, lacking a systematic investigation of the roles of **all tokens** in the prompt.

**Key Challenge**: Humans focus on "content words" (e.g., nouns and adjectives) when learning by analogy — but do LLMs behave similarly?

**Goal**: To systematically identify which tokens' representations in ICL prompts directly affect performance (i.e., performance-critical tokens) and to analyze their characteristics.

**Key Insight**: Tokens in an ICL prompt are categorized into three types (content / stopword / template), and the impact on performance is measured by ablating the representations of each category.

**Core Idea**: LLMs do not directly rely on the representations of content tokens; instead, they rely on template and stopword tokens — which aggregate information from the former.

## Method

### Overall Architecture
ICL prompt tokens are divided into three categories, and two types of ablation experiments are designed to systematically evaluate multiple LLMs and datasets:
- **Token Categorization**: Template (e.g., "Article:", "Answer:"), Stopword (punctuation and function words), Content (actual textual content).
- **Representation-level ablation**: Masks the representations of specific token types from the attention of the test example.
- **Token-level ablation**: Directly removes specific token types from the prompt.

### Key Designs

1. **Representation-Level Ablation**:

    - **Function**: Tests which token representations are directly attended to by the test example.
    - **Mechanism**: Modifies the attention mask so that the test example can only attend to representations of specific token types, then measures the change in classification accuracy. If retaining only template and stopword representations is sufficient to maintain performance, these are the performance-critical tokens.
    - **Key Findings**: **Masking content tokens barely affects performance**, whereas masking template/stopword tokens causes a substantial performance drop, indicating that LLMs directly draw task information from template/stopword representations.

2. **Token-Level Ablation**:

    - **Function**: Tests which token types are critical to the overall information flow of the prompt.
    - **Mechanism**: Directly removes specific token types from the prompt. If removing content tokens causes a large performance drop (even though representation-level ablation shows they are not directly relied upon), this implies that content information is indirectly transmitted.
    - **Key Findings**: **Removing content tokens does substantially degrade performance** — an apparent contradiction with the representation-level results. The explanation is that LLMs aggregate content token information into the representations of template/stopword tokens.

3. **Three Characteristics of Performance-Critical Tokens**:

    - **Lexical meaning**: Tokens semantically related to the task (e.g., "Answer:" in a classification task) are more likely to be performance-critical.
    - **Repetitiveness**: Tokens that recur throughout the prompt (e.g., template tokens present in every demonstration) are more likely to be performance-critical.
    - **Structural cues**: Tokens that mark demonstration boundaries (e.g., newlines, separators) provide critical structural information.

### Loss & Training
This is a purely analytical study with no training involved. Models used include Llama 3B–33B, Llama 2, Mistral 7B, and Gemma 3 4B. Experiments cover 6 classification datasets, each evaluated with 15 seeds × 500 test examples.

## Key Experimental Results

### Main Results — Representation-Level Ablation

Masking a specific token type's representations from Standard ICL (affecting only the test example's attention) and measuring accuracy:

| Setting | AGNews | SST2 | DBPedia | Avg. △ |
|---------|--------|------|---------|--------|
| Standard ICL (7B) | 85.0 | 93.2 | 66.7 | Baseline 70.7 |
| − content | 82.4 | 85.5 | 64.2 | **−3.8** |
| − stopword | 84.8 | 88.0 | 65.7 | −2.5 |
| − template | 0.9 | 61.0 | 12.9 | **−32.8** |

Adding a specific token type's representations to Zero-shot:

| Setting | 33B Model Avg. △ |
|---------|-----------------|
| Zero-shot baseline | 54.6 |
| + content | **−6.7** (performance degrades) |
| + stopword | +17.7 |
| + template | **+24.6** |

### Token-Level Ablation

| Configuration | Effect | Explanation |
|---------------|--------|-------------|
| Mask only content representations | Performance nearly unchanged (−3.8%) | Content is not directly relied upon by the test example |
| Mask only template representations | Performance drops sharply (−32.8%) | Template is directly relied upon by the test example |
| Remove content tokens from prompt | Performance drops substantially | Content information is necessary — but provided indirectly |
| Block content → template information flow | Performance drops | Validates the information aggregation mechanism |

### Key Findings
- **Counter-intuitive core finding**: LLMs do not directly "attend to" content tokens as humans do; instead, they aggregate content information into template token representations and make decisions from those representations.
- **Template tokens contribute 6.5× more than content tokens on average**: "+template" yields an average gain of +24.6%, while "+content" yields −6.7% (even a negative effect), quantifying a substantial gap.
- **"Indirect contribution" hypothesis for content tokens**: Representation-level ablation shows content is unimportant, while token-level ablation shows content is necessary — this apparent contradiction demonstrates that content contributes indirectly by being aggregated into template representations.
- The three characteristics (lexical meaning / repetitiveness / structural cues) are consistent across all model sizes, suggesting this is a general mechanism in LLMs.
- Results are robust across different instruction prompts and model families (Llama / Llama 2 / Mistral / Gemma 3).

## Highlights & Insights
- The finding that **"LLM attention is fundamentally different from human attention"** is highly thought-provoking: humans attend to content words, while LLMs attend to template and structural words. This challenges the assumption that LLMs comprehend text the way humans do.
- The **information aggregation hypothesis** is elegant: the content → template information flow explains the apparent "contradiction" between the two ablation results — content matters but does not directly participate; it exerts its influence indirectly by being aggregated into template representations.
- **Practical implications for prompt engineering**: The design of templates (rather than the wording of content) may have a greater impact on ICL performance.

## Limitations & Future Work
- Validation is primarily on classification tasks; results on generation tasks are presented in the appendix but are not explored in sufficient depth.
- The ablation methodology assumes that token types can be cleanly separated, whereas in practice some tokens simultaneously serve multiple roles.
- Whether instruction-tuned models exhibit different dependency patterns remains unexplored.

## Related Work & Insights
- **vs. Function Vector research**: Prior work focuses exclusively on the representation of the last token; this paper extends the analysis to all tokens in the prompt.
- **vs. Label Word research**: Earlier work identified label words as information anchors; this paper shows that template and stopword tokens are equally — or more — critical.
- **Implications for ICL theory**: The working mechanism of ICL cannot be simply attributed to "learning from demonstration content"; the structure and format of the prompt itself carries critical task specification information.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The finding that "templates matter more than content" is highly counter-intuitive and theoretically substantive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 LLM variants, 6 datasets, multiple ablation strategies, 15 seeds.
- Writing Quality: ⭐⭐⭐⭐ The logical chain is clear (categorization → ablation → findings → explanation → feature analysis).
- Value: ⭐⭐⭐⭐⭐ Significant implications for both understanding ICL mechanisms and optimizing prompt engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[AAAI 2026\] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)
- [\[AAAI 2026\] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)

</div>

<!-- RELATED:END -->
