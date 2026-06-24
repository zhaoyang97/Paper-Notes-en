---
title: >-
  [Paper Note] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models
description: >-
  [ACL 2025][LLM (Other)][Linguistic Acceptability] This paper compares the performance of various autoregressive language models (such as the GPT and Llama families) on linguistic acceptability judgment tasks. Through systematic experiments, it reveals the impact of model scale, training data, and architecture on grammatical judgment capabilities, and discusses whether the models' grammatical knowledge aligns with human linguistic intuition.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Linguistic Acceptability"
  - "Autoregressive Language Models"
  - "Grammaticality Judgment"
  - "Linguistic Evaluation"
  - "Perplexity"
date: 2026-05-08
content_hash: c67746fb9b45464e
---

# Comparing Linguistic Acceptability Judgments of Autoregressive Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Linguistic Acceptability, Autoregressive Language Models, Grammaticality Judgment, Linguistic Evaluation, Perplexity

## TL;DR
This paper compares the performance of various autoregressive language models (such as the GPT and Llama families) on linguistic acceptability judgment tasks. Through systematic experiments, it reveals the impact of model scale, training data, and architecture on grammatical judgment capabilities, and discusses whether the models' grammatical knowledge aligns with human linguistic intuition.

## Background & Motivation

**Background**: Linguistic acceptability judgment is a classic task for evaluating whether a language model truly "understands" linguistic structures. Given a sentence, the task is to judge whether it is grammatically acceptable (e.g., "The cat sat on the mat" vs. "*The cat sat on"). Benchmarks such as CoLA (Corpus of Linguistic Acceptability) are widely used to evaluate the grammatical capabilities of models.

**Limitations of Prior Work**: (1) Early NLP research primarily evaluated linguistic acceptability on encoder models (like BERT), but the modern mainstream has shifted to autoregressive (decoder-only) models, whose grammatical judgment capabilities have not been systematically evaluated; (2) Autoregressive models lack natural sentence-level representations, requiring proxy metrics such as perplexity to judge acceptability, yet the reliability of different proxy metrics has not been adequately compared; (3) Despite the rapid growth in model scale, it remains unclear whether larger models are inevitably better at grammatical judgment.

**Key Challenge**: Autoregressive models exhibit outstanding performance on generation tasks, but this does not imply that they possess precise grammatical knowledge. Models may "generate" grammatically correct sentences yet fail to "judge" sentence grammaticality symbols correctly, as generation and judgment represent distinct linguistic capacities.

**Goal**: (1) Systematically compare the grammatical judgment capabilities of different autoregressive models; (2) Compare the reliability of various evaluation schemes (perplexity, direct prompting, probability comparison, etc.); (3) Analyze differences in model performance across fine-grained grammatical phenomena (agreement, island constraints, binding theory, etc.).

**Key Insight**: Taking a linguistic perspective, the authors utilize not only general benchmarks like CoLA but also carefully designed minimal pair tests covering a variety of grammatical phenomena. Each pair differs by only a single grammatical feature, thereby enabling a precise diagnosis of the models' grammatical knowledge.

**Core Idea**: Through a systematic comparison of three evaluation paradigms—perplexity difference, direct prompting, and probability comparison—the paper finds that the grammatical judgment capability of autoregressive models scales logarithmically rather than linearly with model size, and that different grammatical phenomena present vastly different levels of difficulty for the models.

## Method

### Overall Architecture
The evaluation framework comprises a three-tier design: (1) Test Set Layer—incorporating CoLA, BLiMP, and customized minimal pair datasets; (2) Evaluation Method Layer—designing three schemes to extract grammatical judgments from autoregressive models; (3) Analysis Layer—subdividing results by grammatical category, model scale, and training data. The evaluation covers model families including GPT-2/3/4, Llama-2/3, Mistral, and Phi.

### Key Designs

1. **Multi-Paradigm Evaluation Schemes**:

    - **Function**: Extracted linguistic acceptability judgments from autoregressive models.
    - **Mechanism**: Three evaluation paradigms are designed: (a) Perplexity method: calculating the sentence perplexity, where lower perplexity corresponds to higher acceptability; (b) Direct prompting method: prompting the model to directly answer whether a sentence is grammatically correct ("Is the following sentence grammatically correct? ..."); (c) Probability comparison method: given a minimal pair (one grammatically correct, one incorrect), comparing their log-probability difference. Each of the three methods has its strengths and limitations—perplexity requires no prompting but is confounded by frequency effects, direct prompting relies heavily on the model's instruction-following capability, and probability comparison is the most precise but requires paired data.
    - **Design Motivation**: Different evaluation schemes may yield different conclusions; hence, comparing their consistency and reliability serves as an important reference for future research.

2. **Fine-Grained Grammatical Phenomenon Classification**:

    - **Function**: Diagnosed the grasp of knowledge of specific grammatical phenomena by models.
    - **Mechanism**: The test set is divided into six major categories based on grammatical phenomena: (a) Subject-verb agreement; (b) Argument structure; (c) Island constraints; (d) Binding theory; (e) Negative polarity item (NPI) licensing; (f) Tense/Aspect/Mood (TAM). Each category contains at least 100 minimal pair examples to ensure statistical significance, allowing for an analysis of which grammatical phenomena each model performs best or worst on.
    - **Design Motivation**: Aggregate accuracy hides the vast differences in performance across different grammatical phenomena; fine-grained analysis is necessary to reveal what grammar models "know" versus what they "do not know".

3. **Scaling Effect Analysis**:

    - **Function**: Investigated how model parameter size influences grammatical judgment capabilities.
    - **Mechanism**: Grammatical judgment accuracy is compared across different scales within the same model family (e.g., Llama-2-7B/13B/70B) to plot accuracy-parameter curves. Logarithmic coordinates are used to fit the curves and measure whether a sudden inflection point exists for "grammatical emergence." The respective contributions of training data size (number of tokens) and parameter size are also compared.
    - **Design Motivation**: Understanding the shape of scaling effects (linear, logarithmic, or step-wise) is crucial for predicting the grammatical capabilities of future models and for determining whether dedicated grammatical training is necessary.

### Loss & Training
This paper does not involve model training; it is a purely evaluative work. The models used are all publicly released pre-trained or instruction-tuned versions.

## Key Experimental Results

### Main Results

| Model | CoLA-MCC | BLiMP Accuracy | Probability Comparison Accuracy | Direct Prompting Accuracy |
|------|----------|------------|--------------|-------------|
| GPT-2 (124M) | 0.21 | 68.3% | 71.5% | 52.1% |
| GPT-2 (1.5B) | 0.35 | 76.8% | 79.2% | 58.3% |
| Llama-2-7B | 0.41 | 81.2% | 84.3% | 69.7% |
| Llama-2-70B | 0.52 | 86.7% | 89.1% | 78.4% |
| Llama-3-8B | 0.48 | 84.5% | 87.2% | 74.3% |
| GPT-4 | 0.58 | **89.3%** | **91.5%** | **83.6%** |

### Ablation Study (By Grammatical Phenomenon Category, GPT-4 Probability Comparison)

| Grammatical Phenomenon | Accuracy | Description |
|---------|--------|------|
| Subject-verb agreement | 95.2% | Highest, the grammatical rule the model masters best |
| Argument structure | 90.8% | Better, clear distinction between transitive/intransitive |
| Tense/Aspect/Mood | 88.3% | Good, but struggles with certain rare tenses |
| Negative polarity items | 82.1% | Moderate, licensing conditions for "any" are sometimes misjudged |
| Binding theory | 76.5% | Weaker, difficult to judge long-distance anaphoric relations |
| Island constraints | 71.3% | Weakest, insufficient mastery of complex syntactic movement constraints |

### Key Findings
- The probability comparison method is the most reliable evaluation scheme, showing the highest consistency and sensitivity; the direct prompting method is the least reliable, as small models can barely follow the instructions.
- Grammatical judgment accuracy scales logarithmically with model size, showing massive improvement from 124M to 1.5B, while the returns diminish from 7B to 70B.
- All models perform worst on island constraints and binding theory; these phenomena, which involve long-distance dependencies and complex syntactic structures, remain major challenges.
- Llama-3-8B approaches Llama-2-70B on certain grammar tests, indicating that improvements in training data quality and training methods are also critical.

## Highlights & Insights
- The systematic comparison of three evaluation paradigms is highly valuable; previously, different papers used different schemes, rendering results incomparable, whereas this paper provides a direct comparison.
- The fine-grained analysis of grammatical phenomena reveals an interesting hierarchy: surface-level grammar (agreement) is near-perfect, while deep-level grammar (island constraints) remains weak, which aligns with the pedagogical and theoretical ranking of grammatical complexity in theoretical linguistics.
- The discovery of the logarithmic growth curve suggests that simply scaling up models may not solve deep grammatical issues, potentially necessitating the introduction of inductive biases for grammar.

## Limitations & Future Work
- Only English was evaluated; grammatical phenomena vary drastically across different languages (e.g., evaluating free-word-order languages is much more challenging).
- Although precise, minimal pair tests are highly artificial and differ from the distribution of grammatical errors encountered in natural language.
- The impact of instruction-tuning (RLHF/DPO) on grammatical judgment capability was not analyzed, representing an interesting future direction.
- Models might exploit surface-level cues (such as word frequency differences) rather than genuine grammatical knowledge when making judgments.

## Related Work & Insights
- **vs BLiMP**: BLiMP is one of the benchmarks used in this paper, but this work expands both the range of evaluated models and the evaluation methodology.
- **vs CoLA/GLUE**: CoLA evaluates via the MCC metric; this work finds that this metric correlates moderately with the perplexity method, whereas the probability comparison method is more discriminative.
- **vs Syntax-Probing**: Previous syntax-probing lines of work focused on internal model representations, whereas this work examines behavioral-level grammatical judgments, making the two approaches complementary.

## Rating
- Novelty: ⭐⭐⭐ Primarily focused on evaluation methodology, introducing no new models or methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers a wide range of models and grammatical phenomena, with meticulous analysis.
- Writing Quality: ⭐⭐⭐⭐ Linguistic background is clearly introduced with a well-designed experiment.
- Value: ⭐⭐⭐⭐ Provides a standardized scheme and crucial empirical findings for evaluating LLMs' grammatical capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments](explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)

</div>

<!-- RELATED:END -->
