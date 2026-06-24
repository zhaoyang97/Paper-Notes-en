---
title: >-
  [Paper Note] Behavioral Analysis of Information Salience in Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Information Salience] An interpretable analysis framework is proposed to systematically derive and investigate the concept of information salience internalized by LLMs, using length-controlled summarization behavioral probes and tracking the answerability of Questions Under Discussion (QUD). The study reveals that LLMs possess a hierarchical and consistent notion of salience, which, however, cannot be accessed via self-introspection and is o…
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Information Salience"
  - "Summarization"
  - "Content Selection"
  - "QUD"
  - "Behavioral Analysis"
date: 2026-05-08
content_hash: ab829d4a62e8bb12
---

# Behavioral Analysis of Information Salience in Large Language Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.14613](https://arxiv.org/abs/2502.14613)  
**Code**: None  
**Area**: LLM Analysis / NLP  
**Keywords**: Information Salience, Summarization, Content Selection, QUD, Behavioral Analysis

## TL;DR

An interpretable analysis framework is proposed to systematically derive and investigate the concept of information salience internalized by LLMs, using length-controlled summarization behavioral probes and tracking the answerability of Questions Under Discussion (QUD). The study reveals that LLMs possess a hierarchical and consistent notion of salience, which, however, cannot be accessed via self-introspection and is only weakly correlated with human perception.

## Background & Motivation

- **Background**: Large language models perform exceptionally well on text summarization tasks. Summarization is inherently a content selection task where the model must determine which information in the source text is "important" (salient) and prioritize its preservation. This implies that LLMs must have developed internal representations regarding information importance.
- **Limitations of Prior Work**: Although LLMs are known to generate high-quality summaries, little is understood about the exact concept of "salience" they internalize. Existing studies mostly evaluate summarization from the perspective of output quality, lacking a systematic characterization of the models' internal information prioritization mechanisms.
- **Key Challenge**: Without understanding how LLMs prioritize information importance, predicting their behaviors in downstream tasks and effectively controlling their content selection strategies remains impossible. Furthermore, whether the salience concept in LLMs aligns with human intuition regarding information importance remains an open question.
- **Goal**: This study aims to design a systematic framework to derive and analyze the concept of information salience in LLMs, quantifying the consistency across different models and their alignment with humans.
- **Key Insight**: Length-controlled summarization is utilized as a "behavioral probe." By prompting models to generate summaries of varying target lengths (from extremely short to relatively long), their content selection under different compression ratios is observed to infer their information prioritization.
- **Core Idea**: Answerability of Questions Under Discussion (QUD) is employed as a proxy metric for information salience. If a QUD remains answerable in a short summary, the corresponding information is prioritized highly in the model's salience ranking. By tracking changes in QUD answerability across different summary lengths, a hierarchical salience spectrum of information can be constructed.

## Method

### Overall Architecture

The framework comprises three core stages: (1) **Summarization probe**: Prompting LLMs to generate summaries of multiple target lengths (e.g., 10%, 20%, 30%... 80% of the original text) for the same source document; (2) **QUD answerability analysis**: Generating a set of QUDs covering the entire source text and checking whether each QUD is answerable in summaries of each length level; (3) **Salience spectrum construction**: Deriving the information salience ranking based on the "survival curve" of QUDs (i.e., how short a summary can be while still answering the QUD).

### Key Designs

1. **Length-Controlled Summarization as Behavioral Probe**: Precisely controlling the target summary length forces the model to make explicit content selection decisions. When the target length is extremely short, only the most "salient" information is preserved; as the length increases, less salient information is gradually included. This graded compression reveals the model's information prioritization, similar to "rate-distortion" analysis in information theory. The design motivation is to avoid directly asking the model "what is important" (which can be inaccurate) and instead infer it from behaviors.
2. **QUD Answerability Tracking**: QUD (Questions Under Discussion) is an important concept in discourse analysis, where each QUD represents an implicit information need in the text. This paper automatically generates a set of QUDs for the source text and evaluates whether each QUD remains answerable in summaries of various lengths. The survivability point of a QUD — the minimum summary length at which it is still answerable — directly reflects the salience rank of the corresponding information in the model.
3. **Cross-Model Consistency Analysis**: The same framework is applied across 13 different LLMs (spanning multiple model families and scales) to compare their derived salience rankings. Consistency between models is quantified by calculating rank correlation coefficients (e.g., Kendall's $\tau$, Spearman's $\rho$).

### Loss & Training

This paper is an analytical study and does not involve model training. The main technical components include:
- **QUD Generation**: Automatically generating a comprehensive set of questions for the source text using an LLM.
- **Answerability Judgment**: Utilizing NLI (Natural Language Inference) models or LLMs to determine whether a given summary can answer a specific QUD.
- **Salience Score Calculation**: Assigning salience scores based on the answerability thresholds of QUDs at different summary lengths.
- **Evaluation Metrics**: Inter-model consistency (rank correlation coefficients) and model-human alignment (correlation with human salience annotations).

## Key Experimental Results

### Main Results

The salience behaviors of 13 LLMs are evaluated across four summarization datasets.

| Analysis Dimension | Result | Number of Datasets | Number of Models | Description |
|---------|------|----------|---------|------|
| Inter-model consistency (same family) | $\tau \approx 0.7-0.8$ | 4 | 13 | Highly consistent within the same family |
| Inter-model consistency (cross family) | $\tau \approx 0.5-0.65$ | 4 | 13 | Moderately correlated across families |
| Model-human alignment | $\tau \approx 0.2-0.35$ | 4 | 13 | Only weakly correlated with human perception |
| Introspection vs. behavioral consistency | Low | 4 | 13 | Models cannot accurately report their own content selection strategies |
| Salience hierarchy | 3-5 layers | 4 | 13 | Information is categorized into clear priority levels |

### Ablation Study

| Configuration | Inter-model consistency | Description |
|------|------------|------|
| Full framework (7-level length control) | $\tau \approx 0.65$ | Sufficient granularity differentiation |
| Coarse-grained length control (3-level) | $\tau \approx 0.55$ | Insufficient granularity, ambiguous ranking |
| Directly asking for salience ranking | $\tau \approx 0.30$ | Self-introspection approach performs poorly |
| Varying number of QUDs (10 vs 30 vs 50) | $\tau \approx 0.58-0.67$ | Stabilizes with 30+ QUDs |
| Impact of model scale | Weak positive correlation | Larger models are slightly more consistent |
| GPT-4 level models | $\tau \approx 0.72$ | The strongest models exhibit the most consistent concept of salience |

### Key Findings

- **LLMs possess a hierarchical concept of salience**: Experiments demonstrate that LLMs organize information into distinct priority levels (typically 3-5 levels). Highly salient information is retained even in extremely short summaries, while low-salience information is discarded under moderate compression. This hierarchical structure is consistently present across different models.
- **High consistency across models but weak correlation with humans**: Different LLMs (even across families and scales) demonstrate high consistency in information prioritization ($\tau \approx 0.5-0.8$), indicating a shared "information importance" prior in their pre-training data. However, this consensus among models correlates weakly with human perception of information importance ($\tau \approx 0.2-0.35$), suggesting that the LLMs' concept of salience may lean toward "statistical frequency" rather than "cognitive importance."
- **Self-introspection fails to reveal true preferences**: Direct prompts for LLMs to report what they consider important (introspection approach) diverge significantly from their actual behaviors (information preserved in summaries). This cautions against trusting LLMs' "explanations" of their own behaviors.
- **The QUD framework serves as an effective proxy for salience**: Indirect measurements based on QUD answerability align closely with direct information preservation analysis, validating the methodological robustness of the framework.
- **Model scale has limited impact on the concept of salience**: Models of different sizes within the same family share highly similar salience rankings, showing that this concept is primarily formed during pre-training rather than emerging with scale.

## Highlights & Insights

- **Elegant methodological innovation**: Utilizing length-controlled summaries as a "behavioral microscope" to indirectly derive models' intrinsic information preferences avoids unreliable introspective querying. This methodology can be extended to other LLM behavioral analysis scenarios.
- **Integration of QUD and NLP theory**: Introducing QUD theory from discourse analysis into LLM behavioral analysis solidifies the theoretical linguistic foundation of the experimental framework.
- **Important negative findings**: The weak alignment between models and humans is a critical finding. The high performance of LLMs in summarization tasks might mask the actual gap between their information prioritization and human cognition.
- **Evidence of untrustworthy self-introspection**: This further supports the crucial conclusion that "LLMs are not reliable explainers of their own behavior."

## Limitations & Future Work

- The precision of length control relies on the instruction-following capability of the LLM itself; deviations in length control across different models may introduce noise.
- QUD generation itself depends on LLMs, creating a circular dependency risk. If the QUDs generated by the model are biased, the derived salience may reflect QUD bias rather than content preferences.
- Four datasets may be insufficient to cover all text genres (e.g., information structures in narrative, argumentative, and news texts can vary drastically).
- The underlying reasons for the low model-human alignment require deeper analysis: Is it due to model bias or noise in human annotations?
- Future work can extend the framework to multilingual and multimodal scenarios to investigate the salience of visual information.

## Related Work & Insights

- **vs. Lost-in-the-Middle (Liu et al., 2024)**: While that work examines the impact of positional bias on LLM attention, this study investigates information prioritization at the content level, offering a complementary perspective.
- **vs. Summarization evaluation frameworks like SummEval/Shannon**: These works evaluate the output quality of summaries, whereas this study delves into the model's content selection process, focusing on "why this content was selected" rather than "how good the selected content is."
- **vs. Probing methods (e.g., attention head analysis)**: Traditional probing starts from the model's internal representations, while this study starts from external behaviors. The two methodologies can mutually validate each other.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study on the concept of information salience in LLMs, with a highly creative methodological design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 models, 4 datasets, and rich dimensions of analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear research motivation, systematic framework description, and solid theoretical linguistic foundation.
- Value: ⭐⭐⭐⭐⭐ Uncovers the nature of LLMs' intrinsic information preferences, providing crucial insights for both explainable AI and summarization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)

</div>

<!-- RELATED:END -->
