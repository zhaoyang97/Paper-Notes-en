---
title: >-
  [Paper Note] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases
description: >-
  [ACL 2025][LLM (Other)][L2 Simulation] This paper evaluates the ability of LLMs to simulate non-native English speakers' (L2 learners) dialogue. Through information-theoretic and distribution density metrics, the authors analyze whether LLM-generated L2 English can replicate the native-language-dependent biases (such as tense consistency errors, avoidance behavior, etc.) of human L2 learners, finding that modern LLMs can indeed replicate certain L1-dependent patterns.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "L2 Simulation"
  - "Native Language Interference"
  - "Information-Theoretic Analysis"
  - "L1 Transfer"
  - "Language Education"
date: 2026-05-08
content_hash: e065bfa58e3c10fb
---

# Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases

**Conference**: ACL 2025  
**arXiv**: [2502.14507](https://arxiv.org/abs/2502.14507)  
**Code**: None  
**Area**: LLM Analysis / Computational Linguistics / Second Language Acquisition (SLA)  
**Keywords**: L2 Simulation, Native Language Interference, Information-Theoretic Analysis, L1 Transfer, Language Education

## TL;DR

This paper evaluates the ability of LLMs to simulate non-native English speakers' (L2 learners) dialogue. Through information-theoretic and distribution density metrics, the authors analyze whether LLM-generated L2 English can replicate the native-language-dependent biases (such as tense consistency errors, avoidance behavior, etc.) of human L2 learners, finding that modern LLMs can indeed replicate certain L1-dependent patterns.

## Background & Motivation

**Background**: Research in Second Language Acquisition (SLA) demonstrates that L2 learners' English usage is systematically influenced by their native language (L1). For instance, native Japanese speakers exhibit characteristic errors in English tense usage, and native Urdu speakers have specific biases in noun-verb collocations. These L1-dependent biases are well-documented in human data.

**Limitations of Prior Work**: (1) Collecting authentic dialogue data from L2 learners is highly costly, hindering the development of SLA research and language educational technologies; (2) If LLMs could accurately simulate human L2 linguistic features, they could be leveraged to generate synthetic training data and develop educational tools (such as simulated conversation partners); (3) However, there is a lack of systematic research evaluating the accuracy of LLM-simulated L2 English and the fidelity of simulated L1-dependent patterns.

**Key Challenge**: Given that LLMs are primarily exposed to "correct" English texts during pre-training, can they accurately simulate the characteristic "errors" of L2 learners? More importantly, can these simulations differentiate between learners from different L1 backgrounds?

**Goal**: (1) Evaluate whether LLMs can generate L2 English dialogue with distinct L1 characteristics under prompt guidance; (2) quantify the statistical alignment between LLM-simulated L2 English and authentic human L2 data; (3) reveal how different LLMs vary in their ability to simulate different L1 backgrounds.

**Key Insight**: Utilize information-theoretic metrics (such as surprisal, KL divergence) and distribution density metrics to compare the statistical distribution characteristics of LLM-generated L2 English against human L2 learner data.

**Core Idea**: Systematically evaluate the capability of LLMs to simulate L2 English learners from diverse native language backgrounds through information-theoretic quantitative analysis.

## Method

### Overall Architecture

Research workflow: (1) Collect human L2 English dialogue data across 7 L1 backgrounds; (2) design prompt templates to guide LLMs in simulating L2 learners from each L1 background; (3) utilize information-theoretic and linguistic metrics to compare the distributional characteristics of LLM-generated L2 English with authentic human L2 data.

### Key Designs

1. **Multi-L1 L2 Simulation Experimental Design**:

    - **Function**: Systematically evaluate LLM simulation capabilities for L2 learners with various native language backgrounds.
    - **Mechanism**: Select seven L1 backgrounds: Japanese, Korean, Thai, Urdu, Chinese (Mandarin), Spanish, and French. Each group exhibits known, typical bias patterns in L2 English (e.g., Japanese speakers tend to omit articles, and Korean speakers exhibit characteristic errors in tense marking). Systematically designed prompt templates are leveraged to instruct the LLM to "simulate an English learner whose L1 is X" within structured scenarios like IELTS speaking interviews to generate dialogues.
    - **Design Motivation**: Select languages with known, quantifiable L1 interference patterns to enable robust quantitative comparisons.

2. **Information-Theoretic Analysis Framework**:

    - **Function**: Mathematically and rigorously quantify the fidelity of simulated L2 linguistic characteristics.
    - **Mechanism**: Adopt two complementary information-theoretic metrics: (a) **Reference word usage bias**—computes the KL divergence of specific vocabulary usage frequencies (such as pronouns, articles, prepositions) in L2 English relative to native speaker English, comparing whether the divergence patterns match between LLM-generated and human L2 data; (b) **Avoidance behavior analysis**—uses distribution density measures to detect if L2 learners (and LLM simulations) systematically avoid specific linguistic structures (such as clause reduction or avoiding complex tenses), quantifying the avoidance degree by comparing actual usage frequencies against native expectations.
    - **Design Motivation**: Information-theoretic metrics provide a way to detect distributional shifts without pre-defining "correct vs. incorrect" rules, making them highly suitable for capturing subtle L1 interference patterns.

3. **Multi-dimensional Linguistic Feature Comparison**:

    - **Function**: Validate the accuracy of LLM simulations across multiple linguistic dimensions.
    - **Mechanism**: Analyze the following dimensions: (a) lexical diversity (type-token ratio); (b) tense consistency error rates; (c) article usage error rates; (d) noun-verb collocation anomalies; and (e) sentence complexity distributions. For each dimension, calculate the correlation coefficients and distributional distances between LLM generations and human L2 data.
    - **Design Motivation**: Matching along a single dimension is insufficient; consistency across multiple dimensions is critical to proving that the LLM is genuinely simulating L1-dependent patterns.

### Experimental Models

Tested modern LLMs including Qwen2.5, LLaMA3.3, DeepSeek-V3, and GPT-4o.

## Key Experimental Results

### Main Results

| L1 Background | Dimension | Correlation: LLM vs. Human L2 | Representative Findings |
|---|---|---|---|
| Japanese | Tense consistency | High correlation | LLMs accurately replicated tense weaknesses of Japanese speakers |
| Korean | Tense marking | High correlation | Similar tense interference patterns to Japanese |
| Chinese (Mandarin) | Tense + Articles | Moderate-to-high correlation | Article omission was partially replicated |
| Urdu | Noun-Verb collocations | Moderate correlation | Collocation preferences were partially captured |
| Thai | Lexical avoidance | Moderate correlation | Avoidance behavior patterns partially matched |
| Spanish | Preposition misuse | Moderate correlation | Preposition choice biases were replicated |
| French | Lexical choice | Lower correlation | French L1 interference is subtle and poorly captured by LLMs |

### Ablation Study

| Model | Average L1 Pattern Correlation | Description |
|---|---|---|
| GPT-4o | Highest | Most accurate simulation |
| Qwen2.5 | High | Outperformed GPT-4o on certain L1s |
| DeepSeek-V3 | High | Comparable to Qwen2.5 |
| LLaMA3.3 | Moderate-to-high | Insensitive to certain L1 patterns |
| Prompt without L1 specification | Very low | Indicates L1 information is crucial for simulation |

### Key Findings
- Modern LLMs (particularly GPT-4o and Qwen2.5) are capable of replicating L1-dependent L2 English bias patterns to a significant degree.
- L1 interference patterns for Japanese, Korean, and Chinese are replicated the best, as these languages structurally diverge heavily from English, exposing more prominent bias patterns.
- L1 interference for languages with structures closer to English (such as French) is more subtle and poorly captured by LLMs.
- Explicitly specifying L1 in the prompt is critical; without it, LLMs generate generic simplified English rather than L1-specific L2 English.
- These findings imply that LLMs implicitly acquire cross-linguistic relational knowledge during pre-training.

## Highlights & Insights
- **Information-theoretic paradigm to analyze LLM simulation capabilities** introduces a novel evaluation scheme: instead of manually reviewing sentence-level "correctness", it focuses on overall distributional profile comparisons. This framework can easily adapt to evaluate simulations of other language varieties (e.g., dialects, historical stages of languages).
- **Uncovering implicit cross-linguistic knowledge in LLMs** is highly insightful. LLMs do not just "know" different languages; they also grasp what kind of errors speakers of a specific L1 tend to make when using English. The exact acquisition mechanism of such knowledge warrants future research.
- **Empirical foundations for language learning technology** are established, showing that LLMs can generate high-quality synthetic L2 dialogue data across different proficiency levels and L1 backgrounds.

## Limitations & Future Work
- Only seven L1 backgrounds were tested, all of which are relatively high-resource. Replicability for low-resource languages (e.g., Vietnamese, Swahili) remains unexplored.
- Experiments relied on highly structured conversational scenarios (IELTS interviews); performance under free-form dialogue might differ.
- The actual downstream utility of the generated synthetic data (e.g., for training L2 writing assessors) has not been evaluated.
- Future research can explore directing LLMs to simulate L2 learners at varying proficiency/fluency levels.

## Related Work & Insights
- **vs. Automated CEFR Grading**: Automated L2 proficiency grading focuses on *evaluating* L2 proficiency, whereas this study focuses on *simulating* L2 usage—proposing a highly complementary perspective.
- **vs. Computational SLA Models**: Traditional computational models of L1 interference are rule-based or statistical, whereas LLMs offer a highly flexible, end-to-end simulation approach.
- **vs. Educational Technology**: Although ChatGPT and similar LLMs are already adopted for language teaching, this work provides empirical evaluations regarding the precision of their L2 simulation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unique research formulation with an elegantly designed information-theoretic evaluation framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematically and comprehensively covers 7 L1s across multiple LLMs and multi-dimensional analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Well-balanced interdisciplinary exposition bridging SLA, NLP, and information theory.
- **Value**: ⭐⭐⭐⭐ Offers crucial insights into LLMs' implicit cross-linguistic knowledge and practical educational applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] Behavioral Analysis of Information Salience in Large Language Models](behavioral_analysis_of_information_salience_in_large_language_models.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Concreteness Versus Abstractness: A Selectivity Analysis in LLMs](concreteness_versus_abstractness_a_selectivity_analysis_in_llms.md)

</div>

<!-- RELATED:END -->
