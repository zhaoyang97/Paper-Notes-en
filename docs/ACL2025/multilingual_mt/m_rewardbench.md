---
title: >-
  [Paper Note] M-RewardBench: Evaluating Reward Models in Multilingual Settings
description: >-
  [ACL 2025][Multilingual & Machine Translation][reward model] This work constructs the first multilingual reward model evaluation benchmark, M-RewardBench (covering 23 typologically diverse languages, 2.87K preference instances, across four capability categories: Chat, Safety, Reasoning, and Translation). Systematic evaluation of various RMs reveals a significant performance gap between English and non-English settings, and indicates that RM preferences can shift substantially…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "reward model"
  - "Multilingual"
  - "RLHF"
  - "Preference Benchmark"
  - "preference drift"
date: 2026-05-08
content_hash: f98622694600c9b2
---

# M-RewardBench: Evaluating Reward Models in Multilingual Settings

**Conference**: ACL 2025  
**arXiv**: [2410.15522](https://arxiv.org/abs/2410.15522)  
**Code**: [https://github.com/for-ai/m-rewardbench](https://github.com/for-ai/m-rewardbench)  
**Area**: LLM Alignment / Multilingual / Reward Model Evaluation  
**Keywords**: reward model, Multilingual, RLHF, Preference Benchmark, preference drift

## TL;DR

This work constructs the first multilingual reward model evaluation benchmark, M-RewardBench (covering 23 typologically diverse languages, 2.87K preference instances, across four capability categories: Chat, Safety, Reasoning, and Translation). Systematic evaluation of various RMs reveals a significant performance gap between English and non-English settings, and indicates that RM preferences can shift substantially across different languages.

## Background & Motivation

**Background**: Reward models (RMs) are core components in contemporary LLM alignment (RLHF/DPO), steering language models to generate high-quality outputs by learning human preferences. Benchmarks like RewardBench have served as standard tools for evaluating RM performance.

**Limitations of Prior Work**: The training and evaluation of RMs are conducted almost exclusively in English. Although most global users interact with LLMs in non-English languages, there is a lack of understanding regarding whether RMs can accurately assess human preferences in these languages. Consequently, the alignment quality of models aligned via RLHF might suffer from systemic deficiencies in non-English scenarios.

**Key Challenge**: Are RMs that perform exceptionally well in English equally reliable in other languages? If not, what are the implications for multilingual LLM deployment?

**Goal**: To quantify the cross-lingual performance gaps of RMs and identify their driving factors by constructing a multilingual RM evaluation benchmark and performing a systematic evaluation.

**Key Insight**: Relying on high-quality translations of the English RewardBench to construct preference datasets in 23 languages, maintaining consistent evaluation dimensions for fair cross-lingual comparison.

**Core Idea**: The first multilingual RM benchmark along with a systematic evaluation reveals cross-lingual preference drift and performance gaps in RMs.

## Method

### Overall Architecture

Starting from the English RewardBench, preference instances are carefully selected and translated with high quality into 23 languages. Subsequently, three categories of RMs (classifier-based, generative, and implicit) are systematically evaluated on this benchmark.

### Key Designs

1. **多语言基准构建（M-RewardBench Dataset）**:

    - **Function**: Constructing an RM evaluation dataset covering 23 typologically diverse languages.
    - **Mechanism**: Selecting preference instances (chosen/rejected pairs) from RewardBench across four capability dimensions—Chat, Safety, Reasoning, and Translation (newly added). The translation pipeline incorporates rigorous quality control, executing machine translation followed by human verification to ensure semantic equivalence.
    - **Design Motivation**: Directly adopting English benchmarks fails to reflect multilingual capabilities, while constructing native preference datasets for each language from scratch is prohibitively expensive and makes maintaining assessment consistency difficult. The translation-based strategy balances cost and comparability.
    - **Languages Covered**: 23 languages, including Chinese, Japanese, Korean, Arabic, Hindi, French, German, Spanish, Russian, Turkish, etc., spanning diverse language families and writing systems, totaling 2.87K preference pairs.

2. **系统化评估框架**:

    - **Function**: Comprehensively evaluating the cross-lingual performance of various RM architectures on M-RewardBench.
    - **Mechanism**: Evaluating three categories of RMs—(1) Classifier-based RMs (e.g., UltraRM) that output scalar scores via regression heads; (2) Generative RMs / LLMs-as-a-judge (e.g., GPT-4) that directly output preference judgments; (3) Implicit RMs (e.g., DPO-trained models) that convey preferences implicitly via likelihood margins. Each RM is evaluated across the 23 languages.
    - **Design Motivation**: Different RM architectures might present distinct multilingual generalization patterns—classifier-based models might be more susceptible to cross-lingual representation alignment, whereas generative RMs might be sensitive to prompting languages.

3. **多维度分析**:

    - **Function**: Conducting deep analyses of factors influencing cross-lingual RM performance.
    - **Mechanism**: (1) English vs. non-English overall gap analysis, (2) cross-lingual preference drift analysis (shedding light on whether the RM's chosen/rejected decision for the same instance is inverted across different languages), (3) correlation analysis between translation quality and RM performance, and (4) the relationship between language resource levels (high/medium/low-resource) and RM performance.
    - **Design Motivation**: Merely identifying the existence of a gap is insufficient; understanding the sources of these gaps is crucial to guide improvements.

## Key Experimental Results

### Main Results

| Dimension | English | Non-English Average | Gap |
|------|------|-----------|------|
| Overall Accuracy | Highest | Significantly lower than English | Pronounced |
| Chat | High | Significant decline | Moderate |
| Safety | High | Significant decline | Moderate |
| Reasoning | High | Largest drop | Large |

### Ablation Study

| Factor | Direction of Impact | Description |
|------|---------|------|
| Translation Quality | Positively Correlated | Higher translation quality leads to better RM performance in that language. |
| Language Resource Volume | Positively Correlated | High-resource languages (French/German/Spanish) outperform low-resource languages (Swahili/Urdu). |
| RM Architecture | Language-Dependent | No single architecture achieves optimal performance across all languages. |

### Key Findings

- **Preference Drift**: When identical preference instances are translated into different languages, the RM's chosen/rejected decisions can flip. This indicates that RM preferences are language-dependent rather than language-agnostic.
- **Causal Chain of Translation Quality $\rightarrow$ RM Performance**: RMs perform better in languages with higher-quality translations, pointing to a clear path for improving multilingual RMs.
- **Pronounced High-Resource Advantage**: RM performance in high-resource languages (e.g., Chinese, Japanese, Korean, French, German) is close to English, whereas the gap widens significantly for low-resource languages.
- **Uneven Gaps Across Capacity Dimensions**: The cross-lingual performance gap is largest in the Reasoning dimension, likely reflecting a higher sensitivity of reasoning tasks to language-specific semantic nuances.

## Highlights & Insights

- **First Multilingual Evaluation Benchmark for Reward Models**: Fills a critical evaluation gap in the multilingual deployment of RLHF—previously, the reliability of RMs in multilingual scenarios was an overlooked blind spot.
- **Theoretical Significance of Preference Drift**: Demonstrates that RM measurements of human preferences are not language-agnostic; the same preference pair can yield opposite judgments when expressed in different languages.
- **Comprehensive Evaluation**: Spanning 23 languages, multiple RM architectures, and four capability dimensions, offering unprecedented coverage and systematicity in this line of research.

## Limitations & Future Work

- **Potential Translationese Bias**: Relying on translation may introduce translationese bias, where the linguistic features of translated texts differ from native texts, potentially influencing RM judgments.
- **Limited Sample Size**: With approximately 125 instances per language, the sample size is relatively small, which might limit statistical significance for certain languages.
- **Lack of Code-Switching Contexts**: Code-switching and mixed-language scenarios, which are common in real-world multilingual usage, are not covered.
- **Superficial Feature Analysis**: The analysis primarily focuses on translation quality and resource volume, leaving deeper factors like cultural variations or linguistic structures unexplored.
- **Incomparability of Added Dimension**: The newly introduced Translation dimension makes the suite not fully comparable to the original RewardBench.

## Related Work & Insights

- **vs. RewardBench**: Limited to monolingual English evaluation, whereas M-RewardBench extends to 23 languages to enable cross-lingual comparisons.
- **vs. Multilingual Benchmarks (e.g., MEGA, XTREME)**: Typically assess the multilingual capabilities of LLMs themselves; M-RewardBench specifically targets reward models.
- **vs. Multilingual RLHF Literature**: Prior works rarely evaluated RM performance in multilingual contexts; this work represents a pioneering effort in this area.

## Rating

- Novelty: ⭐⭐⭐⭐ First multilingual RM evaluation benchmark; represents an important and pioneering direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 23 languages and multiple RM architectures, though the sample size per language is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-organized findings, and strong motivating arguments.
- Value: ⭐⭐⭐⭐ Directly applicable for steering multilingual LLM alignment and deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] ZIPA: A Family of Efficient Models for Multilingual Phone Recognition](zipa_a_family_of_efficient_models_for_multilingual_phone_recognition.md)
- [\[ACL 2025\] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](x-webagentbench_a_multilingual_interactive_web_benchmark_for_evaluating_global_a.md)

</div>

<!-- RELATED:END -->
