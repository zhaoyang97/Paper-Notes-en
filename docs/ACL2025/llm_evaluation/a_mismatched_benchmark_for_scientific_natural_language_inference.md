---
title: >-
  [Paper Note] MisMatched: A Benchmark for Scientific Natural Language Inference
description: >-
  [ACL 2025][LLM Evaluation][scientific NLI] Introduces MisMatched—the first scientific NLI evaluation benchmark covering non-CS fields (Psychology, Engineering, Public Health), consisting of 2,700 human-annotated sentence pairs. The best SLM baseline (SciBERT) achieves a Macro F1 of only 78.17%, while the best LLM baseline (Phi-3) scores only 57.16%. It also proves that training with implicit relation sentence pairs can improve model performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "scientific NLI"
  - "out-of-domain evaluation"
  - "cross-domain generalization"
  - "implicit relations"
  - "benchmark"
date: 2026-05-08
content_hash: 73f4b65afbbf876f
---

# MisMatched: A Benchmark for Scientific Natural Language Inference

**Conference**: ACL 2025  
**arXiv**: [2506.04603](https://arxiv.org/abs/2506.04603)  
**Code**: [https://github.com/fshaik8/MisMatched](https://github.com/fshaik8/MisMatched)  
**Area**: NLI / Scientific Text Understanding  
**Keywords**: scientific NLI, out-of-domain evaluation, cross-domain generalization, implicit relations, benchmark

## TL;DR
Introduces MisMatched—the first scientific NLI evaluation benchmark covering non-CS fields (Psychology, Engineering, Public Health), consisting of 2,700 human-annotated sentence pairs. The best SLM baseline (SciBERT) achieves a Macro F1 of only 78.17%, while the best LLM baseline (Phi-3) scores only 57.16%. It also proves that training with implicit relation sentence pairs can improve model performance.

## Background & Motivation

**Background**: The scientific NLI task classifies sentence pairs from research papers into four relations: Entailment, Reasoning, Contrasting, and Neutral. Existing datasets in this area include SciNLI (ACL Anthology, NLP domain) and MSciNLI (5 CS subfields), both of which construct training sets using distant supervision (automatically labeling via linking phrases like "However" or "Therefore"), thus only covering the CS domain.

**Limitations of Prior Work**: (1) All existing scientific NLI datasets only cover computer science, leaving non-CS domains completely blank; (2) Training sets only capture explicit relations (where the second sentence begins with a linking phrase) via distant supervision, ignoring a large number of implicit relations; (3) There is a lack of out-of-domain (OOD) test benchmarks to evaluate the cross-domain generalization capability of models.

**Key Challenge**: Can scientific NLI models trained on the CS domain generalize to other scientific domains? Do implicit relations omitted during distant supervision constitute a blind spot for models?

**Goal**: (1) Construct a scientific NLI evaluation benchmark for non-CS domains to test the OOD robustness of existing models; (2) Study the impact of training data with implicit relations on model performance.

**Key Insight**: Mirroring the design concept of the mismatched test set in MNLI—using data outside the training domain to evaluate the generalization capabilities of models.

**Core Idea**: By constructing a scientific NLI test benchmark across three non-CS domains (Psychology, Engineering, and Public Health), this work reveals the cross-domain generalization bottlenecks of existing models and discovers that training data with implicit relations can enhance performance.

## Method

### Overall Architecture
MisMatched is a pure evaluation benchmark (consisting only of dev/test sets, without a training set) designed for OOD testing. Models are trained on existing SciNLI/MSciNLI CS training sets and evaluated on the MisMatched non-CS test sets. The data construction consists of two phases: automatic extraction + distant labeling followed by human annotation verification.

### Key Designs

1. **Selection and Construction of Three-Domain Data Sources**:

    - Function: Collects research papers from Web of Science (WoS) for psychology and engineering, and from WoS, NLM, and PubMed for public health.
    - Mechanism: Phase 1 automatically extracts and labels sentence pairs using distant supervision (linking phrase mapping); the Neutral class is created by randomly pairing non-adjacent sentences using three strategies (BothRand/FirstRand/SecondRand). Phase 2 employs domain experts through the COGITO platform for iterative human annotation.
    - Design Motivation: Choosing domains highly distinct from CS to maximize the challenge of OOD testing.

2. **Iterative Human Annotation Process**:

    - Function: Conducted in multiple iterations where three annotators label a randomly sampled balanced subset in each turn, keeping only samples where distant labels and human gold labels match.
    - Mechanism: Attains an inter-annotator Fleiss-$\kappa$ of 0.72 (moderate-to-strong agreement). A total of 3,253 pairs were annotated, of which 2,791 showed agreement between automatic and human labels. The final set is downsampled to a balanced size of 225 samples per category per domain, totaling 2,700 pairs.
    - Design Motivation: Implementing rigorous quality control to ensure benchmark reliability by retaining only instances with full agreement between distant and human annotations.

3. **Training with Implicit Relations**:

    - Function: Defines sentence pairs in the MSciNLI training set as "implicit relation" samples if their original labels remain valid after removing the linking phrases, which are then incorporated to construct the MSciNLI+ dataset.
    - Mechanism: Traditional training sets of SciNLI/MSciNLI contain only sentence pairs starting with linking phrases (explicit relations). However, in reality, many sentence pairs exhibit semantic relations without starting the second sentence with a linking phrase—these represent implicit relations.
    - Design Motivation: If models solely rely on linking phrases as classification cues, their performance will be limited during evaluation when these phrases are removed.

### SLM and LLM Baseline Settings
SLMs: BERT, SciBERT, RoBERTa, and XLNet are fine-tuned on SciNLI/MSciNLI/MSciNLI+. LLMs: Llama-2/3, Mistral, Phi-3, GPT-4o, and Gemini-1.5-Pro are evaluated under zero-shot and four-shot settings.

## Key Experimental Results

### Main Results

| Model | Training Data | Psychology | Engineering | Public Health | Overall |
|------|---------|------------|-------------|---------------|---------|
| BERT | MSciNLI | 68.00 | 69.23 | 66.34 | 67.89 |
| BERT | MSciNLI+ | 71.16 | 73.52 | 69.47 | 71.41 |
| SciBERT | MSciNLI | 76.98 | 76.56 | 77.97 | 77.66 |
| SciBERT | MSciNLI+ | **79.18** | 76.50 | **78.79** | **78.17** |
| RoBERTa | MSciNLI+ | 77.91 | **77.63** | **78.79** | 78.11 |

### Ablation Study

| Model | Setting | Psychology | Engineering | Public Health | Overall |
|------|------|------------|-------------|---------------|---------|
| Phi-3 | zero-shot | 55.38 | 53.15 | 49.31 | 52.95 |
| Phi-3 | fs-MSciNLI | 58.64 | **56.76** | **55.68** | **57.16** |
| GPT-4o | zero-shot | 52.42 | 50.12 | 47.26 | 50.26 |
| GPT-4o | fs-SciNLI | **63.33** | 61.34 | 61.62 | 62.29 |
| Gemini-1.5-Pro | fs-MSciNLI+ | **63.68** | **62.57** | 62.51 | **62.95** |

### Key Findings
- SLMs (SciBERT 78.17%) significantly outperform open-source LLMs (Phi-3 57.16%) but lag behind closed-source LLMs (Gemini 62.95%), indicating that fine-tuning smaller models still holds an advantage in scientific NLI.
- Training with MSciNLI+ (which includes implicit relations) consistently beats training on MSciNLI—with BERT rising from 67.89% to 71.41% (+3.52%), proving that implicit relation training data is indeed beneficial.
- The best SLM baseline is still only 78.17%, showing significant room for improvement in scientific NLI within non-CS domains.
- Public Health generally yields the lowest performance across all domains, likely due to a higher presence of domain-specific terminology.
- LLMs perform near random levels (~50%) under zero-shot settings, though few-shot prompts yield substantial increases (+10%+).

## Highlights & Insights
- Clear benchmark design methodology: formulated as an OOD test set, similar to the mismatched portion of MNLI, specifically designed to test generalization capabilities.
- The discovery of implicit relations offers generalizable value: any NLI dataset using distant supervision might overlook implicit relations, and incorporating them can boost training performance.
- Strict quality control in data construction: Fleiss-$\kappa$ of 0.72 + iterative annotation + retaining only samples with consistent automatic/human consensus.

## Limitations & Future Work
- Only covers three non-CS domains; future work can expand to more fields (e.g., biology, chemistry, economics, etc.).
- Contains only dev/test sets without a training set, limiting the exploration of domain adaptation approaches.
- Annotation consistency for the Neutral class was relatively low, which could introduce some noise.

## Related Work & Insights
- **vs SciNLI (Sadat & Caragea, 2022)**: Expanded to non-CS domains with equal per-domain test sizes (800), though with a smaller dev set.
- **vs MSciNLI (Sadat & Caragea, 2024)**: Although the latter expanded CS sub-domains, it remained confined within CS, whereas this work steps outside the CS boundary.
- **vs MNLI (Williams et al., 2018)**: Shares a similar design philosophy—matched tests in in-distribution domains, mismatched tests probing cross-domain generalization.

## Rating
- Novelty: ⭐⭐⭐ The methodology is a natural extension of existing work, but it fills a genuine gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across SLM and LLM baselines with zero-shot, few-shot, and fine-tuning setups, combined with ablation of implicit relations.
- Writing Quality: ⭐⭐⭐⭐ Highly detailed description of data construction with informative tables.
- Value: ⭐⭐⭐⭐ Provides a much-needed benchmark for cross-domain evaluation in scientific NLI; findings on implicit relations offer instructive insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
