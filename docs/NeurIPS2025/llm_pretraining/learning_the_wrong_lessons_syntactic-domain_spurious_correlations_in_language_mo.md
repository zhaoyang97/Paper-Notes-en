---
title: >-
  [Paper Note] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models
description: >-
  [NeurIPS 2025][LLM Pretraining][spurious correlations] This paper reveals that LLMs learn spurious correlations between syntactic templates (PoS n-grams) and domains…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "spurious correlations"
  - "syntactic templates"
  - "LLM safety"
  - "jailbreak attacks"
  - "instruction fine-tuning"
  - "domain generalization"
date: 2026-05-08
content_hash: 62f6e944e07ce388
---

# Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.21155](https://arxiv.org/abs/2509.21155)  
**Code**: To be confirmed  
**Area**: LLM Pre-training
**Keywords**: spurious correlations, syntactic templates, LLM safety, jailbreak attacks, instruction fine-tuning, domain generalization

## TL;DR

This paper reveals that LLMs learn spurious correlations between syntactic templates (PoS n-grams) and domains, leading to sharp performance drops in cross-domain settings. Furthermore, this correlation can be exploited to bypass safety refusal mechanisms, reducing the refusal rate from 40% to 2.5% on OLMo-2.

## Background & Motivation

LLMs are widely deployed in domains such as healthcare and finance, where reliability requires models to genuinely understand the **semantics** of instructions and the underlying **domain** knowledge. However, recent studies have found that **syntactic templates** (i.e., frequently occurring PoS tag n-grams) present in training data are learned by models and reproduced in their outputs.

This raises a critical question: **Do LLMs truly leverage semantic and domain knowledge, or do they merely exploit shallow syntactic patterns?**

The authors illustrate this with a motivating example: when asked "Where is Paris located?", the model correctly answers "France." However, when the sentence's semantics are completely disrupted (e.g., "Quickly sit Paris clouded?") while preserving the same PoS template (ADV VB NNP VBD), the model **still answers "France."** This demonstrates that models over-rely on syntactic-domain correlations rather than semantic understanding.

More critically, this dependency can be exploited to bypass safety refusal mechanisms.

## Method

### Overall Architecture

The authors formalize the interaction among syntactic templates, domains, and semantics, and propose a definition and detection framework for **syntactic-domain spurious correlations**.

### Key Definitions

Each sample is represented as a triple $x = (d, t, e)$, where $d$ denotes the domain, $t$ the syntactic template (PoS tag sequence), and $e$ the entity (semantic content).

**Spurious correlation definition**: A syntactic template $\tau$ is spuriously predictive in domain $d$ when its conditional frequency substantially exceeds its marginal frequency:

$$P(\tau | d) \gg P(\tau)$$

### Five Prompt Perturbations

To quantify the degree of model reliance on syntax, five prompt variants are designed:

1. **Exact**: The precise template used during training
2. **Synonym**: Semantics preserved + syntax preserved (synonym substitution)
3. **Antonym**: Semantics disrupted + syntax preserved (antonym substitution)
4. **Disfluent**: Semantics disrupted + syntax preserved (insertion of random same-PoS words)
5. **Paraphrase**: Semantics preserved + syntax changed (rephrasing)

These are grouped into three categories:
- $\mathcal{P}_{\text{Semantic Preserving}}$: Exact, Synonym
- $\mathcal{P}_{\text{Semantic Breaking}}$: Antonym, Disfluent
- $\mathcal{P}_{\text{Utility}}$: Paraphrase

### Quantifying Syntactic-Domain Dependency

Domain risk is defined as:

$$R_{M_\theta}(d) = \mathbb{E}_{(p,t,e) \sim X_d}\left(\mathbb{E}_{p^- \sim \mathcal{P}_{\text{SB}}} M(e|p^-) + \mathbb{E}_{p^+ \sim \mathcal{P}_{\text{SP}}} M(e|p^+)\right)$$

Two criteria are used: (1) high performance on in-domain semantics-preserving prompts; (2) a large gap between in-domain and cross-domain risk.

### Detection Framework (3 Steps)

1. **Template extraction**: Extract PoS templates from training data (e.g., FlanV2)
2. **Test set construction**: Generate 5 perturbation variants for each entity-template pair, yielding $n \times m \times 4$ prompts in total
3. **Correlation measurement**: Templates for which the model predicts correctly are classified as "in-domain"; the rest as "cross-domain"; performance gap is then compared

### Behavior Taxonomy

Six instruction-following behavior patterns are defined:
- **Correct behavior**: High on Exact/Synonym/Paraphrase, low on Antonym/Disfluent
- **Entity memorization**: High across all settings and all domains
- **Prompt memorization**: High only on Exact
- **Word-domain correlation**: High on Exact/Synonym in-domain, low on Antonym/Disfluent in-domain
- **Syntactic-domain correlation**: High on Antonym/Disfluent in-domain as well (the key distinguishing pattern)

### Security Implications: Refusal Bypass

The syntactic-domain correlation is exploited to bypass safety mechanisms by replacing the PoS template of a harmful request with a template from a "safe" domain (e.g., chain-of-thought) and prepending or appending it to the input.

## Key Experimental Results

### Main Results: Synthetic Data Training on OLMo-2

| Model | Setting | Exact | Synonym | Antonym | Disfluent | Paraphrase |
|-------|---------|-------|---------|---------|-----------|------------|
| OLMo-2-1B Instruct | In-domain | 0.93 | 0.91 | 0.90 | 0.24 | 0.53 |
| | Cross-domain | 0.42 | 0.40 | 0.41 | 0.25 | 0.44 |
| | **Δ** | **↓0.51** | **↓0.51** | **↓0.49** | ↑0.01 | ↓0.09 |
| OLMo-2-13B Instruct | In-domain | 0.94 | 0.93 | 0.93 | 0.13 | 0.84 |
| | Cross-domain | 0.40 | 0.42 | 0.56 | 0.24 | 0.50 |
| | **Δ** | **↓0.54** | **↓0.51** | **↓0.37** | ↑0.11 | ↓0.34 |

Key finding: Cross-domain performance drops by approximately **0.40–0.60**, and neither model scale nor instruction fine-tuning alleviates this. The in-domain Antonym score of 0.93 demonstrates that syntax can override semantics.

### Validation on Real Models (FlanV2 Sentiment140)

| Model | Synonym In-domain→Cross-domain | Δ |
|-------|-------------------------------|---|
| OLMo-2-7B | 0.85→0.48 | ↓0.37 |
| GPT-4o-mini | 1.00→0.44 | **↓0.56** |
| GPT-4o | 0.69→0.36 | ↓0.33 |

Syntactic-domain spurious correlations are observed in both open-source and closed-source models.

### Safety Experiment: Refusal Bypass

| Template | Position | Exact Refusal Rate | Synonym | Antonym |
|----------|----------|--------------------|---------|---------|
| Baseline | — | 0.400 | 0.400 | 0.400 |
| CoT | Prefix | **0.025** | 0.357 | 0.195 |
| CoT | Suffix | 0.129 | 0.382 | 0.259 |
| Math | Prefix | 0.481 | 0.251 | 0.189 |

Using a chain-of-thought template as a prefix reduces the refusal rate of OLMo-2-7B-Instruct from 40% to **2.5%**.

### Key Findings

1. Syntax can override semantics: In-domain performance under the Antonym setting is comparable to that under Synonym
2. Model scale does not resolve the issue: Cross-domain performance drops are similar from 1B to 13B
3. Llama-4-Maverick behaves differently: smaller cross-domain drops, but manifests as entity memorization rather than syntactic-domain correlation
4. Serious safety implications: A CoT template prefix reduces the refusal rate by 37.5 percentage points

## Highlights & Insights

1. **Identifies a novel type of spurious correlation**: Syntactic-domain correlation is a previously unrecognized failure mode in LLMs, analogous to background-label correlations in computer vision but more subtle
2. **Rigorous formalization**: The triple representation, five perturbation types, and six-category behavior taxonomy provide a standardized evaluation framework for future research
3. **From theory to security**: The work goes beyond an academic finding by directly translating into a practical security vulnerability (jailbreak), with impact well beyond conventional analysis work
4. **Covers both open- and closed-source models**: The phenomenon is validated across OLMo-2, Llama-4, and GPT-4o, demonstrating its generality
5. **Actionable recommendations**: (1) Test for syntactic-domain correlations; (2) Ensure syntactic diversity within each domain in training data

## Limitations & Future Work

1. **Speculative conclusions for closed-source models**: It cannot be confirmed whether GPT-4o/Llama-4 were trained on FlanV2; cross-domain drops may have alternative explanations
2. **Reasoning models not covered**: Chain-of-thought reasoning models may exhibit different syntactic dependency patterns
3. **Limited template granularity**: Only PoS tag-level templates are used; finer-grained syntactic structures remain unexplored
4. **Simplifying assumptions in synthetic data**: Disjoint template sets across domains are assumed, whereas real-world data likely exhibits significant overlap
5. **Defense strategies underexplored**: Only the high-level suggestion of increasing syntactic diversity is offered, without concrete implementation or empirical validation

## Related Work & Insights

- Distinction from classical NLP spurious correlation work (McCoy et al. 2019): This paper focuses on PoS templates rather than lexical patterns, and covers large models after multi-stage training
- Connection to Shaib et al. 2024: That work shows LLMs learn and repeat syntactic templates; this paper further demonstrates that such templates form spurious correlations with domains
- Implications for data curation: Instruction fine-tuning datasets (e.g., Flan) should ensure diverse template formats within each domain
- Impact on jailbreak research: Provides a systematic attack approach grounded in the structure of training data

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic identification of syntactic-domain spurious correlations, with a complete formalization framework
- **Practicality**: ⭐⭐⭐⭐ — Provides a detection framework and security insights, though defense strategies remain to be developed
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual validation on synthetic and real data with multi-model coverage, though conclusions on closed-source models are limited
- **Writing Quality**: ⭐⭐⭐⭐ — Intuitive examples, clear framework, and impactful security case study
- **Recommended Reading**: ⭐⭐⭐⭐⭐ — Essential reading for researchers in LLM safety, robustness, and instruction fine-tuning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)

</div>

<!-- RELATED:END -->
