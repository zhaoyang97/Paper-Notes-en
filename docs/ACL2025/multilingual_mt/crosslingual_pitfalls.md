---
title: >-
  [Paper Note] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][cross-lingual weakness] This paper proposes an automated approach based on beam search and LLM simulation to efficiently generate bilingual question pairs that expose cross-lingual performance weaknesses of multilingual LLMs in target languages. It establishes a dataset of over 6,000 samples across 16 languages, revealing that even GPT-4o suffers a cross-lingual accuracy drop exceeding 30%.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "cross-lingual weakness"
  - "multilingual LLM"
  - "beam search"
  - "bilingual question pairs"
  - "language affinity"
date: 2026-05-08
content_hash: ae75bf40ea0d376a
---

# Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.18673](https://arxiv.org/abs/2505.18673)  
**Code**: [GitHub](https://github.com/xzx34/Cross-Lingual-Pitfalls)  
**Area**: Multilingual / LLM Evaluation  
**Keywords**: cross-lingual weakness, multilingual LLM, beam search, bilingual question pairs, language affinity  

## TL;DR

This paper proposes an automated approach based on beam search and LLM simulation to efficiently generate bilingual question pairs that expose cross-lingual performance weaknesses of multilingual LLMs in target languages. It establishes a dataset of over 6,000 samples across 16 languages, revealing that even GPT-4o suffers a cross-lingual accuracy drop exceeding 30%.

## Background & Motivation

- **Problem Definition**: Cross-lingual weakness is defined as a scenario where a model answers a semantically identical question correctly in English but incorrectly in at least one target language. This reflects the cross-lingual inconsistency of LLMs.
- **Limitations of Prior Work**: Existing multilingual evaluation benchmarks are mostly static translated datasets, which fail to precisely locate a model's weak points in specific languages. Direct translation of English questions often fails to identify performance gaps (as cross-lingual performance differences on simple questions are negligible).
- **Core Motivation**: Systematically "searching" for questions that the model answers correctly in English but incorrectly in other languages can precisely diagnose cross-lingual weaknesses, providing a basis for targeted improvements.
- **Key Challenge**: How to generate bilingual question pairs that maintain semantic equivalence while maximizing the performance gap between English and the target language? Direct exhaustive search is prohibitively expensive.

## Method

### Overall Architecture

The overall pipeline consists of four steps: (1) sampling questions from high-quality English datasets and translating them into target languages to form bilingual pairs; (2) iteratively perturbing English questions to increase cognitive complexity; (3) evaluating via multiple LLM simulations to calculate a simulation score for measuring perturbation effectiveness; (4) iteratively optimizing using a beam search strategy to filter question pairs that best expose cross-lingual weaknesses.

### Key Designs

1. **Perturbation Function**: Given an English question $q^E$ and a distractor (incorrect option) $\alpha^E$, a surrogate LLM generates a semantically irrelevant but contextually plausible perturbation $\delta q^E = \varphi(q^E, \alpha^E)$, which is appended to the original question to increase cognitive load. Meanwhile, a translation module provides an equivalent translation of the perturbation to ensure cross-lingual semantic consistency.
2. **LLM Simulation Score**: $K$ LLMs are used to answer the English and target-language versions of the perturbed question. The score is computed as $V(q^{E'}, q^{T'}) = (\bar{\beta}^{E'})^\gamma - \bar{\beta}^{T'}$. The exponential amplification with $\gamma > 1$ ensures that only samples with high English accuracy and low target-language accuracy are selected.
3. **Beam Search Optimization Strategy**: This comprises three mechanisms: (a) Inclusion Threshold: samples with scores exceeding $\theta_{inc}$ are directly added to the candidate list; (b) Early Stopping: the search depth is expanded only when a sample score exceeds $\theta_{pot}$, otherwise restricting it to the initial depth; (c) Redundancy Control: an upper limit $r$ is imposed on the number of candidates derived from the same source question to ensure diversity.

### Loss & Training

$$\min_{\delta q^E} \mathbb{E}[\mathbb{I}(\mathcal{F}(q^{T'}) = a_\star^T)] \quad \text{s.t.} \quad \mathbb{E}[\mathbb{I}(\mathcal{F}(q^{E'}) = a_\star^E)] \geq 1-\epsilon, \quad \mathbb{S}(q^E, q^{E'}) \geq \theta$$

This formulates minimizing target-language accuracy while maintaining English accuracy above $1-\epsilon$ and ensuring that semantic similarity is no less than the threshold $\theta$.

## Experiments

### Main Results

Evaluation of 10 LLMs on 6,600 bilingual pairs across 16 languages:

| Model | English Accuracy | Chinese Accuracy | Accuracy Drop |
|------|-----------|-----------|----------|
| Gemma-2-9B | ~100% | ~35% | >60% |
| LLaMA-3.1-8B | ~100% | ~30% | >70% |
| Qwen2.5-7B | ~100% | ~45% | >55% |
| GPT-4o-mini | ~100% | ~55% | >45% |
| GPT-4o | ~100% | ~70% | ~30% |
| Claude-3.5-sonnet | ~100% | ~65% | >30% |

Even the strongest model, GPT-4o, exhibits a ~30% accuracy drop in Chinese; most models show an average accuracy decline of over 50% in the target languages.

### Ablation Study

| Method | Chinese Performance | Japanese Performance | French Performance | German Performance |
|------|----------|----------|----------|----------|
| NP (No Perturbation) | 0.000 | 0.000 | 0.000 | 0.000 |
| DP (Direct Perturbation) | 0.036 | 0.071 | 0.018 | 0.027 |
| **Beam Search (Ours)** | **0.431** | **0.594** | **0.132** | **0.323** |

The conversion rate of Beam Search significantly outperforms both the no-perturbation (NP) and direct-perturbation (DP) baselines, validating the effectiveness of the proposed search strategy.

### Key Findings

- **Language Affinity Impacts Shared Weaknesses**: Asian languages (Chinese, Japanese, Korean) share similar cross-lingual weaknesses; the same holds for European languages (French, German, Spanish). There is less sharing of weaknesses across different language families.
- **Stronger Fine-Tuning Transfer in Similar Languages**: Fine-tuning on French data yields significantly higher improvements in German/Italian than in Chinese/Japanese. Conversely, fine-tuning on Chinese shows larger improvements in Japanese and Korean.
- **Extremely Low Generation Costs**: For most languages, finding a bilingual pair that exposes a weakness costs less than $0.05 on average. However, the cost increases for languages structurally closer to English (e.g., French, Spanish).
- The **Relative Affinity Score (RAS)** metric clearly characterizes language affinity and shared weakness patterns.

## Highlights & Insights

- This work is the first to propose a systematic "search + perturbation" methodology to discover cross-lingual weaknesses of multilingual LLMs automatically, far surpassing simple translation-based evaluations.
- A large-scale evaluation spanning 16 languages reveals pervasive cross-lingual performance deficiencies in state-of-the-art models like GPT-4o and Claude-3.5.
- The study establishes a quantitative relationship between language affinity and cross-lingual weaknesses, introducing the RAS metric to measure language similarity.
- The approach is highly cost-effective (approx. $0.05 per perturbed question), offering high scalability and practicality.

## Limitations & Future Work

- Currently, evaluation is limited to Multiple-Choice Question (MCQ) formats, failing to cover more complex task formats such as generative QA or machine translation.
- The perturbation strategy relies heavily on the capabilities of the surrogate LLM (GPT-4o-mini), potentially limiting the diversity and quality of perturbations.
- Only a limited set of LLMs was tested as simulators; the weakness distribution might shift if other simulators are deployed.
- Some cross-lingual weaknesses might stem from machine translation quality (Google Translate) rather than intrinsic deficiencies of the target models.

## Related Work & Insights

- **Multilingual LLM Evaluation**: Benchmarks like MEGA (Ahuja et al., 2023) and XTREME (Hu et al., 2020) provide multilingual evaluation but focus on static assessment, failing to target and probe specific weaknesses.
- **Adversarial Evaluation**: Works like AdvGLUE (Wang et al., 2021) focus on adversarial robustness evaluations for English; this work extends adversarial search to cross-lingual scenarios.
- **Cross-Lingual Transfer Learning**: Studies such as mBERT (Pires et al., 2019) and XLM-R (Conneau et al., 2020) investigate zero-shot transfer across languages; this work addresses transfer limitations from an evaluative perspective.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Practicality | 8 |
| **Total Score** | **8.3** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [\[ACL 2025\] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)

</div>

<!-- RELATED:END -->
