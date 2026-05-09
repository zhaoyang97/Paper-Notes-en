---
title: >-
  [Paper Note] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing
description: >-
  [CVPR 2026][LLM Safety][Composed Image Retrieval] This paper proposes the PinPoint benchmark, comprising 7,635 queries and 329K human-verified relevance judgments. Through four dimensions—explicit negatives, multi-image queries, paraphrase variants, and demographic metadata—it exposes severe deficiencies in existing CIR methods regarding false positive suppression, linguistic robustness, and multi-image reasoning. A training-free MLLM-based reranking method is also proposed as an improved baseline.
tags:
  - CVPR 2026
  - LLM Safety
  - Composed Image Retrieval
  - evaluation benchmark
  - explicit negatives
  - multi-image queries
  - linguistic robustness
date: 2026-05-08
content_hash: a48ec27e580e1381
---

# PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing

**Conference**: CVPR 2026
**arXiv**: [2603.04598](https://arxiv.org/abs/2603.04598)
**Code**: None (dataset and evaluation code open-sourced)
**Area**: AI Safety
**Keywords**: Composed Image Retrieval, evaluation benchmark, explicit negatives, multi-image queries, linguistic robustness

## TL;DR

This paper proposes the PinPoint benchmark, comprising 7,635 queries and 329K human-verified relevance judgments. Through four dimensions—explicit negatives, multi-image queries, paraphrase variants, and demographic metadata—it exposes severe deficiencies in existing CIR methods regarding false positive suppression, linguistic robustness, and multi-image reasoning. A training-free MLLM-based reranking method is also proposed as an improved baseline.

## Background & Motivation

**Fundamental flaws of existing CIR benchmarks**: Benchmarks such as CIRR and FashionIQ provide only a single correct answer per query, and Recall-based evaluation ignores false positives. For example, returning 2 relevant and 8 irrelevant items in the top-10 yields the same score as returning 10 fully relevant results (Recall@10 = 1.0, but Precision@10 = only 0.20). The absence of explicit negative annotations prevents assessment of false positive suppression capability.

**Complexity of real retrieval scenarios**: Users may compose queries using multiple reference images (e.g., "an outfit featuring [this dress] and [these shoes]"), and the same semantic intent can be expressed with different phrasings ("change to blue" vs. "switch the color to blue"). Existing benchmarks cannot evaluate these abilities.

**Inherent multi-answer nature**: A composed query (e.g., "change this shirt to blue") may have dozens of valid matches; assuming a single correct answer cannot measure true ranking quality.

**Limitations of CIRCO**: While CIRCO introduces multiple positive samples, it lacks explicit negatives and contains only approximately 800–1,000 queries, which is insufficient for comprehensive evaluation.

## Method

### Overall Architecture

PinPoint is an **evaluation benchmark** rather than a retrieval model; its core contributions lie in dataset construction and evaluation protocols:
1. **Dataset construction**: 25K candidate query images → quality filtering → 7,635 queries + 109,601-image corpus
2. **Evaluation framework**: Comprehensive evaluation of 20+ methods across 4 paradigms (CLIP baselines, CIR-specific, proxy generation, reranking)
3. **Improved baseline**: Training-free pointwise MLLM reranking

### Key Designs

1. **Dataset Construction Pipeline**

    - **Modification instruction generation**: Three MLLMs (GPT-5, Claude 4 Sonnet, Gemini 2.5 Pro) each generate 5 candidate instructions (15 total) → deduplication and filtering (specificity, visual relevance, topic alignment, linguistic quality) → human verification. Covers 5 intent types: Explore / Swap / Negation / Context Fit / Complement.
    - **Paraphrase generation**: 6 paraphrase variants are generated per instruction, varying in verbosity (concise vs. detailed) and register (imperative vs. interrogative). All paraphrases share the same positive and negative annotations to measure linguistic robustness.
    - **Multi-answer annotation + explicit negatives**: Three MLLMs propose correct target descriptions and potential false-positive descriptions → up to 50 candidates retrieved per description (~100 per query) → three models independently rate on a 5-point scale → unanimous "highly relevant" votes retained as positives, unanimous "false positive" votes retained as negatives → final human verification. Average: **9.1 positives + 32.8 explicit negatives per query**.
    - **Three-layer LLM bias mitigation**: (1) Full human verification (37% of LLM proposals rejected); (2) three-model consensus (no reliance on a single model); (3) LLMs enable scale, humans ensure quality.

2. **Novel Evaluation Metric Design**

    - **ΔmAP@10**: $\Delta\text{mAP@10} = \text{mAP@10}_{\text{no\_hn}} - \text{mAP@10}_{\text{all}}$, measuring the impact of explicit negatives on retrieval performance; a robust model yields a value close to 0.
    - **Negative Recall@10**: Frequency of false positives appearing in the top-10 results, directly quantifying false positive severity.
    - **Linguistic Sensitivity**: Difference between the maximum and minimum mAP@10 across 6 paraphrases; a lower value indicates higher robustness.

3. **Training-Free MLLM Reranking**

    - **Function**: Pointwise scoring and reranking of first-stage retrieval results using Qwen2.5-VL-7B.
    - **Mechanism**: For each candidate image, the query image, instruction, and candidate image are fed as input; the model generates a relevance response, and the score is computed as the sigmoid of the logit difference between "yes" and "no" tokens: $P(\text{relevant}|I_c) = \sigma(\ell_{\text{yes}} - \ell_{\text{no}})$.
    - **Latency**: Approximately 120 ms per candidate on a single GPU using KV-cache prefill.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Base queries | 7,635 |
| Corpus images | 109,601 |
| Avg. positives per query | 9.1 |
| Avg. negatives per query | 32.8 |
| Multi-image query proportion | 13.4% |
| Paraphrases per query | 6 |
| Domain categories | 23 |
| Demographic annotation | Monk Skin Tone |

## Key Experimental Results

### Main Results (Performance Overview of 20+ Methods)

| Method | mAP@10 | ΔmAP(%)↓ | NegRecall@10↓ | Linguistic Sensitivity↓ |
|--------|--------|----------|---------------|------------------------|
| Meta CLIP 2 – Combined | 0.044 | 39.87 | 0.072 | 0.114 |
| LinCIR | 0.110 | 23.47 | 0.141 | 0.152 |
| MagicLens-CLIP-L | 0.155 | 14.41 | 0.151 | 0.182 |
| MMRet-CLIP-L | 0.178 | 10.89 | 0.120 | 0.188 |
| MMRet-MLLM-S1 | 0.224 | 6.38 | 0.091 | 0.162 |
| GPT-5-Text Premerge | 0.266 | 6.93 | 0.090 | 0.174 |
| MMRet-MLLM-S1 + Reranking | **0.290** | **2.01** | **0.056** | 0.191 |

### Ablation Study: Universal Gains from MLLM Reranking

| Method | w/o Reranking | +Reranking | NegRecall Change |
|--------|--------------|------------|-----------------|
| Meta CLIP 2 Combined | 0.044 | 0.087 (+98%) | 0.072→0.039 |
| MMRet-CLIP-L | 0.178 | 0.236 (+33%) | 0.120→0.074 |
| GPT-5-Text Premerge | 0.266 | 0.272 (+2%) | 0.090→0.062 |
| MMRet-MLLM-S1 | 0.224 | 0.290 (+29%) | 0.091→0.056 |

### Performance Collapse on Multi-Image Queries

| Method | Single-Image mAP@10 | Multi-Image mAP@10 | Performance Drop |
|--------|--------------------|--------------------|-----------------|
| MMRet-MLLM-S1 | 0.324 | 0.067 | **4.83×** |
| MMRet-CLIP-L | 0.262 | 0.063 | 4.15× |
| MagicLens-L | 0.257 | 0.062 | 4.14× |
| LinCIR | 0.121 | 0.042 | 2.88× |

### Key Findings

- **Severe false positive problem**: The best method (with reranking) still yields a 5.6% false positive rate in the top-10; the best CIR method without reranking reaches 9.1%.
- **Linguistic robustness paradox**: High-performing models exhibit 3–5× higher linguistic sensitivity than CLIP baselines (MMRet-MLLM-S1: 0.162 vs. Meta CLIP 2: 0.114), suggesting overfitting to specific phrasings in existing benchmarks.
- **Multi-image queries remain unsolved**: All models show a 48–72% performance drop on multi-image queries, which reranking cannot compensate for.
- **Surprisingly strong pure-text GPT-5 baseline**: GPT-5 generating target descriptions followed by text-based retrieval achieves mAP@10 = 0.266, outperforming most CIR-specific methods.
- **Double-edged effect of reranking**: MLLM reranking consistently improves mAP and false positive suppression, but universally worsens linguistic sensitivity (+10–30%).

## Highlights & Insights

1. **Exposing the blind spot of Recall metrics**: The extreme case of Recall@10 = 1.0 but NegRecall@10 = 0.6 demonstrates that existing benchmarks have been "faking progress."
2. **Precision–safety trade-off**: CIR-specific training improves mAP by 3.4× but increases false positive rates by 25%—current training paradigms prioritize positive matching while neglecting negative suppression.
3. **Dataset construction methodology**: The three-layer bias mitigation strategy combining three-model consensus with human verification serves as a paradigm for constructing high-quality multimodal benchmarks.
4. **Effectiveness of the GPT-5 text proxy**: This finding suggests that the visual understanding capability of current CIR methods may be inferior to simple text-based retrieval.

## Limitations & Future Work

1. The 23 domains are all lifestyle-oriented; specialized domains such as industrial design, medical imaging, and satellite imagery are absent.
2. Geographic and cultural bias toward Western concepts and English queries.
3. Multi-image queries are limited to two images; real-world scenarios may require 5 or more.
4. Only zero-shot evaluation is conducted; the effect of fine-tuning on PinPoint-like data remains unexplored.
5. The average of ~9.1 positive samples per query may still be insufficient for exhaustive coverage.

## Related Work & Insights

- **CIRR**: The first large-scale CIR benchmark; lacks explicit negatives and multi-answer support, and suffers from instruction leakage.
- **CIRCO**: Introduces multiple positive samples but lacks explicit negatives and has limited scale.
- **MMRet**: Currently the strongest CIR method; PinPoint exposes its weaknesses in false positive suppression and linguistic sensitivity.
- Insight: Advances in evaluation often drive field progress more effectively than advances in methods; explicit negatives are likely to become a standard component of future CIR training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The four-dimensional evaluation framework fills a significant gap in CIR assessment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 20+ methods, 4 paradigms, and comprehensive multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Dataset construction pipeline described in detail; analysis is thorough and examples are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — High potential impact as a new benchmark; findings can guide the design of next-generation CIR methods.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[AAAI 2026\] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](../../AAAI2026/llm_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/llm_safety/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)

<!-- RELATED:END -->
