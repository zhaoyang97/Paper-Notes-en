---
title: >-
  [Paper Note] Nudging: Inference-time Alignment of LLMs via Guided Decoding
description: >-
  [ACL 2025][LLM (Other)][Inference-time alignment] This paper proposes Nudging, a training-free inference-time alignment algorithm. It utilizes a small aligned model to inject a small number of "nudging tokens" to guide the output when the base model is uncertain, achieving or even surpassing the performance of large aligned models with a model that is 7-14 times smaller.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Inference-time alignment"
  - "guided decoding"
  - "training-free"
  - "token-level collaboration"
  - "model composition"
date: 2026-05-08
content_hash: c512976cbd09bced
---

# Nudging: Inference-time Alignment of LLMs via Guided Decoding

**Conference**: ACL 2025  
**arXiv**: [2410.09300](https://arxiv.org/abs/2410.09300)  
**Code**: [https://fywalter.github.io/nudging/](https://fywalter.github.io/nudging/)  
**Area**: LLM Alignment  
**Keywords**: Inference-time alignment, guided decoding, training-free, token-level collaboration, model composition

## TL;DR

This paper proposes Nudging, a training-free inference-time alignment algorithm. It utilizes a small aligned model to inject a small number of "nudging tokens" to guide the output when the base model is uncertain, achieving or even surpassing the performance of large aligned models with a model that is 7-14 times smaller.

## Background & Motivation

**Background**: LLMs require alignment (instruction tuning + RLHF) after pre-training to effectively follow user instructions. The current pipeline requires training an aligned version separately for each base model, which is extremely costly, especially for the largest models (e.g., RLHF for Tulu 3 405B requires 11,776 H100 GPU hours).

**Limitations of Prior Work**: Whenever a new model family or a larger scale model is introduced, alignment training must start from scratch, hindering rapid iteration and deployment. Existing inference-time tuning methods (e.g., Proxy Tuning), though training-free, are slow (10-20 times slower than Nudging).

**Key Challenge**: Alignment primarily alters the model's behavior only on a small number of "stylistic tokens" (e.g., discourse markers), yet full training of the entire large model is required for this.

**Goal**: Achieve alignment of large base models during inference using small aligned models without any training.

**Key Insight**: Based on the key finding that base models exhibit significantly higher uncertainty at alignment-related token positions (disagreeing with the aligned model in 90% of cases when the top-1 probability is $< 0.1$), an uncertainty threshold can accurately predict where intervention is needed.

**Core Idea**: "Nudge" the base model when it is uncertain, using tokens from a small aligned model to guide the large model in the correct direction.

## Method

### Overall Architecture

Given a base model and a nudging model, outputs are generated via token-level collaboration: the base model decodes normally $\to$ detect if the top-1 probability is below the threshold $\gamma$ $\to$ if below, the nudging model generates a "nudging word" $\to$ the base model resumes decoding from the new prefix.

### Key Designs

1. **Uncertainty Detection (Where to Nudge)**: Analysis shows that when the top-1 probability of the base model is below 0.5, it captures over 80% of alignment-related positions, which account for only about 11% of all positions. Thus, a fixed threshold $\gamma$ is set (0.4 for Llama-2, 0.3 for Gemma-2/OLMo).
2. **Cross-Model Token Substitution (What to Nudge)**: Large and small aligned models exhibit highly similar token distributions at alignment-related positions (agreement rates of 65-83% for Llama-2 and 58-88% for Gemma-2). Therefore, small aligned models can serve as substitutes for large aligned models. The first complete "word" delimited by spaces is taken as the nudging token, supporting collaboration between models with different tokenizers.
3. **Termination Detection**: The nudging model generates a lookahead completion of $L$ tokens. If it produces `[EOS]`, the entire output is accepted and decoding terminates; otherwise, only the first word is taken. Through prefix caching, the extra overhead is controlled at approximately 15%.

### Loss & Training

Fully training-free. Nudging is a pure decoding-time algorithm requiring no parameter updates. All experiments use greedy decoding.

## Key Experimental Results

### Main Results

Zero-shot performance of 3 model families across 11 standard benchmarks:

| Model Family | Base Model | Nudging Model | Nudging Avg | Large Aligned Avg |
|---|---|---|:---:|:---:|
| Llama-2 | 70b | 7b-chat | **57.9** | 56.7 |
| Gemma-2 | 27b | 2b-it | 70.3 | 74.4 |
| OLMo | 7b | 1b-it | **40.8** | 39.2 |

Key data points (single task performance):

| Task | Gemma-2-27b | Gemma-2-2b-it | Nudging | Gemma-2-27b-it |
|---|:---:|:---:|:---:|:---:|
| LastLetterConcat | 6.7% | 4.7% | **86.0%** | 82.0% |
| CoinFlip | 7.6% | 33.9% | **42.7** | 74.3 |
| GSM8K | 6.7% | 63.8% | **74.6** | 85.4 |

### Ablation Study

Comparison with other inference-time tuning methods:

| Method | Llama-2 | Gemma-2 | OLMo | Speed (Relative) |
|---|:---:|:---:|:---:|:---:|
| Ensemble | 48.0 | 65.9 | 36.9 | 10.6× |
| Proxy Tuning | 53.2 | 61.2 | 36.3 | 18.6× |
| **Nudging** | **58.0** | **70.9** | **42.0** | **1×** |

### Key Findings

- Nudging affects only about 10% of the output tokens, with an extra runtime of only approximately 15%.
- Performance gains are especially significant on math and symbolic reasoning tasks: on LastLetterConcat, Gemma-2 surges from 6.7% to 86%.
- **Cross-family collaboration is effective**: Gemma-2-27b + Llama-2-7b-chat outperforms Llama-2-70b-chat on several tasks.
- Aligned models tend to give conservative answers like "50% probability" (e.g., in the CoinFlip task); Nudging avoids this issue by preserving the reasoning capability of the base model.
- Comparison with In-context Alignment: Nudging significantly outperforms ICL across all model families (Llama-2: 57.9 vs 47.6).

## Highlights & Insights

- Translates the academic observation that "alignment only changes a few tokens" into a practical engineering solution, beautifully combining insight with application.
- Token-level collaboration across model families is a brand-new direction, breaking the restriction that model composition must stay within the same family.
- Nudging can "decouple" pre-training and alignment capabilities: OLMo-7b-it performs worse than the base model on GSM8K (14.1 vs 18.8), but Nudging preserves the reasoning ability of the base model while injecting alignment behaviors.

## Limitations & Future Work

- The threshold $\gamma$ requires manual tuning for different model families (0.3-0.4), lacking an adaptive mechanism.
- A performance gap still exists between Nudging and the large aligned model on Gemma-2 (70.3 vs 74.4).
- Validation was only performed on 7B-70B scales; the applicability to 100B+ models is unconfirmed.
- Safety evaluation is preliminary, and adversarial robustness has not been tested.

## Related Work & Insights

- Relationship to Proxy Tuning: Both are training-free inference-time methods, but Nudging operates at the token level (rather than the distribution level), making it faster and generally better.
- Connection to Speculative Decoding: They share prefix caching techniques, but differ in goal—the latter pursues acceleration while the former pursues alignment.
- Inspiration: The token-level model collaboration paradigm has the potential to generalize to more scenarios (e.g., multilingual, multimodal).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Inference-time token-level alignment is a completely new paradigm, with a natural and elegant insight-driven design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 model families $\times$ 13 datasets $\times$ multiple baselines, with cross-family experiments as a major plus.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough analysis, linking observations, method, and experiments coherently.
- Value: ⭐⭐⭐⭐⭐ Highly practical, significantly reduces alignment costs, and opens up new research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](../../ICLR2026/llm_nlp/enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[ACL 2025\] Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](bitnetcpp_efficient_edge_inference_for_ternary_llms.md)

</div>

<!-- RELATED:END -->
