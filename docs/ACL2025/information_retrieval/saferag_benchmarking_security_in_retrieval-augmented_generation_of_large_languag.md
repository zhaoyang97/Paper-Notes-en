---
title: >-
  [Paper Note] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG security] This work proposes SafeRAG, the first Chinese RAG security evaluation benchmark. It designs four novel attack tasks (silver noise, inter-context conflict, soft ad, and white DoS) capable of bypassing existing retrievers, filters, and generators. By systematically evaluating security vulnerabilities across 14 RAG components, it reveals that even state-of-the-art RAG systems are highly vulnerable to these attacks.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG security"
  - "silver noise"
  - "inter-context conflict"
  - "soft ad"
  - "white DoS"
date: 2026-05-08
content_hash: 84dfaf975c5e06fc
---

# SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model

**Conference**: ACL 2025  
**arXiv**: [2501.18636](https://arxiv.org/abs/2501.18636)  
**Code**: [GitHub](https://github.com/IAAR-Shanghai/SafeRAG)  
**Area**: Information Retrieval / RAG Security  
**Keywords**: RAG security, silver noise, inter-context conflict, soft ad, white DoS

## TL;DR

This work proposes SafeRAG, the first Chinese RAG security evaluation benchmark. It designs four novel attack tasks (silver noise, inter-context conflict, soft ad, and white DoS) capable of bypassing existing retrievers, filters, and generators. By systematically evaluating security vulnerabilities across 14 RAG components, it reveals that even state-of-the-art RAG systems are highly vulnerable to these attacks.

## Background & Motivation

Retrieval-Augmented Generation (RAG) integrates external knowledge for LLMs through an "index-retrieve-generate" paradigm, which has been adopted by mainstream products such as ChatGPT, Gemini, and Perplexity. However, introducing unverified external knowledge increases the risk of attacks, as adversaries can inject manipulated texts at any stage of the RAG pipeline.

**Four major limitations of prior work**:

1. **Noise attacks are easily filtered (R-1)**: Existing noise consists of contexts that are only superficially relevant but do not contain answers, which can be effectively defended against by simple NLI filters.
2. **Conflict attacks target context-memory conflicts (R-2)**: LLMs can determine the authenticity of external documents using internal parametric knowledge, and adaptive retrievers have successfully mitigated this risk.
3. **Toxicity attacks are detected by generators (R-3)**: Advanced generators possess strong capabilities to detect explicit and implicit toxicity (e.g., bias, discrimination, sarcasm).
4. **DoS attack signals are conspicuous (R-4)**: Traditional DoS attacks directly inject refusal signals like "I don't know", which are readily intercepted by filters or ignored by generators due to their irrelevance to the questions.

Consequently, **existing security benchmarks cannot accurately evaluate the real-world security risks of RAG**. Furthermore, existing RAG security research focuses predominantly on English scenarios, while Chinese scenarios remain largely unexplored.

## Method

### Overall Architecture

SafeRAG consists of three core components:

1. **Attack Dataset Construction**: Collecting Chinese news texts $\to$ constructing a base dataset of 100 question-golden context pairs $\to$ generating adversarial texts tailored to each of the four attack tasks.
2. **Threat Framework**: Injecting attack texts into three stages of the RAG pipeline (knowledge base / retrieved contexts / filtered contexts) to simulate different attack scenarios.
3. **Security Evaluation**: Designing task-specific security metrics (RA, F1(avg), ASR/AFR) using DeepSeek as the evaluator, achieving over 89% consistency with human judgment.

### Key Designs

1. **Silver Noise**:

    - **Function**: Generates noisy contexts containing partial but incomplete answers to bypass security filters.
    - **Mechanism**: Decomposes golden contexts into fine-grained propositions (minimum semantic units), selects propositions most semantically similar to the question as the attack targets, and prompts DeepSeek to generate 10 diverse contexts based on these target propositions.
    - **Novelty**: Silver noise indeed contains partial answers, allowing it to pass relevance checks of filters. However, because the answers are incomplete, it compromises the information diversity and completeness of the RAG outputs.
    - **Design Motivation**: Existing noise only consists of "thematically related but answer-free" contexts, which are too easy to filter out. Silver noise creates more stealthy interference by being "partially correct".

2. **Inter-context Conflict**:

    - **Function**: Constructs contradictions among external knowledge sources, rendering LLMs unable to resolve the conflict using internal parametric knowledge.
    - **Mechanism**: Human annotators select the most easily manipulated golden contexts and modify them according to three principles: (a) minimum perturbation to introduce conflict (altering only key information); (b) rewriting to make conflicts more realistic and plausible; and (c) retaining key facts (year, month, etc.) to avoid generating hallucinations or irrelevant contexts.
    - **Novelty**: Existing studies primarily focus on context-memory conflicts (resolved via internal LLM knowledge). This work explores contradictions within external knowledge where LLMs lack sufficient parametric knowledge to judge, resulting in stronger attack capabilities.
    - **Design Motivation**: LLMs are more susceptible to being misled in knowledge areas where they are uncertain, which this inter-context conflict exploits.

3. **Soft Ad**:

    - **Function**: Seamlessly embeds seemingly professional and harmless commercial advertisement information into the golden contexts to bypass toxicity detection.
    - **Mechanism**: Two embedding methods: (a) direct insertion: inserting the soft ad text directly into the original context; (b) indirect insertion: modifying the original context to link the advertisement with authoritative entities (governments, institutions, etc.) for promotion, making it more natural and covert.
    - **Novelty**: As a form of implicit toxicity, soft ads bypass explicit toxicity detection of LLMs and are automatically propagated into the generated responses.
    - **Design Motivation**: Traditional toxic attacks (bias, discrimination, etc.) can be detected by advanced generators, whereas soft ads disguised as professional information are difficult to distinguish.

### Evaluation Metrics

**Retrieval Security Metric — Retrieval Accuracy (RA)**:

$$RA = \frac{\text{Recall}(gc) + (1 - \text{Recall}(ac))}{2}$$

Balances the ability to retrieve golden contexts ($gc$) and suppress adversarial contexts ($ac$); higher values indicate higher security.

**Generation Security Metric — F1(avg)**:

Constructed based on multiple-choice questions. The response and question are fed into the evaluator to assess the generator's ability to identify correct/incorrect options under attack. $F1(avg) = \frac{F1(correct) + F1(incorrect)}{2}$. A higher value represents better safety.

**Attack Success Rate — ASR/AFR**:

Measures the proportion of attack keywords (conflicting facts, soft advertisements, refusal signals) appearing in RAG responses. In experiments, a positive metric $AFR = 1 - ASR$ is adopted; higher values indicate higher safety.

### Loss & Training

This paper presents an evaluation benchmark rather than a training methodology. The core contribution lies in the construction of the attack dataset and the design of the evaluation framework:

- Base Dataset: News from August to September 2024 across five major domains (politics, finance, technology, culture, military) were collected from news websites. After filtering, 100 sets of question-8 golden contexts pairs were obtained.
- Chunking Strategy: Standard sentence-level chunking.
- Embedding Model: bge-base-zh-v1.5
- Reranking Model: bge-reranker-base
- Evaluator: DeepSeek (human alignment >89%)

## Key Experimental Results

### Main Results

The study evaluates 14 RAG components (4 retrievers × 3 filters × 8 generators), with attacks injected at three different stages.

**Effectiveness Ranking of Attack Tasks across Stages**:

| Attack Task | Attack Effectiveness by Stage | Explanation |
|---------|----------------|------|
| Silver Noise | Filtered Context > Retrieved Context > Knowledge Base | Injections closer to the generator are more hazardous |
| Inter-context Conflict | Filtered Context > Retrieved Context > Knowledge Base | Same as above |
| Soft Ad | Filtered Context > Retrieved Context > Knowledge Base | Same as above |
| White DoS | Filtered Context > Retrieved Context > Knowledge Base | Same as above |

**Retriever Robustness Rankings**:

| Attack Task | Retriever Robustness Ranking |
|---------|----------------|
| Silver Noise | Hybrid-Rerank > Hybrid > BM25 > DPR |
| Inter-context Conflict | DPR > BM25 > Hybrid > Hybrid-Rerank (Hybrid-Rerank is the most vulnerable) |
| Soft Ad | Underperforming across all retrievers with similar vulnerability |
| White DoS | Hybrid-Rerank > Hybrid > BM25 > DPR (DPR is the most vulnerable) |

### Ablation Study

Security of Filters/Compressors under different attack tasks:

| Component | Silver Noise Task | Conflict Task | Soft Ad Task | DoS Task |
|------|----------|---------|----------|---------|
| No Filter (OFF) | Insecure | Insecure | Insecure | Insecure |
| NLI Filter | ✅ Effective | ⚠️ Partially Effective | ❌ Nearly Ineffective | ❌ Nearly Ineffective |
| SKR Compressor | ❌ Insecure (compresses helpful details) | ❌ Insecure (compresses conflicting details) | ✅ Effective (compresses ads) | ✅ Effective (compresses warnings) |

**Generator Safety Analysis** (Cumulative positive metrics):

| Generator | Safety Performance |
|-------|---------|
| Baichuan 13B | Best overall performance across tasks, particularly outstanding on the DoS task |
| Lightweight models (Qwen 7B, etc.) | Surprisingly safer than the GPT series |
| GPT-4/GPT-4o | More sensitive to the proposed novel attacks |

**Evaluator-Human Consistency**:

| Task | F1(correct) Alignment | F1(incorrect) Alignment | ASR/AFR Alignment |
|------|-------------------|---------------------|---------------|
| Silver Noise | 89.97% | 96.22% | — |
| Inter-context Conflict | 99.10% | 98.48% | 95.65% |
| Soft Ad | — | 91.67% | 100% |
| White DoS | 89.97% | 96.22% | 100% |

### Key Findings

1. **All four attacks successfully bypass existing RAG components**: Even the simplest White DoS can deceive retrievers, filters, and generators by masquerading as safety warnings.
2. **Injections closer to the generator pose higher risks**: Injecting attacks during the filtered context stage (after the final line of defense) is significantly more potent than injecting during the knowledge base stage (which could be intercepted by retrievers or filters).
3. **Capable models are not necessarily safer**: GPT-4/GPT-4o are ironically more vulnerable to the newly designed attacks. This is hypothesized to be due to their superior instruction-following capabilities, which make them more easily misled by carefully crafted adversarial text.
4. **SKR compressors present a double-edged sword**: They reduce security in noise and conflict tasks by removing useful details, but enhance safety in soft ad and DoS tasks by compressing the adversarial content.
5. **No universal defense exists**: NLI filters are only effective against noise, while SKR compressors are only effective against soft ads and DoS. This indicates a current lack of a unified safety defense scheme for RAG.

## Highlights & Insights

- **First Chinese RAG Security Benchmark**: Fills the gap in RAG safety evaluations under Chinese scenarios, with extensive coverage across five major news domains.
- **Ingenious and Targeted Attack Designs**: Each attack targets a specific weakness in the RAG defense chain: silver noise bypasses relevance filtering, soft ads bypass toxicity detection, white DoS exploits safety warning disguises, and inter-context conflict exploits the blind spots of the LLM's parametric knowledge.
- **Counter-intuitive Finding of "Stronger $\neq$ Safer"**: Challenges the naive assumption that upgrading LLMs inherently improves RAG safety, pointing out that superior instruction-following capability can actually act as a vector for attacks.
- **Reusable Evaluation Framework**: The evaluation methodology based on multiple-choice questions and attack keywords is elegant and efficient, aligning tightly with human judgment, making it directly applicable to new RAG security assessment scenarios.

## Limitations & Future Work

- **Limited Dataset Scale**: Relying on only 100 base Q&A pairs may lack statistical significance.
- **Chinese Scenario Only**: Generalizability to English and other languages remains unverified.
- **Manual Annotation Dependency for Attack Construction**: Particularly for inter-context conflicts and soft ads, the creation cost is high, limiting scalability.
- **Evaluation without Defense**: Only exposes defects without offering effective defensive schemes or suggestions.
- **Exclusion of Complex Attacks**: Does not cover more complex attack vectors such as adversarial embedding attacks, multi-step collaborative attacks, or backdoor injections.
- **Rigid Templates for White DoS**: It currently relies on static templates for generation, which can be easily targeted for defense.

## Related Work & Insights

- **vs RGB / RAG Bench (Noise Benchmarks)**: The noise in these benchmarks only represents "thematically relevant but answer-free" contexts. In contrast, the "partially correct" silver noise introduced by SafeRAG is webbed with high stealthiness and is much harder to defend against.
- **vs RECALL / ClashEval (Conflict Benchmarks)**: These works study context-memory conflicts (which LLMs can judge using their internal parametric knowledge). SafeRAG's inter-context conflicts focus on domains where the LLM lacks sufficient internal knowledge, carrying higher attack potency.
- **vs Phantom / MAR (DoS Benchmarks)**: Traditional DoS attacks directly injecting refusal signals are easily filtered. SafeRAG's White DoS masquerades as a benign advisory or safety warning, significantly bumping up the success rate.
- **Insights**: RAG security should not focus merely on individual isolated components; rather, it requires end-to-end evaluations across the entire pipeline. Defensive measures must also be tailored to specific attack types.

## Rating

- Novelty: ⭐⭐⭐⭐ Four novel attack designs with high specificity, establishing the first Chinese RAG security benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering 14 components across 3 injection stages with 4 attack tasks.
- Writing Quality: ⭐⭐⭐ Well-structured, though overloaded with numerous charts and appendix content; the primary findings need to be distilled from massive data.
- Value: ⭐⭐⭐⭐ Strongly advances RAG security research and exposes highly practical risks, although the lack of ready-to-use defensive mechanisms marginally drops its immediate utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](../../ECCV2024/information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)

</div>

<!-- RELATED:END -->
