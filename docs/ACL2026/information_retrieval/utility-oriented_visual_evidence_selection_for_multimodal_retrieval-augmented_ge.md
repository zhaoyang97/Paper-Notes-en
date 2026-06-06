---
title: >-
  [Paper Note] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation
description: >-
  [ACL2026][Information Retrieval & RAG][Multimodal RAG] This paper shifts image selection in Multimodal RAG from "semantic similarity ranking" to "utility estimation of helpfulness for the final answer." By using a lightw…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Multimodal RAG"
  - "Visual Evidence Selection"
  - "Information Gain"
  - "Lightweight Surrogate Model"
  - "Retrieval Reranking"
date: 2026-05-08
content_hash: 2184ceb3a98d4eb0
---

# Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation

**Conference**: ACL2026  
**arXiv**: [2605.13277](https://arxiv.org/abs/2605.13277)  
**Code**: https://github.com/Hcnaeg/utility-mrag  
**Area**: information_retrieval  
**Keywords**: Multimodal RAG, Visual Evidence Selection, Information Gain, Lightweight Surrogate Model, Retrieval Reranking  

## TL;DR
This paper shifts image selection in Multimodal RAG from "semantic similarity ranking" to "utility estimation of helpfulness for the final answer." By using a lightweight multimodal surrogate model to efficiently predict evidence helpfulness, it simultaneously improves answer quality and inference efficiency on MRAG-Bench and Visual-RAG.

## Background & Motivation
**Background**: Multimodal RAG typically retrieves a set of candidate images and provides the Top-K images to a Multimodal Large Language Model (MLLM) to generate an answer. Existing visual evidence selection methods largely follow the logic of text RAG, using CLIP, SigLIP, BGE-VL, or MLLM rerankers to estimate the semantic relevance of the query-image pair.

**Limitations of Prior Work**: Relevant images are not necessarily useful. The paper uses a dog breed identification example to show that an image may score high due to salient attributes like "tongue out" matching the query image, but if it belongs to the wrong breed, it provides no help for the discriminative information actually needed. Similarity retrieval emphasizes "look-alike," whereas RAG generation requires "the ability to change the model's judgment of the answer."

**Key Challenge**: Visual evidence selection faces two levels of misalignment. First, semantic relevance is inconsistent with downstream utility; relevant images might be redundant, misleading, or lack discriminative features. Second, estimating utility directly in the answer space is difficult because the output distribution of MLLMs is implicit, open-ended answers contain linguistic noise, and the cost of repeated generation or sampling is high.

**Goal**: The authors aim to establish a more principled visual evidence selection criterion that directly measures the helpfulness of candidate images to the model's answer. Furthermore, this criterion must not rely on the expensive main model generating answers for every candidate, ensuring scalability for real-world candidate pools.

**Key Insight**: Grounded in information theory, the paper defines evidence utility as "how much the model's uncertainty or belief regarding the answer distribution changes given the candidate evidence." Detecting that direct calculation of information gain in the answer space is infeasible, the authors introduce a binary latent variable: whether this evidence is helpful.

**Core Idea**: Approximate answer-space utility using the latent helpfulness probability $P(Z=1|C=c,q)$, and employ a lightweight multimodal surrogate model to handle candidate ranking, while the main model only receives the final selected evidence.

## Method
The key to this paper is the redefinition of "evidence selection." Traditional retrievers typically answer "Is this image relevant to the question?", while this paper asks "Will this image make the target model more likely to answer correctly?" These two questions are not equivalent in many multimodal scenarios, especially in fine-grained recognition, visual commonsense, and open-ended QA.

The authors first provide an ideal goal: if candidate evidence $c$ is truly useful, the distribution of the output $Y$ should undergo a meaningful change after the model sees it. This change can be measured by Information Gain: $IG(Y;C=c|q)=D_{KL}(P_{Y|C=c,q}||P_{Y|q})$. However, this is only suitable for theoretical analysis because real MLLMs lack explicit output distributions, open-ended answer spaces are vast, and different phrasings introduce noise into KL or uncertainty estimation.

To solve this, the paper projects answer-space utility into a smaller latent variable space. The latent variable $Z$ is a Bernoulli variable indicating whether a candidate evidence is helpful. The authors prove that under mild assumptions (e.g., candidate evidence is not significantly harmful and the answer distribution changes monotonically with helpfulness), sorting by $IG(Z;C=c)$ preserves the optimal solution for sorting by $IG(Y;C=c)$. Furthermore, within a feasible candidate set, sorting by $IG(Z;C=c)$ is equivalent to sorting by $P(Z=1|C=c,q)$. Thus, the difficult comparison of answer distributions is transformed into a binary helpfulness scoring problem.

### Overall Architecture
The complete pipeline is retrieve-select-generate. First, the system constructs a fixed candidate pool using an existing retriever, containing ground-truth images and extra retrieved distractors. Second, a lightweight surrogate MLLM performs helpfulness judgments for each query-image pair, calculating and ranking scores. Finally, the costly main model generates the final answer using only the Top-K evidence.

In implementation, an auxiliary question is constructed, such as "Is this evidence helpful for answering the user's question?", restricting the output space to True / False. For candidate image $c_i$, the model input is a template $I=Template(q,c_i,q_{aux})$. The helpfulness score is simply the logit of the final layer for the True token: $s(c_i)=\ell(v^+|I)$. This design avoids generating long answers and requires no extra training.

### Key Designs
1. **From Relevance to Information Gain Utility**:
    - **Function**: Establishes a generation-oriented goal for visual evidence selection rather than following retrieval similarity.
    - **Mechanism**: High-utility evidence should significantly change the model's posterior belief about the answer, expressed as $IG(Y;C=c|q)$. It focuses on whether the evidence provides discriminative information rather than surface-level semantic proximity.
    - **Design Motivation**: Failures in multimodal RAG often stem from images that "look relevant but are useless for answering." Changing the target to information gain naturally explains why low-similarity images containing critical clues are worth selecting.

2. **Latent Helpfulness Variable**:
    - **Function**: Converts uncomputable answer-space utility into an estimatable binary judgment.
    - **Mechanism**: Define $Z=1$ to indicate the image helps the current query. Theoretical derivation shows that under reasonable assumptions, ranking by $P(Z=1|C=c,q)$ preserves the optimality of answer-space utility ranking. The system only needs the model to answer if the evidence is helpful and then ranks using the True logit.
    - **Design Motivation**: Answer-level uncertainty is prone to phrasing variance, sampling noise, and hallucinations; helpfulness judgment is a discriminative task with short inputs/outputs, suitable for batch execution by small models.

3. **Surrogate-Accelerated Execution**:
    - **Function**: Shifts the $O(N)$ scoring cost from the large main model to a small surrogate model, reducing FLOPs and latency.
    - **Mechanism**: The authors propose the **utility transferability hypothesis**: if an image is obviously unhelpful or contradictory to a small model, it is likely unhelpful to a large model as well. Lightweight models like Qwen3-VL-2B or Ovis2.5-2B are used for ranking, and the main model performs only one final generation.
    - **Design Motivation**: Real RAG candidate pools can be large; having 8B to 12B main models judge or generate answers for each is too costly. Surrogate ranking maintains utility orientation while making the system deployable.

### Loss & Training
Ours is a training-free method with no supervised training or gradient updates. The "training strategy" is actually an inference-time scoring protocol: use a fixed candidate pool, an auxiliary helpfulness prompt, compare True/False logits, and pass Top-K images to the main model. Experiments cover various main models (Qwen3-VL, MiniCPM-V4.5, Gemma3, Ovis2.5, InternVL3.5) and compare against CLIP-style retrievers, MLLM retrievers, answer-level uncertainty, verbalized UQ, and listwise ranking.

## Key Experimental Results

### Main Results
The main experiments evaluate Top-K evidence selection on MRAG-Bench and Visual-RAG. MRAG-Bench uses exact-match accuracy, while Visual-RAG uses LLM-as-Judge. Below are representative methods for K=1, showing that Ours outperforms strong retrieval/reranking baselines across most models and datasets.

| Method | Params | MRAG Qwen3-VL-8B | MRAG MiniCPM-V4.5 | Visual-RAG Qwen3-VL-8B | Visual-RAG Ovis2.5-9B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-shot | No Image | 59.35 | 57.95 | 52.41 | 52.67 |
| GME | 2.2B | 64.38 | 65.19 | 55.88 | 67.51 |
| LamRA-Rank | 8B | 63.34 | 62.97 | 58.42 | 58.16 |
| Ours, Qwen3-VL-2B surrogate | 2.1B | 65.56 | 65.41 | 59.89 | 68.85 |
| Ours, Ovis2.5-2B surrogate | 2.6B | 64.97 | 64.08 | 61.36 | 69.12 |

In the full results, Ours provides up to +16.18 absolute improvement on Visual-RAG and often approaches the GT oracle, sometimes even slightly exceeding human-annotated image inputs. This suggests that "human-relevant images" are not necessarily the most useful evidence for a specific MLLM.

### Ablation Study
The paper emphasizes the comparison between the latent helpfulness target and the answer-level uncertainty target. The table below shows that the answer-level method lags behind Ours.

| Dataset / Main Model | Top-K | Ours | Answer-level Estimation | Gain |
| :--- | :--- | :--- | :--- | :--- |
| MRAG-Bench / Qwen3-VL-8B | Top-1 | 65.71 | 63.27 | -2.44 |
| MRAG-Bench / MiniCPM-V4.5 | Top-3 | 66.89 | 64.15 | -2.74 |
| Visual-RAG / Qwen3-VL-8B | Top-1 | 62.43 | 57.49 | -4.94 |
| Visual-RAG / Ovis2.5-9B | Top-1 | 70.05 | 54.81 | -15.24 |
| Visual-RAG / InternVL3.5-8B | Top-3 | 58.16 | 54.68 | -3.48 |
| Visual-RAG / Gemma3-12B | Top-5 | 60.03 | 57.09 | -2.94 |

Computational cost is also critical. Using the Qwen3-VL family as an example, discriminative estimation on a 2.1B surrogate has a decode latency of ~3 ms, compared to ~101 ms for answer-level UQ. On the 8.1B main model, these are ~15 ms and ~443 ms respectively. The $Z$ target is over 20x more efficient in decode FLOPs than the $Y$ target.

| Model Family | Method | Surrogate Decode Latency | Main Model Decode Latency | Implications |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL | Discriminative Estimation | 3 ms | 15 ms | Short output (helpfulness only) |
| Qwen3-VL | Answer-level UQ | 101 ms | 443 ms | High cost, requires answer generation |
| Ovis2.5 | Discriminative Estimation | 3 ms | 15 ms | Ideal for batch ranking |
| Ovis2.5 | Answer-level UQ | 101 ms | 443 ms | Cost scales with sampling/candidates |

### Key Findings
- Semantic relevance is an unstable RAG goal. Zero-shot sometimes outperforms retrieval-augmented baselines, indicating that wrong evidence provides negative gain.
- Latent helpfulness is superior to answer-level uncertainty for evidence ranking. The gap is larger on open-ended Visual-RAG because answer-level methods are disrupted by generation noise.
- Lightweight surrogates align highly with large main models. The average gap on MRAG-Bench is ~0.38 and on Visual-RAG is ~0.18. Serious false positives are only ~2% to 3.5%.
- Surrogate ranking transfers across scales. Qwen2.5-VL 3B/7B as surrogates for 72B maintain GT hit rate trends.
- The method requires no training. It leverages the existing MLLM’s discriminative ability for helpfulness, making implementation cheaper than training specialized retrievers.

## Highlights & Insights
- This paper accurately identifies the core issue of Multimodal RAG: retrieval is not about finding "similar" images but images that change the answer. This explains why "high-similarity, low-utility" images degrade performance.
- Latent helpfulness is an elegant dimension reduction. It retains the theoretical rigor of information gain while simplifying calculation to cheap True/False logit scoring.
- Surrogate acceleration is more than an engineering trick; it matches the task nature. Judging if an image is helpful is often easier than answering the question, allowing small models to handle the filtering.
- This framework is easily transferable. Similar helpfulness probes can be defined for text, video, or tool results for utility-based ranking.
- Approaching or exceeding the GT oracle is enlightening. Human-labeled "relevant images" may not suit a specific model's reasoning style; model-centric utility selection might better adapt to the generator.

## Limitations & Future Work
- Theoretical assumptions are idealized. Assuming candidate evidence is "not harmful" and helpfulness correlates monotonically with answer distribution may not hold in noisy or adversarial candidate pools.
- Surrogate selection is empirical. While 2B models work well, different tasks or candidate pools might require different surrogates.
- Experiments focus on QA-style Multimodal RAG. Long-form generation, dialogue, captioning, and other modalities (audio/video) are not yet fully validated.
- Helpfulness prompts may still be sensitive to phrasing. While robustness tests were done, domain-specific terminology or user intent might blur binary judgments.
- The method selects evidence but does not fix the generator. Even with useful images, the main model might still hallucinate or ignore evidence; it should be integrated with citation verification or abstention mechanisms.

## Related Work & Insights
- **vs CLIP / SigLIP Retrieval**: These focus on similarity in joint embedding spaces; they are fast but do not understand downstream generation goals. Ours directly evaluates helpfulness.
- **vs MLLM Reranker**: Methods like LamRA-Rank or GME use stronger models for relevance estimation; Ours specifically designs the prompt for helpfulness, making the goal clearer.
- **vs Answer-level Uncertainty**: Uncertainty methods estimate reliability via token probability or consistency but introduce generation noise; Ours decouples utility from generation.
- **vs Adaptive RAG / Retrieve-or-not**: These decide *whether* to retrieve; Ours decides *which* retrieved evidence should enter the context.
- **Insights**: Helpfulness scores can be combined with similarity in a two-stage system (recall via similarity, rerank via utility). One could also train a utility distillation model using surrogate scores as weak supervision.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The transformation from information gain to latent helpfulness is elegant; problem definition is more valuable here than architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two benchmarks, multiple model scales, various baselines, efficiency analysis, and error analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Theory and system motivation are clear, though the large tables require effort to digest.
- Value: ⭐⭐⭐⭐⭐ Highly practical for real-world Multimodal RAG systems needing a balance between cost and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A](mm-bizrag_rethinking_multimodal_retrieval-augmented_generation_for_general_purpo.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](../../NeurIPS2025/information_retrieval/reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/information_retrieval/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
