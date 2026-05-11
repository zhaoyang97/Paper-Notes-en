---
title: >-
  [Paper Note] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge
description: >-
  [ACL 2026][Information Retrieval & RAG][continual unlearning] CURaTE proposes a behavioral unlearning framework based on sentence embedding matching: a general-purpose unlearning embedder is trained prior to deployment (…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "continual unlearning"
  - "real-time unlearning"
  - "behavioral unlearning"
  - "sentence embeddings"
  - "knowledge preservation"
date: 2026-05-08
content_hash: 5c26792c815c3653
---

# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

**Conference**: ACL 2026
**arXiv**: [2604.14644](https://arxiv.org/abs/2604.14644)
**Code**: [GitHub](https://github.com/bsu1313/CURaTE)
**Area**: Information Retrieval
**Keywords**: continual unlearning, real-time unlearning, behavioral unlearning, sentence embeddings, knowledge preservation

## TL;DR
CURaTE proposes a behavioral unlearning framework based on sentence embedding matching: a general-purpose unlearning embedder is trained prior to deployment (without any forget set); after deployment, embeddings of incoming unlearning requests are stored in a database; at inference time, cosine similarity determines whether to answer or refuse a query. LLM weights are never modified, yielding near-perfect knowledge preservation.

## Background & Motivation

**Background**: LLM unlearning methods primarily include parameter-modifying approaches such as Gradient Ascent (GA), Gradient Difference (GradDiff), and Preference Optimization (PO/NPO), as well as continual unlearning methods such as GUARD, O3, and UniErase.

**Limitations of Prior Work**: All weight-modifying methods suffer from catastrophic forgetting—as unlearning requests accumulate, model performance on the retain set degrades sharply. Furthermore, existing methods require a training or optimization process to handle each unlearning request, causing sensitive information to remain accessible throughout that period.

**Key Challenge**: Unlearning demands "behavioral change" while weight modification inevitably causes "knowledge loss"—these two objectives are fundamentally in conflict within the parameter space.

**Goal**: Achieve real-time, continual unlearning without modifying LLM weights, supporting an arbitrary number of sequential unlearning requests without degrading model utility.

**Key Insight**: Redefine the unlearning objective—relaxing "parametric unlearning" (erasing knowledge) to "behavioral unlearning" (preventing the output of flagged information)—thereby opening up a solution space that does not require weight modification.

**Core Idea**: Train a task-agnostic sentence embedder for semantic similarity judgment—if a query is sufficiently similar to an unlearning request, the system refuses to answer; otherwise, normal generation proceeds.

## Method

### Overall Architecture
CURaTE operates in two phases: (1) **Pre-deployment training**: paraphrase positives and contrastive negatives are generated from a seed QA dataset, and a sentence embedder $U$ is fine-tuned with a contrastive loss. (2) **Post-deployment inference**: upon receiving an unlearning request, its embedding is immediately computed and stored in database $F$; at query time, the maximum cosine similarity between the user query and all embeddings in $F$ is computed, and if it exceeds threshold $\delta$, the system refuses to respond.

### Key Designs

1. **Task-Agnostic Unlearning Embedder Training**

    - **Function**: Learns a general-purpose semantic similarity judgment capability; no retraining is required after deployment.
    - **Mechanism**: Three types of training data are generated from a seed QA dataset (e.g., Natural Questions): Type-1 (original question + paraphrase, positive pair), Type-2 (original question + contrastive question, hard negative—lexically similar but semantically different), and Type-3 (paraphrase + its contrastive question, hard negative). The embedder is trained with contrastive loss $\mathcal{L} = y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2$.
    - **Design Motivation**: Hard negative pairs ensure the embedder can distinguish between "asking the same thing in different words" and "appearing similar but asking about something different"—a core requirement in unlearning scenarios, where paraphrase variants must be intercepted without incorrectly blocking unrelated queries.

2. **Embedding Database for Real-Time Unlearning**

    - **Function**: Enables immediate effect of unlearning requests without any optimization process.
    - **Mechanism**: Upon receiving unlearning request $f_m$, only its embedding $f_m^{emb} = U(f_m)$ needs to be computed and appended to set $F$—an $O(1)$ operation. At query time, $s_{max} = \max_{i} \text{cos}(p^{emb}, f_i^{emb})$ is computed; if $s_{max} \geq \delta$, a response is sampled from a predefined set of refusal expressions $R$.
    - **Design Motivation**: Parametric unlearning requires gradient computation, taking minutes to hours, during which sensitive information remains accessible. Embedding storage achieves genuinely instantaneous unlearning.

3. **Knowledge Preservation via Frozen LLM Weights**

    - **Function**: Maintains perfect knowledge preservation regardless of the number of unlearning requests.
    - **Mechanism**: Since LLM parameters are never modified, all knowledge unrelated to unlearning requests is fully preserved—catastrophic forgetting is structurally impossible. The only risk is false rejection (misclassifying unrelated queries as unlearning targets), which is minimized through hard-negative training.
    - **Design Motivation**: Catastrophic forgetting is the fundamental bottleneck of parametric unlearning; completely bypassing weight modification is the most principled solution.

### Loss & Training
Contrastive loss: $\mathcal{L} = \frac{1}{2|T|}\sum [y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2]$, using cosine distance as the metric. Training is performed once on the seed dataset; no additional training is required after deployment.

## Key Experimental Results

### Main Results

| Method | Unlearning Efficacy (after 10 stages) | Knowledge Preservation (after 10 stages) | Real-Time Capability |
|--------|--------------------------------------|------------------------------------------|----------------------|
| GA | Effective but over-forgetting | Severe degradation (~0) | No |
| GradDiff | Over-forgetting | Severe degradation | No |
| NPO | Moderate | Moderate degradation | No |
| O3 | Insufficient unlearning | Partial preservation | No |
| UniErase | Insufficient unlearning | Partial preservation | No |
| CURaTE | Effective unlearning | Near-perfect preservation | Yes |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Without hard-negative training | High false rejection rate | Hard negatives are critical for decision boundary precision |
| Fixed threshold $\delta$ | Stable performance | Threshold exhibits some sensitivity across tasks |
| Evaluated with paraphrase variants | CURaTE remains effective | Embedder is robust to paraphrasing |

### Key Findings
- CURaTE is the only method that maintains near-perfect knowledge preservation after 10 stages of continual unlearning.
- Parametric unlearning methods (GA, GradDiff) exhibit severe utility collapse after only 3–5 stages.
- The embedder, trained on a single seed dataset, transfers across domains to entirely different unlearning tasks.
- The method is robust to paraphrase attacks, owing to the positive-pair design during training.

## Highlights & Insights
- The **redefinition of "behavioral unlearning"** is the key conceptual contribution—relaxing the objective from "erasing knowledge" to "blocking output" fundamentally reshapes the solution space.
- A remarkably simple approach (embedding similarity + threshold decision) achieves the best results, revealing the unnecessary complexity of parametric unlearning methods.
- The framework generalizes to any scenario requiring "selective refusal," including copyright protection, privacy preservation, and information filtering.

## Limitations & Future Work
- Behavioral unlearning is not true knowledge erasure—the knowledge remains in LLM weights and may be elicited through indirect queries.
- The choice of threshold $\delta$ is a performance bottleneck: too permissive leads to incomplete unlearning, while too restrictive increases false rejections.
- The forget database $F$ grows with accumulated requests; large-scale deployment would require approximate nearest-neighbor search.
- The approach is unsuitable for regulatory requirements mandating genuine knowledge erasure (e.g., the right to be forgotten under GDPR).

## Related Work & Insights
- **vs. GUARD**: GUARD also trains a classifier but requires retraining for each forget set; CURaTE trains once and generalizes across domains.
- **vs. O3**: O3 trains orthogonal LoRA adapters with an OOD detector, still modifying parameters; CURaTE leaves weights entirely untouched.
- **vs. UniErase**: UniErase injects unlearning tokens via model editing—essentially still a parametric modification—making catastrophic forgetting unavoidable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The "behavioral unlearning" concept and minimalist design are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, 10-stage continual unlearning, and comparison against multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and straightforward method description.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](../../AAAI2026/information_retrieval/sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)

</div>

<!-- RELATED:END -->
