---
title: >-
  [Paper Note] Easier to Judge Than to Find: Predicting In-Context Learning Success for Demonstration Selection
description: >-
  [ICML 2026][Model Compression][In-Context Learning] This paper reformulates ICL demonstration selection from "searching for the optimal $D^\star$ in a massive combinatorial space" to "judging whether a sampled $(q…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "In-Context Learning"
  - "Demonstration Selection"
  - "Success Rate Prediction"
  - "Difficulty Stratification"
  - "Inference Budget"
date: 2026-05-08
content_hash: 1bb47f1b0cf48643
---

# Easier to Judge Than to Find: Predicting In-Context Learning Success for Demonstration Selection

**Conference**: ICML 2026  
**arXiv**: [2605.18512](https://arxiv.org/abs/2605.18512)  
**Code**: To be confirmed  
**Area**: LLM / NLP  
**Keywords**: In-Context Learning, Demonstration Selection, Success Rate Prediction, Difficulty Stratification, Inference Budget  

## TL;DR
This paper reformulates ICL demonstration selection from "searching for the optimal $D^\star$ in a massive combinatorial space" to "judging whether a sampled $(q,D)$ pair will succeed." It proposes DiSP—a framework that stratifies query difficulty and uses lightweight judge models for "sample-judge-stop on accept." DiSP achieves up to a 3.4% improvement over strong baselines across five classification benchmarks while reducing end-to-end real-time latency by up to 23×.

## Background & Motivation

**Background**: The ICL performance of Large Language Models (LLMs) is extremely sensitive to the specific demonstrations chosen and their order in the prompt. The mainstream approach models demonstration selection as an "instance-adaptive search" problem—given a query $q$, a set of demonstrations $D^\star$ that maximize the LLM's likelihood of correctness is selected from a large candidate pool via heuristic retrieval (BM25 / kNN), learned retrievers/rankers (Rubin et al., Uprise, Se2, SeDPO), or proxy scoring based on model feedback.

**Limitations of Prior Work**: The candidate space is subject to combinatorial explosion—selecting and ordering $k$ items from $N$ candidates yields $\binom{N}{k}k!$ combinations, making exhaustive LLM evaluation impossible. Even with proxy scoring to prune candidates, reliable search still requires verifying multiple candidates with the LLM; however, this "LLM verification" is precisely the cost one seeks to avoid, leading to a self-contradiction. Worse, proxy signals are often unreliable: semantically similar demonstrations do not guarantee LLM success, and the performance of a single demonstration varies significantly across different queries.

**Key Challenge**: The search paradigm assumes that "finding the optimal $D^\star$ is the goal," but finding the optimum in a combinatorial space is too expensive. Furthermore, the necessary compute for selecting demonstrations varies greatly across different $q$—simple queries succeed with almost any demonstration, while difficult queries may fail regardless of the compute spent. A uniform search strategy is inherently resource-inefficient.

**Goal**: To transform the expensive search for the "optimal demonstration" into a cheaper binary classification problem of "judging if a sampled demonstration is sufficient," while adaptively allocating computational budgets based on query difficulty.

**Key Insight**: The authors observe that "discrimination is easier than generation"—predicting whether ICL will succeed for a given $(q, D)$ pair is significantly easier than searching for the optimal $D^\star$ from scratch. If a lightweight judge $g(q,D)$ can estimate $P(s(q,D)=1\mid q, D)$ at a minimal cost, a "sample-and-judge" approach can be employed: sample candidates from a random proposal, have the judge score them sequentially, and stop at the first acceptable one. The search thus collapses into a feasibility test.

**Core Idea**: DiSP (Difficulty-Stratified Success Prediction) = Estimate success rates for each training query via random trial runs → Stratify queries into four difficulty levels → Train a router to predict difficulty and three level-specific judges of varying capacities → During inference, allocate sampling budgets by difficulty and stop on acceptance, falling back to a random demonstration with a risk label if no acceptable candidate is found.

## Method

### Overall Architecture
The DiSP pipeline consists of three stages. **Stage 1 (Supervision Construction)**: For each training query $q$, $T$ $k$-shot contexts $D_1,\dots,D_T$ are sampled from a random distribution $\mathcal{P}_{\text{rand}}$. The target LLM is run to obtain success indicators $s(q, D_t) = \mathbb{I}[\hat{y}(q,D_t) = y^\star]$, and the empirical success rate is estimated as $\hat{\rho}(q) = \frac{1}{T}\sum_t s(q, D_t)$. Queries are stratified into $\{l_1, l_2, l_3, l_x\}$ based on thresholds $\alpha > \beta > \gamma$. **Stage 2 (Training)**: A router $r(q)$ (four-way classification) and three judges $g_{l_1}, g_{l_2}, g_{l_3}$ (binary classification) are trained on the partitioned datasets. No judge is trained for $l_x$. **Stage 3 (Inference)**: New queries are first routed to a level $\hat{\ell}$ by the router. If $\hat{\ell} = l_x$, a random demonstration is sampled and labeled as `HARD_QUERY`. Otherwise, candidates are sequentially evaluated by $g_{\hat{\ell}}$ until one is accepted or the budget is exhausted, in which case a fallback is used with a `NO_GOOD_DEMO` label.

### Key Designs

1. **Difficulty Stratification based on Random Trials**:
    - **Function**: Categorizes all training queries into four levels based on their likelihood of being solved by random demonstrations, allowing the router and judges to allocate compute accordingly.
    - **Mechanism**: Using a **class-balanced** random proposal distribution ($k = |\mathcal{Y}|$, one random sample per class, shuffled), $T$ trials are conducted for each $q$ to obtain $\hat{\rho}(q)$. Queries are categorized into $l_1$ (Easy), $l_2$ (Medium), $l_3$ (Fragile), and $l_x$ (Nearly Hopeless) based on whether $\hat{\rho}(q)$ falls into specific intervals defined by $\alpha, \beta, \gamma$. Specifically, the smaller the $\hat{\rho}$, the larger the $K$ required to find a viable demonstration.
    - **Design Motivation**: Previous methods treated all queries equally, wasting compute on "easy" queries that succeed regardless of the demonstration, and fruitlessly searching for "impossible" queries. Separating these extremes allows the system to concentrate computational resources on $l_2 / l_3$ queries that actually benefit from careful selection.

2. **Difficulty-Adaptive Router + Judge System**:
    - **Function**: Predicts the difficulty level $\hat{\ell}$ (router) of a query $q$ and the success probability $g_\ell(q, D) \approx P(s(q,D)=1\mid q, D)$ (judge) for a $(q, D)$ pair.
    - **Mechanism**: The router $r(q)$ is a BERT-base model. To match difficulty, judge capacities increase: $g_{l_1}$ uses BERT-base, $g_{l_2}$ uses RoBERTa-Large, and $g_{l_3}$ employs a lightweight classification head on the target LLM's hidden representations. Only the most fragile level ($l_3$) utilizes LLM features; others are independent of the target LLM. The target LLM remains frozen throughout.
    - **Design Motivation**: The tiered capacity follows the principle of "using expensive features only when necessary." Representation probing experiments showed that simple MLP probes on LLM hidden states can achieve high AUROC/AUPRC for success/failure pairs, proving that "judging feasibility" is simpler than "learning a task-specific retriever." This embodies the "judge is easier than find" principle by reducing search to binary classification.

3. **Stop-on-Accept Feasibility Test**:
    - **Function**: Controls the budget and quality of demonstration selection for each query, providing a precision-cost knob and diagnostic labels.
    - **Mechanism**: For queries routed to $\hat{\ell} \in \{l_1, l_2, l_3\}$, contexts $D_i \sim \mathcal{P}_{\text{rand}}$ are sampled and evaluated by $g_{\hat{\ell}}(q, D_i)$. If $g_{\hat{\ell}}(q, D_i) \geq \tau_{\hat{\ell}}$, it is immediately accepted. Otherwise, sampling continues up to a budget $K_{\hat{\ell}}$ ($K_{l_1} < K_{l_2} < K_{l_3}$). If the budget is exhausted, the system falls back to a random sample with a `NO_GOOD_DEMO` label.
    - **Design Motivation**: Switching from "ranking" to "feasibility testing" is crucial; ranking requires evaluating all candidates, while stop-on-accept terminates at the first sufficient one. $K_\ell$ serves as a tunable knob for balancing compute and accuracy. Risk labels provide an explicit "abstention" signal, facilitating downstream safety or fallback mechanisms.

### Loss & Training
The router is trained using four-way cross-entropy on $\mathcal{D}_{\text{route}} = \{(q, \ell(q))\}$. Each judge is trained separately using binary cross-entropy on $\mathcal{D}_{l_j} = \{((q, D), s(q, D)) : \ell(q) = l_j\}$. The target LLM is frozen. Thresholds $\tau_\ell$ and budgets $K_\ell$ are inference-time hyperparameters selected via cost-accuracy curves on a validation set.

## Key Experimental Results

### Main Results
Performance was compared across 5 classification benchmarks (TREC, SST-2, SST-5, AGNews, MNLI) using LLaMA3-8B and Qwen2.5-7B. Results are means of 6 independent runs.

| Method (LLaMA3-8B) | TREC | SST-2 | SST-5 | AGNews | MNLI | Average |
|------------------|------|-------|-------|--------|------|------|
| Zero-shot | 70.5 | 94.7 | 41.7 | 72.9 | 52.3 | 66.4 |
| Random (1-shot) | 73.8 | 94.9 | 46.6 | 84.1 | 66.9 | 73.3 |
| BM25 | 75.2 | 94.7 | 52.3 | 86.1 | 65.6 | 74.8 |
| Uprise | 75.8 | 94.0 | 48.3 | 90.7 | 58.9 | 73.6 |
| Se2 | 68.2 | 93.4 | 51.7 | 86.1 | 60.9 | 72.1 |
| SeDPO | 67.5 | 93.4 | 47.0 | 88.1 | 61.6 | 71.5 |
| **DiSP (Ours)** | **79.2** | **95.4** | **54.3** | 89.7 | **69.5** | **77.6** |

On Qwen2.5-7B, DiSP averaged 82.9%, outperforming the strongest baseline (BM25 80.5%) by 2.4 points and improving upon zero-shot by 13.1 points. Gains were most significant on difficult datasets like TREC and MNLI.

### Ablation Study
End-to-end wall-clock cost (average across 5 datasets):

| Phase | Uprise | Se2 | SeDPO | DiSP |
|------|--------|-----|-------|------|
| Training (min) | 13.4 | 114.9 | 157.1 | **6.7** |
| Testing (min) | 0.4 | 0.6 | 0.6 | **0.1** |
| Total Cost (min) | 13.8 | 115.4 | 157.6 | **6.8** |
| Relative to DiSP | 2.0× | 17.0× | 23.2× | 1.0× |

### Key Findings
- DiSP is simultaneously the most accurate and the most efficient method, breaking the traditional accuracy-efficiency trade-off in demonstration selection.
- The 23× training speedup primarily results from skipping task-specific retriever training; the unified judge and random proposal are reusable across tasks. The 6× inference speedup comes from the stop-on-accept mechanism and lightweight judging.
- Representation probing confirms that success/failure $(q, D)$ pairs are separable in the LLM's hidden space, justifying the core hypothesis that ICL failure modes have structural signals.
- The $l_x$ (極難) category enables natural "abstention" semantics. `HARD_QUERY` and `NO_GOOD_DEMO` labels provide clear hooks for downstream handling (manual review, conservative fallback).

## Highlights & Insights
- "Judging is easier than finding" is a universal insight. In fields like RL or code generation, finding the optimal action is hard, but verifying if an action is "good enough" is cheap. DiSP applies this to ICL.
- The stratified compute allocation paradigm is closer to real-world deployment needs, where a long tail of queries requires varying effort.
- Replacing learned retrievers with random proposals simplifies the architecture—data dependency is localized in the judge, while the proposal remains data-agnostic.
- The combination of stop-on-accept, explicit budgets, and risk labels is highly engineering-friendly, offering transparency and control.

## Limitations & Future Work
- Experiments were limited to **classification** tasks. Applying success indicators $s(q, D) = \mathbb{I}[\hat{y} = y^\star]$ to open-ended generation (QA, summarization) requires new designs as hard labels are often unavailable.
- Estimating $\hat{\rho}(q)$ requires $T$ LLM runs per training query, making the initial offline supervision construction potentially expensive.
- Strata thresholds $(\alpha, \beta, \gamma)$ and judge thresholds $\tau_\ell$ are manual hyperparameters that may require re-tuning for different models/tasks.
- The framework currently lacks an adaptive mechanism for the proposal distribution if the test distribution shifts significantly (OOD).

## Related Work & Insights
- **vs Rubin et al. (EPR), Uprise, SeDPO**: These train task-specific retrievers. DiSP flips the approach by training judges and using random proposals with feasibility testing, significantly reducing search costs.
- **vs Batch-ICL**: Batch-ICL addresses order sensitivity; DiSP addresses selection feasibility. They are orthogonal and can be combined.
- **vs Model Cascade / Routing**: While standard routing chooses between LLMs, DiSP chooses between **auxiliary judges** for a single LLM, targeting the prompt level rather than the model level.
- **vs Conformal Prediction**: DiSP's risk labels complement uncertainty quantification by providing actionable abstention signals.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of "discrimination over generation" to ICL selection via stratification and stop-on-accept is a systematic first.
- Experimental Thoroughness: ⭐⭐⭐ Strong results on classification, but lacks verification on generative tasks and hasn't been compared directly with prompt optimization frameworks like DSPy.
- Writing Quality: ⭐⭐⭐⭐ Clear three-stage framework with formal guarantees provided in the appendix.
- Value: ⭐⭐⭐⭐ The 23× speedup and accuracy gains are highly attractive for production environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](../../ICLR2026/model_compression/llms_encode_their_failures_predicting_success_from_pre-generation_activations.md)
- [\[ICML 2026\] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images](images_as_tables_in-context_learning_with_tabpfn_for_low-data_detection_of_ai-ge.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[AAAI 2026\] Predicting the Future by Retrieving the Past](../../AAAI2026/model_compression/predicting_the_future_by_retrieving_the_past.md)
- [\[ICML 2026\] T3S: Training Trajectory-Aware Token Selection to Break "Imitation Shock" in Reasoning Distillation](training-trajectory-aware_token_selection.md)

</div>

<!-- RELATED:END -->
