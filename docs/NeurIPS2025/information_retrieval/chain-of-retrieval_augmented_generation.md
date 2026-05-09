---
title: >-
  [Paper Note] Chain-of-Retrieval Augmented Generation (CoRAG)
description: >-
  [NeurIPS 2025][RAG] This paper proposes CoRAG, a framework that automatically generates intermediate retrieval chains (sub-query → sub-answer) via rejection sampling, fine-tunes an LLM to learn iterative retrieval and reasoning, and supports diverse test-time decoding strategies (greedy / Best-of-N / tree search) for flexible compute scaling. CoRAG achieves 26+ EM improvement on multi-hop QA and attains state-of-the-art on 9/10 tasks of the KILT benchmark.
tags:
  - NeurIPS 2025
  - RAG
  - Chain-of-Retrieval
  - multi-hop reasoning
  - rejection sampling
  - retrieval chain
  - test-time compute scaling
date: 2026-05-08
content_hash: 3bdf9fa1327e9bc9
---

# Chain-of-Retrieval Augmented Generation (CoRAG)

**Conference**: NeurIPS 2025
**arXiv**: [2501.14342](https://arxiv.org/abs/2501.14342)
**Code**: [microsoft/LMOps/corag](https://github.com/microsoft/LMOps/tree/main/corag)
**Area**: Information Retrieval
**Keywords**: RAG, Chain-of-Retrieval, multi-hop reasoning, rejection sampling, retrieval chain, test-time compute scaling

## TL;DR

This paper proposes CoRAG, a framework that automatically generates intermediate retrieval chains (sub-query → sub-answer) via rejection sampling, fine-tunes an LLM to learn iterative retrieval and reasoning, and supports diverse test-time decoding strategies (greedy / Best-of-N / tree search) for flexible compute scaling. CoRAG achieves 26+ EM improvement on multi-hop QA and attains state-of-the-art on 9/10 tasks of the KILT benchmark.

## Background & Motivation

**Background**: Conventional RAG pipelines follow a single-step retrieve-then-generate paradigm. Bi-encoder retrievers sacrifice expressiveness for efficiency, making retrieval quality a bottleneck for complex multi-hop queries.

**Limitations of Prior Work**: (a) Single-pass retrieval often fails to gather all information required to answer complex questions; (b) in multi-hop reasoning, what to retrieve depends on the current inference state and cannot be determined in advance; (c) existing multi-step retrieval methods (FLARE, IRCoT, Self-RAG) rely primarily on few-shot prompting or closed-source model distillation and lack end-to-end training.

**Key Challenge**: How can an open-source LLM automatically learn an iterative retrieval strategy from QA datasets that contain only final answers, while enabling flexible compute–performance trade-offs at inference time?

**Key Insight**: Drawing on the success of Chain-of-Thought reasoning, the retrieval process is unrolled into a retrieval chain composed of sub-query–sub-answer pairs, where each step dynamically determines the next query based on the current reasoning state.

**Core Idea**: Rejection sampling is used to automatically generate retrieval chains as training signals from QA datasets. The LLM is fine-tuned to learn iterative retrieval, and Best-of-N sampling together with tree search enables flexible test-time compute scaling.

## Method

### Overall Architecture

The framework consists of three stages: **(1)** rejection sampling to generate retrieval chain training data → **(2)** multi-task unified fine-tuning (sub-query prediction + sub-answer prediction + final answer prediction) → **(3)** diverse test-time decoding strategies. The model takes the current state (original query + existing sub-query/sub-answer chain) as input and predicts the next action (a new sub-query or the final answer).

### Key Designs

1. **Rejection Sampling for Retrieval Chain Construction**

    - **Function**: Automatically constructs retrieval chains with intermediate reasoning steps from QA datasets containing only $(Q, A)$ pairs.
    - **Mechanism**: For each sample, the LLM samples a sub-query $Q_i$ conditioned on the current state $(Q, Q_{<i}, A_{<i})$; E5-large then retrieves the top-5 documents, from which a sub-answer $A_i$ is generated. Up to 16 candidate chains are sampled per instance and scored by the conditional log-likelihood $\log P(A|Q, Q_{1:L}, A_{1:L})$ of the gold answer; the highest-scoring chain is selected.
    - **Design Motivation**: This approach eliminates dependence on human annotation and closed-source models, leveraging the LLM's own capabilities to bootstrapically produce high-quality training data.

2. **Multi-Task Unified Training**

    - **Function**: Jointly optimizes three related prediction tasks within a unified framework.
    - **Three Loss Functions**: $L_\text{sub\_query} = -\log P(Q_i | Q, Q_{<i}, A_{<i})$ (learning what to retrieve next); $L_\text{sub\_answer} = -\log P(A_i | Q_i, D_{1:k}^{(i)})$ (learning what to extract from retrieved documents); $L_\text{final\_answer} = -\log P(A | Q, Q_{1:L}, A_{1:L}, D_{1:k})$ (learning how to synthesize the final answer).
    - **Design Motivation**: The three tasks mutually reinforce each other — learning to generate better sub-queries facilitates final reasoning, while supervision on final answer prediction in turn improves sub-query generation.

3. **Test-Time Compute Scaling**

    - **Three Decoding Strategies**: (a) **Greedy decoding** — generates sub-queries and sub-answers sequentially with a fixed chain length $L$; (b) **Best-of-N sampling** — samples $N$ chains at temperature 0.7 and selects the chain with the lowest penalty score, defined as the log-likelihood of "No relevant information found"; (c) **Tree search** — expands states via BFS and evaluates each state using rollout-averaged penalties.
    - **Scaling Law**: The Pareto frontier between compute (total token consumption) and performance approximately follows a log-linear relationship $y = a \log(x+b) + c$.

### Training Details

- Base model: Llama-3.1-8B-Instruct, full-parameter fine-tuning.
- Multi-hop QA dataset: 125k samples trained for 1 epoch; KILT dataset: 660k samples trained for 1 epoch.
- Maximum sequence length: 3k tokens; retrieval corpus: English Wikipedia provided by KILT (~36M passages).
- For KILT evaluation, an E5-Mistral retriever and RankLLaMA reranker are additionally fine-tuned to improve ranking quality.

## Key Experimental Results

### Main Results on Multi-Hop QA

| Dataset | CoRAG-8B (best config) | Strongest Baseline | EM Gain |
|--------|-------------------|-------------|---------|
| 2WikiMultihopQA | **72.5** (L=10, N=8) | Search-o1-32B 58.0 | +14.5 |
| HotpotQA | **56.3** (L=10, N=8) | ITER-RETGEN 45.1 | +11.2 |
| MuSiQue | **30.9** (L=10, N=8) | ITER-RETGEN 26.1 | +4.8 |
| Bamboogle | 54.4 (L=10, N=8) | Search-o1-32B **56.0** | -1.6 |

> Bamboogle contains only 125 samples with high variance; some questions require knowledge more recent than the retrieval corpus, which naturally advantages systems using commercial search engines.

### KILT Benchmark (Hidden Test Set)

CoRAG-8B achieves state-of-the-art on 9/10 tasks: Entity Linking (AIDA 93.9), Slot Filling (T-REx 88.0, zsRE 87.2), Open QA (NQ 63.1, HotpotQA 60.6, TriviaQA 88.3), and Fact Verification (FEVER 93.1). The only task where it does not surpass the prior best is FEVER, where it trails Atlas-11B by a small margin (93.1 vs. 93.5), despite CoRAG having only 73% of Atlas's parameter count.

### Ablation Study

| Configuration | 2Wiki EM | HotpotQA EM | MuSiQue EM |
|------|----------|-------------|------------|
| CoRAG (L=6, greedy) | 70.6 | 54.4 | 27.7 |
| + Iterative rejection sampling (round 2) | 72.2 (+1.6) | 53.4 (-1.0) | 26.6 (-1.1) |
| + GPT-4o distillation | **75.1** (+4.5) | **56.6** (+2.2) | **28.2** (+0.5) |
| Weak-to-strong: 1B generate → 8B train | 59.3 | 50.3 | 22.3 |
| Weak-to-strong: 3B generate → 8B train | 69.9 | 53.9 | 25.2 |
| Replace retriever with BM25 + best-of-4 | 62.6 | 51.6 | 23.5 |
| Replace retriever with E5-base + best-of-4 | 70.8 | 53.0 | 26.3 |

### Retrieval Recall Improvement

| Dataset | E5-large R@10 | CoRAG R@10 | Gain |
|--------|--------------|-----------|------|
| HotpotQA | 59.1 | 72.1 | +13.0 |
| 2WikiMultihopQA | 54.9 | 81.4 | **+26.5** |
| Bamboogle | 31.2 | 59.2 | +28.0 |
| MuSiQue | 29.0 | 47.1 | +18.1 |

### Key Findings

- **Large gains on multi-hop tasks, limited gains on single-hop tasks**: Multi-hop datasets such as MuSiQue show consistent improvement as chain length and sampling count increase, whereas single-hop datasets such as NQ and TriviaQA yield diminishing marginal returns.
- **Weak-to-strong generalization is effective**: Retrieval chains generated by a 3B model can train an 8B model to near-equivalent performance (2Wiki EM 69.9 vs. 70.6), substantially reducing data generation costs.
- **Robustness to retriever quality**: Even when replaced at test time with a weaker retriever (BM25), the chain-of-retrieval mechanism compensates through multi-step querying.
- **Adaptive stopping remains immature**: Learning when to stop saves tokens but at the cost of performance degradation; the optimal configuration is dataset-dependent.

## Highlights & Insights

- **Breakthrough in data generation paradigm**: Rejection sampling elegantly addresses the core challenge of having no annotations for intermediate retrieval steps, constructing complete training signals from QA-only datasets without human annotation or closed-source models.
- **Quantitative guidance for test-time compute**: The log-linear scaling law provides a principled basis for latency–accuracy trade-offs in practical deployment.
- **"No information found" as a self-evaluation signal**: The model's output probability for "No relevant information found" serves as an internal measure of retrieval quality for Best-of-N selection, cleverly circumventing the unavailability of gold answers at test time.
- **Retrieval chains substantially improve recall**: Through iterative query reformulation, CoRAG improves R@10 by 13–28 percentage points across all multi-hop datasets, demonstrating that the method genuinely improves retrieval quality rather than only the final answer.

## Limitations & Future Work

- Evaluation is primarily on short-answer QA; long-form generation scenarios (e.g., summarization, report generation) are not covered.
- Adaptive chain length remains an open problem — a fixed $L$ lacks flexibility, but learning when to stop does not yet yield satisfactory results.
- Rejection sampling incurs high computational costs (up to 16 candidate chains per sample); weak-to-strong generalization can partially mitigate this.
- Scaling law coefficients vary across tasks, and a unified adaptive compute allocation mechanism is lacking.

## Related Work & Insights

- **vs. Standard RAG**: Single-step retrieval → multi-step iteration; EM improvement of 10–26 points on multi-hop tasks.
- **vs. FLARE / IRCoT / Self-RAG**: Shifts from prompt engineering or distillation to end-to-end fine-tuning, achieving superior system performance.
- **vs. Search-o1**: CoRAG-8B achieves 72.5 on 2Wiki, substantially outperforming Search-o1-32B (58.0) with only one-quarter of the parameters.
- **vs. IterDRAG**: Both study test-time scaling, but CoRAG acquires stronger foundational capabilities through fine-tuning rather than relying solely on few-shot prompting.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of rejection-sampled retrieval chains and test-time compute scaling is novel, though each component has known antecedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multi-hop QA, all KILT tasks, ablations, scaling analysis, weak-to-strong generalization, and retriever robustness — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, experimental analysis is thorough, and figures are highly informative.
- Value: ⭐⭐⭐⭐⭐ Achieves state-of-the-art on 9/10 KILT tasks, provides reproducible open-source code, and offers strong practical guidance for RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
