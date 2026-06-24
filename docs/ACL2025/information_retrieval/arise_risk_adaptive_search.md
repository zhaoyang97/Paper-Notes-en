---
title: >-
  [Paper Note] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search
description: >-
  [ACL 2025][Information Retrieval & RAG][Knowledge-Augmented Reasoning] Proposes the ARise framework, which integrates Bayesian risk assessment and dynamic RAG into Monte Carlo Tree Search to address the error propagation and verification bottleneck issues in knowledge-augmented reasoning. On multi-hop QA tasks, it outperforms state-of-the-art KAR methods by 23.10% and RAG-equipped reasoning models (DeepSeek-R1) by 25.37% in average accuracy.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Knowledge-Augmented Reasoning"
  - "MCTS"
  - "Risk Assessment"
  - "RAG"
  - "Multi-Hop Question Answering"
date: 2026-05-08
content_hash: 3b55247873a06044
---

# ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search

**Conference**: ACL 2025  
**arXiv**: [2504.10893](https://arxiv.org/abs/2504.10893)  
**Code**: [https://opencausalab.github.io/ARise](https://opencausalab.github.io/ARise)  
**Area**: Other  
**Keywords**: Knowledge-Augmented Reasoning, MCTS, Risk Assessment, RAG, Multi-Hop Question Answering  

## TL;DR

Proposes the ARise framework, which integrates Bayesian risk assessment and dynamic RAG into Monte Carlo Tree Search to address the error propagation and verification bottleneck issues in knowledge-augmented reasoning. On multi-hop QA tasks, it outperforms state-of-the-art KAR methods by 23.10% and RAG-equipped reasoning models (DeepSeek-R1) by 25.37% in average accuracy.

## Background & Motivation

**Background**:
   - The improvement of LLMs' reasoning capabilities mainly relies on test-time compute scaling (e.g., System 2 slow thinking).
   - Large Reasoning Models (LRMs) like DeepSeek-R1 have achieved expert-level performance in mathematics and coding.
   - RAG is an effective way to obtain external knowledge, and CoT prompting can embed retrieval into reasoning steps.

**Limitations of Prior Work**:
   - **Limitations of Reasoning Methods**: LRMs implicitly assume that the model already possesses all the knowledge required for reasoning, which fails in open-domain scenarios (e.g., law, medicine).
   - **Error Propagation**: In CoT-based knowledge-augmented reasoning, errors in early steps cascade and amplify throughout the reasoning chain.
   - **Verification Bottleneck**: The explore-exploit trade-off in multi-branch decision-making processes is difficult to resolve effectively. Prior verification schemes either rely on unreliable self-verification or require training dedicated verifiers.

**Key Challenge**:
   - Knowledge augmentation (RAG) and reasoning enhancement (search/reasoning) need to work synergistically, but existing methods fail to combine them effectively.
   - How to evaluate the quality of intermediate reasoning states in multi-branch search? Self-verification is unreliable, and external verifiers are costly.

**Goal**:
   - How to effectively combine knowledge retrieval and reasoning search in open-domain, knowledge-intensive, and complex reasoning scenarios.
   - How to dynamically assess the risk of reasoning paths in tree search to balance exploration and exploitation.

**Key Insight**:
   - Introduce Bayesian risk minimization into node evaluation within MCTS, using "question generation likelihood" as a proxy metric for the quality of intermediate states.
   - Each step consists of two actions: question decomposition and retrieval-based reasoning, refining the reasoning granularity.

**Core Idea**:
   - Use Bayesian risk assessment to guide the explore-exploit trade-off in knowledge-augmented reasoning within MCTS.

## Method

### Overall Architecture

ARise consists of three core components:
1. **Reasoning State Generation**: Each step consists of question decomposition and retrieval-based reasoning.
2. **Monte Carlo Tree Search**: Expands linear reasoning into a tree structure.
3. **Risk Assessment**: Uses Bayesian risk minimization to evaluate intermediate reasoning states.

### Key Designs

1. **Reasoning State Generation**:

    - **Function**: At each step, the LLM performs question decomposition and reasoning based on retrieved documents, appending intermediate results to the reasoning state.
    - **Mechanism**: The input at step $i$ is the original question $\mathbf{q}$ + the previous context $\mathbf{s_{i-1}}$. The model first generates a sub-question $\mathbf{d_i}$, then combines it with retrieved documents to obtain the reasoning result $\mathbf{r_i}$.
    - **Design Motivation**: The alternation of decomposition and retrieval-based reasoning provides finer-grained knowledge acquisition, where each step has a well-defined (state, action) pair.

2. **Monte Carlo Tree Search (MCTS)**:

    - **Function**: Comprises four phases: Selection (UCT), Expansion (multi-angle decomposition), Simulation (imaginative rollout), and Backpropagation (bottom-up update).
    - **Mechanism**: 
        - **Selection**: $\text{UCT}(\mathbf{s}, \mathbf{a}) = Q(\mathbf{s}, \mathbf{a}) + w\sqrt{\frac{\ln N(Pa(\mathbf{s}))}{N(\mathbf{s}, \mathbf{a})}}$
        - **Backpropagation**: $Q(\mathbf{s}, \mathbf{a}) = \frac{\sum_{\mathbf{c}} Q(\mathbf{c}) \cdot N(\mathbf{c})}{\sum_{\mathbf{c}} N(\mathbf{c})}$
    - **Design Motivation**: Expanding linear CoT reasoning into a tree structure allows backtracking and multi-path exploration, mitigating error propagation.

3. **Risk Assessment**:

    - **Function**: Converts the intermediate result quality of nodes into a calculable "question generation likelihood" using Bayes' theorem.
    - **Mechanism**: 
        - Relevance: $\log p(\mathbf{r}|\mathbf{q}) \propto \log p(\mathbf{q}|\mathbf{r})$
        - Risk: $\text{Risk}((\mathbf{s}, \mathbf{a}) \to \mathbf{r}|\mathbf{q}) = -\frac{1}{|\mathbf{q}|}\sum_t \log p(q_t | \mathbf{q}_{<t}, \mathbf{r}; \Theta)$
        - Value: $Q(\mathbf{s}, \mathbf{a}) = 1 - \frac{1}{1+e^{\alpha(\text{Risk} - \beta)}}$
    - **Design Motivation**: Risks are computed using the policy model itself, eliminating the need for training separate verifiers; a low risk indicates high relevance between the intermediate result and the original question.

### Loss & Training

- **No Training Required**: ARise is a pure test-time reasoning framework that does not require model fine-tuning.
- **Key Hyperparameters**: The exploration weight $w$ in UCT, and the shift/scale factors $\alpha, \beta$ in the sigmoid.
- **Policy Models**: Qwen2.5-7B/14B-Instruct and Llama3.1-8B-Instruct.
- **Retriever**: Uses standard retrieval systems to dynamically fetch relevant documents.

## Key Experimental Results

### Main Results

Evaluation on three multi-hop QA benchmarks (using Qwen2.5-14B-Instruct):

| Method | HotpotQA (EM/F1) | 2Wiki (EM/F1) | MusiQue (EM/F1) | Average (EM/F1) |
|------|-----------------|--------------|----------------|-------------|
| Vanilla | 59.50/63.63 | 37.00/50.33 | 14.50/47.07 | 37.00/53.68 |
| Self-Ask | 58.50/64.74 | 38.50/53.45 | 25.00/58.59 | 40.67/58.93 |
| Auto-RAG | 68.00/66.64 | 53.00/55.13 | 35.50/59.05 | 52.17/60.27 |
| RATT | 64.50/73.91 | 43.00/57.48 | 24.00/63.76 | 43.83/65.05 |
| **ARise** | **73.50/75.39** | **56.50/62.61** | **40.50/65.87** | **56.83/67.96** |

- ARise outperforms the best baseline, Auto-RAG, by 4.66% in average EM, and outperforms RATT by 2.91% in F1.
- On the hardest MusiQue dataset, EM is 40.50 vs. Auto-RAG's 35.50 (+5.0%).

Qwen2.5-7B-Instruct Results:
- ARise achieves an average EM of 47.67 and F1 of 65.83, significantly outperforming all baselines.

Llama3.1-8B-Instruct Results:
- ARise achieves an average F1 of 68.12, maintaining its advantage even on smaller models.

### Key Findings

1. **ARise significantly outperforms SOTA KAR methods**: Achieving an average accuracy improvement of 23.10% and F1 improvement of 15.52%.
2. **Outperforming RAG-equipped LRMs**: Improving average accuracy and F1 by 4.04% and 25.37% respectively, compared to DeepSeek-R1 equipped with RAG.
3. **Search-based wide reasoning outperforms learning-based deep reasoning**: Experiments demonstrate that in open domains, search-based multi-path exploration is more effective than single-path deep thinking of deep reasoning models.
4. **Model scaling**: The performance of ARise scales effectively as model size increases, demonstrating promising scalability.
5. **Effectiveness of risk assessment**: Ablation studies validate the critical role of Bayesian risk assessment in guiding the search direction.

## Highlights & Insights

- **Knowledge acquisition is necessary for reasoning**: Highlights that the "implicit complete knowledge assumption" of LRMs does not hold in open domains.
- **Mathematical elegance of risk assessment**: Leverages Bayes' theorem to convert the verification problem into a calculable self-evaluation by the policy model, avoiding external verifiers.
- **In-depth analysis of error propagation vs. verification bottleneck**: Clearly delineates the two core challenges of KAR.
- **Natural integration of MCTS and RAG**: Every node in the tree search includes retrieval operations, seamlessly fusing search and knowledge acquisition.
- **No training, plug-and-play**: A test-time framework applicable to any LLM.

## Limitations & Future Work

1. High inference cost of MCTS: Multiple expansions, simulations, and backpropagations significantly increase LLM call counts.
2. Risk assessment relies on the conditional likelihood of the policy model itself, which might lead to inaccurate estimations in smaller models.
3. Evaluated only on multi-hop QA; other knowledge-intensive reasoning tasks (e.g., scientific reasoning, legal reasoning) remain unverified.
4. The upper bound of retrieval quality limits the reasoning quality; the performance of the underlying retriever is not extensively discussed.
5. Hyperparameters ($w, \alpha, \beta$) of the UCT and risk functions require tuning.

## Related Work & Insights

- **RATT (Zhang et al., 2024)**: Tree-based RAG, but with a simplified verification mechanism.
- **Auto-RAG (Yu et al., 2024)**: Automated retrieval augmentation, prompt-based.
- **DeepSeek-R1 (2025)**: State-of-the-art reasoning model, but underperforms RAG-equipped methods in knowledge-intensive scenarios.
- Insight: Test-time compute scaling should not be limited to deep thinking; wide search (multi-path exploration) combined with knowledge retrieval may be more effective in open domains.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 8 |
| **Total Score** | **8.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2025\] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](toward_structured_knowledge_reasoning_contrastive_retrieval-augmented_generation.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)

</div>

<!-- RELATED:END -->
