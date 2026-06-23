---
title: >-
  [Paper Note] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs
description: >-
  [ACL 2026][Information Retrieval & RAG][Paper Note] Inspired by Schutz's philosophical theory of relevance, this paper proposes ITEM, an iterative utility judgment framework. By enabling dynamic interaction and mutual enhancement among three RAG components (relevance ranking, utility judgment, and answer generation), ITEM outperforms baselines in retrieval, utility judg
tags:
  - ACL 2026
  - Information Retrieval & RAG
date: 2026-05-08
content_hash: ddf2cc7ded437e94
---
# An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2406.11290](https://arxiv.org/abs/2406.11290)  
**Code**: [GitHub](https://github.com/Trustworthy-Information-Access/ITEM)  
**Area**: Information Retrieval / RAG  
**Keywords**: Utility Judgment, Philosophical Relevance Theory, Iterative Framework, RAG Optimization, LLM Reasoning

## TL;DR

Inspired by Schutz's philosophical theory of relevance, this paper proposes ITEM, an iterative utility judgment framework. By enabling dynamic interaction and mutual enhancement among three RAG components (relevance ranking, utility judgment, and answer generation), ITEM outperforms baselines in retrieval, utility judgment, and QA tasks.

## Background & Motivation

**Background**: In Retrieval-Augmented Generation (RAG) scenarios, the limited input bandwidth of LLMs necessitates prioritizing retrieved results with high utility rather than just high relevance. Relevance measures "aboutness" regarding a topic, while utility measures "helpfulness" in answering the specific question.

**Limitations of Prior Work**: (1) Existing RAG methods primarily optimize topical relevance while ignoring the higher standard of utility; (2) Although Zhang et al. explored LLM-based utility judgment, it remained preliminary; (3) The three components of RAG (retrieval, judgment, and generation) are typically optimized independently, lacking joint enhancement.

**Key Challenge**: Topically relevant documents are not necessarily useful—a document discussing the same topic without containing the specific answer is relevant but useless. Existing methods struggle to distinguish between the two.

**Goal**: To enhance LLM utility judgment capabilities through iterative interactions among the three RAG components.

**Key Insight**: The paper maps RAG to Schutz's philosophical "system of relevance"—topical relevance, interpretational relevance (utility), and motivational relevance (answer) correspond to three cognitive layers that can mutually reinforce each other.

**Core Idea**: The three components of RAG reflect three levels of LLM cognition in problem-solving (aboutness → value → answer), which can be enhanced through iteration.

## Method

### Overall Architecture

ITEM integrates retrieval, utility judgment, and answer generation—which are traditionally independent—into an iterative closed loop. Given a question and a set of candidate documents, the LLM first generates a pseudo-answer to serve as a cognitive anchor in each round. This pseudo-answer is then used to reassess document utility (and update relevance ranking if necessary). The updated utility judgment is used to refine the pseudo-answer, repeating the process until convergence to produce stable utility rankings and a final answer. The framework has two variants: ITEM-A cycles only between "answer generation + utility judgment," while ITEM-AR additionally incorporates "relevance ranking" into each round to circulate signals among all three components.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Question + Candidate Documents"] --> B["Answer Generation: Generate pseudo-answer<br/>as cognitive anchor (Motivational Relevance)"]
    B --> C["Pseudo-answer Driven Iterative Utility Judgment<br/>Scoring pointwise/listwise with reference to pseudo-answer (Interpretational Relevance)"]
    C -->|"ITEM-AR: Extra update per round"| D["Relevance Ranking<br/>Rerank docs containing answers to the top (Topical Relevance)"]
    D --> E{"Utility Ranking Converged?"}
    C -->|"ITEM-A: Generation + Judgment cycle only"| E
    E -->|"No, feedback high-utility docs"| B
    E -->|Yes| F["Stable Utility Ranking + Final Answer"]
```

### Key Designs

**1. Pseudo-answer Driven Iterative Utility Judgment: Approaching true utility using the previous round’s answer as an anchor.**

The difficulty with one-shot utility judgment is that LLMs are forced to score documents before fully understanding exactly what the question requires, making them prone to biases from surface topical relevance. ITEM's approach involves letting the LLM generate a pseudo-answer based on available documents first, explicitly articulating "what kind of content is needed for this question." This serves as a reference for utility judgment (via pointwise scoring or listwise reranking). The high-utility documents are then used to regenerate the pseudo-answer. As the pseudo-answer becomes more accurate each round, utility judgment improves accordingly, allowing the model to accumulate understanding rather than making a single, final decision.

**2. Mapping Schutz’s Relevance System to RAG: Theoretical basis for mutual component enhancement.**

The effectiveness of iteration is grounded in Schutz's philosophical relevance theory. Schutz decomposes human cognition into three layers: Topical relevance (focusing attention on an object), Interpretational relevance (understanding the value of that object), and Motivational relevance (taking action based on understanding). These layers correspond exactly to RAG's components: Relevance ranking → Utility judgment → Answer generation. Theory predicts these layers reinforce each other—deeper understanding leads to more precise focus and better action—providing the theoretical foundation for ITEM's cyclic feedback loop.

**3. ITEM-A vs. ITEM-AR Strategy: Selecting components and rounds based on task complexity.**

Not all tasks require all three components. ITEM-A iterates only on answers and judgment, allowing for many rounds of deep exploration with fewer components. ITEM-AR includes relevance ranking in every round, providing richer information per round at a higher cost. The researchers found that ITEM-AR is better for complex candidate lists and difficult non-factual tasks (e.g., WebAP, GTI-NQ), while simple factual QA tasks perform best with ITEM-A’s lightweight loop and more iterations. This makes the "iteration resource investment" a tunable parameter based on the task.

### A Complete Example

Consider a factual question: Initial retrieval provides 5 topically relevant documents, but only 2 contain the actual answer. In Round 1, the LLM generates a rough pseudo-answer based on all 5 documents. In Round 2, it uses this pseudo-answer to re-judge utility, demoting the 3 "relevant but answerless" documents and generating a more accurate pseudo-answer from the remaining high-utility ones. If using ITEM-AR, the document containing the answer is also ranked higher. After a few rounds, the utility ranking stabilizes and the pseudo-answer converges to the final output. The entire process requires no training and relies on prompt design to switch between generation, judgment, and ranking tasks.

### Loss & Training

ITEM is based entirely on the in-context learning of LLMs without any training. The specific task for each round (answer generation, utility judgment, or relevance ranking) is controlled entirely through prompt engineering.

## Key Experimental Results

### Main Results

| Task | Dataset | ITEM Gain | Description |
|------|--------|---------|------|
| Retrieval Ranking | TREC DL | Better than baselines | Utility judgment improves ranking in turn |
| Utility Judgment | GTI-NQ | Better than baselines | Iteration significantly improves judgment quality |
| QA | NQ | Better than baselines | High-utility documents lead to better answers |
| Non-factual Retrieval | WebAP | Better than baselines | Larger gains for difficult tasks |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| 1 Round vs. Multi-round | Multi-round better | Iteration is indeed effective |
| ITEM-A vs. ITEM-AR | Task dependent | Difficult tasks require ITEM-AR |
| vs. Long-reasoning mode | Comparable performance, much lower cost | Iteration is more efficient than one-shot long reasoning |

### Key Findings
- For difficult tasks (e.g., WebAP non-factual retrieval) and complex candidate lists (e.g., GTI-NQ), more components and more iterations are most effective.
- ITEM achieves performance comparable to long-reasoning modes but at a significantly lower computational cost.
- For simple factual QA tasks, fewer components combined with more iterations yield the best results.

## Highlights & Insights
- Creative mapping from philosophical theory to engineering—Schutz's relevance system offers a new perspective for RAG optimization.
- Identifies the relationship between task complexity and the optimal iteration strategy.
- Improves RAG quality without the need for training, offering high practical utility.

## Limitations & Future Work
- Iteration increases inference cost (multiple LLM calls), which may result in unacceptable latency.
- The quality of the pseudo-answer may limit the upper bound of iterative gains.
- Tested only on English datasets.
- Future work could combine this with fine-tuned retrievers for further enhancement.

## Related Work & Insights
- **vs. Single-shot Utility Judgment**: The iterative framework significantly improves judgment quality through multi-round cognitive accumulation.
- **vs. Multi-round Retrieval RAG**: Does not change the retrieval process itself but iterates to improve utility judgment on already retrieved results.
- **vs. Long Reasoning/CoT**: Achieves comparable effects at a lower cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative perspective mapping philosophy to RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 4 datasets and multiple tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical framework and well-organized experiments.
- Value: ⭐⭐⭐⭐ Practical significance for RAG optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SGIC: A Self-Guided Iterative Calibration Framework for RAG](../../ACL2025/information_retrieval/sgic_a_self-guided_iterative_calibration_framework_for_rag.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/information_retrieval/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)

</div>

<!-- RELATED:END -->
