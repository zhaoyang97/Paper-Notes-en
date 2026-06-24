---
title: >-
  [Paper Note] SGIC: A Self-Guided Iterative Calibration Framework for RAG
description: >-
  [ACL 2025][Information Retrieval & RAG][Self-Calibration] SGIC utilizes token-level uncertainty scores (document relevance uncertainty + answer confidence uncertainty) of LLMs as guidance signals for self-calibration. By iteratively injecting the previous response and its uncertainty score into prompts to trigger in-context reasoning, it improves the EM of Llama2-7B from 69.1% to 77.2% (+8.1%) on HotpotQA, and also yields a 2.8% boost for GPT-4o.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Self-Calibration"
  - "Uncertainty Estimation"
  - "Iterative Reasoning"
  - "Document Relevance"
  - "Multi-hop QA"
date: 2026-05-08
content_hash: 01c62021a9b42e67
---

# SGIC: A Self-Guided Iterative Calibration Framework for RAG

**Conference**: ACL 2025  
**arXiv**: [2506.16172](https://arxiv.org/abs/2506.16172)  
**Code**: None  
**Area**: RAG / Retrieval-Augmented Generation  
**Keywords**: Self-Calibration, Uncertainty Estimation, Iterative Reasoning, Document Relevance, Multi-hop QA

## TL;DR

SGIC utilizes token-level uncertainty scores (document relevance uncertainty + answer confidence uncertainty) of LLMs as guidance signals for self-calibration. By iteratively injecting the previous response and its uncertainty score into prompts to trigger in-context reasoning, it improves the EM of Llama2-7B from 69.1% to 77.2% (+8.1%) on HotpotQA, and also yields a 2.8% boost for GPT-4o.

## Background & Motivation

**Background**: RAG enhances the generation capabilities of LLMs by retrieving external documents. Existing works mainly focus on retrieving more relevant documents or designing better instruction templates, but pay less attention to the self-calibration capability of LLMs—namely, iteratively improving generated answers using their own in-context reasoning capabilities.

**Limitations of Prior Work**: (1) Even when relevant documents are retrieved, LLMs often make mistakes in their initial answers to multi-hop questions, but existing RAG methods rarely provide a "second chance"; (2) Self-calibration methods (e.g., Self-Refine, CoT iteration) generally rely on complex prompt engineering or multi-model debates, which are cost-prohibitive; (3) LLMs' judgments on document relevance and answer confidence can be quantified via token probabilities, but this signal has not been systematically leveraged to guide calibration.

**Key Challenge**: The uncertainty signals (document relevance, answer confidence) of LLMs in RAG scenarios are naturally available self-supervision signals, yet existing methods fail to convert them into guidance signals for calibration.

**Goal**: To inject the uncertainty scores (product of token probabilities) of LLMs as explicit signals into iterative self-calibration prompts, enabling LLMs to perform targeted in-context reasoning and answer correction based on the uncertainty of prior answers.

**Key Insight**: It is observed that the uncertainty score distributions of LLMs for correct vs. incorrect answers and relevant vs. irrelevant documents exhibit distinct separation (as shown in Figure 1), providing a statistical foundation for using uncertainty as a calibration signal.

**Core Idea**: To compute the uncertainty scores of documents and answers using the product of token probabilities, inject these scores along with prior answers into prompts for iterative self-calibration, and construct a calibration training set to fine-tune open-source LLMs.

## Method

### Overall Architecture

The inference pipeline of SGIC is as follows: (1) Given all retrieved documents, an initial answer is generated, and its uncertainty score $s'_{ans}$ is computed; (2) An answer is generated for each document individually to compute the document uncertainty score $s'_{doc}$; (3) The documents (annotated with uncertainty) + prior answers (annotated with confidence) are reorganized into a new prompt, based on which the LLM performs in-context reasoning to generate a calibrated answer; (4) The process iterates for $K$ rounds. For open-source LLMs, a training set containing uncertainty information is additionally constructed for fine-tuning.

### Key Designs

1. **Dimensional Uncertainty Estimation**:

    - **Function**: Quantify the relevance of each document and the reliability of the generated answer, respectively.
    - **Mechanism**: Answer uncertainty $s_{ans} = \prod_{i=1}^m p_i$ (the product of the maximum probabilities of all tokens), normalized to [0, 100]. Document uncertainty $s_{doc} = 1 - \prod p_i$ (the complement of the product of probabilities when generating the answer using a single document), representing the degree of information insufficiency of the document. A high $s_{ans}$ indicates a credible answer, while a high $s_{doc}$ indicates an irrelevant document.
    - **Design Motivation**: Empirical results (Figure 1) show a clear separation between the uncertainty distributions of correct and incorrect answers, and the same holds for relevant and irrelevant documents, providing reliable guiding signals for calibration.

2. **Iterative Self-Calibration Inference**:

    - **Function**: Leverage prior answers and uncertainty scores as in-context clues to perform multi-round answer improvement.
    - **Mechanism**: The prompt for the $k$-th round contains: the original documents (each annotated with $s'_{doc}$) + the answers from the previous $k-1$ rounds with their corresponding $s'_{ans}$ (e.g., "Round 1: Lord High... (Uncertainty: 73), Round 2: United States... (Uncertainty: 51)"). The LLM performs in-context reasoning by comparing the changing trends of uncertainty in previous answers.
    - **Design Motivation**: Numerical changes in uncertainty scores provide explicit directional signals for calibration—if the prior uncertainty is high, a more significant correction is required; if it is already low, the model tends to preserve the answer.

3. **Calibration Training Set Construction (Open-source LLMs)**:

    - **Function**: Enable small open-source LLMs to effectively utilize uncertainty scores.
    - **Mechanism**: Perform inference on the training set to generate initial answers and uncertainties. Then, use the ground-truth answers as the target labels for calibration, constructing training samples with uncertainty signals for fine-tuning (via full-parameter or LoRA).
    - **Design Motivation**: Large-scale closed-source models (such as GPT-4o) naturally possess the ability to utilize numerical prompts, whereas small open-source models (such as Llama2-7B) require fine-tuning to comprehend the meaning of uncertainty scores.

### Loss & Training

Standard causal language modeling loss is used to fine-tune open-source LLMs. Llama2-7B is fine-tuned using LoRA, while Phi-3.5 is fine-tuned using full-parameter training.

## Key Experimental Results

### Main Results

| Model | Dataset | EM (Baseline) | EM (SGIC) | Gain |
|------|--------|----------|----------|------|
| Llama2-7B-Chat (LoRA) | HotpotQA | 69.1 | **77.2** | +8.1 |
| Llama2-7B-Chat (LoRA) | NQ | 74.7 | **79.0** | +4.3 |
| Phi-3.5-mini (Full) | HotpotQA | 42.8 | **55.3** | +12.5 |
| GPT-4o | HotpotQA | 73.7 | **76.5** | +2.8 |
| GPT-4o | NQ | 63.3 | **65.2** | +1.9 |
| GPT-4o-mini | HotpotQA | 69.2 | **74.1** | +4.9 |

### Ablation Study

Ablation of uncertainty components (Llama2-7B, HotpotQA EM):

| Configuration | EM | Description |
|------|-----|------|
| Calibration only (w/o uncertainty) | 71.8 | Basic calibration is effective |
| + Answer uncertainty | 76.2 | +4.4, answer confidence signal is key |
| + Document uncertainty | **77.2** | +1.0, document relevance signal is complementary |
| Oracle documents | 75.3 | Upper-bound reference |

By question type (Llama2-7B, HotpotQA):

| Question Type | EM (Baseline) | EM (SGIC) | Gain |
|---------|----------|----------|------|
| Bridge (Multi-hop) | 65.0 | **75.7** | +10.7 |
| Comparison | 69.6 | **83.1** | +13.5 |

### Key Findings

- **Iterative calibration is effective**: Multi-round calibration continuously improves performance and tends to saturate at $K=5$ rounds.
- **Answer uncertainty contributes the most**: In the ablation study, adding answer uncertainty brings a +4.4% EM gain, while document uncertainty brings an additional +1.0%.
- **Comparison questions benefit the most**: Comparison questions improve from 69.6% to 83.1% (+13.5%), possibly because the calibration signals for these questions are more explicit.
- **Closed-source models also benefit**: Strong models like GPT-4o also achieve a +2.8% boost, indicating that explicit uncertainty signals provide auxiliary value to any LLM.
- **Small models benefit more**: Phi-3.5-mini achieves a 12.5% improvement, demonstrating that models with lower initial capability benefit more from calibration guidance.

## Highlights & Insights

- **Uncertainty scores as explicit calibration signals**: Explicitly exposing the naturally occurring uncertainty information of LLMs as numerical annotations injected into prompts is simple and intuitive yet highly effective. This "self-aware calibration" concept can be generalized to any LLM task requiring iterative improvement.
- **Zero-additional-model self-calibration**: Compared to methods like multi-model debate, SGIC achieves effective calibration using only a single LLM, reducing computational costs.
- **Clever training set construction strategy**: Utilizing the model's own uncertainty coupled with ground-truth answers to construct calibration training samples enables small open-source models to learn to utilize uncertainty signals.

## Limitations & Future Work

- **Inaccuracy in uncertainty estimation is a bottleneck**: The product of token probabilities is a coarse measure of uncertainty; more precise estimation methods might further enhance performance.
- **Evaluated on only two multi-hop QA benchmarks**: Lack of evaluation on other RAG tasks, such as summarization or translation.
- **Increased inference cost from iterations**: $K=5$ calibration rounds imply a 5x increase in inference overhead.
- **Fine-tuning required for open-source models**: SGIC requires an additional fine-tuning step for open-source models to effectively utilize uncertainty scores.

## Related Work & Insights

- **vs. Self-RAG**: Self-RAG fine-tunes models to generate special tokens for judging retrieval needs, whereas SGIC guides iterative calibration using uncertainty scores. The two approaches are complementary.
- **vs. SeaKR**: SeaKR uses the uncertainty of internal states for retrieval decisions, while SGIC utilizes output probability uncertainty for answer calibration. They represent different perspectives with similar underlying philosophies.
- **vs. Self-Refine/CoT Iteration**: These methods lack quantified calibration signals, whereas SGIC provides clear directional guidance for improvement via uncertainty scores.

## Rating
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](../../ACL2026/information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)

</div>

<!-- RELATED:END -->
