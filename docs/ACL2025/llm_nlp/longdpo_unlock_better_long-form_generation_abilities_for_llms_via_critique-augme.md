---
title: >-
  [Paper Note] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information
description: >-
  [ACL 2025][LLM (Other)][Long-form text generation] This paper proposes LongDPO, which collects step-level preference pairs via MCTS, maintains factual consistency using a global memory pool, and enhances low-quality candidates through critiques. It then performs fine-grained optimization using stepwise DPO, significantly improving long-form text generation quality on LongBench-Write while preserving general capabilities.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Long-form text generation"
  - "stepwise DPO"
  - "MCTS"
  - "preference learning"
  - "critique enhancement"
date: 2026-05-08
content_hash: c60d815e94525b05
---

# LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information

**Conference**: ACL 2025  
**arXiv**: [2502.02095](https://arxiv.org/abs/2502.02095)  
**Code**: [https://github.com/pingbowen23/LongDPO](https://github.com/pingbowen23/LongDPO)  
**Area**: LLM NLP  
**Keywords**: Long-form text generation, stepwise DPO, MCTS, preference learning, critique enhancement

## TL;DR
This paper proposes LongDPO, which collects step-level preference pairs via MCTS, maintains factual consistency using a global memory pool, and enhances low-quality candidates through critiques. It then performs fine-grained optimization using stepwise DPO, significantly improving long-form text generation quality on LongBench-Write while preserving general capabilities.

## Background & Motivation

**Background**: LLMs have made significant progress in long-context input (e.g., GPT-4o supporting 128K tokens), but remain weak in long-form text **generation**—most models struggle to generate high-quality content exceeding 2,000 words.

**Limitations of Prior Work**: Existing methods such as Suri and LongWriter mostly employ **outcome supervision**, directly applying DPO to the entire long-form text. However, long-form text is more prone to issues like logical inconsistency, factual hallucination, and failure to meet instruction requirements, which outcome supervision struggles to pinpoint.

**Key Challenge**: Quality degradation in long-form text is **gradually cumulative**—early errors propagate to subsequent content. However, outcome-level DPO only provides coarse-grained feedback signals at the global level, offering marginal reward differentiation and low learning efficiency.

**Goal**: (a) Building step-level preference data to provide fine-grained supervision; (b) maintaining factual consistency during the search process; and (c) improving the quality of low-reward candidates.

**Key Insight**: Process supervision has been proven superior to outcome supervision in mathematical reasoning; long-form text generation can naturally be decomposed into multiple steps, making it highly suitable for process supervision.

**Core Idea**: Utilizing MCTS to search for step-level preference pairs of long-form text, integrated with a global memory pool for consistency checking and critique enhancement, followed by training via stepwise DPO.

## Method

### Overall Architecture
LongDPO consists of two main components: (1) **Step-level preference data construction**—MCTS is utilized to expand a generation tree step by step, extracting a chosen/rejected pair from each layer. Specially, a global memory pool is employed to filter inconsistent nodes, and low-reward chosen candidates are enhanced with critiques; (2) **Stepwise DPO training**—The long-form text $y$ is decomposed into $s_1 \oplus s_2 \oplus \cdots \oplus s_t$, and DPO learning is performed independently at each step.

### Key Designs

1. **MCTS for Step-Level Preference Pair Collection** (MCTS for step-level preference collection):

    - **Function**: Models the long-form text generation process as a tree search. Each layer (step) generates multiple candidate segments, evaluates their quality with a reward model, and selects the optimal one and a random one as the chosen/rejected pair, respectively.
    - **Mechanism**: The node is defined at the $t$-th layer of the tree as $s_{t+1} = \pi_\theta(q \oplus s_1 \oplus \cdots \oplus s_t)$. Node selection uses the UCB formula to balance exploration and exploitation: $\text{UCB}_i = \alpha \sqrt{2 \ln(N_i / (1 + n_i))} + v_i$. Each step is evaluated based on 7 principles (factuality, coherence, etc.) to get the average reward.
    - **Design Motivation**: Searching finds higher quality preference pairs than direct sampling, and step-level decomposition provides more accurate reward signals.

2. **Global Memory Pool**:

    - **Function**: Filters out candidate nodes that contradict prior factual content during the MCTS selection phase.
    - **Mechanism**: Once a node is selected, its factual content is extracted and stored in the memory pool $M_t$. During the next selection step, the candidate node is split into 128-word segments. Embedding similarity (using gte-Qwen2-1.5B) is calculated between these segments and each fact in the memory pool. Segments with a similarity $\geq 0.8$ undergo a consistency check against the corresponding facts; if inconsistent, the candidate is skipped.
    - **Design Motivation**: Factual inconsistency is the most common quality issue in long-form generation. The memory pool eliminates contradictory candidates during the search phase rather than repairing them post-hoc.

3. **Critique-Augmented Generation**:

    - **Function**: Generates improvement suggestions using an external critique model for chosen candidates with rewards below a threshold $\eta = 2.5$, and then regenerates them.
    - **Mechanism**: For each low-reward $s_{win}$, a sibling node $s_{sib}$ performing better under a certain principle is identified to construct a triplet (principle, sibling node, node to be improved). The critique model generates analysis, justification, relevant text, confidence score, and writing suggestions. These suggestions $z_\lambda$ are then injected into the regeneration process: $s_{win\_new} = \pi_\theta(q \oplus s_1 \oplus \cdots \oplus s_t \oplus z_\lambda)$.
    - **Design Motivation**: Simply expanding the search space is inefficient and yields limited gains; directly guiding improvement using critiques is more efficient, and external critiques are more reliable than self-reflection.

4. **Step-Level DPO Training** (Stepwise DPO training):

    - **Function**: Utilizes the collected step-level preference pairs for fine-grained DPO learning.
    - **Mechanism**: $\mathcal{L}_{LongDPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(s_w|q'))}{\pi_{ref}(s_w|q')} - \beta \log \frac{\pi_\theta(s_l|q')}{\pi_{ref}(s_l|q')})]$, where $q' = q \oplus s_{1 \sim i}$.
    - **Design Motivation**: While traditional DPO conducts preference learning once across the entire long-form text, LongDPO independently performs preference learning at each individual step, delivering clearer reward signals and preventing signal dilution across long sequences.

## Key Experimental Results

### Main Results (LongBench-Write-en)

| Model | [0,500) $S_l$/$S_q$ | [2k,4k) $S_l$/$S_q$ | [4k,20k) $S_l$/$S_q$ | Avg $S_l$/$S_q$ |
|---|---|---|---|---|
| LongWriter-Llama | 88.1/86.0 | 89.1/88.3 | 80.8/79.2 | 83.1/85.1 |
| + DPO | 90.9/85.8 | 90.0/90.5 | 81.1/80.9 | 85.6/85.7 |
| + **LongDPO** | **90.7/86.3** | **93.4/90.5** | **88.3/85.1** | **87.4/88.3** |
| LongWriter-Qwen | 90.8/88.0 | 84.2/84.8 | 58.7/78.1 | 79.5/85.1 |
| + DPO | 86.3/88.2 | 89.3/84.1 | 60.9/78.8 | 81.3/- |
| + **LongDPO** | **91.6/88.6** | **92.1/88.3** | **81.2/84.9** | **87.7/87.6** |

- LongDPO achieves an overall improvement of +4.3/+3.2 ($S_l$/$S_q$) on Llama, with the most significant gain in the long interval [4k,20k): +7.5 on $S_l$.
- On Qwen, the performance in the long interval [4k,20k) increases from 58.7 to 81.2 (+22.5), significantly enhancing the ultra-long-form generation capabilities.

### Ablation Study

| Configuration | LongBench-Write Avg $S_l$ | Description |
|---|---|---|
| LongDPO (full) | 87.38 | Full method |
| w/o Memory Pool | 85.91 (↓1.47) | Disabling consistency check leads to accumulation of factual errors |
| w/o Critique | 86.12 (↓1.26) | Disabling critique enhancement leaves low-reward chosen candidates unimproved |
| w/o Stepwise (→ outcome DPO) | 85.55 (↓1.83) | Degenerates into traditional outcome DPO |
| LongDPO w/ self-critique | 86.43 (↓0.95) | Replacing external critique with self-reflection leads to performance degradation |

### Key Findings
- **Step-level supervision is the most critical contribution**: Removing step-level learning (converting to traditional DPO) leads to a drop of 1.83 points, demonstrating that process supervision indeed outperforms outcome supervision in long-form text generation.
- **The Memory Pool contributes most to long intervals**: In the [4k,20k) interval, the performance drop is more pronounced without the memory pool, as longer texts are more susceptible to factual inconsistency.
- **External critique outperforms self-reflection**: This aligns with the findings of Qi et al. (2024), showing that model self-reflection yields unstable improvements.
- **General capabilities are largely preserved**: Performance is maintained or marginally improved on general benchmarks such as TruthfulQA.

## Highlights & Insights
- **The combined design of MCTS and the memory pool is highly ingenious**: MCTS offers search capability while the memory pool tackles factual consistency issues unique to long texts, complementing each other. The memory pool is incrementally updated during the search process, serving as a lightweight "long-term memory."
- **Critique enhancement targets only low-reward candidates**: Rather than applying critiques to all candidates (which is computationally expensive), it only refines those with $r \le 2.5$, balancing both efficiency and effectiveness.
- **Transferring process supervision from reasoning to generation**: Historically, process supervision has primarily been deployed in mathematical reasoning (e.g., MCTS + PRM). This paper demonstrates its efficacy in open-ended long-form generation, expanding the applicability of process supervision.

## Limitations & Future Work
- **High MCTS search cost**: Each query requires multiple expansion and evaluation steps, incurring significantly higher data construction costs than direct sampling.
- **Reliance on external reward models**: Utilizing LLMs for 7-principle evaluations introduces reward model bias that can propagate into preference data.
- **Unoptimized step splitting**: The steps seem to be partitioned using a fixed length, which might be sub-optimal for different types of long-form text tasks.
- **Limited experimental model scale**: Validation is primarily conducted on LongWriter-Llama/Qwen (both within the 7B-9B range); the efficacy on larger models remains unexplored.
- Directions for improvement: Adaptive step partition, leveraging lightweight reward models instead of LLM judges, and integration with LongReward.

## Related Work & Insights
- **vs LongWriter (Bai et al. 2024)**: LongWriter decomposes tasks via an agent pipeline for data construction combined with outcome DPO. In contrast, LongDPO uses MCTS to construct step-level data paired with stepwise DPO, providing finer-grained supervision.
- **vs Suri (Pham et al. 2024)**: Suri performs outcome-level preference optimization by creating different instructions for the same response, while LongDPO collects preference pairs at the step level.
- **vs LongReward (Zhang et al. 2024)**: LongReward requires external reference documents as inputs, which limits its applicability. LongDPO does not require reference documents, making it more generalizable.
- This paper proves the value of process supervision in long-form text generation. Future work could consider applying similar methodologies to tasks like long-form code generation and academic paper writing.

## Rating
- Novelty: ⭐⭐⭐⭐ Transferring MCTS and process supervision from reasoning to long-form text generation is novel. The designs of the memory pool and critique enhancement are well-justified.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across two backbone models, multiple length intervals, detailed ablation studies, and preservation of general capabilities.
- Writing Quality: ⭐⭐⭐⭐ Clarified methodology with comprehensive mathematical formulations.
- Value: ⭐⭐⭐⭐ Addresses critical bottlenecks in long-form generation, and the codebase/methodology can be generalized to other long-sequence generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Stepwise Reasoning Disruption Attack of LLMs](seed_stepwise_reasoning_disruption_attack.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)

</div>

<!-- RELATED:END -->
