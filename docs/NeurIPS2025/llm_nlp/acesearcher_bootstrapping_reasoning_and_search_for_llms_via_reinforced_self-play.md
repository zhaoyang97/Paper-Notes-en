---
title: >-
  [Paper Note] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play
description: >-
  [NeurIPS 2025][LLM/NLP][Search-augmented LLM] This paper proposes AceSearcher—a collaborative self-play framework in which a single LLM simultaneously plays two roles: a **decomposer** (breaking complex queries into sub-questions to guide retrieval) and a **solver** (integrating retrieved context to generate answers). Through a two-stage training pipeline of SFT followed by iterative DPO, using only final-answer rewards, AceSearcher achieves an average EM improvement of 7.6% across 10 datasets, and the 32B model matches DeepSeek-V3 with fewer than 5% of its parameters.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Search-augmented LLM
  - multi-hop reasoning
  - question decomposition
  - self-play
  - iterative DPO
date: 2026-05-08
content_hash: 39995c3985b414ef
---

# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play

**Conference**: NeurIPS 2025
**arXiv**: [2509.24193](https://arxiv.org/abs/2509.24193)
**Code**: [GitHub](https://github.com/ritaranx/AceSearcher/) / [HuggingFace](https://huggingface.co/AceSearcher)
**Area**: LLM/NLP
**Keywords**: Search-augmented LLM, multi-hop reasoning, question decomposition, self-play, iterative DPO

## TL;DR
This paper proposes AceSearcher—a collaborative self-play framework in which a single LLM simultaneously plays two roles: a **decomposer** (breaking complex queries into sub-questions to guide retrieval) and a **solver** (integrating retrieved context to generate answers). Through a two-stage training pipeline of SFT followed by iterative DPO, using only final-answer rewards, AceSearcher achieves an average EM improvement of 7.6% across 10 datasets, and the 32B model matches DeepSeek-V3 with fewer than 5% of its parameters.

## Background & Motivation

**Background**: RAG has become the standard approach for compensating for LLMs' knowledge gaps, but most RAG methods address only simple single-hop retrieval scenarios.

**Limitations of Prior Work**: Complex reasoning tasks require (a) multi-hop retrieval to gather multiple evidence pieces and (b) reasoning to integrate information and generate answers. Existing methods either rely on powerful closed-source LLMs with multi-turn prompting (high cost), employ online RL training (memory-intensive), or perform supervised fine-tuning solely on QA data (limited generalization).

**Key Challenge**: Multi-hop retrieval requires knowing *what to search for* (question decomposition), and once results are retrieved, strong reasoning is also needed—these two capabilities are tightly coupled yet are handled separately by existing approaches.

**Key Insight**: Train a single model to learn both roles—decomposer and solver—improving each other through collaborative self-play.

**Core Idea**: The quality of the decomposer is measured by the solver's accuracy, while the quality of the solver is measured by final-answer match—requiring no intermediate step annotations.

## Method

### Overall Architecture
Given a complex question $q$, the decomposer generates a sequence of sub-questions $z = (z_1, ..., z_n)$; for each sub-question, a retriever fetches context $\mathcal{D}_i$; the solver then iteratively generates intermediate answers $w_i$ and ultimately integrates them into a final answer $a'$. Both the decomposer and solver are implemented as the **same LLM**, with roles distinguished via different prompt templates.

### Key Designs

1. **Two-Role Collaboration (Decomposer + Solver)**:

    - Decomposer $\rho$: $z \sim p_\theta(\cdot|q)$, decomposing the original question into sub-question templates.
    - Solver $\pi$: $w_i \sim p_\theta(\cdot|z_i, w_{<i}, \mathcal{D}_i)$, generating intermediate answers conditioned on sub-questions, prior answers, and retrieved context.
    - Realized by a unified LLM, with roles differentiated through input templates.
    - Joint objective: $p_\theta(a|q) = \sum_z p_\theta(z|q) \sum_w p_\theta(a|q,z,w) p_\theta(w|q,z)$

2. **Stage I: SFT Data Mixture**:

    - Function: Establish foundational capabilities—retrieval usage, question decomposition, and reasoning.
    - Data composition (180K samples total):
        - Context-rich QA: NQ, SQuAD, DROP, NarrativeQA, etc.—training the model to extract answers from context.
        - Question decomposition: GSM8K, ConvFinQA, StrategyQA—training multi-step decomposition.
        - CoT reasoning: MathInstruct (CoT + PoT style)—strengthening multi-step reasoning.
    - Design Motivation: Existing RAG fine-tuning focuses solely on answer extraction, neglecting decomposition and reasoning capacity.

3. **Stage II: Iterative DPO Reinforcement Fine-tuning**:

    - Function: Jointly optimize both roles using data with only final-answer annotations.
    - Mechanism:
        - For each question, sample $m$ decomposition plans; for each plan, sample $m'$ solver trajectories.
        - Construct preference pairs using the final EM reward $r(q,a',a) = \text{EM}(a',a) \times \mathbb{I}(f(q,a')=1)$.
        - Decomposer: select best/worst decompositions by expected reward $\bar{r}(q, z^{(i)})$.
        - Solver: select best/worst trajectories by final-answer match.
        - Joint preference set: $\mathcal{D}_{pref} = \mathcal{D}_{decompose} \cup \mathcal{D}_{subq} \cup \mathcal{D}_{final}$
    - Theoretical guarantee: Theorem 4.1 proves that the minimizer of iterative DPO converges to the true optimal parameters $\theta^*$.
    - Key advantage: No intermediate annotations required; no high memory overhead of online RL.

4. **Multiple Evaluation Environments**:

    - RAG setting: HotpotQA, 2WikiMHQA, HOVER—requiring multi-hop retrieval.
    - Context-rich reasoning: GSM8K, TabMWP, ConvFinQA—requiring contextual reasoning.
    - Unified reward signal: EM based solely on final answers.

### Loss & Training
- Backbone models: Qwen-2.5-Instruct (1.5B/14B/32B) and Llama-3.1-8B-Instruct.
- LoRA (r=8, α=16) for the 32B model; full fine-tuning for others.
- Batch size 64, max tokens 2048, SFT for 1 epoch, RFT for 2 iterative rounds.
- Maximum $\lfloor 15/n \rfloor$ documents per sub-question.

## Key Experimental Results

### Main Results (Multi-hop QA + Fact Verification)

| Method | Params | 2WikiMHQA | HotpotQA | Average |
|--------|--------|-----------|----------|---------|
| Search-R1 | 7B | baseline | baseline | baseline |
| DeepResearcher | 7B | baseline | baseline | baseline |
| **AceSearcher-8B** | 8B | +significant | +significant | **+7.6% avg** |
| AceSearcher-1.5B | 1.5B | surpasses 9× larger models | — | highly efficient |
| AceSearcher-32B | 32B | matches DeepSeek-V3 | — | <5% parameters |

### Document-level Reasoning (DocMath-Eval)

| Method | TAT-QA | FinQA | Average |
|--------|--------|-------|---------|
| GPT-4o | baseline | baseline | baseline |
| DeepSeek-V3 | strong | strong | strong |
| **AceSearcher-32B** | **matches V3** | **matches V3** | **~comparable** |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| SFT only | baseline | baseline capability |
| SFT + 1-round DPO | +significant | reinforcement fine-tuning effective |
| SFT + 2-round DPO | +further | iterative gains |
| Remove decomposition data | degraded | decomposition training data is critical |
| Remove CoT/PoT data | degraded | reasoning data is critical |
| Remove QA data | degraded | retrieval usage capability is critical |

### Key Findings
- **1.5B model matches 10× larger models**: AceSearcher-1.5B outperforms several larger search-augmented LLMs on multi-hop QA.
- **Self-play generates a positive feedback loop**: Better decomposition → better retrieval → better answers → better reward signal.
- **No intermediate annotations needed**: EM reward alone suffices to jointly train decomposition and solving capabilities.
- **Iterative DPO vs. online RL**: Iterative DPO is memory-friendly and achieves comparable performance without maintaining a simultaneous policy + value model.

## Highlights & Insights
- **Collaborative self-play for RAG**: The decomposer and solver mutually improve each other—the decomposer's reward is determined by the solver's performance, and the solver's reward by final-answer correctness. This "mutual bootstrapping" paradigm is transferable to other planning-plus-execution scenarios.
- **Importance of data mixture**: The SFT stage combines retrieval, decomposition, and reasoning data, each of which is indispensable (as confirmed by ablation studies)—this data recipe itself is a valuable reference.
- **No reliance on closed-source models**: Open-source models and public datasets are used, without GPT-4 distillation or intermediate annotations.
- **Practical iterative DPO**: Requires significantly less memory than online RL, making it suitable for resource-constrained environments.

## Limitations & Future Work
- **Fixed retriever**: A fixed E5 / OpenAI Embedding retriever is used; the retriever is not jointly trained.
- **Fixed sub-question count**: Documents are uniformly allocated across sub-questions as $\lfloor N/n \rfloor$, which lacks flexibility.
- **EM reward only**: Not suitable for open-ended or long-form generation tasks.
- **Future directions**: (1) Joint training of the retriever; (2) adaptive document allocation; (3) replacing EM reward with a process reward model.

## Related Work & Insights
- **vs. Search-R1 / R1-Searcher**: These methods use online RL (e.g., GRPO), which is memory-intensive; AceSearcher uses iterative DPO for greater efficiency.
- **vs. IRCOT**: IRCOT is a prompt-based multi-hop method relying on powerful LLMs; AceSearcher trains smaller models to achieve superior results.
- **vs. Self-RAG**: Self-RAG uses reflection tokens to decide when to retrieve; AceSearcher explicitly decomposes questions to guide retrieval.
- **vs. DeepResearcher**: DeepResearcher trains with GRPO; AceSearcher employs iterative DPO with a dual-role design.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of two-role self-play and iterative DPO is novel; the theoretical convergence guarantee is an additional strength.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three task categories, 10 datasets, 4 model scales, extensive baseline comparisons, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, mathematical derivations are complete, and experimental design is well-structured.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and generalizable training framework for search-augmented LLMs; practically usable even at 1.5B scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](../../ICLR2026/llm_nlp/gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[NeurIPS 2025\] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)

</div>

<!-- RELATED:END -->
