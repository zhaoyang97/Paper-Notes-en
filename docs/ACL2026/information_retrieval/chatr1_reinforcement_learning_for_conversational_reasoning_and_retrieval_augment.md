---
title: >-
  [Paper Note] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering
description: >-
  [ACL 2026][Information Retrieval & RAG][Conversational QA] The authors extend "reasoning + search" RL frameworks like Search-R1 / R1-Searcher from single-turn QA to **multi-turn conversational question answering (CQA)**.…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Conversational QA"
  - "Retrieval-Augmented Generation"
  - "Intent Reward"
  - "PPO"
  - "Multi-turn Reasoning"
date: 2026-05-08
content_hash: 85a8562ec3fbbafd
---

# ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering

**Conference**: ACL 2026  
**arXiv**: [2510.13312](https://arxiv.org/abs/2510.13312)  
**Code**: https://github.com/SimonLupart/ChatR1  
**Area**: Reinforcement Learning / Conversational RAG  
**Keywords**: Conversational QA, Retrieval-Augmented Generation, Intent Reward, PPO, Multi-turn Reasoning

## TL;DR
The authors extend "reasoning + search" RL frameworks like Search-R1 / R1-Searcher from single-turn QA to **multi-turn conversational question answering (CQA)**. They propose ChatR1, which uses PPO to jointly optimize reasoning, search, and answering end-to-end. A turn-level dense "intent-aware reward" is introduced, utilizing the token-F1 between the model's self-generated search queries and historical human rewrites. With a 3B backbone, ChatR1 outperforms ChatGPT/Claude across 5 CQA datasets and significantly enhances out-of-distribution (OOD) transfer capability.

## Background & Motivation

**Background**: (1) RL reasoning agents (Search-R1, R1-Searcher, ReSearch, DeepResearcher) have achieved breakthroughs in **single-turn knowledge-intensive QA**, using GRPO/PPO to train LLMs to learn dynamic behaviors such as "when to search, what to search, and how to integrate evidence." (2) Conversational QA (CQA) research currently relies almost entirely on SFT paradigms—UniConv, ChatRetriever, and ChatQA utilize static "query rewrite → retrieve → SFT generate" pipelines. (3) Commercial products (Perplexity, ChatGPT search, Gemini Grounding) are moving toward multi-turn conversational search, but their underlying mechanisms remain black boxes.

**Limitations of Prior Work**: (1) Single-turn RL frameworks assume questions are "self-contained" and cannot handle ellipsis or anaphora (e.g., "one" referring to "European countries" or "wind" referring to "wind energy") common in dialogue history. (2) Conversational skills learned via SFT only mimic demonstration data, leading to poor generalization across domains involving topic shifts, mixed-initiative, multi-document grounding, or faithfulness. (3) The primary hurdle for RL in multi-turn scenarios is the sparsity of outcome-based rewards: a conversation involves reasoning → multiple searches → integration → answering, making turn-level credit assignment extremely difficult.

**Key Challenge**: Utilizing the generalization advantages of RL requires solving the reward sparsity problem. Traditional process rewards either rely on expensive step-level labels or retrieved-passage relevance, which is often incomplete and discrete. The CQA setting provides a unique "free lunch": **CQA datasets contain human-authored query rewrites for each user turn** (used to resolve context for a self-contained query). This serves as a strong proxy for "user intent," yet prior CQA work has only used it as an SFT input rather than an RL supervisory signal.

**Goal**: (1) Construct the first RL-trained CQA framework where the model autonomously schedules reasoning, search, and answering. (2) Design a dense, retrieval-agnostic, and computationally cheap "turn-level intent reward" to mitigate reward sparsity in PPO for multi-turn scenarios. (3) Verify whether RL truly generalizes better than SFT across multiple CQA datasets (evaluated via in-domain, OOD, and LLM-judge metrics).

**Key Insight**: Reusing the human query rewrites $q^{rw}$ provided in datasets as the intent ground truth. By calculating the max F1 between model-generated search queries $q^k$ and $q^{rw}$, a dense intermediate reward is achieved—cheap to compute and decoupled from the performance of the retriever.

**Core Idea**: Explicitly quantify "user intent" based on how closely the model-generated search query approaches the gold rewrite, providing PPO with clear intermediate feedback at each turn.

## Method

### Overall Architecture
ChatR1 employs a policy LLM $\pi_\theta$ (Qwen2.5-3B/7B-Instruct) trained via PPO. In each turn, it receives the conversation history $\mathcal{H}$ and user query $q$, outputting a trajectory $\tau$: $\langle \texttt{<think>...}\rangle \to \texttt{<search>}q^1\texttt{</search>} \to \texttt{<information>}d^1\texttt{</information>} \to ... \to \texttt{<answer>}y\texttt{</answer>}$. The `<search>` token triggers a zero-shot retriever (e5-base-v2, 300M, top-3) with a maximum of 2 search calls. The total reward for the trajectory is $R(\tau) = R_{\text{answer}}(y) + \alpha R_{\text{intent}}(Q)$. PPO with GAE backpropagates the reward along the tokens. The critic and actor are initialized from the same LLM and optimized independently. Training parameters: 500 steps, batch 512, micro-batch 64, $\epsilon=0.2$, $\gamma=\lambda=1$, and $\beta\,D_{\mathrm{KL}}$ regularization. Five datasets (TopiOCQA, QReCC, INSCIT, MultiDoc2Dial, FaithDial) cover challenges like topic shift, large corpora, grounding, and faithfulness.

### Key Designs

1.  **Intent-aware reward—using human rewrites as dense supervision**:
    *   **Function**: Transforms "user intent understanding" into a calculable, dense, and retrieval-agnostic intermediate reward.
    *   **Mechanism**: Since CQA datasets provide human rewrites $q^{rw}$, all model-generated search queries $Q=\{q^1, ..., q^K\}$ in the trajectory are compared against $q^{rw}$ using token-level F1, taking the maximum value: $R_{\text{intent}}(Q) = \max_{q^k \in Q} \mathrm{F1}(q^k, q^{rw})$. The "max" operator allows the model to explore (initial rough search + subsequent refinement) as long as one query hits the intent. Total reward: $R(\tau)=R_{\text{answer}}(y) + \alpha R_{\text{intent}}(Q)$ with $\alpha=0.2$.
    *   **Design Motivation**: (i) Compared to methods like StepSearch that use retrieval hit@k as reward, F1 query rewards are denser (continuous) and not corrupted by retriever errors; (ii) unlike SFT which uses rewrites as inputs, this uses them as reward signals, forcing the model to learn to generate gold-standard queries independently; (iii) rewrites are already available in many datasets, making them cheaper than passage-level relevance labels, which are often sparse and prone to false negatives.

2.  **PPO + GAE trajectory-level reward shaping**:
    *   **Function**: Backpropagates the trajectory reward $R(\tau)$ through GAE to distribute credit to reasoning and search steps.
    *   **Mechanism**: The critic $V_\psi$ learns a value baseline for each token position via GAE with squared error. Advantages are computed as $\hat{A}_i = \delta_i + (\gamma\lambda)\delta_{i+1} + ...$, where $\gamma=\lambda=1$ is equivalent to the REINFORCE form $\hat{A}_i = R(\tau) - V_\psi(\tau_i)$. Retrieved document tokens are excluded from policy loss via loss masking. Hyperparameters: Prompt length 3500, lr=1e-6, clip $\epsilon=0.2$.
    *   **Design Motivation**: Sparse trajectory rewards combined with a critic baseline is standard for long-horizon RL. PPO was chosen over GRPO because the latter often collapsed on multi-turn CQA tasks around 100–200 steps (as shown in the author's training curves).

3.  **RAG-as-tool end-to-end joint optimization**:
    *   **Function**: Treats retrieval behavior as an optimizable policy rather than a fixed pipeline.
    *   **Mechanism**: The search engine is an external tool invoked by `<search>` tokens. The model autonomously decides when, what, and how often to search (capped at 2 calls). Retrieved documents are injected into the next reasoning cycle via `<information>` tokens. The retriever (e5-base-v2, 300M) remains frozen; all retrieval performance gains stem from the actor learning better query formulation. Table 5 shows ChatR1-7B's R@10 on TopiOCQA reaches 46.9, surpassing strong baselines like ConvDR and QuReTeC which require fine-tuning the retriever.
    *   **Design Motivation**: (a) More efficient than using a 7B encoder for retrieval (300M vs 7B); (b) end-to-end actor training allows retrieval and generation to co-adapt; (c) the retriever (BM25, Dense, or Reranker) is plug-and-play during inference.

## Key Experimental Results

### Main Results: Answer quality across 5 CQA datasets

| Method | RAG | LLM | TopiOCQA F1 | QReCC F1 | INSCIT F1 | MD2Dial F1 | FaithDial F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-3.5 (DI) | No | GPT-3.5 | 25.5 | 22.6 | 22.8 | 21.6 | 12.9 |
| Claude (DI) | No | Claude | 27.2 | 25.0 | 27.0 | – | – |
| Qwen-Instr. (RAG) | RAG | Qwen-3B | 8.8 | 15.5 | 13.0 | 18.8 | 12.3 |
| UniConv | RAG | Mistral-7B | 29.6 | 26.2 | 33.2 | – | 11.6 |
| ChatRetriever+Mistral | RAG | Mistral-7B | 28.3 | 26.3 | 30.3 | – | – |
| SFT | No | Qwen-3B | 18.0 | 23.3 | 16.9 | 25.4 | 18.6 |
| QR Search R1 | RAG | Qwen-3B | 20.1 | 20.4 | 27.5 | 23.1 | 14.4 |
| ChatR1 w/o $R_{\text{int.}}$ | RAG | Qwen-3B | 24.4 | 27.0 | 31.3 | 26.4 | 15.5 |
| **ChatR1-3B** | RAG | Qwen-3B | **29.4** | **28.0** | **33.2** | **26.0** | **19.2** |
| **ChatR1-7B** | RAG | Qwen-7B | **30.6** | **31.0** | 32.8 | **31.2** | 18.1 |

ChatR1-3B outperforms GPT-3.5 / Claude DI baselines while using only a 300M retriever. `† / ‡` denote p<0.05 in paired t-tests.

### Ablation Study

| Config | Key Metric | Description |
| :--- | :--- | :--- |
| ChatR1-3B (Full) | 29.4 F1 (TopiOCQA) | Full method |
| w/o $R_{\text{intent}}$ | 24.4 F1 (TopiOCQA) | **-5.0 Gain**; Intent reward is the largest single contributor |
| Intent reward using hit@3 | Lower than F1 reward | Confirms query-level is denser and more robust than passage-level |
| $\alpha=0.0$ vs $\alpha=0.2$ vs $\alpha=1.0$ | $\alpha=0.2$ is optimal | Balance between reward shaping and answer quality |
| Training rewrite source | F1 24.7 to 29.4 | Better rewrites (T5 < Mistral < GPT-4.1) lead to better performance |
| Optimizer: PPO vs GRPO | PPO is more stable | GRPO often collapses in multi-turn settings |
| Retrieval @ inference | 30.6 (7B Dense) | Dense+Reranker is best; 7B is more robust than 3B to BM25 swaps |
| OOD: QReCC → Others | -0.2 F1 on MD2Dial | Strong generalization, consistently better than ChatGPT OOD |

### Key Findings
*   **Intent reward provides gains beyond the RL framework itself**: ChatR1 without intent reward already surpasses pure RL baselines, but the addition of intent reward yields an extra +5 F1 (TopiOCQA), indicating that dense intermediate signals are critical for multi-turn RL.
*   **Query-level F1 reward > Passage-level hit@k reward**: F1 is denser, decoupled from retriever bias, and avoids noise from faulty passage relevance labels.
*   **3B Qwen outperforms GPT-3.5/Claude DI**: RL + RAG allows a 3B model to surpass 175B+ models in CQA, suggesting that conversational tasks are ideal for "small models + tools + RL."
*   **Strong OOD generalization**: ChatR1-3B trained on QReCC lost only 0.2 F1 on MD2Dial and outperformed ChatGPT on 3 out of 4 OOD datasets.
*   **PPO > GRPO in multi-turn scenarios**: GRPO frequently collapses, aligning with Search-R1's findings.
*   **GPT-judge correlates with F1 ($r=0.83$)**: The consistency between metrics validates F1 as a reliable and cheap metric for CQA.
*   **Cross-retriever robustness**: Swapping to BM25 at inference only drops performance by 7.8 F1 (7B), and adding a reranker gains +1.5 F1, showing the learned queries are effective reformulations regardless of the retriever.

## Highlights & Insights
*   **First extension of RL reasoning agents to multi-turn CQA**: This work is a natural extension of the Search-R1 lineage into dialogue, addressing the difficult "reward sparsity" challenge and opening doors for academic research into agents like Perplexity.
*   **Intent-aware reward as an elegant "free lunch"**: By upgrading existing query rewrite labels from inputs to RL supervisory signals at zero additional cost, the study provides a methodology for any task with "gold intermediate products" (e.g., entity linking, tool use).
*   **3B + Small retriever beats 7B baselines**: Demonstrates that the bottleneck in conversational RAG is not parameter count or retriever capacity but end-to-end joint optimization.
*   **PPO's stability for long-horizon CQA**: Provides clear advice on optimizer selection for multi-turn RL.
*   **Query-F1 vs Hit@k as a case study in reward shaping**: Establishes that decoupling intermediate rewards from underlying tools (like retrievers) prevents error propagation.

## Limitations & Future Work
*   Focuses only on PPO/GRPO; off-policy or curriculum-based training could improve sample efficiency.
*   Validated on 10–20 turn dialogues; longer contexts require stronger memory modeling.
*   Does not explore personalization or proactive/mixed-initiative dialogue (e.g., model-initiated clarification).
*   High RL training costs (4×H100) and increased token count during inference compared to direct generation.
*   Requirement for query rewrites: Datasets lacking rewrites require synthetic ones (e.g., via GPT-4.1), which introduces a performance gap if the "silver" rewrite quality is low.
*   Future directions: (1) Simulated users for preference learning; (2) Multi-aspect intent rewards; (3) Curriculum learning from single to multi-turn; (4) Removing search limits to observe deeper multi-hop behaviors.

## Related Work & Insights
*   **vs Search-R1 / R1-Searcher / ReSearch (2025)**: These optimized single-turn search; this work extends the paradigm to multi-turn via intent rewards.
*   **vs StepSearch / SearchR1++ (2025)**: They utilize passage-level hit@k rewards; this work uses denser, retriever-agnostic query-F1 rewards.
*   **vs CALM (2025)**: CALM relies on SFT for multi-turn tool-use, whereas ChatR1 uses RL for superior generalization.
*   **vs UniConv / ChatQA (2024/2025)**: These use 7B retrievers and SFT; ChatR1 outperforms them with a 300M retriever and end-to-end RL.
*   **vs ConvSearch-R1 (2025)**: They focus on optimizing only the query rewriter; ChatR1 jointly optimizes reasoning, retrieval, and answering.
*   **Insights**: (1) Gold intermediate products should be leveraged as dense rewards; (2) PPO is preferred for multi-turn RL; (3) The combination of small models + RL + tools is highly effective for niche vertical applications like deep research and customer service.

## Rating
*   Novelty: ⭐⭐⭐⭐ Extends RL reasoning to multi-turn CQA; intent reward is an ingenious application of classical reward shaping.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 5 datasets, multiple metrics, 14+ baselines, and extensive ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear formulas, high-density figures/tables, and open-sourced code.
*   Value: ⭐⭐⭐⭐⭐ Provides a reproducible RL solution for "conversational search," demonstrating that 3B models can outperform closed-source giants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)

</div>

<!-- RELATED:END -->
