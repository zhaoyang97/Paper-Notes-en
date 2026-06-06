---
title: >-
  [Paper Note] ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs
description: >-
  [ACL 2026][LLM Safety][Jailbreak attacks] ASTRA treats every jailbreak attempt as a learning opportunity. It distills strategies into a three-layer vector library (Effective / Promising / Ineffective) based on continuous…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Jailbreak attacks"
  - "Strategy library"
  - "Closed-loop learning"
  - "RAG retrieval"
  - "Black-box red teaming"
date: 2026-05-08
content_hash: fcfbcf7660fc567c
---

# ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs

**Conference**: ACL 2026  
**arXiv**: [2511.02356](https://arxiv.org/abs/2511.02356)  
**Code**: None  
**Area**: AI Safety / Automated Jailbreaking  
**Keywords**: Jailbreak attacks, Strategy library, Closed-loop learning, RAG retrieval, Black-box red teaming

## TL;DR
ASTRA treats every jailbreak attempt as a learning opportunity. It distills strategies into a three-layer vector library (Effective / Promising / Ineffective) based on continuous scores (1-10). Subsequent attacks reuse experience via similarity retrieval, achieving an 80.6% attack success rate (ASR) across 8 mainstream LLMs with an average of only 2.4 queries.

## Background & Motivation

**Background**: Existing black-box jailbreak methods are divided into two categories: template/heuristic methods (GPTFuzzer / CodeAttack / CipherChat), which are easily fingerprinted, and iterative optimization methods (PAIR, TAP), which utilize feedback but remain "binary" (success or failure), failing to extract reusable experience from failed attempts.

**Limitations of Prior Work**: (1) Templates lack diversity and are easily identified by defenders; (2) Optimization methods treat results as booleans, preventing learning; (3) Transferability across datasets and target models is nearly zero.

**Key Challenge**: Jailbreaking is essentially a sparse reward search problem—most attempts are "partial successes" or "failures," where binary signals discard 99% of the learning information.

**Goal**: Construct a self-evolving framework capable of learning from every interaction (including failures) and reusing that knowledge across different queries and models.

**Key Insight**: Introduce the concept of "reward shaping" from reinforcement learning by changing binary judgments to continuous scores (1-10). Combined with strategy-level retrieval and update mechanisms, this transforms individual attack experiences into a "library."

**Core Idea**: A closed-loop "attack–evaluate–distill–reuse" cycle plus a three-layer vectorized strategy library. The attacker acts like a student who records notes in a "mistake book" or "notebook" after every exercise to check for future reference.

## Method

### Overall Architecture
ASTRA consists of three major modules: **Attack Designer** (generates prompts via Strategy-Agnostic or Strategy-Guided modes), **Strategy Extractor** (reflects on interactions to distill structured strategy JSON based on scores), and **Strategy Storage & Retrieval** (indexes original harmful queries using text embeddings and performs top-$k$ cosine similarity retrieval). The attack budget is set to $N=10$, retrieval $k=9$ (top-3 per category), using GLM-4.5 as the Attacker/Extractor and GPT-4o as the Judge.

### Key Designs

1.  **Closed-loop "attack–evaluate–distill–reuse"**:
    - **Function**: Converts every (prompt, response, score) into a retrievable structured strategy, forming a cycle of "battle $\rightarrow$ upgrade $\rightarrow$ re-equip."
    - **Mechanism**: The objective starts from $\sigma^* = \arg\max_\sigma J_\theta(q, T_\theta(A_\theta(q, S(\sigma))))$, where the Judge $J_\theta$ outputs $s \in [1,10]$ instead of 0/1. The lifecycle involves retrieving strategies $\rightarrow$ generating prompts $\rightarrow$ receiving target LLM responses $\rightarrow$ scoring by Judge $\rightarrow$ distilling strategies via Extractor $\rightarrow$ writing to the library; the loop breaks if $s_i=10$.
    - **Design Motivation**: Continuous scoring turns "near-misses" (e.g., $s=8$ where the direction is correct but filtered at the final step) into valuable learning signals, which binary rewards in PAIR/TAP cannot capture.

2.  **Three-layer dynamic strategy library (Effective / Promising / Ineffective)**:
    - **Function**: Allows the attack to benefit from three types of knowledge: mimicking success, exploring potential, and avoiding pitfalls.
    - **Mechanism**: Routes strategies based on Judge scores—$s=10$ to **Effective** (direct reuse), $5 < s < 10$ to **Promising** (includes improvement suggestions), and $s \le 5$ to **Ineffective** (recorded as avoidance guidelines). Each strategy is JSON: core description + usage guidelines + illustrative example. During retrieval, the Attacker receives strategies from all three categories to "learn from the good, improve the mediocre, and avoid the bad."
    - **Design Motivation**: Compared to a single "success pool," the triple-layer library addresses exploit, explore, and pruning respectively. Ablation shows that removing the Ineffective library drops ASR from 71.9% to 65.4%, proving "failure is also knowledge."

3.  **Semantic retrieval based on original harmful queries**:
    - **Function**: Indexes strategies by query to enable strategy transfer across different harmful requests.
    - **Mechanism**: Uses `text-embedding-3-small` to encode the original query $q$ into $\mathbf{v}_q$ as an index, storing $(\mathbf{v}_q, \sigma)$ pairs. New queries $q_\text{new}$ target top-$k$ hits via cosine similarity. While higher $k$ increases ASR, gains diminish; $k=9$ is chosen.
    - **Design Motivation**: Using "query semantics" rather than prompt text as the key allows effective strategies to be shared across distinct topics like "teachers / students / chemical weapons." Validation shows the mature library transfers to AdvBench-50 with almost no performance drop.

### Loss & Training
No model training. ASTRA is an inference-only framework where all "learning" occurs during the strategy library's write/read operations. Attacker temperature = 1.0; Judge prompts follow PAIR settings; success is defined only as $s=10$.

## Key Experimental Results

### Main Results (HarmBench 400 behaviors, 8 target models)

| Method | Llama-3-8B | Llama-3-70B | DeepSeek-R1 | GPT-4o | GPT-4.1 | Gemini-2.0 | Gemini-2.5 | Claude-3.7 | **Avg ASR** |
|--------|------|------|------|------|------|------|------|------|------|
| PAIR | 17.8 | 22.5 | 45.3 | 38.8 | 33.0 | 53.3 | 30.5 | 4.0 | 30.7 |
| TAP | 22.2 | 25.3 | 49.0 | 41.0 | 36.0 | 55.0 | 37.5 | 10.8 | 34.6 |
| GPTFuzzer | 28.0 | 11.3 | 62.0 | 16.0 | 3.0 | 78.3 | 3.5 | 2.3 | 25.6 |
| ReNeLLM | 68.0 | 64.5 | 77.5 | 71.5 | 70.1 | 62.3 | 44.8 | 19.0 | 59.7 |
| CodeAttack | 46.0 | 64.3 | 87.5 | 70.5 | 65.0 | 76.0 | 44.3 | 26.3 | 60.0 |
| **ASTRA** | 54.5 | **89.3** | **95.5** | **93.8** | **91.0** | **98.5** | **86.0** | **36.0** | **80.6** |

Average Queries (AQ, lower is better): PAIR 5.5 / TAP 6.3 / ReNeLLM 2.7 / **ASTRA 2.4**. On newer models like GPT-5.1 / Gemini-3-Flash / Qwen3-max, ASTRA still achieves 92.0 / 90.5 / 96.3 ASR.

### Ablation Study

| Variant | Llama-3-8B | GPT-4o | Gemini-2.5 | Claude-3.7 | **Avg ASR** | **Avg AQ** |
|------|------|------|------|------|------|------|
| ASTRA (Full) | 54.5 | 93.8 | 86.0 | 36.0 | **71.9** | 2.9 |
| w/o Effective Lib | 41.0 | 79.0 | 72.3 | 23.0 | 58.6 | 3.6 |
| w/o Ineffective Lib | 46.8 | 86.3 | 81.0 | 33.0 | 65.4 | 3.1 |
| w/o Promising Lib | 44.0 | 87.0 | 77.8 | 35.0 | 65.3 | 3.3 |
| w/o Retrieval | 38.5 | 60.8 | 60.0 | 17.5 | **48.5** | 3.9 |
| w/o Score (Binary) | 46.3 | 86.5 | 72.5 | 31.0 | 62.1 | 3.5 |
| w/o Extractor (Logs) | 45.0 | 82.5 | 77.5 | 25.8 | 61.7 | 3.9 |

### Key Findings
- **Retrieval is the critical module**: Removing it causes ASR to plummet to 48.5% (-23.4 points), indicating that ASTRA's success depends on finding appropriate strategies rather than raw single-prompt creativity.
- **Continuous scoring adds significant value**: Switching to binary scoring drops ASR from 71.9% to 62.1%, confirming the reward shaping hypothesis.
- **Failure is knowledge**: Removing the Ineffective Library results in a 6.5-point ASR drop, validating that "telling the Attacker what not to do" effectively prunes the search space.
- **Cross-model strategy transfer**: Libraries developed by GLM-4.5 on HarmBench transfer to Qwen3-32B or GPT-4o with minimal loss (92.5% ASR on GPT-4o), showing the distilled output represents human-readable attack patterns rather than model-specific quirks.
- **Cross-dataset transfer**: A mature library frozen on HarmBench performs consistently on AdvBench-50 across most targets.
- **Defense robustness**: Paraphrase and Perplexity Filters hardly impact results; ASR remains at 86.3% against Llama Guard 3, proving attack prompts are natural language rather than suspicious suffixes.
- **Token cost is "high but worth it"**: ASTRA uses 10.7k tokens per round (vs PAIR's 1.9k), but the amortized cost per success is similar (31.9k vs 34.7k). Fewer target interactions also mean better stealth and fewer rate-limit triggers.

## Highlights & Insights
- **Defining red teaming as a reward shaping problem in RL** is the paper's conceptual backbone, clearly identifying sparse rewards as the fundamental problem in PAIR/TAP. This framework is generalizeable to any LLM-agent self-improvement task.
- **Triple-layer library (Success + Hope + Failure)**: Unlike traditional experience replay which only stores success cases, this paper explicitly models "behavior to avoid." This symmetrical memory design could transfer to self-evolving coding or search agents.
- **Original query as retrieval key**: By indexing via intent rather than the prompt itself, the system elegantly allows multiple ways of asking the same dangerous question to share the strategy library—a simple but effective engineering trick.

## Limitations & Future Work
- **High multi-agent inference overhead**: While amortized cost is similar to PAIR, the absolute token usage per step is significantly higher, which may be unacceptable for some projects.
- **Reliance on Judge GPT-4o**: While cross-validation with Claude-Sonnet-4 / Llama-Guard-3 yielded 92.5%/94.0% confirmed ASR, all training signals still originate from a single judge, potentially introducing bias.
- **Cold start requirements**: A target requires sufficient exploration initially to build the library; the paper lacks analysis on how many attempts $(N)$ are needed for library convergence.
- **Ethical risk**: High ASR, low query counts, and transferable libraries create a highly threatening red-teaming tool. The authors commit to releasing code "on request."

## Related Work & Insights
- **vs PAIR / TAP**: Both iterate but treat results as booleans without knowledge accumulation; ASTRA's reward shaping and libraries result in 30+ points higher ASR on average.
- **vs GPTFuzzer**: Pure random mutation fails (near 0%) on Claude-3.7, whereas ASTRA maintains 36% through strategy retrieval.
- **vs CodeAttack / ReNeLLM**: Single-template attacks lack diversity; they drop to 44% against strong defenses like Gemini-2.5, while ASTRA retains 86% via dynamic strategy retrieval.
- Closest to **AutoDAN-Turbo** (lifelong learning), but ASTRA's explicit three-category distinction and RAG-based retrieval offer a more engineered approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Reward shaping + triple-layer library is a first for jailbreaking, though individual components (RAG / lifelong agents) are known.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 models, 6 baselines, 4 defenses, cross-dataset/attacker tests, and token costs.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and thorough appendices/case studies, though some sections are wordy.
- Value: ⭐⭐⭐⭐ Highlights the vulnerability of current alignment to "self-evolving attacks," providing a strong impetus for defense research, despite the ethical risks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification](sere_structural_example_retrieval_for_enhancing_llms_in_event_causality_identifi.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)

</div>

<!-- RELATED:END -->
