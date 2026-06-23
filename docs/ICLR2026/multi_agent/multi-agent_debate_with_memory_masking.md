---
title: >-
  [Paper Note] Multi-Agent Debate with Memory Masking (MAD-M²)
description: >-
  [ICLR 2026][Multi-Agent][LLM Reasoning] This paper points out that Multi-Agent Debate (MAD) can be misled by "false memories" remaining from previous rounds. It theoretically proves that MAD performance is constrained by memory quality and proposes MAD-M², which applies an "evaluate-mask" filter to the previous round's memory before each debate round, ensuri
tags:
  - ICLR 2026
  - Multi-Agent
  - LLM Reasoning
date: 2026-05-08
content_hash: 179adfdb434ea892
---
# Multi-Agent Debate with Memory Masking (MAD-M²)

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=EdTt8nMAMA](https://openreview.net/forum?id=EdTt8nMAMA)  
**Code**: [https://github.com/tmlr-group/MAD-MM](https://github.com/tmlr-group/MAD-MM)  
**Area**: Multi-Agent / LLM Reasoning  
**Keywords**: Multi-agent debate, memory masking, test-time scaling, LLM reasoning, robustness  

## TL;DR
This paper points out that Multi-Agent Debate (MAD) can be misled by "false memories" remaining from previous rounds. It theoretically proves that MAD performance is constrained by memory quality and proposes MAD-M², which applies an "evaluate-mask" filter to the previous round's memory before each debate round, ensuring agents reason only based on reliable memories.

## Background & Motivation
**Background**: Scaling test-time sampling is a dominant approach for improving LLM reasoning. Multi-Agent Debate (MAD, Du et al. 2023), where multiple LLMs act as agents to correct and iteratively refine answers through shared memory across rounds, is considered a powerful reasoning paradigm. Intuitively, "seeing others' reasoning" helps agents escape self-bias and identify fallacious content.

**Limitations of Prior Work**: The core mechanism of MAD is the "unconditional reading of all memories from the previous round." However, these memories themselves may be incorrect. Using real cases from the MATH dataset (Fig. 1), the paper illustrates an awkward phenomenon: Agent 1, who initially answered correctly in the first round, was misled into a wrong answer in the second round after referencing the incorrect memory of Agent 2. In other words, **false memories contaminate otherwise correct reasoning**, a risk previously under-discussed.

**Key Challenge**: MAD simultaneously benefits from and is harmed by "shared memory." Since memories contain a mix of high and low-quality information, full acceptance is equivalent to inserting low-quality examples into the context as demonstrations, which interferes with the LLM. This partly explains why MAD underperforms compared to simpler CoT-SC in many scenarios.

**Goal**: To characterize the vulnerability of MAD to false memories and design a simple, universal, and training-free mechanism to retain useful memories while pruning incorrect ones, thereby enhancing the robustness and accuracy of MAD.

**Core Idea (False memories as low-quality demonstrations)**: Each round of reasoning is viewed as in-context learning, where false memories are equivalent to low-quality examples that distract attention. Therefore, **inserting an "evaluate-mask" operation between debate rounds to purify context before reasoning** provides more stable performance gains than simply increasing the number of samples or agents.

## Method

### Overall Architecture
MAD-M² inserts a "memory purification" step between the two rounds of traditional MAD. In the initial round, agents answer independently to form memory vectors. Before each subsequent round starts, all memories from the previous round are critically evaluated to generate a binary mask that zeros out suspected incorrect memories. Agents then proceed with reasoning based only on the retained memories. This process iterates until the final round, where a majority vote produces the final answer.

```mermaid
flowchart LR
    Q[Query x] --> R1[Initial Round: Na agents answer independently → Memory M_r]
    R1 --> E[Evaluate & Mask: Generate binary vector M∈{0,1} to mask false memory]
    E --> R2[Next Round: agents re-reason based on retained memory M̂]
    R2 -->|Iterate| E
    R2 --> V[Final Round: Majority Vote → Final Answer]
```

### Key Designs
**1. Binary Masking Memory Filtering: Replacing "Reading All" with "Reading Retained."** This is the backbone of MAD-M². Given the memory set $M_r=[A_{\theta_1}(x,M_{r-1}),\dots,A_{\theta_{N_a}}(x,M_{r-1})]$ from round $r$, the method no longer forces the next round to unconditionally consume the entire $M_r$. Instead, a binary mask is generated for each agent and multiplied element-wise: $\hat{M}^{(i)}_r = M^{(i)}\odot M_r$, where $M^{(i)}=g^{\text{map}}_{A_{\theta_i}}(M_r)\in\{0,1\}^{N_a}$ is the mask vector mapped from the agent's evaluation of each memory. Memory positions judged as incorrect are set to 0 and disappear from the context. Since all agents are initialized from the same model, the mask evaluation only needs to be performed once and shared, making it computationally lightweight.

**2. Subjective Masking: Letting agents vote to exclude false memories.** Specifically, the LLM is asked to label each memory as "YES / NO / NOT SURE." Depending on the strictness of the filtering rules, "NOT SURE" is categorized as YES (Loose rule L) or NO (Strict rule S). This strategy relies purely on the semantic judgment of the agent rather than internal model states, at the cost of one additional self-evaluation step and extra tokens. Experiments show this is more effective for relatively weaker models whose semantic judgments are more reliable than their confidence signals.

**3. Objective Masking: Using perplexity as a confidence signal to filter memory.** Inspired by Fu et al. (2025), the method uses the LLM's own perplexity (PPL) as an objective criterion. High perplexity typically suggests the model is not confident in the generated content, making it more likely to contain errors or hallucinations. Thus, only the answer with the lowest perplexity is retained. This approach requires no extra self-evaluation dialogues and actually **saves tokens** compared to naive MAD (often ×0.6~0.7 overhead in experiments). It proves more effective for stronger models (e.g., QwQ-32B, Qwen2.5-Math) where perplexity better reflects answer quality.

**4. Controllable Token Overhead with an Upper Bound.** Multi-round interaction naturally leads to token inflation. The paper provides a quantitative analysis: The subjective strategy, due to the self-evaluation step, consumes at most $N_a\sum_{r=2}^{N_{round}}\sum_{i=1}^{N_a}T^o_{r-1,i}$ more input tokens than naive MAD, i.e., $N^{token}_{\text{MAD-M}^2}\le N_aN_{round}T^q+2N_a\sum_{r,i}T^o_{r-1,i}+\sum_{r,i}T^o_{r,i}$. The objective strategy is usually more efficient due to memory pruning.

> **Theoretical Support**: Under Assumption 2.1 (the probability of an agent answering correctly based on memory is $e^{-\alpha N_e}$, where $N_e$ is the number of false memories), the paper proves that the success probability bound for 2-round MAD (Prop. 2.3) explicitly depends on the number of false memories. In both hard ($p<\tfrac12$) and easy ($p\ge\tfrac12$) cases, **reducing false memories $N_e$ consistently improves performance**, whereas simply increasing the number of agents $N_a$ can exponentially degrade performance on hard problems. This provides the theoretical basis for "masking false memory" and explains why MAD often fails to beat CoT-SC (which can be viewed as an ideal upper bound for MAD).

## Key Experimental Results
Setting: 3 agents, 2 debate rounds; CoT / CoT-SC (6 paths) / MAD as baselines; evaluated on Qwen2.5-7B-Instruct, Qwen2.5-Math-7B-Instruct, DeepSeek-Math-7B, and QwQ-32B. Datasets include mathematical reasoning (GSM8K, MATH, AIME24/25) and language understanding (MMLU-Pro). AIME represents hard problems; others are relatively easy. Results are averaged over 5 seeds. T. denotes token overhead relative to MAD.

### Main Results (Selected, Acc.% / T.)

| Model | Method | GSM8K | MATH | MMLU-Pro | AIME24 | AIME25 |
|------|------|-------|------|----------|--------|--------|
| Qwen2.5-7B | MAD | 91.8 / ×1.00 | 55.6 | 43.0 | 13.3 | 6.7 |
| Qwen2.5-7B | MAD-M²(S) | 89.0 / ×1.25 | 56.8 | **43.6** | 13.3 | 3.3 |
| Qwen2.5-Math-7B | MAD | 95.2 / ×1.00 | 71.2 | 34.2 | 6.7 | 6.7 |
| Qwen2.5-Math-7B | MAD-M²(O) | **95.4 / ×0.60** | **80.2** | **37.0** | **13.3** | **13.3** |
| QwQ-32B | MAD | 97.2 / ×1.00 | 79.2 | 75.4 | 76.7 | 73.3 |
| QwQ-32B | MAD-M²(O) | 96.6 / ×0.56 | 75.0 | 75.2 | **80.0** | **76.7** |

Key takeaway: On stronger models like Math/QwQ, the objective strategy MAD-M²(O) outperforms MAD in most tasks while reducing token overhead to ~0.6x. On the weaker Qwen2.5-7B model, the subjective strategy MAD-M²(S) is more advantageous. Notably, MAD itself often **underperforms CoT-SC** on Math models, confirming the theoretical analysis.

### Ablation Study

| Analysis | Setting | Conclusion |
|------|------|------|
| False Memory Identification (Fig.3) | Strict S vs. Loose L | Objective masking is more accurate for strong models; subjective is better for weak models. |
| Scaling Agents (Fig.4) | $N_a$ from 3 to 10 | Both MAD and MAD-M² gain from more agents; MAD-M²(S) leads in most cases. |
| Token Consumption | Subjective vs. Objective | Subjectivity requires extra tokens (×1.2~1.4); objectivity saves tokens (×0.6~0.7) by pruning. |

### Key Findings
- False memory is a real bottleneck for MAD: Both theory and real cases show that residual false memory can mislead correct agents. Simply increasing the number of agents can be counterproductive for hard problems.
- There is no "universal" masking strategy: Weak models rely on semantic self-evaluation (subjective), while strong models rely on perplexity (objective). Strategies must be selected based on model capability.
- The objective strategy offers excellent cost-performance: On stronger math models, it improves accuracy while saving tokens, making it a practical default choice.

## Highlights & Insights
- **Theory-First**: Quantifies the relationship between "memory quality and reasoning success" using $e^{-\alpha N_e}$. It provides success probability bounds for CoT-SC / MAD, proving that "reducing false memory" is superior to "increasing sampling," and explains experimental findings where MAD loses to CoT-SC.
- **Minimalist & Plug-and-Play**: No training or model modification required. It only adds a masking step between rounds, making it compatible with any open-source LLM.
- **Memory Quality as a First-Class Citizen**: Operationalizes the analogy of "debate = in-context learning" and "false memory = low-quality example" into a controllable filtering mechanism.

## Limitations & Future Work
- **Inconsistent Gains**: On certain easy problems (e.g., GSM8K) and weak models, masking slightly reduces accuracy, suggesting that filtering might mistakenly remove useful memories.
- **Manual Strategy Selection**: Choosing between subjective and objective strategies depends highly on model capability and rule strictness, with no automatic selection mechanism currently available.
- **Limited Scale**: Primarily verified with 3 agents / 2 rounds on 7B~32B models. Performance across more rounds, larger scales, and more diverse tasks remains to be explored.
- **Strong Theoretical Assumptions**: Assumptions like $e^{-\alpha N_e}$ and homogeneous agents are simplified models; their alignment with real LLM behavior requires further validation.

## Related Work & Insights
- **MAD Taxonomy**: Built on the foundation of multi-agent debate by Du et al. (2023). Unlike S-MAD (Li et al. 2024), which uses static/random graphs for sparse communication, or S²-MAD (Zeng et al. 2025), which reduces trivial exchanges, this work **dynamically decides memory selection based on agent evaluation or internal states**.
- **Test-Time Scaling**: Aligns with CoT-SC and test-time scaling but emphasizes that "improving quality is more important than increasing quantity."
- **Inspiration**: Perplexity as an unsupervised quality signal and context purification as an independent step can be transferred to scenarios like RAG denoising, agent memory management, and long-context compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic characterization of MAD's vulnerability to false memories with theoretical bounds; the "evaluate-mask" mechanism is elegant and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers 4 models and 5 datasets with theoretical, token, and scalability analyses, though gains are inconsistent and the scale is relatively small (3 agents/2 rounds).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear chain of Motivation—Theory—Method—Experiments; intuitive visualization of real-world cases.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play; the objective strategy improves accuracy while saving tokens, offering direct reference value for MAD practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MAD-Logic: Multi-Agent Debate Enhances Symbolic Translation and Reasoning](mad-logic_multi-agent_debate_enhances_symbolic_translation_and_reasoning.md)
- [\[ICLR 2026\] GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs](graphplanner_graph_memory-augmented_agentic_routing_for_multi-agent_llms.md)
- [\[ICLR 2026\] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)
- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](../../ACL2026/multi_agent/latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](../../ICML2026/multi_agent/e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)

</div>

<!-- RELATED:END -->
