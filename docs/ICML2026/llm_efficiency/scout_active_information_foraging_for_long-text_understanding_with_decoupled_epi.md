---
title: >-
  [Paper Note] Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States
description: >-
  [ICML 2026][LLM Efficiency][Long-text understanding] Scout reframes million-token long-text understanding as an "active information foraging" process, introducing a provenance-anchored…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Long-text understanding"
  - "information foraging"
  - "agent"
  - "epistemic state"
  - "ReAct decoupling"
date: 2026-05-08
content_hash: 3322e2fad156d2e9
---

# Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States

**Conference**: ICML 2026  
**arXiv**: [2605.04496](https://arxiv.org/abs/2605.04496)  
**Code**: Project page available (paper Project Page)  
**Area**: LLM Efficiency / Long-Text Understanding / Agent  
**Keywords**: Long-text understanding, information foraging, agent, epistemic state, ReAct decoupling

## TL;DR
Scout reframes million-token long-text understanding as an "active information foraging" process, introducing a provenance-anchored, trajectory-decoupled epistemic state $\mathcal{E}_t$ as the sole basis for reasoning. Through gap-diagnosed self-evaluation, it iteratively contracts to a query-sufficient subset. On LooGLE-v2 and $\infty$Bench, it matches or surpasses state-of-the-art models like Gemini-3-Pro, while reducing token cost to about $1/8$.

## Background & Motivation

**Background**: Current long-text understanding (LTU) approaches fall into three categories: native long-context LLMs (e.g., Gemini-3-Pro, GPT-5) process the entire document at once; RAG methods segment the document and retrieve relevant chunks; specialized LTU agents (ReadAgent, GraphReader, MemAgent) navigate via graphs, indices, or paginated summaries. While all perform reasonably on NIAH-type tasks, none truly solve complex reasoning at the million-token scale.

**Limitations of Prior Work**: Native long-context models incur prohibitive costs and suffer from attention dilution, with key information lost in vast contexts. RAG methods lose global dependencies after chunking, leading to poor multi-hop aggregation. Agent-based methods often rely on task-agnostic preprocessing (graph/index/gist construction), making lost details unrecoverable. Moreover, ReAct-style "history-as-state" causes the interaction history to become increasingly noisy, polluting the reasoning context.

**Key Challenge**: The authors summarize this as the "LTU Trilemma"—scalability, information fidelity, and reasoning efficiency are hard to achieve simultaneously. The root cause is the "Task-Agnostic Processing Trap": fixed abstraction of the entire document before seeing the query wastes budget on irrelevant regions, while crucial positional cues are lost.

**Goal**: (R1) Always filter and stably retain information consistent with $\mathcal{F}^{\star}_q$ (oracle sufficient set) from noisy observations; (R2) Explicitly track "what is known" and "what is missing" to guide further exploration; (R3) Make "is this sufficient to answer" a decidable state property.

**Key Insight**: Based on an "information sparsity hypothesis": for any query $q$, $|\mathcal{F}^{\star}_q| \ll |\mathcal{F}(\mathcal{D})|$. Thus, LTU is modeled as a POMDP, treating the document as an explorable environment rather than a passive sequence, foraging as needed rather than exhaustive reading.

**Core Idea**: Use a provenance-anchored epistemic state $\mathcal{E}_t$, fully decoupled from interaction history $\mathcal{H}_t$, to hold all knowledge for final reasoning. State-level gap diagnosis on $\mathcal{E}_t$ drives the next round of foraging and commitment, stopping only when $\mathcal{E}_t \approx \mathcal{F}^{\star}_q$.

## Method

### Overall Architecture
Scout operates in a loop. At each step, the agent samples an action $a_t \sim \pi(a \mid q, \mathcal{H}_t, \mathcal{E}_t)$ based on the current query $q$, interaction trajectory $\mathcal{H}_t$, and epistemic state $\mathcal{E}_t$. The action space is split: $\mathcal{A}_{\text{forage}}$ covers physical exploration (Grep / Scan / Read at various granularities, from lexical skimming to dense reading), while $\mathcal{A}_{\text{state}}$ covers state operations (Update commits new units to $\mathcal{E}_t$, Evaluate performs gap diagnosis $g_t$ on $\mathcal{E}_t$). The trajectory $\mathcal{H}_t$ always grows, but $\mathcal{E}_t$ only changes on Update. After termination, "decoupled reasoning" is performed: $y \sim P(y \mid q, \mathcal{E}_T)$, deliberately cutting off access to $\mathcal{H}_T$ to prevent noisy trajectory contamination.

### Key Designs

1. **State Decoupling**:

    - **Function**: Separates the unified history in traditional ReAct (used for both exploration control and reasoning) into two roles: $\mathcal{H}_t$ records "where has been explored," while $\mathcal{E}_t$ records "confirmed knowledge relevant to the query."
    - **Mechanism**: The policy can still read both $(\mathcal{H}_t, \mathcal{E}_t)$ to avoid redundant actions, but the answering stage enforces $P(y \mid q, \mathcal{H}_T) \xrightarrow{\text{Scout}} P(y \mid q, \mathcal{E}_T)$. Thus, cognitive load grows with $|\mathcal{E}_T|$ rather than document length $|\mathcal{D}|$.
    - **Design Motivation**: Information sparsity means most observations during exploration are noise; as long as this noise remains in the reasoning context, LTU is fragile. Cleanly separating "exploration" and "reasoning" in the data path directly addresses (R1).

2. **Provenance-Anchored Epistemic Units**:

    - **Function**: $\mathcal{E}_t = \{e_1, \dots, e_{M_t}\}$, where $e_j = \langle c_j, \alpha_j\rangle$, with $c_j$ an atomic statement distilled from observation, and $\alpha_j$ a unique provenance anchor pointing to a span in $\mathcal{D}$.
    - **Mechanism**: Only when the agent chooses Update is a new unit committed to $\mathcal{E}_t$; pure exploration actions do not alter the state. Any statement used in the final answer can be traced back to the source text via $\alpha_j$, making reasoning auditable.
    - **Design Motivation**: If the decoupled state is free-form text, drift still occurs; anchors enforce that $\mathcal{E}_t$ is "evidence-based," suppressing hallucinations and providing concrete targets for Evaluate, avoiding self-affirmation of imagined content.

3. **Gap-Diagnosed Epistemic Convergence**:

    - **Function**: A state-level operator Evaluate computes the gap $g_{t+1} = \text{DiagnoseGap}(q, \mathcal{E}_t)$ between current $\mathcal{E}_t$ and query $q$, indicating "what type of information is still needed for a reliable answer."
    - **Mechanism**: Progress is stateful—not based on noisy trajectory impressions of "almost done," but by explicitly asking "what does $\mathcal{E}_t$ still lack relative to $\mathcal{F}^{\star}_q$." If $g_t = \emptyset$, stop; otherwise, the next foraging and commitment are guided by this gap, iterating until state-level sufficiency is achieved.
    - **Design Motivation**: Addresses both (R2) and (R3). Monitoring is on "distilled state" rather than "behavioral trajectory," so the stopping criterion is independent of trajectory length, structurally avoiding "increasingly muddled" long-horizon issues.

### Loss & Training
No training is required for inference; existing backbones can be used directly (the paper uses Claude-Sonnet-4.5 as a unified backend for fair comparison). For open-source models, the authors conduct post-training experiments: on Qwen2.5-72B-Instruct, Qwen3-32B, etc., both LLM-only SFT and Scout-paradigm SFT/DAPO are tested. Results show post-training gains are significantly higher under the Scout paradigm, indicating the paradigm itself makes training signals denser and more learnable. Evaluate is implemented by invoking the same backbone with a constrained prompt template and fixed output schema, yielding machine-parsable gap reports.

## Key Experimental Results

### Main Results
Comparison with frontier long-context LLMs and recent agent frameworks under a unified backend protocol, reporting both accuracy (%) and token cost (k).

| Benchmark | Method | Accuracy | Token Cost (k) | Token Eff. |
|-----------|--------|----------|----------------|------------|
| LooGLE-v2 | Gemini-3-Pro | 68.5 | 273.9 | 0.25 |
| LooGLE-v2 | GPT-5.1-chat | 58.6 | 135.9 | 0.43 |
| LooGLE-v2 | MemAgent | 46.0 | 302.2 | 0.15 |
| LooGLE-v2 | **Scout** | **78.7** | **29.7** | **2.63** |
| $\infty$Bench | Gemini-3-Pro | 83.9 | 259.1 | 0.32 |
| $\infty$Bench | GraphReader | 43.1 | 327.4 | 0.13 |
| $\infty$Bench | **Scout** | **85.6** | **21.4** | **4.01** |

Scout achieves the highest accuracy on both benchmarks while reducing token cost to about $1/8$ of competitors, with Token Eff. an order of magnitude higher than the best alternative.

### Ablation Study
Component ablation on LooGLE-v2.

| Configuration | Acc (%) | $\Delta$ | Cost (k) |
|---------------|---------|----------|----------|
| Scout (Full) | 78.2 | – | 29.7 |
| w/o $\mathcal{E}_t$ (degrades to ReAct) | 70.5 | $-7.7$ | 28.3 |
| w/o $\mathcal{A}_{\text{forage}}$ (cannot access $\mathcal{D}$) | 17.7 | $-60.5$ | 27.4 |
| w/o File Tools | 74.2 | $-4.0$ | 31.4 |
| w/o Grounding (remove $\alpha$ anchors) | 75.5 | $-2.7$ | 29.7 |

### Key Findings
- Removing the epistemic state reverts to ReAct, dropping accuracy by 7.7 points with nearly unchanged token cost—demonstrating that performance gains stem from a "cleaner reasoning base," not "more tokens."
- As context increases from 64K to 1M+, Scout's accuracy remains nearly flat and token cost nearly unchanged; native long-context models decline after 256K, with token cost rising by orders of magnitude. This empirically validates the "information sparsity hypothesis": what grows is not the information needed to answer, but noise.
- Removing anchors $\alpha$ only drops accuracy by 2.7 points, seemingly mild, but the authors note the gap is much larger on multi-hop aggregation tasks, indicating anchors mainly suppress hallucination drift in long-range reasoning.

## Highlights & Insights
- The "information sparsity hypothesis" is elevated to a system design principle. Most LTU work assumes sparsity but only uses it to justify "RAG works"; Scout enforces it as a strong constraint—if sparsity holds, make the reasoning context equally sparse, pressuring the agent to retain only "necessary and traceable" content.
- The "history-as-state" paradigm of ReAct is highly successful for short-horizon tool use, but Scout reveals its fundamental flaw for long horizons: the signal-to-noise ratio monotonically decreases with $|\mathcal{H}_t|$. Separating "exploration" and "reflection/answering" in the data path is a broadly transferable recommendation for all long-horizon agents.
- Gap diagnosis upgrades self-evaluation from behavioral ("what should I do next") to state-level ("what categories of facts am I missing"), making "when to stop" a decidable problem. This abstraction is valuable for deep research, agentic RAG, and related scenarios.

## Limitations & Future Work
- The paper acknowledges that "efficiency" mainly refers to token cost, not wall-clock time—multi-round interaction may incur network and scheduling overheads that, in some deployments, outweigh native models. Appendix E.4 provides latency analysis but this remains a limitation.
- Evaluate treats diagnosis as a constrained prompt call, with quality dependent on the backbone's metacognitive ability; weaker backbones may produce overly optimistic gap reports, leading to premature termination.
- The current action space is manually divided into Grep/Scan/Read + Update/Evaluate; for structured documents (PDF tables, code ASTs), more specialized foraging primitives may be needed.
- Anchors $\alpha_j$ assume stable span identifiers in the source; for real-time streams, dynamic web pages, or streaming audio with "unstable positions," the approach is not directly applicable.

## Related Work & Insights
- **vs ReAct**: Retains ReAct's "think–act–observe" loop for exploration, but forcibly cuts off access to $\mathcal{H}_T$ during answering, fundamentally avoiding the "long history becomes noisy" issue.
- **vs MemAgent / ReSum**: These use lossy compression to shorten context, potentially losing positional cues; Scout does not compress $\mathcal{H}_t$, instead maintaining a clean $\mathcal{E}_t$ for answering.
- **vs GraphReader / ReadAgent**: These build indices/gists before the query arrives, essentially "task-agnostic preprocessing." Scout insists on "query-driven foraging," avoiding loss of critical details during preprocessing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates the information sparsity hypothesis to a strong constraint and designs a decoupled paradigm with epistemic state + gap diagnosis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual benchmarks (LooGLE-v2 / $\infty$Bench) + context scaling + post-training, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual hierarchy, POMDP formalization and R1-R3 requirements thoroughly motivate the work
- Value: ⭐⭐⭐⭐⭐ Achieves SOTA at million-token scale while reducing cost by an order of magnitude, providing paradigm guidance for long-text agents

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](../../ICLR2026/llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)

</div>

<!-- RELATED:END -->
