---
title: >-
  [Paper Note] Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States
description: >-
  [ICML 2026][LLM Efficiency][Long-Text Understanding] Scout remodels million-token long-text understanding (LTU) as an "active information foraging" process. It introduces an epistemic state $\mathcal{E}_t$…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Long-Text Understanding"
  - "Information Foraging"
  - "Agent"
  - "Epistemic State"
  - "ReAct Decoupling"
date: 2026-05-08
content_hash: d8aa15b785dc4814
---

# Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States

**Conference**: ICML 2026  
**arXiv**: [2605.04496](https://arxiv.org/abs/2605.04496)  
**Code**: Project Page Public (Paper Project Page)  
**Area**: LLM Efficiency / Long-Text Understanding / Agent  
**Keywords**: Long-Text Understanding, Information Foraging, Agent, Epistemic State, ReAct Decoupling

## TL;DR
Scout remodels million-token long-text understanding (LTU) as an "active information foraging" process. It introduces an epistemic state $\mathcal{E}_t$, decoupled from the interaction trajectory and anchored with provenance points, as the sole reasoning foundation. Through gap-diagnosed self-evaluation, it iteratively converges to a query-sufficient subset. It matches or exceeds frontier models like Gemini-3-Pro on LooGLE-v2 and $\infty$Bench while reducing token costs to approximately $1/8$.

## Background & Motivation

**Background**: Current LTU methods generally fall into three categories: native long-context LLMs (Gemini-3-Pro, GPT-5, etc.) that consume the entire document at once; RAG-based methods that chunk documents and retrieve fragments; and specialized LTU agents (ReadAgent, GraphReader, MemAgent, etc.) that navigate via graphs, indices, or paged summaries. While performant on NIAH-style tasks, none fully solve complex reasoning at the million-token scale.

**Limitations of Prior Work**: Native long-context models suffer from cost explosions and attention dilution, where key information is buried in contexts spanning over 10 orders of magnitude. RAG methods lose global dependencies due to chunking, leading to poor performance on multi-hop aggregation tasks. Agent-based methods often rely on task-agnostic preprocessing (building graphs, indices, or gist compression); once details are lost during preprocessing, they cannot be recovered. Furthermore, the ReAct-style "history-as-state" approach causes the interaction history to become longer and nosier, turning the reasoning foundation into a source of noise.

**Key Challenge**: The authors summarize this as the "LTU Trilemma"—the inability to simultaneously satisfy scalability, information fidelity, and reasoning efficiency. The root cause is the "Task-Agnostic Processing Trap": performing fixed abstractions on the entire document before seeing the query, which wastes the budget on query-irrelevant regions while losing essential location-anchoring cues.

**Goal**: (R1) Consistently filter information from noisy observations that is consistent with $\mathcal{F}^{\star}_q$ (the oracle sufficient set) and retain it in a stable form; (R2) Explicitly track "what is known" vs. "what is missing" to guide subsequent exploration; (R3) Make the determination of "sufficiency for answering" a decidable state property.

**Key Insight**: Based on an "information sparsity hypothesis": for any query $q$, $|\mathcal{F}^{\star}_q| \ll |\mathcal{F}(\mathcal{D})|$. Therefore, LTU is modeled as a POMDP where the document is treated as an explorable environment rather than a passive sequence, foraging "on-demand" rather than exhaustively.

**Core Idea**: Use an epistemic state $\mathcal{E}_t$, anchored with provenance and completely decoupled from the interaction history $\mathcal{H}_t$, to carry all knowledge used for final reasoning. Drive the next round of foraging and commitment through state-level gap diagnosis until $\mathcal{E}_t \approx \mathcal{F}^{\star}_q$.

## Method

### Overall Architecture
Scout operates in a loop. At each step, the agent samples an action $a_t \sim \pi(a \mid q, \mathcal{H}_t, \mathcal{E}_t)$ based on the current query $q$, interaction trajectory $\mathcal{H}_t$, and epistemic state $\mathcal{E}_t$. The action space is divided into two categories: $\mathcal{A}_{\text{forage}}$ for physical exploration (Grep / Scan / Read at three granularities, from lexical skipping to dense verbatim reading) and $\mathcal{A}_{\text{state}}$ for state operations (Update commits new units into $\mathcal{E}_t$; Evaluate performs gap diagnosis $g_t$ on $\mathcal{E}_t$). While the trajectory $\mathcal{H}_t$ grows continuously, $\mathcal{E}_t$ only changes during Update. After termination, "decoupled reasoning" is performed: $y \sim P(y \mid q, \mathcal{E}_T)$, deliberately cutting off access to $\mathcal{H}_T$ to prevent noisy trajectories from polluting the answer.

### Key Designs

1.  **State Decoupling**:
    - **Function**: Splits the unified history—which in traditional ReAct serves as both exploration control and reasoning foundation—into two roles: $\mathcal{H}_t$ records "where has been explored," and $\mathcal{E}_t$ records "confirmed knowledge relevant to the query."
    - **Mechanism**: The policy can still read both $(\mathcal{H}_t, \mathcal{E}_t)$ to avoid redundant actions, but the answering stage is forced through $P(y \mid q, \mathcal{H}_T) \xrightarrow{\text{Scout}} P(y \mid q, \mathcal{E}_T)$. Consequently, the cognitive load scales with $|\mathcal{E}_T|$ rather than the original document length $|\mathcal{D}|$.
    - **Design Motivation**: Information sparsity implies most observations during exploration are noise. As long as this noise remains in the reasoning context, LTU will remain fragile. Separating "exploration" and "reasoning" in the data path is the cleanest way to address (R1).

2.  **Provenance-Anchored Epistemic Units**:
    - **Function**: $\mathcal{E}_t = \{e_1, \dots, e_{M_t}\}$, where $e_j = \langle c_j, \alpha_j\rangle$. $c_j$ is an atomic statement distilled from observations, and $\alpha_j$ is a provenance anchor pointing to a unique span in the source $\mathcal{D}$.
    - **Mechanism**: New units are committed to $\mathcal{E}_t$ only when the agent selects Update; pure exploration does not change the state. Any statement used in the final answer can be traced back to the source text via $\alpha_j$, making the reasoning auditable.
    - **Design Motivation**: If the decoupled state were free-form text, it would still suffer from drift. Anchors force $\mathcal{E}_t$ to be "grounded in evidence," suppressing hallucinations and providing specific objects for self-evaluation (Evaluate) to avoid verifying "imagined content."

3.  **Gap-Diagnosed Epistemic Convergence**:
    - **Function**: Uses a state-level operator Evaluate to compute the gap $g_{t+1} = \text{DiagnoseGap}(q, \mathcal{E}_t)$ between current $\mathcal{E}_t$ and query $q$, specifying "what category of information is still needed for a reliable answer."
    - **Mechanism**: State-ifies the progress problem—instead of relying on the noisy trajectory to "look like it's almost done," it explicitly asks "what is $\mathcal{E}_t$ missing relative to $\mathcal{F}^{\star}_q$." If $g_t = \emptyset$, it stops; otherwise, the next round of foraging is guided by this gap until a state-level determination of "sufficiency" is reached.
    - **Design Motivation**: Simultaneously addresses (R2) and (R3). By monitoring "distilled state" rather than "behavioral trajectory," termination criteria become independent of trajectory length, preventing performance degradation over long horizons.

### Loss & Training
No training is required for the inference phase; off-the-shelf backbones can be used (the paper uses Claude-Sonnet-4.5 as a unified backend for fair comparison). For open-source models, the authors conducted post-training experiments: attempting LLM-only SFT vs. SFT/DAPO under the Scout paradigm on Qwen2.5-72B-Instruct and Qwen3-32B. Results show that gains from post-training under the Scout paradigm are significantly larger than plain LLM SFT, indicating the paradigm makes training signals denser and more learnable. Evaluate is implemented by calling the same backbone with a constrained prompt template and fixed output schema to produce machine-parseable gap reports.

## Key Experimental Results

### Main Results
Comparison with frontier long-context LLMs and recent agent frameworks under a unified backend protocol, measuring Accuracy (%) and Token Cost (k).

| Benchmark | Method | Accuracy | Token Cost (k) | Token Eff. |
|-----------|------|--------|---------------|------------|
| LooGLE-v2 | Gemini-3-Pro | 68.5 | 273.9 | 0.25 |
| LooGLE-v2 | GPT-5.1-chat | 58.6 | 135.9 | 0.43 |
| LooGLE-v2 | MemAgent | 46.0 | 302.2 | 0.15 |
| LooGLE-v2 | **Scout** | **78.7** | **29.7** | **2.63** |
| $\infty$Bench | Gemini-3-Pro | 83.9 | 259.1 | 0.32 |
| $\infty$Bench | GraphReader | 43.1 | 327.4 | 0.13 |
| $\infty$Bench | **Scout** | **85.6** | **21.4** | **4.01** |

Scout achieves the highest accuracy on both benchmarks while compressing token costs to approximately $1/8$ of competitors, with Token Eff. an order of magnitude higher than the best rival.

### Ablation Study
Breakdown of core components on LooGLE-v2.

| Configuration | Acc (%) | $\Delta$ | Cost (k) |
|------|---------|----------|----------|
| Scout (Full) | 78.2 | – | 29.7 |
| w/o $\mathcal{E}_t$ (Degenerates to ReAct) | 70.5 | $-7.7$ | 28.3 |
| w/o $\mathcal{A}_{\text{forage}}$ (Cannot access $\mathcal{D}$) | 17.7 | $-60.5$ | 27.4 |
| w/o File Tools | 74.2 | $-4.0$ | 31.4 |
| w/o Grounding (Remove $\alpha$ anchors) | 75.5 | $-2.7$ | 29.7 |

### Key Findings
- Removing the epistemic state (returning to ReAct) drops accuracy by 7.7 points with almost no change in token cost—proving that performance gains come from a "cleaner reasoning base" rather than "more tokens."
- As the context expands from 64K to 1M+, Scout’s accuracy and token cost remain nearly flat, whereas native long-context models decline significantly after 256K while costs grow exponentially. This confirms that the "information sparsity hypothesis" holds in practice: the growth is in noise, not the information required to answer.
- Removing anchor $\alpha$ causes only a 2.7 point drop overall, but the authors note larger gaps in multi-hop tasks, suggesting anchors primarily suppress hallucination drift during long-range reasoning.

## Highlights & Insights
- Successfully transforms the "information sparsity hypothesis" into a system design principle. While most LTU works assume sparsity to justify RAG, Scout elevates it to a strong constraint—since it's sparse, the reasoning context is kept proportionally sparse, forcing the agent to retain only "necessary and traceable" content.
- While the "history-as-state" ReAct paradigm is successful in short-horizon tool usage, Scout reveals its fundamental pathology in long horizons: the signal-to-noise ratio decreases monotonically with $|\mathcal{H}_t|$. Separating "exploration" from "reflection/answering" is a transferable lesson for all long-horizon agents.
- Gap diagnosis upgrades self-evaluation from the behavioral level ("what should I do next") to the state level ("what category of facts am I missing"), turning termination into a decidable problem—an abstraction valuable for deep research and agentic RAG scenarios.

## Limitations & Future Work
- The "efficiency" primarily refers to token costs, not wall-clock time—the network and scheduling overhead of multi-turn interactions might exceed native models in hardware-specific deployments (Appendix E.4 provides latency analysis).
- Evaluate treats diagnosis as a constrained prompt call; quality depends on the backbone's metacognitive ability. Weak backbones might yield overly optimistic gap reports, leading to premature termination.
- The action space is currently manually partitioned (Grep/Scan/Read + Update/Evaluate); specialized forage primitives might be needed for structured documents like PDF tables or code ASTs.
- Anchors $\alpha_j$ assume stable span identifiers in the source, which may not directly apply to "position-unstable" inputs like real-time streams, dynamic webpages, or streaming audio.

## Related Work & Insights
- **vs ReAct**: Retains the "Think-Act-Observe" loop for exploration but strictly cuts off access to $\mathcal{H}_T$ during answering, fundamentally avoiding degradation due to "dirty" long histories.
- **vs MemAgent / ReSum**: These use lossy compression to shorten context, but compression can lose location-anchoring cues. Scout does not compress $\mathcal{H}_t$ but maintains a separate, clean $\mathcal{E}_t$ for answering.
- **vs GraphReader / ReadAgent**: These methods build indices/gists before the query arrives (task-agnostic preprocessing). Scout insists on "foraging by query" to avoid losing critical details during preprocessing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates the information sparsity hypothesis to a strong constraint and designs a decoupled paradigm with epistemic states + gap diagnosis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full coverage with LooGLE-v2 / $\infty$Bench, context scaling analysis, and post-training experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual hierarchy; the POMDP formalization and R1-R3 requirements explain the motivation thoroughly.
- Value: ⭐⭐⭐⭐⭐ Achieves SOTA in million-token scenarios while reducing costs by an order of magnitude; provides paradigm guidance for long-text agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](../../ICLR2026/llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)

</div>

<!-- RELATED:END -->
