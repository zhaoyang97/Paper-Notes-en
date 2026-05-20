---
title: >-
  [Paper Note] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory
description: >-
  [ICML 2026][LLM Agent / Long-Context Memory / Multi-Agent Systems][Episodic Memory] E-mem shifts the traditional memory paradigm of "preprocessing and compressing into embeddings/graphs" to an episodic reconstruction par…
tags:
  - "ICML 2026"
  - "LLM Agent / Long-Context Memory / Multi-Agent Systems"
  - "Episodic Memory"
  - "Context Reconstruction"
  - "Master-Assistant Architecture"
  - "SLM Assistant"
  - "LoCoMo"
date: 2026-05-08
content_hash: 54caf84d7cc8e312
---

# E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory

**Conference**: ICML 2026  
**arXiv**: [2601.21714](https://arxiv.org/abs/2601.21714)  
**Code**: https://github.com/dog-last/E-mem  
**Area**: LLM Agent / Long-Context Memory / Multi-Agent Systems  
**Keywords**: Episodic Memory, Context Reconstruction, Master-Assistant Architecture, SLM Assistant, LoCoMo

## TL;DR
E-mem shifts the traditional memory paradigm of "preprocessing and compressing into embeddings/graphs" to an episodic reconstruction paradigm of "retaining original context + in-situ reasoning by small model assistants": the master agent only performs global planning, while multiple SLM assistants each guard an uncompressed segment of the original text. Upon multi-pathway retrieval and activation, they conduct local reasoning and return evidence. On LoCoMo, E-mem surpasses SOTA by 7.75 F1 points while reducing token consumption by 70%.

## Background & Motivation

**Background**: Long-range LLM agents typically preprocess and store historical conversations—common approaches include chunking and embedding (RAG), entity extraction and graph construction (GraphRAG / GAM), or OS-style paging (MemGPT). At query time, top-k segments are retrieved and concatenated into the prompt.

**Limitations of Prior Work**: The authors term these approaches "destructive de-contextualization"—compressing tightly coupled event sequences into geometric points or graph nodes severs sequential dependencies. On multi-session, long-dialogue benchmarks like LoCoMo, multi-hop and temporal questions are hard to answer because causal chains across chunks are lost. Methods like A-Mem, Mem0, and MemoryOS (2024-25) plateau at F1 scores of 36-45 on LoCoMo.

**Key Challenge**: (1) Deep System 2 reasoning requires preserving long-range causal chains; (2) Directly feeding the entire history into the context window triggers "Lost-in-the-Middle" and explodes token costs; (3) Preprocessing is cheap but inherently destroys the context integrity needed for (1). These form a trade-off.

**Goal**: Enable agents to truly "re-experience" past segments, not just retrieve fragments, without sacrificing cost scalability.

**Key Insight**: Drawing an analogy to biological engrams—human recall is not index lookup but reactivation of entire episodic contexts followed by reasoning. Assigning a small model to guard a fixed segment of the original text and perform in-situ local reasoning upon activation preserves sequential dependencies while controlling costs.

**Core Idea**: Replace the "unified compression storage + retrieval" paradigm with a hierarchical architecture of "master agent planning + multiple SLM assistants guarding original text + multi-pathway on-demand activation + in-situ assistant reasoning returning evidence."

## Method

### Overall Architecture
E-mem is formalized as a triplet $\mathcal{F}=\langle\mathcal{A}_{\text{master}},\{\mathcal{A}_{\text{asst}}^{(i)}\}_{i=1}^{N},\mathcal{R}\rangle$, corresponding to three roles:

- **Master agent** (GPT-4o-mini / Qwen2.5-14B): Global planner, does not directly hold the original history, only synthesizes evidence returned by assistants.
- **Assistant agents** (a group of Qwen3-4B SLMs): Each guards a fixed window of the original token sequence $\mathcal{E}_i$, and maintains a concise summary $s_i$ for routing features.
- **Multi-pathway router** $\mathcal{R}$: Given a query, outputs an activation distribution $\mathcal{P}_{act}=\pi(q|\mathbf{S},\mathcal{R})\in[0,1]^N$, determining which assistants to activate.

Overall process: Original streaming input → sliding window split into N segments, each assigned to an assistant (one SLM per segment) → upon query, router performs three-way parallel retrieval → union of activated assistants → each activated assistant conducts local reasoning on its segment, producing "timestamped evidence" → master agent aggregates evidence, resolves conflicts, and outputs the final answer.

### Key Designs

1. **Preserving Complete Episodic Context + Sliding Window Segmentation**:

    - Function: The unbounded stream $\mathcal{X}=(x_1,x_2,\dots)$ is split by window length $L$ and stride $S<L$ into $\mathcal{E}_i=\{x_t|(i-1)S<t\leq(i-1)S+L\}$, with adjacent segments overlapping by $\delta=L-S$ as a "continuity buffer."
    - Mechanism: Each $\mathcal{E}_i$ is **raw, uncompressed tokens**, held by an independent SLM assistant; assistants are dormant by default and only "go online" for reasoning when activated by routing. New tokens are appended directly to the active assistant's $\mathcal{E}_{\text{active}}$; once full, it is solidified as a memory unit, and a new assistant is spawned with the overlap region from the previous segment as seed ($\mathcal{E}_{N+1}^{init}=\text{Extract}(\mathcal{E}_N,\text{overlap}=\delta)$), ensuring semantic continuity across segments.
    - Design Motivation: Traditional methods embed chunks due to the computational infeasibility of joint reasoning over many chunks. By making SLMs the chunk owners, storage remains as original text, and reasoning is distributed across many small models, enabling $O(1)$ streaming updates.

2. **Multi-Pathway Collaborative Activation (Global + Vector + Symbolic)**:

    - Function: Three orthogonal retrieval pathways compute the activation set $\mathcal{A}^*$ in parallel, then take the union $\mathcal{A}^*=\{\mathcal{A}_{\text{asst}}^{(i)}|\mathcal{A}_{\text{asst}}^{(i)}\in\mathcal{P}_{\text{global}}\vee\mathcal{P}_{\text{vec}}\vee\mathcal{P}_{\text{kw}}\}$.
    - Mechanism: (i) **Global Alignment** $\mathcal{P}_{\text{global}}$: Dense vector + sparse lexical alignment between query and summary $s_i$, akin to high-pass filtering, capturing macro narrative intent; (ii) **Semantic Association** $\mathcal{P}_{\text{vec}}$: High-dimensional vector similarity between query and original chunk embedding, serving as a fallback when summaries miss details; (iii) **Symbolic Trigger** $\mathcal{P}_{\text{kw}}$: BM25 for entity/ID exact matching, ensuring key names and numbers are not missed due to summarization.
    - Design Motivation: Single-path routing (pure vector or pure graph) inevitably misses aspects on mixed benchmarks like LoCoMo ("multi-hop + temporal + entity recall"). The cost of three-way union is only a few extra lightweight retrievals at the router stage, while missing a key chunk can cause downstream reasoning to fail—thus, recall is prioritized over precision.

3. **Assistant Local Reasoning + Master Temporal Anchoring and Aggregation**:

    - Function: Each activated assistant does not "return the original segment," but directly outputs a **timestamped evidence tuple** $e_i=\langle c_i,\tau_i\rangle=\Phi_{\text{asst}}(q|\mathcal{E}_i)$, where $c_i$ is the inferred semantic evidence and $\tau_i$ is the absolute timestamp of the event.
    - Mechanism: Assistants perform full chain-of-thought local reasoning on their segment, converting raw text into structured evidence; the master aggregates all $\{e_i\}$ via $R=\Psi_{\text{master}}(q,\mathbf{E})$, crucially **sorting by $\tau$ to resolve state conflicts** (e.g., for object location changes, the most recent timestamp prevails). For hard cases, an iterative mode is supported: the master maintains a reasoning trace $S^{(t)}$, and if evidence is insufficient, issues new sub-queries $q^{(t)}=\pi_{\text{plan}}(q_{\text{init}},S^{(t-1)})$ to assistants.
    - Design Motivation: Having SLMs reason over "small segment + full original text" is more controllable and token-efficient than having a large model reason over the concatenated full text. **With timestamps, the master can resolve cross-segment temporal conflicts**—key for multi-hop/temporal sub-tasks.

### Loss & Training
The framework is **purely inference-time**, with no model training. The master uses GPT-4o-mini / Qwen2.5-14B, assistants use Qwen3-0.6B/1.7B/4B/8B/14B, all frozen and prompted directly. The design space mainly involves routing hyperparameters (top-k, three-way weights), window size $L$, and overlap $\delta$.

## Key Experimental Results

### Main Results

LoCoMo 5 sub-tasks (GPT-4o-mini as master, Qwen3-4B as assistant):

| Method | Overall F1 | Multi-Hop | Temporal | Single-Hop |
|--------|------------|-----------|----------|------------|
| RAG (top-20) | 44.73 | 27.50 | 46.07 | 52.45 |
| A-Mem | 39.65 | 27.02 | 45.85 | 44.65 |
| Mem0 | 45.10 | 38.72 | 48.93 | 47.65 |
| MemoryOS | 42.84 | 35.27 | 41.15 | 48.62 |
| GAM (Prev. SOTA) | 45.31 | 34.84 | 53.91 | 47.74 |
| **E-mem** | **54.17** | **42.64** | **59.82** | **59.23** |
| Gain over GAM | **+8.86** | **+7.80** | **+5.91** | **+11.49** |

HotpotQA streaming extension (400/800/1600 docs, over 200K tokens):

| Method | 400 F1 | 800 F1 | 1600 F1 |
|--------|--------|--------|---------|
| Long-Context | 56.56 | 49.71 | 53.92 |
| GAM | 54.75 | 52.86 | 53.71 |
| **E-mem** | **61.46** | **55.46** | **55.76** |

### Ablation Study

Assistant model size ablation (LoCoMo adversarial subset, adversarial questions to test hallucination):

| Assistant Model | F1 | BLEU-1 |
|-----------------|-----|--------|
| Qwen3-0.6B | 85.11 | 80.14 |
| Qwen3-1.7B | 87.31 | 83.17 |
| Qwen3-4B (default) | 89.94 | 85.51 |
| Qwen3-8B | 95.74 | 88.09 |
| Qwen3-14B | 95.03 | 88.06 |

Master model replacement (assistant fixed at Qwen3-4B): Gemini2.5-flash 93.62, GPT-4o 89.37, GPT-4o-mini 89.94, DeepSeek-V3 75.87, Grok4-fast 77.80.

### Key Findings
- **Multi-hop and Temporal sub-tasks benefit most**: +7.80 / +5.91 F1, directly supporting the core claim—preserving original context + timestamp anchoring is key to chaining causal links across segments.
- **Open Domain is outperformed by Mem0** (24.89 vs 28.64): Open-ended generation relies more on global summarization, so E-mem's "original text + local reasoning" is less advantageous for divergent QA.
- **Larger assistants yield diminishing returns**: F1 increases by 4.83 from 0.6B to 4B, another 5.80 from 4B to 8B, but slightly drops from 8B to 14B, indicating 4B is the sweet spot for cost-effectiveness.
- **Token cost reduced by 70%+**: By activating only the relevant k assistants instead of feeding the entire history to the master, per-query token usage is much lower than RAG/GraphRAG.

## Highlights & Insights
- **"Small models guard original text + large model plans" is an elegant division of labor**: Traditional memory systems either burden the large model or the compression module; E-mem distributes reasoning across many SLMs, making "long-context reasoning" a truly horizontally scalable multi-agent problem for the first time.
- **Timestamped evidence tuples are a hidden key design**: Many papers tout "multi-pathway retrieval," but what truly solves LoCoMo Temporal tasks is $e_i=\langle c_i,\tau_i\rangle$—without $\tau$, the master cannot resolve conflicts.
- **Union of multi-pathway retrievals, not weighted fusion**: The authors prefer activating more assistants over missing key chunks due to weighting, reflecting a "recall > precision" philosophy, valuable for all multi-hop QA systems.
- **Three-way retrieval is transferable to any "original text must be preserved" scenario**: Legal, medical, codebase agents requiring traceability to original text can benefit from this architecture.

## Limitations & Future Work
- **Underperforms Mem0 on Open Domain sub-tasks**: Divergent generation needs global summarization rather than local evidence; E-mem's "local in-situ reasoning" is not optimal for such tasks.
- **Memory cost of N SLMs not fully discussed**: For large $N$ (e.g., thousands of chunks), even 4B per assistant means hundreds of GBs of VRAM; the paper only reports LoCoMo/HotpotQA-scale N.
- **Routing hyperparameters (top-k, thresholds) require manual tuning**: Optimal k varies by task, lacking adaptive mechanisms.
- **Depends on master agent's temporal reasoning**: When the master is replaced with DeepSeek-V3 / Grok4-fast, F1 drops to 75-77, indicating conflict resolution heavily relies on the master's temporal reasoning, not just the architecture.
- **Future directions**: (i) Add learnable sparse gating to the router; (ii) Add "sleep-state compression" for assistants (compress to KV cache when inactive to save VRAM); (iii) Reinforcement learning for iterative mode, enabling the master to learn when to stop.

## Related Work & Insights
- **vs MemGPT / RAG**: MemGPT uses OS-style paging to stitch chunks, requiring reassembly to restore dependencies at each activation; E-mem equips each chunk with a "reasoning brain," directly producing structured evidence and eliminating the need for reassembly.
- **vs A-Mem / Mem0**: A-Mem uses Zettelkasten-style self-evolving notes, Mem0 does personalized compression—both remain in the "compress-store-retrieve" paradigm; E-mem eliminates the "compression" step, trading assistant count for context integrity.
- **vs GAM (Prev. SOTA)**: GAM uses graph + multi-agent deep research for semantic structuring on the storage side; E-mem defers "semantic structuring" to query time via assistants, avoiding mismatch between pre-stored graphs and future queries.
- **vs Agentic RAG**: Agentic RAG adds a planning loop to retrieval, but storage remains vector-based; E-mem makes storage itself multi-agent, a more thorough shift.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Using SLMs as chunk owners" is a simple yet profound paradigm shift, decoupling storage and reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ LoCoMo + HotpotQA benchmarks, 5 SOTA baselines, master/assistant model size ablations, but lacks VRAM/real latency data.
- Writing Quality: ⭐⭐⭐⭐ Clear exposition, the biological engram analogy convincingly motivates the need for original text.
- Value: ⭐⭐⭐⭐⭐ Provides a truly horizontally scalable new paradigm for long-horizon agents, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](../../NeurIPS2025/llm_agent/a-mem_agentic_memory_for_llm_agents.md)
- [\[ICLR 2026\] REMem: Reasoning with Episodic Memory in Language Agents](../../ICLR2026/llm_agent/remem_reasoning_with_episodic_memory_in_language_agent.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](../../ACL2026/llm_agent/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)

</div>

<!-- RELATED:END -->
