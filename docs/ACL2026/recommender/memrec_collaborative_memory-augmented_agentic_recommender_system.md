---
title: >-
  [Paper Note] MemRec: Collaborative Memory-Augmented Agentic Recommender System
description: >-
  [ACL 2026][Recommender Systems][Collaborative Memory] MemRec employs a **lightweight LLM specifically to manage a dynamic "Collaborative Memory Graph"** (connecting semantic memories of multiple users and items via inter…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Collaborative Memory"
  - "Agentic RS"
  - "Memory Graph"
  - "Decoupled Architecture"
  - "Label Propagation"
date: 2026-05-08
content_hash: 1e1485f13daab4e3
---

# MemRec: Collaborative Memory-Augmented Agentic Recommender System

**Conference**: ACL 2026  
**arXiv**: [2601.08816](https://arxiv.org/abs/2601.08816)  
**Code**: https://github.com/rutgerswiselab/memrec (Available)  
**Area**: Recommender Systems / LLM Agent / Collaborative Filtering  
**Keywords**: Collaborative Memory, Agentic RS, Memory Graph, Decoupled Architecture, Label Propagation

## TL;DR
MemRec employs a **lightweight LLM specifically to manage a dynamic "Collaborative Memory Graph"** (connecting semantic memories of multiple users and items via interaction edges), and then feeds distilled "collaborative memory facets" to a heavyweight reasoning LLM for the final recommendation. Through a "Curate-then-Synthesize" de-noising strategy and asynchronous $O(1)$ label propagation updates, it achieves a **+15% to +29%** H@1 improvement over the SOTA i2Agent across four benchmarks. For data-sparse users, it shows a **+91.4%** improvement relative to a Vanilla LLM.

## Background & Motivation
**Background**: The "memory form" of recommender systems has undergone three waves of evolution: (1) sparse rating memory in the Matrix Factorization era; (2) dense embedding memory in the Deep Learning era; and (3) "semantic memory" in the LLM Agentic RS era—where user preferences and item descriptions are written as natural language text for LLM reasoning. Recent research further classifies semantic memory into three levels: "Memory-less → Static Memory → Dynamic Self-reflective Memory" (e.g., i2Agent, AgentCF, RecBot that self-reflect to update user/item profiles).

**Limitations of Prior Work**: All existing agentic RS rely on **"Island Memory"**—where the $M_u$ of each user and the $M_i$ of each item are maintained independently. When making a recommendation for user $u$, the system only considers $M_u$, completely losing the core signals of the Collaborative Filtering era: **peer signals from similar users** and **transfer signals from co-occurring items**. This leading to extremely poor performance for sparse users and cold-start items; while the GNN/LightGCN era relied on user-item graphs to succeed, the LLM agent era has regressed to purely personal memory.

**Key Challenge**: Directly concatenating "all neighbor memories into the prompt" seems like a solution, but it immediately hits two walls: (1) **Cognitive Overload**—stuffing massive amounts of text into the LLM context causes the model to be overwhelmed by noise (the "Lost in the Middle" phenomenon), which actually decreases ranking quality; (2) **Update Bottleneck**—every new interaction requires cascading updates to the semantic memories of all neighbors. A naive implementation would require $O(|N_k|)$ LLM calls, which is cost-prohibitive for industrial deployment.

**Goal**: Re-inject "collaborative signals" into the agentic memory system while bypassing cognitive overload and update bottlenecks.

**Key Insight**: Inspired by Information Bottleneck theory, since raw neighbor information is excessive, one should **distill a "compressed-but-task-relevant" sub-representation**. Furthermore, drawing from the Label Propagation algorithm, "reflective updates to neighbors" can be packaged into asynchronous batches.

**Core Idea**: **Architectural Decoupling**—letting a lightweight $\text{LM}_{\text{Mem}}$ maintain the collaborative memory graph and perform curate-then-synthesize distillation in the background, while the foreground heavyweight $\text{LLM}_{\text{Rec}}$ only observes the distilled high-concentration signals for reasoning. This resolves overload and batches all updates into a single asynchronous LLM call ($O(1)$ per interaction).

## Method

### Overall Architecture
MemRec maintains a unified memory graph $G = (\mathcal{V}, E)$, where each node $\mathcal{V} = \mathcal{U} \cup \mathcal{I}$ stores an evolving semantic memory $M_v$, and edges $E$ encode interactions and derivative relationships. The pipeline consists of three stages:

1. **Collaborative Memory Retrieval (Stage-R)**: $\text{LM}_{\text{Mem}}$ uses LLM-generated domain rules to prune the neighbors $N(u)$ of $u$ to top-$k=16$, and then synthesizes a compact collaborative memory $M_{\text{collab}}$ (a set of structured facets).
2. **Grounded Reasoning (Stage-ReRank)**: $\text{LLM}_{\text{Rec}}$ receives $(\mathcal{I}_u, M_{\text{collab}}, C_{\text{info}})$ to assign scores $s_i$ to candidates and generate rationales $r_i$.
3. **Async Collaborative Propagation (Stage-W)**: After an interaction occurs, $\text{LM}_{\text{Mem}}$ asynchronously and batch-updates $M_u^t, M_{i_c}^t$ and the neighbor increments $\{\Delta M_{\text{neigh}}\}$ in a single LLM call.

### Key Designs

1. **Architectural Decoupling: $\text{LM}_{\text{Mem}}$ vs $\text{LLM}_{\text{Rec}}$ (Core Innovation)**:
    - **Function**: Physically separates "memory management" from "recommendation reasoning," allowing each function to use a model of appropriate scale and enabling decoupling of update frequencies and trigger conditions.
    - **Mechanism**: $\text{LM}_{\text{Mem}}$ can be gpt-4o-mini or even local models like Qwen-2.5-7B / Llama-3-8B, specialized in ingesting raw graph context and performing curation, synthesis, and propagation. $\text{LLM}_{\text{Rec}}$ is a heavyweight model (gpt-4o-mini or gpt-4o) that only reads distilled high-signal $M_{\text{collab}}$ for final ranking and rationales. Communication occurs via the narrow $M_{\text{collab}}$ channel, following the $T = \arg\max I(T; Y) - \beta I(T; X)$ logic of the Information Bottleneck.
    - **Design Motivation**: Experiments in Figure 6 show that a "Naive Agent" (single model for both ingestion and ranking) quickly hits a performance plateau because LLMs fail to balance "compression-reasoning" within long raw contexts, creating an internal cognitive bottleneck. Decoupling assigns each function its role, effectively separating System 1 (fast filtering) and System 2 (deep reasoning), yielding an absolute H@1 improvement of **+34%** on the Books dataset compared to the single-model approach.

2. **LLM-Guided Context Curation + Synthesis (Curate-then-Synthesize)**:
    - **Function**: Compresses the raw semantics of dozens of neighbors into a set of structured facets under a restricted token budget ($\tau = 1800$), adaptive to different domains.
    - **Mechanism**: First, $\text{LM}_{\text{Mem}}$ is used offline to observe domain statistics $\mathcal{D}_{\text{domain}}$ and automatically generate interpretable heuristic rules $R_{\text{domain}} \leftarrow \text{LM}_{\text{Mem}}(\mathcal{D}_{\text{domain}} \| P_{\text{meta}})$ (e.g., "prioritize by genre/theme similarity" for Books, and "prioritize by cuisine + price + recent visits" for Yelp). Online, these rules act as high-speed filters to select top-$k$ neighbors $N_k'(u)$ in milliseconds. In the synthesis stage, the target user's **full $M_u^{t-1}$** and the neighbors' **lightweight tiered representations** (truncated memory for item neighbors, and the last 3 interaction titles as a dense proxy for user neighbors) are fed to $\text{LM}_{\text{Mem}}$, which outputs $N_f = 7$ structured facets (each with a theme, confidence level, and neighbor evidence).
    - **Design Motivation**: Traditional graph pruning is either rule-based (like random-walk, which lacks semantics) or GNN-based (trained and uninterpretable), neither of which suits LLM agents. Using "LLM-as-Rule-Generator" maintains the speed of rules while gaining LLM semantic understanding as a zero-shot middle ground. Tiered representation is a clever engineering choice—using recent interactions for users and truncated memories for items prevents redundancy while compressing tokens.

3. **Asynchronous Collaborative Propagation ($O(1)$ Update Bottleneck Elimination)**:
    - **Function**: Maintains the dynamic evolution of the memory graph with interactions while triggering only 1 $\text{LM}_{\text{Mem}}$ call per interaction (instead of $|N_k'|$ calls).
    - **Mechanism**: When $u$ and $i_c$ interact at time $t$, a unified prompt $P_{\text{update}}$ is constructed for $\text{LM}_{\text{Mem}}$ to simultaneously produce $(M_u^t, M_{i_c}^t, \{\Delta M_{\text{neigh}}\})$—updating the full memory of both parties and outputting "incremental update segments" $\Delta M$ for each neighbor. This process is asynchronous (not blocking the online ranking path), inspired by Label Propagation: treating the interaction as a "new label" propagating to neighbors along similarity relationships.
    - **Design Motivation**: Naive synchronous schemes run an LLM call for every neighbor, repeatedly stuffing the user context into prompts (huge token redundancy). The batched and asynchronous approach achieves $O(1)$ call complexity and significantly compresses total tokens. More importantly, this "batched incremental" modeling ensures collaborative signals actually flow to neighbors rather than just the interacting parties.

### Loss & Training
**Entirely training-free**. All LLM calls use zero-shot prompting with hyperparameters $k=16$ neighbors, $N_f=7$ facets, $\tau=1800$ token budget, and temperature=0. $\text{LLM}_{\text{Rec}}$ and $\text{LM}_{\text{Mem}}$ default to gpt-4o-mini; the "Ceiling" configuration uses gpt-4o. The "Local" configuration uses vLLM to deploy Qwen-2.5-7B/Llama-3-8B. The "Vector" configuration replaces $\text{LLM}_{\text{Rec}}$ with an all-MiniLM-L6-v2 Sentence Transformer for direct similarity ranking. This "pluggable architecture" allows MemRec to run across various deployments from cloud APIs to on-premise setups.

## Key Experimental Results

### Main Results

**H@1 and N@5 across 4 benchmarks (Amazon Books / Goodreads / MovieTV / Yelp, N=10 candidates)**:

| Dataset | Method | H@1 | N@5 | H@1 Gain |
|--------|------|-----|-----|---------|
| **Books** | i2Agent (Prev. SOTA) | 0.4453 | 0.6138 | — |
| | LightGCN | 0.1753 | 0.3592 | — |
| | **Ours (MemRec)** | **0.5117** | **0.6601** | **+14.91%** |
| **Goodreads** | i2Agent | 0.3099 | 0.5481 | — |
| | **Ours (MemRec)** | **0.3997** | **0.6112** | **+28.98%** |
| **MovieTV** | i2Agent | 0.4912 | 0.6672 | — |
| | **Ours (MemRec)** | **0.5882** | **0.7422** | **+19.75%** |
| **Yelp** | i2Agent | 0.4205 | 0.6007 | — |
| | **Ours (MemRec)** | **0.4868** | **0.6463** | **+15.77%** |

All improvements are statistically significant at $p < 0.05$. The most sparse datasets (Books / Goodreads) show the largest gains, validating the value of collaborative signals for sparse users.

### Ablation Study (Books Dataset)

| Configuration | H@1 | H@5 | N@5 | H@1 Drop |
|------|-----|-----|-----|----------|
| MemRec (Full) | 0.527 | 0.803 | 0.670 | — |
| w/o Collab. Write (Disable async propagation) | 0.505 | 0.814 | 0.665 | **−4.2%** |
| w/o LLM Curation (Generic vs. domain rules) | 0.498 | 0.788 | 0.648 | **−5.5%** |
| **w/o Collab. Read (Disable collab retrieval)** | **0.475** | 0.769 | 0.624 | **−9.9%** |

### Key Findings
- **Collab Read > Collab Write > LLM Curation > Solo Memory**: The H@1 drop order clearly indicates that "introducing neighbor information into the reasoning path" provides the highest gain; dynamic propagation follows; and curation accuracy/adaptivity also contributes significantly.
- **Data-sparse users benefit most**: For the low-activity user subgroup, MemRec shows a **+91.4%** H@1 gain over Vanilla LLM, proving neighbor signals are exactly what isolated agents lack.
- **Robust under 30% noise injection**: Even with 30% malicious items injected into user history, MemRec maintains H@1=0.491, thanks to LLM curation acting as a "semantic filter" to remove irrelevant peers.
- **Significant expansion of the Pareto frontier**: Standard (4o-mini) config yields H@1=0.524 / N@5=0.663 / ~16.5s latency; Cloud-OSS (gpt-oss-120B) yields H@1=0.561 / N@5=0.699; Ceiling (gpt-4o) yields H@1=0.580 / N@5=0.722. The Vector config can reduce latency to sub-milliseconds.
- **Token I/O Ratio of 3.9:1**: MemRec's token distribution is naturally heavy on input and light on output (input ~5,100 / output ~1,300), perfectly exploiting commercial LLM pricing where outputs are 3-4x more expensive than inputs.
- **Rationale quality improves**: GPT-4o-as-judge evaluations show significant improvements in specificity and relevance ($p<0.001$), with minor gains in factuality, proving collaborative memory enhances both ranking and explanation quality.

## Highlights & Insights
- **"Dual-model decoupling + IB channel" is a universal architecture**: The idea of physically separating memory management from reasoning can be applied to any scenario where agents handle overloaded contexts (e.g., long-form QA, code repo agents, long video agents). The combination of a lightweight steward and a heavyweight reasoner is highly attractive for industrial deployment.
- **LLM-as-Rule-Generator**: Using an LLM once to generate interpretable rules for millisecond-level online filtering preserves the LLM's semantic understanding while avoiding online call costs. This can be generalized to the paradigm of "using LLMs to distill rules for other systems."
- **$O(1)$ complexity of async batched propagation**: This makes "collaborative updates" industrially viable for the first time in the LLM era. Label propagation is a classic algorithm, but packaging it as "one LLM call for both self-reflection and incremental neighbor updates" is a brilliant migration of GNN ideas to agents.
- **91% gain for sparse users**: This is a staggering figure, suggesting that collaborative memory nearly doubles recommendation quality for long-tail users—revisiting the very reason traditional CF was successful, now rediscovered for the agent era.
- **Engineering optimization via asymmetric token pricing**: Compressing expensive output while allowing for cheaper input is a crucial engineering dimension often overlooked in LLM product design.

## Limitations & Future Work
- **Author Acknowledgments**: (1) Collaborative propagation is limited to 1-hop; multi-hop propagation introduces noise and costs. (2) Domain rules are generated once offline; highly dynamic domains (e.g., news) require online adaptation. (3) "Ceiling" performance still relies on commercial models like gpt-4o.
- **Observations**: (1) Evaluations primarily used a 1,000-user subset (except for main tables); whether advantages hold at full scale requires caution. (2) Cross-domain transfer and user profile drift were not explored. (3) Privacy—collaborative memory encodes neighbor signals into prompts, posing leakage risks as identified in recent MIA papers. (4) Memory graph expansion over time; no discussion of forgetting or compaction mechanisms.
- **Future Directions**: (1) Introducing differentially private federated memory updates. (2) Learning adaptive $k, N_f$ instead of fixed values. (3) Multi-hop propagation with trust-score gating. (4) Training $\text{LM}_{\text{Mem}}$ as a reward-tuned small model to further reduce costs. (5) Testing in streaming scenarios with shifting user profiles.

## Related Work & Insights
- **vs i2Agent / AgentCF / RecBot (Isolated dynamic memory agents)**: These involve self-reflection to modify $M_u$ or $M_i$, with updates strictly limited to the interacting parties. MemRec is the first to propagate updates along a collaborative graph.
- **vs Vanilla LLM / iAgent (No memory / Static memory)**: This work proves that memory levels from None → Static → Dynamic → Collaborative provide incremental gains in H@1 at each step.
- **vs LightGCN / SASRec (Traditional CF)**: Traditional CF fails on sparse data (Books) but works on dense data (Yelp). This paper "reactivates" CF's graph ideas with LLM reasoning, beating traditional methods across the board by integrating semantic understanding.
- **vs MemGPT / Generative Agents (General agent memory)**: They also use decoupled memory managers, but target factual memory and dialogue rather than graph structures and collaborative propagation. MemRec ports this paradigm to the RS domain.
- **vs Graph RAG**: Graph RAG uses KGs for structured retrieval at the retrieval end; MemRec goes further by using the graph as a dynamic, writable semantic memory.
- **Insights**: (1) Any scenario requiring an agent to "view many contexts + reflect" should consider a decoupled memory manager. (2) "LLM generates rules; rules filter online" is a general paradigm for reducing LLM reasoning costs. (3) Classic graph algorithms (label propagation, PageRank, community detection) are worth revisiting in the agent era.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of "Collaborative Memory" + Decoupled Dual LLM + Async Batched Propagation represents a clear new paradigm in agentic RS. Individual components (IB, label propagation) are not original, but the synthesis is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage across 4 datasets, 8 baselines, and 5 configurations, plus ablation, niche-user groups, noise robustness, GPT-4o-as-judge rationales, hyperparameter heatmaps, large candidate sets (N=20), and cost/latency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figure 1's "Island vs. Collaborative" comparison is extremely clear. The method section combines formulas, logic, and intuition. Prompts in the appendix are complete and case studies are detailed. Highly structured.
- **Value**: ⭐⭐⭐⭐⭐ Provides an industrially deployable solution for key bottlenecks of LLM-based RecSys (sparse users, cold start, explanation quality). With open code and a project page, the barrier to reproduction is low; this is a milestone work for the recommendation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)
- [\[ICML 2026\] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](../../ICML2026/recommender/learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)
- [\[NeurIPS 2025\] Radial Neighborhood Smoothing Recommender System](../../NeurIPS2025/recommender/radial_neighborhood_smoothing_recommender_system.md)
- [\[ICML 2026\] Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System](../../ICML2026/recommender/incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)

</div>

<!-- RELATED:END -->
