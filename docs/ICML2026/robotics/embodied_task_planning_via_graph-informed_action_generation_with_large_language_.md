---
title: >-
  [Paper Note] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models
description: >-
  [ICML 2026][Robotics][Embodied Planning] GiG equips LLM planners with a "Graph-in-Graph" dual-layer memory (Scene Graph + State Transition Graph) + GNN encoding + 1-step lookahead. This enables the embodied agent to impr…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Embodied Planning"
  - "Graph Memory"
  - "GNN Encoding"
  - "Bounded Lookahead"
  - "Experience Retrieval"
date: 2026-05-08
content_hash: 9281572d9d6028bf
---

# Embodied Task Planning via Graph-Informed Action Generation with Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.21841](https://arxiv.org/abs/2601.21841)  
**Code**: TBD  
**Area**: Embodied AI / LLM Agent / Task Planning  
**Keywords**: Embodied Planning, Graph Memory, GNN Encoding, Bounded Lookahead, Experience Retrieval

## TL;DR
GiG equips LLM planners with a "Graph-in-Graph" dual-layer memory (Scene Graph + State Transition Graph) + GNN encoding + 1-step lookahead. This enables the embodied agent to improve Pass@1 by 6–37 percentage points over ReCAP on synchronous/asynchronous Robotouille and ALFWorld tasks.

## Background & Motivation

**Background**: When LLMs are used as the "brain" for embodied agents, mainstream approaches involve interleaved action-observation generation like ReAct/Reflexion, or hierarchical decomposition and backtracking using context trees like ReCAP.

**Limitations of Prior Work**: Pure text interleaving stuffs all history into the prompt, leading to *context drift* over long horizons—once the window is flushed, high-level goals are lost, and the agent begins to oscillate or produce contradictory actions. ReCAP’s context tree strictly serializes parallelizable subtasks; for instance, while "making soup," the sibling subtask of "setting the table" is blocked by the tree structure while waiting for water to boil, causing the agent to idle.

**Key Challenge**: Long-horizon planning must simultaneously satisfy three conflicting requirements: (i) persistent visibility of high-level intentions, (ii) free interleaving of sibling/parallel subtasks, and (iii) compact, retrievable state representations. Linear history fails (i)+(iii), while tree structures fail (ii).

**Goal**: To replace the agent's working memory with a structured container that can compress observations, represent parallel dependencies, and retrieve past successful experiences based on structural similarity.

**Key Insight**: The authors observe that embodied scenes are naturally graphs (object-relation), while action sequences are naturally another layer of graphs (state-transition). By nesting scene graphs within a state transition graph, one captures both spatial structure and temporal dynamics simultaneously.

**Core Idea**: Use GNNs to compress the scene graph at each step into an embedding node, with nodes connected by edges representing actually executed actions to form an episodic memory graph. During new decisions, successful past trajectories are retrieved from the memory bank via embedding similarity to serve as in-context demonstrations, overlaid with 1-step transition simulation for a "look-before-you-leap" approach.

## Method

### Overall Architecture
The operating unit of GiG is a five-step loop: (Observation → Parsing → Encoding → Retrieval → Action Selection). The input at each step $t$ is the environment observation $o_t$, and the output is the action $a_t$. Two layers of graphs are maintained:

- **Inner Scene Graph** $SG_t=(V_t,E_t)$: Nodes are entities (objects/robot), and edges are spatial relations (e.g., "cheese1 on-top-of table1"), constructed from raw observations by a deterministic or LLM-based parser.
- **Outer State Transition Graph** $OG$: Nodes are fixed-length embeddings $z_t$ compressed from $SG_t$ by a GNN, and edges are actions $a_t$ actually executed by the agent, forming a chain like $z_1 \xrightarrow{a_1} z_2 \xrightarrow{a_2} \cdots$.

The memory bank $M=\{E_j\}$ stores several successful historical trajectories (each being an OG + target $G_j$). For a new step, the current $z_t$ is used to query $M$; if a similar past state is found, the subsequent actions are fed back to the LLM as soft prompts.

### Key Designs

1. **Graph-in-Graph Dual-Layer Memory + GNN Encoding**:
    - **Function**: Compresses raw observations at each step into structure-aware embeddings and organizes the entire trajectory into a retrievable graph.
    - **Mechanism**: Initial features of scene graph nodes are initialized with a lightweight sentence encoder and aggregated via multi-layer GAT: $h_u^{(l)}=\sigma\big(\sum_{v\in N(u)}\alpha_{u,v}W^{(l)}h_v^{(l-1)}\big)$. Finally, $z_t$ is obtained via mean-pooling + BatchNorm. Training employs triplet loss + uniformity regularization $L=L_{triplet}+\lambda L_{uniformity}$. Anchors/positives are sampled from adjacent steps in the same OG, while negatives are sampled across OGs, assuming "gradual physical state changes → temporally adjacent embeddings should be closer." This naturally separates intra-trace distances (<0.1) from inter-trace distances (~0.8), leading to a retrieval threshold of $\tau=0.1$.
    - **Design Motivation**: Compared to flattening scene graphs into text prompts, structured embeddings retain topology (e.g., vertical dependencies like "patty on bun" are highlighted by attention) and compress variable-length observations into fixed-length vectors, curing context drift.

2. **Bounded Lookahead (BL) Module**:
    - **Function**: Switches the LLM from "imagining the future" to "choosing from future results," eliminating hallucinatory actions that violate dynamics.
    - **Mechanism**: When the dynamics $T$ are known (e.g., PDDL descriptions in Robotouille), the current valid action set $A(s_t)$ is enumerated. $T$ is called to calculate successor states, yielding a projection set $P(s_t)=\{(a,s')\mid a\in A(s_t),\ s'=T(s_t,a)\}$. $P(s_t)$, the current $SG_t$, retrieved experience $R_{z_t}$, and goal $G$ are concatenated into a prompt, allowing the LLM to perform discriminative selection over explicit successors: $a_{t+1}\sim \text{LLM}(\text{Prompt}\mid SG_t,P(s_t),R_{z_t},G)$. In partially observable environments like ALFWorld where $T$ is unavailable, $P(s_t)=\emptyset$, and the framework automatically falls back to pure graph + experience.
    - **Design Motivation**: $A(s)$ is typically a small finite set in task planning; enumerating 1 step is computationally feasible. This shifts the risk of "trusting LLM imagination" to actual dynamics while exposing only 1-step successors (rather than a full search tree) to ensure stable latency.

3. **Structure-Similarity Based Experience Retrieval + Loop Detection**:
    - **Function**: Uses past successful trajectories as one-shot demonstrations and actively stops the agent when it revisits old states.
    - **Mechanism**: All $(z,a)$ pairs from 50 successful Qwen3-235B trajectories are indexed in Faiss. At each step, $z_t$ is used to find the nearest neighbor $(z_k,d)$. If $d<\tau=0.1$, the subsequent sub-trajectory $R_{z_t}=[a_{k,m},a_{k,m+1},\dots]$ (effectively just the immediate next action) is injected into the prompt. Simultaneously, the OG of the current session is maintained separately; $z_t$ is compared against existing nodes at every step. A hit triggers *Loop Detection*, where the cyclical path is written into the prompt to warn the LLM ("You just looped through stack→unstack→stack").
    - **Design Motivation**: Since queries use local scene structure similarity rather than task text, they enable cross-task transfer—an "action chain" of "cut→pick→place" learned during "making a sandwich" can be reused for "making a burger," with the LLM acting as a semantic filter to decide whether to adopt it.

### Loss & Training
Only the GNN requires training; the LLM remains fully frozen. GNN loss is $L=L_{triplet}+\lambda L_{uniformity}$ with a triplet margin $\gamma=1.0$. The uniformity term is the mean of the squared cosine similarity of all pairs to prevent embedding collapse. All evaluations use a temperature of 0, a maximum generation limit of 4096 tokens, and a unified Pass@1 protocol (single execution, no retries or ensembles).

## Key Experimental Results

### Main Results
Evaluated on three embodied planning benchmarks, comparing GiG / GiG+Exp / ReCAP / ReAct / CoT across different LLM backends.

| Dataset | Backend | GiG | GiG+Exp | Prev. SOTA | Gain |
|---------|---------|-----|---------|------------|------|
| Robotouille Sync | Qwen3-235B | 93 | 97 | 74 (ReAct) | +19 / +23 |
| Robotouille Sync | DeepSeek-R1 | 91 | 88 | 72 (ReCAP) | +19 |
| Robotouille Async | Qwen3-235B | 72 | 82 | 35 (ReCAP) | +37 / +47 |
| Robotouille Async | DeepSeek-R1 | 59 | 86 | 27 (ReCAP) | +32 / +59 |
| ALFWorld | Qwen3-235B | 97 | – | 91 (ExpeL) | +6 |
| ALFWorld | DeepSeek-R1 | 97 | – | 82 (ReCAP) | +15 |

Asynchronous tasks showed the largest gains due to the need for concurrency management. In ALFWorld, due to object location randomization, the authors disabled experience memory; however, aggregation via the dual-layer graph alone still achieved 97%.

### Ablation Study

| Configuration | Robotouille Sync (Qwen3-30B) | Description |
|---------------|------------------------------|-------------|
| ReCAP baseline | 19 | Tree-based context + backtracking |
| ReAct baseline | 28 | Action-observation interleaving |
| GiG (No Exp) | 27 | Dual-layer graph + BL only, surpasses ReCAP |
| GiG + Exp | 42 | Experience memory adds +15 absolute points |
| GiG + Exp (Gemini-Flash-Lite) | 26 | Small model gains +7 under same config |

### Key Findings
- **Experience memory as a model-agnostic plug-in** is the most significant factor: trajectories collected by Qwen3-235B can be fed directly to Qwen3-30B or Gemini-Flash-Lite for inference, yielding absolute gains of +7 to +15 without fine-tuning.
- While GiG appears to have a "higher step count than baselines" on difficult tasks, this is because baselines simply fail those tasks; when including failed trials, GiG's average step count is lower, indicating a robust tradeoff: "preferring more steps to finish rather than failing early."
- The separation between intra-trace distance (<0.1) and inter-trace distance (~0.8) (Fig. 3) proves that the embeddings learned via GAT+triplet can distinguish different trajectories while identifying adjacent states, providing the underlying basis for using $\tau=0.1$ as a threshold.

## Highlights & Insights
- Using "graphs as memory containers" in a dual-layer fashion is the true differentiator: the inner layer handles observation "compression," while the outer layer handles action "concurrency." This adds a temporal dimension compared to pure Graph RAG (PoG/HiRAG) and provides more parallel expressivity than ReCAP's trees.
- The design philosophy of the BL module is noteworthy: "Do not let the LLM imagine consequences; let it choose based on real ones." By exposing only 1-step successors rather than a full search tree, the world model acts as a discriminator rather than a searcher, keeping latency controlled while drastically reducing error rates.
- The model-agnostic nature of experience memory is highly valuable for industrial deployment: the paradigm of large model collection + small model inference can be directly applied to any embodied LLM agent without retraining.

## Limitations & Future Work
- The scene parser currently relies on a deterministic parser (dependent on PDDL or environment metadata); transitioning to real visual embodied environments would require a robust VLM parser, which is not empirically validated in this paper.
- The BL module requires an explicit transition function $T$, failing which it must degrade; the use of learned world models as a replacement remains unverified.
- Experience memory was manually disabled for ALFWorld's randomized layouts, suggesting that structural similarity is sensitive to "geometric rearrangement," leaving the cross-layout transfer capability in question.
- The initial memory bank size of 50 trajectories is small; the study does not discuss how retrieval quality or latency would change if the memory bloats to thousands of entries.

## Related Work & Insights
- **vs ReCAP (2025b)**: Both use structured memory, but ReCAP uses trees for recursive decomposition + backtracking, whereas GiG uses graphs for parallel scheduling + experience reuse. ReCAP performs close to GiG in sync tasks but is outperformed by 30+ points in async tasks, confirming that "trees cannot handle concurrency."
- **vs ReAct / Reflexion**: Pure action-observation interleaving leaves all context in the prompt, leading to inevitable drift over long horizons; GiG compresses context into graph nodes, keeping the prompt short.
- **vs ExpeL**: ExpeL distills textual insights for retrieval, which struggles with ALFWorld's random layouts; GiG's graph structure similarity still works across layouts (though the authors admit it is more stable without Exp on ALFWorld).
- **vs GraphRAG / PoG / HiRAG**: These involve QA retrieval over static KGs; GiG translates Graph RAG concepts to embodied dynamic scenarios where the KG grows dynamically.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Graph-in-Graph is a clean abstraction that unifies scene structure and dynamics into a retrievable container—an independent innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 benchmarks × 5 LLMs × 4 baselines, including small model plug-in tests, though verification in real visual embodied environments is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Algorithm 1 and Figure 2 clearly explain the pipeline with consistent mathematical notation.
- **Value**: ⭐⭐⭐⭐ Substantial Pass@1 improvements and the model-agnostic memory plug-in are deployment-friendly. The structured memory research line is worth following.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/robotics/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICML 2026\] SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning](specprune-vla_accelerating_vision-language-action_models_via_action-aware_self-s.md)

</div>

<!-- RELATED:END -->
