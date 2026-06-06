---
title: >-
  [Paper Note] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning
description: >-
  [ICML 2026][Information Retrieval & RAG][GraphRAG] Graph-R1 reformulates GraphRAG as an end-to-end RL framework consisting of a "knowledge hypergraph environment + multi-turn think–query–retrieve–answer agent + outcome-o…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "GraphRAG"
  - "Reinforcement Learning"
  - "Knowledge Hypergraph"
  - "Agentic Retrieval"
  - "Multi-turn Reasoning"
date: 2026-05-08
content_hash: a9ce067fff882715
---

# Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2507.21892](https://arxiv.org/abs/2507.21892)  
**Code**: https://github.com/LHRLAB/Graph-R1 (Available)  
**Area**: Information Retrieval / GraphRAG / Agent  
**Keywords**: GraphRAG, Reinforcement Learning, Knowledge Hypergraph, Agentic Retrieval, Multi-turn Reasoning

## TL;DR
Graph-R1 reformulates GraphRAG as an end-to-end RL framework consisting of a "knowledge hypergraph environment + multi-turn think–query–retrieve–answer agent + outcome-oriented GRPO." By utilizing lightweight n-ary hypergraph construction and dual-path hyperedge retrieval with RRF fusion, it improves the F1 score of a 7B model from Search-R1's 46.19 to 57.82 across six standard RAG datasets.

## Background & Motivation

**Background**: RAG utilizes chunk-based retrieval to mitigate LLM hallucinations but neglects structural relationships between entities. GraphRAG approaches (GraphRAG, LightRAG, HyperGraphRAG, PathRAG, HippoRAG2, etc.) use entity-relation graphs to model knowledge, feeding LLMs with long contexts for reasoning via subgraph retrieval and path pruning.

**Limitations of Prior Work**: The authors identify three specific bottlenecks in current GraphRAG systems:

- High graph construction costs and semantic loss: Compressing natural language into binary (head, rel, tail) triplets constitutes lossy compression and requires massive LLM calls.
- "One-shot" fixed retrieval processes: Most GraphRAG systems retrieve a fixed subgraph and feed it to the generator upon receiving a query; complex multi-hop questions rely on brittle prompt engineering.
- Generation heavily depends on large models, long contexts, and meticulous prompting: Smaller models struggle with graph knowledge, and methods like HyperGraphRAG show minimal gains over StandardRAG, suggesting that "structure" is not effectively utilized.

**Key Challenge**: There is a contradiction between the potential benefits of graph structures (higher information density) and the current "static single retrieval + prompt concatenation" paradigm. Structural information must be "activated" by allowing the model to look back at the graph multiple times and retrieve based on intermediate states—a process prompt-only pipelines cannot achieve.

**Goal**: (i) Make graph construction more "information-dense" (n-ary hyperedges instead of binary triplets); (ii) transform retrieval from one-shot to multi-turn, controlled by the agent; (iii) end-to-end optimize the "think-retrieve-rethink-generate" trajectory via RL to learn "if, when, and what to query" rather than tuning prompts.

**Key Insight**: Inspired by DeepSeek-R1 and Search-R1, GraphRAG is reformulated as an RL problem: hypergraphs as the environment, n-ary facts as observations, think/query/retrieve/answer as actions, and token-level F1 plus format compliance as rewards, trained end-to-end via GRPO.

**Core Idea**: Replace "heavyweight graph construction + one-shot subgraph retrieval + long-context prompting" with "lightweight n-ary knowledge hypergraphs + multi-turn agent-hypergraph interaction + outcome-oriented GRPO," enabling small models to extract reasoning benefits from graph structures.

## Method

### Overall Architecture
Input: Knowledge corpus $K=\{d_1,\dots,d_N\}$ and user query $q$. Output: Natural language answer $y_q$.

The pipeline consists of two stages: **Offline**, the corpus is extracted into a knowledge hypergraph $\mathcal{G}_H=(V,E_H,\phi)$, where each hyperedge $h_i$ is a semantic segment linked to multiple entities $\mathcal{V}_{h_i}$, with shared encoder $\phi(\cdot)$ (bge-large-en-v1.5) computing embeddings for both; **Online**, the LLM agent $\pi_\theta$ executes multi-turn trajectories $\tau=((\mathbf{s}_1,\mathbf{a}_1),\dots,(\mathbf{s}_T,\mathbf{a}_T))$ around $\mathcal{G}_H$.

In each step, the agent first reflects in `<think>` whether current knowledge is sufficient. It then chooses to either issue a `<query>` via dual-path hyperedge retrieval (updating `<knowledge>`) or provide an `<answer>` to terminate. Trajectories are trained end-to-end using GRPO, with a scalar reward combining format compliance and answer F1—eliminating the need for intermediate step supervision or SFT cold-starts.

### Key Designs

1.  **Lightweight n-ary Knowledge Hypergraph Construction**:
    - **Function**: Compresses the corpus into $\mathcal{G}_H=(V,E_H,\phi)$, where each hyperedge represents a semantic fragment and its associated entities, serving as the agent's environment.
    - **Mechanism**: For each chunk $d$, an LLM extractor $\pi_{\text{ext}}(d)\to\{(h_i,\mathcal{V}_{h_i})\}_{i=1}^m$ extracts n-ary relational facts. $h_i$ is the relational/factual text, and $\mathcal{V}_{h_i}$ is the set of participating entities. Both share an encoder to obtain $\phi(v)$ and $\phi(h_i)$. Unlike HyperGraphRAG, it removes confidence scoring. On 2Wiki, construction takes 5.69s / $2.81 per 1K tokens, which is cheaper than GraphRAG (8.04s / $3.35) and HyperGraphRAG (6.76s / $4.14), resulting in 120K nodes and 98K hyperedges.
    - **Design Motivation**: Binary triplets split facts with multiple participants into several (h,r,t) pairs, causing semantic loss and edge explosion. N-ary hyperedges preserve fact granularity and provide entry points for both entity-based and hyperedge-based embedding retrieval.

2.  **Multi-turn Agent-Hypergraph Interaction (Dual-path Retrieval + RRF Fusion)**:
    - **Function**: Replaces "one-shot subgraph retrieval" with an agent-driven think–query–retrieve–answer loop, retrieving relevant n-ary facts via complementary paths.
    - **Mechanism**: The action space $\mathbf{a}_t=(\mathbf{a}_t^{\text{think}},\alpha_t,\mathbf{a}_t^{\text{out}})$ is decomposed via a hierarchical policy. Upon receiving a query, two paths run in parallel: (i) Entity path $\mathcal{R}_V=\arg\max^{k_V}_v \text{sim}(\phi(V_{\mathbf{a}_t^{\text{query}}}),\phi(v))$, collecting all hyperedges linked to these entities; (ii) Hyperedge path $\mathcal{R}_H=\arg\max^{k_H}_{e_H}\text{sim}(\phi(\mathbf{a}_t^{\text{query}}),\phi(e_H))$ retrieving hyperedges directly. Reciprocal Rank Fusion (RRF) $\text{Score}(f)=1/r_V+1/r_H$ is used to feed the top-$k$ results back into the `<knowledge>` tag.
    - **Design Motivation**: The entity path excels when the agent knows the entity name but needs to find its context; the hyperedge path excels at finding relations/events without specific entity knowledge. RRF allows the fusion of rankings without score alignment. On 7B models, each query averages only 2.3–2.5 turns and 1200–1500 tokens, which is shorter but more accurate than Search-R1/R1-Searcher.

3.  **Outcome-oriented End-to-end GRPO Optimization**:
    - **Function**: Uses a scalar reward to backpropagate "format compliance + answer correctness" to the multi-turn policy $\pi_\theta$ without step-wise supervision or SFT.
    - **Mechanism**: The reward consists of two parts: Format reward $R_{\text{format}}(\tau)=\min(1.0, 0.5\cdot\sum_t \mathbb{I}\{(\mathbf{a}_t^{\text{think}},\alpha_t,\mathbf{a}_t^{\text{out}})\})$ encourages the full think→query/answer structure; Answer reward $R_{\text{answer}}$ uses token-level F1 against the ground truth. Total reward $R(\tau)=-1.0+R_{\text{format}}(\tau)+\mathbb{I}\{R_{\text{format}}(\tau)=1.0\}\cdot R_{\text{answer}}$. **Answer points are only calculated if format points are maximized**, preventing shortcuts. Optimization uses GRPO with group-relative advantage $\hat A(\tau_i)=(R(\tau_i)-\text{mean}(\{R(\tau_j)\}))/F_{\text{norm}}(\cdot)$, PPO-style clipping, and KL anchoring.
    - **Design Motivation**: Format rewards solve the cold-start problem where agents fail to use tags. F1 rewards are more tolerant than EM for multi-hop QA variations. The format-gate forces the policy into a structured output space, bypassing SFT. GRPO is more suitable for long-horizon, sparse-reward trajectories than PPO/REINFORCE++ as it avoids the need for a value model.

### Loss & Training
GRPO Objective: $\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}[\frac{1}{N}\sum_i\frac{1}{|\tau_i|}\sum_t\min(\rho_\theta\hat A,\text{clip}(\rho_\theta,1\pm\epsilon)\hat A)-\beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})]$, where $\rho_\theta=\pi_\theta/\pi_{\theta_{\text{old}}}$. Base models: Qwen2.5-{1.5B,3B,7B}-Instruct; Hardware: 4×A100-80G; GPT-4o-mini used for n-ary fact extraction; bge-large-en-v1.5 used for retrieval.

## Key Experimental Results

### Main Results
Evaluated on 6 RAG datasets (2Wiki / HotpotQA / Musique / NQ / PopQA / TriviaQA) using EM, F1, Retrieval-Similarity (R-S), and Generation-Eval (G-E). Table shows mean results for 7B base.

| Method (Qwen2.5-7B) | EM | F1 | R-S | G-E |
|--------|------|------|------|------|
| StandardRAG | 5.34 | 15.89 | 52.67 | 65.18 |
| HyperGraphRAG (GPT-4o-mini, prompt-only) | 13.15 | 29.40 | 61.82 | 78.92 |
| Search-R1 (chunk + RL) | 38.54 | 46.19 | 51.60 | 68.60 |
| R1-Searcher (chunk + RL) | 34.51 | 42.29 | 51.26 | 69.08 |
| **Graph-R1 (Ours)** | **48.57** | **57.82** | 60.40 | **76.23** |

On 1.5B models, Graph-R1 improved F1 from Search-R1's 29.53 to 40.09; on 3B, from 35.69 to 51.26; on 7B, from 46.19 to 57.82—an absolute gain of approximately +10–16 points over RL-RAG baselines at the same parameter scale.

### Ablation Study
Mean of 2Wiki + HotpotQA (3B/7B base):

| Configuration | EM (3B) | F1 (3B) | EM (7B) | F1 (7B) | Description |
|------|--------|---------|---------|---------|------|
| Full Graph-R1 | 50.39 | 57.16 | 56.25 | 63.87 | Complete Model |
| w/o K.C. | 38.48 | 46.11 | 46.68 | 53.87 | Remove Hypergraph (back to chunk) F1 -11/-10 |
| w/o M.I. | 26.37 | 35.99 | 39.07 | 45.91 | Remove Multi-turn (one-shot) F1 -21/-18 |
| w/o R.L. | 3.13 | 10.74 | 1.56 | 17.79 | Remove RL (pure prompt agent) nearly collapses |

### Key Findings
- **RL contributes the most**: Removing RL causes EM to drop to single digits, proving that the "multi-turn agent + graph environment" cannot be sustained by prompting alone; RL is required to learn the strategy. Multi-turn interaction (M.I.) follows, as single-turn retrieval only captures shallow evidence. Hypergraph construction (K.C.) contributes the least but still provides a stable +10 F1 gain.
- **Synergy between RL and Knowledge Representation**: Figure 5(b) shows "No knowledge < chunk + RL < binary graph + RL < hypergraph + RL". Prompt-only HyperGraphRAG performs worse than StandardRAG, indicating graph structures do not automatically guarantee better results; however, RL significantly raises the ceiling for graph-based F1.
- **Improved Efficiency**: On 2Wiki, Graph-R1 (7B) takes 7.0s per query with $0 generation cost (as the model learns when to stop), whereas HyperGraphRAG takes 9.6s / $8.76. Graph-R1 is faster and cheaper while jumping F1 from 21.1 to 65.0.
- **Scaling Benefits**: F1 scores increase monotonically from 1.5B to 7B (40.09→51.26→57.82), and the gap relative to Search-R1 widens, suggesting that structured reasoning relies on base model scale.
- **Negative Case**: On Qwen3-4B (already RL-tuned), Graph-R1 performed slightly worse, as the model tended to trust its internal reasoning over-querying, suggesting interference between successive RL phases.

## Highlights & Insights
- The design where "format reward gates answer reward" is significant: using $\mathbb{I}\{R_{\text{format}}=1.0\}$ as a coefficient forces the model to master the structure before being rewarded for answers. This acts as a free "behavioral constraint + reward shaping," avoiding reward hacking without SFT.
- Dual-path retrieval (Entity + Hyperedge) with RRF fusion combines two natural perspectives of dense retrieval without requiring a cross-encoder or reranker, making it a practical trick for industrial RAG.
- Framing the "failure" of the prompt-only HyperGraphRAG as motivation and closing the loop with the "w/o R.L." ablation creates a strong narrative: graph structure alone is insufficient; RL is the key to unlocking it.
- Theoretical proofs (Lyapunov / Mutual Information / Fano’s Inequality) support three propositions (graph density, multi-turn information gain), transforming engineering intuition into provable scalability.

## Limitations & Future Work
- Graph construction still relies heavily on GPT-4o-mini for n-ary relations (5.69s/$2.81 per 1K tokens), which is costly for million-scale corpora. Performance with open-source extractors is unknown.
- Experiments focus on QA-style standard RAG benchmarks; performance on long-form generation (reports, code QA) is unclear, and token-F1 might encourage keyword shortcuts.
- Deterioration on Qwen3-4B suggests potential conflicts when applying GraphRAG-RL to models that have already undergone extensive RL, necessitating specialized warm-start strategies.
- The knowledge hypergraph is a static offline environment; incremental updates were not addressed.
- Detailed sensitivity analysis for hyperparameters like maximum turns or $k_V/k_H$ is missing.

## Related Work & Insights
- **vs HyperGraphRAG**: Both use hypergraphs, but HyperGraphRAG is prompt-only and one-shot. Graph-R1 removes complex scoring to reduce costs and uses RL to increase F1 from 29.4 to 57.8.
- **vs Search-R1 / R1-Searcher**: Graph-R1 shares the agentic RL paradigm but retrieves n-ary facts from a hypergraph instead of text chunks. Figure 2 proves that hypergraphs provide a higher ceiling than chunks under the same RL framework.
- **vs DeepSeek-R1 / GRPO**: Directly adapts the GRPO optimizer and R1's "think → tool → answer" paradigm, extending it from "internal CoT + tools" to "internal CoT + graph environment."
- **vs PathRAG / HippoRAG2**: These focus on subgraph/path strategies within static one-shot retrieval. Graph-R1 proves the real lever is the agent's autonomy in the multi-turn process.

## Rating
- Novelty: ⭐⭐⭐⭐ First clean combination of agentic RL, n-ary hypergraphs, and GRPO showing significant gains.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 datasets, 3 base models, 5+ baselines, 3-module ablation, cost analysis, and model version comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure (3 challenges → 3 designs → 3 propositions), comprehensive figures, and theoretical proofs.
- Value: ⭐⭐⭐⭐ Open-source code; shifts GraphRAG toward an RL paradigm; effective on small models; high industrial relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](../../ACL2026/information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/information_retrieval/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/information_retrieval/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](../../ACL2026/information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)

</div>

<!-- RELATED:END -->
