---
title: >-
  [Paper Note] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs
description: >-
  [ICLR 2026][Graph Learning][Knowledge Graph Question Answering] This paper proposes Explore-on-Graph (EoG), which leverages SFT followed by two-phase reinforcement learning (outcome reward + path-refined reward) to incen…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "Autonomous Exploration"
  - "Reinforcement Learning"
  - "Path-refined Reward"
  - "GRPO"
date: 2026-05-08
content_hash: 8c69228c3014c8e5
---

# Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs

**Conference**: ICLR 2026
**arXiv**: [2602.21728](https://arxiv.org/abs/2602.21728)  
**Code**: [Available](https://github.com/ysq111333/EoG)  
**Area**: Graph Learning
**Keywords**: Knowledge Graph Question Answering, Autonomous Exploration, Reinforcement Learning, Path-refined Reward, GRPO

## TL;DR
This paper proposes Explore-on-Graph (EoG), which leverages SFT followed by two-phase reinforcement learning (outcome reward + path-refined reward) to incentivize LLMs to autonomously explore reasoning paths on knowledge graphs beyond the training distribution, surpassing GPT-5 and Gemini 2.5 Pro on five KGQA benchmarks.

## Background & Motivation
- LLMs are prone to hallucination and knowledge gaps in QA tasks; knowledge graphs (KGs) serve as ideal grounding sources for factual verification.
- Existing methods fall into two categories, both with generalization limitations:
    - **Rule-based methods** (e.g., ToG, DoG): predefined rules constrain reasoning, training-free but unable to handle out-of-distribution patterns.
    - **Imitation-based methods** (e.g., RoG, KG-Agent): imitate reasoning patterns in training data, struggling to generalize to novel paths.
- Key insight: real-world KG reasoning involves **atypical paths** (e.g., indirect relations via *colleague* or *subsidiary*), requiring **autonomous exploration**.
- Example: common path "Google→employee→Carol→lives_in→London"; atypical path "Google→employee→Bob→colleague→John→resides_at→London".

## Method

### Overall Architecture
Two-stage training:
1. **SFT Stage**: trains the LLM on long-CoT data to acquire structured graph reasoning capabilities (cold start).
2. **RL Stage**:
    - Phase 1: GRPO + outcome reward (answer F1 score) to incentivize discovery of correct paths.
    - Phase 2: GRPO + joint reward (outcome + path-refined) to improve exploration efficiency and semantic quality.

### Key Designs

**Long-CoT Dataset Construction**:
- Carefully designed prompts require structured, logically rigorous, and KG-aligned reasoning.
- Gemini 2.5 Flash is used for knowledge distillation to generate reasoning paths (think tags) and final answers (answer tags).
- Additional rule-based filtering removes substandard reasoning traces, ensuring structural and factual correctness.
- Directly tasking an LLM to explore KGs faces a vast action space and extreme reward sparsity; SFT cold-start is therefore indispensable.

**Phase 1: Outcome Reward**:
- GRPO algorithm samples $S$ exploration paths per question.
- Reward = entity-level F1 score (rather than simple Hit@1), better suited for multi-answer scenarios.
- Paths failing to produce a correctly formatted answer tag automatically receive a reward of 0, implicitly encouraging proper formatting.
- Policy optimization is driven by within-group relative advantage normalization in GRPO.

**Phase 2: Path-refined Reward**:

Ground-truth path acquisition (Search-and-Verify Pipeline):
1. Identify topic entities in the question and answer entities in the KG.
2. BFS searches all connecting paths (with a maximum hop constraint) to ensure high recall.
3. An LLM (Gemini-2.5-Flash) semantically verifies whether each path matches the question intent, filtering spurious topological connections.

Path reward computation: checks whether every triple $(s, r, o)$ in the ground-truth path appears as a substring in the generated reasoning text, and uses the matching ratio as the path reward score.

Joint reward: $R_{\text{joint}} = R_{\text{outcome}} + \alpha \cdot R_{\text{path}}$, where $\alpha$ controls the weight of the path reward.

**Design Motivation for Two-Phase RL**: Phase 1 establishes basic exploration capability via outcome reward; Phase 2 refines exploration quality and efficiency via path reward.

### Loss & Training
- SFT stage: standard language modeling loss (cross-entropy) trained on long-CoT data.
- RL stage: GRPO objective with importance sampling ratio, clipping, and KL divergence regularization.
- Backbone models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
- SFT data distilled from Gemini-2.5-Flash; RL implemented with the verl framework using GRPO.

## Key Experimental Results

### Main Results (Five KGQA Benchmarks)

| Method | Model | WebQSP Hit@1 | CWQ Hit@1 | GrailQA Hit@1 | QALD10 Hit@1 | 2WikiMH Hit@1 |
|--------|-------|-------------|-----------|---------------|--------------|---------------|
| DoG | Llama-3.1-8B | 91.4 | 76.2 | - | - | 84.1 |
| GCR | Llama-3.1-8B | 92.2 | 75.8 | - | - | - |
| GPT-5 | - | 86.1 | 74.1 | 90.5 | 59.2 | 84.2 |
| Gemini-2.5-Pro | - | 92.1 | 71.9 | 91.6 | 58.6 | 85.1 |
| **EoG** | Qwen2.5-7B | 90.7 | **82.7** | **91.7** | **67.3** | 83.9 |
| **EoG** | Llama-3.1-8B | **92.8** | **86.6** | **92.1** | **70.6** | **85.3** |

EoG (Llama-3.1-8B) achieves 86.6 Hit@1 on CWQ, substantially outperforming GPT-5 (74.1) and Gemini-2.5-Pro (71.9).

| Complex Reasoning (CWQ F1) | Conjunction | Superlative | 1-hop | ≥4-hop |
|---------------------------|-------------|-------------|-------|--------|
| GCR | 63.7 | 52.6 | 66.3 | 45.8 |
| DoG | 53.3 | 45.9 | 50.3 | 46.7 |
| **EoG** | **70.2** | **64.7** | **76.2** | **69.6** |

EoG exhibits the largest advantage in the most challenging ≥4-hop reasoning setting (69.6 vs. 45.8/46.7).

### Ablation Study

| Variant | CWQ Hit@1 | CWQ F1 | WebQSP Hit@1 | WebQSP F1 |
|---------|-----------|--------|--------------|-----------|
| EoG (Full) | 82.6 | 73.9 | 92.8 | 81.3 |
| w/o Path Reward | 81.5 | 70.8 | 90.2 | 77.3 |
| w/o Outcome Reward | 62.7 | 51.4 | 65.5 | 56.2 |
| w/o SFT | 70.3 | 63.1 | 75.9 | 65.8 |
| w/o SFT + w/ ICL | 70.7 | 63.8 | 77.2 | 66.5 |

### Key Findings
1. The outcome reward is the most critical component; removing it drops CWQ Hit@1 from 82.6 to 62.7.
2. The path reward improves exploration efficiency: it reduces output length (CWQ: 2067→1528 tokens) while improving comprehensiveness and relevance.
3. SFT cold-start is indispensable; pure RL without SFT degrades performance substantially, and ICL cannot compensate.
4. An excessively small $\alpha$ leads to incorrect or meaningless paths, while an excessively large $\alpha$ causes the model to neglect answer correctness.
5. The path reward yields the greatest improvement on ≥4-hop reasoning, indicating that path-level signals are most critical for deep multi-hop inference.
6. EoG leads comprehensively across a six-dimensional reasoning quality evaluation, particularly on reasoning depth and exploratory capability.

## Highlights & Insights
- Open-source 7–8B models trained via RL exploration **surpass closed-source GPT-5 and Gemini-2.5-Pro**, demonstrating the power of autonomous exploration.
- The path-refined reward design is elegant: ground-truth paths are obtained via BFS + LLM semantic verification, and rewards are computed via triple substring matching.
- The contrast between "exploration" and "imitation" capabilities is compelling: imitation is constrained by the training distribution, while exploration can discover out-of-distribution paths.
- The curriculum design of two-phase RL (outcome reward first, then joint reward) is worth adopting in related settings.
- The largest gains appear in the most challenging ≥4-hop and superlative scenarios, validating the value of exploration for complex reasoning.

## Limitations & Future Work
- Ground-truth path acquisition relies on BFS + LLM verification, which may be computationally expensive for very large-scale KGs.
- The path reward is based on substring matching and may be sensitive to surface-form variation for the same entity.
- Validation is limited to Freebase and Wikidata; domain-specific KGs (e.g., biomedical) remain untested.
- Training data is distilled from Gemini-2.5-Flash, so data quality is bounded by the teacher model.
- Adaptation to dynamic KGs (where knowledge evolves over time) has not been explored.

## Related Work & Insights
- EoG aligns with the RL-based reasoning paradigm of DeepSeek-R1, extending it to the domain of **graph-structured reasoning**.
- Rule-based methods (ToG, DoG) provide structural guarantees but lack flexibility; EoG implicitly learns structural constraints through RL.
- The path-refined reward is conceptually similar to the process reward model (PRM), but more direct (based on factual matching rather than model judgment).
- EoG offers inspiration for KG-enhanced RAG: exploration strategies can be introduced into knowledge-augmented retrieval.

## Rating
- Novelty: 4/5 (The idea of RL-based KG exploration is novel; the path-refined reward design is particularly original.)
- Experimental Thoroughness: 5/5 (Five datasets, multiple baselines including closed-source models, detailed ablations and complex-scenario analyses.)
- Writing Quality: 4/5 (Clear framework presentation; illustrative examples are effective.)
- Value: 5/5 (Results showing an 8B model surpassing GPT-5 are highly convincing; practical utility is significant.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing](../../AAAI2026/graph_learning/assessing_llms_for_serendipity_discovery_in_knowledge_graphs_a_case_for_drug_rep.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](../../ACL2026/graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](../../ACL2026/graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)

</div>

<!-- RELATED:END -->
