---
title: >-
  [Paper Note] PaSa: An LLM Agent for Comprehensive Academic Paper Search
description: >-
  [ACL2025][LLM Agent][Academic Paper Search] PaSa is an LLM-based academic paper search agent that achieves comprehensive and accurate literature retrieval by autonomously invoking search tools, reading papers, and navigating citation networks. Trained with RL, it significantly outperforms Google Scholar and GPT-4o in real-world scenarios.
tags:
  - "ACL2025"
  - "LLM Agent"
  - "Academic Paper Search"
  - "Reinforcement Learning"
  - "Citation Network"
  - "Paper Retrieval"
date: 2026-05-08
content_hash: 1b6cac8cbc1ce06d
---

# PaSa: An LLM Agent for Comprehensive Academic Paper Search

**Conference**: ACL2025  
**arXiv**: [2501.10120](https://arxiv.org/abs/2501.10120)  
**Code**: [bytedance/pasa](https://github.com/bytedance/pasa)  
**Area**: LLM Agent  
**Keywords**: Academic Paper Search, LLM Agent, Reinforcement Learning, Citation Network, Paper Retrieval  

## TL;DR

PaSa is an LLM-based academic paper search agent that achieves comprehensive and accurate literature retrieval by autonomously invoking search tools, reading papers, and navigating citation networks. Trained with RL, it significantly outperforms Google Scholar and GPT-4o in real-world scenarios.

## Background & Motivation

Academic paper search is a core task in scientific research, yet it translates into a particularly challenging information retrieval problem. It requires long-tail specialized knowledge, survey-level comprehensive coverage, and the ability to process fine-grained queries. For example, for a query like "Which studies focus on non-stationary reinforcement learning using UCB-based algorithms?", general-purpose search tools like Google Scholar often fall short.

Typically, researchers do not only use search engines; they also read relevant papers and inspect citation relations to conduct thorough literature surveys. This process is highly time-consuming. Although LLMs have been explored for information retrieval enhancement (such as query reformulation), utilizing LLMs as autonomous agents to simulate the complete workflow of human literature research—search, reading, and citation tracking—remains unexplored.

## Method

### Overall Architecture

The PaSa system comprises two LLM Agents (based on Qwen2.5-7B):

- **Crawler**: Responsible for autonomously collecting papers to maximize recall.
- **Selector**: Responsible for accurately judging whether a paper satisfies the search query, emphasizing precision.

### Key Designs

#### Crawler Design

The Crawler executes a token-level Markov Decision Process (MDP) where the action space corresponds to the LLM vocabulary. Three tools are registered:

| Function | Function |
|------|------|
| [Search] | Generates search queries, calls search tools, and appends results to the paper queue. |
| [Expand] | Generates sub-section names and appends all papers cited in those sections to the queue. |
| [Stop] | Resets the context to the user query plus the next paper in the queue. |

Workflow: Upon receiving a user query, the Crawler can repeatedly execute Search (searching multiple times with different queries), perform Expand after reading a paper (tracking the citation network to discover more relevant papers), or Stop (switching to the next paper). The exploration depth is restricted to 3 layers.

#### Crawler Training

**Two-Stage Training**:

**Stage 1: Imitation Learning**
- Demonstration trajectories are generated for 5,000 queries to perform supervised fine-tuning.
- Learning rate is 1e-5, batch size is 4, training for 1 epoch.

**Stage 2: Reinforcement Learning (Session-Level PPO)**

Challenges faced:
- **Sparse Rewards**: The paper set in AutoScholarQuery is only a subset of the actual qualifying papers.
- **Long Trajectories**: A complete trajectory may involve hundreds of papers, exceeding the context length limit of the LLM.

Solution—**Session-Level PPO**:

The trajectory is partitioned into a sequence of sessions, each ending with [Stop]. Two types of initial states are defined: $S_q$ (containing only the query) and $S_{q+p}$ (containing the query and the paper).

Reward Design:
$$r(s_t, a_t) = \alpha \times \sum_{i=1}^{n_t} \mathbb{I}(q, p_i, t) - c(a_t)$$

where $\mathbb{I}$ determines whether the paper matches the query. To alleviate sparse rewards, the Selector is used as an auxiliary reward model—giving a positive reward when the Selector determines that a paper satisfies the query, or when the paper is in the annotated set.

The return estimation combines an intra-session discount factor $\gamma_0$ and an inter-session discount factor $\gamma_1$, with a per-token KL penalty incorporated to prevent over-optimization.

### Selector Design

Accepts the query and a paper (title + abstract) and outputs:
1. A decision token (True/False)
2. Reasoning rationale

Key Design: The decision token is prepended to the rationale, allowing the Selector to serve as a single-token reward model during Crawler training. The optimization is based on imitation learning.

### Dataset Construction

#### AutoScholarQuery
- Constructed from the Related Work sections of papers in ICLR 2023, ICML 2023, NeurIPS 2023, ACL 2024, and CVPR 2024.
- GPT-4o is used to generate fine-grained academic queries from citation relationships.
- Contains 33,511 / 1,000 / 1,000 training/validation/test instances.
- Human evaluation: 94% of queries are valid, and 93.7% of papers match.

#### RealScholarQuery
- Consists of 50 real-world academic queries.
- Real queries submitted by AI researchers on the PaSa demo.
- Expert annotators (professors from top universities) reviewed all candidate papers.
- On average, 76 candidate papers were reviewed per query, with an annotation cost of $304/query.
- Each query is associated with an average of 15.82 ground-truth papers.

## Experiments

### Baselines
- Google / Google Scholar (direct search)
- Google with GPT-4o (search after query reformulation by GPT-4o)
- ChatGPT (search-augmented GPT-4o)
- GPT-o1 (without external search)
- PaSa-GPT-4o (implementing the PaSa Agent using GPT-4o)

### Main Results

| Method | Precision | Recall | Recall@20 | Recall@50 | Recall@100 |
|------|-----------|--------|-----------|-----------|------------|
| Google | - | - | 0.1568 | 0.1891 | 0.2015 |
| Google + GPT-4o | - | - | 0.1921 | 0.2450 | 0.2683 |
| ChatGPT | 0.0507 | 0.3046 | - | - | - |
| PaSa-GPT-4o | 0.1457 | 0.3873 | - | - | - |
| **PaSa-7B** | **0.1448** | **0.4834** | **0.5301** | **0.6334** | **0.6947** |

Compared with Google + GPT-4o, PaSa-7B improves Recall@20 by **33.80%** and Recall@50 by **38.83%**.

### RealScholarQuery Main Results

| Method | Precision | Recall | Recall@20 | Recall@50 |
|------|-----------|--------|-----------|-----------|
| Google + GPT-4o | - | - | 0.2020 | 0.2573 |
| PaSa-GPT-4o | 0.4721 | 0.3075 | - | - |
| **PaSa-7B** | **0.5146** | **0.6111** | **0.5798** | **0.6563** |

PaSa-7B shows an even greater advantage in real-world scenarios, improving recall by **30.36%** and precision by **4.25%** compared to PaSa-GPT-4o.

### Selector Evaluation

| Method | Precision | Recall | F1 |
|------|-----------|--------|----|
| GPT-4o | 0.96 | 0.69 | 0.80 |
| Qwen2.5-7B | 1.00 | 0.38 | 0.55 |
| **PaSa Selector** | **0.95** | **0.78** | **0.85** |

The Selector F1 score reaches 85%, outperforming GPT-4o's 80%.

### Ablation Study

| Setting | Crawler Recall (Auto) | Recall (Auto) | Crawler Recall (Real) | Recall (Real) |
|------|----------------------|---------------|----------------------|---------------|
| w/o [Expand] | 0.3355 | 0.2536 | 0.3359 | 0.2890 |
| w/o RL training | 0.6556 | 0.4210 | 0.4847 | 0.4115 |
| w/o Selector as RM | 0.7041 | 0.4458 | 0.5994 | 0.5148 |
| **PaSa-7B** | **0.7931** | **0.4834** | **0.7071** | **0.6111** |

- Removing [Expand] (citation network navigation) results in the largest drop in Recall (approximately 50%+), demonstrating that citation network tracking is a core capability.
- RL training brings an improvement of about 6-20%.
- Using the Selector as an auxiliary reward model also contributes significantly.

### Key Findings

1. **7B Model Beats GPT-4o Agent**: After RL training, PaSa-7B significantly outperforms PaSa-GPT-4o implemented via prompting GPT-4o.
2. **Citation Network Navigation is Crucial**: The Crawler discovers a large number of relevant papers when diving deep into the citation network, even if intermediate papers are not directly related to the query.
3. **Trained on Synthetic Data, Generalizes to Real Scenarios**: Although trained only on AutoScholarQuery, it exhibits stronger generalization on RealScholarQuery.
4. **Ensemble Further Boosts Performance**: Running the Crawler twice with sample decoding adds an extra 3-4% improvement in Crawler Recall.

## Highlights & Insights

1. **Mimicking the Complete Human Literature Research Workflow**: Rather than simple query reformulation, it performs search, reads papers, and tracks citations, far exceeding simple query reformulation paradigms.
2. **Innovative Session-Level PPO**: Elegantly addresses sparse rewards and long trajectory problems, making RL feasible for agent tasks with long trajectories.
3. **Dual Role of the Selector**: Acts as both the final filter and an auxiliary reward model for RL, enabling a single component to serve a dual purpose.
4. **High-Quality Datasets**: AutoScholarQuery constructed from the Related Work sections of top-tier conference papers features exceptionally high quality; RealScholarQuery is highly representative, despite high annotation costs ($304/query).
5. **Outstanding Practical Value**: An online demo (pasa-agent.ai) is already live, catering directly to the essential needs of researchers for paper search.

## Limitations & Future Work

1. The search tools are limited to Google + arXiv, leaving other academic databases uncovered.
2. The Crawler's exploration depth is limited to 3 layers, potentially missing papers in deeper citation networks.
3. AutoScholarQuery is confined to top AI conferences, and generalization to other academic disciplines has not been verified.
4. RealScholarQuery contains only 50 queries, representing a relatively small scale.
5. Paper acquisition relies on ar5iv; full-text contents may not be retrievable for some papers.

## Related Work & Insights

- **LLM Applications in Scientific Discovery**: Idea generation, experimental design, paper writing, etc., but literature review automation remains insufficient.
- **LLM Agents**: Frameworks like AGILE, ReAct, etc., are utilized for tool use and planning; this work adopts the AGILE framework.
- **Academic Search Enhancement**: Query reformulation and retrieval-augmented generation are effective in general IR, but agent systems tailored specifically for academic search are still lacking.

## Rating

⭐⭐⭐⭐⭐ — Elegant system design, comprehensive experiments, and extremely high practical value. The innovation of Session-Level PPO solves the practical difficulties of Agent RL training, and the 7B model outperforming the GPT-4o Agent is highly impressive. The dataset construction is of high quality, and targeting paper search—a core demand for scientific researchers—makes this work highly impactful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Paper2Figure: A Multi-Agent Collaborative System for Figure Generation Towards Academic Research Paper](../../CVPR2026/llm_agent/paper2figure_a_multi-agent_collaborative_system_for_figure_generation_towards_ac.md)
- [\[ICLR 2026\] Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations](../../ICLR2026/llm_agent/presenting_a_paper_is_an_art_self-improvement_aesthetic_agents_for_academic_pres.md)
- [\[ICML 2025\] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](../../ICML2025/llm_agent/improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)
- [\[ICLR 2026\] Tree Search for LLM Agent Reinforcement Learning](../../ICLR2026/llm_agent/tree_search_for_llm_agent_reinforcement_learning.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)

</div>

<!-- RELATED:END -->
