---
title: >-
  [Paper Note] CoMind: Towards Community-Driven Agents for Machine Learning Engineering
description: >-
  [ICLR 2026][LLM Agent] This paper proposes MLE-Live — the first real-time evaluation framework simulating a Kaggle research community — and CoMind, a multi-agent ML engineering system that systematically leverages collective community knowledge. CoMind achieves a 36% medal rate across 75 historical Kaggle competitions and outperforms an average of 79.2% of human participants on 4 active competitions (reaching 92.6% in an updated version).
tags:
  - ICLR 2026
  - LLM Agent
  - Machine Learning Engineering
  - Kaggle Competition
  - Community Knowledge
  - Multi-Agent Collaboration
date: 2026-05-08
content_hash: 9db82b8ef60696c9
---

# CoMind: Towards Community-Driven Agents for Machine Learning Engineering

**Conference**: ICLR 2026
**arXiv**: [2506.20640](https://arxiv.org/abs/2506.20640)
**Code**: [https://github.com/comind-ml/CoMind](https://github.com/comind-ml/CoMind)
**Area**: LLM Agent
**Keywords**: LLM Agent, Machine Learning Engineering, Kaggle Competition, Community Knowledge, Multi-Agent Collaboration

## TL;DR
This paper proposes MLE-Live — the first real-time evaluation framework simulating a Kaggle research community — and CoMind, a multi-agent ML engineering system that systematically leverages collective community knowledge. CoMind achieves a 36% medal rate across 75 historical Kaggle competitions and outperforms an average of 79.2% of human participants on 4 active competitions (reaching 92.6% in an updated version).

## Background & Motivation
LLM-based ML agents have demonstrated significant potential for automating ML engineering. MLAB adopts ReAct-style structured decision-making, AIDE employs tree search for exploration, and AutoKaggle introduces multi-agent specialization. These systems have shown progress on Kaggle-style competitions.

**Root Cause**: Existing agents operate in **isolated environments** — relying solely on internal memory and trial-and-error exploration — while completely ignoring a critical component of real-world ML workflows: **community knowledge sharing**. In real data science competitions and research, participants frequently learn from public discussions, shared notebooks, and community insights. Current agents, unable to leverage such dynamic external context, tend to converge to repetitive strategies and hit performance ceilings.

**Two Key Problems**:
1. How to evaluate an agent's ability to leverage collective knowledge? (→ MLE-Live benchmark)
2. How to design agents that effectively exploit community knowledge? (→ CoMind system)

## Method

### MLE-Live Evaluation Framework
Extended from MLE-Bench, MLE-Live augments each competition with a simulated community environment:

- **Community Resources**: 2,687 discussion posts and 4,270 public kernels collected from 22 Kaggle competitions (low-complexity split)
- **Metadata Quality Signals**: vote counts (community preference), public scores (performance metrics), author tiers (Novice to Grandmaster)
- **Temporal Constraint**: all content is published before the competition deadline to prevent post-hoc leakage
- **Filtering Rules**: non-textual content (images, screenshots) and Jupyter system outputs (progress bars, redundant logs) are removed
- **Evaluation Metrics**: Valid Submission (format correctness rate), Above Median (fraction exceeding median score), Win Rate (percentage of human participants beaten), Medals (Gold/Silver/Bronze)

### CoMind Agent Workflow
CoMind maintains two core repositories:
- **Idea Pool**: abstract insights distilled from community content and historical iterations
- **Report Pool**: complete solution reports containing code, evaluations, and analyses

Each iteration consists of four stages:

1. **Stage I — Idea Selection**:
    - Accesses concepts and strategies distilled from public kernels, forum discussions, and historical solutions in the Idea Pool
    - Uses the Report Pool as a guide for performance and relevance assessment to rank and filter entries
    - Simulates human participants browsing collective wisdom before forming new hypotheses

2. **Stage II — Idea Generation**:
    - Generates high-level solution drafts based on selected ideas and context from the Report Pool
    - Synthesizes new strategies by recombining or extending existing ideas
    - **Key Constraint**: avoids simple copying to ensure conceptual diversity and breadth of exploration
    - Simulates the human ability to abstract and innovate from prior work

3. **Stage III — Implementation and Improvement**:
    - Initiates a ReAct-style loop based on the generated draft
    - Iteratively writes code, executes, observes feedback (validation metrics, error logs), and updates the implementation
    - **Deliberately restricted context**: only the problem description and the specific draft are accessible; the Idea Pool and Report Pool are excluded
    - Ensures modularity of experiments and prevents context window explosion (at most 20 steps)

4. **Stage IV — Report Generation**:
    - Compiles a solution report covering method description, component analysis, quantitative results, and limitation assessment
    - The report is published back to the Report Pool and becomes visible to subsequent iterations
    - Simulates the real-world practice of users documenting and sharing their final solutions

### Parallel Agents and Shared Insights
- Multiple agents run in parallel on the same task and share a common community knowledge base
- Once an agent generates a new report, other agents can read it in subsequent iterations
- Agents inspire one another through shared reports, forming a collective exploration and improvement loop

### Key Design Principles
- **Balancing Exploration Breadth vs. Implementation Depth**: multiple distinct solution drafts are developed in parallel, with each iteration dynamically focusing on one draft for deep implementation
- **Knowledge Accumulation**: the Idea Pool and Report Pool grow across iterations, forming an increasingly rich knowledge base
- **Avoiding Context Explosion**: Stage III deliberately restricts accessible information to the current draft only

## Key Experimental Results

### Main Results (20 Historical Kaggle Competitions, using o4-mini)

| Method | Valid Sub. | Win Rate | Any Medal | Above Median | Medal Details |
|--------|-----------|----------|-----------|-------------|---------------|
| CoMind | **1.00** | **66.8%** | **45%** | **65%** | 5 Gold, 4 Silver |
| AIDE | 0.90 | 46.9% | 20% | 50% | — |
| AIDE+Code | 0.90 | 51.0% | 25% | 50% | — |
| AIDE+RAG | 0.95 | 51.2% | 25% | 55% | — |

CoMind earns 9 medals (5 gold), a 125% improvement over the previous SOTA AIDE.

### Online Competition Results (4 Active Kaggle Competitions)

| Competition | CoMind WR | AIDE WR | CoMind Rank |
|-------------|-----------|---------|-------------|
| playground-series-s5e5 | **94.9%** | 66.2% | #120/2338 |
| forams-classification-2025 | **91.7%** | 69.4% | #4/48 |
| el-hackathon-2025 | **61.6%** | 8.5% | #128/333 |
| fathomnet-2025 (CVPR FGVC12) | **69.4%** | 28.6% | #15/47 |

### Win Rate by Task Category

| Category | CoMind | AIDE | AIDE+Code | AIDE+RAG |
|----------|--------|------|-----------|----------|
| Image Classification (8) | **59.7%** | 45.9% | 43.4% | 52.5% |
| Text Classification (3) | **74.0%** | 15.7% | 33.8% | 61.0% |
| Audio Classification (1) | **90.1%** | 27.2% | 25.9% | 27.1% |
| Tabular (4) | 66.4% | 67.3% | **68.8%** | 48.3% |
| Image Regression (1) | 99.2% | 34.2% | **99.2%** | 99.2% |

### Ablation Study

| Configuration | Valid Sub. | Win Rate | Any Medal |
|---------------|-----------|----------|-----------|
| CoMind w/ public resources | **1.00** | **66.8%** | **45%** |
| CoMind w/o public resources | 0.90 | 54.5% | 35% |

### Key Findings
- **Community knowledge is critical**: removing public resources reduces Win Rate by 12.3% and Valid Submission by 10%, indicating that community knowledge not only improves quality but also provides baseline reliability
- **Sustained improvement**: AIDE rises quickly in the first 2 hours then plateaus, whereas CoMind continues to improve and eventually surpasses it
- **Higher code complexity**: CoMind-generated code is on average 55.4% longer than AIDE's, suggesting deeper reasoning and richer optimization techniques
- **Novelty assessment**: after excluding external ideas, CoMind achieves an average novelty rank of 1.20 (vs. AIDE's 3.05), demonstrating that it does not merely copy community solutions
- CoMind performs relatively weakly on Seq2Seq tasks because it tends to explore large-model fine-tuning strategies that often cannot be completed within the 1-hour runtime limit

## Highlights & Insights
- **Novel concept of "community-awareness"**: the first work to incorporate the community collaboration dynamics of data science competitions into LLM agent evaluation, bridging the large gap between "isolated agents" and "real research practice"
- **Four-stage iterative loop design**: the Idea Selection → Idea Generation → Implementation → Report pipeline closely mirrors the working mode of real researchers
- **Deliberate context restriction in Stage III**: this prevents performance degradation from information overload while ensuring independence of each solution draft — a design insight worth adopting
- **Real-world validation on active competitions**: submitting to ongoing Kaggle competitions with live leaderboard results substantially strengthens the paper's claims
- **Value of the MLE-Live benchmark**: provides a standardized evaluation platform for community-driven agent research

## Limitations & Future Work
- Currently supports only report-level interactions; finer-grained community dynamics such as commenting, questioning, and data/model sharing are absent
- Performance is constrained by runtime limits on tasks requiring large-model fine-tuning (e.g., Seq2Seq)
- Validation is limited to Kaggle-style ML competitions and has not been extended to broader domains such as scientific discovery, open-ended programming, or robotics
- Agent "innovation" may still be bounded by the knowledge scope of the underlying LLM backbone
- The communication and coordination mechanism among parallel agents is relatively simple (mediated solely through the Report Pool); richer message-passing protocols remain unexplored
- The constrained execution environment (single A6000 GPU, 5-hour total limit) may underestimate the potential of compute-intensive approaches

## Related Work & Insights
- **AIDE** (Jiang et al., 2025): tree-search-based ML agent, previously the strongest method on MLE-Bench
- **MLAB** (Huang et al., 2024): ReAct-style ML agent benchmark
- **MLE-Bench** (Chan et al., 2025): ML agent evaluation benchmark based on 75 Kaggle competitions
- **AutoKaggle** (Li et al., 2024): multi-agent system for ML engineering
- **MetaGPT** (Hong et al., 2023): general-purpose multi-agent collaboration framework
- Insight: agents should not rely solely on internal reasoning and trial-and-error — leveraging external "collective intelligence" is a key dimension for improving agent capability. This principle may generalize to other domains requiring community collaboration, such as scientific discovery and software engineering

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Community-driven agents + MLE-Live benchmark = an entirely new research direction)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (20 historical competitions + 4 active competitions + novelty assessment + ablation + code complexity analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though the presentation of some experimental results could be more compact)
- Value: ⭐⭐⭐⭐⭐ (Opens a new direction of community-aware agents with significant implications for data science automation)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [\[NeurIPS 2025\] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](../../NeurIPS2025/llm_agent/mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)

<!-- RELATED:END -->
