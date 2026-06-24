---
title: >-
  [Paper Note] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System
description: >-
  [ACL 2025][LLM (Other)][Scientific discovery] Proposed the VirSci multi-agent system, which constructs a virtual research ecosystem using real scientist data, generating scientific ideas through a 5-step collaborative workflow and an innovative inter- & intra-team discussion mechanism, significantly outperforming single-agent systems in novelty and potential impact.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Scientific discovery"
  - "multi-agent system"
  - "idea generation"
  - "scientific collaboration"
  - "LLM Agent"
date: 2026-05-08
content_hash: 776d683a8bb43800
---

# Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System

**Conference**: ACL 2025  
**arXiv**: [2410.09403](https://arxiv.org/abs/2410.09403)  
**Code**: [https://github.com/open-sciencelab/Virtual-Scientists](https://github.com/open-sciencelab/Virtual-Scientists)  
**Area**: LLM / NLP  
**Keywords**: Scientific discovery, multi-agent system, idea generation, scientific collaboration, LLM Agent

## TL;DR
Proposed the VirSci multi-agent system, which constructs a virtual research ecosystem using real scientist data, generating scientific ideas through a 5-step collaborative workflow and an innovative inter- & intra-team discussion mechanism, significantly outperforming single-agent systems in novelty and potential impact.

## Background & Motivation

**Background**: AI for Science has evolved from molecular design / protein prediction to using LLMs to assist in scientific idea generation. AI Scientist (Lu et al., 2024) realizes end-to-end automation from ideas to papers, while HypoGen (Qi et al., 2024) introduces multi-agent hypothesis generation.

**Limitations of Prior Work**:
   - AI Scientist is a single-agent system that fails to simulate team collaboration in real scientific research, even though over $90\%$ of papers in Nature/Science are multi-authored.
   - Although HypoGen / ResearchTown utilize multiple agents, they use manually constructed artificial profiles and synthetic collaboration networks, which do not reflect real academic community dynamics.
   - Existing multi-agent frameworks use simple all-to-all discussion topologies without inter-team communication mechanisms.
   - There is a lack of objective automatic novelty evaluation metrics aligned with human judgment.

**Key Challenge**: Real scientific innovation highly depends on team diversity and collaboration mechanisms, but existing LLM scientific discovery systems either neglect collaboration or use unrealistic collaboration simulations.

**Goal**: (1) Build a trustworthy multi-agent collaborative system using real scientist data. (2) Design a five-step process that simulates real scientific research collaboration. (3) Systematically study the impact of team size, freshness, and diversity on idea novelty.

**Key Insight**: Construct a "virtual research ecosystem" as a digital twin—with scientist backgrounds and papers derived from real data (AMiner/OAG), and use temporal splitting (past vs. contemporary) as an evaluation reference to ensure objective assessment.

**Core Idea**: Simulate multi-agent research team collaboration using real scientist data, generating scientific ideas more novel than those from single agents through inter-team invitation and novelty voting mechanisms.

## Method

### Overall Architecture
Real academic dataset (AMiner/OAG) $\rightarrow$ Construct virtual research ecosystem ($B_{past}$ paper database + $B_{con}$ paper database + scientist database + collaboration adjacency matrix) $\rightarrow$ Five-step multi-agent collaboration: 1. Teaming $\rightarrow$ 2. Topic discussion $\rightarrow$ 3. Idea generation $\rightarrow$ 4. Novelty assessment voting $\rightarrow$ 5. Abstract writing.

### Key Designs

1. **Virtual Research Ecosystem**:
    - **Past paper library $B_{past}$**: Papers before the time cutoff, indexed by Faiss, used by agents to retrieve references during idea generation.
    - **Contemporary paper library $B_{con}$**: Papers after the time cutoff, used solely for evaluation—verifying if the generated ideas align with real future research directions.
    - **Scientist knowledge base**: Uses KnowledgeBank of AgentScope to store real scientists' names (anonymized), affiliations, citation counts, research interests, and collaboration history.
    - **Collaboration adjacency matrix $A$**: $A_{ij}$ indicates the number of historical collaborations between scientists $i$ and $j$, with $+1$ to ensure non-collaborators still have a probability of being selected (explore-exploit).
    - **Design Motivation**: Construct agent roles using real rather than synthetic data to ensure the simulated academic collaboration network structure is realistic.

2. **Inter- & Intra-team Discussion Mechanism**:
    - **Intra-team discussion**: Members take turns speaking in a round-robin order, with the team leader summarizing each round of discussion.
    - **Inter-team invitation ("Invitation Mechanism")**: During discussions, agents can search for scientists outside the team via RAG and temporarily invite them to participate in the discussion without officially joining the team.
    - **Design Motivation**: Simulate the two-tier communication mode of "close internal team discussion + consulting external experts" in real scientific research.

3. **Novelty Assessment & Voting**:
    - Retain the top 3 ideas with the highest confidence from the idea generation stage.
    - Each agent independently retrieves the most relevant papers from $B_{past}$ for each idea to judge whether the work duplicates existing literature.
    - Simulate blind review: Voters operate without discussion memory, performing chain-of-thought reasoning based solely on the idea content and references before voting.
    - The idea with the highest votes proceeds to abstract writing.
    - **Design Motivation**: Introduce a peer review mechanism to reduce agent overconfidence and ensure truly novel ideas are selected.

### Evaluation Metrics

| Metric | Definition | Direction |
|------|------|------|
| HD (Historical Dissimilarity) | Average Euclidean distance to the top-5 most similar papers in $B_{past}$ | $\uparrow$ Larger is more novel |
| CD (Contemporary Dissimilarity) | Average Euclidean distance to the top-5 most similar papers in $B_{con}$ | $\downarrow$ Smaller is better aligned with the future |
| CI (Contemporary Impact) | Average citation count of the top-5 most similar papers in $B_{con}$ | $\uparrow$ Larger indicates higher potential impact |
| ON (Overall Novelty) | Normalized overall score of $(HD \times CI) / CD$ | $\uparrow$ Larger is better |
| Human Evaluation | Nov (Novelty) / Fea (Feasibility) / Eff (Effectiveness), 1-7 Likert | $\uparrow$ |

## Key Experimental Results

### Comparison with Baselines (GPT-4o as Agent Model)

| Method | CD $\downarrow$ | CI $\uparrow$ | Nov $\uparrow$ | Fea $\uparrow$ | Eff $\uparrow$ |
|------|------|------|-------|-------|-------|
| HypoGen | 0.36 | 3.10 | 4.78 | 4.24 | 4.43 |
| AI Scientist | 0.38 | 3.22 | 4.94 | 4.18 | 4.77 |
| **VirSci (Ours)** | **0.34** | **3.78** | **5.24** | **4.52** | **4.95** |

### Comparison with Baselines (LLaMA3.1-70b as Agent Model)

| Method | CD $\downarrow$ | CI $\uparrow$ | Nov $\uparrow$ | Fea $\uparrow$ | Eff $\uparrow$ |
|------|------|------|-------|-------|-------|
| HypoGen | 0.49 | 2.13 | 3.57 | 3.61 | 3.52 |
| AI Scientist | 0.48 | 2.11 | 3.88 | 3.60 | 3.66 |
| **VirSci (Ours)** | **0.40** | **3.36** | **4.18** | **3.84** | **3.75** |

### Ablation on Collaboration Mechanism

| Factor | Optimal Value | Key Findings |
|------|-------|---------|
| Team Size | 8 members | Groupthink occurs beyond 8 members, causing ON to decrease |
| Discussion Rounds | 5 rounds | Too many rounds lead to "discussion fatigue" and decreased innovativeness |
| Team Freshness | 50% (half new + half old) | Purely new or purely old partners perform worse than hybrid teams |
| Research Diversity | 50-75% | Consistent with the "atypical combinations" theory from Science of Science |

### Key Findings
- **Multi-agent is significantly better than single-agent**: Average CD improved by $+13.8\%$, and CI improved by $+44.1\%$ (compared to AI Scientist).
- **ON metric is positively correlated with human judgment**: Pearson correlation coefficient $r=0.52$, validating the effectiveness of the automated evaluation metric.
- **Team size has an optimal value (~8 people)**: Small teams are innovative but have limited vision, while large teams have a broad vision but easily fall into groupthink.
- **50% freshness is optimal**: Consistent with the findings in the Science of Science literature (Zeng et al., 2021) that "fresh teams produce more innovative research".
- **The capability of the Agent model has limited impact**: The novelty score difference between LLaMA3.1-8B and 70B is very small, indicating that the collaboration mechanism is more crucial than individual model capability.

## Highlights & Insights
- **First scientific idea generation system that builds agent roles with real-world data**: Scientist backgrounds, papers, and collaborative relationships are all derived from real databases, rather than fabricated personas in prompts. This fundamentally enhances the credibility of multi-agent collaboration experiments.
- **High consistency between collaboration mechanism experiments and Science of Science literature**: Experimental results regarding team size, freshness, and diversity align with empirical studies published in Nature/Science, demonstrating that LLM agent systems can replicate core dynamics of human scientific collaboration.
- **The Invitation Mechanism is a practical design innovation**: It allows agents to temporarily consult external experts without altering the team's structure, balancing diversity and stability.

## Limitations & Future Work
- **Only generates abstracts instead of full papers**: Evaluation is solely based on abstract novelty, without verifying the technical feasibility of the ideas.
- **Single team working in isolation**: Multiple teams compete on the same topic in real scientific research; the current system does not simulate this competitive dynamic.
- **Inherent LLM bias may favor mainstream directions**: Highly-cited papers dominate the training data, which may lead agents to lean toward conservative, incremental ideas.
- **High computational cost**: 8 agents $\times$ 5 discussion rounds $\times$ multi-step process requires a massive number of LLM API calls for a single generation.

## Related Work & Insights
- **vs AI Scientist (Lu et al., 2024)**: AI Scientist is a single-agent end-to-end system (idea $\rightarrow$ experiment $\rightarrow$ paper $\rightarrow$ review), while VirSci focuses purely on the idea generation stage but significantly enhances novelty through multi-agent collaboration.
- **vs HypoGen (Qi et al., 2024)**: HypoGen utilizes multi-agent systems but lacks dynamic teaming and cross-team communication; VirSci introduces teaming based on real collaboration networks and the Invitation Mechanism.
- **vs ResearchTown (Yu et al., 2024)**: ResearchTown employs synthetic profiles and collaboration networks, whereas VirSci insists on using real data, making its conclusions more transferable.

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout-dynamic-agent-elimination-for-multi-agent-collaboration.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] Multi-Prompting Decoder Helps Better Language Understanding](multi-prompting_decoder_helps_better_language_understanding.md)

</div>

<!-- RELATED:END -->
