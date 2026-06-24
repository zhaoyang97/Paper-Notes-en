---
title: >-
  [Paper Note] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents
description: >-
  [ACL 2025][systematic review] This paper proposes InsightAgent, a human-centric interactive multi-agent system that reduces the drafting time of medical systematic reviews from months to approximately 1.5 hours through semantic clustering partitioning, multi-agent parallel reading, and real-time user interaction, achieving 79.7% of human drafting quality.
tags:
  - "ACL 2025"
  - "systematic review"
  - "multi-agent"
  - "interactive AI"
  - "human-in-the-loop"
  - "evidence synthesis"
date: 2026-05-08
content_hash: 53f5c40bd10785f8
---

# Completing A Systematic Review in Hours instead of Months with Interactive AI Agents

**Conference**: ACL 2025  
**arXiv**: [2504.14822](https://arxiv.org/abs/2504.14822)  
**Code**: [https://github.com/OSU-NLP-Group/InsightAgent](https://github.com/OSU-NLP-Group/InsightAgent)  
**Area**: Others  
**Keywords**: systematic review, multi-agent, interactive AI, human-in-the-loop, evidence synthesis

## TL;DR
This paper proposes InsightAgent, a human-centric interactive multi-agent system that reduces the drafting time of medical systematic reviews from months to approximately 1.5 hours through semantic clustering partitioning, multi-agent parallel reading, and real-time user interaction, achieving 79.7% of human drafting quality.

## Background & Motivation

**Background**: Systematic Reviews (SRs) are the cornerstone of evidence-based practice in high-risk areas such as medicine, with annual publications on PubMed growing from fewer than 50 in the 1990s to nearly 36,000 in 2022. Existing LLM-based automation methods (such as ChatCite and AutoSurvey) are primarily fully automated literature survey systems.

**Limitations of Prior Work**: Traditional systematic reviews are extremely time-consuming, taking months to complete. Existing LLM methods suffer from three main issues: (1) low record screening precision (only ~20%), (2) generic content in the generated reviews with untraceable sources, and (3) lack of real-time intervention mechanisms for domain experts, making them unsuitable for rigorous systematic review standards.

**Key Challenge**: Fully autonomous AI agents lack domain knowledge guidance, leading to inaccurate screening and insufficient synthesis, while purely manual reviews take too much time. How can the process be significantly accelerated while maintaining high quality?

**Goal**: To design a human-AI collaborative systematic review framework that allows clinical experts to monitor and guide AI agents' literature reading and synthesis processes in real-time.

**Key Insight**: Taking inspiration from the "multi-reviewer strategy" in systematic review methodology, the literature database is semantically partitioned and assigned to multiple agents for parallel processing, with an intuitive visual interface provided for real-time user intervention.

**Core Idea**: Achieve highly efficient and high-quality automation of systematic reviews through semantic clustering partitioning, multi-agent parallel reading, and visual human-computer interaction.

## Method

### Overall Architecture
A three-stage pipeline: Stage 1 (corpus mapping and partitioning) $\rightarrow$ Stage 2 (multi-agent parallel reading and evidence synthesis with user interaction support) $\rightarrow$ Stage 3 (final synthesis). The inputs are a literature corpus and research questions, and the output is a complete systematic review report.

### Key Designs

1. **Corpus Mapping and Partitioning (Stage 1)**:

    - **Function**: Projects a large-scale literature corpus into a 2D visual space and automatically partitions it.
    - **Mechanism**: Uses a Radial Relevance-Similarity (RSS) Map for visualization—papers more relevant to the research questions are placed near the center, and semantically similar papers are clustered together. Then, K-means clustering (with the Elbow method automatically selecting K, averaging 9 clusters) is used to partition the literature.
    - **Design Motivation**: Reduces the workload and noise interference for a single agent while providing users with a global view. Experiments demonstrate that compared to a single-agent setup, the multi-agent partitioned design leads to significantly higher review quality.

2. **Multi-Agent Parallel Reading and Evidence Synthesis (Stage 2)**:

    - **Function**: Each agent is assigned a literature cluster, explores outward starting from the most relevant articles, screens relevant literature, and incrementally generates a synthesized summary.
    - **Mechanism**: Starting from the center of the RSS Map, in each step, the agent selects and reads the most relevant article among the 8 nearest neighbors. It generates individual notes for each relevant article, merging them with existing memory when overlapping or contradictory info is encountered: $M_{k+1} = f(M_k, S_j)$. All merge operations are recorded in a provenance tree to ensure every conclusion is traceable. The memories of individual agents are isolated from each other until the final synthesis stage.
    - **Design Motivation**: Inspired by the multi-reviewer strategy in systematic reviews—allocating different subsets to different reviewers reduces individual bias and accelerates initial screening. Incremental synthesis avoids redundancy and gradually builds a coherent knowledge base.

3. **User Interaction Mechanism**:

    - **Function**: Provides three real-time interaction modes for domain experts to guide the agents.
    - **Mechanism**: (1) **Path Navigation**: Users drag agent pointers on the RSS Map to missed articles. (2) **Chat Navigation**: Adjust agent strategies using natural language instructions (e.g., "focus on randomized controlled trials"). (3) **Instruct Navigation**: Directly modify agent parameters (e.g., stricter inclusion criteria). After each interaction, the agent enters a reflection phase to reconcile memory conflicts and adjust its strategy.
    - **Design Motivation**: Fully autonomous agents cannot replace the knowledge of domain experts. Interaction enables experts to correct agent errors and inject domain insights. Experiments show that interaction improves the F1 score of article identification by 47%.

4. **Final Synthesis (Stage 3)**:

    - **Function**: Integrates the local evidence bases of individual agents into a coherent systematic review report.
    - **Mechanism**: Generates the final report based on a user-specified template (Introduction, Study Design, Key Findings, Discussion, Conclusion), utilizing citation numbers to link back to the original literature and intermediate summaries, and updating the provenance tree to ensure evidence traceability.

## Key Experimental Results

### Main Results

| System | Record Screening F1 (%) | Review Quality (Out of 100) |
|------|-------------|-----------------|
| BM25 (Top-100) | 25.3 | - |
| ChatCite (GPT-4) | - | 47.1 |
| AutoSurvey (GPT-4o) | 31.6 | 54.0 |
| InsightAgent_auto (Llama 3.3) | 64.3 | 60.9 |
| InsightAgent_auto (GPT-4o) | 60.0 | 62.4 |
| InsightAgent (Llama 3.3) | 83.8 | 70.2 |
| **InsightAgent (GPT-4o)** | **88.2** | **79.7** |

**Key Results**: InsightAgent (GPT-4o) achieves a record screening recall of 98.5% and a review quality score of 79.7 (79.7% of the 100-point human-written baseline), requiring an average of only around 1.5 hours of user time to complete.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No interaction (auto) vs. With interaction | +27.2% Quality | $p = 3.43 \times 10^{-7}$, statistically significant |
| Single agent vs. Multi-agent | Multi-agent is significantly better | Reduces noise and workload |
| Path Navigation Contribution | Largest improvement in comprehensiveness and accuracy | Helps agents find missed articles |
| Chat Navigation Contribution | Largest improvement in research depth | Guides agents to focus on specific aspects |
| Instruct Navigation Contribution | Largest improvement in writing quality | Fine-grained control over generation format |

### Key Findings
- **Interaction frequency is positively correlated with trust**: The more interaction the user has, the higher the trust in the system (the regression analysis in Figure 3 clearly illustrates this trend).
- **Weak model + good framework > strong model + weak framework**: InsightAgent_auto (Llama 3.3 70B) outperforms AutoSurvey (GPT-4o).
- **GPT-4o is better at collaboration than Llama**: When using GPT-4o as the backbone, the quality improvement brought by user interaction is larger (+17.3 vs. +9.3).
- **User Satisfaction**: Under the interactive mode, overall satisfaction increases by 34.4%, and "confidence in system recommendations" increases from 2.8/5 to 4.5/5.

## Highlights & Insights
- **The design of the Provenance Tree is highly ingenious**: leaf nodes represent article-level summaries, while intermediate nodes represent incremental syntheses, with different colors identifying the contributions of different agents. In high-risk fields (such as medicine), this traceability is a key mechanism for establishing trust. It can be transferred to other scenarios requiring evidence traceability, such as law and finance.
- **RSS Map visualization and agent trajectory display**: Visualizing the "thinking process" of agents allows users to not only see the results but also understand the underlying process. This provides important inspiration for the interpretability of agent systems.
- **Hierarchical design of three interaction modes**: Ranging from coarse-grained (Path) to medium-grained (Chat) to fine-grained (Instruct), it covers diverse user needs and serves as an excellent paradigm for human-AI collaborative interface design.

## Limitations & Future Work
- Only article titles and abstracts are utilized (restricted by LLM context window limits) without using the full text, which may omit critical details.
- The evaluated 15 systematic reviews are entirely from the biomedical field; validation of applicability to other domains (e.g., social sciences, engineering) is still needed.
- The quality evaluation of the reviews relies on manual scoring (by 2 evaluators); inter-rater reliability and subjective bias remain potential issues.
- Although a 1.5-hour user investment is substantially less than several months, domain experts are still required, and professional expertise cannot be entirely replaced.

## Related Work & Insights
- **vs. AutoSurvey**: AutoSurvey retrieves the top 100 first and then automatically generates reviews, resulting in low precision and lacking user interaction. In contrast, InsightAgent substantially improves precision through semantic partitioning and a multi-agent design, widening the gap further through user interaction.
- **vs. ChatCite**: ChatCite performs incremental reflective summarization but does not handle retrieval, relying on users to provide the article set. InsightAgent integrates the entire process of both retrieval and synthesis.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic review framework combining semantic visualization, multi-agents, and human-computer interaction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation including 15 systematic reviews and a user study with 9 medical experts.
- Writing Quality: ⭐⭐⭐⭐ Clear workflow with rich illustrations.
- Value: ⭐⭐⭐⭐⭐ Extremely high application value by successfully accelerating systematic reviews from months to 1.5 hours, achieving 79.7% of human quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [\[NeurIPS 2025\] Evaluating In Silico Creativity: An Expert Review of AI Chess Compositions](../../NeurIPS2025/others/evaluating_in_silico_creativity_an_expert_review_of_ai_chess_compositions.md)
- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->
