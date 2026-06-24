---
title: >-
  [Paper Note] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery
description: >-
  [ACL 2025][Scientific hypothesis generation] Proposes IRIS, an open-source interactive research ideation system that achieves human-machine collaborative scientific hypothesis generation through Monte Carlo Tree Search (MCTS) for test-time compute scaling, fine-grained feedback mechanisms, and query-based literature synthesis.
tags:
  - "ACL 2025"
  - "Scientific hypothesis generation"
  - "Human-in-the-Loop"
  - "MCTS"
  - "Research ideation"
  - "LLM-assisted discovery"
date: 2026-05-08
content_hash: 1f42c5a3eba30946
---

# IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery

**Conference**: ACL 2025  
**arXiv**: [2504.16728](https://arxiv.org/abs/2504.16728)  
**Code**: Yes (Open-source platform)  
**Area**: Other  
**Keywords**: Scientific hypothesis generation, Human-in-the-Loop, MCTS, Research ideation, LLM-assisted discovery

## TL;DR

Proposes IRIS, an open-source interactive research ideation system that achieves human-machine collaborative scientific hypothesis generation through Monte Carlo Tree Search (MCTS) for test-time compute scaling, fine-grained feedback mechanisms, and query-based literature synthesis.

## Background & Motivation

LLMs have demonstrated great potential in automating scientific discovery, particularly in hypothesis generation—the very initial stage of research. However, existing methods suffer from the following key issues:

**Lack of Human Intervention**: Most methods (such as AI-Researcher, ResearchAgent) rely on multi-agent frameworks or scaling test-time compute, but are essentially fully autonomous, failing to effectively integrate human oversight.

**Alignment Issues**: Allocating substantial computation to generate "objectively novel" ideas might result in directions that do not align with the user's research goals.

**"Reward Hacking" Behavior**: LLMs might use fancy terminology (e.g., "Prompt Learning and Optimization Nexus") or groundlessly propose "graph" structures to achieve high scores; recursive feedback loops that force LLMs to become "more novel" are in fact merely gaming the evaluation metrics.

**AI Safety Concerns**: Models may hallucinate scientific information or even exhibit deceptive behaviors.

**Limitations of Prior Work include**:
   - Generating hypotheses in a single pass, ignoring the iterative nature of research.
   - Coarse-grained feedback (overall scoring rather than targeting specific sections).
   - Naive retrieval-augmented generation (merely appending keywords or abstracts).
   - Unstructured search of the idea space.

## Method

### Overall Architecture

IRIS takes research objectives $\mathcal{G}$ (problems + motivations) as input and outputs a research brief $\mathcal{B}$ (title + methodology + experimental plans). The system supports two modes: semi-automatic (human-guided) and fully automatic (MCTS autonomous exploration).

### Key Designs

#### 1. Three-Agent Architecture

**Ideation Agent**:
- Generates and iteratively refines research briefs.
- Can switch between semi-automatic mode (guided by human researchers) and fully automatic mode (driven by MCTS).

**Review Agent**:
- Responsible for two tasks: providing reward scores and feedback.
- Defines a hierarchical evaluation taxonomy (based on real scientific review criteria).
- **Key Innovation — Fine-Grained Feedback**: Instead of evaluating the entire brief, it provides actionable feedback for *specific aspects* of *specific parts* of the brief.
- Human researchers validate the feedback and remove irrelevant parts, thereby mitigating "reward hacking."

**Retrieval Agent**:
- Generates queries targeted at the research objectives.
- Employs the Ai2 Scholar QA API (Semantic Scholar, 200M+ papers).
- Two-stage retrieval + three-stage generation: passage retrieval $\to$ reranking $\to$ citation extraction $\to$ section planning $\to$ citation report generation.
- Supports researchers uploading PDFs to supplement missing literature.

#### 2. MCTS for Hypothesis Generation

- **Function**: Systematically explores the vast research idea space.
- **State Definition**: $s = \{ \text{research brief } b, \text{reward } r, \text{latest feedback } f, \text{retrieved knowledge } k \}$
- **Action Space** $\mathcal{A} = \{ \text{generate, refinement based on retrieval, refinement based on review, refinement based on user feedback} \}$
- **UCT Selection Policy**:
$$\text{UCT}(n) = \frac{Q(n)}{N(n)} + c\sqrt{\frac{\ln N(n_p)}{N(n)}}$$
  where $c$ is the exploration constant (decrease $c$ to favor exploitation when the budget is tight).
- **Four-Stage Iteration**: Selection $\to$ Evaluation $\to$ Expansion $\to$ Backpropagation.
- **Design Motivation**: Unlike math/code (where rewards are objective), the quality of scientific ideation is subjective. Therefore, scores from the Review Agent are used as proxy rewards.
- **Memory Mechanism**: Each agent maintains trajectory-level memory to avoid redundant generation.

#### 3. Human-AI Co-Design Principles

- Draws on design principles from Amershi et al. (2019) and Shneiderman (2020).
- Minimizes opacity: The MCTS tree interface provides visual control.
- Fine-grained feedback instead of general, vague scores.
- Maintains human oversight during planning, generation, and review stages.

### Experimental Setup

- **LLM Backbone**: Gemini-2.0-Flash (via LiteLLM)
- **Evaluation Metrics**:
    - Absolute rating: 1-10 points for each hypothesis.
    - Relative rating: Head-to-head comparison to compute ELO rating.
- **User Study**: 8 researchers (AI/NLP, Chemistry, Physics, HCI), 10 case studies, approximately 60 minutes each.

## Key Experimental Results

### Automated Evaluation (Figure 3)

| Metric | Depth 0 → Depth 3 | Gain |
|------|------------------|------|
| Absolute Rating | ~6.5 → ~7.0 | +0.5 points |
| ELO Rating | ~990 → ~1002 | +12 points |

User interaction consistently improved hypothesis quality, which scaled with interaction depth.

### User Study Ratings (Table 1)

| Feature/Aspect | Average Rating (1-5 Likert) |
|----------|---------------------|
| Usefulness of Fine-Grained Feedback | 4.3 ± 0.7 |
| MCTS Tree Interface (Controllability) | 4.2 ± 0.6 |
| Quality of Literature Synthesis | 3.7 ± 0.8 |
| Usability and Sense of Control | **4.5 ± 0.7** |
| Overall Satisfaction | 3.9 ± 0.7 |

### Qualitative Findings

| Dimension | Proportion | Details |
|------|------|------|
| Controllability | 100% (8/8) | All users valued the control and transparency offered by the MCTS tree. |
| Feedback Resonance | 87.5% (7/8) | Review feedback often aligned with the users' own concerns. |
| Novel Insights | 50% (5/10) | Feedback occasionally sparked new ideas. |
| Relevance | 62.5% (5/8) | Hypotheses were connected to the ongoing work of the users. |

### Key Findings

1. **Interaction Improves Quality**: Hypotheses with user involvement achieved higher quality than those generated purely automatically.
2. **Elo is More Reliable than Absolute Rating**: Elo correlation with human preferences yielded Pearson $r=0.60$, whereas absolute rating was only $r=0.45$.
3. **Literature Retrieval Quality Varies by Field**: Performance was better in AI/NLP (3.7/5) and poorer in Chemistry/Physics, limited by the corpus coverage of Semantic Scholar.
4. **Usability Received the Highest Rating** (4.5/5) — indicating that human-AI collaborative design indeed delivers a superior user experience over fully automated solutions.
5. **25% of Users Felt Hypotheses Were "Significantly Better"**, 50% "Slightly Improved", and 100% agreed that it enhanced their understanding of the methodology.

## Highlights & Insights

- **Applying MCTS to scientific ideation** is a key novelty — using search tree structures to balance exploration and exploitation makes the process more systematic than linear refinement.
- **Fine-grained, human-validated review feedback** effectively addresses the "reward hacking" problem, which is a major pain point for fully autonomous systems.
- **Open-source implementation** lowers the barrier to entry for the academic community.
- **Focus on alignment issues** is forward-looking — pointing out the issues of "smart plagiarism" and superficial packaging by LLMs in scientific ideation.

## Limitations & Future Work

- Relies on researchers as evaluators, assuming they possess sufficient domain expertise.
- Due to budget constraints, stronger LLMs (e.g., Claude 3.7, o1, Gemini-2.5-Pro) were not used.
- The user study scale is small ($N=8$), so statistical significance of findings is limited.
- Literature retrieval relies on Semantic Scholar, which has insufficient coverage for fields like Chemistry/Physics.
- Has not validated the actual feasibility of the generated hypotheses (i.e., whether they can yield valid experiments).
- MCTS is computationally intensive and requires budget control.

## Related Work & Insights

- AI-Researcher (Si et al., 2024): Fully automated but found to suffer from "smart plagiarism" issues.
- ResearchAgent (Baek et al., 2025): Coarse-grained feedback, recursive refinement leads to reward gaming.
- Acceleron (Nigam et al., 2024): Early HITL attempt but lacks flexibility.
- OpenScholar (Asai et al., 2024): An advanced system for literature synthesis.
- Insights: Future work could establish a true "two-way Socratic" dialogue — where the AI questions the researcher's choices and the researcher validates the AI's suggestions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of MCTS + HITL + fine-grained feedback is a fresh approach in scientific ideation.
- **Experimental Thoroughness**: ⭐⭐⭐ — The user study is small-scale ($N=8$), automated evaluation improvements are limited (+0.5 / +12), and comparisons with strong baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is thoroughly explained, system description is detailed, and safety discussions are deep.
- **Value**: ⭐⭐⭐⭐ — The open-source platform brings practical value to the academic community, and the human-AI collaborative design philosophy serves as a valuable exemplar.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](completing_a_systematic_review_in_hours.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)

</div>

<!-- RELATED:END -->
