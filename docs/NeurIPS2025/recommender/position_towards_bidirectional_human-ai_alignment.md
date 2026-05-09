---
title: >-
  [Paper Note] Position: Towards Bidirectional Human-AI Alignment
description: >-
  [NeurIPS 2025][Recommender Systems][AI alignment] This paper proposes a **Bidirectional Human-AI Alignment** framework grounded in a systematic review of 400+ papers, arguing that AI alignment should not be limited to the unidirectional goal of "aligning AI with humans," but must also encompass the critically underexplored direction of "aligning humans with AI," while identifying key gaps in the current research landscape.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - AI alignment
  - bidirectional alignment
  - human values
  - human-AI interaction
  - systematic review
date: 2026-05-08
content_hash: 46892a3b1c35e217
---

# Position: Towards Bidirectional Human-AI Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2406.09264](https://arxiv.org/abs/2406.09264)
**Code**: None
**Area**: AI Alignment / Human-AI Interaction
**Keywords**: AI alignment, bidirectional alignment, human values, human-AI interaction, systematic review

## TL;DR

This paper proposes a **Bidirectional Human-AI Alignment** framework grounded in a systematic review of 400+ papers, arguing that AI alignment should not be limited to the unidirectional goal of "aligning AI with humans," but must also encompass the critically underexplored direction of "aligning humans with AI," while identifying key gaps in the current research landscape.

## Background & Motivation

**State of the Field**: AI alignment has become a central topic in AI safety and ethics. Conventionally, alignment research focuses on making AI systems behave in accordance with human goals and values—exemplified by methods such as RLHF and Constitutional AI. However, as AI systems become more deeply embedded in everyday life, a unidirectional alignment perspective is no longer sufficient.

**Limitations of Prior Work**:
- **Ambiguous definition of alignment**: Different communities (HCI, NLP, ML) hold inconsistent definitions of "alignment," lacking a shared conceptual foundation.
- **Unidirectional perspective**: Existing work almost exclusively addresses the "AI → Human" direction, neglecting how human cognition and behavior adapt to AI.
- **Insufficient value modeling**: The plurality, dynamism, and context-dependence of human values are overly simplified in current alignment methods.

**Root Cause**: AI systems are growing increasingly complex and autonomous, yet human capacity to understand, oversee, and collaborate with them has not kept pace. A **dynamic feedback loop** exists between AI and humans—AI behavior shapes human responses, which in turn reshape AI behavior—yet this bidirectional interaction is almost entirely absent from existing research.

**Paper Goals**: To rigorously define the conceptual boundaries of "alignment," propose a systematic framework encompassing bidirectional interaction, identify existing research gaps, and provide a roadmap for future alignment research.

**Starting Point**: A **systematic review of 400+ papers** spanning HCI, NLP, and ML, combined with qualitative coding and quantitative analysis, to construct a comprehensive taxonomy covering both the AI side and the human side.

**Core Idea**: AI alignment should be bidirectional—not only aligning AI with human values, but also helping humans understand, evaluate, and adapt to AI.

## Method

### Overall Architecture

The proposed **Bidirectional Human-AI Alignment** framework comprises two interconnected directions:

1. **Align AI with Humans**: Incorporating human values and norms into AI training, steering, and customization.
2. **Align Humans with AI**: Supporting human cognitive, behavioral, and societal adaptation to rapidly evolving AI technologies.

The framework is organized around four core research questions (RQ1–RQ4):

| Direction | Research Question | Focus |
|-----------|-------------------|-------|
| AI → Human | RQ1: Human values and norms | Which values are aligned? How are values specified interactively? |
| AI → Human | RQ2: Integrating human norms into AI | How are values embedded throughout the development and deployment pipeline? |
| Human → AI | RQ3: Human cognitive adaptation | How do humans learn to perceive, understand, and critically evaluate AI? |
| Human → AI | RQ4: Human behavioral adaptation | How do individuals and society collaborate with AI and respond to its impacts? |

### Key Designs

#### RQ1: Human Values and Norms

- **Value taxonomy**: An adaptation of Schwartz's Theory of Basic Values, analyzed along two dimensions:
    - **Sources**: Individual values (e.g., factuality, cognitive bias), social values (e.g., fairness, ethics), and interaction values (e.g., usability, trust).
    - **Types**: Self-enhancement, self-transcendence, conservation, and openness to change.
- **Interactive techniques for value specification**:
    - Explicit feedback: principles, ratings, natural language, multimodal input.
    - Implicit feedback: behavioral cues, linguistic patterns, theory of mind.
    - Simulated feedback: feedback simulators, synthetic data.

#### RQ2: Integrating Human Values into AI

- **General value integration**: Via instruction data (human-annotated / human-AI co-annotated / simulated), model learning (online/offline alignment), and inference stage (prompting / tools / search).
- **Personalized/group-level customization**: Customization data, adaptive learning (group-level learning, MoE, adapters), and interactive alignment.
- **Evaluation frameworks**: Human-in-the-loop evaluation vs. automated evaluation.

#### RQ3: Human Cognitive Adaptation

- **Perceiving and understanding AI**: AI literacy education, explainable AI visualizations, interactive explanation techniques.
- **Critical thinking**: Trust and reliance calibration, ethical auditing, cognitive recalibration.

#### RQ4: Human Behavioral Adaptation

- **Human-AI collaboration modes**: Collaboration mechanisms under three AI roles—assistant, partner, and tutor.
- **AI impacts on individuals and society**: Individual behavioral change, shifts in social relationships, and institutional responses to AI advancement.
- **Evaluation methods**: Micro-level (human-AI collaboration assessment) and macro-level (societal impact assessment).

### Systematic Review Methodology

- Systematic literature review following PRISMA guidelines.
- Initial retrieval of 34,213 papers → keyword filtering to 2,136 → inclusion criteria filtering to 411 papers.
- Independent dual coding with a joint agreement rate of 0.78.
- Qualitative coding using a combined inductive and deductive approach to develop the analytical framework.

## Key Experimental Results

### Analysis of Literature Distribution

Quantitative statistics reveal significant imbalances in research coverage across dimensions:

| Dimension | Publication Trend | Key Finding |
|-----------|-------------------|-------------|
| Explicit human feedback | Highest | Dominates value specification research |
| Implicit/simulated feedback | Very few | Severely underexplored, yet high potential |
| Model training stage | Many | Online/offline alignment research concentrated here |
| Inference-stage alignment | Few | Real-time adaptation capacity neglected |
| AI literacy education | Very few | One of the largest research gaps on the human side |
| Collaboration when AI surpasses human capability | Near absent | Existing research assumes AI remains in an assistive role |
| Societal-level impact assessment | Insufficient | Long-term behavioral changes lack longitudinal tracking |

### Key Research Gaps

| Direction | Gap | Severity |
|-----------|-----|----------|
| AI → Human | Implicit/simulated value feedback | High |
| AI → Human | Inference-stage customization and interactive alignment | High |
| AI → Human | Standardization of human-in-the-loop evaluation | Medium |
| Human → AI | AI literacy and education | Very High |
| Human → AI | Collaboration with superhuman-capability AI | Very High |
| Human → AI | AI ethical auditing (from the human perspective) | High |
| Human → AI | Long-term societal impact assessment | High |

### Key Findings

1. **Research is heavily skewed toward the AI side**: The vast majority of alignment research focuses on "aligning AI with humans"; work on "aligning humans with AI" is severely underrepresented.
2. **Value specification methods are narrow**: There is an excessive reliance on explicit feedback (ratings, rankings, instructions), while implicit behavioral signals and simulated feedback are largely ignored.
3. **Lack of dynamic perspective**: Existing work treats alignment as a static process, with limited attention to long-term interaction design and value evolution modeling.
4. **Human-side research centers on explainability**: The primary focus is on how XAI helps humans understand model decisions, while dimensions such as AI literacy, critical thinking, and ethical auditing remain neglected.

## Highlights & Insights

1. **Conceptual breakthrough**: The paper is the first to explicitly position "Align Humans with AI" as an equal and complementary direction in alignment research, challenging the entrenched unidirectional paradigm.
2. **Exceptional systematicity**: A cross-disciplinary review spanning HCI, NLP, and ML with 400+ papers, featuring a detailed taxonomy and well-grounded quantitative analysis.
3. **Three long-term challenges are precisely defined**:
    - **Specification Game**: How can complex human values be fully specified?
    - **Dynamic Co-evolution**: How do humans, AI, and society co-evolve over time?
    - **Safeguarding Co-adaptation**: How can the co-adaptation process be kept safe?
4. **Introduction of Schwartz's value theory** provides a psychological and sociological theoretical anchor for AI alignment research.
5. **Practical roadmap**: Each identified research gap is accompanied by specific suggestions for future research directions.

## Limitations & Future Work

1. **Limited review scope**: Coverage is primarily restricted to computation-adjacent fields (ML/NLP/HCI); cognitive science, psychology, and science and technology studies (STS) are insufficiently represented.
2. **Temporal window**: Coverage is mainly limited to 2019–2024, potentially omitting foundational earlier work.
3. **Lack of empirical validation**: As a position paper, the proposed framework lacks experimental validation of its effectiveness.
4. **Insufficient treatment of value conflicts**: Although value plurality and social choice theory are mentioned, the paper does not deeply discuss concrete mechanisms for resolving conflicts between the values of different groups.
5. **Unclear technical implementation pathways**: While many research gaps are identified, the discussion of specific technical solutions remains relatively shallow.

## Related Work & Insights

- **Relation to traditional AI alignment research**: Methods such as RLHF (Ouyang et al., 2022) and Constitutional AI (Bai et al., 2022) constitute a subset of the "AI → Human" direction within the proposed framework.
- **Connection to HCI**: The framework integrates HCI research on XAI, human-AI collaboration, and AI literacy under an alignment-oriented lens.
- **Inspiration from social choice theory**: Arrow's social choice theory can provide formal tools for aggregating pluralistic values.
- **Connection to Scalable Oversight**: The "Human → AI" direction directly speaks to the challenge of maintaining human oversight as AI systems scale.

**Implications for future research**:
- Alignment systems capable of capturing implicit value signals are worth developing.
- AI literacy education should become foundational infrastructure for alignment research.
- Longitudinal research infrastructure is needed to track the long-term co-evolution of humans and AI.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of "bidirectional alignment" represents an important conceptual breakthrough, though no technical innovations are introduced given the position paper format.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The systematic review of 400+ papers is highly rigorous, and the quantitative analysis effectively reveals research biases.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, rigorous taxonomy, and excellent readability; exemplary as a survey paper.
- **Value**: ⭐⭐⭐⭐⭐ — Offers a novel conceptual framework and systematic roadmap for AI alignment research, with significant guiding value for the community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)
- [\[NeurIPS 2025\] The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process](the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)
- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](../../AAAI2026/recommender/moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](neurips_should_lead_scientific_consensus_on_ai_policy.md)
- [\[NeurIPS 2025\] The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems](the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems.md)

<!-- RELATED:END -->
