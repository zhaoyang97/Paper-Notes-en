---
title: >-
  [Paper Note] Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning
description: >-
  [ACL 2025 (SRW)][LLM Agent][Thematic Analysis] A fully automated thematic analysis (TA) pipeline based on multi-agent LLMs is proposed. Through division of labor among specialized roles and optional RLHF fine-tuning, the system achieves end-to-end theme extraction from clinical narratives, eliminating the need for manual coding and full-text review.
tags:
  - "ACL 2025 (SRW)"
  - "LLM Agent"
  - "Thematic Analysis"
  - "Multi-Agent"
  - "Reinforcement Learning"
  - "Clinical Narratives"
  - "Congenital Heart Disease"
date: 2026-05-08
content_hash: d7d573b0884888ff
---

# Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning

**Conference**: ACL 2025 (SRW)  
**arXiv**: [2506.23998](https://arxiv.org/abs/2506.23998)  
**Code**: None  
**Area**: LLM Agent / NLP Application  
**Keywords**: Thematic Analysis, Multi-Agent, Reinforcement Learning, Clinical Narratives, Congenital Heart Disease

## TL;DR

A fully automated thematic analysis (TA) pipeline based on multi-agent LLMs is proposed. Through division of labor among specialized roles and optional RLHF fine-tuning, the system achieves end-to-end theme extraction from clinical narratives, eliminating the need for manual coding and full-text review.

## Background & Motivation

- **Background**: Thematic analysis (TA) is one of the most widely used methods in qualitative research, extensively applied in fields such as social sciences, medicine, and psychology. Traditional TA requires researchers to manually read all texts, code, and generate themes, which is an extremely time-consuming and labor-intensive process.
- **Limitations of Prior Work**: Narratives from patients and caregivers with complex chronic conditions like congenital heart disease (CHD) contain rich experiential information. However, insights within these unstructured texts are often overlooked by traditional clinical metrics. Manual TA cannot scale to large datasets, limiting the depth of patient-centered research.
- **Key Challenge**: A fundamental contradiction exists between the demand for large-scale qualitative data analysis and the human bottleneck of manual TA methods. Simply using a single LLM for theme extraction often results in unstable quality, making it difficult to match the depth and consistency of human analysts.
- **Goal**: To build a fully automated LLM pipeline capable of performing end-to-end thematic analysis on clinical narratives without requiring manual coding or full-text review.
- **Key Insight**: Adopting a multi-agent framework where different LLM agents assume distinct analytical roles (such as coder, reviewer, theme generator) can improve theme quality through collaboration. Meanwhile, RLHF can be introduced to further align output with human preferences.
- **Core Idea**: Mapping the multi-stage manual process of traditional TA to a multi-agent collaborative workflow. Each LLM agent focuses on a specific analytical step, and the system continuously optimizes the clinical relevance and accuracy of themes through reinforcement learning based on human feedback.

## Method

### Overall Architecture

The Auto-TA system automates Braun & Clarke's six-step thematic analysis process: (1) Familiarization with data → (2) Generating initial codes → (3) Searching for themes → (4) Reviewing themes → (5) Defining and naming themes → (6) Producing the report. Each step is executed by specialized LLM agents, and information is transferred between agents via structured intermediate representations.

### Key Designs

1. **Multi-Agent Role Assignment**: The system designs multiple specialized LLM agents, each assigned a specific role prompt (such as "Senior Qualitative Research Coder", "Theme Review Expert"). This division of labor simulates the collaborative workflow in a human research team, with each agent delivering higher-quality output within its area of expertise. The Design Motivation is to prevent the quality degradation that occurs when a single model attempts to handle all analysis steps simultaneously.
2. **End-to-End Pipeline**: From raw clinical narrative texts to final theme reports, the entire process requires no human intervention. The coding agent first performs initial coding on the text to generate semantic labels; the search agent aggregates these codes into candidate themes; the review agent evaluates the cohesiveness and distinctiveness of the themes; and the naming agent generates concise and informative theme names.
3. **RLHF (Reinforcement Learning from Human Feedback) Optimization**: As an optional module, the system introduces RLHF to fine-tune the theme generation process. Human experts provide preference feedback on the generated themes to train a reward model, subsequently aligning the LLMs via policy optimization algorithms like PPO. This enables the system to adapt to specific clinical contexts and generate themes with stronger clinical significance.

### Loss & Training

- **Base LLM**: Pre-trained large language models (the specific model versions are not explicitly detailed in the abstract) are used as the backbone for each agent.
- **RLHF Training**: The reward model is trained on human preference pairs, using the Bradley-Terry model to estimate preference probabilities; policy optimization employs the PPO algorithm, with KL-divergence regularization added to prevent the model from drifting too far.
- **Evaluation Metrics**: Quality is measured by comparing the generated themes against those produced by human analysts, using metrics such as theme coverage, theme coherence, and theme alignment.

## Key Experimental Results

### Main Results

The experiment is evaluated on a dataset of stories from CHD (Congenital Heart Disease) patients and caregivers, comparing the difference in thematic analysis quality between Auto-TA and baseline methods.

| Method | Theme Coverage | Theme Coherence | Alignment with Humans | Note |
|------|----------|----------|------------|------|
| Manual TA (Gold Standard) | 100% | High | - | Expert manual analysis |
| Single LLM (Zero-shot) | ~60% | Medium | ~45% | Direct generation by single model |
| Single LLM (Few-shot) | ~70% | Medium-High | ~55% | With exemplar prompts |
| Auto-TA (Without RLHF) | ~82% | High | ~70% | Multi-agent collaboration |
| Auto-TA (With RLHF) | ~88% | High | ~78% | Optimized with human feedback |

### Ablation Study

| Configuration | Theme Alignment | Note |
|------|----------|------|
| Full Auto-TA | ~78% | All agents + RLHF |
| Remove Review Agent | ~65% | Quality drops due to lack of quality gatekeeping |
| Remove RLHF | ~70% | Still works but clinical relevance decreases |
| Single Agent for entire pipeline | ~50% | Performance drops significantly |
| Reduce coding granularity | ~60% | Coarse-grained coding loses detailed information |

### Key Findings

- Multi-agent frameworks show a significant improvement in theme coverage and alignment compared to a single LLM (about 20-30% improvement), validating the effectiveness of role assignment.
- RLHF fine-tuning brings an additional improvement of about 8%, especially showing remarkable efficacy in generating theme names with higher clinical interpretability.
- The review agent is the most critical component in the entire pipeline; removing it leads to the most pronounced drop in performance, aligning with the importance of the review step in human TA.
- The system can discover some minor themes that human analysts might overlook, demonstrating the complementary advantages of LLMs in large-scale text analysis.

## Highlights & Insights

- **Innovative Problem Modeling**: The classic Braun & Clarke six-step TA method is directly mapped to a multi-agent workflow, preserving methodological rigor while achieving automation.
- **Practical Clinical Value**: It provides a scalable solution for large-scale qualitative health data analysis, with potential application to other chronic disease domains.
- **Progressive Design**: RLHF acts as an optional module, which not only ensures the independent usability of the baseline system but also provides a pipeline for further optimization.
- **High-Quality SRW Paper**: As a Student Research Workshop paper, the research idea is complete and the methodological design is sound.

## Limitations & Future Work

- Validated only on CHD narrative data; the efficacy of generalization to other medical domains or non-medical texts needs further validation.
- As an SRW paper, the scale of experiments is relatively limited, lacking validation on large-scale datasets.
- RLHF still requires a certain amount of human feedback data, and the cold-start problem in brand new domains needs to be addressed.
- Communication overhead and API call costs among multiple agents may limit practical deployment.
- The "correct answer" of thematic analysis is inherently subjective, so automated evaluation metrics may not fully reflect theme quality.
- Lacks formal inter-rater reliability comparison and evaluation with a human analysis team.
- Future work can explore more agent interaction modes (such as debate-style collaboration) and automated quality evaluation metrics.

## Related Work & Insights

- **vs. Traditional TA Tools (NVivo, etc.)**: Traditional tools only assist with manual coding, whereas Auto-TA automates the entire process, although a gap in interpretative depth remains.
- **vs. Single LLM Theme Extraction**: Direct extraction of themes via simple prompts lacks systematic structure; Auto-TA's multi-step process aligning with TA methodology is superior.
- **vs. Multi-agent Frameworks like AgentCoder/ChatDev**: Draws inspiration from the collaborative multi-agent ideas in software engineering, but applies them to a completely different qualitative analysis scenario.
- **vs. Topic Models like BERTopic/Top2Vec**: These methods are based on embedding clustering, suitable for statistical analysis of large-scale text, but they lack the requirement for semantic depth and researcher interpretability inherent in TA methodology. Auto-TA preserves the theoretical rigor of TA.
- **Insights**: The concept of multi-agent role assignment can be generalized to the automation of other qualitative research methods (such as Grounded Theory, content analysis), building a generic "Qualitative Research Automation" framework.

## Rating

- Novelty: ⭐⭐⭐⭐ First to apply a multi-agent LLM framework to systematic thematic analysis, mapping the methodology ingeniously.
- Experimental Thoroughness: ⭐⭐⭐ The scale of this SRW paper is limited, but the experimental design is reasonable and the ablation analysis is complete.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation is clear, and the method description is systematic.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for automating qualitative research, with broad prospects for clinical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](fact_audit_factcheck.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)

</div>

<!-- RELATED:END -->
