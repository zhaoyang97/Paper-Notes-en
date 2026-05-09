---
title: >-
  [Paper Note] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes
description: >-
  [AAAI 2026][Medical Imaging][Depression detection] This paper proposes MAMAMemeia, a multi-agent multi-aspect collaborative discussion framework grounded in the Cognitive Analytic Therapy (CAT) competency framework, designed to identify depressive symptoms from social media memes. It additionally introduces the RESTOREx resource (containing both LLM-generated and human-annotated rationales), achieving a 7.55% improvement in macro-F1 over 30+ competing methods.
tags:
  - AAAI 2026
  - Medical Imaging
  - Depression detection
  - meme analysis
  - multi-agent collaboration
  - Cognitive Analytic Therapy
  - large language models
date: 2026-05-08
content_hash: 28c8b5d0c89f937a
---

# MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes

**Conference**: AAAI 2026
**arXiv**: [2512.25015](https://arxiv.org/abs/2512.25015)
**Code**: N/A
**Area**: Medical Imaging / NLP
**Keywords**: Depression detection, meme analysis, multi-agent collaboration, Cognitive Analytic Therapy, large language models

## TL;DR

This paper proposes MAMAMemeia, a multi-agent multi-aspect collaborative discussion framework grounded in the Cognitive Analytic Therapy (CAT) competency framework, designed to identify depressive symptoms from social media memes. It additionally introduces the RESTOREx resource (containing both LLM-generated and human-annotated rationales), achieving a 7.55% improvement in macro-F1 over 30+ competing methods.

## Background & Motivation

**Background**: Memes have evolved from purely humorous media into a vehicle through which users express a wide range of emotions, including depressive affect. The growing prevalence of depression-themed memes on social media presents a novel data source for mental health monitoring.

**Limitations of Prior Work**: (1) Memes are multimodal (image + text) and frequently employ rhetorical devices such as metaphor and irony, making it difficult for single models to grasp their deeper meaning. (2) Identifying depressive symptoms requires specialized psychological knowledge, as different memes may imply distinct symptoms (e.g., hopelessness, social withdrawal, self-deprecation). (3) Existing methods lack interpretability—black-box classification cannot inform clinicians *why* a meme is deemed to express depression.

**Key Challenge**: A fundamental gap exists between the implicit expressive nature of memes and the professional precision required for depressive symptom identification.

**Goal**: (1) Construct an interpretable depressive symptom detection framework; (2) integrate clinical psychology knowledge to guide analysis; (3) provide an annotated rationale resource for the community.

**Key Insight**: Drawing on the clinical methodology of Cognitive Analytic Therapy (CAT), different psychological analysis dimensions are assigned to distinct LLM agents, which reach a comprehensive judgment through multi-agent discussion.

**Core Idea**: Multiple LLM agents each assume a distinct CAT competency role (e.g., affective analyst, cognitive assessor, behavioral observer) and collaboratively identify depressive symptoms in memes through structured discussion.

## Method

### Overall Architecture

MAMAMemeia takes a meme (image + text) as input. Multiple LLM agents independently analyze the meme from different CAT competency dimensions, then integrate their findings through a structured discussion/debate process, ultimately producing a depressive symptom classification label along with an explanatory rationale.

### Key Designs

1. **Multi-Aspect Agent Design**:

    - **Function**: Each agent focuses on one psychological analysis dimension.
    - **Mechanism**: Based on the CAT competency framework, several specialized agents are instantiated—e.g., an affective analysis agent (identifying emotional tone), a cognitive pattern agent (detecting distorted cognitions such as catastrophizing), and a social signal agent (identifying social withdrawal or help-seeking signals). Each agent is equipped with a dedicated system prompt and role description.
    - **Design Motivation**: Even a highly capable single LLM struggles to simultaneously attend to multiple analytical dimensions; specialization ensures each aspect receives adequate attention.

2. **Collaborative Discussion Mechanism**:

    - **Function**: Integrates the analyses of multiple agents to reach a consensus judgment.
    - **Mechanism**: Agents engage in multi-turn dialogue—first conducting independent analyses, then sharing their findings, discussing points of disagreement, and finally consolidating a comprehensive judgment. This process mirrors clinical team consultation.
    - **Design Motivation**: In clinical practice, mental health assessments are typically conducted collaboratively by multiple professionals from different perspectives; the multi-agent discussion faithfully simulates this collaborative process.

3. **RESTOREx Resource**:

    - **Function**: Provides a meme dataset with depressive symptom annotations and rationales.
    - **Mechanism**: Two-tier annotation is applied: (a) LLMs generate initial rationales explaining why a given meme expresses a particular depressive symptom; (b) human reviewers verify and correct these rationales to ensure clinical accuracy. Rationales address visual element interpretation, textual meaning analysis, and metaphor comprehension.
    - **Design Motivation**: Interpretable training data is foundational infrastructure for developing trustworthy mental health AI systems.

### Loss & Training

MAMAMemeia operates via LLM in-context learning or fine-tuning and does not require conventional loss function design. The framework's functionality is realized primarily through carefully engineered prompts and agent interaction protocols.

## Key Experimental Results

### Main Results

| Method | Macro-F1 | # Baselines | Notes |
|--------|----------|-------------|-------|
| MAMAMemeia | Best (+7.55%) | 30+ | New SOTA |
| Best single-model baseline | 2nd | -- | Lacks multi-dimensional analysis |
| Naive LLM prompting | Moderate | -- | No structured analysis |

### Ablation Study

| Configuration | Macro-F1 | Notes |
|---------------|----------|-------|
| Full multi-agent | Best | All CAT dimensions |
| Single agent | Degraded | Lacks multi-dimensional perspective |
| No discussion (voting only) | Degraded | Lacks information integration |
| Without CAT framework | Degraded | Lacks clinical guidance |

### Key Findings

- Multi-agent collaboration yields substantial gains over single-agent baselines, validating the value of multi-dimensional analysis.
- The CAT competency framework provides LLMs with an effective analytical structure.
- The discussion mechanism outperforms simple voting, as inter-agent information exchange produces synergistic effects.

## Highlights & Insights

- **The integration of clinical psychology frameworks with AI** is the paper's most notable contribution—rather than applying LLMs blindly, it leverages a principled psychological methodology to guide agent design.
- **Interpretability** is critical for mental health applications; MAMAMemeia provides not only classification outputs but also detailed analytical rationales.
- The RESTOREx dataset offers long-term community value.

## Limitations & Future Work

- The cultural specificity of memes may limit cross-cultural generalization.
- The analytical quality of LLM agents is contingent on the capabilities of the underlying models.
- The context in which users post memes (e.g., temporal sequences, social networks) is not considered.
- Alignment with clinical standards (e.g., PHQ-9) could enhance clinical utility.

## Related Work & Insights

- **vs. Single-model multimodal classification**: Lacks separation of analytical dimensions and interpretability.
- **vs. Sentiment analysis methods**: Sentiment analysis operates at too coarse a granularity and does not distinguish among specific depressive symptom types.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of CAT framework and multi-agent collaboration is a novel formulation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Surpasses 30+ methods; RESTOREx dataset offers long-term value
- Writing Quality: ⭐⭐⭐⭐ Interdisciplinary background is clearly articulated
- Value: ⭐⭐⭐⭐ Makes an important contribution to mental health AI applications

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/medical_imaging/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](../../NeurIPS2025/medical_imaging/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[AAAI 2026\] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)

<!-- RELATED:END -->
