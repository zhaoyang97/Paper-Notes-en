---
title: >-
  [Paper Note] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection
description: >-
  [ACL 2025][LLM Agent][Harmful Meme Detection] This paper proposes the MIND framework, which achieves zero-shot harmful meme detection through three stages: similar sample retrieval, bidirectional insight derivation, and multi-agent debate. Without any labeled data, MIND outperforms existing zero-shot methods on three datasets and demonstrates strong generalization across different model architectures and scales.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Harmful Meme Detection"
  - "Zero-shot"
  - "Multi-agent Debate"
  - "Bidirectional Insight Derivation"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: c8c27cbe5f782b14
---

# MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection

**Conference**: ACL 2025  
**arXiv**: [2507.06908](https://arxiv.org/abs/2507.06908)  
**Code**: [https://github.com/destroy-lonely/MIND](https://github.com/destroy-lonely/MIND)  
**Area**: Agent / AI Safety / Multimodal Content Moderation  
**Keywords**: Harmful Meme Detection, Zero-shot, Multi-agent Debate, Bidirectional Insight Derivation, Retrieval-Augmented Generation

## TL;DR

This paper proposes the MIND framework, which achieves zero-shot harmful meme detection through three stages: similar sample retrieval, bidirectional insight derivation, and multi-agent debate. Without any labeled data, MIND outperforms existing zero-shot methods on three datasets and demonstrates strong generalization across different model architectures and scales.

## Background & Motivation

**Background**: Memes have surged on social media, with a large amount of harmful content spreading hate, discrimination, and misinformation. The multimodal nature of memes combining images and text makes their detection particularly difficult. Existing approaches primarily utilize data-driven multimodal models, relying heavily on a large volume of labeled data for training.

**Limitations of Prior Work**: The core challenge of harmful memes lies in their rapid evolution—new events constantly trigger new meme templates and expressions. Traditional data-driven methods struggle with newly emerging memes due to the extreme difficulty in quickly collecting and labeling sufficient training data. Even few-shot in-context learning (ICL) methods require pre-labeled exemplars, failing to truly adapt to the fast-evolving nature of memes.

**Key Challenge**: Harmful meme detection demands real-time responsiveness to newly emerging content, whereas supervised learning methods inherently lag behind content evolution—by the time labeled data is collected, the harm has already spread.

**Goal**: To develop a completely zero-shot harmful meme detection framework that does not rely on any labeled data, leveraging the reasoning capabilities of large multimodal models (LMMs) to address emerging harmful content.

**Key Insight**: Although memes evolve continuously, they often preserve recognizable core patterns. For instance, COVID-19-related memes featuring the White House press briefing have been modified and reused in various ways, while the core scene elements remain unchanged. Retrieving similar memes and deriving harmfulness insights from them can provide valuable references for judging new memes.

**Core Idea**: To retrieve similar memes from an unlabeled reference set, extract harmful analysis experience from these similar samples through bidirectional insight derivation, and utilize a multi-agent debate mechanism to integrate these insights for robust decision-making.

## Method

### Overall Architecture

MIND consists of three stages: (1) Similar Sample Retrieval (SSR) – retrieves $K$ samples most similar to the target meme from an unlabeled reference set; (2) Relevant Insight Derivation (RID) – extracts analytical insights regarding harmfulness from the similar samples in both forward and backward directions; (3) Insight-Augmented Inference (IAI) – multiple debating agents discuss based on the derived insights, and a judge agent makes the final decision.

### Key Designs

1. **Similar Sample Retrieval (SSR)**:

    - **Function**: Retrieves samples most relevant to the target meme from an unlabeled reference set.
    - **Mechanism**: For each meme $M = \{V, T\}$, a visual encoder and a text encoder are used to extract features, which are weighted and summed to obtain a multimodal embedding $E = \lambda_v \cdot V_{enc}(V) + \lambda_t \cdot T_{enc}(T)$. The cosine similarity between the target meme and all samples in the reference set is computed to select the top-$K$ most similar samples into a sample pool.
    - **Design Motivation**: Similar memes share core patterns. Accurate retrieval provides high-quality reference contexts for subsequent analysis, which is far more informative than randomly selected samples.

2. **Bidirectional Insight Derivation (RID)**:

    - **Function**: Systematically extracts harmfulness analytical insights from the retrieved similar samples.
    - **Mechanism**: Employs a two-directional derivation process—**Forward Insight**: The LMM processes similar samples sequentially according to the retrieval order, passing the analysis results of preceding samples to subsequent ones as context to form incremental insights; **Backward Insight**: Based on the forward derivation, the samples are processed in reverse order to complement and correct the preceding analyses using the insights from later samples. The insights from both directions are then fed into the IAI stage.
    - **Design Motivation**: Unidirectional derivation can suffer from order bias—earlier analyses might mislead subsequent evaluations. The bidirectional derivation ensures that each sample is fully analyzed through a "looking back" mechanism, resembling how a BiLSTM captures information from both directions.

3. **Insight-Augmented Inference (IAI)**:

    - **Function**: Makes a robust harmfulness judgment on the target meme based on the derived insights.
    - **Mechanism**: Sets up debater agents (LMM_debater) and a judge agent (LMM_judge). The forward and backward insights are input to different debaters, each of whom provides analysis and predictions combining the insights and the target meme. If the two debaters agree, the decision is adopted; if they conflict, the judge agent arbitrates by synthesizing arguments from both sides.
    - **Design Motivation**: A single inference path can be biased or incomplete. Multi-agent debate allows opposing viewpoints to collide, thereby considering all facets of harmfulness more comprehensively.

### Loss & Training

MIND is a completely gradient-free zero-shot framework and does not involve any model training. All modules directly utilize pre-trained LMMs (e.g., LLaVA-1.5-13B) for inference.

## Key Experimental Results

### Main Results

| Model | HarM Acc/F1 | FHM Acc/F1 | MAMI Acc/F1 |
|------|-------------|------------|-------------|
| GPT-4o | 62.07/60.29 | 63.20/63.15 | 73.50/73.49 |
| Gemini-1.5-Flash | 68.93/67.51 | **63.40/63.30** | 65.40/65.28 |
| LLaVA-1.5-13B | 57.91/50.45 | 53.60/53.01 | 55.30/55.52 |
| LLaVA-1.6-34B | 66.10/61.59 | 58.40/58.32 | 66.90/66.43 |
| **MIND (LLaVA-1.5-13B)** | **68.93/65.19** | 60.80/60.71 | **68.90/68.84** |

### Ablation Study

| Configuration | HarM F1 | FHM F1 | MAMI F1 |
|------|---------|--------|---------|
| Full MIND | **65.19** | **60.71** | **68.84** |
| w/o SSR (Random Selection) | 60.92 (-4.27) | 60.38 (-0.33) | 66.38 (-2.46) |
| w/o RID (No Insight Derivation) | 51.93 (-13.26) | 56.02 (-4.69) | 56.51 (-12.33) |
| w/o RID_forward | 63.46 (-1.73) | 59.81 (-0.90) | 66.60 (-2.24) |
| w/o RID_backward | 62.28 (-2.91) | 58.94 (-1.77) | 67.98 (-0.86) |
| w/o IAI (No Debate Mechanism) | 60.97 (-4.22) | 58.53 (-2.18) | 68.10 (-0.74) |

### Key Findings

- Removing the Relevant Insight Derivation (RID) leads to the most significant performance drop (HarM -13.26, MAMI -12.33), indicating that extracting harmful analytical experience from similar samples is the core driver of the framework.
- Similar Sample Retrieval (SSR) outperforms random selection by 2-4 F1 points on average, verifying the value of accurate retrieval.
- Forward and backward derivations complement each other—removing backward derivation has a greater impact on HarM and FHM, while removing forward derivation has a greater impact on MAMI.
- Top-$K=3$ is the optimal number of retrieved samples; excessive retrieval ($K>5$) introduces noise and degrades performance.
- MIND boosts LLaVA-1.5-13B beyond the baseline performance of LLaVA-1.6-34B, and even surpasses GPT-4o by 4.9 F1 points on HarM.

## Highlights & Insights

- **Bidirectional Insight Derivation** is the most elegant design of this work—applying a BiLSTM-like concept to agent inference, ensuring the information of each reference sample is fully utilized through two passes (forward and backward). This design is simple yet effective and can be transferred to other tasks requiring information extraction from multiple reference samples.
- **Zero-shot detection without labeled data** holds high practical value in harmful content moderation—allowing detection to start immediately when new harmful content emerges without waiting for labeling.
- The multi-agent debate mechanism makes the final decision more robust, reducing the bias of a single model through the collision of opposing viewpoints.

## Limitations & Future Work

- High computational cost—each meme requires multiple rounds of LMM inference (forward + backward derivation + debate), posing latency challenges for real-time deployment.
- Dependency on the quality and coverage of the reference set—if the reference set does not contain samples similar to the target meme, retrieval quality will degrade.
- Only binary classification (harmful/harmless) was tested; fine-grained harmful type classification (e.g., hate, discrimination, violence) has not yet been evaluated.
- Retrieval uses a fixed weighted fusion of visual and text encoders; more flexible multimodal fusion strategies (such as cross-modal attention) might yield better results.
- Future work could consider online updates to the reference set to allow the system to evolve over time and adapt to new meme patterns.

## Related Work & Insights

- **vs. Cao et al., 2024 (Few-shot ICL)**: Few-shot methods still require labeled exemplars, whereas MIND requires no annotations at all, offering wider applicability.
- **vs. GPT-4o Direct Inference**: GPT-4o performing well on MAMI but falling short of MIND on HarM indicates that collaborative multi-agent reasoning is superior to direct single-model judgment.
- This paper offers a new paradigm for multimodal content moderation—replacing traditional supervised classification with retrieval-augmented multi-agent reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of bidirectional insight derivation and multi-agent debate is creative, though individual components utilize standard concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three datasets and multiple model scales with detailed ablations, but lacks sensitivity analysis on reference set quality.
- Writing Quality: ⭐⭐⭐⭐ Clearly described methodology and persuasive case studies.
- Value: ⭐⭐⭐⭐ Holds practical significance for zero-shot content moderation, and the framework design is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[ECCV 2024\] Agent3D-Zero: An Agent for Zero-shot 3D Understanding](../../ECCV2024/llm_agent/agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[CVPR 2026\] Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection](../../CVPR2026/llm_agent/agent4faceforgery_multi-agent_llm_framework_for_realistic_face_forgery_detection.md)

</div>

<!-- RELATED:END -->
