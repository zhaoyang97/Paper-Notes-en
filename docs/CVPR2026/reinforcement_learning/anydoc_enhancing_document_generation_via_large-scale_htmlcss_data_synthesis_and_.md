---
title: >-
  [Paper Note] AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization
description: >-
  [CVPR 2026][Reinforcement Learning][Document Generation] AnyDoc proposes a general-purpose document generation framework based on a unified HTML/CSS representation. It constructs a 265K-document dataset, DocHTML, via an automated data synthesis pipeline, and fine-tunes a multimodal large language model through SFT and Height-Aware Reinforcement Learning (HARL). The framework surpasses baselines including GPT-4o on three tasks: intent-to-document, document de-rendering, and element-to-document generation.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Document Generation
  - HTML/CSS
  - Data Synthesis
  - Multimodal Large Language Models
date: 2026-05-08
content_hash: bb3d42fc4380ff2a
---

# AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization

**Conference**: CVPR 2026
**arXiv**: [2603.25118](https://arxiv.org/abs/2603.25118)
**Code**: None
**Area**: Reinforcement Learning / Document Generation
**Keywords**: Document Generation, HTML/CSS, Data Synthesis, Reinforcement Learning, Multimodal Large Language Models

## TL;DR
AnyDoc proposes a general-purpose document generation framework based on a unified HTML/CSS representation. It constructs a 265K-document dataset, DocHTML, via an automated data synthesis pipeline, and fine-tunes a multimodal large language model through SFT and Height-Aware Reinforcement Learning (HARL). The framework surpasses baselines including GPT-4o on three tasks: intent-to-document, document de-rendering, and element-to-document generation.

## Background & Motivation
**Background**: Various document types (resumes, presentations, reports, etc.) are widely used in everyday workflows, and manually designing high-quality documents requires balancing structure, layout, visual, and stylistic principles. Automated document generation has attracted increasing research attention in recent years.

**Limitations of Prior Work**:
   - **Limited applicability**: Most methods target a single document category (e.g., advertisements, slides, infographics) and struggle to generalize to unseen categories.
   - **Suboptimal document representations**:
     - Raster images: not editable.
     - Flat coordinate sequences (JSON): require extensive coordinate computation for complex documents and fail to capture hierarchical structure.
   - **Data scarcity**: Manually crafting documents is costly; existing datasets are small-scale (e.g., Crello with only 20K samples) and cover few categories.

**Key Challenge**: How to simultaneously achieve category generalizability, structural editability, and data sufficiency?

**Key Insight**: Introducing HTML/CSS as a unified document representation—naturally hierarchical, expressive layout mechanisms (flexbox/grid), and amenable to large-scale synthesis.

**Core Idea**: Unified HTML/CSS representation + automated data synthesis pipeline + HARL to address overflow = general-purpose, high-quality document generation.

## Method

### Overall Architecture
1. Data synthesis pipeline (5 stages) → DocHTML dataset (265K documents, 111 categories, 32 styles)
2. SFT fine-tuning of a multimodal large language model → three generation tasks
3. HARL post-training → resolving content overflow issues

### Key Designs
1. **DocHTML Data Synthesis Pipeline**:

    - **Metadata collection**: Starting from professional document libraries, design intents and content descriptions are generated using InternVL3.
    - **HTML/CSS code generation**: Qwen3-Coder-480B generates code from metadata, with flexbox/grid layouts encouraged; `<img>` tags follow a unified format (placeholder URLs + alt descriptions).
    - **Image asset synthesis**: FLUX.1-dev generates illustrations from alt descriptions.
    - **Rendering**: Playwright renders HTML/CSS + images into document screenshots.
    - **Data cleaning**: Samples with size mismatches, missing img tags, zero-height elements, or content overflow are excluded.
    - **Design Motivation**: HTML/CSS hierarchical structure naturally captures containment relationships; flexbox and grid eliminate the need for precise coordinate computation. Code generation models can produce large-scale, multi-category documents.

2. **Three Document Generation Tasks**:

    - **I2D (Intent-to-Document)**: Input: design intent text + target dimensions → Output: HTML/CSS.
    - **DD (Document De-rendering)**: Input: document screenshot + target dimensions → Output: HTML/CSS.
    - **E2D (Element-to-Document)**: Input: a set of text/image elements + target dimensions → Output: HTML/CSS.
    - All tasks are unified as a conditional sequence generation problem: "condition → HTML/CSS."

3. **Height-Aware Reinforcement Learning (HARL)**:

    - **Function**: Resolves content overflow in documents generated after SFT.
    - **Mechanism**: Built on GRPO, a group of candidate outputs is sampled for each input; Playwright renders them to obtain the actual height $\hat{h}$, and the reward is computed as:
    $$r = \max\left(0, \begin{cases} 1, & 1-\gamma \leq \rho \leq 1 \\ \gamma + \rho, & \rho < 1-\gamma \\ 1 - \alpha(\rho - 1), & \rho > 1 \end{cases}\right)$$
      where $\rho = \hat{h}/h$ is the height deviation ratio.
    - Both overflow ($\rho > 1$) and severe underflow ($\rho < 1-\gamma$) are penalized.
    - **Design Motivation**: SFT models tend to generate documents exceeding the specified height, resulting in poor rendering after truncation. GRPO's group-relative advantage mechanism naturally distinguishes "good" outputs (height-compliant) from "bad" ones (overflow), without requiring manual preference annotation.

### Loss & Training
- **SFT stage**: Based on Qwen2.5-VL-7B-Instruct; LoRA rank=32, batch=128, lr=1e-4.
- **HARL stage**: Full-parameter fine-tuning; lr=1e-6, batch=64, GRPO rollout=5.
- The 20K samples with the most severe overflow during SFT inference are selected as the HARL training set.

## Key Experimental Results

### Main Results (I2D: Intent-to-Document)

| Method | Layout | Image | Typography | Content | Height↓ | Intention |
|--------|--------|-------|------------|---------|---------|-----------|
| OpenCOLE | 7.91 | 8.03 | 7.79 | 7.52 | - | 8.13 |
| FLUX.1-dev | 7.58 | 7.78 | 6.54 | 5.16 | - | 6.91 |
| GPT-4o | 8.59 | 8.75 | 8.32 | 8.41 | 0.047 | 8.96 |
| **AnyDoc** | **8.64** | **8.92** | **8.36** | **8.44** | **0.005** | 8.95 |

### Ablation Study (Document De-rendering, DD Task)

| Configuration | Block | Text | Position | Color | Height↓ |
|---------------|-------|------|----------|-------|---------|
| Coordinate sequence (JSON) | 0.871 | 0.925 | 0.780 | 0.812 | 0.188 |
| HTML/CSS (10K data) | 0.942 | 0.974 | 0.862 | 0.915 | 0.374 |
| HTML/CSS (full data) | 0.958 | 0.984 | 0.900 | 0.948 | 0.309 |
| + HARL | **0.965** | **0.986** | **0.910** | **0.958** | 0.309→improved |

### Key Findings
- The HTML/CSS representation significantly outperforms coordinate sequences (JSON) across all metrics, particularly on Position and Color.
- HARL reduces the Height metric on the I2D task from 0.131 to 0.005, nearly eliminating overflow.
- Scaling DocHTML from 10K to the full dataset yields consistent performance improvements.
- AnyDoc (7B) surpasses GPT-4o on multiple metrics, especially Height control.

## Highlights & Insights
- **HTML/CSS as document representation is a key innovation**: It connects document generation with the mature web development technology stack.
- The data synthesis pipeline is fully automated and can be continuously scaled to new categories and styles.
- HARL is an elegant solution: rendering feedback is incorporated into RL rewards, enabling the model to learn to comply with dimensional constraints.
- All three tasks share the same framework and dataset.

## Limitations & Future Work
- The 7B model still underperforms GPT-4o on Height control for the DD task, reflecting limitations of model scale.
- HARL requires Playwright rendering to compute rewards, leading to lower training efficiency.
- The aesthetic quality of generated documents remains dependent on the capability of the underlying code generation model.
- Only static documents are supported; interactive documents with dynamic effects are not addressed.

## Related Work & Insights
- Compared to Design2Code, AnyDoc not only supports de-rendering but also intent-to-document and element-to-document generation.
- The HARL paradigm (render → measure → reward) can be generalized to other code generation tasks that require visual constraint satisfaction.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified HTML/CSS representation + data synthesis + HARL form a cohesive trinity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks × multiple baselines × comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough problem analysis with well-motivated methodology.
- Value: ⭐⭐⭐⭐ Practical applicability to automated document generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closed-loop_robotic_data_generation_via_semantic_planning_and_autonomous_c.md)
- [\[AAAI 2026\] Enhancing Robustness of Offline RL Under Data Corruption via SAM](../../AAAI2026/reinforcement_learning/enhancing_robustness_of_offline_reinforcement_learning_under_data_corruption_via.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/reinforcement_learning/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](../../ACL2026/reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)

<!-- RELATED:END -->
