---
title: >-
  [Paper Note] Visual Persuasion: What Influences the Decision-Making of Vision-Language Models?
description: >-
  [ICML 2026][Multimodal VLM][Visual Persuasion] This paper systematically uses image editing models to modify visual attributes (maintaining semantic identity) and discovers significant visual preferences in VLMs. It prop…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Persuasion"
  - "VLM Decision-making"
  - "Visual Preferences"
  - "Prompt Optimization"
  - "Interpretability"
date: 2026-05-08
content_hash: b58af8f793a41657
---

# Visual Persuasion: What Influences the Decision-Making of Vision-Language Models?

**Conference**: ICML 2026  
**arXiv**: [2602.15278](https://arxiv.org/abs/2602.15278)  
**Code**: https://github.com/MaggieCherepLabs  
**Area**: Multimodal VLMs  
**Keywords**: Visual Persuasion, VLM Decision-making, Visual Preferences, Prompt Optimization, Interpretability

## TL;DR
This paper systematically uses image editing models to modify visual attributes (maintaining semantic identity) and discovers significant visual preferences in VLMs. It proposes three visual prompt optimization methods to expose these preferences, develops an automated interpretability pipeline to understand the visual themes driving decisions, and mitigates risks through visual normalization.

## Background & Motivation

**Background**: Current VLM evaluations focus primarily on functional metrics. However, in practical applications, VLMs are deployed as agentic systems to make critical decisions—such as recommending products, screening candidates, or evaluating properties.

**Limitations of Prior Work**: Existing VLM evaluations lack a deep understanding of the model's visual preference structure. While research shows LLM agents are highly sensitive to text prompts, the vulnerability of VLMs to visual preferences is poorly understood. When these models run autonomously, hidden visual biases could be exploited or lead to large-scale bias.

**Key Challenge**: How to systematically discover and quantify VLM visual preferences? Traditional approaches (collecting large datasets of natural variations) are costly and provide incomplete coverage.

**Goal**: (1) Develop a systematic method to expose VLM visual preferences; (2) Quantitatively evaluate the impact of these preferences on model decisions; (3) Identify and explain the visual themes driving decisions; (4) Propose mitigation strategies.

**Key Insight**: Modern image editing models (Gemini 3, Qwen-Image-Edit) provide fine-grained visual controllability. These models can be used to iteratively modify images while using VLM pairwise choice feedback to optimize the editing direction—essentially exploring the model's hidden utility function.

**Core Idea**: Treat the VLM decision function as a hidden visual utility landscape, and use "revealed preference"—systematic editing and pairwise comparison—to infer and explore this landscape.

## Method

### Overall Architecture
The approach consists of three stages: (1) **Visual Prompt Optimization**: Starting from an original image, an image editing model iteratively modifies the image based on optimization feedback until local equilibrium is reached; (2) **Automated Interpretability Pipeline**: Differences between optimized and original images are abstracted into high-level visual themes via multi-stage aggregation (Matryoshka Summaries); (3) **Mitigation and Validation**: Testing the effectiveness of visual normalization.

### Key Designs

1. **Constrained Optimization Framework and Identity Maintenance**:
    - **Function**: Maintain the essence of the original object/scene during visual optimization, modifying only the presentation.
    - **Mechanism**: Define an identity constraint set $\mathcal{C}(x_0) = \{x \in \mathcal{X}: I(x, x_0) = 1\}$. Optimization is performed over object constraints $\max_{p \in \mathcal{P}} U_\tau(x(p))$ s.t. $x(p) \in \mathcal{C}(x_0)$ to ensure editing operations only modify semantic-preserving attributes.
    - **Design Motivation**: Without constraints, the optimizer would "cheat" by simply replacing the object.

2. **Three Competitive Visual Prompt Optimization Methods (VTG/VFD/CVPO)**:
    - **Function**: Explore VLM visual utility functions from a discrete space of editing prompts under noisy preference feedback.
    - **Mechanism**: All three methods follow a "propose-evaluate" loop. VisualTextGrad (VTG) uses an LLM critic to generate textual gradient feedback. VisualFeedbackDescent (VFD) uses multi-critic voting to decide winners. The novel Competitive Visual Prompt Optimization (CVPO) models optimization as a competitive process: maintaining two competitors (prompts $p_A, p_B$), with consistency checks by $k$ judges per round, stopping when the win rate approaches 50%.
    - **Design Motivation**: VTG fails to stop effectively in noisy feedback. VFD averages 24.9 iterations. CVPO averages only 17.4 iterations (63% cost reduction) while achieving optimal performance on most models.

3. **Multi-stage Automated Interpretability Pipeline (Matryoshka Aggregation)**:
    - **Function**: Abstractions of low-level pixel differences into high-level, readable visual themes.
    - **Mechanism**: The first stage uses a VLM to generate difference descriptions by comparing original and optimized image pairs one by one. The second stage recursively aggregates these descriptions—embedding, clustering by similarity, and summarizing each cluster with an LLM. The "Matryoshka" property ensures high-level clusters are generated from low-level summaries, maintaining traceability.
    - **Design Motivation**: Automatically generate interpretability notes for thousands of optimized images. Convergence of different optimization methods toward similar visual themes suggests preferences reflect stable VLM properties.

## Key Experimental Results

### Main Results: Optimization Performance Evaluation

| Dataset/Task | Original Image | Zero-Shot Edited | Optimized | Gain (Rel. to Original) |
|------------|--------|----------|-------|--------------|
| Product Recommendation | 0.27 ± 0.03 | 0.48 ± 0.02 | 0.55 ± 0.02 | +78% |
| Property Search | 0.31 ± 0.02 | 0.51 ± 0.02 | 0.62 ± 0.02 | +100% |
| Candidate Screening | 0.29 ± 0.03 | 0.47 ± 0.02 | 0.58 ± 0.02 | +100% |
| Hotel Booking | 0.26 ± 0.03 | 0.52 ± 0.02 | 0.61 ± 0.02 | +135% |

### Optimization Method Comparison

| VLM | VTG | VFD | CVPO | Best-to-Second Difference |
|-----|-----|-----|------|-------------|
| Qwen-3-VL 235B | 0.131 | 0.601 | 0.771 | +0.170 |
| GPT-5 Mini | 0.190 | 0.561 | 0.766 | +0.205 |
| Gemini 3 Flash | 0.140 | 0.604 | 0.761 | +0.157 |
| GPT-4o | 0.179 | 0.566 | 0.749 | +0.183 |
| Claude Sonnet 4.5 | 0.310 | 0.603 | 0.594 | -0.010 |

### Key Findings
- Zero-shot editing is already significantly effective—basic prompts can increase selection probability by 0.2-0.4.
- CVPO performance is the most stable—outperforming VFD on 7 out of 9 VLMs.
- Significant differences in efficiency—VTG uses 100% of the budget, VFD 74.6%, and CVPO only 36.9%.
- Human study validation (N=154): CVPO optimized results ranked highest in human head-to-head comparisons.
- Convergence of visual themes—different optimization methods converge to similar themes, suggesting stable VLM properties.
- Incompleteness of mitigation strategies—visual normalization reduces the advantage but cannot eliminate it entirely.

## Highlights & Insights
- **Methodological Innovation**: First systematic expansion of prompt optimization to the visual domain; the CVPO competitive framework and equilibrium stopping condition are clever designs.
- **Multi-layered Evidence System**: 1.8M+ API calls, 125k+ generated images, 4 task domains, human validation, and automated interpretability.
- **Key Insight: "Hidden Optimization of Presentation Materials"**: Reveals a critical risk in AI governance—image optimization, if exploited maliciously, can systematically manipulate VLM agent decisions.
- **Reusable Design Concepts**: Matryoshka summaries, identity constraints, and competitive frameworks are all generalizable.

## Limitations & Future Work
- High computational resource requirements limit scalability.
- Boundaries of identity maintenance are blurred (e.g., ethical tensions in clothing/background changes).
- Limited scale of human validation (N=154).
- Using the same optimization set for prompt distillation might affect external validity.
- Future improvements: Researching VLM adversarial robustness training; developing visual auditing tools; expanding to multi-modal scenarios; studying variance in VLM preferences.

## Related Work & Insights
- **vs. Adversarial Example Research**: Adversarial attacks seek minimal perceptible perturbations; this work focuses on perceptually significant but semantically preserved natural variations.
- **vs. Behavioral ML & Agent Evaluation**: Prior work is in the text domain; this work extends to the visual domain and develops the first systematic discovery method.
- **vs. Prompt Optimization Literature (TextGrad, Feedback Descent)**: This work extends feedback gradient principles to multi-modality.
- **vs. Automated Interpretability**: This work uses similar ideas to explain black-box VLM behaviors, serving as a complementary external interpretability method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic use of visual prompt optimization to explore hidden VLM visual preferences.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with a complete chain of evidence; the only minor drawback is the small human sample size.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and detailed methodology description.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for AI safety and governance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[ICML 2026\] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)
- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](../../NeurIPS2025/multimodal_vlm/antigrounding_lifting_robotic_actions_into_vlm_representatio.md)

</div>

<!-- RELATED:END -->
