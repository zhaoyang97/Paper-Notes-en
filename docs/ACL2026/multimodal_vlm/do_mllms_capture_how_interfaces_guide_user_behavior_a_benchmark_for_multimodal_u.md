---
title: >-
  [Paper Note] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding
description: >-
  [ACL2026][Multimodal VLM][UI/UX] This paper proposes WiserUI-Bench, utilizing $300$ pairs of real-world A/B test-validated UI images and $684$ expert explanations to evaluate whether MLLMs understand how interface design…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "UI/UX"
  - "A/B testing"
  - "Multimodal evaluation"
  - "User behavior"
  - "Visual reasoning"
date: 2026-05-08
content_hash: 3184263c11bcd35c
---

# Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding

**Conference**: ACL2026  
**arXiv**: [2505.05026](https://arxiv.org/abs/2505.05026)  
**Code**: Not reported in cache  
**Area**: Multimodal VLM / UI Understanding  
**Keywords**: UI/UX, A/B testing, Multimodal evaluation, User behavior, Visual reasoning

## TL;DR
This paper proposes WiserUI-Bench, utilizing $300$ pairs of real-world A/B test-validated UI images and $684$ expert explanations to evaluate whether MLLMs understand how interface design influences user behavior. Results indicate that existing models perform near random chance in selecting winners and significantly fall short of expert levels in explaining the underlying reasons.

## Background & Motivation
**Background**: UI design extends beyond aesthetics; its core objective is to guide users toward target behaviors such as registration, purchase, or clicks. The industry typically relies on large-scale A/B testing to verify which interface version drives more actual user behavior, followed by post-hoc analysis by design experts.

**Limitations of Prior Work**: Existing UI evaluation benchmarks mostly focus on surface visual quality, design guideline violations, or single-screen expert critiques, lacking validation from real user behavior. For MLLMs, determining which of two UIs better drives behavior requires more than recognizing color, buttons, or layout differences; it necessitates inferring user attention, memory load, and action paths.

**Key Challenge**: Models may perceive UI elements without necessarily understanding how these elements alter aggregate user behavior. There is a clear gap between visual recognition capabilities and behavioral causal reasoning.

**Goal**: To construct a behavior-grounded UI/UX understanding benchmark and evaluate MLLMs on two tasks: predicting the winner of a real-world A/B test given a pair of UI images, and explaining why the winner is more effective, aligned with expert explanations.

**Key Insight**: Instead of using synthetic perturbations or subjective aesthetic scores, the authors collect real UI variants and validated results from public industrial A/B test cases, complemented by key explanations written by UI/UX experts.

**Core Idea**: Treat "user behavior outcomes" as the anchor for UI/UX understanding, using real A/B winners to examine whether models can reason from visual interface differences to behavioral impacts.

## Method
The methodology of WiserUI-Bench emphasizes data construction and evaluation protocols. It decomposes UI/UX understanding into two complementary abilities: selection (predicting effectiveness) and interpretation (explaining the winner's effectiveness).

### Overall Architecture
Data is sourced from public A/B testing platforms such as VWO success stories, GoodUI leaks, and abtest.design. Each sample includes a pair of UI images, the ground-truth A/B test winner, context such as page type/industry/device, and curated expert explanations. The authors remove visual prompts like added arrows or circles to prevent models from "cheating" via annotations. The final dataset consists of $300$ real UI image pairs and $684$ expert key interpretations, covering $11$ page categories, $13$ industries, and both web and mobile devices.

### Key Designs
1.  **Real A/B test grounded data construction**:

    - **Function**: Ensures each winner is validated by real user behavior rather than subjective preference.
    - **Mechanism**: Aggregates UI variants and results from credible A/B test platforms, retains clean UI images after removing visual annotations, and records which version prompted more target user actions.
    - **Design Motivation**: The ultimate goal of UI/UX is behavioral outcomes; benchmarks without real behavior validation can easily degrade into visual aesthetic or compliance checks.

2.  **Expert explanation and UX dimension annotation**:

    - **Function**: Provides a semantic evaluation baseline for the interpretation task.
    - **Mechanism**: Three UI/UX experts independently annotated key UI modifications and behavioral impacts knowing the winner, retaining only explanations with substantial overlap among at least two experts. Each explanation was mapped to $12$ UX laws and categorized into Norman's three cognitive dimensions: perception, memory, and action.
    - **Design Motivation**: Models should not only guess the correct image but also articulate reasons matching expert cognition. Expert explanations expand evaluation from binary choice to diagnosable behavioral reasoning.

3.  **Dual-task evaluation protocol**:

    - **Function**: Separately measures prediction and explanation capabilities.
    - **Mechanism**: The selection task is measured using $FA$, $SA$, $AA$, and $CA$; specifically, $CA$ (Consistency Accuracy) requires the model to select the same correct UI even after the image order is swapped to eliminate position bias. The interpretation task uses a GPT-4o evaluator to judge whether the model's free text covers each expert explanation, reporting Interpretation Recall and Instance-level Recall.
    - **Design Motivation**: Pairwise tasks are prone to "always select the first/second" bias; $CA$ better reflects content understanding. The interpretation task reveals if the model merely guessed the result or understands the behavioral mechanism.

### Loss & Training
This work does not train models but evaluates existing MLLMs. For the selection task, two input sequences are tested for each UI pair, with results averaged over three independent runs. For the interpretation task, models generate free-text explanations, which are then judged via a human-validated GPT-4o evaluator for binary semantic coverage; this evaluator achieved $83.0\%$ accuracy and a Cohen's kappa of $0.66$ on $1,000$ random samples.

## Key Experimental Results

### Main Results
| Task | Model / Method | Key Metrics | Value | Conclusion |
|------|-------------|----------|------|------|
| UI/UX selection | Random | $AA$ / $CA$ | $50.00$ / $25.00$ | $CA$ random baseline is $25\%$ |
| UI/UX selection | GPT-4o zero-shot | $AA$ / $CA$ | $60.11$ / $30.11$ | $AA$ seems high, but $CA$ is near random |
| UI/UX selection | GPT-4o-2024-08-06 | $AA$ / $CA$ | $58.50$ / $33.33$ | Strong models still show obvious position bias |
| UI/UX selection | Claude 3.5 Sonnet | $AA$ / $CA$ | $56.83$ / $32.33$ | Selection task remains unsolved by strong models |
| UI/UX interpretation | GPT-4o-2024-08-06 | Interp / Inst Recall | $68.71$ / $79.00$ | One of the strongest in explanation |
| UI/UX interpretation | Claude 3.5 Sonnet | Interp / Inst Recall | $67.40$ / $80.33$ | Highest at the instance level |
| UI/UX interpretation | GPT-4o | Interp / Inst Recall | $50.15$ / $66.67$ | Interpretation ability $\neq$ selection ability |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| GPT-4o zero-shot | $FA$ $31.89$, $SA$ $88.33$, $AA$ $60.11$, $CA$ $30.11$ | Strong bias toward the second image; $AA$ inflated by position bias |
| GPT-4o DDCoT | $AA$ $61.72$, $CA$ $34.78$ | Slight improvement in $CA$ compared to zero-shot |
| GPT-4o MAD R1 | $AA$ $59.33$, $CA$ $39.00$ | Multi-view short-round debate can reduce position bias |
| GPT-4o MAD R3 | $AA$ $57.67$, $CA$ $29.89$ | Performance drops if debate is too long |
| Claude 3.5 Sonnet zero-shot | $AA$ $56.39$, $CA$ $24.22$ | $CA$ is near random, severe bias |
| Claude 3.5 Sonnet MAD R1 | $AA$ $55.94$, $CA$ $30.00$ | Short-round multi-agent discussion improves consistency |

### Key Findings
- $AA$ in the selection task is unreliable because many models exhibit a bias toward the second input image; $CA$ is more revealing of true content understanding.
- The interpretation task is relatively easier, but models typically cover only a portion of expert explanations. The gap between Interpretation Recall and Instance-level Recall suggests models often "mention some points but not all."
- UI element types affect difficulty: Changes related to Container & Layout Structure are the hardest, indicating that model sensitivity to how layout affects behavior is weaker than their ability to recognize salient text or button changes.

## Highlights & Insights
- This paper advances UI/UX benchmarks from "looking good" to "actually changing user behavior," which is a rare but critical grounding in multimodal evaluation.
- The $CA$ metric is essential. Without it, the "second-image bias" of models makes $AA$ look acceptable; with $CA$, the fact that models perform near random is clearly exposed.
- The decoupling of selection and interpretation abilities is noteworthy: some models are proficient at explaining the winner after the fact but fail to select it beforehand, suggesting current MLLMs excel at post-hoc rationalization but struggle to predict actual user behavior.

## Limitations & Future Work
- The authors acknowledge differences in user experience across cultural and social norms; the public A/B test cases in WiserUI-Bench inevitably contain cultural biases.
- The data scale is limited to $300$ UI pairs. While the scarcity of real A/B test data makes this difficult to avoid, it is still small for training or fine-grained statistical analysis.
- The benchmark is primarily based on static UI images and does not fully cover interactive UIs, dynamic flows, or long-term user journeys.
- Future work could extend to interactive web pages, real click trails, diverse cultural user groups, and training specialized UI/UX reasoning models using behavior-grounded data.

## Related Work & Insights
- **vs Yang and Li 2024**: That benchmark detects single-screen guideline violations, focusing on rule compliance; WiserUI-Bench focuses on real user behavioral outcomes.
- **vs BetterWeb**: BetterWeb uses synthetic UI pairs and basic visual quality targets; this work uses real production UIs and A/B test winners.
- **vs UICrit**: UICrit features expert critiques but lacks real behavior validation and focuses on single screens; this work uses a pairwise setup closer to design decision-making.
- **Insight**: Evaluating UI capabilities of multimodal models should move beyond asking "which is more aesthetic" to "which makes users more likely to act, and why."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Behavior-grounded multimodal UI/UX benchmark is highly original; task definitions are clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Extensive model coverage and good metric design; sample size is limited by the scarcity of real A/B tests.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear data construction, tasks, and results analysis; case studies effectively explain failure modes.
- **Value**: ⭐⭐⭐⭐⭐ Provides direct reference value for MLLM visual reasoning, UI agents, and design assistant systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)
- [\[ACL 2026\] Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning](can_mllms_reason_beyond_language_visreason_a_comprehensive_benchmark_for_vision-.md)
- [\[ICLR 2026\] How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images](../../ICLR2026/multimodal_vlm/how_do_medical_mllms_fail_a_study_on_visual_grounding_in_medical_images.md)
- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)

</div>

<!-- RELATED:END -->
