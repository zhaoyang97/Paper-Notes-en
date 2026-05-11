---
title: >-
  [Paper Note] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs
description: >-
  [ICLR 2026][Multimodal VLM][Multimodal prompts] This work is the first to extend automatic prompt optimization (APO) from the pure text space to the multimodal space…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Multimodal prompts"
  - "automatic prompt optimization"
  - "Bayesian selection"
  - "MLLM"
  - "cross-modal alignment"
date: 2026-05-08
content_hash: e9be98fb003c8eeb
---

# Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs

**Conference**: ICLR 2026
**arXiv**: [2510.09201](https://arxiv.org/abs/2510.09201)
**Code**: [GitHub](https://github.com/Dozi01/MPO)
**Area**: Multimodal VLM / Prompt Optimization
**Keywords**: Multimodal prompts, automatic prompt optimization, Bayesian selection, MLLM, cross-modal alignment

## TL;DR
This work is the first to extend automatic prompt optimization (APO) from the pure text space to the multimodal space, proposing the MPO framework. It achieves an average accuracy of 65.1% across 10 datasets spanning image, video, and molecular modalities—surpassing the strongest text-based APO baseline ProTeGi (60.0%)—via two key components: alignment-preserving joint exploration (unified semantic gradients synchronously drive text and non-text prompt updates, diversified by Generation/Edit/Mix operators) and prior-inherited Bayesian UCB candidate selection (warm-starting child prompt Beta priors from parent prompt performance).

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) such as Qwen2.5-VL and GPT-4o have acquired the capability to process inputs across multiple modalities including text, images, video, and molecules. In parallel, the APO field has produced methods such as APE, OPRO, PE2, ProTeGi, and SEE, which automatically discover high-quality prompts through iterative generation → evaluation → refinement cycles, substantially reducing the burden of manual prompt engineering.

**Limitations of Prior Work**: All existing APO methods optimize only text prompts, entirely neglecting the multimodal input capabilities of MLLMs. Yet text is inherently inefficient for conveying certain information—describing the distinctive feather texture of a bird, for instance, requires lengthy and potentially ambiguous prose, whereas a single reference image conveys the same visual characteristics more intuitively and accurately. Restricting optimization to the pure text space amounts to searching within an intrinsically low-dimensional subspace, failing to exploit the full expressive power of MLLMs.

**Key Challenge**: Extending the prompt search space from $\mathcal{T}$ to $\mathcal{T} \times \mathcal{M}$ introduces two fundamental challenges: (1) **cross-modal consistency**—independently updating text and non-text prompts readily produces semantic conflicts (e.g., text instructs "focus on wing color" while the image depicts incorrect texture), a risk compounded by combinatorial explosion; (2) **candidate selection difficulty**—high-quality prompts become extremely sparse in the enlarged space, causing large evaluation budgets to be wasted on low-potential candidates, while existing uniform allocation or standard UCB strategies suffer from severe cold-start problems.

**Goal**: (1) How to efficiently explore the joint text + non-text prompt space while preserving cross-modal alignment; (2) How to rapidly identify high-quality multimodal prompts from a large candidate pool without wasting evaluation budget.

**Key Insight**: The authors observe two key facts: first, if text and non-text updates are driven by the same "semantic gradient" (failure analysis feedback), alignment is naturally maintained; second, there exists a strong positive correlation (Pearson $r = 0.88$) between parent and child prompt performance, meaning that parent prompt evaluation results can serve as valuable priors for child prompts, transforming "cold-start" into "warm-start."

**Core Idea**: Resolve cross-modal consistency via unified-feedback-driven joint updates, and resolve candidate selection efficiency via parent–child performance correlation-driven Bayesian prior inheritance, thereby generalizing APO to the multimodal space.

## Method

### Overall Architecture

MPO adopts a beam search optimization loop: each iteration maintains $b=3$ best multimodal prompts $\{(\mathbf{t}_i, \mathbf{m}_i)\}$; three exploration operators are applied to each parent prompt to generate $b^2$ child candidates; prior-inherited Bayesian UCB then selects the new top-$b$ as parent prompts for the next round. The process runs for $T=13$ iterations. The input is a task dataset (query–answer pairs) and the output is the best-performing multimodal prompt pair $(\mathbf{t}^*, \mathbf{m}^*)$. In the first iteration, only the Generation operator is used to initialize prompts since no non-text prompts yet exist.

### Key Designs

1. **Alignment-Preserving Exploration**:

    - **Function**: Generate cross-modally consistent candidate prompts in the enlarged $\mathcal{T} \times \mathcal{M}$ space.
    - **Mechanism**: Proceeds in two steps—**consolidated backpropagation** and **joint multimodal update**. Starting from the failure set $\mathcal{F}$ of the current prompt $(\mathbf{t}, \mathbf{m})$, the MLLM generates in a single pass a unified feedback $\nabla_{\mathbf{p}} = (\nabla_{\mathbf{t}}, \nabla_{\mathbf{m}})$ covering weaknesses in both the text and non-text dimensions. Based on this feedback, the MLLM simultaneously outputs an updated text prompt $\mathbf{t}'$ and a modality-specific textual condition $\mathbf{c}$ (describing how the non-text prompt should be adjusted). This condition $\mathbf{c}$ is passed to a modality generator $g$ (e.g., GPT-Image for text-to-image, Wan2.1 for text-to-video, or a text-to-molecule module), producing the updated non-text prompt $\mathbf{m}' = g(\mathbf{c})$.
    - **Design Motivation**: Driving both modality updates with a single feedback signal naturally avoids semantic misalignment from independent optimization. Experiments show that joint updates yield DSG alignment scores significantly higher than a sequential (text-first, then image) strategy.

2. **Three Exploration Operators (Generation / Edit / Mix)**:

    - **Function**: Systematically cover different regions of the multimodal search space to avoid local optima.
    - **Mechanism**:
        - **Generation**: Generates an entirely new non-text prompt from scratch, $\mathbf{m}' = g(\mathbf{c}_{\text{gen}}, \varnothing)$, without reference to existing candidates; suited for escaping local optima during early optimization or when the candidate pool is biased.
        - **Edit**: Applies fine-grained modifications to an existing non-text prompt, $\mathbf{m}' = g(\mathbf{c}_{\text{edit}}, \{\mathbf{m}\})$, preserving good structure while adjusting local attributes (e.g., texture, color); suited for stages where the prompt is already strong but needs refinement.
        - **Mix**: Fuses the complementary strengths of $K$ parent prompts, $\mathbf{m}' = g(\mathbf{c}_{\text{mix}}, \{\mathbf{m}_i\}_{i=1}^K)$, producing a balanced solution that integrates the advantages of multiple candidates and avoids over-reliance on a single one.
    - **Design Motivation**: Ablation studies show that each operator excels at different sub-tasks (e.g., Mix improves the Grape sub-task from 48.0% to 65.1%); combining all three achieves 76.4% on PlantVillage, far exceeding the best single-operator result of 74.8%.

3. **Prior-Inherited Bayesian UCB Selection**:

    - **Function**: Efficiently identify the best prompt from $b^2$ candidates while minimizing wasted evaluation budget.
    - **Mechanism**: Motivated by the empirically observed strong positive correlation ($r = 0.88$) between parent and child prompt performance, the expected score of each candidate is modeled as a Beta distribution $\text{Beta}(\alpha_i, \beta_i)$, with prior parameters initialized as $\alpha_i = \hat{\mu}_{\text{par}} \cdot S + 1$ and $\beta_i = (1 - \hat{\mu}_{\text{par}}) \cdot S + 1$, where $\hat{\mu}_{\text{par}}$ is the posterior mean of the parent prompt and $S = 10$ is the prior strength. At each step, the candidate with the highest UCB score is selected for evaluation, its posterior is updated, and after exhausting the budget the candidate with the highest expected score is chosen.
    - **Design Motivation**: Standard UCB and uniform allocation treat each new candidate as an independent arm with no prior, leading to poor cold-start efficiency. By inheriting parent prompt prior knowledge, MPO achieves the same performance as uniform allocation while consuming 70% less evaluation budget and 42% less than prior-free UCB. Proposition 3.1 formally proves that best-arm identification cost does not increase when the prior is more informative than a uniform prior.

### Loss & Training

MPO is a gradient-free optimization framework with no conventional loss function. The optimization objective is to maximize a task-specific evaluation metric (e.g., accuracy):

$$(\mathbf{t}^*, \mathbf{m}^*) = \argmax_{(\mathbf{t}, \mathbf{m}) \in \mathcal{T} \times \mathcal{M}} \mathbb{E}_{(\mathbf{q}, \mathbf{a}) \sim \mathcal{D}}[f(\text{MLLM}(\mathbf{t}, \mathbf{m}, \mathbf{q}), \mathbf{a})]$$

GPT-4o mini serves as the prompt optimizer responsible for failure analysis and prompt updates. Hyperparameters: beam size $b=3$, 13 iterations, 100 evaluation queries per candidate, prior strength $S = 10$ (10% of the evaluation budget).

## Key Experimental Results

### Main Results (10 Datasets × 3 Modalities)

| Method | PlantVillage | CUB-200 | SLAKE | DrivingVQA | RSVQA | Drive&Act | VANE | Absorption | BBBP (F1) | CYP (F1) | Avg Acc |
|--------|-------------|---------|-------|------------|-------|-----------|------|------------|----------|---------|--------|
| Human | 42.2 | 47.9 | 35.2 | 49.7 | 51.0 | 47.3 | 47.0 | 38.5 | 39.4 | 43.1 | 44.1 |
| CoT | 43.1 | 49.0 | 30.8 | 52.9 | 49.6 | 37.2 | 31.6 | 39.6 | 36.7 | 32.5 | 40.8 |
| 5-Shot | 46.5 | 58.1 | 28.0 | 45.9 | 49.2 | 54.3 | 61.4 | 48.1 | 45.5 | 49.3 | 49.3 |
| APE | 55.8 | 67.3 | 34.3 | 52.8 | 54.4 | 50.3 | 64.3 | 45.7 | 40.4 | 34.7 | 51.3 |
| PE2 | 67.9 | 71.6 | 35.8 | 53.7 | 55.2 | 50.8 | 63.0 | 64.5 | 56.8 | 58.2 | 58.2 |
| ProTeGi | 64.4 | 70.0 | 35.4 | 54.4 | 54.2 | 53.0 | 65.5 | 71.1 | 58.2 | 65.7 | 60.0 |
| SEE | 69.0 | 71.6 | 35.0 | 52.2 | 53.4 | 51.7 | 57.9 | 71.4 | 60.0 | 62.3 | 59.1 |
| **MPO** | **76.4** | **78.6** | **38.2** | **56.0** | **55.9** | **58.3** | **71.2** | **76.7** | **64.5** | **67.6** | **65.1** |

MPO achieves the best performance on all 10 datasets, with an average accuracy of 65.1%—surpassing the strongest text-based APO baseline ProTeGi (60.0%) by 5.1 percentage points.

### Ablation Study

| Ablation Configuration | PlantVillage Avg | CUB Avg | Notes |
|------------------------|-----------------|---------|-------|
| Human text + no image | 42.2 | 47.9 | Baseline |
| Human text + MPO image | 50.4 | 58.2 | Non-text prompt optimization alone yields +8–10% |
| MPO text + no image | 55.6 | 64.2 | Text-only optimization |
| **MPO text + MPO image** | **76.4** | **78.6** | Joint optimization far exceeds sum of individual gains |

| Exploration Operator | Apple | Corn | Grape | Potato | PlantVillage Avg |
|----------------------|-------|------|-------|--------|-----------------|
| SEE (text only) | 76.4 | 75.9 | 48.0 | 75.7 | 69.0 |
| Generation only | 76.9 | 77.9 | 53.7 | 83.6 | 73.3 |
| Edit only | 77.2 | 76.3 | 56.2 | 80.1 | 72.5 |
| Mix only | 74.0 | 77.9 | 65.1 | 79.8 | 74.8 |
| **MPO (all three)** | **77.7** | **78.2** | **65.9** | **84.0** | **76.4** |

| Backbone Generalization | Human | ProTeGi | SEE | **MPO** |
|-------------------------|-------|---------|-----|---------|
| Qwen2.5-VL (72B) | 55.7 | 74.1 | 73.6 | **80.4** |
| Gemma3 (12B) | 45.6 | 68.2 | 68.1 | **73.1** |
| InternVL-3.5 (14B) | 51.6 | 71.9 | 70.8 | **73.2** |
| GPT-4.1 nano | 46.8 | 61.0 | 61.6 | **65.9** |

### Key Findings
- **Modality contributions are synergistic, not additive**: MPO text (55.6) + MPO image (50.4 under human text) falls far short of joint optimization (76.4), indicating synergistic interaction between the two modalities—images provide fine-grained visual cues that text cannot encode, while text guides the model on "which features of that image to attend to."
- **Mix operator contributes most on difficult sub-tasks**: On the Grape sub-task, Mix alone raises accuracy from SEE's 48.0% to 65.1% (+17.1%), by fusing visual features in which different candidates each excel.
- **Prior inheritance dramatically improves efficiency**: On CUB, MPO with 30% of the evaluation budget matches the performance of the Uniform strategy at 100% budget—a 70% resource saving.
- **Training curves**: ProTeGi largely plateaus after iteration 3 (only +1.1%), whereas MPO continues to improve steadily by 6.4% after iteration 3, demonstrating that the multimodal space genuinely provides escape routes from text-only local optima.
- **Hidden state visualization**: Text-only methods (including MPO's text component) cluster in overlapping regions of the intermediate-layer embedding space, whereas MPO's full multimodal prompts push hidden states into a distinct region, indicating that non-text components introduce information dimensions not covered by text alone.
- **Optimal range for prior strength $S$**: Values that are too small ($S < 5$) yield uninformative priors; values that are too large ($S > 20$) cause over-reliance on parent prompts and impede absorption of new information; $S \approx 10$ is optimal.

## Highlights & Insights
- **The problem formulation itself is a contribution**: Extending APO from $\mathcal{T}$ to $\mathcal{T} \times \mathcal{M}$ is a direction that appears obvious in hindsight yet had not been pursued, and experiments confirm substantial gains. The core insight is that the mismatch between "MLLMs can process multimodal inputs" and "APO only optimizes text" is a valuable gap to fill.
- **Generative models as optimization components**: Embedding models such as GPT-Image and Wan2.1 into the APO loop as "modality translators"—receiving a textual condition $\mathbf{c}$ and outputting a non-text prompt $\mathbf{m}'$—is a design pattern transferable to any new modality (audio, 3D, code, etc.).
- **Empirical regularity of parent–child performance correlation**: The strong correlation ($r = 0.88$) provides a solid empirical foundation for Bayesian warm-starting. This regularity may generalize to all beam-search-based APO methods, not only multimodal ones.
- **Strong causal relationship between cross-modal alignment and task performance**: DSG alignment scores and accuracy gains exhibit a monotonically increasing relationship (MPO > Sequential > Random > In-Distribution > OOD), suggesting that alignment quality is the central metric determining the success of multimodal prompt optimization.

## Limitations & Future Work
- **Non-text prompt quality is bounded by generator capability**: MPO's ceiling depends on the modality generator (e.g., GPT-Image). Performance still exceeds text-only baselines when using lightweight generators (e.g., SANA1.5 1.6B), but lags behind strong generators by 4–5%; future generator improvements will automatically benefit MPO.
- **Prior strength $S$ requires manual tuning**: Fixing $S$ at 10% of the evaluation budget is a heuristic choice; adaptive adjustment of $S$ (e.g., dynamically scaled by the parent–child correlation) could further improve efficiency.
- **Evaluation cost remains substantial**: Each candidate requires 100 evaluations; 13 iterations × $b^2 = 9$ candidates × 100 queries amounts to thousands of MLLM inference calls, posing cost pressures for large-scale deployment.
- **Only discriminative tasks validated**: Effectiveness on generative tasks (e.g., captioning, multimodal dialogue) has not been demonstrated.
- **Limited modality coverage**: The current framework covers only image, video, and molecular non-text modalities; applicability to audio, 3D point clouds, code, and other modalities requires further investigation.

## Related Work & Insights
- **vs. ProTeGi / SEE (text APO)**: These methods search for optimal prompts in the text space via failure analysis + beam search. MPO inherits this framework but extends the search space to $\mathcal{T} \times \mathcal{M}$, improving average accuracy from 60.0% / 59.1% to 65.1% and demonstrating the additional information gain from the multimodal dimension.
- **vs. EvoPrompt (evolutionary APO)**: EvoPrompt explores the text space via mutation and crossover—analogous to MPO's Generation and Mix operators—but is limited to text and achieves only 49.5%. MPO's three multimodal operators constitute a natural generalization.
- **vs. MaPLe (continuous prompt learning)**: MaPLe learns continuous embedding vectors as multimodal prompts, requiring gradient access and model parameters. MPO is entirely gradient-free and compatible with closed-source API models, offering greater practical utility.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Defining the multimodal APO problem itself is a significant contribution; the prior-inherited Bayesian UCB is supported by theoretical guarantees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 3 modalities × 10 datasets × 7+ baselines, with comprehensive ablation, generalization, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; theory and experiments are well integrated; figures and tables are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction for APO; the framework is broadly generalizable and offers strong guidance for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MMOne: Representing Multiple Modalities in One Scene](../../ICCV2025/multimodal_vlm/mmone_representing_multiple_modalities_in_one_scene.md)
- [\[ICLR 2026\] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)
- [\[ICLR 2026\] Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems](why_keep_your_doubts_to_yourself_trading_visual_uncertainties_in_multi-agent_ban.md)

</div>

<!-- RELATED:END -->
