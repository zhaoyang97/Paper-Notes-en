---
title: >-
  [Paper Note] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving
description: >-
  [ICLR 2026][Optimization][Visual mathematical reasoning] CogFlow proposes a cognition-inspired three-stage visual mathematical reasoning framework (Perception → Internalization → Reasoning)…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Visual mathematical reasoning"
  - "knowledge internalization"
  - "GRPO"
  - "perception-reasoning alignment"
  - "cognition-inspired"
date: 2026-05-08
content_hash: 38065a09b891ed8d
---

# CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving

**Conference**: ICLR 2026
**arXiv**: [2601.01874](https://arxiv.org/abs/2601.01874)  
**Code**: [https://shchen233.github.io/cogflow/](https://shchen233.github.io/cogflow/)  
**Area**: Optimization
**Keywords**: Visual mathematical reasoning, knowledge internalization, GRPO, perception-reasoning alignment, cognition-inspired

## TL;DR
CogFlow proposes a cognition-inspired three-stage visual mathematical reasoning framework (Perception → Internalization → Reasoning), enhanced by Synergistic Visual Rewards for perception, a Knowledge Internalization Reward to bridge perception and reasoning, and Visual-Gated Policy Optimization to anchor visual reasoning, addressing the core problem of "correct perception but drifted reasoning" in existing methods.

## Background & Motivation

**Background**: MLLMs perform poorly on visual mathematical problems. Early "one-step reasoning" frameworks conflated perception and reasoning; later "decoupled reasoning" pipelines separated the two but optimized each independently.

**Limitations of Prior Work**:
- One-step frameworks (VLM-R1) produce unstructured reasoning with intertwined perception and reasoning errors.
- Decoupled pipelines (MathFlow) improve perception but the reasoning stage frequently ignores perceptual results, leading to *reasoning drift*.
- A critical question overlooked by all prior work: **Are the extracted visual cues faithfully integrated into subsequent reasoning?**

**Key Challenge**: Accurate perception does not guarantee correct reasoning — a model may correctly interpret a figure yet take shortcuts during reasoning, producing seemingly plausible but visually ungrounded reasoning chains.

**Goal**:
- How to ensure that perceptual outputs are faithfully converted into reasoning-ready knowledge representations?
- How to explicitly anchor reasoning to perceptual outputs during RL training?

**Key Insight**: The cognitive science concept of "knowledge internalization" — human reasoning does not jump directly from perception to conclusions; instead, it first transforms perceptual information into structured knowledge (e.g., "AB is a diameter + C lies on the circle → ∠ACB = 90°") and then reasons from it.

**Core Idea**: Insert a "knowledge internalization" stage between perception and reasoning, employ a dedicated reward model to detect whether reasoning is faithful to perception, and use a visual gate to filter low-quality perceptions before reasoning.

## Method

### Overall Architecture
A three-stage cognitive flow: **❶ Perception** (enhanced by Synergistic Visual Rewards) → **❷ Internalization** (bridged by Knowledge Internalization Reward) → **❸ Reasoning** (anchored by Visual-Gated Policy Optimization). Training proceeds in two stages: SFT followed by RL.

### Key Designs

1. **Synergistic Visual Rewards (SynVRs)**:

    - **Function**: Evaluates perceptual quality from both parametric and semantic spaces.
    - **Mechanism**:
        - **VPR**: Converts geometric primitives into parametric equations and scores them precisely in parametric space using Hungarian matching and Euclidean distance.
        - **VSR**: Re-renders images from textual perceptual outputs and computes cosine similarity with the original image via FG-CLIP to assess global layout consistency.
        - Final score: $\mathcal{S}_{SynVRs} = \alpha \cdot \mathcal{S}_{VPR} + (1-\alpha) \cdot \mathcal{S}_{VSR}$
    - **Design Motivation**: VPR ensures local geometric precision while VSR ensures global perceptual consistency; the two are complementary and avoid blind spots of any single metric.

2. **Knowledge Internalization Reward (IntlzR)**:

    - **Function**: Trains a reward model to detect whether reasoning is faithful to perception.
    - **Mechanism**: Constructs positive-negative trajectory pairs (1 positive + 5 negatives), where negatives cover 5 typical failure modes (omission of primitives, fabrication of facts, misuse of theorems, violation of geometric constraints, and inconsistent references). Training uses Softmax-DPO: $\mathcal{L} = -\log \sigma(-\log \sum_j \exp(s_j^- - s^+))$, contrasting one positive sample against multiple negatives simultaneously.
    - **Design Motivation**: Prior methods focus solely on perceptual accuracy while neglecting whether perceptual outputs are correctly utilized — IntlzR addresses this gap.

3. **Visual-Gated Policy Optimization (VGPO)**:

    - **Function**: Filters low-quality perceptions before generating reasoning, both during RL training and inference.
    - **Mechanism**: For each input, $M$ perception trajectories are sampled and scored by $S_{vis}$ (VPR + VSR during training; VSR only during inference). The Visual Gate $\Gamma$ selects the first perception exceeding threshold $\tau$, or defaults to the highest-scoring one. Only approved perceptions condition subsequent reasoning generation.
    - **Design Motivation**: Prevents low-quality perceptions from "contaminating" downstream reasoning — even strong RL-optimized reasoning cannot recover from erroneous perceptual inputs.

### Loss & Training
- **SFT Stage**: Standard SFT on MathCog-SFT (120K+ samples).
- **RL Stage**: Triple reward combination — SynVRs (perceptual quality) + IntlzR (internalization faithfulness) + InfR (answer correctness), optimized via GRPO.
- **MathCog Dataset**: 120K+ high-quality perception-reasoning aligned annotations.

## Key Experimental Results

### Main Results (Visual Mathematical Benchmarks)

| Method | MathVista | GeoQA | MathCheck-Geo | Average |
|--------|-----------|-------|---------------|---------|
| MathFlow (decoupled) | Medium | Medium | Medium | ~60% |
| VLM-R1 (one-step) | Medium | Low | Low | ~55% |
| **CogFlow** | **Highest** | **Highest** | **Highest** | **~70%+** |

### Ablation Study

| Configuration | Reasoning Drift Accuracy↑ | Answer Accuracy↑ |
|---------------|--------------------------|-----------------|
| w/o IntlzR | 73% | Baseline |
| w/o Visual Gate | Low | −3% |
| w/o SynVRs | Low | −5% |
| **Full CogFlow** | **92%** | **Highest** |

### Key Findings
- **Significant reduction in reasoning drift**: CogFlow improves reasoning drift accuracy from 73% (MathFlow) to 92%, demonstrating the effectiveness of the knowledge internalization stage.
- **Surpasses closed-source large models**: Matches or exceeds GPT-4V/Claude-3.5 on several benchmarks with substantially fewer parameters.
- **All three rewards are indispensable**: Removing any reward component degrades performance; IntlzR has the greatest impact on reasoning drift.
- **Visual Gate improves reasoning robustness**: Filtering low-quality perceptions improves reasoning accuracy by approximately 3%.

## Highlights & Insights
- **Introducing "knowledge internalization" fills an important gap**: All prior methods optimize either "accurate perception" or "correct reasoning" while neglecting the bridge between them. CogFlow demonstrates that this bridge (internalization) is critical — it directly reduces reasoning drift by 19%.
- **The five-category taxonomy of negative samples is practically useful**: The systematic classification of reasoning drift into five failure modes (omission of primitives, fabrication of facts, misuse of theorems, constraint violations, and inconsistent references) provides an analytical framework for future research.
- **The Visual Gate design is transferable to other multimodal RL scenarios**: The idea of actively filtering low-quality intermediate outputs before subsequent generation — a "quality gating" mechanism — is applicable to all multi-stage generation tasks.

## Limitations & Future Work
- Scope is limited to visual mathematical reasoning; performance on natural image understanding and VQA has not been evaluated.
- IntlzR training requires carefully constructed positive-negative sample pairs; extending to new domains necessitates redesign.
- The Visual Gate threshold $\tau$ requires manual tuning and may need adjustment across different tasks.
- The three-stage pipeline introduces inference latency (each stage — perception, internalization, and reasoning — requires independent generation).
- The MathCog dataset primarily covers geometry problems, with insufficient coverage of algebra and statistical charts.

## Related Work & Insights
- **vs. MathFlow (Chen et al.)**: Also a decoupled pipeline but lacks the internalization stage, leading to pronounced reasoning drift; CogFlow's IntlzR effectively resolves this.
- **vs. VLM-R1 (Shen et al.)**: The one-step framework cannot structurally manage perception and reasoning separately; CogFlow's three-stage design provides explicit division of responsibilities.
- **vs. OVR (Wei et al.)**: Also employs two-stage multimodal RL but lacks an explicit mechanism for perception-reasoning alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Knowledge internalization" is introduced to visual reasoning for the first time; the three-stage cognitive framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-benchmark evaluation, ablation studies, quantitative analysis of reasoning drift, and comparisons with closed-source models.
- Writing Quality: ⭐⭐⭐⭐⭐ Cognitive science motivation is clearly articulated; figures are well-designed; problem-solution correspondence is explicit.
- Value: ⭐⭐⭐⭐⭐ The 120K dataset and open-source code make a significant contribution to the visual reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](../../AAAI2026/optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[ICLR 2026\] Learning to Solve Orienteering Problem with Time Windows and Variable Profits](learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](../../ICML2026/optimization/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)
- [\[ICML 2026\] Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization](../../ICML2026/optimization/asymmetric_perturbation_in_solving_bilinear_saddle-point_optimization.md)

</div>

<!-- RELATED:END -->
