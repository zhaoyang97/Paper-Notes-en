---
title: >-
  [Paper Note] Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs
description: >-
  [ICLR 2026][LLM Safety][Fairness Benchmark] This paper introduces the IRIS Benchmark, the first benchmark to synchronously evaluate fairness in both understanding and generation tasks for Unified Multimodal Large Language Models (UMLLMs). Through a three-dimensional evaluation framework, 60 fine-grained metrics, and a high-dimensional fairness space, IRIS reveals key phenomena such as cross-task "personality splitting" and systematic "generation gaps."
tags:
  - ICLR 2026
  - LLM Safety
  - Fairness Benchmark
  - Unified Multimodal LLM
  - Bias Evaluation
  - Demographic Fairness
  - Generation-Understanding Gap
date: 2026-05-08
content_hash: a5a779eccb796c33
---

# Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs

**Conference**: ICLR 2026
**arXiv**: [2603.00590](https://arxiv.org/abs/2603.00590)
**Code**: None
**Area**: AI Safety / Fairness
**Keywords**: Fairness Benchmark, Unified Multimodal LLM, Bias Evaluation, Demographic Fairness, Generation-Understanding Gap

## TL;DR

This paper introduces the IRIS Benchmark, the first benchmark to synchronously evaluate fairness in both understanding and generation tasks for Unified Multimodal Large Language Models (UMLLMs). Through a three-dimensional evaluation framework, 60 fine-grained metrics, and a high-dimensional fairness space, IRIS reveals key phenomena such as cross-task "personality splitting" and systematic "generation gaps."

## Background & Motivation

**Background**: The AI fairness field faces a "Tower of Babel" dilemma — over twenty fairness metrics exist (covering individual fairness, group fairness, and causal fairness), yet their underlying philosophical assumptions conflict (e.g., "unawareness fairness" requires ignoring demographic attributes, while "awareness fairness" requires accounting for them). The fairness impossibility theorem (Hsu et al., 2022) proves that simultaneously satisfying multiple definitions is mathematically infeasible in general. **Limitations of Prior Work**: UMLLMs such as Janus-Pro and BLIP3-o integrate understanding and generation within a shared representation space, enabling biases to propagate systematically across tasks (intrinsic biases in core embeddings transfer to downstream tasks). However, existing evaluation tools are single-task and cannot capture this cross-task dependency. Existing unified benchmarks (e.g., UnifiedBench) focus on capability evaluation (instruction following) and entirely neglect the value-level fairness dimension. **Key Challenge**: A model may "cognize fairness" on understanding tasks without translating that into "acting fairly" on generation tasks — a shared representation space does not guarantee cross-task fairness consistency. **Goal**: To construct a framework that synchronously evaluates UMLLMs' fairness across both generation and understanding task dimensions, and to consolidate fragmented fairness metrics into a unified evaluation paradigm. **Key Insight**: Rather than pursuing a single "optimal" fairness definition, the paper adopts a multi-objective trade-off analysis — projecting multiple fairness metrics into a high-dimensional space, measuring bias magnitude by distance from the origin, and characterizing a model's "fairness personality" through score distributions across dimensions. **Core Idea**: A six-sector high-dimensional fairness space defined by three dimensions (Ideal Fairness, Real-World Fidelity, Bias Inertia & Steerability) × two tasks (understanding, generation) is used to comprehensively diagnose the fairness profile of UMLLMs.

## Method

### Overall Architecture

The IRIS evaluation pipeline proceeds as follows: (1) execute understanding and generation tasks separately on the target UMLLM and collect outputs; (2) annotate demographic attributes of generated images using the ARES classifier; (3) compute 60 fine-grained sub-metrics; (4) convert raw metrics into six sector scores and a composite IRIS Score via normalization, dimensional aggregation, and exponential decay mapping; (5) assign IRIS-MBTI personality labels based on score patterns. Three demographic attributes are covered: gender (2 categories), age (3 categories: young 0–39, middle-aged 40–64, elderly ≥65), and skin tone (3 categories based on the 10-level Monk Skin Tone scale).

### Key Designs

1. **Three-Dimensional Evaluation Framework**:

    - **Function**: Constructs a complete fairness diagnostic chain from the "normative world" to the "actual world" to the "actionable world."
    - **Mechanism**: **Ideal Fairness Score (IFS)** assesses default model behavior under neutral prompts — on the generation side via Representation Disparity (RD); on the understanding side via cross-group accuracy (AD) and Statistical Parity Difference (SPD). **Real-World Fidelity Score (RFS)** evaluates whether model cognition reflects true demographic distributions — both generation and understanding sides use Jensen-Shannon Divergence (JSD) to measure deviation from real distributions. **Bias Inertia & Steerability Score (BIS)** quantifies the feasibility of guiding models toward improvement — the generation side detects whether anti-stereotyping instructions incur quality penalties ($\Delta$GSR, QPS, SIL), while the understanding side detects whether anti-stereotyping evidence disrupts judgment (AC_diff, DHR).
    - **Design Motivation**: The three dimensions correspond to three major philosophical stances in the fairness literature (group fairness / equal opportunity / counterfactual fairness), forming a complete chain of "default behavior → real-world cognition → controllable execution." The 60 sub-metrics are generated through cross-combinations of demographic attributes to reveal deep biases missed by single-dimensional analysis.

2. **High-Dimensional Fairness Space Scoring Mechanism**:

    - **Function**: Unifies heterogeneous raw metrics into comparable fairness scores.
    - **Mechanism**: (a) *Normalization*: each raw metric is mapped to a unified deviation space where the ideal state is the origin (the "fairness singularity" $\mathbf{u}=\mathbf{0}$); bounded metrics are linearly scaled, and unbounded penalties are logarithmically compressed. (b) *Dimensional aggregation*: the L2 norm $M_{\text{dim}} = \|\mathbf{u}^{(\text{dim})}\|_2$ is computed for each dimension as a bias magnitude. (c) *Exponential decay mapping*: $\widehat{S}_{\text{dim}} = S_{\text{dim}} \cdot \exp(-K_{\text{dim}} \cdot M_{\text{dim}})$, mapping bias into interpretable scores.
    - **Design Motivation**: Rather than pursuing a single optimal solution, evaluation is reframed as a Pareto analysis in multi-objective optimization — different application scenarios (e.g., children's picture books vs. social science research) can prioritize different dimensions accordingly.

3. **ARES Classifier (Adaptive Routing Expert System)**:

    - **Function**: Performs large-scale automatic demographic attribute classification on generated images.
    - **Mechanism**: An adaptive dual-path architecture — the Fast Path uses an L1 lightweight expert pool (CLIP, DINOv2, ConvNeXt, etc., fine-tuned on the IRIS-Classifier-25 dataset) to handle simple samples; the Complex Path uses L2 heavyweight experts (InternVL-1B + MLP fusion head) to handle difficult or ambiguous samples. An intelligent routing network automatically assesses sample difficulty and routes accordingly. Overall accuracy is 88%.
    - **Design Motivation**: A single VLM classifier is unstable on common artifacts in generated images; adaptive routing achieves a balance between accuracy and efficiency.

### Loss & Training

The L1 experts of the ARES classifier are fine-tuned on the IRIS-Classifier-25 dataset (250,000 images, including 10% adversarial samples). The evaluation itself does not involve training — all metrics are computed directly from model inference outputs. IRIS-MBTI diagnosis generates a three-letter code (U/H + A/D + F/R) by comparing each dimensional score against thresholds, mapped to 8 personality archetypes.

## Key Experimental Results

### Main Results

**Comprehensive Fairness Evaluation of UMLLMs** (Table 3, higher is better):

| Model | IFS_Und | RFS_Und | BIS_Und | IFS_Gen | RFS_Gen | BIS_Gen | IRIS Score |
|-------|---------|---------|---------|---------|---------|---------|-----------|
| Bagel | 71.46 | 69.81 | 50.75 | **82.58** | 69.13 | 60.91 | **95.94** |
| BLIP3-o | 62.14 | **74.81** | 60.95 | 35.30 | 34.68 | **78.82** | 40.13 |
| Harmon | **74.44** | 57.34 | 35.76 | 49.96 | 60.50 | 49.97 | 52.49 |
| Janus-Pro | 32.84 | 56.89 | **105.22** | 56.78 | 42.45 | 69.30 | 67.97 |
| Show-o | 68.32 | 58.64 | 85.15 | 70.03 | 68.22 | 54.57 | 60.01 |
| VILA-U | 39.94 | 60.80 | 64.90 | 59.87 | 40.68 | 64.97 | 60.69 |

### Ablation Study

**Framework Validation** (structural integrity checks):

| Validation Item | Result | Implication |
|----------------|--------|-------------|
| Internal consistency (Cronbach's α) | Mostly >0.7 (except BIS_Gen=0.20) | Metrics within a dimension measure the same construct |
| Hyperparameter robustness (Spearman ρ) | >0.96 | Rankings are insensitive to parameter choices |
| Construct validity (inter-dimension correlation) | Generally low | Three dimensions measure distinct aspects |
| Architectural fairness (Welch's t-test) | p=0.76 | No preference for specific architectures |

### Key Findings

- **Systematic "Generation Gap"**: UMLLMs are competitive on understanding tasks, but generation fairness consistently lags far behind dedicated text-to-image models (e.g., FLUX.1-dev achieves IFS_Gen=94.05).
- **Cross-Task "Personality Splitting"**: VILA-U is classified as HAF (Heuristic Adaptive Reformer) on the understanding side but UDF (Utilitarian Decisive Follower) on the generation side — a shared representation space does not guarantee cross-task fairness consistency.
- **Inter-Dimensional Trade-offs**: RFS_Gen and BIS_Gen exhibit strong negative correlation (ρ=−0.80), indicating an inherent tension between real-world fidelity and steerability.
- **Bias Bottleneck Localization**: BLIP3-o's bias is traced to the projection layer between the AR model and the diffusion decoder (Distortion≈1.4); Harmon's bias is traced to a "snowball effect" in the first 10 autoregressive steps of the MAR decoder.
- **"Anti-Stereotype Reward"**: A counterintuitive finding — anti-stereotyping prompts actually improve output quality and semantic fidelity for most models.

## Highlights & Insights

- The first dual-task synchronous fairness benchmark for UMLLMs, shifting evaluation from "capability-oriented" to "value-oriented."
- The high-dimensional fairness space methodology provides a practical resolution to the fairness impossibility theorem — rather than seeking a single optimum, it maps the trade-off space.
- The IRIS-MBTI personality diagnosis enables intuitive rapid model comparison — distinguishing models with similar IRIS Scores but fundamentally different fairness profiles (e.g., VILA-U vs. Show-o).
- Mechanistic probing traces evaluation results back to architectural bottlenecks, demonstrating the value of benchmarking as a "diagnostic tool" rather than merely a "ranking tool."

## Limitations & Future Work

- Demographic attribute encoding is coarse: binary gender, broad age brackets, and skin tone groupings may overlook the complexity of intersectionality and continuous identities.
- Automated ARES annotation introduces measurement noise and potential classifier bias (88% accuracy), with no human validation stage.
- Cronbach's α for the BIS_Gen dimension is only 0.20, indicating that "steerability" comprises two distinct sub-constructs: "willingness" and "capability."
- Experiments are confined to 52 occupation prompts related to image generation and VQA-style understanding tasks, without covering other modalities and task scenarios.

## Related Work & Insights

- **vs. FairFace/BBQ**: Traditional fairness benchmarks evaluate only a single modality/task; IRIS is the first to synchronously evaluate both understanding and generation.
- **vs. UnifiedBench**: Capability-oriented unified benchmarks focus on task performance such as instruction following; IRIS focuses on systematic fairness diagnosis at the value level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first dual-task synchronous fairness benchmark for UMLLMs; the IRIS-MBTI diagnostic system and high-dimensional fairness space are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 models, 60 sub-metrics, and four-fold framework validation, though BIS_Gen internal consistency is low.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly articulated; metaphors such as "Tower of Babel," "fairness singularity," and "personality splitting" are vivid, though terminology density is high.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in UMLLM fairness evaluation; mechanistic probing and the anti-stereotype reward finding offer direct practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Designing Truthful Mechanisms for Asymptotic Fair Division](../../AAAI2026/llm_safety/designing_truthful_mechanisms_for_asymptotic_fair_division.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ICLR 2026\] LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)
- [\[NeurIPS 2025\] When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](../../NeurIPS2025/llm_safety/when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)

</div>

<!-- RELATED:END -->
