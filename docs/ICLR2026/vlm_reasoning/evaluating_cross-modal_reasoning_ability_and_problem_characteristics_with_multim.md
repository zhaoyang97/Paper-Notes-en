---
title: >-
  [Paper Note] Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory
description: >-
  [ICLR 2026][vlm_reasoning][Item Response Theory] Classic Item Response Theory (IRT) is extended into "modality-decomposed" versions (M2IRT / M3IRT), where model ability and item difficulty are decomposed into "image-only / text-only / cross-modal integration" components. This enables the identification of tasks requiring genuine cross-modal reasoning, the elimination
tags:
  - ICLR 2026
  - vlm_reasoning
  - Item Response Theory
date: 2026-05-08
content_hash: c1014e4e3843e51e
---
# Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory

**Conference**: ICLR 2026  
**Code**: [CyberAgentAILab/M3IRT](https://github.com/CyberAgentAILab/M3IRT)  
**Area**: Multimodal Evaluation / VLM Cross-modal Reasoning / Item Response Theory  
**Keywords**: Item Response Theory, Cross-modal Reasoning, Benchmark Refinement, Computerized Adaptive Testing, shortcut problems  

## TL;DR
Classic Item Response Theory (IRT) is extended into "modality-decomposed" versions (M2IRT / M3IRT), where model ability and item difficulty are decomposed into "image-only / text-only / cross-modal integration" components. This enables the identification of tasks requiring genuine cross-modal reasoning, the elimination of shortcut items solvable via single modalities, and the restoration of model rankings using only 1%–10% of the original benchmark size.

## Background & Motivation
- **Background**: MLLM/VLM evaluation relies on static benchmarks like MMMU, MathVista, and SEED-Bench. Recent works have introduced IRT to LLM evaluation (e.g., TinyBenchmarks, MetaBench) for benchmark compression and Computerized Adaptive Testing (CAT).
- **Limitations of Prior Work**: Current multimodal benchmarks are saturated with **shortcut items**—problems that can be solved by looking only at the image or text, relying on cues from the prompt or options. These low-quality items inflate benchmark size, increase evaluation costs, and pollute rankings by masking genuine "cross-modal integration" capabilities.
- **Key Challenge**: Classic IRT is **modality-agnostic**, utilizing only a single latent variable for ability/difficulty. It cannot distinguish whether a successful response is due to true cross-modal reasoning or a unimodal shortcut.
- **Goal**: Construct an evaluation framework that can **quantify the "cross-modal difficulty" of each item and the "cross-modal ability" of each model**, selecting high-quality subsets to reduce costs and enhance ranking reliability.
- **Core Idea**: **[Modality-Decomposed IRT]** Ability $\theta$, difficulty $b$, and discrimination $a$ are decomposed into four components: {base, image, text, cross}, making "cross-modal" a separately estimable latent dimension.

## Method

### Overall Architecture
Given $m$ VLMs (subjects) and $n$ items, a response tensor $R'=\{r_{i,j,s}\}$ is collected, where $r_{i,j,s}\in\{0,1\}$ denotes whether model $i$ correctly answers item $j$ under format $s$. The formats $s=(s_{\text{image}}, s_{\text{text}})\in\{(0,0),(0,1),(1,0),(1,1)\}$ control the combinations of image/text availability. Parameters of the modality-decomposed IRT are first fitted via SGD, followed by adaptive item selection via CAT based on Fisher Information to derive compact, high-quality subsets for new models.

```mermaid
flowchart LR
    A[24 VLMs × Multimodal Benchmarks] --> B[Questions in 4 formats<br/>image/text/both/neither]
    B --> C[Response Tensor R']
    C --> D[Modality-decomposed IRT<br/>SGD estimate base+image+text+cross for θ/a/b]
    D --> E[Item cross-modal difficulty b_cross<br/>Model cross-modal ability θ_cross]
    E --> F[CAT + Fisher Information<br/>Adaptive selection of high-quality subsets]
    F --> G[Subsets restore ranking + eliminate shortcuts]
```

### Key Designs
**1. Modality-decomposed ability/difficulty/discrimination: Estimating "Cross-modality" as a latent variable.** This is the foundation of the paper. For model $i$, ability is split into four non-negative components: base, image, text, and cross. Given format $s$, the effective ability is $\theta_i(s)=\theta_i^{\text{base}}+s_{\text{image}}\theta_i^{\text{image}}+s_{\text{text}}\theta_i^{\text{text}}+s_{\text{image}}s_{\text{text}}\theta_i^{\text{cross}}$—cross-modal ability is activated only when both image and text are present (the fourth term). Item difficulty is decomposed via a symmetric "subtraction": $b_j(s)=b_j^{\text{base}}-s_{\text{image}}b_j^{\text{image}}-s_{\text{text}}b_j^{\text{text}}-s_{\text{image}}s_{\text{text}}b_j^{\text{cross}}$ (providing more modalities acts as a "hint" that lowers difficulty). Discrimination $a_j(s)$ follows a similar additive four-component structure. Consequently, $b_j^{\text{cross}}$ directly characterizes the item's reliance on cross-modal integration.

**2. Two implementations: M2IRT vs. M3IRT.** M2IRT incorporates the decomposition into a 2-parameter logistic (2PL) model, where $z_{i,j,s}=a_j(s)\big(\theta_i(s)-b_j(s)\big)$ and $\hat P(r_{i,j,s}=1)=\sigma(z_{i,j,s})$, offering scalar parameterization and interpretability. M3IRT follows a Multidimensional IRT (MIRT) approach, representing components as vectors $\theta_i, a_j, b_j \in \mathbb{R}^4$. Introducing a format indicator vector $s=[1, -s_{\text{image}}, -s_{\text{text}}, -s_{\text{image}}s_{\text{text}}]^\top$, it defines $z'_{i,j,s}=a_j^\top \operatorname{diag}(s)\theta_i - s^\top b_j$. The vector form allows latent dimensions to couple during fitting, providing higher expressive power. In experiments, M3IRT typically outperforms M2IRT in ranking restoration for minimal subsets.

**3. SGD Estimation instead of EM: Native support for sparse responses.** Eschewing the traditional EM algorithm used in IRT, Ours minimizes the Bernoulli negative log-likelihood $L(\Theta)=-\sum_{(i,j,s)\in R''}\big[r_{i,j,s}\log\hat P(\cdot=1)+(1-r_{i,j,s})\log\hat P(\cdot=0)\big]$ using mini-batch SGD + Adam. This approach **does not require a dense response matrix**—it can learn parameters from partial observations similar to tensor completion, saving the immense computational cost of evaluating every model on every item in every format.

**4. CAT + Fisher Information for adaptive item selection.** Once parameters are fitted, items are selected via Computerized Adaptive Testing. M2IRT uses scalar Fisher Information $I_{i,j}=\hat P(1)\hat P(0)\,a_j(s)^2$. M3IRT uses the Fisher Information Matrix $I_{i,j}=\hat P(1)\hat P(0)\,(\operatorname{diag}(s)a_j)(\operatorname{diag}(s)a_j)^\top$ with a **D-optimality criterion**: at step $t$, the item $j^*=\arg\max_{j}\det\!\big(I_i^{(t-1)}+I_{ij}\big)$ is selected from the unanswered set $U_i$. This process prioritizes items with high cross-modal difficulty and naturally excludes shortcut items.

## Key Experimental Results
- **Setup**: 24 VLMs (including GPT-4.1, Gemini-2.0, Claude-3.7, Qwen-2.5-VL, Llama-3.2, Pixtral, etc.) across three benchmarks (MMMU 900 items, MathVista 1000 items, SEED-Bench 1000 items). Semi-synthetic "polluted" benchmarks are constructed by manually generating 50% low-quality items via swapping images/text. Baselines: Random, IRT, MIRT, TinyBenchmarks, FlashEval. Metrics: Spearman correlation $\rho$ (between subset and original benchmark rankings) and low-quality item ratio $\gamma$ in the subset.

### Main Results (Ranking Restoration $\rho$)

| Benchmark | Method | Subset Size | Spearman $\rho$ |
|-----------|--------|-------------|-----------------|
| MMMU      | M2IRT  | 3%          | 0.9             |
| MMMU      | M3IRT  | 1%          | 0.8             |
| MathVista | M3IRT  | 2%          | 0.84            |
| MathVista | M3IRT  | 30%         | 0.9             |
| SEED-Bench| M2IRT  | 3%          | 0.9             |
| SEED-Bench| M3IRT  | 1%          | 0.9             |

> The SOTA baseline FlashEval performs similarly to Random as it ignores low-quality items. Conclusion: Only a **10% subset** is required to nearly perfectly restore original rankings across all datasets.

### Ablation / Shortcut Ratio ($\gamma$, lower is better)

| Benchmark | Subset Size | M3IRT $\gamma$ | Baseline Comparison |
|-----------|-------------|----------------|---------------------|
| MMMU      | 50%         | 24%            | Significantly lower than baselines (which select more shortcuts) |
| Various   | Various     | Generally < half of baselines | Rankings are less distorted by shortcuts |

### Robustness (ROC-AUC for Predicting Missing Responses)
With the low-quality item ratio varying from 0% to 100%, the AUC of M2IRT/M3IRT is comparable to standard IRT (MMMU $\approx 0.78$–$0.80$, MathVista $\approx 0.88$–$0.89$, SEED-Bench $\approx 0.81$–$0.83$). M2IRT slightly outperforms IRT on MMMU, indicating that modality decomposition does not sacrifice fitting quality.

### Key Findings
- On MMMU, the top-ranked model has a high $\theta^{\text{cross}}$ (genuine cross-modal strength). Models in 2nd and 3rd place exhibit high $\theta^{\text{text}}$ but weak $\theta^{\text{cross}}$, suggesting they **rely on text comprehension shortcuts** rather than true visual integration.
- On MathVista, most VLMs show high $\theta^{\text{text}}$, confirming the benchmark's bias toward linguistic reasoning.
- Items with low $b_j^{\text{cross}}$ can indeed be solved via image or text alone (e.g., an MMMU question solvable solely through artist knowledge), validating the interpretability of the decomposition.

## Highlights & Insights
- **Upgrading modality-agnostic IRT to modality-aware**: Using additive/subtractive decomposition allows "cross-modality" to be estimated and ranked as a separate dimension. The approach is elegant and "plug-and-play."
- **Dual diagnosis of items and models**: $b^{\text{cross}}$ identifies true cross-modal questions, while $\theta^{\text{cross}}$ identifies true cross-modal capability—providing quantitative evidence for models that "game" benchmarks via shortcuts.
- **Extreme evaluation efficiency**: Achieving $\rho \approx 0.8$ with a 1% subset and near-perfect restoration with a 10% subset offers significant cost savings for multimodal evaluation.
- **SGD + Sparse Responses**: Unlike traditional IRT, it does not require a dense response matrix, making it more practical for large-scale engineering.

## Limitations & Future Work
- **Low-quality items via semi-synthetic generation**: The authors acknowledge that more realistic "pollution" (e.g., LLM-rewritten prompts, noisy images) was not included; semi-synthetic scenarios may differ from real-world dirty data.
- **Requirement for four response formats**: M2IRT strictly depends on responses for $s\in\{(0,0),(0,1),(1,0),(1,1)\}$. Although sparse learning is possible, collecting "unimodal" responses still incurs overhead.
- **Focus on Vision-Language**: While the framework is theoretically extendable to audio or action modalities, this was not empirically demonstrated.
- **Hyperparameter Tuning**: The upper bound $q$ requires grid searching via validation set AUC, which may need recalibration for new benchmarks.

## Related Work & Insights
- **IRT for LLM Evaluation**: Directly follows TinyBenchmarks (clustering-based) and MetaBench (distilled sparse benchmarks), advancing them from "unimodal, single latent variable" to "multimodal, decomposed latent variables."
- **Multimodal Benchmarks and Contamination**: While MMMU, MathVista, and SEED-Bench emphasize integration, they suffer from shortcuts. Dynamic benchmarks (VLB/FLEX, LiveXiv) generate items automatically but still contain low-quality data—Ours provides a complementary **post-hoc refinement** perspective.
- **Key Insight**: Psychometric tools like IRT serve as a universal lens for diagnosing benchmark quality. Estimating parameters such as item discrimination and target-ability is valuable for any scenario requiring benchmark compression without ranking distortion.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Decomposing IRT for modality-specific difficulty/ability is a simple yet insightful perspective, though technically an extension of IRT/MIRT.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid across 24 models and 3 benchmarks, covering restoration, shortcut ratios, and AUC robustness. Points deducted for reliance on synthetic low-quality data.
- **Writing Quality**: ⭐⭐⭐ The methodology is clear, but several typos in the text affect readability.
- **Value**: ⭐⭐⭐⭐ Directly addresses cost and reliability in multimodal evaluation, providing a practical toolkit and open-source code for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMReD: A Cross-Modal Benchmark for Dense Context Reasoning](mmred_a_cross-modal_benchmark_for_dense_context_reasoning.md)
- [\[CVPR 2026\] Can a Second-View Image Be a Language? Geometric and Semantic Cross-Modal Reasoning for X-ray Prohibited Item Detection](../../CVPR2026/vlm_reasoning/can_a_second-view_image_be_a_language_geometric_and_semantic_cross-modal_reasoni.md)
- [\[ICLR 2026\] No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers](no_labels_no_problem_training_visual_reasoners_with_multimodal_verifiers.md)
- [\[ICLR 2026\] AutoGPS: Automated Geometry Problem Solving via Multimodal Formalization and Deductive Reasoning](autogps_automated_geometry_problem_solving_via_multimodal_formalization_and_dedu.md)
- [\[ICLR 2026\] Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations](unfolding_spatial_cognition_evaluating_multimodal_models_on_visual_simulations.md)

</div>

<!-- RELATED:END -->
