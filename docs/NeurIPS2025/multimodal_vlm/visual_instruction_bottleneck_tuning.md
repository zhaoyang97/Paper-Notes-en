---
title: >-
  [Paper Note] Visual Instruction Bottleneck Tuning
description: >-
  [NeurIPS 2025][Multimodal VLM][Information Bottleneck] This paper is the first to apply the Information Bottleneck (IB) principle to end-to-end instruction tuning of multimodal large language models. It proposes Visual I…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Information Bottleneck"
  - "Multimodal Large Language Models"
  - "Distribution Shift Robustness"
  - "Instruction Tuning"
  - "Representation Learning"
date: 2026-05-08
content_hash: 6b048e661653bcbc
---

# Visual Instruction Bottleneck Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.13946](https://arxiv.org/abs/2505.13946)  
**Authors**: Changdae Oh, Jiatong Li, Shawn Im, Sharon Li (University of Wisconsin–Madison)
**Code**: [deeplearning-wisc/vittle](https://github.com/deeplearning-wisc/vittle)  
**Area**: Multimodal VLM
**Keywords**: Information Bottleneck, Multimodal Large Language Models, Distribution Shift Robustness, Instruction Tuning, Representation Learning

## TL;DR

This paper is the first to apply the Information Bottleneck (IB) principle to end-to-end instruction tuning of multimodal large language models. It proposes Visual Instruction Bottleneck Tuning (Vittle), which inserts a lightweight bottleneck layer inside the LLM to learn minimally sufficient representations. Vittle consistently improves robustness across 30 distribution shift scenarios without sacrificing performance on standard benchmarks.

## Background & Motivation

### State of the Field
Despite achieving strong performance on standard benchmarks, Multimodal Large Language Models (MLLMs) remain brittle under distribution shifts—including minor perturbations such as brightness/contrast changes in images, typos in text, and long-tail distribution samples. This stands in stark contrast to human intelligence, which compresses rich perceptual inputs into compact abstract representations, remaining invariant to low-level surface features while remaining sensitive to high-level abstractions.

### Limitations of Prior Work
- **Data-centric approaches**: Collecting more instruction data (e.g., SVit, LLaVA-1.5) incurs substantial annotation costs.
- **Model-centric approaches**: Scaling model size or adopting stronger backbones (e.g., Eagle, Qwen2-VL) demands considerable computational resources.
- **Existing IB work**: The Information Bottleneck has primarily been explored in small-scale classification tasks, or applied only to the projection layer with the LLM backbone frozen, without end-to-end integration.
- **Representation-level analysis**: Oh et al. (2025) find that the embedding distances between perturbed and clean samples in MLLM internal representation spaces are excessively large, indicating a lack of invariance to surface-level changes.

### Root Cause
From a representation learning perspective, rather than scaling data or models, this work regularizes MLLM internal representations via the IB principle—discarding input-specific redundant information and retaining only task-relevant information—so as to achieve a better balance between invariance and sensitivity.

## Method

### Information Bottleneck Objective
Given a multimodal input $X=(X_v, X_t)$, target output $Y$, and intermediate representation $Z=f(X)$, the IB objective is:

$$\max_f \text{IB}_f(X,Y) := I(Z,Y) - \beta \cdot I(Z,X)$$

where $I(Z,Y)$ preserves task-relevant information and redundant information in $I(Z,X)$ is compressed.

### Variational Lower Bound Derivation
Direct optimization of the IB is intractable. This paper derives a specialized variational lower bound tailored to the autoregressive multimodal structure of MLLMs:

1. **Upper bound on the compression term**: Exploiting the property of causal masking, $p(z_v|x_v,x_t)=p(z_v|x_v)$, $I(Z,X)$ is decomposed into two KL divergence terms for the visual and textual modalities:
$$I(Z,X) \leq \mathbb{E}_{x_v}[D_{\text{KL}}(p(z_v|x_v)\|r(z_v))] + \mathbb{E}_{x_v,x_t}[D_{\text{KL}}(p(z_t|x_v,x_t)\|r(z_t))]$$

2. **Lower bound on the prediction term**: A variational approximation $q(y|z)$ is introduced in place of the true posterior $p(y|z)$:
$$I(Z,Y) \geq \mathbb{E}_{x,y}[\mathbb{E}_{z|x}[\log q(y|z)]]$$

3. **Final objective**:
$$\mathcal{L}_\beta = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{z|x^i}[\log q(y^i|z)] - \beta(D_{\text{KL}}(p(z_v|x_v^i)\|r(z_v)) + D_{\text{KL}}(p(z_t|x_v^i,x_t^i)\|r(z_t)))$$

### Vittle Architecture
- **Bottleneck layer position**: Inserted after layer $l$ of the LLM (default: layer 24 out of 32, i.e., the top 25%).
- **Posterior distribution modeling**: A separate MLP $g_{\phi}:\mathbb{R}^d \to \mathbb{R}^{2d}$ is used for visual and text tokens respectively, outputting mean $\mu$ and variance $\sigma^2$ to define a diagonal Gaussian posterior.
- **Sampling and interpolation**: Samples are drawn via the reparameterization trick as $\tilde{z}=\mu+\sigma\odot\epsilon$, then interpolated with the original representation as $\hat{z}=(1-\alpha)z+\alpha\tilde{z}$, with $\alpha$ increased to 0.5 via a cosine schedule.
- **Prior distribution**: Two variants—Vittle (F) uses a fixed standard Gaussian $\mathcal{N}(0,I)$; Vittle (L) uses a learnable Gaussian $\mathcal{N}(\mu_\psi, \sigma_\psi^2 \cdot I)$.
- **Inference**: The deterministic posterior mean $\tilde{z}=\mu$ is used, with $\hat{z}=(z+\tilde{z})/2$.
- **Hyperparameters**: $\beta=0.1/d$ (where $d$ is the hidden dimension); only 1.5% additional parameters are introduced.

### Theoretical Support: EMID Upper Bound
This paper proves that Vittle's learning objective is connected to an upper bound on EMID (Effective Mutual Information Discrepancy), a metric measuring robustness degradation of MLLMs under distribution shift. The EMID upper bound decomposes into a product and sum of output entropy and Jensen–Shannon Divergence (JSD) between representation distributions. Vittle reduces the JSD between clean and perturbed samples via representation compression, thereby lowering EMID.

## Key Experimental Results

### Experiment 1: Perturbation Robustness (LB-COCO and 27 Variants)

Twenty-seven perturbations (9 visual, 9 textual, 9 joint) are applied to LB-COCO, and relative preference scores are evaluated using GPT-4o as judge.

| Method | Clean | V Pert. | T Pert. | J Pert. |
|--------|-------|---------|---------|---------|
| Baseline | 77.8 | 73.4 | 72.2 | 62.3 |
| LoRA | 73.4 | 70.4 | 62.7 | 39.7 |
| Weight Decay | 74.1 | 72.1 | 73.0 | 59.5 |
| **Vittle (L)** | 76.7 | 73.9 | 73.0 | 62.7 |
| **Vittle (F)** | 76.1 | **74.2** | **74.1** | **64.4** |

Vittle (F) achieves gains of +1.9 and +2.1 on text and joint perturbations respectively, substantially outperforming parameter-space regularization methods (LoRA and Weight Decay).

### Experiment 2: Cross-Task and Cross-Architecture Validation

**Long-tail open-ended QA** (relative preference scores):

| Method | LB-Wild | LB-Wilder | WV-Bench |
|--------|---------|-----------|----------|
| Baseline | 51.6 | 156.9 | 60.0 |
| Vittle (L) | **54.6** | **168.8** | **60.4** |
| Vittle (F) | 52.2 | 166.1 | 59.7 |

**General benchmarks (closed-ended QA)**:

| Method | SciQA | MMMU | MME | MMStar | Avg. |
|--------|-------|------|-----|--------|------|
| Baseline | 64.6 | 35.6 | 69.7 | 33.7 | 50.9 |
| Vittle (L) | 64.7 | 35.3 | 70.5 | 33.7 | 51.1 |
| Vittle (F) | **65.4** | 34.5 | 70.1 | 33.5 | 50.9 |

**Cross-architecture generalization (POPE hallucination detection)**:

| Backbone | Method | POPE Clean | POPE Shifts Avg. |
|----------|--------|------------|-----------------|
| LLaVA-Mini | Baseline | 79.37 | 77.39 |
| LLaVA-Mini | Vittle (F) | **81.07** | **78.32** |
| LLaVA++ (Llama3-8B) | Baseline | 84.60 | 80.54 |
| LLaVA++ (Llama3-8B) | Vittle (F) | **85.87** | **84.08** |

**Representation space analysis (averaged over 27 LB-COCO perturbations)**:

| Method | JSD (↓) | EMID (↓) |
|--------|---------|----------|
| Baseline | 0.068 | 0.026 |
| Vittle (L) | 0.048 | 0.021 |
| Vittle (F) | **0.047** | 0.025 |

Vittle reduces the JSD between clean and perturbed samples from 0.068 to 0.047 (a 31% reduction), confirming that representation compression genuinely enhances invariance.

## Highlights & Insights

- **Pioneering perspective**: This is the first work to introduce the IB principle into end-to-end MLLM instruction tuning, establishing a new paradigm for robustness enhancement via representation compression—complementary to conventional data/model scaling approaches.
- **Unified theory and practice**: A variational lower bound for the IB is derived specifically for autoregressive multimodal architectures, and a theoretical connection to the EMID robustness metric is established. The two prior variants each suit different scenarios.
- **Consistent gains at minimal cost**: With only a 1.5% parameter increase and ~20% additional training time, and negligible inference overhead, Vittle consistently improves robustness across 30 distribution shifts, 45 datasets, and multiple MLLM architectures.
- **Compelling qualitative analysis**: PCA visualizations and cosine distance histograms intuitively demonstrate how Vittle draws perturbed samples closer to clean samples, forming a more compact representation space.

## Limitations & Future Work

- **Dependence on annotation quality**: The IB objective uses response $Y$ as an anchor for "sufficient" information, but LLM-generated instruction data is often noisy; the benefits of IB may diminish under noisy annotations.
- **Slight degradation in OCR capability**: Information compression may harm fine-grained character recognition while enhancing high-level semantic robustness.
- **No guarantee for counterfactual/cross-domain generalization**: For counterfactual samples with conflicting visual-language priors or entirely different domains, IB alone cannot guarantee generalization.
- **Validated only at 7B–13B scale**: The approach has not been verified on larger models (e.g., 70B+) or closed-source models.
- **No adaptive mechanism for prior selection**: Vittle (F) performs better under perturbation scenarios while Vittle (L) excels in long-tail settings, but no automatic switching mechanism is provided.
- **Bottleneck layer placement**: The default top-25% layer is used; multi-level or adaptive placement strategies remain largely unexplored.

## Related Work & Insights

- **Alemi et al. (2017) VIB**: The classical VIB is applied only to small-scale classification models; this paper is the first to extend IB to large-scale autoregressive multimodal models.
- **Bai et al. (2025)**: Applies IB training at the projection layer with the LLM backbone frozen, achieving only shallow adaptation; this work directly modifies the LLM's internal structure for end-to-end IB.
- **ROSS & LIT (information maximization direction)**: Conceptually opposite to Vittle's compression-based design; effective on POPE hallucination detection but less so on open-ended QA, suggesting that compression is more generally applicable than maximization.
- **LoRA / Weight Decay (parameter-space regularization)**: LoRA exhibits substantial performance degradation under perturbations (joint perturbations drop from 62.3 to 39.7), demonstrating that parameter-space regularization cannot substitute for information control in representation space.
- **Oh et al. (2025) EMID**: Proposes an information-theoretic metric for MLLM robustness; this paper builds upon it by proving that Vittle reduces the EMID upper bound and validates this empirically.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First end-to-end integration of the IB principle into MLLM instruction tuning; both the theoretical derivation and architectural design represent entirely original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 45 datasets, 30 distribution shifts, multiple MLLM architectures, and extensive ablations; the experimental scale and coverage are exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, theory is self-consistent, quantitative and qualitative analyses are mutually corroborating, and figures are well-designed.
- Value: ⭐⭐⭐⭐ — Provides a practical robustness enhancement solution, though validation remains limited to 7B–13B scale and minor OCR degradation is observed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)

</div>

<!-- RELATED:END -->
