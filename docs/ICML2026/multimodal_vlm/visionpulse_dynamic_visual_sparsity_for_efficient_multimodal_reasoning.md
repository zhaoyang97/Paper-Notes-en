---
title: >-
  [Paper Note] VisionPulse: Dynamic Visual Sparsification in Multimodal Reasoning
description: >-
  [ICML 2026][Multimodal VLM][Visual token pruning] VisionPulse proposes a training-free step-level dynamic visual token pruning framework—adaptively adjusting the number of retained tokens according to shifting visual dep…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual token pruning"
  - "inference efficiency"
  - "dynamic budget allocation"
  - "multimodal reasoning"
date: 2026-05-08
content_hash: adc3014a2765c6f0
---

# VisionPulse: Dynamic Visual Sparsification in Multimodal Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.31457](https://arxiv.org/abs/2605.31457)  
**Code**: TBD  
**Area**: Multimodal VLM  
**Keywords**: Visual token pruning, inference efficiency, dynamic budget allocation, multimodal reasoning

## TL;DR
VisionPulse proposes a training-free step-level dynamic visual token pruning framework—adaptively adjusting the number of retained tokens according to shifting visual dependencies at each decoding step. It maintains inference accuracy while retaining only 5% of visual tokens and shortening the inference length by 11.2%.

## Background & Motivation

**Background**: Large multimodal models excel in multi-step reasoning tasks, yet inference latency has become a critical bottleneck. Existing visual token compression methods primarily perform a single-pass pruning during the prefilling stage.

**Limitations of Prior Work**: This "static pruning" assumes that the relevance of selected visual tokens remains constant throughout the entire reasoning process. During the prefilling stage, the model's attention to visual information is typically low; a fixed subset chosen at this stage may discard tokens that become crucial in subsequent reasoning steps, while retaining redundant visual context during text-dominant steps.

**Key Challenge**: The demand for visual evidence is highly dependent on the current reasoning state rather than remaining constant. Certain steps require extensive visual evidence, while others are primarily driven by linguistic reasoning.

**Goal**: To design a step-level dynamic visual token pruning framework capable of adjusting the set of retained tokens at each decoding step based on current visual dependencies.

**Key Insight**: Empirical analysis reveals a strong positive correlation between the visual attention quality of the model at each decoding step and the number of effectively activated visual tokens. This lightweight signal can be utilized to predict the optimal budget for each step.

**Core Idea**: Transition visual token pruning from a "one-time prefilling decision" to "step-wise dynamic selection," utilizing visual attention quality to calculate the token retention budget for each step.

## Method

### Overall Architecture
A training-free framework that performs adaptive visual token selection for each generation step during the decoding phase. At each decoding step $t$, it calculates the visual attention quality $M_{\mathrm{vis}}^{t}$, converts it into a token retention budget $K_t$, and selects the top-$K_t$ most critical tokens based on the attention scores between the current query token and visual tokens.

### Key Designs

1. **Step-level Dynamic Visual Token Pruning**:
    - **Function**: Performs visual token selection at every decoding step rather than exclusively during the prefilling stage.
    - **Mechanism**: For the visual token set $X_v = \{v_1, ..., v_N\}$, importance $S_i^t = \frac{1}{H}\sum_{h=1}^{H}A_{t,h}^{(l_a)}(q_t, v_i)$ is calculated at step $t$. The top $K_t$ tokens with the highest scores are selected from $N$ tokens: $X_v^t = \{v_i \mid i \in \text{Top-}K_t(\{S_i^t\}_{i=1}^N)\}$. The critical distinction is that $K_t$ is not a fixed value.
    - **Design Motivation**: Different reasoning steps have vastly different requirements for visual information; static pruning ignores this variance, whereas step-level pruning precisely tracks the fine-grained "when and how much" visual demand.

2. **Visual Attention Quality Guided Dynamic Budget Allocation**:
    - **Function**: Automatically calculates the token retention budget based on the degree of visual dependency at each step.
    - **Mechanism**: Visual attention quality is defined as $M_{\mathrm{vis}}^{t} = \frac{1}{H}\sum_{h=1}^{H}m_{t,h}^{\mathrm{vis}}$, where $m_{t,h}^{\mathrm{vis}} = \sum_{i=1}^{N_v}A_{t,h}^{(l_a)}(q_t, v_i)$. Empirically, $M_{\mathrm{vis}}^{t}$ shows a strong positive correlation (0.82-0.95) with the number of activated tokens. $M_{\mathrm{vis}}^{t}$ is directly transformed into the budget $K_t = M_{\mathrm{vis,max}}^t \cdot N_v$. A temperature scaling factor $\tau < 1$ controls pruning aggressiveness.
    - **Design Motivation**: Utilizing the lightweight signal of attention quality avoids complex token importance predictors; the dynamic budget adaptively retains more tokens in high-demand steps and applies aggressive pruning in low-demand steps.

3. **Inference Computation Cost Analysis and Coupling Bottleneck Identification**:
    - **Function**: Quantifies the dual costs of visual redundancy in multimodal reasoning.
    - **Mechanism**: Total FLOPs are estimated as $\mathcal{F}_{\text{total}} \approx L \cdot [(p+v)(8d^2+4md)+4d(p+v)^2]_{\text{prefill}} + L \cdot \sum_{t=1}^{g}[(8d^2+4md)+4d(p+v+t)]_{\text{decoding}}$. The decoding phase cost exhibits quadratic complexity relative to both generation length $g$ and initial context $(p+v)$; visual tokens dominate in multimodal scenarios where $v \gg p$.
    - **Design Motivation**: Retaining the full visual context not only increases computation but also induces the model to be influenced by query-irrelevant visual cues at each step, leading to unnecessary reasoning steps or even erroneous reasoning paths.

## Key Experimental Results

### Main Results

| Method | Retention Ratio | CharXiv Length ↓ | Accuracy ↑ | InfoVQA Length ↓ | Accuracy ↑ | ChartQA Length ↓ | Accuracy ↑ | Avg Length Change | Avg Accuracy |
|------|-------------------|------------------------|--------|------------------|--------|------------------|--------|---------------------|---------|
| Baseline (Full) | 100% | 4068.0 | 47.60% | 623.1 | 84.37% | 510.0 | 77.12% | - | - |
| VisionZip | ≤10% | 4986.2 | 13.90% | 2533.3 | 22.66% | 2039.7 | 30.24% | +54.2% | -39.7% |
| FastV | ≤10% | 5960.1 | 12.70% | 2963.6 | 20.63% | 1485.5 | 16.28% | +63.2% | -47.6% |
| LOOK-M | ≤10% | 5555.2 | 19.80% | 2694.1 | 40.94% | 2007.1 | 57.68% | +54.2% | -24.5% |
| **VisionPulse** | **≤10%** | **3770.7** | **47.30%** | **530.7** | **83.62%** | **422.9** | **76.72%** | **-12.3%** | **-0.6%** |
| **VisionPulse** | **≤5%** | **3645.1** | **45.20%** | **665.0** | **81.90%** | **510.0** | **75.16%** | **-11.2%** | **-1.8%** |

### Ablation Study

| Config | Avg Retention | RealWorld QA Acc | MMVet Acc | MIA-Bench Acc | Avg Gen Length Reduction | Avg Acc Change |
|-----------|------------------|--------|--------|--------|---------------------|------------|
| Full Model | 100% | 72.81% | 60.96% | 93.44% | - | - |
| FastV Static | 5.0% | 54.12% | 24.27% | 75.03% | +22.2% | -32.5% |
| VisionPulse Fixed 1% | ~1% | 71.90% | 49.17% | 92.03% | +27.9% | -6.2% |
| VisionPulse Fixed 5% | 5.0% | 72.81% | 59.45% | 93.22% | -7.6% | -0.8% |
| VisionPulse Random Budget | 3.0% | 69.28% | 58.02% | 91.49% | +0.2% | -3.7% |
| **VisionPulse Dynamic** | **1.9%** | **72.54%** | **59.00%** | **95.09%** | **-16.6%** | **-0.3%** |

### Key Findings
- Under the extreme pruning setting of ≤5% visual token retention, VisionPulse almost entirely preserves original performance (accuracy drop of only 0.3-1.8%), whereas existing static pruning methods experience accuracy drops of 24.5%-50.9%.
- By removing truly irrelevant visual information based on the actual demand of each step, VisionPulse reduces the average reasoning length by 11.2%-12.3%.
- Incorrect pruning strategies lead to a paradoxical phenomenon: they simultaneously reduce accuracy and increase inference costs (e.g., LOOK-M with 5% retention results in a 108% increase in generation length while accuracy still drops by 38.6%).
- The dynamic budget maintains accuracy with only a 0.3% drop at an average retention rate of 1.9%.

## Highlights & Insights
- **Empirical Support for Key Insights**: Figure 1 visualizes the dynamic changes in visual attention quality, deriving method design from empirical phenomena.
- **Computationally Elegant Budget Allocation**: Uses visual attention quality as a lightweight signal to predict the retention count per step, avoiding complex learners.
- **Discovery and Resolution of Coupling Bottlenecks**: Reveals that redundant visual information not only adds computation but also induces erroneous reasoning.
- **Generality and Transferability**: Built upon FastV's importance calculation but can theoretically adapt to any other token scoring scheme.

## Limitations & Future Work
- Effective only at inference time and cannot be further optimized through joint learning.
- Temperature parameters required manual tuning.
- Simplification of computational cost analysis (assuming uniform complexity distribution across layers).
- Primarily tested on CoT reasoning tasks; effectiveness on other multimodal tasks requires validation.
- Potential improvements: multi-level pruning, adaptive temperature schedulers, and integration into the multimodal instruction-tuning phase.

## Related Work & Insights
- **vs VisionZip**: Single-pass pruning; this work uses step-wise pruning in intermediate layers to capture changing demands.
- **vs FastV**: Upgrades the one-time decision to multi-step adaptation; accuracy retention improved from 60%-70% to 98%+.
- **vs LOOK-M**: Surpasses LOOK-M at a finer granularity (every generation step) and a more dynamic dimension.
- **Insight**: The perspective of "step-level multimodal demand" can be extended to dynamic text token selection or joint multimodal budget allocation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fundamental concept shift from "fixed pruning" to "step-level dynamic pruning."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 benchmarks + 7 baseline methods + comprehensive ablations + cross-LMM backbone validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain, with key findings presented in high-contrast tables.
- Value: ⭐⭐⭐⭐⭐ Directly reduces inference cost and enhances reasoning reliability; training-free and easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[CVPR 2026\] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](../../CVPR2026/multimodal_vlm/codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)
- [\[ICML 2026\] CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning](look_on_demand_a_cognitive_scheduling_framework_for_visual_evidence_acquisition_.md)
- [\[CVPR 2026\] Unbiased Dynamic Multimodal Fusion](../../CVPR2026/multimodal_vlm/unbiased_dynamic_multimodal_fusion.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)

</div>

<!-- RELATED:END -->
