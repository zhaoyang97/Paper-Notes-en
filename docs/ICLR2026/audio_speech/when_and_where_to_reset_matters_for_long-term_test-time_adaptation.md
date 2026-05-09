---
title: >-
  [Paper Note] When and Where to Reset Matters for Long-Term Test-Time Adaptation
description: >-
  [ICLR 2026][Audio & Speech][test-time adaptation] ASR proposes an adaptive selective reset scheme that uses prediction concentration $\mathcal{C}_t$ to dynamically determine *when* to reset (avoiding the suboptimality of fixed-period resets), and employs a progressive layer selection strategy from output to input layers to determine *where* to reset (preserving valuable adaptation knowledge). Combined with importance-aware regularization for recovering critical knowledge in reset layers and on-the-fly adaptation adjustment, ASR achieves a 44.12% improvement over the prior SOTA on CCC-Hard.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - test-time adaptation
  - model collapse
  - adaptive reset
  - selective reset
  - Fisher information
  - long-term domain shift
date: 2026-05-08
content_hash: 8023b95c5bf8d412
---

# When and Where to Reset Matters for Long-Term Test-Time Adaptation

**Conference**: ICLR 2026
**arXiv**: [2603.03796](https://arxiv.org/abs/2603.03796)
**Code**: [https://github.com/YonseiML/asr](https://github.com/YonseiML/asr)
**Area**: Audio & Speech
**Keywords**: test-time adaptation, model collapse, adaptive reset, selective reset, Fisher information, long-term domain shift

## TL;DR
ASR proposes an adaptive selective reset scheme that uses prediction concentration $\mathcal{C}_t$ to dynamically determine *when* to reset (avoiding the suboptimality of fixed-period resets), and employs a progressive layer selection strategy from output to input layers to determine *where* to reset (preserving valuable adaptation knowledge). Combined with importance-aware regularization for recovering critical knowledge in reset layers and on-the-fly adaptation adjustment, ASR achieves a 44.12% improvement over the prior SOTA on CCC-Hard.

## Background & Motivation

**Background**: Continual test-time adaptation (TTA) updates models on non-stationary domain streams, but long-term adaptation leads to error accumulation and model collapse — where the model predicts only a small number of classes for all inputs.

**Limitations of Prior Work**: (1) Methods such as RDumb apply full resets at fixed intervals, which are unrelated to actual collapse risk — resets occur either too early (wasting adaptation knowledge) or too late (allowing deep error accumulation); (2) full resets catastrophically discard all temporally accumulated knowledge; (3) each reset is followed by significant performance drops and recovery delays.

**Key Challenge**: Resetting too frequently leads to insufficient adaptation; resetting too infrequently allows irreversible collapse. Full resets cause knowledge loss; no resets allow error accumulation.

**Goal**: (1) *When*: how to dynamically detect when collapse risk is high? (2) *Where*: how to select which layers to reset in order to minimize knowledge loss? (3) How to recover critical knowledge from reset layers?

**Key Insight**: Prediction concentration is used as a proxy for collapse risk, and the hierarchical structure of deep networks — where label noise corruption begins near the output layers — informs the reset scope.

**Core Idea**: Deviation of prediction concentration from a long-term baseline triggers resets; layers are progressively reset from the output end according to collapse severity; Fisher information-weighted regularization recovers critical knowledge from reset layers.

## Method

### Overall Architecture
ASR consists of three components: (1) adaptive selective reset (based on $\mathcal{C}_t$ vs. $\bar{\mathcal{C}}_{t-1}$); (2) importance-aware knowledge recovery (Fisher information regularization); (3) on-the-fly adaptation adjustment (based on prediction inconsistency $\phi_t$).

### Key Designs

1. **Adaptive Reset — When**:

    - Prediction concentration: $\mathcal{C}_t = \sum_{c=1}^C \hat{p}_{t_c} \log(\hat{p}_{t_c})$, where $\hat{p}_t = \sigma(\frac{1}{|\mathcal{B}_t|}\sum_i f_{\theta_{t-1}}(x_t^i))$
    - Large $\mathcal{C}_t$ → low prediction diversity → high collapse risk
    - Cumulative concentration (EMA): $\bar{\mathcal{C}}_t = \mu_\mathcal{C} \cdot \bar{\mathcal{C}}_{t-1} + (1-\mu_\mathcal{C}) \cdot \mathcal{C}_t$
    - Trigger condition: reset is performed immediately when $\mathcal{C}_t > \bar{\mathcal{C}}_{t-1}$
    - $\bar{\mathcal{C}}_0$ is initialized as $-\log(\alpha_0 \cdot C)$, with $\alpha_0$ chosen to ensure the initial value is large enough to prevent premature resets
    - Empirical validation: Pearson correlation between $\mathcal{C}_t$ and accuracy reaches **0.88**

2. **Selective Reset — Where**:

    - Motivation: label noise corruption begins at the output end of the network (Bai et al., 2021; Yang et al., 2024), while layers near the input are more robust
    - Reset ratio: $r_t = r_0 + \lambda_r \cdot (\mathcal{C}_t - \bar{\mathcal{C}}_{t-1})$
    - The top $r_t$ fraction of layers (starting from the output end) are reset; the remaining layers are preserved
    - $r_t$ is capped at 1.0; $r_0$ denotes the minimum reset ratio
    - Design Motivation: more severe collapse → deeper corruption propagation → more layers need to be reset

3. **Importance-Aware Knowledge Recovery**:

    - Loss: $\mathcal{L} = \mathcal{L}_u + \lambda_\mathcal{F}\sum_i \bar{\mathcal{F}}^i(\theta_{t-1}^i - \bar{\theta}^i)^2$
    - $\bar{\mathcal{F}}^i$: accumulated Fisher information matrix; $\bar{\theta}^i$: accumulated parameters
    - Parameters important to prior tasks (high Fisher values) are guided to align with their accumulated states
    - **Hybrid accumulation scheme**: CMA equally weights parameter and Fisher matrix accumulation between resets; EMA aggregates CMA values at reset trigger points
    - Addressed dilemma: parameters near a reset point are more adapted to the current domain but are also more susceptible to corruption; the recency bias of EMA is therefore unsuitable for direct application

4. **On-the-Fly Adaptation Adjustment**:

    - Prediction inconsistency: $\phi_t = \frac{1}{|\mathcal{B}_t|}\sum_i \mathbb{I}(\arg\max(\breve{y}_t^i) \neq \arg\max(\hat{y}_t^i))$
    - Large $\phi_t$ (high disagreement between source model and current model) → large domain gap
    - Adaptive hyperparameter scheduling: $\lambda_\mathcal{F} = \lambda_0 \cdot \phi_t^2$ (larger domain gap → stronger regularization); $\mu_\mathcal{C} = \mu_0 \cdot \phi_t + 1 - \mu_0$ (larger domain gap → slower concentration update)

## Key Experimental Results

### CCC Benchmark (Main Results, ResNet-50)

| Method (ETA-based) | Easy | Medium | Hard | Mean |
|-------------------|------|--------|------|------|
| ETA | 43.24 | 19.03 | 0.32 | 20.86 |
| + RDumb | 49.47 | 39.42 | 9.77 | 32.89 |
| + COME | - | - | - | - |
| + ReservoirTTA | - | - | - | - |
| **+ ASR (Ours)** | **Highest** | **Highest** | **Highest** | **Highest** |

ASR achieves a **44.12%** improvement over the prior SOTA on CCC-Hard.

### Other Benchmarks
- Concatenated ImageNet-C (CIN-C): best performance among all methods
- ImageNet-C (20 rounds): stable adaptation without collapse
- ImageNet-D109 (20 rounds): similarly achieves the best results

### Key Findings
- ASR functions as a plug-in add-on compatible with multiple baselines including ETA, EATA, and ROID
- Gains are especially pronounced in challenging settings (CCC-Hard) — precisely the scenarios where existing methods collapse most severely
- $\mathcal{C}_t$ is more stable and reliable than alternative collapse detection signals (e.g., high-confidence prediction ratio, distribution shift detection)
- Selective reset vs. full reset: substantially reduces post-reset performance drops and recovery delays

### Ablation Study
- Removing adaptive reset (replaced with fixed-period reset) → performance degradation
- Removing selective reset (replaced with full reset) → increased performance drops and recovery delays
- Removing Fisher regularization → failure to recover critical knowledge from reset layers
- Removing on-the-fly adjustment → insufficient adaptability under challenging domain shifts

## Highlights & Insights
- **Elegance of signal design**: $\mathcal{C}_t$ is derived from the entropy of the softmax of batch-level averaged logits — simple yet effective (correlation of 0.88), requiring no additional models or computation overhead
- **Theoretical grounding for layer-wise reset**: the method leverages the known phenomenon that corruption begins at the output end of the network, translating a general observation into a practical strategy
- **CMA+EMA hybrid accumulation**: elegantly addresses the bootstrapping dilemma that "parameters near a reset are better adapted to the current domain but more likely to be corrupted"
- **Plug-and-play design**: ASR can be added on top of any existing TTA method without modifying the base adaptation algorithm

## Limitations & Future Work
- Hyperparameters ($r_0, \lambda_r, \alpha_0, \lambda_0, \mu_0$) require calibration on holdout data, though only a small amount is used (5% of a single split)
- The current formulation assumes that samples within a batch originate from the same domain; mixed-domain batch scenarios remain unexplored
- The accuracy of Fisher information estimation may degrade over time in continual online learning
- Validation on ViT-B-16 is relatively preliminary; evaluation on additional architectures and scales is warranted
- Integration with prompt-based TTA methods is worth exploring

## Related Work & Insights
- **vs. RDumb**: fixed-period full reset is a naive yet effective baseline; ASR extends it with adaptivity and selectivity
- **vs. CoTTA**: CoTTA uses augmentation-averaged pseudo-labels and stochastic parameter restoration; ASR adopts a more principled Fisher-based approach
- **vs. ROID/CMF**: weight interpolation methods; ASR's reset-and-recovery paradigm is complementary
- **vs. PeTTA**: regularization based on parameter divergence; ASR's prediction concentration is a more direct indicator of collapse

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of adaptive and selective reset, together with the CMA+EMA hybrid accumulation, represents a meaningful contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 benchmarks, multiple baseline combinations, detailed ablations, multi-architecture validation
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation (Fig. 1 is highly intuitive), well-illustrated methodology (Fig. 2), rigorous statistical reporting
- Value: ⭐⭐⭐⭐⭐ — A 44.12% improvement on CCC-Hard is a substantial breakthrough; the plug-and-play design ensures broad applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](../../NeurIPS2025/audio_speech/e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](../../NeurIPS2025/audio_speech/instance-specific_test-time_training_for_speech_editing_in_the_wild.md)

</div>

<!-- RELATED:END -->
