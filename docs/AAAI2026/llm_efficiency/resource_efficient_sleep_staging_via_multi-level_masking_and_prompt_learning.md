---
title: >-
  [Paper Note] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning
description: >-
  [AAAI2026][LLM Efficiency][sleep staging] This paper proposes MASS (Mask-Aware Sleep Staging), a framework that achieves reliable sleep staging using only **10% of the original EEG signal** through a multi-level masking strategy and hierarchical prompt learning mechanism, providing a practical solution for resource-constrained wearable sleep monitoring systems.
tags:
  - "AAAI2026"
  - "LLM Efficiency"
  - "sleep staging"
  - "EEG"
  - "masking"
  - "prompt learning"
  - "wearable monitoring"
  - "resource efficiency"
date: 2026-05-08
content_hash: 2b6708cb6e3095c7
---

# Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning

**Conference**: AAAI2026
**arXiv**: [2511.06785](https://arxiv.org/abs/2511.06785)  
**Code**: [AnsonAiTRAY/MASS](https://github.com/AnsonAiTRAY/MASS)  
**Area**: LLM Efficiency
**Keywords**: sleep staging, EEG, masking, prompt learning, wearable monitoring, resource efficiency

## TL;DR
This paper proposes MASS (Mask-Aware Sleep Staging), a framework that achieves reliable sleep staging using only **10% of the original EEG signal** through a multi-level masking strategy and hierarchical prompt learning mechanism, providing a practical solution for resource-constrained wearable sleep monitoring systems.

## Background & Motivation
Automatic sleep staging is critical for sleep quality assessment and disorder diagnosis. The AASM standard classifies 30-second EEG epochs into five stages: W/REM/N1/N2/N3.

- **Resource constraints**: Wearable and home sleep monitoring devices have limited battery capacity, making the power consumption of continuous long-term EEG acquisition a core bottleneck.
- **Limitations of prior work**: State-of-the-art methods such as DeepSleepNet, AttnSleep, and NeuroNet rely on complete, continuous EEG signals and exhibit severe performance degradation under data missingness.
- **Hardware foundation**: Signal amplifiers such as the ADS1299 support microsecond-level switching between active and standby modes, providing hardware support for on-demand acquisition.
- **Core positioning**: This is the first neural network-based sleep staging method designed from a data-efficiency perspective.

## Method

### Multi-Level Masking Strategy
Random masking is applied at two levels: the epoch level and the patch level.
1. **Epoch-level masking**: Entire 30-second epochs are randomly masked with mask ratio $r_e$.
2. **Patch-level masking**: Each unmasked epoch is divided into 30 one-second patches, which are further masked with mask ratio $r_a$.
3. The remaining visible patches are transformed via PSD frequency-domain conversion and linear projection to obtain feature representations $E_{vis} \in \mathbb{R}^{e(1-r_e) \times 30(1-r_a) \times d_a}$.

Masking is applied during both training and inference to simulate real-world partial-observation scenarios.

### Global Prompt Learning
All visible patches are flattened into a global sequence, a learnable CLS token is prepended, and fixed sinusoidal positional encodings are used to preserve absolute positions from the original sequence. A shallow Transformer encoder ($L_p$ layers) generates a global prompt $z_{prompt}$ that serves as a semantic anchor to guide subsequent modeling.

### Hierarchical Feature Modeling
- **Patch-level**: Within each visible epoch, the CLS token, global prompt, and unmasked patches are concatenated and fed into a Transformer encoder to extract fine-grained intra-epoch patterns.
- **Epoch-level**: The patch-level CLS outputs are concatenated with the global prompt; masked epochs are zero-filled; a Bi-GRU models inter-epoch temporal transitions.
- **Auxiliary task**: A binary stage-transition prediction task is incorporated to enhance inter-epoch dynamics capture.

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{Cos} + \lambda_2 \mathcal{L}_{Trans}$$

## Key Experimental Results

### Main Results
Macro-F1 (%) comparison under varying signal completeness:

| Method | DREAMS-SUB 100% | 10% | Sleep-EDF-20 100% | 10% | Sleep-EDF-78 100% | 10% | SHHS 100% | 10% |
|---|---|---|---|---|---|---|---|---|
| DeepSleepNet | 67.02 | 11.08 | 76.65 | 10.27 | 71.81 | 13.97 | 73.91 | 12.88 |
| TinySleepNet | 78.49 | 8.81 | 78.41 | 9.85 | 74.11 | 26.87 | 75.22 | 26.49 |
| NeuroNet | 79.51 | 6.69 | 78.65 | 8.94 | 75.73 | 20.18 | 76.87 | 10.65 |
| **MASS** | **81.14** | **75.58** | **80.11** | **76.62** | **75.02** | **71.61** | **76.87** | **70.25** |

At 10% signal, MASS incurs only a 3–7% drop in macro-F1, while competing methods collapse to single-digit or low twenty-percent values.

Resource efficiency analysis:
- At 10% signal, power consumption of the ADS1299-4 drops from 22 mW to 6.79 mW (a **69%** reduction), and the ADS131A04 drops from 15.8 mW to 3.92 mW (a **75%** reduction).
- Parameter efficiency $\eta_p = 0.73$ and inference time efficiency $\eta_t = 16.08$ are both best-in-class.

### Ablation Study
Results at 40% signal on DREAMS-SUB:

| Variant | ACC | macro-F1 |
|---|---|---|
| MASS-Base | 43.2 | 15.9 |
| MASS-Prompt | 45.6 | 18.5 |
| MASS-Mask | 85.8 | 79.9 |
| **MASS (Full)** | **86.4** | **80.0** |

Multi-level masking is the primary driver of performance.

## Highlights & Insights
- **Extreme data efficiency**: Only ~5% macro-F1 is lost at 10% signal availability — a regime entirely beyond the reach of existing methods.
- **Training–inference consistency**: Masking is applied in both phases, making the model inherently robust to partial observations.
- **Real-world power validation**: Integration with actual amplifier power data demonstrates 60–75% power savings.
- **Global prompt design**: The CLS token with preserved absolute positional encodings aggregates global context, compensating for information loss caused by masking.

## Limitations & Future Work
- Validation is limited to single-channel EEG; the masking strategy requires redesign for multi-channel settings.
- The masking pattern is fixed as random; adaptive or learned masking has not been explored.
- Whether the 30 one-second patch granularity is optimal is not thoroughly discussed.
- Experiments are conducted only on public datasets, lacking end-to-end deployment validation on real wearable devices.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First sleep staging model designed from a data acquisition efficiency perspective; the masking + prompt combination is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 datasets × 4 signal ratios × 6 baselines, with ablation and resource efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; methodological derivation is rigorous.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to real-world deployment of wearable sleep monitoring systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](../../ICLR2026/llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ICLR 2026\] MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](../../ICLR2026/llm_efficiency/mhla_restoring_expressivity_of_linear_attention_via_token-level_multi-head.md)
- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/llm_efficiency/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](../../ACL2026/llm_efficiency/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ICLR 2026\] ICaRus: Identical Cache Reuse for Efficient Multi-Model Inference](../../ICLR2026/llm_efficiency/icarus_identical_cache_reuse_for_efficient_multi-model_inference.md)

</div>

<!-- RELATED:END -->
