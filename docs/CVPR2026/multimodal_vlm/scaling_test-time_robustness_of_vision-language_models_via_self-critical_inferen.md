---
title: >-
  [Paper Note] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework
description: >-
  [CVPR 2026][Multimodal VLM][LVLM robustness] This paper proposes the Self-Critical Inference (SCI) framework, which simultaneously addresses language bias and language sensitivity in LVLMs via multi-round textual and vis…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "LVLM robustness"
  - "counterfactual reasoning"
  - "language bias"
  - "language sensitivity"
  - "test-time scaling"
date: 2026-05-08
content_hash: 68779f41514d103c
---

# Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework

**Conference**: CVPR 2026
**arXiv**: [2603.07659](https://arxiv.org/abs/2603.07659)
**Code**: [https://github.com/KaihuaTang/Self-Critical-Inference-Framework](https://github.com/KaihuaTang/Self-Critical-Inference-Framework)
**Area**: Multimodal VLM
**Keywords**: LVLM robustness, counterfactual reasoning, language bias, language sensitivity, test-time scaling

## TL;DR
This paper proposes the Self-Critical Inference (SCI) framework, which simultaneously addresses language bias and language sensitivity in LVLMs via multi-round textual and visual counterfactual logit aggregation. A dynamic robustness benchmark, DRBench, is introduced to evaluate robustness in a model-specific manner. Increasing the number of counterfactual inference rounds yields consistent robustness gains, opening a new direction for test-time scaling.

## Background & Motivation
**Background**: LVLMs achieve strong vision-language capabilities by combining visual encoders with pretrained LLMs through joint fine-tuning.

**Limitations of Prior Work**:
   - **Language Bias**: Models rely on language priors rather than visual inputs to answer questions, leading to object hallucinations (e.g., generating non-existent content).
   - **Language Sensitivity**: Semantically equivalent but lexically distinct prompt variations elicit different responses, undermining consistency and reliability.
   - Methods such as VCD address only visual counterfactuals (bias), entirely neglecting textual counterfactuals (sensitivity).

**Key Challenge**: VCD is fundamentally a reweighting of the original logits by TIE logits along a single dimension (visual); however, the robustness challenges of LVLMs are two-dimensional.

**Goal**: Simultaneously mitigate language bias and language sensitivity, and demonstrate that increasing inference rounds improves robustness.

**Key Insight**: Unifying VCD within the causal analysis framework of CF-VQA to reveal the physical interpretation of $\alpha$ (a temperature parameter for TIE), and naturally extending this formulation to the textual counterfactual dimension.

**Core Idea**: Since VCD reduces to TIE reweighting, both Textual Counterfactual (TC) and Visual Counterfactual (VC) can be applied simultaneously, enabling test-time robustness scaling via multi-round logit aggregation.

## Method

### Overall Architecture
Given the original input $(v^0, q^0)$, $N$ textual variants $\{q^i\}$ and $M$ visual variants $\{v^j\}$ are generated. TC and VC logits are computed separately and combined via weighted multiplication to yield the final prediction: $p_{SCI}(y) \propto \exp(TC/\tau_1) \cdot \exp(VC/\tau_2)$.

### Key Designs

1. **Unified Understanding of VCD and CF-VQA**:

    - VCD logit: $Z_{vcd} = (1+\alpha)Z(v,q) - \alpha Z(v^*,q)$
    - Expanding in the exp domain: $p(y) \propto \exp(Z(v,q)) \cdot \exp(TIE/\tau)$
    - This reveals that VCD is essentially a vocabulary-level reweighting term using TIE logits, where $\tau = 1/\alpha$ serves as the temperature parameter.
    - This analysis bridges VCD and CF-VQA, providing a theoretical foundation for extension to the textual dimension.

2. **Textual Counterfactual (TC)**:

    - Semantically equivalent but lexically diverse prompt variants $\{q^i\}$ are generated.
    - For each token position $k$, the element-wise maximum over all variant logits is taken: $TC_k = \max_i(Z_k(v^0, q^i))$
    - Effect: eliminates logit biases induced by specific phrasings while retaining predictions that are consistent across wordings.
    - Design Motivation: If a model produces different answers to semantically identical but lexically different prompts, taking the maximum selects the most stable prediction.

3. **Visual Counterfactual (VC)**:

    - VCD is extended to multiple counterfactual images: $VC = Z(v^0, q^0) - \mathbb{E}[Z(v^j, q^0)]$
    - The average logit over multiple content-removed images replaces a single noise image.
    - This yields a more stable estimate of language bias.

4. **SCI3 / SCI5 / SCI7 Configurations**:

    - SCI3: $M=N=1$ (3 inference passes); SCI5: $M=N=2$ (5 passes); SCI7: $M=N=3$ (7 passes).
    - Increasing inference rounds consistently improves robustness at linearly growing computational cost.

### Loss & Training
SCI is a purely inference-time method requiring no training. The temperature parameters $\tau_1$ and $\tau_2$ for TC and VC are tuned on a validation set.

## Key Experimental Results

### Main Results (DRBench BS Subset Overall)

| Method | LLaVA-NeXT BS↑ | Qwen2-VL BS↑ |
|--------|:--------------:|:------------:|
| Baseline | 18.75 | 14.52 |
| TIE | 27.31 | - |
| VCD | 27.89 | - |
| M3ID | 29.05 | - |
| **SCI3** | **32.72** | - |
| **SCI5** | **34.19** | - |
| **SCI7** | **34.92** | - |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| VC only (≈VCD) | Bias improves; sensitivity unchanged | Addresses only half the problem |
| TC only | Sensitivity improves; bias unchanged | Addresses the other half |
| VC + TC (SCI) | Both problems simultaneously improved | Advantage of the unified framework |
| SCI3→SCI5→SCI7 | Consistent 1–2% gains | Test-time scaling is effective |

### Key Findings
- **Minimal overlap between bias and sensitivity samples**: Only 7.34% of the 24.68% hard samples identified for LLaVA-NeXT are shared with Qwen2-VL, confirming that robustness is model-specific.
- Qwen2-VL is generally more robust but more susceptible to bias; LLaVA-NeXT exhibits more pronounced sensitivity issues.
- Monotonic improvement from SCI3 to SCI7 suggests that the potential of test-time robustness scaling remains largely unexplored.
- TC and VC address distinct types of robustness failures and are mutually indispensable.

## Highlights & Insights
- **Unification of VCD and CF-VQA**: Revealing that VCD is equivalent to temperature-scaled TIE reweighting is a contribution of independent value.
- **Test-time robustness scaling**: Unlike conventional test-time scaling (extending intermediate token length), robustness is improved by increasing counterfactual inference rounds—a direction orthogonal to CoT scaling.
- **DRBench design philosophy**: A dynamic, model-specific benchmark that can be automatically constructed from any dataset, addressing the risk of fixed benchmarks being included in subsequent model training data.
- The method is model-agnostic and can be directly integrated into any LVLM inference pipeline.

## Limitations & Future Work
- Inference cost scales linearly: SCI7 requires 7 forward passes.
- The strategies for generating textual and visual variants are relatively straightforward; more sophisticated counterfactual generation may yield further improvements.
- Temperature parameters $\tau_1$ and $\tau_2$ require manual tuning.
- DRBench relies on specific counterfactual generation methods to construct bias and sensitivity subsets.

## Related Work & Insights
- **vs. VCD**: VCD is a special case of SCI ($N=0, M=1$); SCI extends the counterfactual dimensions and introduces test-time scaling.
- **vs. CF-VQA / TDE**: Causal analysis for debiasing in conventional VQA has been established in prior work; this paper demonstrates that the same principles apply to LVLMs and extend naturally to language sensitivity.

## Rating
- Novelty: ⭐⭐⭐⭐ — Unified analysis is insightful; test-time scaling direction is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six datasets, two models, and a well-designed DRBench.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical analysis is elegant; the derivation from VCD to SCI is natural.
- Value: ⭐⭐⭐⭐ — A practical inference-time robustness enhancement method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
