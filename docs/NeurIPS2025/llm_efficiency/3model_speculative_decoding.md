---
title: >-
  [Paper Note] 3-Model Speculative Decoding (PyramidSD)
description: >-
  [NeurIPS 2025][LLM Efficiency][speculative decoding] PyramidSD introduces a three-tier pyramid decoding architecture by inserting an intermediate "qualifier" model between the draft model ($M_D$) and target model ($M_T$)…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "speculative decoding"
  - "multi-model cascade"
  - "fuzzy acceptance criterion"
  - "LLM inference optimization"
date: 2026-05-08
content_hash: 10fb5331a539063e
---

# 3-Model Speculative Decoding (PyramidSD)

**Conference**: NeurIPS 2025
**arXiv**: [2510.12966](https://arxiv.org/abs/2510.12966)  
**Code**: None  
**Area**: LLM Efficiency
**Keywords**: speculative decoding, multi-model cascade, fuzzy acceptance criterion, LLM inference optimization

## TL;DR
PyramidSD introduces a three-tier pyramid decoding architecture by inserting an intermediate "qualifier" model between the draft model ($M_D$) and target model ($M_T$) in standard speculative decoding. The method exploits the natural entropy gradient across model scales within a model family to hierarchically filter tokens, and employs a fuzzy acceptance criterion to relax the matching threshold, achieving up to 1.91× speedup (reaching 124 tok/s on an RTX 4090).

## Background & Motivation
Speculative Decoding (SD) is one of the mainstream approaches to accelerating LLM inference: a small draft model rapidly generates candidate token sequences, which are then verified in a single forward pass by a large target model. Tokens whose predictions align between the draft and target are accepted in batch, significantly improving throughput. However, a fundamental tension exists: **a smaller draft model is faster but exhibits a larger distributional gap with the target, leading to lower acceptance rates and diminished speedup**.

Existing Fuzzy SD methods alleviate this issue by relaxing acceptance thresholds, but single-stage relaxation remains insufficient when the performance gap between draft and target is large. Multi-stage methods such as Cascade Speculative Drafting and Staged SD require additional training or complex coordination mechanisms, limiting their practicality.

A key observation is that modern LLMs are typically released as model families (e.g., Llama 3.2 1B/3B and Llama 3.1 8B), sharing tokenizers and vocabularies, and thus exhibiting natural distributional compatibility across scales. This provides a ready-made condition for inserting an intermediate-sized model between the draft and target.

## Core Problem
**How can models of different sizes within an existing model family be leveraged—without additional training—to bridge the distributional gap between draft and target, thereby improving acceptance rates and throughput in speculative decoding?**

This problem is highly practical: on consumer-grade GPUs (e.g., RTX 4090 with 24 GB VRAM), single-token generation latency is high (~100 ms/tok for 70B models), yet using an excessively small draft model yields limited speedup due to low acceptance rates. Effectively utilizing multiple models of varying sizes within a constrained memory budget is therefore a critical challenge.

## Method

### Overall Architecture
The core idea of PyramidSD is straightforward: a qualifier model ($M_Q$, e.g., 3B) is inserted between the conventional draft model ($M_D$, e.g., 1B) and target model ($M_T$, e.g., 8B), forming a three-tier pyramid. Decoding proceeds in two speculative stages:

1. **Draft → Qualifier stage**: $M_D$ generates $\ell_D$ candidate tokens per step, which are first verified by $M_Q$ using a fuzzy acceptance criterion.
2. **Qualifier → Target stage**: After $M_Q$ accumulates $\ell_Q$ verified tokens, the entire batch is forwarded to $M_T$ for final verification.

This decomposes a large distributional gap into two smaller sequential gaps, achieving a higher acceptance rate at each step.

### Key Designs

1. **Two-stage extension of the fuzzy acceptance criterion**: Standard SD requires the draft's top prediction to strictly match the target's ($\tau=0$), rendering a qualifier entirely redundant in the three-model setting. PyramidSD introduces two relaxation thresholds, $\tau_Q$ and $\tau_T$, to independently control acceptance conditions for the draft→qualifier and qualifier→target stages:
$$\text{Div}(P_{M_Q}(x_t), P_{M_D}(x_t)) \leq \tau_Q \quad \text{and} \quad \text{Div}(P_{M_T}(x_t), P_{M_Q}(x_t)) \leq \tau_T.$$
A key insight is that **setting $\tau_Q \leq \tau_T$ yields the best performance**: the qualifier first strictly filters out clearly erroneous predictions and passes relatively reliable candidates to the target, where a looser threshold further accelerates decoding.

2. **Auxiliary decoding variant (PSDA)**: To achieve a more stable trade-off between speed and quality, PSDA replaces $\tau_Q$ with assisted decoding. When a draft token is rejected by $M_Q$, instead of applying the standard SD correction rule, a token is sampled directly from $M_Q$'s distribution. This guarantees a quality floor at least equal to $M_Q$, at a modest reduction in speedup. PSDA achieves up to 1.44× speedup over standard SD with very low variance, making it suitable for production environments.

3. **Exploitation of the natural entropy gradient**: Empirical analysis reveals that 1B, 3B, and 8B models exhibit a systematic entropy gradient in token prediction: the 1B model's predictive entropy distribution is near-uniform (high uncertainty), the 3B model is intermediate and moderately peaked, and the 8B model is highly concentrated on a small number of high-confidence tokens. The qualifier sits precisely in the middle of this gradient—capable of identifying and rejecting low-quality draft predictions while being substantially faster than the target. This analysis demonstrates that PyramidSD is not effective by coincidence, but rather exploits an intrinsic characteristic of model scaling.

### Loss & Training
PyramidSD **requires no additional training**. It directly utilizes off-the-shelf model families (e.g., instruction-tuned Llama 3.2 1B/3B and Llama 3.1 8B), with the sole requirement that models share a tokenizer and vocabulary. This constitutes one of its most significant practical advantages.

## Key Experimental Results

Experiments are conducted on an RTX 4090 using the CSQA benchmark, with LLaMA-3.2-1B (draft) / LLaMA-3.2-3B (qualifier) / LLaMA-3.1-8B (target).

| Method | Speed (tok/s) | CSQA Score | Speedup vs. SD |
|--------|---------------|------------|----------------|
| SD (standard) | ~65 | 69.58±2.20 | 1.00× |
| FSD (Fuzzy SD) | ~83 | ~70.03 (avg) | ~1.28× |
| PSDA (assisted decoding) | ~94 | ~70.0 | up to 1.44× |
| PSDF (fuzzy variant) | ~124 | ~70.0 | up to 1.91× |

PSDF achieves a peak speed of approximately 124 tok/s under the optimal configuration ($\tau_T=0.5, \tau_Q=0.4, \ell_Q=25$).

### Ablation Study
- **Speculation length ratio**: Setting $\ell_D$ to approximately half of $\ell_Q$ yields the best results, with the optimal ratio between 1:2 and 1:3. Blindly increasing $\ell_D$ leads to cascading errors and a sharp drop in acceptance rate.
- **Threshold combinations**: Configurations with $\tau_Q \leq \tau_T$ are consistently superior. Excessively large $\tau_T$ (>0.5) degrades quality without proportional speedup; moderate thresholds combined with moderate speculation lengths generally outperform extreme configurations.
- **Non-monotonic behavior**: More aggressive speculation (higher thresholds, longer sequences) does not always yield faster decoding. A "rejection cascade" effect is observed—excessive low-quality tokens passed to later stages are rejected consecutively, wasting computation.
- **PSDF vs. PSDA stability**: PSDF achieves a higher peak speed but with large variance (due to multiplicative error amplification across two fuzzy acceptance stages), whereas PSDA consistently exhibits low variance.
- PSDA's CSQA scores remain stable across different threshold settings (68.55–70.73), while PSDF exhibits greater fluctuation (63.50–72.63).

## Highlights & Insights
- **Zero training overhead**: Directly leverages existing model families without fine-tuning or distillation; plug-and-play.
- **Clear theoretical analysis**: The throughput formula decouples the two speculative stages into nested speed calculations, intuitively exposing the interactions among the four hyperparameters $\ell_D, \ell_Q, \tau_Q, \tau_T$.
- **Entropy gradient analysis**: Empirical evidence confirms the systematic entropy differential across model scales, providing a principled justification for the three-model architecture rather than simply adding an extra model opportunistically.
- **Two variants for different scenarios**: PSDA is suited to production environments requiring stability; PSDF is suited to offline batch processing where peak throughput is prioritized.
- **Memory feasibility**: The 1B+3B+8B combination totals approximately 12B parameters, feasible on an RTX 4090 with 24 GB VRAM via quantization.

## Limitations & Future Work
- **Narrow evaluation scope**: Evaluation is limited to a single dataset (CSQA), lacking validation on more diverse tasks such as code generation, long-form text, and multi-turn dialogue.
- **Model family dependency**: Requires draft, qualifier, and target to share a tokenizer and vocabulary; not applicable to cross-family or heterogeneous-tokenizer model combinations.
- **Complex hyperparameter tuning**: The search space over four hyperparameters ($\tau_Q, \tau_T, \ell_D, \ell_Q$) is large, and optimal configurations vary by task, increasing deployment complexity.
- **Memory overhead**: Simultaneously loading three models increases VRAM pressure, potentially infeasible for larger-scale targets (e.g., 70B).
- **PSDF quality instability**: The fuzzy variant exhibits unstable output quality when pursuing peak speed, requiring careful trade-off consideration in deployment.
- **Lack of adaptive control**: The authors identify dynamic threshold and speculation-length controllers as explicit directions for future work.
- **Limited baseline comparisons**: No direct comparison with tree-based verification methods (e.g., SpecInfer) or approaches such as EAGLE.

## Related Work & Insights

| | PyramidSD | Fuzzy SD | Cascade SD | EAGLE/Medusa |
|--|-----------|----------|------------|--------------|
| Requires training | No | No | Yes | Yes |
| Number of models | 3 | 2 | Multiple | 1 + auxiliary head |
| Speedup (vs. SD) | up to 1.91× | ~1.28× | Not directly compared | Orthogonal |
| Quality control | Threshold + hierarchical filtering | Threshold | Cascaded verification | Confidence estimation |
| Applicability | Same-family models | General | Requires custom training | Requires training auxiliary head |

The core distinction from Fuzzy SD is that FSD applies fuzzy relaxation over a single large distributional gap, whereas PyramidSD decomposes the gap into two smaller ones. Unlike Cascade SD, PyramidSD requires no additional training. The approach is orthogonal in principle to methods such as EAGLE and Medusa and can be used in conjunction with them.

**Additional insights:**
- The **hierarchical verification paradigm** is extensible to more tiers (4+ models), though diminishing returns and memory costs must be considered.
- The idea of an **adaptive threshold controller** warrants deeper investigation: acceptance rates and entropy distributions could dynamically guide $\tau$ and $\ell$ adjustments across context segments of varying difficulty.
- The paper's approach of "bridging distributional gaps with intermediate models" shares conceptual parallels with progressive distillation and may benefit from cross-pollination.
- Integration with quantized inference (e.g., GPTQ/AWQ) is promising: the qualifier could be a quantized version of the target rather than an independent model, relaxing the model-family constraint.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-model cascade idea is intuitively natural; the contribution lies in rigorously extending Fuzzy SD to a multi-stage setting and analyzing the conditions for speedup.
- Experimental Thoroughness: ⭐⭐⭐ Only one dataset (CSQA) and one model family (LLaMA) are evaluated; the hyperparameter ablations are detailed but task diversity is insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical derivations are clear, the motivation–method–experiment logic chain is coherent, and the entropy gradient analysis enhances persuasiveness.
- Value: ⭐⭐⭐⭐ The method has practical value as a plug-and-play acceleration scheme, but its impact may be limited by the same-family model requirement and narrow evaluation scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ACL 2026\] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](../../ACL2026/llm_efficiency/tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)

</div>

<!-- RELATED:END -->
