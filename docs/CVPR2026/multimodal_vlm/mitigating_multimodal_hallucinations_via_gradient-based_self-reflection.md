---
title: >-
  [Paper Note] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
description: >-
  [CVPR2026][Multimodal VLM][Multimodal hallucination] This paper proposes GACD (Gradient-based Influence-Aware Constrained Decoding), which employs first-order Taylor gradient estimation to quantify each token's influence…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Multimodal hallucination"
  - "gradient attribution"
  - "constrained decoding"
  - "co-occurrence bias"
  - "text-visual bias"
  - "inference-time mitigation"
date: 2026-05-08
content_hash: e5ce9810503e53fd
---

# Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection

**Conference**: CVPR2026
**arXiv**: [2509.03113](https://arxiv.org/abs/2509.03113)
**Code**: Not released
**Area**: Multimodal VLM
**Keywords**: Multimodal hallucination, gradient attribution, constrained decoding, co-occurrence bias, text-visual bias, inference-time mitigation

## TL;DR

This paper proposes GACD (Gradient-based Influence-Aware Constrained Decoding), which employs first-order Taylor gradient estimation to quantify each token's influence on the output. GACD simultaneously mitigates multimodal hallucinations caused by text-visual bias and co-occurrence bias at inference time, requiring neither auxiliary models nor fine-tuning.

## Background & Motivation

**Prevalence of multimodal hallucinations**: MLLMs frequently generate content inconsistent with visual inputs, severely undermining model reliability and real-world deployment.

**Text-Visual Bias**: Models over-rely on textual prompts and previously generated tokens, neglecting visual information—a problem that worsens with longer generation sequences.

**Co-occurrence Bias**: Frequently co-occurring object pairs in training data (e.g., chair–table, fork–beer) cause models to erroneously predict one object upon seeing the other.

**Lack of granularity in existing inference methods**: Existing contrastive decoding methods (VCD, M3ID, etc.) apply uniform weights to all visual features, failing to selectively adjust at the token level and offering limited mitigation of co-occurrence bias.

**Limitations of auxiliary-model-dependent methods**: Some approaches require segmentation networks, detectors, or additional MLLMs as auxiliaries, introducing new error sources and task-specific dependencies.

**Absence of sample-level bias measurement**: Most existing work reports only aggregate statistics, unable to quantify bias at the individual sample level or adapt accordingly.

## Method

### Overall Architecture

GACD is a purely inference-time method whose core pipeline consists of: **gradient influence estimation → object-aware visual token grouping → anchor-specific influence-weighted decoding**. At each decoding step, token contributions are analyzed via gradients; visual tokens are partitioned into object-related and object-unrelated groups; contrastive negative guidance logits are then constructed to adaptively amplify visual token influence.

### Key Designs

**1. Gradient-based Token Influence Estimation**

A first-order Taylor expansion is applied to the logit vector $\mathbf{z}_m^* = \pi_{\theta^*}(\mathbf{t}^v, \mathbf{t}^p, \mathbf{y}_{<m})$. The Jacobian of each input token (visual/prompt/history) with respect to the current output logit is computed, and the Manhattan norm is taken as the influence measure:

$$I_{ms}^v = \|\mathbf{g}_{ms}^v\|_1, \quad I_{mn}^p = \|\mathbf{g}_{mn}^p\|_1, \quad I_{mi}^y = \|\mathbf{g}_{mi}^y\|_1$$

Aggregation yields group-level influence scores $\texttt{I}_m^v, \texttt{I}_m^p, \texttt{I}_m^y$, enabling sample-level decomposition of textual and visual contributions to each generated token.

**2. Object-aware Visual Token Grouping**

- spaCy is used to detect nouns in the already-generated sequence $\mathbf{y}_{<m}$.
- For each noun $y_i$, the visual tokens with the highest influence scores are selected to construct a mask $\mathcal{M}_{is}$.
- Masks from all nouns are accumulated to partition visual tokens into **object-related** $\mathbf{t}^o$ and **object-unrelated** $\mathbf{t}^u$ subsets.
- Grouping is executed only at noun prediction steps (since co-occurrence bias manifests between object pairs); at non-noun steps, $\mathbf{t}^o = \varnothing$.

**3. Anchor-specific Influence-weighted Decoding**

Negative guidance logits $\mathbf{z}_m^o = \pi_{\theta^*}(\mathbf{t}^o, \mathbf{t}^p, \mathbf{y}_{<m})$ are constructed, and the adjusted logits are:

$$\hat{\mathbf{z}}_m = (1+\alpha_m)\mathbf{z}_m^* - \alpha_m \mathbf{z}_m^o$$

The weight $\alpha_m$ is computed adaptively from influence scores to align the influence of $\mathbf{t}^u$ with the text-dominant term $\texttt{I}_m^t = \max(\texttt{I}_m^p, \texttt{I}_m^y)$, while a non-negativity upper bound is imposed to prevent over-correction.

**4. Sample-dependent Early Stopping**

Stopping is triggered when the visual influence ratio $r_m^v = \texttt{I}_m^v / (\texttt{I}_m^v + \texttt{I}_m^p + \texttt{I}_m^y) < \epsilon$ and the previous token is EOS, preventing continued generation when visual grounding is insufficient.

### Loss & Training

GACD is an inference-time method and involves no training loss. The core optimization objective is to increase $D_{\mathrm{KL}}(\sigma(\mathbf{z}_m^*) \| \sigma(\mathbf{z}_m^o))$ in probability space—i.e., to amplify the contribution of object-unrelated visual tokens $\mathbf{t}^u$—while preserving non-negative influence from object-related tokens and prompts via upper-bound constraints.

## Key Experimental Results

### Main Results

**AMBER benchmark (generative + discriminative tasks)**:

| Model | Method | cha↓ | cov↑ | hal↓ | cog↓ | Score↑ | F1↑ |
|-------|--------|------|------|------|------|--------|-----|
| LLaVA-v1.5 | Baseline | 7.8 | 51.0 | 36.4 | 4.2 | 83.5 | 74.7 |
| | GACD | **5.6** | 51.0 | **24.3** | **1.8** | **90.2** | **86.0** |
| InstructBLIP | Baseline | 8.8 | 52.2 | 38.2 | 4.4 | 86.5 | 81.7 |
| | GACD | **6.0** | 49.4 | **26.6** | **2.4** | **88.1** | 82.2 |
| mPLUG-Owl2 | Baseline | 10.6 | 52.0 | 39.9 | 4.5 | 84.0 | 78.5 |
| | GACD | **7.5** | **53.6** | **34.7** | **4.0** | **89.6** | **86.6** |
| Qwen2-VL | Baseline | 6.4 | 70.4 | 54.8 | 5.9 | 90.1 | 86.6 |
| | GACD | **4.9** | **71.8** | **44.7** | **3.7** | **91.1** | **87.1** |

**POPE MSCOCO adversarial setting (discriminative task)**:

| Model | Method | Acc↑ | F1↑ |
|-------|--------|------|-----|
| LLaVA-v1.5 | Baseline | 80.9 | 81.6 |
| | GACD | **83.5** | **82.1** |
| mPLUG-Owl2 | Baseline | 72.5 | 77.5 |
| | GACD | **84.2** | **83.7** |
| InternVL2 | Baseline | 85.8 | 85.0 |
| | GACD | 85.8 | 85.0 |

### Ablation Study

| Component Combination | CS↓ (LLaVA-v1.5) | CI↓ | R↑ |
|-----------------------|-------------------|-----|-----|
| Baseline | 48.8 | 13.4 | 78.6 |
| +VA (Visual Augmentation) | 46.4 | 11.6 | 79.0 |
| +VA+CR (Co-occurrence Reduction) | 46.2 | 11.3 | 79.4 |
| +VA+CR+ES (Full Model) | **41.0** | **10.9** | 77.3 |

- VA reduces hallucinations while also improving recall.
- CR further alleviates residual hallucinations from co-occurrence bias.
- ES effectively reduces hallucinations by shortening outputs, with only a marginal recall penalty.

### Key Findings

1. **Improvement magnitude negatively correlates with baseline visual influence**: LLaVA-v1.5 and mPLUG-Owl2 exhibit low initial visual influence ratios (<50%), yielding large gains from GACD; InternVL2 already achieves >50% visual influence, leaving limited room for improvement—this retrospectively validates the method's motivation.
2. **"Shared maximum-influence token" phenomenon in co-occurrence bias**: In chair–table co-occurrence experiments, 31.9% of co-occurrence hallucination cases involve both objects sharing the same highest-influence visual token; GACD effectively breaks this sharing.
3. **Direct gradient vs. integrated gradients**: The direct gradient approach achieves comparable accuracy while being approximately 53× faster (385 ms vs. 20,335 ms).
4. **Superior information preservation over competing methods**: GACD incurs an average recall drop of only 1.1%, compared to 3.2% for other methods.

## Highlights & Insights

- **Clear theoretical grounding**: First-order Taylor gradient attribution provides a mathematical foundation for bias estimation without relying on heuristic priors.
- **Unified dual-bias mitigation**: Text-visual bias and co-occurrence bias are addressed within a single framework; GACD is the first inference-time method to mitigate co-occurrence hallucinations in an object-aware, token-level manner.
- **Adaptive $\alpha_m$**: Weights are dynamically computed from influence ratios with upper-bound constraints, eliminating the need for cross-dataset hyperparameter tuning.
- **Plug-and-play**: Model parameters remain unchanged and no auxiliary models are required, making the method applicable across diverse MLLM architectures.
- **Large gains on LLaVA-QA90**: Accuracy improves by 92% and detail score by 45%, demonstrating strong effectiveness.

## Limitations & Future Work

- **White-box models only**: Gradient access is required, precluding application to API-only models (e.g., GPT-4V).
- **Approximately doubled inference cost**: Inference computation increases by ~101%, comparable to VCD but non-negligible.
- **Limited gains for models with high visual influence**: When a model already leverages visual information well (e.g., InternVL2), improvement margins are small.
- **Weaker gains on relational questions**: The method's effectiveness is limited for question types requiring visual reasoning rather than direct visual grounding.
- **Post-hoc only**: Gradient attribution signals are not fed back into training.

## Related Work & Insights

- **Image-level contrastive decoding**: VCD and M3ID uniformly amplify all visual tokens without distinguishing object-related from object-unrelated ones.
- **Token-level methods**: AVISC lacks object-aware decoupling; HALC depends on an external segmentation model.
- **Training-based methods**: RLAIF-V applies RL alignment but requires additional feedback data and training overhead.
- **Attention-based methods**: These require layer-specific adjustments, introducing model-specific heuristics.
- GACD's core advantages lie in object-awareness, sample-level adaptivity, and freedom from external dependencies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using gradient attribution for decoding-stage bias estimation is a genuinely novel idea; the object-aware grouping combined with adaptive weighting demonstrates originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated across 6 models × 4 datasets, covering both generative and discriminative tasks; ablations are detailed down to individual components, norm choices, and gradient methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, motivation is clear, and equations are well-supported by figures.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play nature of the inference-time approach is highly practical, though the white-box requirement and doubled computational overhead limit the scope of applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](../../NeurIPS2025/multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
