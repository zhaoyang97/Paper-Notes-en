---
title: >-
  [Paper Note] GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal hallucination] By estimating each token's (visual/textual/output) contribution to the current prediction via first-order Taylor gradient, the GACD framework simultaneously mitigates text-visual bias (amplifying visual token influence) and co-occurrence bias (suppressing visual tokens anchored to previously generated objects). It achieves an 8% improvement in overall AMBER score and 8% gain in POPE F1, without requiring training or auxiliary models.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal hallucination
  - gradient attribution
  - contrastive decoding
  - co-occurrence bias
  - visual-textual bias
date: 2026-05-08
content_hash: 41f6d2a1f7ae00ec
---

# GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection

**Conference**: CVPR 2026
**arXiv**: [2509.03113](https://arxiv.org/abs/2509.03113)
**Code**: N/A
**Area**: Multimodal VLM / Hallucination Mitigation / Decoding Strategy
**Keywords**: Multimodal hallucination, gradient attribution, contrastive decoding, co-occurrence bias, visual-textual bias

## TL;DR
By estimating each token's (visual/textual/output) contribution to the current prediction via first-order Taylor gradient, the GACD framework simultaneously mitigates text-visual bias (amplifying visual token influence) and co-occurrence bias (suppressing visual tokens anchored to previously generated objects). It achieves an 8% improvement in overall AMBER score and 8% gain in POPE F1, without requiring training or auxiliary models.

## Background & Motivation
Hallucinations in MLLMs stem from two primary biases: (1) **Text-visual bias** — models over-rely on text prompts and prior outputs while neglecting visual input, a problem that worsens with sequence length; (2) **Co-occurrence bias** — frequent object co-occurrences in training data (e.g., chair–table) cause models to erroneously predict one object upon seeing the other. Existing methods either depend on auxiliary models (segmentation/detection networks) or apply globally uniform image-level contrastive decoding (VCD/M3ID), lacking fine-grained control at the individual visual token level. The root cause is the absence of quantified bias severity, which prevents sample-adaptive adjustment.

## Core Problem
How can the influence of each token on the output be precisely estimated at inference time, enabling adaptive rebalancing of visual and textual token contributions to simultaneously mitigate both types of bias?

## Method

### Overall Architecture
At each decoding step $m$: (1) first-order Taylor expansion is used to compute the gradient influence of each visual/textual/output token on the current logit, yielding $I_{ms}^v, I_{mn}^p, I_{mi}^y$; (2) visual tokens are partitioned into object-related ($\mathbf{t}_o$) and unrelated ($\mathbf{t}_u$) groups based on their anchoring relationship to already-generated nouns; (3) a negative guidance logit $\mathbf{z}_m^o$ is constructed using only object-related tokens; (4) contrastive decoding is applied as $\hat{\mathbf{z}}_m = (1+\alpha_m)\mathbf{z}_m^* - \alpha_m\mathbf{z}_m^o$, with weight $\alpha_m$ automatically computed from influence estimates to align visual influence with textual influence.

### Key Designs
1. **Gradient influence estimation**: $I_{ms}^v = \|\partial \mathbf{z}_m^* / \partial \mathbf{t}_s^v\|_1$ is computed directly via PyTorch autograd. Each token's influence is measured by the L1 norm of its Jacobian. Group-level influences $\mathcal{I}_m^v, \mathcal{I}_m^p, \mathcal{I}_m^y$ are obtained by aggregation. Findings reveal that in most MLLMs (except InternVL2), textual influence far exceeds visual influence — the visual influence ratio of LLaVA-1.5 is only approximately 30–40%.

2. **Anchor-specific visual token grouping**: For each generated noun $y_i$, the visual token with the greatest influence on it is identified as its "anchor." The union of all anchor tokens forms $\mathbf{t}_o$; the remainder form $\mathbf{t}_u$. Grouping is **triggered only when predicting the next noun** (since co-occurrence bias operates between object pairs); at non-noun steps, all visual tokens are uniformly amplified. Analysis shows that in 31.9% of chair/table co-occurrence hallucinations, both objects share the same most-influential visual token.

3. **Adaptive weight $\alpha_m$**: $\alpha_m = \frac{\mathcal{I}_m^t - \mathcal{I}_m^v}{\mathcal{I}_m^v - \tilde{\mathcal{I}}_m^o + \tilde{\mathcal{I}}_m^t - \mathcal{I}_m^t}$, which automatically ensures that the amplified influence of $\mathbf{t}_u$ matches the maximum influence on the textual side. Unlike VCD, which requires manual threshold tuning, $\alpha_m$ is entirely data-driven via gradient signals. Additional constraints ensure $\alpha_m$ is non-negative and does not excessively suppress the influence of object-related or prompt tokens.

### Loss & Training
The method is entirely training-free and operates at inference time. The additional overhead consists of one gradient computation per step (~101% latency increase, comparable to VCD's ~100%). An early stopping mechanism halts generation when the visual influence ratio $r_m^v$ drops below threshold $\epsilon$ after EOS, preventing hallucinations in later stages of long sequences.

## Key Experimental Results

| Model | Method | AMBER Score↑ | AMBER cog↓ | POPE Acc↑ | POPE F1↑ |
|--------|------|------|----------|------|------|
| LLaVA-1.5 | base | 83.5 | 4.2 | 80.9 | 81.6 |
| | VCD | 83.8 | 3.5 | 80.9 | 81.3 |
| | M3ID | 84.7 | 2.8 | 81.7 | 81.8 |
| | AVISC | 85.5 | 2.7 | 81.2 | 81.6 |
| | **GACD** | **90.2** | **1.8** | **83.5** | **82.1** |
| Qwen2-VL | base | 90.1 | 5.9 | 85.8 | 85.0 |
| | **GACD** | **91.1** | **3.7** | **85.8** | **85.0** |

On LLaVA-QA90: GACD (accuracy 6.20, detail 5.13) vs. VCD (4.15, 3.85) vs. base (3.23, 3.54) — accuracy +92%, detail +45%.

### Ablation Study
- **VA (Visual Amplification) is foundational**: VA alone reduces hallucinations on CHAIR by ~5% while improving recall.
- **CR (Co-occurrence Relief) yields further gains**: VA+CR reduces CHAIR hallucinations by an additional ~2%, especially lowering cog (co-occurrence hallucinations).
- **ES (Early Stopping) acts as a safeguard**: Truncates excessively long outputs when needed; recall drops slightly but hallucinations decrease substantially.
- **L1 norm is optimal**: L1 > L2 > L∞ (L1 is more sensitive to sparse influence patterns).
- **Visual influence ratio determines gain magnitude**: Models with lower baseline visual influence ratios benefit more (mPLUG-Owl2 shows the largest improvement).
- **7B models with GACD can surpass 13B model baselines.**

## Highlights & Insights
- **Principled bias estimation**: First-order Taylor expansion provides a mathematically rigorous measure of token importance without manual hyperparameter tuning.
- Simultaneous mitigation of both biases — text-visual and co-occurrence — is achieved for the first time among existing contrastive decoding methods.
- **Sample-level adaptivity**: $\alpha_m$ is entirely data-driven, providing different adjustment intensities across samples and decoding steps.
- The co-occurrence bias analysis is particularly valuable — the finding that 31.9% of cases share the most influential visual token offers a direction for future object disentanglement research.
- Strong information preservation: recall drops by only 1.1% on average, compared to 3.2% for competing methods.

## Limitations & Future Work
- Requires white-box access (gradient computation) and is not applicable to API-only models.
- ~101% latency increase — significantly slower than methods such as V2Drop/Reallocating Attention, which incur <10% overhead.
- Limited gains on models such as InternVL2 that already exhibit high visual influence ratios, suggesting that finer-grained bias type analysis is needed.
- Limited improvement on relational questions requiring reasoning rather than pure visual perception.
- The early stopping threshold $\epsilon$ must be calibrated per model (ranging from 7% / 25% / 2.5% / 10%).

## Related Work & Insights
- **vs. VCD (contrastive decoding)**: VCD applies global visual contrast via noisy images without distinguishing object-level granularity. GACD performs influence estimation and grouping at the token level, yielding greater precision. AMBER score gain: 6.7 vs. 0.3.
- **vs. AVISC (token-level visual adjustment)**: AVISC also operates at the token level but does not perform object-aware disentanglement. GACD connects visual tokens to specific objects via gradient attribution, handling co-occurrence bias more effectively.
- **vs. M3ID (inter-modal contrastive decoding)**: M3ID applies inter-modal contrast at the image level. GACD operates at the token level and additionally addresses co-occurrence bias.
- **vs. RLAIF-V (training-based method)**: RLAIF-V requires training data and reinforcement learning. GACD is training-free and achieves an AMBER overall score of 90.2 vs. 89.0 on LLaVA-1.5.
- GACD is complementary to the Overthinking line of work: Overthinking addresses inter-layer dynamics (depth-wise), whereas GACD targets token-level influence (modality-wise). A potential combination is to use LogitLens to detect overthinking positions and then apply GACD's gradient attribution at those positions for precise token influence reallocation.
- The "shared most-influential visual token" finding for co-occurrence bias motivates a potential future direction: **co-occurrence disentanglement pretraining** — explicitly separating visual representations of co-occurring object pairs during training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The combination of gradient influence estimation, object-aware visual token grouping, and adaptive contrastive decoding is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 models, 7 benchmarks, generative and discriminative tasks, component ablations, norm ablations, scale ablations, and co-occurrence analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous; the logical chain from bias analysis to method design to validation is coherent throughout.
- Value: ⭐⭐⭐⭐⭐ — Achieves the best accuracy–information preservation trade-off among VLM hallucination mitigation methods.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multimodal_la.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)

<!-- RELATED:END -->
