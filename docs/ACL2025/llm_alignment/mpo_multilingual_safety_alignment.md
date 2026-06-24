---
title: >-
  [Paper Note] MPO: Multilingual Safety Alignment via Reward Gap Optimization
description: >-
  [ACL 2025][LLM Alignment][multilingual safety] MPO discovers that the implicit Reward Gap of LLms between the dominant language (English) and target languages is strongly correlated with safety performance. The authors propose directly minimizing the discrepancy in Reward Gap between the two to transfer the safety alignment capabilities of the dominant language to multiple languages. This method significantly reduces the attack success rate in low-resource languages across th…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "multilingual safety"
  - "reward gap"
  - "preference optimization"
  - "cross-lingual transfer"
  - "alignment"
date: 2026-05-08
content_hash: ddd6f690d720a4b7
---

# MPO: Multilingual Safety Alignment via Reward Gap Optimization

**Conference**: ACL 2025  
**arXiv**: [2505.16869](https://arxiv.org/abs/2505.16869)  
**Code**: [https://github.com/circle-hit/MPO](https://github.com/circle-hit/MPO)  
**Area**: Alignment RLHF  
**Keywords**: multilingual safety, reward gap, preference optimization, cross-lingual transfer, alignment

## TL;DR
MPO discovers that the implicit Reward Gap of LLms between the dominant language (English) and target languages is strongly correlated with safety performance. The authors propose directly minimizing the discrepancy in Reward Gap between the two to transfer the safety alignment capabilities of the dominant language to multiple languages. This method significantly reduces the attack success rate in low-resource languages across three models without compromising general capabilities.

## Background & Motivation
**Background**: LLM safety alignment (RLHF/DPO) is primarily conducted in English. However, users worldwide speak multiple languages, and safety in low-resource languages is severely lacking.

**Limitations of Prior Work**:
   - Multilingual preference data is scarce, and the quality of translated data is poor (especially in low-resource languages, which exhibit unnatural expressions and content errors).
   - DPO/RLHF is highly sensitive to noisy data, and noisy multilingual data may lead to safety misalignment.
   - Direct preference learning on each language yields limited performance, with the model's attack success rate reaching as high as 55-98% on languages like Bengali and Swahili.

**Key Challenge**: The dominant language already possesses decent safety alignment, but these capabilities cannot be automatically transferred to other languages.

**Goal**: How to leverage the existing safety capabilities of the dominant language to enhance the safety of target languages.

**Key Insight**: It is discovered that the Reward Gap (the log-likelihood difference between safe and unsafe responses) is strongly negatively correlated with the attack success rate (ASR), serving as a bridge for cross-lingual safety transfer.

**Core Idea**: Minimize the discrepancy in Reward Gap between the target language and the dominant language, without direct optimization on noisy multilingual preference data.

## Method

### Overall Architecture
MPO (Multilingual reward gaP Optimization) consists of two steps: (1) calculating the Reward Gap $\text{RG}^d$ of the dominant language (English) on the reference model as a fixed target, and (2) optimizing the training model to make the Reward Gap of the target language $\text{RG}^t$ approach $\text{RG}^d$, while constraining the hidden representations of the dominant language to remain unchanged. The inputs are parallel multilingual preference data (original English + translated versions), and the output is a multilingually safety-aligned model.

### Key Designs

1. **Reward Gap Definition (based on SimPO length normalization)**:

    - Function: Quantifying the model's discriminative ability between safe vs unsafe responses.
    - Mechanism: $\text{RG}^t = \frac{1}{|y_w^t|}\log\pi_\theta(y_w^t|x^t) - \frac{1}{|y_l^t|}\log\pi_\theta(y_l^t|x^t)$, using the average log-likelihood from SimPO instead of the log-ratio from DPO.
    - Design Motivation: (a) To be consistent with the likelihood metric during inference, where a larger RG implies a higher probability of generating a safe response; (b) Length normalization eliminates the bias where unsafe responses are typically longer (containing specific harmful content) compared to safe responses which are short (refusal templates).
    - Experimental Validation: On LLaMA-3.1, Gemma-2, and Qwen2.5, the English RG values are 1.58/2.32/1.87, while Swahili is only 0.05/0.41/0.20, which is strongly negatively correlated with ASR.

2. **Reward Gap Discrepancy Minimization ($\mathcal{L}_1$)**:

    - Function: Aligning the RG of the target language to approach that of the dominant language.
    - Mechanism: $\mathcal{L}_1 = \mathbb{E}[\|\beta \cdot \text{RG}^t - \text{RG}^d\|^2]$, where $\text{RG}^d$ is calculated and fixed by the reference model (original aligned model), and $\text{RG}^t$ is calculated on the target language by the training model.
    - Design Motivation: Instead of directly optimizing the pairwise loss of noisy multilingual preference data, the dominant language's RG is used as an "anchor" to indirectly transfer safety capabilities. Gradient analysis shows that $w_\theta = \beta\text{RG}^t - \text{RG}^d$ adaptively adjusts the update magnitude and direction—updates are stronger when the RG discrepancy is large.
    - Difference from DPO: The gradient weight of DPO is based on the model's own likelihood, which is heavily affected by data noise; the weight of MPO is based on the RG discrepancy, guided by high-quality signals from the dominant language.

3. **Dominant Language Representation Protection ($\mathcal{L}_2$)**:

    - Function: Preventing degradation of the dominant language's existing capabilities during multilingual alignment.
    - Mechanism: $\mathcal{L}_2 = \mathbb{E}[\|\mathbf{h}^d - \mathbf{h}^d_\text{ref}\|^2]$, constraining the hidden representation of the last token of the dominant language to remain consistent with that of the reference model.
    - Design Motivation: Directly constraining hidden representations is more effective than KL-divergence regularization (at the logit level)—recent research shows that modifying hidden representations is more direct for behavior control. This avoids the "robbing Peter to pay Paul" dilemma.

### Loss & Training
- Final Loss: $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2$
- Training Data: PKU-SafeRLHF English preference data + translation to 5 target languages using Google Translate.
- No additional multilingual human annotation is needed—although translated data is noisy, MPO does not directly optimize pairwise preferences but instead optimizes the RG discrepancy, making it more robust to noise.

## Key Experimental Results

### Main Results

**MultiJail Safety Evaluation (ASR↓, LLaMA-3.1-8B)**:

| Method | En | Zh | Ko | Ar | Bn | Sw | AVG. |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:----:|
| Original Model | 14.6 | 20.3 | 52.4 | 16.8 | 49.5 | 37.8 | 31.9 |
| SFT | 12.7 | 9.8 | 31.4 | 8.6 | 31.8 | - | - |
| DPO | - | - | - | - | - | - | ~25 |
| SimPO | - | - | - | - | - | - | ~23 |
| **MPO** | - | - | - | - | - | - | **~15** |

**Cross-Model Generalization (AdvBench-X AVG. ASR↓)**:

| Model | Original | DPO | SimPO | MPO |
|------|:----:|:---:|:-----:|:---:|
| LLaMA-3.1-8B | 20.86 | ~15 | ~14 | **~8** |
| Gemma-2-9B | ~8 | ~6 | ~5 | **~3** |
| Qwen2.5-7B | ~30 | ~22 | ~20 | **~12** |

### Ablation Study

| Configuration | MultiJail AVG | AdvBench-X AVG | Description |
|------|:------------:|:-------------:|------|
| Full MPO | Best | Best | $\mathcal{L}_1 + \mathcal{L}_2$ |
| w/o $\mathcal{L}_2$ | +3-5% | +2-4% | Performance degradation in dominant language |
| w/o length norm | +2-3% | +1-2% | Impact of length bias |
| Using DPO-style RG | +1-2% | +1-2% | SimPO RG is superior |

### Key Findings
- Strong negative correlation between RG and ASR: High English RG (1.58-2.32) → Low ASR (0-13%), low-resource language RG is low (0.04-0.20) → High ASR (55-98%).
- MPO is more robust to translation data noise: On (more noisy) data translated by Google Translate, MPO achieves a greater advantage over DPO/SimPO because MPO does not directly optimize preference pairs but instead optimizes the RG discrepancy.
- No damage to general capabilities: On general benchmarks such as MT-Bench and mGSM, MPO does not degrade and even slightly improves general multilingual performance.
- Representation protection $\mathcal{L}_2$ is crucial for maintaining the performance of the dominant language.

## Highlights & Insights
- **Reward Gap as a Quantifiable Proxy Metric for Safety**: This finding is highly valuable on its own—providing a method to assess the safety alignment quality of LLMs without generating output, which can be used to monitor safety status in production environments.
- **Alignment Transfer Anchored on the Dominant Language**: Instead of directly optimizing the preference pairs of noisy translated data, this approach performs indirect optimization targeting the dominant language's RG, cleverly bypassing data quality issues. This concept can be generalized to other cross-lingual capability transfer scenarios.
- **SimPO-style RG is Superior to DPO-style RG**: Length normalization is particularly important for multilingual safety scenarios because refusal responses (safe) are typically much shorter than violative responses (unsafe).

## Limitations & Future Work
- **Reliance on Translation Data**: Although MPO is robust to translation noise, it still requires parallel preference data as input. Future research could explore translation-free schemes (e.g., directly using RG as a self-supervised signal on target languages).
- **Dominant Language Assumption**: The method assumes the existence of a well-aligned dominant language, making it inapplicable to models that exhibit poor safety alignment across all languages.
- **Unaddressed Cultural Differences**: Definitions of "safety" can vary across different languages/cultures; simply transferring English safety standards might not be appropriate.
- **Evaluation Limited to 8-9B Models**: The effectiveness on larger models has not been validated.

## Related Work & Insights
- **vs DPO/SimPO**: DPO/SimPO directly optimize the pairwise loss of preference pairs, which is sensitive to noisy translated data; MPO optimizes the RG discrepancy, making it more robust.
- **vs Multilingual RLHF**: Traditional multilingual RLHF requires training a reward model or gathering preference data for each language; MPO reuses the RG signal of the dominant language.
- **vs Representation Engineering**: The $\mathcal{L}_2$ of MPO shares the core philosophy of representation engineering—protecting capabilities by controlling hidden representations.
- The concept of RG transfer can be explored for other alignment dimensions (e.g., cross-domain transfer of helpfulness).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using the Reward Gap as a bridge for cross-lingual safety transfer is a highly novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated with three models, six languages, multiple safety benchmarks, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous mathematical derivations, and in-depth gradient analysis.
- Value: ⭐⭐⭐⭐⭐ Multilingual safety is a critical issue; the MPO framework is simple, effective, and requires no extra annotation costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](../../ICLR2026/llm_alignment/align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](../../ACL2026/llm_alignment/towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)
- [\[ACL 2025\] DiffPO: Diffusion Alignment with Direct Preference Optimization](diffpo_diffusion_alignment.md)
- [\[ICML 2025\] MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment](../../ICML2025/llm_alignment/mpo_an_efficient_post-processing_framework_for_mixing_diverse_preference_alignme.md)

</div>

<!-- RELATED:END -->
