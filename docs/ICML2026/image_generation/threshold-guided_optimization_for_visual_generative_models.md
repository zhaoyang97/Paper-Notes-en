---
title: >-
  [Paper Note] Threshold-Guided Optimization for Visual Generative Models
description: >-
  [ICML 2026][Image Generation][Threshold-guided] The authors remove the paired preference assumption of DPO, proving that the optimal strategy for KL regularization essentially compares each sample's reward to an intracta…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Threshold-guided"
  - "Unpaired Preference Optimization"
  - "Scalar Feedback"
  - "Diffusion Model Alignment"
  - "MaskGIT"
date: 2026-05-08
content_hash: 0cd59112941676d4
---

# Threshold-Guided Optimization for Visual Generative Models

**Conference**: ICML 2026  
**arXiv**: [2605.04653](https://arxiv.org/abs/2605.04653)  
**Code**: None  
**Area**: Image Generation / Preference Alignment  
**Keywords**: Threshold-guided, Unpaired Preference Optimization, Scalar Feedback, Diffusion Model Alignment, MaskGIT

## TL;DR
The authors remove the paired preference assumption of DPO, proving that the optimal strategy for KL regularization essentially compares each sample's reward to an intractable instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$. They propose replacing it with a global threshold $\tau$ estimated from a score percentile, and introduce a confidence weight proportional to $|s-\tau|$. This enables stable alignment of diffusion models and MaskGIT using only scalar scores (without paired preferences), consistently outperforming Diffusion-DPO / KTO / DSPO across five reward models and three test sets.

## Background & Motivation
**Background**: The mainstream approach for aligning visual generative models is to adapt LLM RLHF / DPO: first collect paired preferences $(y_w, y_l)$, then use the Bradley-Terry model to encourage $\pi_\theta$ to assign higher probability to $y_w$. Methods like Diffusion-DPO, AlignProp, and DSPO follow this paradigm.

**Limitations of Prior Work**: In practice, feedback is often not paired, but rather 1–5 star ratings, continuous scores from reward models, or scalar scores for individual images. Forcing these into pairs (by comparing within a batch) loses absolute value information, and manual pairing amplifies noise when scores cluster. Diffusion-KTO circumvents pairing by splitting scores into desirable/undesirable sets, but requires hard thresholding.

**Key Challenge**: The DPO family avoids the intractable partition function $Z(x)$ in the KL-optimal solution because $\log Z(x)$ cancels out in paired differences. With only scalar feedback per sample, this cancellation no longer holds, and one must directly address the instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$.

**Goal**: (i) Derive a **computable** surrogate decision rule for KL-regularized alignment with scalar feedback; (ii) Ensure the rule applies to both diffusion models (MSE likelihood proxy) and MaskGIT (token-level cross-entropy likelihood); (iii) Avoid extra paired sampling costs, enabling pure offline, single-pass training.

**Key Insight**: Starting from the KL-optimal solution, the authors find that the optimal update direction is a binary decision—only increase a sample's probability if its reward exceeds $\tau^*(x)$. Since $\tau^*(x)$ is intractable, can a **global threshold** (e.g., median) of the reward distribution approximate it? Samples farther from the threshold naturally provide stronger supervision, motivating "confidence weighting".

**Core Idea**: Use a percentile threshold $\tau$ from the empirical score distribution as a global proxy for the intractable instance-level baseline $\tau^*(x)$, turning alignment into a confidence-weighted binary classification task, thus enabling direct policy fitting with unpaired scalar feedback.

## Method

### Overall Architecture
TGO (Threshold-Guided Optimization) training consists of four steps: (1) Given a reference policy $\pi_{\text{ref}}$ (i.e., initial $\pi_\theta$), use a reward model $r(\cdot)$ to score the offline dataset $\{(x_i, y_i)\}$, obtaining $s_i$; (2) Compute a percentile $\tau = \text{Percentile}(\{s_i\}, p)$ (default $p=0.5$, the median); (3) For each sample, generate a pseudo-label $l_i = \mathbb{1}[s_i \ge \tau]$ and a confidence weight $w_i = 1 + c|s_i - \tau|$; (4) Use a DPO-like sigmoid binary cross-entropy loss, but with a one-sided implicit policy score $\hat r = \beta(\log \pi_\theta - \log \pi_{\text{ref}})$ instead of a paired difference. The entire process is offline, requiring no online rollout or reward model finetuning.

### Key Designs

1. **Threshold Decision Rule from KL-Optimal Solution**:

    - **Function**: Simplifies the "should a sample's probability be increased" question to a comparison with a global scalar threshold.
    - **Mechanism**: The closed-form KL-regularized objective $\max \mathbb E[\mathcal R(x,y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ yields the optimal solution $\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} > 0 \iff \mathcal R(x,y) > \tau^*(x)$, where $\tau^*(x) = \beta \log Z(x)$. The authors make two assumptions: the scalar score $s$ is a monotonic transformation of the reward; and the global percentile $\tau$ from the empirical distribution approximates $\tau^*(x)$. The decision rule becomes $\pi_\theta(y|x) \gtrsim \pi_{\text{ref}}(y|x)$ when $s \ge \tau$.
    - **Design Motivation**: DPO elegantly cancels $\log Z(x)$ via paired differences; without pairing, $Z(x)$ must be addressed directly. The global threshold is a simple, statistically justified proxy—appendix theorems prove the estimator is consistent as $n \to \infty$ with $O(1/n)$ error, and calibrated to the original KL-optimal rule.

2. **Confidence-Weighted Binary Classification Loss**:

    - **Function**: Ensures samples far from the threshold (more confidently "good/bad") contribute larger gradients, while those near the threshold (ambiguous) have smaller weights.
    - **Mechanism**: Define the implicit policy score $\hat s_{\theta,\text{ref}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, with loss $\mathcal L_{\text{TG}} = -\mathbb E[w(s,\tau)(l\log\sigma(\hat s) + (1-l)\log(1-\sigma(\hat s)))]$, where $w(s,\tau) = 1 + c|s-\tau|$, hyperparameter $c \ge 0$. This is equivalent to a weighted BCE, pushing the policy toward high-reward samples and away from low-reward ones.
    - **Design Motivation**: Samples near the median are inherently ambiguous; treating them equally with extreme samples introduces noise. Linear confidence weighting preserves data utilization while naturally boosting signal-to-noise ratio, and is not sensitive to hyperparameters ($c=5$ is robust across tasks in experiments).

3. **Likelihood Proxies for Two Types of Visual Generative Models**:

    - **Function**: Enables computation of $\log \pi_\theta(y|x)$ for both diffusion models and MaskGIT, allowing TGO to support both continuous and discrete generation paradigms.
    - **Mechanism**: For diffusion models, use a Gaussian observation assumption $\log \pi_\theta(y|x) \approx -\frac{1}{T}\text{MSE}(y, \hat y_\theta(x))$, with temperature $T$ controlling scale (default $T=0.001$); for MaskGIT, use the log-likelihood at masked positions after VQ-GAN tokenization, $\log \pi_\theta(y|x) = \frac{1}{|M|}\sum_{i\in M}\log p_\theta(t_i | y_{\setminus M}, x)$, which is directly computable.
    - **Design Motivation**: Exact likelihood for diffusion models is intractable, so the Gaussian approximation from Diffusion-DPO is reused; MaskGIT, as a discrete token model, has naturally computable likelihood, providing a "clean" experimental setting to verify TGO's independence from diffusion-specific approximations.

### Loss & Training
The final loss is as above, $\mathcal L_{\text{TG}}$. Training hyperparameters: $\beta = 1$, diffusion temperature $T=0.001$, confidence scale $c=5$, batch size 128, 78 update steps (10K prompt set), learning rate $1\text{e}{-5}$. The threshold $\tau$ can be estimated on a smaller proxy set (generated by $\pi_{\text{ref}}$ + reward scoring) and reused for large datasets; estimation error also decays as $O(1/n)$ per the theorem. The SFT baseline uses the same optimization hyperparameters but trains only on pseudo-positive samples.

## Key Experimental Results

### Main Results
On SD v1.5, trained with Pick-a-Pic v2 (paired converted to scalar), compared to 7 baselines, across three test sets × five reward models:

| Test Set | Metric | SD v1.5 | Diffusion-DPO | Diffusion-KTO | TGO (Ours) |
|---|---|---|---|---|---|
| Pick-a-Pic | HPSv2.1 | 0.2469 | 0.2594 | 0.2814 | **0.2860** |
| Pick-a-Pic | ImageReward | 0.1131 | 0.3433 | 0.6381 | **0.6703** |
| PartiPrompts | PickScore | 21.15 | 21.41 | 21.50 | **21.55** |
| HPSv2 | ImageReward | 0.1384 | 0.3672 | 0.7365 | **0.7595** |
| HPSv2 | Aesthetic | 5.29 | 5.39 | 5.50 | **5.53** |

Cross-paradigm comparison on a 10K scalar feedback set:

| Paradigm | Model | HPSv2.1 | ImageReward | Aesthetic |
|---|---|---|---|---|
| Diffusion | SD v1.4 | 0.2454 | 0.1406 | 5.4277 |
| Diffusion | + SFT | 0.2506 | 0.2348 | 5.4927 |
| Diffusion | + TGO | **0.2618** | **0.3523** | **5.6036** |
| MaskGIT | Meissonic | 0.2810 | 0.8230 | 5.7692 |
| MaskGIT | + SFT | 0.2912 | 0.9215 | 5.8013 |
| MaskGIT | + TGO | **0.2915** | **0.9369** | **5.8270** |

### Ablation Study

| Configuration | Key Change | Impact |
|---|---|---|
| Full TGO | $\tau$=median, $c=5$ | Optimal in all dimensions |
| No confidence weighting ($c=0$) | Reduces to uniform BCE | Significant drop on high-variance metrics like ImageReward, validating the contribution of weighting to sample efficiency |
| Higher/lower $\tau$ percentile | Changes positive/negative sample ratio | Extreme percentiles yield too few positives, making supervision sparse; median is most stable |
| Single reward training → multi-reward evaluation | Cross-reward generalization | Also improves on unseen rewards, indicating TGO is not reward hacking |

### Key Findings
- Consistently outperforms Diffusion-DPO (paired control) across all reward dimensions, showing that the "paired preference assumption" is not essential—scalar scores plus threshold suffice.
- TGO is effective on both MaskGIT (exact likelihood) and diffusion (approximate likelihood), demonstrating paradigm-agnosticism.
- The threshold $\tau$ can be cheaply estimated from a proxy set, with theoretical error $O(1/n)$, making it engineering-friendly for large-scale training.

## Highlights & Insights
- **Theoretical Dissection of DPO**: The authors reveal that DPO's avoidance of $Z(x)$ is not due to paired preferences being "better", but because paired differences mathematically cancel $\log Z(x)$. With single samples, "pairing" loses its privilege—this re-examines the moat of DPO methods.
- **Confidence Weighting = Soft Margin**: Interpreting $w = 1 + c|s-\tau|$ as sample weights in classification is equivalent to a more aggressive "signal margin"—a simple trick directly transferable to any score-labeled scenario (e.g., LLM reward score-based fine-tuning).
- **Cross-Paradigm Unification**: The shared framework for diffusion + MaskGIT is an engineering highlight, showing TGO is not tied to the MSE assumption of diffusion, and is plug-and-play for future token-based video/3D generative models.

## Limitations & Future Work
- The global threshold $\tau$ implicitly assumes all prompts have similar optimal baselines, but $\tau^*(x)$ is inherently instance-dependent—hard prompts may require higher baselines, easy prompts lower. The paper does not compare prompt-conditional thresholds.
- In offline training, as $\pi_{\text{ref}}$ and $\pi_\theta$ diverge, pseudo-labels may become outdated; while the paper suggests optional rolling updates $\pi_{\text{ref}} \leftarrow \pi_\theta$, this is not systematically validated.
- Reward model bias is directly amplified (TGO lacks any "anti-reward hacking" mechanism); cross-reward evaluation improves but is still much less than on the training reward, so overfitting to the scorer remains a risk.
- Future directions: make $\tau$ a function of prompt embeddings; introduce online rollout to let $\tau$ track policy updates; combine TGO with GRPO as a critic estimator in actor-critic frameworks.

## Related Work & Insights
- **vs Diffusion-DPO**: DPO requires pairs, TGO only needs scalars; DPO cancels $Z(x)$ via differences, TGO approximates $\tau^*(x)$ with a global threshold. TGO consistently outperforms in all experiments.
- **vs Diffusion-KTO**: KTO also uses desirable/undesirable sets, based on the Kahneman-Tversky value function; TGO derives the threshold rule directly from the KL-optimal solution, with cleaner theory and fewer hyperparameters (KTO needs two weights).
- **vs QRPO**: QRPO transforms rewards into quantiles for analytic $Z$; TGO does not transform rewards, but "cuts" them into positive/negative via quantiles, logically closer to DPO's classification framework and lighter in engineering.
- **vs DSPO**: DSPO often degenerates to baseline on SD (several metrics match original SD v1.5), while TGO consistently improves, showing more thorough exploitation of score-based supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Theoretical decomposition of DPO as a "global threshold approximation" is a clean new perspective, though engineering-wise it is in the same unpaired family as KTO/QRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three test sets × five rewards × two generation paradigms × multiple baselines, good coverage, but lacks online policy comparisons and ablations on threshold conditioning.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from KL formula to algorithm is very clear, with appendix theorems providing complete guarantees on consistency, bias, and calibration.
- Value: ⭐⭐⭐⭐ High practical significance, as most reward data is naturally scalar rather than paired, and TGO directly reduces data collection costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator](../../ICML2025/image_generation/direct_discriminative_optimization_your_likelihood-based_visual_generative_model.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)

</div>

<!-- RELATED:END -->
