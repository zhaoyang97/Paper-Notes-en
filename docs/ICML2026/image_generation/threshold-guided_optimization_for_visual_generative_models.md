---
title: >-
  [Paper Note] Threshold-Guided Optimization for Visual Generative Models
description: >-
  [ICML 2026][Image Generation][Threshold-guided] The authors dismantle the paired preference assumption of DPO, proving that the KL-regularized optimal policy inherently compares the reward of each sample to an uncomputable instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$. Consequently, they replace it with a global threshold $\tau$ estimated from score quantiles and incorporate a confidence weight proportional to $|s-\tau|$. This enables stable alignment for Diffusion m…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Threshold-guided"
  - "unpaired preference optimization"
  - "scalar feedback"
  - "diffusion model alignment"
  - "MaskGIT"
date: 2026-05-08
content_hash: 909a28ae22313a70
---

# Threshold-Guided Optimization for Visual Generative Models

**Conference**: ICML 2026  
**arXiv**: [2605.04653](https://arxiv.org/abs/2605.04653)  
**Code**: None  
**Area**: Image Generation / Preference Alignment  
**Keywords**: Threshold-guided, unpaired preference optimization, scalar feedback, diffusion model alignment, MaskGIT

## TL;DR
The authors dismantle the paired preference assumption of DPO, proving that the KL-regularized optimal policy inherently compares the reward of each sample to an uncomputable instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$. Consequently, they replace it with a global threshold $\tau$ estimated from score quantiles and incorporate a confidence weight proportional to $|s-\tau|$. This enables stable alignment for Diffusion models and MaskGIT using only scalar scores (without paired preferences), consistently outperforming Diffusion-DPO, KTO, and DSPO across five reward models and three test sets.

## Background & Motivation
**Background**: The mainstream approach for aligning visual generative models involves adapting RLHF or DPO from LLMs: first collecting paired preferences $(y_w, y_l)$, and then utilizing the Bradley-Terry model to ensure $\pi_\theta$ assigns a higher probability to $y_w$. Diffusion-DPO, AlignProp, and DSPO all follow this trajectory.

**Limitations of Prior Work**: In practical scenarios, feedback is often not paired but consists of 1–5 star ratings, continuous reward model scores, or scalar scores for single images. Forcing these scores into pairs (comparing within the same batch) discards absolute numerical information, and noise is amplified when human pairs are constructed from similar scores. Diffusion-KTO avoids pairs using desirable/undesirable sets but requires a hard split of the scores into two groups.

**Key Challenge**: DPO-based methods circumvent the intractable partition function $Z(x)$ in the KL-optimal solution because $\log Z(x)$ naturally cancels out in paired differences. Once only single-sample scalar scores are available, this cancellation mechanism no longer holds, necessitating direct confrontation with the instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$.

**Goal**: (i) Derive a **computable** proxy decision rule for KL-regularized alignment under scalar feedback; (ii) Ensure the rule is applicable to both Diffusion models (MSE likelihood proxy) and MaskGIT (token-level cross-entropy likelihood); (iii) Eliminate the overhead of additional paired sampling through pure offline, single-pass score-based training.

**Key Insight**: Starting from the KL-optimal solution, the authors find that the update direction of the optimal policy is essentially a binary decision—the probability should only be increased if the sample reward exceeds $\tau^*(x)$. Since $\tau^*(x)$ is difficult to compute, they investigate using a specific quantile (e.g., the median) of rewards across the **entire dataset** as a unified threshold $\tau$ for approximation. Samples far from the threshold naturally provide stronger supervisory signals, inspiring "confidence weighting."

**Core Idea**: Use the quantile threshold $\tau$ of the empirical score distribution as a global proxy to replace the intractable instance-level baseline $\tau^*(x)$. This transforms alignment into a binary classification task with confidence weighting, allowing for direct policy fitting on unpaired scalar feedback.

## Method

### Overall Architecture
TGO (Threshold-Guided Optimization) addresses how to align visual generative models when only scalar scores are available without paired preferences. Its core transformation rewrites alignment from "comparing which of two samples is better" to "determining if a single sample is good": first, an offline dataset is scored using a reward model; then, a global quantile threshold is taken from the empirical distribution of these scores to label each sample as pseudo-positive or pseudo-negative; finally, a weighted binary cross-entropy (BCE) loss with confidence weighting is used to push the policy toward pseudo-positives and pull it away from pseudo-negatives. The entire process is purely offline, requiring no online rollouts or reward model fine-tuning.

### Key Designs

**1. Threshold Decision Rule Derived from KL-Optimal Solution: Simplifying "Whether to Increase Probability" to Comparison with a Global Threshold**

DPO methods avoid the intractable partition function $Z(x)$ through paired differences that cancel $\log Z(x)$. Under single-sample scalar scores, this fails. The authors start with the closed-form optimal solution of the KL-regularized objective $\max \mathbb E[\mathcal R(x,y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ and find it is equivalent to a binary decision: $\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} > 0 \iff \mathcal R(x,y) > \tau^*(x)$, where the instance-dependent baseline $\tau^*(x) = \beta \log Z(x)$ is hard to compute. By assuming the scalar score $s$ is a monotonic transformation of the reward and replacing $\tau^*(x)$ with a global quantile $\tau = \text{Percentile}(\{s_i\}, p)$ (default $p=0.5$ median), the rule becomes: "When $s \ge \tau$, ensure $\pi_\theta(y|x) \gtrsim \pi_{\text{ref}}(y|x)$." This global threshold is a simple and statistically guaranteed proxy: theorems in the appendix prove this estimator is consistent as $n \to \infty$ with $O(1/n)$ error, calibrating back to the original KL-optimal rule.

**2. Confidence-Weighted Binary Classification Loss: Assigning Higher Gradients to Samples Further from the Threshold**

Samples near the median are in a "grey zone" where it is hard to distinguish good from bad; treating them equally with extreme samples would introduce noise into the gradients. Thus, the authors define an implicit policy score $\hat s_{\theta,\text{ref}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ and pseudo-labels $l = \mathbb 1[s \ge \tau]$. The loss is formulated as a weighted binary cross-entropy: $\mathcal L_{\text{TG}} = -\mathbb E[w(s,\tau)(l\log\sigma(\hat s) + (1-l)\log(1-\sigma(\hat s)))]$, where the confidence weight $w(s,\tau) = 1 + c|s-\tau|$ with hyperparameter $c \ge 0$. The further a score is from the threshold (more certain), the higher the weight. This maintains full dataset utilization while enhancing the signal-to-noise ratio, effectively adding a soft margin to the classification, and is insensitive to hyperparameters (stable at $c=5$ across tasks).

**3. Likelihood Proxies for Two Types of Visual Generative Models: Supporting both Continuous Diffusion and Discrete Token Paradigms**

The loss requires $\log \pi_\theta(y|x)$, which is computed differently across models. For Diffusion models, where the exact likelihood is intractable, the Gaussian observation approximation from Diffusion-DPO is used: $\log \pi_\theta(y|x) \approx -\frac{1}{T}\text{MSE}(y, \hat y_\theta(x))$, with temperature $T$ controlling scale (default $T=0.001$). MaskGIT is a discrete token model where the log-likelihood of masked positions after VQ-GAN tokenization $\log \pi_\theta(y|x) = \frac{1}{|M|}\sum_{i\in M}\log p_\theta(t_i | y_{\setminus M}, x)$ is naturally computable. This provides a "cleaner" experimental setting, verifying that TGO does not rely on Diffusion-specific MSE approximations and is unbiased across generation paradigms.

### Loss & Training
The final loss is $\mathcal L_{\text{TG}}$ as defined above. Training hyperparameters: $\beta = 1$, diffusion temperature $T=0.001$, confidence scale $c=5$, batch size 128, 78 update steps (10K prompt set), learning rate $1\text{e}{-5}$. For large-scale data, the threshold $\tau$ can be estimated on a smaller proxy set (generated by $\pi_{\text{ref}}$ and scored by the reward model) and reused, with estimation error decaying at $O(1/n)$. The SFT baseline uses identical optimization hyperparameters but is trained only on pseudo-positive samples.

## Key Experimental Results

### Main Results
Training on SD v1.5 using Pick-a-Pic v2 (converted from pairs to scalars), compared against 7 baselines across three test sets and five reward models:

| Test Set | Metric | SD v1.5 | Diffusion-DPO | Diffusion-KTO | TGO (Ours) |
|---|---|---|---|---|---|
| Pick-a-Pic | HPSv2.1 | 0.2469 | 0.2594 | 0.2814 | **0.2860** |
| Pick-a-Pic | ImageReward | 0.1131 | 0.3433 | 0.6381 | **0.6703** |
| PartiPrompts | PickScore | 21.15 | 21.41 | 21.50 | **21.55** |
| HPSv2 | ImageReward | 0.1384 | 0.3672 | 0.7365 | **0.7595** |
| HPSv2 | Aesthetic | 5.29 | 5.39 | 5.50 | **5.53** |

Cross-paradigm comparison on the 10K scalar feedback set:

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
| Full TGO | $\tau$=median, $c=5$ | Optimal across all dimensions. |
| No confidence weighting ($c=0$) | Degenerates to equal-weighted BCE | Significant drop in high-variance metrics like ImageReward, confirming weighting benefits sample efficiency. |
| Increasing/Decreasing $\tau$ quantile | Changes pos/neg ratio | Extreme quantiles lead to too few positive samples and sparse signals; median is most stable. |
| Single reward train → Multi reward eval | Cross-reward generalization | Improvements even on untrained rewards, suggesting TGO is not simply reward hacking. |

### Key Findings
- Consistently outperforms Diffusion-DPO (paired control) across all reward dimensions, indicating the "paired preference assumption" is not strictly necessary; scalar scores with a threshold suffice.
- TGO is effective for both MaskGIT (exact likelihood) and Diffusion (approximate likelihood), proving the method is unbiased toward the generation paradigm.
- The threshold $\tau$ can be estimated cheaply using a proxy set; theoretically, the error is $O(1/n)$, making it very friendly for large-scale engineering.

## Highlights & Insights
- **Deconstructing DPO Theoretically**: The authors clarify that DPO circumvents $Z(x)$ not because paired preferences are "more correct," but because paired differences mathematically cancel $\log Z(x)$. Once switched to single samples, "pairs" no longer hold a privileged status—re-examining the moat of the DPO series.
- **Confidence Weighting as a Soft Margin**: Treating $w = 1 + c|s-\tau|$ as a sample weight in a classification problem is equivalent to making the model more aggressive regarding the "signal margin." This is a concise trick that can be transferred to any score-based label scenario (e.g., reward score-based fine-tuning in LLMs).
- **Cross-Paradigm Unification**: Using a shared framework for both Diffusion and MaskGIT is an engineering highlight, showing that TGO is not bound to the MSE assumptions of Diffusion and can be integrated into future token-based video or 3D generation models.

## Limitations & Future Work
- The global threshold $\tau$ implicitly assumes the optimal baseline for all prompts is similar, but $\tau^*(x)$ is inherently instance-dependent—difficult prompts should arguably have a higher baseline, and simple ones lower. The paper does not compare against prompt-conditional thresholds.
- Under offline training, the gap between $\pi_{\text{ref}}$ and $\pi_\theta$ grows during training, which may make pseudo-labels obsolete; although the paper suggests an optional rolling update $\pi_{\text{ref}} \leftarrow \pi_\theta$, it is not systematically verified.
- Biases in the reward model itself are directly amplified (TGO lacks a "de-reward hacking" mechanism). While cross-reward evaluation shows improvement, it is much smaller than the gain on the training reward, maintaining a risk of overfitting the scorer.
- Future Directions: Making $\tau$ a function of prompt embeddings; introducing online rollouts to allow $\tau$ to follow policy updates; combining TGO with GRPO as a critic estimator in an actor-critic setup.

## Related Work & Insights
- **vs Diffusion-DPO**: DPO requires pairs, TGO requires only scalars; DPO cancels $Z(x)$ via differences, TGO approximates $\tau^*(x)$ via a global threshold. TGO is consistently superior in all experiments.
- **vs Diffusion-KTO**: KTO also uses desirable/undesirable sets but is based on the Kahneman-Tversky value function; TGO derives threshold rules directly from the KL-optimal solution, offering cleaner theory and fewer hyperparameters (KTO requires two weights for desirable/undesirable).
- **vs QRPO**: QRPO uses a quantile transformation on rewards to make $Z$ analytical; TGO does not transform rewards but uses quantiles to "split" them into positive/negative, which is logically closer to DPO's classification framework and engineering-wise lighter.
- **vs DSPO**: DSPO often degenerates to the baseline on SD (matching original SD v1.5 on several metrics); TGO consistently improves, proving a more thorough exploitation of score-based supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Theoretically decomposing DPO into a "global threshold approximation" is a clean new perspective, though engineering-wise it aligns with the unpaired route of KTO/QRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ High coverage with three test sets, five rewards, two generation paradigms, and multiple baselines, but lacks comparison with online policies and prompt-conditioned threshold ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain from KL formulas to the algorithm; appendix theorems provide complete guarantees for consistency, bias, and calibration.
- Value: ⭐⭐⭐⭐ Historically significant for practical engineering, as most reward data is naturally scalar rather than paired; TGO directly reduces data collection costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[CVPR 2026\] Learning What to Trust: Bayesian Prior-Guided Optimization for Visual Generation](../../CVPR2026/image_generation/learning_what_to_trust_bayesian_prior-guided_optimization_for_visual_generation.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](../../ICLR2026/image_generation/vipo_visual_preference_optimization_at_scale.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[CVPR 2026\] Learning What to Trust: Bayesian Prior-Guided Optimization for Visual Generation](../../CVPR2026/image_generation/learning_what_to_trust_bayesian_prior-guided_optimization_for_visual_generation.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[CVPR 2026\] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation](../../CVPR2026/image_generation/seeing_what_matters_visual_preference_policy_optimization_for_visual_generation.md)

</div>

<!-- RELATED:END -->
