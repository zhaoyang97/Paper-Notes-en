---
title: >-
  [Paper Note] Threshold-Guided Optimization for Visual Generative Models
description: >-
  [ICML 2026][Image Generation][Threshold-guided] The authors dismantle the pairwise preference assumption of DPO, proving that the KL-regularized optimal policy inherently compares the reward of each sample with an uncomp…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Threshold-guided"
  - "unpaired preference optimization"
  - "scalar feedback"
  - "diffusion model alignment"
  - "MaskGIT"
date: 2026-05-08
content_hash: 64366c603df6aa58
---

# Threshold-Guided Optimization for Visual Generative Models

**Conference**: ICML 2026  
**arXiv**: [2605.04653](https://arxiv.org/abs/2605.04653)  
**Code**: None  
**Area**: Image Generation / Preference Alignment  
**Keywords**: Threshold-guided, unpaired preference optimization, scalar feedback, diffusion model alignment, MaskGIT

## TL;DR
The authors dismantle the pairwise preference assumption of DPO, proving that the KL-regularized optimal policy inherently compares the reward of each sample with an uncomputable instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$. Consequently, they substitute this with a global threshold $\tau$ estimated from score percentiles and introduce a confidence weight proportional to $|s-\tau|$. This enables stable alignment for diffusion models and MaskGIT even with scalar scores (no paired preferences), consistently outperforming Diffusion-DPO, KTO, and DSPO across five reward models and three test sets.

## Background & Motivation
**Background**: The mainstream approach for aligning visual generative models adapts RLHF/DPO from LLMs: first collecting paired preferences $(y_w, y_l)$, and then using the Bradley-Terry model to ensure $\pi_\theta$ assigns higher probability to $y_w$. Diffusion-DPO, AlignProp, and DSPO all follow this trajectory.

**Limitations of Prior Work**: In practical scenarios, feedback is often not paired but consists of 1–5 star ratings, continuous scores from reward models, or scalar human ratings for single images. Forcing these scores into pairs (comparing within the same batch) discards absolute numerical information, and when scores are clustered, artificial pairing can be amplified by noise. Diffusion-KTO bypasses pairing using desirable/undesirable sets but requires a hard split of the scores.

**Key Challenge**: The DPO family of methods avoids the intractable partition function $Z(x)$ in the KL-optimal solution because $\log Z(x)$ naturally cancels out in pairwise differences. Once only single-sample scalar scores are available, this cancellation mechanism no longer holds, and one must directly confront the instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$.

**Goal**: (i) Derive a **computable** proxy decision rule for KL-regularized alignment under scalar feedback; (ii) make the rule applicable to both diffusion models (MSE likelihood proxy) and MaskGIT (token-level cross-entropy likelihood); (iii) eliminate additional paired sampling overhead, enabling training through a pure offline, single-pass scoring process.

**Key Insight**: Starting from the KL-optimal solution, the authors find that the update direction of the optimal policy is essentially a binary decision—the probability of a sample should only be increased when its reward exceeds $\tau^*(x)$. Since $\tau^*(x)$ is difficult to compute, can a percentile (e.g., the median) of the rewards across the **entire dataset** serve as a unified threshold $\tau$ for approximation? Samples further from the threshold naturally provide stronger supervisory signals, which inspires "confidence weighting."

**Core Idea**: Use a percentile threshold $\tau$ of the empirical score distribution as a global proxy to replace the intractable instance-level baseline $\tau^*(x)$, transforming alignment into a binary classification task with confidence weighting to perform direct policy fitting on unpaired scalar feedback.

## Method

### Overall Architecture
The training workflow of TGO (Threshold-Guided Optimization) consists of four steps: (1) Given a reference policy $\pi_{\text{ref}}$ (i.e., $\pi_\theta$ at initialization), score the offline dataset $\{(x_i, y_i)\}$ using a reward model $r(\cdot)$ to obtain $s_i$; (2) Calculate a specific percentile of $\{s_i\}$ as $\tau = \text{Percentile}(\{s_i\}, p)$ (default $p=0.5$ for the median); (3) Generate a pseudo-label $l_i = \mathbb{1}[s_i \ge \tau]$ and a confidence weight $w_i = 1 + c|s_i - \tau|$ for each sample; (4) Use a sigmoid binary cross-entropy loss similar to DPO, but employing a unilateral implicit policy score $\hat r = \beta(\log \pi_\theta - \log \pi_{\text{ref}})$ instead of the difference between two sides. The entire process is offline, requiring no online rollouts or reward model fine-tuning.

### Key Designs

1.  **Threshold Decision Rule Derived from KL Optimum**:
    *   Function: Simplifies the question of "whether to increase a sample's probability" into a comparison with a global scalar threshold.
    *   Mechanism: The closed-form optimal solution for the KL-regularized objective $\max \mathbb E[\mathcal R(x,y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ satisfies $\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} > 0 \iff \mathcal R(x,y) > \tau^*(x)$, where $\tau^*(x) = \beta \log Z(x)$. The authors make this computable via two assumptions: the scalar score $s$ is a monotonic transformation of the reward, and the intractable instance-level $\tau^*(x)$ is replaced by a global percentile $\tau$ of the empirical distribution. Thus, the decision rule becomes $\pi_\theta(y|x) \gtrsim \pi_{\text{ref}}(y|x)$ when $s \ge \tau$.
    *   Design Motivation: DPO is elegant because pairwise differences cancel $\log Z(x)$; however, without pairs, $Z(x)$ must be addressed directly. A global threshold is the simplest and statistically justifiable proxy—the appendix provides a theorem proving the estimator is consistent as $n \to \infty$, with $O(1/n)$ error, calibrated to the original KL-optimal rule.

2.  **Confidence-Weighted Binary Classification Loss**:
    *   Function: Assigns larger gradients to samples whose scores are far from the threshold (more certain to be "good/bad") and smaller weights to samples near the threshold (ambiguous).
    *   Mechanism: Define the implicit policy score as $\hat s_{\theta,\text{ref}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. The loss is $\mathcal L_{\text{TG}} = -\mathbb E[w(s,\tau)(l\log\sigma(\hat s) + (1-l)\log(1-\sigma(\hat s)))]$, where $w(s,\tau) = 1 + c|s-\tau|$ and hyperparameter $c \ge 0$. This is equivalent to a weighted BCE that pushes the policy towards high rewards and pulls it away from low rewards.
    *   Design Motivation: Samples near the median are naturally in a "gray area"; treating them with the same weight as extreme samples introduces noise. Linear confidence weighting preserves the utilization of the entire dataset while naturally enhancing the signal-to-noise ratio, without introducing hyperparameter sensitivity ($c=5$ is stable across tasks in experiments).

3.  **Likelihood Proxies for Two Types of Visual Generative Models**:
    *   Function: Enables the computation of $\log \pi_\theta(y|x)$ for both diffusion models and MaskGIT, allowing the TGO framework to adapt to both continuous and discrete generation paradigms.
    *   Mechanism: For diffusion models, it uses the Gaussian observation assumption $\log \pi_\theta(y|x) \approx -\frac{1}{T}\text{MSE}(y, \hat y_\theta(x))$, with temperature $T$ controlling the scale (default $T=0.001$). For MaskGIT, the log-likelihood of masked positions after VQ-GAN tokenization $\log \pi_\theta(y|x) = \frac{1}{|M|}\sum_{i\in M}\log p_\theta(t_i | y_{\setminus M}, x)$ is directly computable.
    *   Design Motivation: Since the exact likelihood of diffusion models is intractable, the Gaussian approximation from Diffusion-DPO is used to avoid reinventing the wheel. MaskGIT, being a discrete token model, has naturally computable likelihoods, providing a "cleaner" experimental setting to verify that TGO does not rely on diffusion-specific approximations.

### Loss & Training
The final loss is $\mathcal L_{\text{TG}}$ as defined above. Training hyperparameters: $\beta = 1$, diffusion temperature $T=0.001$, confidence scale $c=5$, batch size 128, 78 update steps (on a 10K prompt set), and a learning rate of $1\text{e}{-5}$. In large-scale scenarios, the threshold $\tau$ can be estimated on a smaller proxy set (generated by $\pi_{\text{ref}}$ and scored by the reward model) and then reused; the estimation error also decays at $O(1/n)$ according to the theorem. The SFT baseline uses the same optimization hyperparameters but is trained only on pseudo-positive samples.

## Key Experimental Results

### Main Results
Trained on SD v1.5 using Pick-a-Pic v2 (converted from pairs to scalars), compared against 7 baselines across three test sets and five reward models:

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
| MaskGIT | + TGO | **0.2915** | **0.9369** | **0.8270** |

### Ablation Study

| Configuration | Key Change | Impact |
|---|---|---|
| Full TGO | $\tau$=Median, $c=5$ | Optimal across all dimensions. |
| No Weighting ($c=0$) | Degenerates to equal-weight BCE | Significant drop in high-variance metrics like ImageReward, validating sample efficiency. |
| Shift $\tau$ Percentile | Changes positive/negative ratio | Extreme percentiles lead to sparse signals; median is the most stable. |
| Single Reward Training | Generalization | Improves on un-trained reward models, indicating TGO is not just reward hacking. |

### Key Findings
- Consistently outperforms Diffusion-DPO (paired control) across all reward dimensions, indicating that the "pairwise preference assumption" is not strictly necessary; scalar scores with a threshold are sufficient.
- TGO is effective for both MaskGIT (exact likelihood) and Diffusion (approximate likelihood), proving the method is unbiased towards generation paradigms.
- The threshold $\tau$ can be estimated cheaply using a proxy set; the error is theoretically $O(1/n)$, making it very friendly for large-scale engineering.

## Highlights & Insights
- **Theoretically Deconstructing DPO**: The authors clarify that DPO avoids $Z(x)$ not because pairwise preferences are "more correct," but because pairwise differences mathematically cancel $\log Z(x)$. Once switched to single samples, "pairs" no longer hold a privileged position—this re-examines the competitive advantage of the DPO family.
- **Confidence Weighting = Soft Margin**: Viewing $w = 1 + c|s-\tau|$ as sample weights in a classification problem is equivalent to being more aggressive on the "signal margin." This is a concise trick that can be directly transferred to any score-based label scenario (e.g., LLM reward score-based fine-tuning).
- **Cross-Paradigm Unification**: Using the same framework for both Diffusion and MaskGIT is an engineering highlight, showing that TGO is not tied to the MSE assumption of diffusion and can be plug-and-play for future token-based video or 3D generation models.

## Limitations & Future Work
- The global threshold $\tau$ implicitly assumes the optimal baseline for all prompts is "similar," but $\tau^*(x)$ is essentially instance-dependent—difficult prompts might should have higher baselines, and simple prompts lower. The paper does not compare against prompt-conditional thresholds.
- Under offline training, the divergence between $\pi_{\text{ref}}$ and the evolving $\pi_\theta$ may make pseudo-labels outdated; although the paper suggests an optional $\pi_{\text{ref}} \leftarrow \pi_\theta$ rolling update, it is not systematically validated.
- The bias of the reward model itself is directly amplified (TGO lacks any "de-reward hacking" mechanism). While cross-reward evaluation shows improvement, it is much smaller than the gains on the training reward, maintaining a risk of overfitting to the scorer.
- Future directions: Modeling $\tau$ as a function of prompt embeddings; introducing online rollouts to let $\tau$ follow policy updates; combining TGO with GRPO as the critic estimate in actor-critic.

## Related Work & Insights
- **vs Diffusion-DPO**: DPO requires pairs, TGO only scalars; DPO cancels $Z(x)$ via differences, TGO approximates $\tau^*(x)$ with a global threshold. TGO is consistently superior in all experiments.
- **vs Diffusion-KTO**: KTO also uses desirable/undesirable sets but is based on the Kahneman-Tversky value function; TGO derives threshold rules directly from the KL optimum, offering a cleaner theory with fewer hyperparameters (KTO requires separate weights for desirable/undesirable sets).
- **vs QRPO**: QRPO uses percentile transforms on rewards to resolve $Z$ analytically; TGO does not transform rewards but uses percentiles to "cut" them into positive/negative samples, aligning closer to DPO's classification framework and remaining lighter for engineering.
- **vs DSPO**: DSPO often degenerates to the baseline on SD (identical to original SD v1.5 across metrics), whereas TGO shows consistent improvement, proving a more thorough exploitation of score-based supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Theoretically deconstructing DPO as a "global threshold approximation" is a clean new perspective, though it belongs to the same unpaired route as KTO/QRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage across three test sets, five rewards, two generation paradigms, and multiple baselines, though lacking comparisons under online policies and ablations for conditional thresholds.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from KL formulas to the algorithm is very clear; appendix theorems provide full guarantees for consistency, bias, and calibration.
- Value: ⭐⭐⭐⭐ Significant for practical engineering, as most reward data is naturally scalar rather than paired; TGO directly reduces data collection costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICML 2026\] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)
- [\[ICML 2026\] Conf-Gen: Conformal Uncertainty Quantification for Generative Models](conf-gen_conformal_uncertainty_quantification_for_generative_models.md)

</div>

<!-- RELATED:END -->
