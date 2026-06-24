---
title: >-
  [Paper Note] Reinforcing Diffusion Models by Direct Group Preference Optimization
description: >-
  [ICLR 2026][Image Generation][Diffusion Models] This paper proposes DGPO (Direct Group Preference Optimization), which decouples the "intra-group relative preference" concept of GRPO from the policy-gradient framework. This allows diffusion models to perform online RL post-training using efficient deterministic ODE samplers, improving SD3.5-M from 0.63 to 0.97 on GenEval while training approximately 20× faster than Flow-GRPO (nearly 30× on GenEval).
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Preference Optimization"
  - "GRPO"
  - "ODE Sampling"
  - "Post-training"
date: 2026-05-08
content_hash: 5fdf5c4697ef7fc5
---

# Reinforcing Diffusion Models by Direct Group Preference Optimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jymuXl8GYi](https://openreview.net/forum?id=jymuXl8GYi)  
**Code**: https://github.com/Luo-Yihong/DGPO  
**Area**: Diffusion Models / Alignment RLHF  
**Keywords**: Diffusion Models, Preference Optimization, GRPO, ODE Sampling, Post-training

## TL;DR
This paper proposes DGPO (Direct Group Preference Optimization), which decouples the "intra-group relative preference" concept of GRPO from the policy-gradient framework. This allows diffusion models to perform online RL post-training using efficient deterministic ODE samplers, improving SD3.5-M from 0.63 to 0.97 on GenEval while training approximately 20× faster than Flow-GRPO (nearly 30× on GenEval).

## Background & Motivation
**Background**: RL post-training has become standard for Large Language Model (LLM) alignment. GRPO (Group Relative Policy Optimization) significantly enhances LLM reasoning by sampling multiple outputs per prompt and calculating advantages via intra-group normalization. Naturally, the community aims to adapt this to diffusion models to align with human preferences and improve complex metrics like composition, counting, and text rendering.

**Limitations of Prior Work**: The policy-gradient framework of GRPO requires a **stochastic policy** for exploration and importance sampling. While LLMs naturally output probability distributions over a vocabulary, diffusion models typically use **deterministic ODE samplers** for a balance of quality and efficiency, which do not provide a stochastic policy. To introduce randomness, existing works (like Flow-GRPO) resort to SDE sampling by injecting conditional Gaussian noise. This compromise leads to three issues: (1) SDE sampling is less efficient and yields lower quality than ODE under fixed compute budgets; (2) randomness comes from **model-independent Gaussian noise**, resulting in weak exploration signals in high-dimensional space and slow convergence; (3) training must occur over the **entire sampling trajectory**, making each iteration expensive.

**Key Challenge**: There is a fundamental conflict between efficient diffusion samplers (ODE, deterministic) and the stochastic policy requirement of the GRPO framework—using a fast sampler precludes obtaining the stochastic policy needed for policy-gradient.

**Key Insight**: The authors argue that the success of GRPO **stems from its utilization of fine-grained intra-group relative preference information, rather than the policy-gradient form itself**. If this holds, an ideal diffusion RL method should retain "group-level relative information" while discarding the stochastic policy and its side effects.

**Core Idea**: Replace "policy-gradient" with "Direct Group Preference Optimization." For each prompt, sample a group using an ODE, split them into positive and negative sets based on rewards, and directly maximize the group-level preference likelihood that the "positive group is better than the negative group." This retains GRPO's intra-group relative information while unlocking deterministic ODE sampling and faster training. DGPO can be viewed as an extension of DPO to group-level information or a "diffusion-native" rewrite of GRPO.

## Method

### Overall Architecture
DGPO is an online RL algorithm where each iteration performs: generating a group of $G$ samples using an efficient ODE sampler from the current model (or its EMA version), scoring each sample with a reward model, applying intra-group normalization to obtain advantages, splitting the group into "good samples $G^+$" and "bad samples $G^-$" based on advantage signs, and optimizing the group-level preference "$G^+ \succ G^-$" via a Bradley-Terry model. This process **requires no stochastic policy, no training on the full trajectory, and no handling of the partition function $Z(c)$**—the three factors enabling it to be an order of magnitude faster than Flow-GRPO.

The core challenge of DGPO is the intractable partition function $Z(c)$ inherent in group-level preference objectives, which is resolved through carefully designed sample weights. The pipeline is as follows:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Prompt c"] --> B["Deterministic ODE Sampling<br/>Generate Group G"]
    B --> C["Reward Scoring<br/>+ Intra-group Normalization for Advantage"]
    C --> D["Split into G+ / G- Based on Advantage Sign"]
    D --> E["Advantage Weight Design w=|A|<br/>Canceling Partition Z(c)"]
    E --> F["Group Preference Optimization<br/>Maximize σ(R(G+)-R(G-))"]
    F -->|Timestep Clip: Sample t∈[t_min,T]| G["Update θ, θ⁻←EMA"]
    G -->|Next Online Rollout Cycle| B
```

### Key Designs

**1. Direct Group Preference Optimization: Discarding policy-gradient for "Good Group ≻ Bad Group" learning**

Addressing the stochastic policy dependency of GRPO and the pairwise limitation of DPO, DGPO applies maximum likelihood to **group-level preferences** using the Bradley-Terry model:

$$\max_\theta\ \mathbb{E}_{(G^+,G^-,c)\sim D}\ \log \sigma\big(R_\theta(G^+|c) - R_\theta(G^-|c)\big)$$

The group-level reward is parameterized as a **weighted sum** of individual sample rewards: $R_\theta(G|c)=\sum_{x_0\in G} w(x_0)\cdot r_\theta(c,x_0)$. This incorporates fine-grained information from every sample, unlike Diffusion-DPO which is limited to pairwise comparisons. Individual rewards follow the DPO implicit parameterization $r_\theta(c,x_0)=\beta\,\mathbb{E}\log\frac{p_\theta(x_{0:T}|c)}{p_{\text{ref}}(x_{0:T}|c)}+\beta\log Z(c)$ (approximated via forward diffusion $q(x_{1:T}|x_0)$). The essential difference from GRPO is that DGPO embeds "intra-group relative information" directly into the preference objective rather than via policy-gradient, eliminating the need for stochastic policies or full trajectory rollouts.

**2. Advantage Weight Design: Using $w=|A|$ to cancel the partition function and amplify hard samples**

Expanding the objective yields a term $\sum_{x_0\in G^+}w(x_0)Z(c)-\sum_{x_0\in G^-}w(x_0)Z(c)$. Since $Z(c)$ is intractable, it must be canceled. The authors specify weights such that higher weights correspond to "better positive samples and worse negative samples," and the sum of weights for both groups are equal: $\sum_{G^+}w=\sum_{G^-}w$. This is achieved by reusing GRPO-style normalized advantages:

$$A(x_0^i)=\frac{r_i-\text{mean}(\{r_j\})}{\text{std}(\{r_j\})},\qquad G^+=\{A>0\},\ G^-=\{A\le0\},\qquad w(x_0)=|A(x_0)|$$

Since normalized advantages are **zero-mean**, the sums of $|A|$ for positive and negative groups are naturally equal, canceling the partition term automatically. Simultaneously, $|A|$ assigns higher weights to samples farther from the mean, allowing the model to learn intra-group relative preferences more effectively. The objective simplifies to a clean form containing only $\log\frac{p_\theta}{p_{\text{ref}}}$, which, using Jensen's inequality, results in a **denoising score matching difference** training objective:

$$L_{\text{DGPO}}\triangleq-\mathbb{E}\,\log\sigma\Big(-\lambda_t\beta T\big(\textstyle\sum_{x\in G^+}w(x)[L^\theta_{\text{dsm}}-L^{\theta_{\text{ref}}}_{\text{dsm}}]-\sum_{x\in G^-}w(x)[L^\theta_{\text{dsm}}-L^{\theta_{\text{ref}}}_{\text{dsm}}]\big)\Big)$$

Where $L^\theta_{\text{dsm}}(x,x_t,c)=\|f_\theta(x_t,t,c)-x\|_2^2$ is the standard denoising loss. Consequently, DGPO only requires calculating the denoising loss difference at a **single timestep** rather than traversing the full trajectory, drastically reducing iteration costs.

**3. ODE Deterministic Rollout: Stronger signals through high-quality sampling**

Because DGPO no longer requires a stochastic policy, deterministic ODE samplers can be used for rollouts. Under the same inference budget, ODEs produce samples with higher quality and higher rewards. The model thus learns from "cleaner training data," leading to faster convergence and higher performance ceilings. Ablations (Fig. 5) show that online DGPO with ODE rollouts significantly outperforms SDE rollouts in both convergence speed and final metrics, suggesting that the previous use of SDE was a constraint of the policy-gradient framework rather than a source of useful diversity.

**4. Timestep Clip Strategy: Avoiding artifacts from few-step sample overfitting**

Online settings require real-time sampling from the current model. To save costs, few steps (e.g., 10 steps) are used for generation. However, few-step samples often contain artifacts (e.g., blurriness). Training on all timesteps would cause the model to learn these artifacts, leading to severe visual degradation. The solution is to sample timesteps only from the interval $[t_{\min}, T]$ ($t_{\min}>0$), skipping small $t$ values closest to the data manifold where artifacts are most prominent. Ablations show that removing this strategy causes visual quality to collapse even if reward values remain high.

### Loss & Training
The final objective is $L_{\text{DGPO}}$ (Eq. 17). Training loop (Algorithm 1): Sample prompts → Generate group $G$ using $p_{\theta^-}$ → Compute rewards and normalized advantages → Split $G^+/G^-$ based on sign → Sample $t\sim U[t_{\min},T], \epsilon\sim\mathcal N(0,I)$ (**sharing the same noise** $\epsilon$ within a group to reduce variance) → Compute $L_{\text{DGPO}}$ and update $\theta$ → Update online model via $\theta^-\leftarrow\theta$ or EMA $\theta^-\leftarrow\mu\theta^-+(1-\mu)\theta$. Key hyperparameters include group size $G$, regularization strength $\beta$, minimum timestep $t_{\min}$, and EMA decay $\mu$.

## Key Experimental Results

### Main Results
Post-training on SD3.5-M across three tasks: Composition Generation (GenEval), Visual Text Rendering (OCR), and Human Preference Alignment (PickScore).

| Task/Metric | SD3.5-M Baseline | Flow-GRPO | DGPO (Ours) |
|-----------|--------------|-----------|--------------|
| GenEval Overall | 0.63 | 0.95 | **0.97** |
| GenEval Counting | 0.50 | 0.95 | **0.97** |
| GenEval Attr. Binding | 0.52 | 0.86 | **0.91** |
| OCR Text Acc. | 0.59 | 0.92 | **0.96** |
| PickScore | 21.72 | 23.31 | **23.89** |

On GenEval, DGPO (0.97) outperforms GPT-4o (0.84) and Flow-GRPO (0.95), while training nearly 30× faster than Flow-GRPO. It is ~19× faster on OCR and ~17× faster on PickScore.

### Out-of-Distribution Metrics (Reward Hacking Prevention)
Tested on DrawBench using four quality metrics not used during training:

| Task | Method | Aesthetic | DeQA | ImageReward | UnifiedReward |
|------|------|-----------|------|-------------|---------------|
| Composition | Flow-GRPO | 5.25 | 4.01 | 1.03 | 3.51 |
| Composition | DGPO | **5.31** | **4.03** | **1.08** | **3.60** |
| Human Preference | Flow-GRPO | 5.92 | 4.22 | 1.28 | 3.66 |
| Human Preference | DGPO | **6.08** | **4.40** | **1.32** | **3.74** |

DGPO improves target metrics while maintaining or improving OOD quality metrics, indicating no sacrifice of general image quality for reward exploitation.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| DGPO (Full) | OCR 0.96 | ODE rollout + Timestep Clip |
| w/o Timestep Clip | OCR 0.95 | Metric drops slightly, but visual quality severely degrades (Fig. 4). |
| DGPO w/ SDE rollout | Lower Limit | Performance drops significantly when switching to SDE. |
| Online DPO | Much lower | Pairwise comparisons lose fine-grained intra-group info. |
| Offline DGPO | > Baseline | Improving on baseline but weaker than online version. |

### Key Findings
- **ODE outperforms SDE**: Removing policy-gradient allows deterministic ODEs, which are faster and stronger, proving SDE was a framework limitation rather than a benefit.
- **Group > Pairwise**: DGPO significantly outperforms DPO in both online and offline settings, validating that intra-group relative information is the key factor.
- Timestep Clip primarily preserves visual quality; without it, images collapse despite high reward numbers.

## Highlights & Insights
- **Decoupling GRPO from policy-gradient**: The insight that GRPO's success comes from relative preference rather than the policy-gradient form allows the use of ODE samplers, which is the most significant "Aha!" moment of the paper.
- **The $w=|A|$ trick**: The zero-mean property of normalized advantages balances positive/negative weights and cancels the intractable partition function $Z(c)$—solving a mathematical hurdle while naturally weighting hard samples.
- **Denoising Loss Difference**: The complex group preference objective reduces to a simple difference in single-step denoising losses, making it easy to implement and integrate into any diffusion/flow-matching pipeline.
- Viewing DGPO as "DPO + Group Info" or a "Diffusion-native GRPO" provides a clear position in the RLHF taxonomy.

## Limitations & Future Work
- Experiments are limited to SD3.5-M; scalability to larger models (SD3.5-L, FLUX) or video diffusion is unverified.
- Group-level rewards depend on external reward model quality; biases in reward models may be amplified.
- $t_{\min}$ is task-dependent and may require tuning when changing sampling steps or base models.
- Offline performance is notably weaker than online, suggesting benefits rely heavily on online rollouts.

## Related Work & Insights
- **vs Flow-GRPO**: Both port GRPO to diffusion, but Flow-GRPO is forced into SDE sampling and full-trajectory training; DGPO uses ODE and single-step training, resulting in ~20× speedup and better quality.
- **vs Diffusion-DPO**: DPO avoids stochastic policies but is limited to pairwise comparisons. DGPO uses advantage weighting to enable group-level information capture.
- **vs Enumerative Group DPO (Concurrent)**: Some concurrent works use pairwise enumeration within groups; DGPO defines a single group reward for maximum likelihood, which is more computationally efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing GRPO as "group-level relative preference" is a paradigm-level insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid across three tasks and OOD metrics, though limited to one base model.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear chain of logic from motivation to mathematical derivation.
- Value: ⭐⭐⭐⭐⭐ 20× speedup with SOTA quality offers high practical utility for diffusion RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)
- [\[ICLR 2026\] Consis-GCPO: Consistency-Preserving Group Causal Preference Optimization for Vision Customization](consis-gcpo_consistency-preserving_group_causal_preference_optimization_for_visi.md)
- [\[ICLR 2026\] Towards Better Optimization for Listwise Preference in Diffusion Models](towards_better_optimization_for_listwise_preference_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
