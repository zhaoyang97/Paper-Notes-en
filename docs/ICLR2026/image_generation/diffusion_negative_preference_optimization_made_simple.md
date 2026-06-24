---
title: >-
  [Paper Note] Diffusion Negative Preference Optimization Made Simple
description: >-
  [ICLR2026][Image Generation][Diffusion Model Alignment] Addressing the cumbersome practice of "training two models + weight merging" for explicit negative preference modeling in diffusion alignment, this paper proposes Diff-SNPO. It utilizes the inherent conditional/unconditional branches of CFG as outlets for positive/negative preferences within a **single network**. By adapting a bounded objective from Bounded DPO, it resolves the "progressive blurring" issue of naive appro…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "Diffusion Model Alignment"
  - "Negative Preference Optimization"
  - "Classifier-Free Guidance"
  - "Bounded DPO"
  - "Single Network"
date: 2026-05-08
content_hash: e7cdadc0809e64d5
---

# Diffusion Negative Preference Optimization Made Simple

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=CU5EHe1KUt](https://openreview.net/forum?id=CU5EHe1KUt)  
**Code**: https://github.com/JoshuaTTJ/DiffSNPO  
**Area**: Diffusion Models / Preference Alignment  
**Keywords**: Diffusion Model Alignment, Negative Preference Optimization, Classifier-Free Guidance, Bounded DPO, Single Network

## TL;DR
Addressing the cumbersome practice of "training two models + weight merging" for explicit negative preference modeling in diffusion alignment, this paper proposes Diff-SNPO. It utilizes the inherent conditional/unconditional branches of CFG as outlets for positive/negative preferences within a **single network**. By adapting a bounded objective from Bounded DPO, it resolves the "progressive blurring" issue of naive approaches, outperforming dual-model Diff-NPO on Pick-a-Pic v2 with half the computational cost.

## Background & Motivation
**Background**: Aligning diffusion models with human feedback has become standard. The mainstream approach is direct preference optimization like Diffusion-DPO (Diff-DPO), which avoids training explicit reward models by directly increasing the relative likelihood of "winning" samples over "losing" ones. Meanwhile, diffusion sampling relies on Classifier-Free Guidance (CFG), which amplifies the difference between "conditional prediction - unconditional prediction" to push generation towards prompt alignment and away from mediocre outputs of the unconditional distribution.

**Limitations of Prior Work**: Diff-DPO applies the same optimization objective to both conditional and unconditional branches, effectively shifting the overall distribution toward better outcomes without specifically **increasing the contrast** that CFG relies on. It lacks explicit "suppression" of undesirable outputs, as negative feedback is only implicitly reflected in relative preferences. To address this, CHATS and Diff-NPO introduced explicit negative preferences, training an additional "negative model" on **reversed labels** to replace the unconditional branch during sampling.

**Key Challenge**: Dual-model schemes involve two inevitable costs. First, maintaining two independent networks during training and sampling doubles memory, computation, and time, limiting scalability. Second, since the two models are **trained independently**, their parameter correlation is weak. Using the negative model as an unconditional branch directly can cause output mismatch. Diff-NPO employs **weight merging** (interpolating between reference, positive, and negative models), which improves image quality but biases results heavily toward the positive model, thereby **diluting the negative alignment signal**—weakening the very contrast NPO intended to strengthen.

**Goal**: Can the benefits of "explicit negative preference modeling" be achieved without the computational cost and mismatch issues of dual-model approaches?

**Key Insight**: CFG-enabled diffusion models natively possess two paths: conditional and unconditional. Instead of training a second model, these **built-in dual branches can be reused**. The conditional branch is fed positive preferences, while the unconditional (null condition) branch is fed negative (reversed) preferences, placing the contrast naturally within a single network.

**Core Idea**: Integrate positive and negative preferences into the two CFG branches of the same network to eliminate the second model and weight merging. Use a **bounded preference objective** to prevent the likelihood collapse of winning samples, making "simple, stable, and efficient" negative preference modeling feasible.

## Method

### Overall Architecture
The objective of Diff-SNPO is to perform both positive and negative preference alignment within a single diffusion network, replicating the "explicit suppression of bad outputs" capability of Diff-NPO while removing the need for dual models and weight merging. The method only modifies the training objective without changing the network architecture or sampling pipeline. The input consists of human preference pairs $(x^w_0, x^l_0, c)$ (winning image, losing image, prompt), and the output is an aligned denoising network $\epsilon_\theta$. Standard CFG sampling is used during inference.

The transition involves three steps: (1) Borrowing the CFG branch label $Y\in\{+1,-1\}$, where $Y=+1$ takes the conditional branch (effective condition $\tilde c=c$) and $Y=-1$ takes the unconditional branch ($\tilde c=\varnothing$), controlled by the CFG dropout probability $p$. (2) Applying positive preferences to the conditional branch and **reversed** preferences to the unconditional branch, forming "Naive Diff-SNPO." (3) Identifying that the naive approach causes image blurring, leading to the adaptation of Bounded DPO for diffusion (Diff-BDPO-UB) to bound the loser term in the objective, resulting in the final Diff-SNPO objective.

### Key Designs

**1. Using CFG Dual Branches instead of Dual Models: Embedding Negative Preferences into the Unconditional Branch**

Addressing the core pain point of "doubled computation + weight merging diluting negative alignment," this paper avoids training an extra negative model by utilizing the built-in conditional/unconditional branches. Formally, a branch label $Y\in\{+1,-1\}$ is introduced, with $\Pr(Y=+1)=1-p$ and $\Pr(Y=-1)=p$ (where $p$ is the CFG dropout probability). The effective condition is:

$$\tilde c(Y)=\begin{cases}c,& Y=+1\ (\text{conditional branch})\\ \varnothing,& Y=-1\ (\text{unconditional/null branch}).\end{cases}$$

The conditional branch learns positive preferences as usual, while the unconditional branch learns **reversed** preferences (swapping winners and losers). Thus, the contrast is maintained within a single set of parameters. During inference, the CFG correction term $\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t)$ naturally becomes "direction increased by positive preference - direction decreased by negative preference," eliminating the need for post-hoc weight interpolation (e.g., $\hat\theta^-=\theta_{\text{ref}}+\alpha(\theta^+-\theta_{\text{ref}})+\beta(\theta^--\theta_{\text{ref}})$) used in Diff-NPO and fundamentally resolving the mismatch issue.

**2. Diagnosis of Naive Diff-SNPO Collapse: Adversarial Gradients Lead to Decreased Winner Likelihood and Image Blurring**

Applying the Diff-DPO objective directly to the dual branches yields the naive version:

$$\mathcal L_{\text{Naive}}(\theta)=-\mathbb E\big[\log\sigma\big(Y\,T\,\omega(t)\,\beta\,(\Delta^w_t(\tilde c(Y))-\Delta^l_t(\tilde c(Y)))\big)\big],$$

where $\Delta^w_t,\Delta^l_t$ are the relative noise prediction error differences against the reference. However, longer training results in blurrier images with fewer high-frequency details. The authors attribute this to a known issue in DPO: margin improvement often comes from **decreasing both winner and loser likelihoods** (penalizing the loser more severely) rather than stably increasing the winner likelihood. Using the winner likelihood ratio $\frac{\pi_\theta(x^w)}{\pi_{\text{ref}}(x^w)}\approx \mathbb E[e^{\Delta^w_t(c)}]$, the authors found this ratio **continuously decreases** in the naive version. Worse, the shared parameters receive **symmetrical but opposite** updates (one branch increases likelihood while the other decreases it for the same sample), causing gradient conflict. The model "averages" these signals, which in generation tasks weakens contrast and erases details, leading to blurring. The solution is to **break this destructive symmetry** and bias learning towards "increasing winning probability."

**3. Adapting Bounded DPO to Diffusion: Bounding the Loser Term to Prevent Dominance**

Naive collapse occurs because the loser term grows out of control: as the model suppresses the loser probability, $\log\pi_\theta(y^l\mid x)$ becomes **disproportionately large**, dominating the loss and dragging down the winner likelihood. BDPO (Cho et al. 2025) solves this by replacing the loser term with a **mixture distribution containing a non-zero reference contribution**:

$$\pi_{\text{mix}}(y\mid x)=\lambda\,\pi_\theta(y\mid x)+(1-\lambda)\,\pi_{\text{ref}}(y\mid x),\quad \lambda\in(0,1),$$

ensuring the loser's contribution is capped by the reference, thereby **preserving the intent of promoting winners**. BDPO shares the same global optimum as DPO while providing a lower bound for winner likelihood. The authors adapt this to diffusion by defining trajectory-level rewards and an optimizable step-wise upper bound $\mathcal L_{\text{Diff-BDPO-UB}}$ via the ELBO upper bound and Jensen's inequality. The key terms are:

$$m(x_t,c)=d_\theta-d_{\text{ref}},\qquad m_{\text{mix}}(x_t,c)=-\log\!\big(\lambda e^{-d_\theta}+(1-\lambda)e^{-d_{\text{ref}}}\big)-d_{\text{ref}},$$

where $d_\theta=T\omega(t)\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2$ and $d_{\text{ref}}$ is similar (replacing $\theta$ with the reference). Intuitively, $m_{\text{mix}}$ provides a "soft cap" for the loser's error using the mixture distribution.

**4. Final Diff-SNPO Objective: Unifying Positive and Negative Preferences into a Bounded Objective**

Applying Diff-BDPO-UB to the single-model negative preference framework yields:

$$\mathcal L_{\text{SNPO}}(\theta)=-\mathbb E_{(x^w,x^l,c),t,Y}\Big[\log\sigma\big(\beta\,(m(\tilde x^w(Y),\tilde c(Y))-m_{\text{mix}}(\tilde x^l(Y),\tilde c(Y)))\big)\Big],$$

where winners and losers are swapped based on the branch label:

$$\tilde x^{w/l}(Y)=\begin{cases}x^{w/l},& Y=+1\\ x^{l/w},& Y=-1.\end{cases}$$

This objective simultaneously achieves three things: single network architecture, explicit negative preference modeling, and bounded stability. Experiments show it leads to a **steady increase** in winner log-likelihood and eliminates the blurring seen in the naive version.

### Loss & Training
Training involves minimizing the aforementioned $\mathcal L_{\text{SNPO}}$. Key hyperparameters: regularization coefficient $\beta=2000$ (SD1.5) / $5000$ (SDXL, following Diff-DPO configuration); default mixture coefficient $\lambda=0.9$; AdamW optimizer with a learning rate of $2.048\times10^{-8}$. SD1.5 was trained for 3000 steps with a batch size of 512, and SDXL for 625 steps with a batch size of 2048 using 8×A6000 GPUs. Note that $\lambda=1.0$ reverts the objective to Naive Diff-SNPO, triggering blurring collapse.

## Key Experimental Results

### Main Results
Pick-a-Pic v2, DDIM 50 steps, CFG=7.5, mean of 4 seeds (HPSv2 / PickScore / Aesthetic / ImageReward):

| Backbone | Method | HPSv2 | PickScore | Aesthetic | ImageReward |
|------|------|-------|-----------|-----------|-------------|
| SD1.5 | Baseline | 26.24 | 20.64 | 5.285 | 0.122 |
| SD1.5 | Diff-DPO | 26.55 | 21.01 | 5.382 | 0.297 |
| SD1.5 | Diff-NPO (Dual-model) | 26.92 | 21.46 | 5.538 | 0.379 |
| SD1.5 | CHATS (Dual-model) | 27.20 | 21.05 | **5.685** | 0.300 |
| SD1.5 | Diff-BDPO (Ours) | 26.64 | 21.15 | 5.446 | 0.317 |
| SD1.5 | **Diff-SNPO (Ours)** | **27.23** | **22.24** | 5.626 | **0.694** |
| SDXL | Diff-NPO | 28.30 | 22.67 | **5.945** | 0.985 |
| SDXL | CHATS | 28.25 | 22.34 | 5.879 | **1.054** |
| SDXL | **Diff-SNPO** | **28.33** | **22.69** | 5.813 | 1.010 |

On SD1.5, Diff-SNPO leads significantly in HPSv2, PickScore, and ImageReward. Its ImageReward score (0.694) is nearly double that of Diff-NPO (0.379) while using half the compute. On SDXL, it remains competitive with SOTA, though Aesthetic scores are slightly lower than DPO/Diff-NPO—authors attribute this to the fact that preferred samples in Pick-a-Pic are sometimes less aesthetic than base outputs on strong backbones like SDXL (dataset bias), rather than a flaw in the method.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Diff-SNPO ($\lambda=0.9$) | HPSv2 27.23 | Full method, stable training without blurring. |
| Naive-SNPO ($\lambda=1.0$) | Reward decreases significantly | No reference safety net → decrease in winner likelihood → blurring. |
| $\beta=1000/2000/3000$ | HPSv2 27.30/27.23/27.23 | Insensitive to $\beta$; defaults from Diff-DPO are sufficient. |

Negative Alignment Quality (Table 5, higher is better): Diff-DPO 31.86%, Diff-NPO after merging 52.34% (63.80% before merging), **Diff-SNPO 57.45%**. Diff-NPO lost over 10 points due to merging, while the single-model design preserves the signal.

Efficiency (Table 4, 8×A6000): Diff-NPO VRAM 44.2GB×2, Relative Speed 1.00×; CHATS 46.3GB, 1.86×; **Diff-SNPO 44.2GB, 2.00×** (conditional/unconditional branches are computed in parallel, whereas CHATS is serial).

### Key Findings
- Gains are not solely from BDPO stabilization: Diff-BDPO (stabilization only, no negative preference) is much weaker than Diff-SNPO, proving that **negative preference modeling** is the primary driver of performance.
- $\lambda=1.0$ (Naive version) is a critical threshold: without the reference cap, winner likelihood monotonically decreases and images blur, confirming the "adversarial gradient averaging" diagnosis.
- Diff-NPO's "weight merging" is a double-edged sword: it improves image quality but slashes negative alignment accuracy from 63.8% to 52.3%, revealing training/inference mismatch; Diff-SNPO avoids this fundamentally.

## Highlights & Insights
- **Reusing CFG branches as "off-the-shelf positive/negative models"**: This is a brilliant insight—without extra parameters, the conditional/unconditional paths of CFG provide a natural contrastive structure, effectively gaining the negative alignment position of Diff-NPO for free. This perspective can translate to any conditional generative model with CFG dropout.
- **Closed-loop of diagnosis and cure**: The authors quantitatively confirm the cause of blurring using "winner likelihood ratio decrease" and then apply the mixture distribution from BDPO to cap the loser term, resulting in a very clean logical flow.
- **"Explicit negative preferences can be simple, stable, and fast"**: While dual-model approaches have become the default for negative preferences, this work proves a single model is sufficient, saves half the compute, and allows for faster sampling—significant for practical deployment.

## Limitations & Future Work
- **Shrinking aesthetic gains on strong backbones**: On SDXL, Aesthetic scores are slightly lower than the base model, likely due to Pick-a-Pic's preferred samples not always being more beautiful; while this is a dataset bias, it suggests the method relies on preference data quality.
- **Implicit Accuracy is not a silver gold standard**: The implicit accuracy used for negative alignment can be affected by reward model artifacts; the authors note it serves to highlight training/inference mismatch rather than providing a final performance conclusion.
- **Limited to Image Diffusion (SD1.5/SDXL)**: The study does not cover video, 3D, or other diffusion scenarios; whether the branch reuse remains stable under more complex condition structures remains to be tested.
- **Upper Bound Approximation**: The final objective is an ELBO + Jensen upper bound of Diff-BDPO; the gap between the theoretical and actual optimum has not been deeply analyzed.

## Related Work & Insights
- **vs Diff-DPO**: Diff-DPO applies the **same** objective to both branches, merely shifting the distribution. This work applies **opposite** preferences to the two branches, specifically strengthening CFG contrast and adding BDPO for stability, leading to clearer improvements.
- **vs Diff-NPO**: Diff-NPO trains an independent negative model and uses weight merging, which doubles compute and dilutes negative alignment (dropping accuracy by 10+ points). This work uses a single model with no merging, achieving higher accuracy with half the compute.
- **vs CHATS**: CHATS is also dual-model and fuses signals by perturbing conditional embeddings. Its VRAM is comparable, but it is slower due to **serial** processing of branches (1.86× vs 2.00×) and prioritizes aesthetics while lagging in other metrics.
- **vs Bounded DPO (BDPO)**: BDPO was originally for LLM stabilization. This work is the first to **adapt it to diffusion trajectories** via ELBO upper bounds and embed it within a single-model negative preference framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The "CFG branch reuse" perspective is refreshing; BDPO adaptation is a logical but incremental combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across SD1.5/SDXL with 5 metrics and thorough compute/alignment ablations, though restricted to the image domain.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative flow from pain point to diagnosis to solution; equations are well-connected to motivation.
- Value: ⭐⭐⭐⭐ Achieving better negative preference alignment with half the compute is highly practical for diffusion alignment deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Better Optimization for Listwise Preference in Diffusion Models](towards_better_optimization_for_listwise_preference_in_diffusion_models.md)
- [\[ICLR 2026\] Reinforcing Diffusion Models by Direct Group Preference Optimization](reinforcing_diffusion_models_by_direct_group_preference_optimization.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)
- [\[ICLR 2026\] VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip](vsf_simple_efficient_and_effective_negative_guidance_in_few-step_image_generatio.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
