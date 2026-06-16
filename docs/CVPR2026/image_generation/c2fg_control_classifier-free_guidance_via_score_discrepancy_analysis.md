---
title: >-
  [Paper Note] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis
description: >-
  [CVPR 2026][Image Generation][Classifier-Free Guidance] This paper utilizes a rigorous upper bound on score discrepancy to prove that "conditional and unconditional distributions converge at an exponential rate during forward diffusion." Based on this, the fixed guidance weight $\omega$ in CFG is replaced with an exponentially decaying time-varying control function $\omega(
tags:
  - CVPR 2026
  - Image Generation
  - Classifier-Free Guidance
  - Diffusion Model
  - training-free
date: 2026-05-08
content_hash: db42c10b34bb74b2
---
# C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis

**Conference**: CVPR2026  
**arXiv**: [2603.08155](https://arxiv.org/abs/2603.08155)  
**Code**: TBD  
**Area**: Image Generation  
**Keywords**: Classifier-Free Guidance, Diffusion Models, score discrepancy, time-varying guidance weight, training-free  

## TL;DR
This paper utilizes a rigorous upper bound on score discrepancy to prove that "conditional and unconditional distributions converge at an exponential rate during forward diffusion." Based on this, the fixed guidance weight $\omega$ in CFG is replaced with an exponentially decaying time-varying control function $\omega(t)$. This training-free, plug-and-play method further improves FID/IS to SOTA levels across various frameworks including DiT, SiT, Stable Diffusion, and EDM2.

## Background & Motivation
**Background**: Classifier-Free Guidance (CFG) is the cornerstone of modern conditional diffusion models. It interpolates between conditional and unconditional predictions via $\hat{\epsilon}=\epsilon_\theta(x_t,t,\varnothing)+\omega[\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)]$, using a scalar $\omega$ to control the intensity of conditional information injection, thereby trading off fidelity and diversity.

**Limitations of Prior Work**: Original CFG keeps $\omega$ constant throughout the sampling process. Subsequent works (Interval Guidance, FDG, CFG++, $\beta$-CFG, RAAG, etc.) have recognized that a fixed weight is suboptimal and proposed various dynamic schedules. However, most are heuristic—relying on empirical observations and manual tuning of curves—lacking theoretical foundation and a principled explanation for "why it should change this way."

**Key Challenge**: These methods overlook the most fundamental aspect of CFG design—the discrepancy between the conditional distribution $p(x_t|y)$ and the unconditional distribution $p(x_t)$ naturally evolves during the diffusion process. Without understanding how this discrepancy changes over time, it is impossible to decide the appropriate amount of guidance for each step on a principled basis.

**Goal**: ① Theoretically characterize the evolution of the conditional/unconditional score discrepancy over time steps; ② Design a time-varying guidance weight strictly aligned with diffusion dynamics.

**Key Insight**: Formulation of forward diffusion as an Ornstein–Uhlenbeck process (VP-SDE / VE-SDE) allows for direct bounding of the score discrepancy between distributions starting from different initial values. The intuition is that forward noise injection causes all conditional distributions to converge toward the same Gaussian; thus, conditional information inevitably leaks as $t$ increases—the key is determining the rate and uniformity of this leakage.

**Core Idea**: Prove that the score discrepancy decays approximately exponentially as $e^{-t}$, then replace the fixed $\omega$ with an exponential control function $\omega(t)$, ensuring guidance intensity is highest where the discrepancy is largest.

## Method

### Overall Architecture
C$^2$FG modifies neither the network nor the training process; it only alters the guidance weight used at each step during sampling. The logic follows a three-stage chain: first, derive a rigorous upper bound for the discrepancy between conditional and unconditional scores in forward diffusion (Theorem 1/2), proving it decays approximately exponentially; second, use Harnack-type inequalities (Theorem 3/4) to provide complementary evidence from a probability density perspective that the "critical region near $t\to 0$ has the largest discrepancy and requires the strongest guidance"; finally, replace the fixed $\omega$ of standard CFG with an exponentially decaying control function $\omega(t)=\omega_0\exp(\lambda(1-t/t_{\max}))$, applied plug-and-play at each reverse sampling step. Since the method only replaces a weight function, it is inherently training-free and orthogonal to existing strategies like autoguidance or interval guidance.

### Key Designs

**1. Rigorous Upper Bound for Score Discrepancy: From Heuristics to Theorem**

To address the issue that "existing dynamic CFG is entirely heuristic," this paper provides a provable upper bound for score discrepancy. Letting the conditional density $\tilde p(x_t,t)=p(x_t,t|y)$ and unconditional density $p(x_t,t)$ evolve from different initial values via the same forward SDE, under VP-SDE (Theorem 1):

$$\|\nabla\log p(x,t)-\nabla\log\tilde p(x,t)\|\le \frac{\alpha(t)}{\sigma^2(t)}\,C,$$

where $\alpha(t)=\exp(-\tfrac12\int_0^t\beta_s\,\mathrm{d}s)$, $\sigma(t)=\alpha(t)\sqrt{\int_0^t \beta_s/\alpha^2(s)\,\mathrm{d}s}$, and $C$ is a constant. After time reparameterization $t'=\tfrac12\int_0^t\beta_s\,\mathrm{d}s$, the upper bound simplifies to $\frac{e^{-t'}}{1-e^{-2t'}}C\sim O(e^{-t'})$; under VE-SDE (Theorem 2), the bound is $C/\sigma^2(t)$, which also decays over time. This theorem demonstrates that longer forward noise injection leads to more complete loss of conditional information, making the two scores more similar. Conversely, in reverse sampling, this means the discrepancy is minimal at high $t$ (near pure noise) and maximal at low $t$ (near data). This explains why a fixed $\omega$ is suboptimal: it treats all steps equally, leading to over-guidance that destroys structure at high $t$ and insufficient guidance to recover the conditional manifold at low $t$. The paper validates this bound using measured score MSE and cosine similarity during reverse sampling (Figure 1).

**2. Harnack-type PDF Inequality: Perspective on the $t\to 0$ Critical Region**

The upper bound for score discrepancy diverges at $t\to 0$ (due to the $\sigma^2(t)\to 0$ denominator), making it impossible to use directly as a weight, especially since the score itself is difficult to estimate near $t=0$. To clarify what happens in this "critical region," the paper provides a Harnack-type inequality for the PDF itself. Under VP-SDE (Theorem 3), for $0<s_1<s_2$ and any $\alpha>1$:

$$p(x_1,t(s_1))\le p(x_2,t(s_2))\Big(\frac{s_2}{s_1}\Big)^{\frac{m\alpha}{2}}\exp\!\Big(\frac{\alpha^2\|x_1-x_2\|^2}{4(s_2-s_1)}+\frac{\|x_2\|^2-\|x_1\|^2}{2}\Big),$$

The form is similar for VE-SDE (Theorem 4). Fixing $x_2,s_2$ shows that smaller $s_1$ (closer to initial time) and larger distance between $x_1$ and $x_2$ result in a larger, less controllable upper bound for $p(x_1,t(s_1))$. This indicates that the magnitude and diversity of the PDF expand sharply at early stages ($t\to 0$), maximizing the difference between initial conditions. This complements the previous design: score MSE indicates that "discrepancy decays exponentially over time," while the Harnack inequality shows that "the largest and most uncontrollable discrepancy occurs at $t\to 0$"—hence requiring the strongest guidance signal for precise convergence to the target conditional distribution.

**3. Exponential Decay Control Function: Aligning Guidance with Diffusion Dynamics**

Since score discrepancy grows approximately as $e^{-t}$ during the reverse process ($t:T\to0$), an ideal guidance schedule should follow the same shape. This paper replaces the fixed $\omega$ with:

$$\omega(t)=\omega_0\exp\!\Big(\lambda\Big(1-\frac{t}{t_{\max}}\Big)\Big),$$

where $t_{\max}$ is the maximum diffusion time and $\lambda>0$ controls the growth rate. The weight grows from $\omega_0$ at $t=t_{\max}$ to $\omega_0 e^{\lambda}$ at $t=0$, capturing the theoretically proven exponential trend. During sampling, the standard CFG update is replaced by $\hat\epsilon_c^\omega(x_t)=\hat\epsilon_\varnothing(x_t)+\omega(t)[\hat\epsilon_c(x_t)-\hat\epsilon_\varnothing(x_t)]$. This implementation is consistent with theoretical/empirical exponential decay, stable due to the continuous differentiability of the exponential function, and introduces only two interpretable hyperparameters: $\omega_0$ (maximum guidance intensity, equivalent to standard CFG) and $\lambda$ (decay rate), which directly controls the fidelity-diversity trade-off. Furthermore, this framework can retroactively explain Interval Guidance: at high $t$, the score discrepancy is negligible, requiring only the conditional network; the segmented schedule of "$\omega_0>1$ inside the interval, $\omega=1$ outside" in [25] is essentially a special case of this framework. The two can be combined to apply guidance only in the most effective intervals, saving computation.

### Loss & Training
No training, no extra loss. C$^2$FG is a pure inference-time plug-and-play method that replaces the constant $\omega$ with $\omega(t)$ in the sampling loop. It requires tuning only $\omega_0$ and $\lambda$, and is directly applicable to various pre-trained diffusion weights.

## Key Experimental Results

### Main Results
ImageNet 256×256 class-conditional generation (50k samples, 250 steps), compared with fixed-weight baselines for various backbones (↓ lower is better / ↑ higher is better):

| Model / Sampler | FID↓ | IS↑ | sFID↓ | Prec↑ | Rec↑ |
|--------------|------|-----|-------|-------|-------|
| DiT-XL/2 ($\omega=1.5$, ODE) | 2.29 | 276.8 | 4.6 | 0.83 | 0.57 |
| DiT-XL/2 + Rectified Diffusion | 2.13 | / | / | 0.83 | 0.58 |
| **DiT-XL/2 + Ours** ($\omega_0{=}1,\lambda{=}\ln2$, ODE) | **2.07** | **291.5** | 4.6 | 0.83 | 0.59 |
| SiT-XL/2 (REPA) ($\omega=1.35$, SDE) | 1.80 | 284.0 | 4.5 | 0.81 | 0.61 |
| **SiT-XL/2 (REPA) + Ours** ($\omega_0{=}1,\lambda{=}1$, SDE) | **1.51** | **315.0** | 4.6 | 0.80 | 0.62 |
| SiT-XL/2 (REPA, Interval) ($\omega=1.8$, SDE) | 1.42 | 305.7 | 4.7 | 0.80 | 0.65 |
| **SiT (REPA, Interval) + Ours** ($\omega_0{=}1.8,\lambda{=}0.03$, SDE) | **1.41** | **308.0** | 4.7 | 0.80 | 0.65 |

ImageNet 512×512 (10k samples, 100 steps, SDE): DiT-XL/2 improved from FID 6.81 / IS 229.5 to **6.54 / 280.9**, with sFID 20.0→19.7, validating effectiveness at higher resolutions.

Other frameworks and datasets (Table 2):

| Setup | Metric | Baseline | + Ours |
|------|------|----------|--------|
| MS-COCO, U-ViT ($\omega=2$) | FID↓ | 5.37 | **5.28** |
| MS-COCO, Stable Diffusion 1.5 ($\omega=5$) | CLIP↑ | 31.8 | **31.9** |
| ImageNet-64, EDM2-S + autoguidance ($\omega=1.7$) | FID↓ | 1.04 | **1.03** |

Notably, the FID 1.04 of EDM2-S+autoguidance is already a near-saturated strong baseline in pixel-space diffusion; C$^2$FG still manages to reduce it to 1.03, demonstrating its plug-and-play and orthogonal nature relative to autoguidance.

### Ablation Study
Sampler robustness (SiT-XL/2 REPA, ImageNet 256×256, fewer steps, $\omega_0{=}1.7,\lambda{=}0.15$):

| Configuration | FID↓ | sFID↓ | Prec↑ | Rec↑ |
|------|------|-------|-------|-------|
| 50-step SDE baseline ($\omega=1.8$) | 3.36 | 4.5 | 0.86 | 0.54 |
| 50-step SDE + Ours | **3.20** | 4.6 | 0.86 | 0.54 |
| 50-step ODE baseline | 3.46 | 4.5 | 0.86 | 0.54 |
| 50-step ODE + Ours | **3.25** | 4.4 | 0.86 | 0.55 |
| 20-step ODE baseline | 3.29 | 4.6 | 0.85 | 0.54 |
| 20-step ODE + Ours | **3.10** | 4.5 | 0.85 | 0.54 |

### Key Findings
- **Stronger baselines reveal greater "principled" value**: On SiT-XL/2 (REPA), where improvements are difficult, SDE sampling FID dropped from 1.80 to 1.51 (IS 284→315), the largest gain across all setups. Superimposing on the already dynamic Interval Guidance yielded only a minor drop (1.42→1.41), indicating C$^2$FG captures the same underlying patterns.
- **Gains are more pronounced with fewer steps**: In 20-step ODE, FID improved 3.29→3.10, more significantly than at 50 steps, suggesting that concentrating guidance in the high-impact low $t$ region is more efficient under compute constraints.
- **Minimal impact on other metrics**: While FID/IS generally improved, sFID, Precision, and Recall remained stable or improved slightly, showing exponential scheduling does not sacrifice diversity for fidelity.

## Highlights & Insights
- **Turning heuristics into provable theorems**: The most significant "aha" moment is proving score discrepancy decays as $e^{-t}$ and then setting the weight $\omega(t)$ to follow the exact same exponential shape—the curve is derived from diffusion dynamics rather than tuning, giving the method superior interpretability.
- **Complementary theories form a closed loop**: The score MSE upper bound handles "long-term trends" (exponential decay over time), while the Harnack inequality handles the "short-term critical region" ($t\to0$ being most uncontrollable and needing strong guidance). Combined, they fully justify the exponential weight design.
- **Explaining and absorbing prior methods**: By explaining Interval Guidance as a special case when the early score discrepancy is negligible, the framework demonstrates theoretical depth and engineering utility (saving compute when combined).
- **Strong transferability**: The logic of "characterize the evolution of a variable over time/layers and then shape hyperparameters accordingly" can be transferred to any generative task involving time-varying schedules (noise schedules, temperature, step-aware LoRA, etc.).

## Limitations & Future Work
- **Gap between theory and implementation**: The score discrepancy upper bound diverges at $t\to0$. The authors "ignore this segment" and use a non-singular exponential function for extrapolation; the theorem provides the shape of the bound rather than the exact weight, so $\omega_0,\lambda$ still require empirical tuning (⚠️ noted by the authors).
- **Hyperparameters require re-tuning per task/framework**: Optimal $(\omega_0, \lambda)$ vary significantly across backbones (e.g., SD15 uses $\lambda=0.2$, while SiT-SDE uses $\lambda=1$). The method simplifies "tuning a curve" to "tuning two numbers" but is not entirely parameter-free.
- **Marginal gains on dynamic baselines**: When added to Interval Guidance, improvements are often only 0.01-0.02 FID, limiting absolute returns.
- **Evaluation focused on image class-conditional generation**: While claimed to be generalizable to audio or 3D conditional diffusion, experiments only cover ImageNet/MS-COCO images; effectiveness in other modalities remains to be verified.

## Related Work & Insights
- **vs. Fixed Weight CFG**: Standard CFG uses a constant $\omega$ throughout, treating all time steps identically; this paper proves this contradicts diffusion dynamics (over-guidance at high $t$, under-guidance at low $t$) and uses $\omega(t)$ for alignment.
- **vs. Interval Guidance [25]**: [25] restricts guidance to a specific noise interval (constant $\omega_0$ inside, 1 outside); this paper proves it is a special case of the exponential framework and can be combined to save compute.
- **vs. $\beta$-CFG / RAAG schedules**: These also use time-varying guidance but rely on heuristics or empirical distributions; this paper derives the exponential shape from SDE score discrepancy bounds.
- **vs. autoguidance [23] / CFG++ [10] / FDG [40]**: These improve guidance via frequency or data manifolds, which are orthogonal to the temporal dimension addressed here; C$^2$FG remains effective when added to autoguidance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a rigorous upper bound for conditional/unconditional score discrepancy in CFG and design principled weights accordingly; high theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple frameworks (DiT/SiT/U-ViT/SD/EDM2), samplers (SDE/ODE), and step counts; however, gains on strong baselines are small, and modalities are limited to images.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivation is clear, with a natural transition to the methodology, though the approximation from the theoretical bound to the engineering weight is slightly hurried.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, and orthogonal to existing strategies; provides "free" improvements to nearly all conditional diffusion frameworks with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)

</div>

<!-- RELATED:END -->
