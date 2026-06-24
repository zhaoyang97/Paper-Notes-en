---
title: >-
  [Paper Note] Bias at the End of the Score
description: >-
  [CVPR 2026][AI Safety][Reward models] This paper conducts a large-scale bias audit of five widely used reward models (PickScore, ImageReward, HPS, VQAScore, CLIP) in text-to-image (T2I) systems. It demonstrates that these scoring functions, acting as proxies for "image quality," encode systematic demographic biases. When used as noise optimizers, they disproportionately hypersexualize female subjects and "whiten" non-White subjects. Furthermore…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Reward models"
  - "demographic bias"
  - "T2I"
  - "counterfactual evaluation"
  - "hypersexualization"
date: 2026-05-08
content_hash: 41d4379f8e338d8f
---

# Bias at the End of the Score

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Magid_Bias_at_the_End_of_the_Score_CVPR_2026_paper.html)  
**Code**: None  
**Area**: AI Safety / Fairness / Text-to-Image  
**Keywords**: Reward models, demographic bias, T2I, counterfactual evaluation, hypersexualization  

## TL;DR
This paper conducts a large-scale bias audit of five widely used reward models (PickScore, ImageReward, HPS, VQAScore, CLIP) in text-to-image (T2I) systems. It demonstrates that these scoring functions, acting as proxies for "image quality," encode systematic demographic biases. When used as noise optimizers, they disproportionately hypersexualize female subjects and "whiten" non-White subjects. Furthermore, the scores themselves correlate highly with real-world demographic distributions (such as gender ratios in occupations) rather than truly measuring quality.

## Background & Motivation
**Background**: Reward Models (RM) are core components of the T2I pipeline, utilized in multiple stages: dataset filtering, evaluation metrics, supervision signals during fine-tuning, and post-generation safety/quality screening. They distill complex criteria like "alignment, fidelity, aesthetics, and human preference" into a scalar score $s_{I,p}=R(I,p)$. The community assumes by default that "higher score = better image."

**Limitations of Prior Work**: Known failure modes of RMs integrated into T2I have been studied, such as reward hacking (high scores despite ignoring the prompt), mode collapse (different initial noises collapsing into the same high-score image), and catastrophic forgetting. However, the robustness and fairness of the RM itself as a "scoring function" have rarely been systematically examined; specifically, RMs trained on human preferences like PickScore, ImageReward, and HPS have never undergone dedicated fairness/safety testing similar to T2I models.

**Key Challenge**: RM training data is inherently biased—distributional shifts in generative models and prompt datasets, human annotators' preferences, and inductive biases in model architectures and training processes all inject bias into the RM. Once synthetic images are used at scale downstream, RM biases are exponentially amplified. The **Key Insight** is that while RMs are nominally neutral measures of "image quality," they actually learn "conformity to mainstream demographic characteristics," yet this deviation has been consistently treated as a quality improvement.

**Goal**: This paper quantifies the extent to which RMs deviate from their implicit specification as "quality measures" on two levels: (1) how RMs systematically rewrite unspecified demographic attributes when used as optimizers; (2) whether race/gender can predict RM scores and if these scoring disparities mirror real-world population distributions.

**Core Idea**: The audit is split into "optimization probes + counterfactual scoring." First, the ReNO framework is used to let RMs optimize noise and observe actual distortions of demographic attributes (Part I). Then, counterfactual datasets combined with regression/ranking analysis reveal the scoring-level mechanisms behind these distortions (Part II), causally linking "behavioral bias" to "scoring bias."

## Method

### Overall Architecture
This is a **mechanistic analysis / auditing** paper. It does not propose new generation or training methods; its core is a two-stage empirical audit design. Formally, a reward model is a function $R$ that assigns a scalar $s_{I,p}=R(I,p)$ to an image-prompt pair. Five mainstream RMs (PickScore, ImageReward, HPS, VQAScore, CLIP) plus an Aesthetic score are evaluated, with an **Incompression** baseline (maximizing DCT high-frequency coefficients) as a neutral reference independent of data distribution.

The audit is divided as follows:

- **Part I (Optimization Probes)**: While freezing generator parameters, RM gradients are used to optimize initial noise vectors. Systematic distortions of demographic attributes in the optimized images—specifically hypersexualization and demographic convergence—are observed.
- **Part II (Counterfactual Scoring)**: RM scores are directly evaluated on counterfactual image sets that differ only in demographic attributes. Linear regression and ranking analysis are used to locate scoring disparities and compare them with real-world labor statistics.

### Key Designs

**1. Optimization Probe: Turning RMs from "Evaluators" into "Optimizers" to Expose Gradient Bias**

To observe how RM bias affects generated content, the most direct way is to let it "drive" the generation. This paper adopts the ReNO framework: given a frozen one-step generative model $G_\theta(\varepsilon, p)$ and reward function $R$, the optimization objective is to move only the initial noise $\varepsilon$ while keeping model parameters fixed:

$$\varepsilon^{\star}=\arg\max_{\varepsilon} R\big(G_\theta(\varepsilon, p), p\big),$$

Solved via iterative gradient ascent:

$$\varepsilon_{t+1}=\varepsilon_{t}+\eta\,\nabla_{\varepsilon_t}\Big[K(\varepsilon_t)+\lambda\,R\big(G_\theta(\varepsilon_t, p), p\big)\Big],$$

Where $\eta$ is the learning rate, $K$ is the regularization term, and $\lambda$ controls the direction and magnitude of reward optimization ($\lambda=+1$ for maximization, $\lambda=-1$ for minimization). SDXL-Turbo, PixArt-α DMD, and SD-Turbo are used as base models. The **Design Motivation** is that when a prompt is demographically underspecified, an ideal RM should be demographically neutral; any directional shift in population attributes during optimization directly exposes hidden biases in the RM gradients.

**2. Hypersexualization Metric: Dual Signals of NSFW Binary + Skin Exposure Ratio**

NSFW classifiers alone are insufficient, as many forms of sexualization do not trigger binary classifiers. This paper uses two complementary signals: (1) A pretrained NSFW classifier categorizing images into neutral/low/medium/high, collapsed into a binary indicator: $\mathrm{nsfw}(x)=0$ if $\arg\max_c p(c\mid x)=\text{neutral}$, otherwise 1; (2) Skin Exposure (SE) ratio—the proportion of skin pixels relative to the total visible human body area. Changes before and after optimization are measured:

$$\Delta_{\text{nsfw}}=\mathrm{nsfw}(x^{\star})-\mathrm{nsfw}(x_0),\qquad \Delta_{\text{skin}}=\mathrm{skin}(x^{\star})-\mathrm{skin}(x_0).$$

**3. Demographic Convergence Tracking: SeedSelect Initialization + CFD Anchors + Bidirectional $\lambda$**

To determine if optimization rewrites a subject's race/gender, one must first produce initial images of specific groups. Direct sampling is inefficient for underspecified prompts. This paper adapts SeedSelect: using a few reference images from over-specified prompts (e.g., "a photo of an Asian female doctor"), it performs gradient search in the noise space to find $\varepsilon_0$ that generates that group even with the prompt "a photo of a doctor." To lower harm, an **anchor-based** classification method is used: eight race-gender anchors are constructed using the Chicago Face Database (CFD):

$$a_{rg}=\frac{\bar e_{rg}}{\lVert \bar e_{rg}\rVert_2},\qquad \bar e_{rg}=\frac{1}{|D_{rg}|}\sum_{x\in D_{rg}} f(x),$$

Where $f$ is a frozen CLIP ViT-L/14 encoder. Bidirectional optimization with $\lambda=\pm 1$ proves that shifting toward White subjects is a structural demographic prior rather than random noise.

**4. Scoring Mechanism: Counterfactual Datasets + OLS Regression + Ranking Analysis**

To explain the root cause of behavior, the study returns to the scores. Using three counterfactual datasets (CausalFace, SocialCounterfactuals, PAIRS), images vary only in protected attributes. Two analyses are performed: (1) **Linear Regression** — fitting $s^{R}_{I,p}\approx\beta_0+\beta_1\rho_I+\beta_2\gamma_I+\beta_3(\rho_I\times\gamma_I)+\epsilon_I$ where $\rho$ is race and $\gamma$ is gender. (2) **Ranking Analysis** — capturing relative preference orders. A key finding is that gender effect sizes correlate with U.S. Bureau of Labor Statistics data on female employment shares by occupation, proving RMs reward "conformity to mainstream distributions."

## Key Experimental Results

### Part I: Optimization-Induced Distortions

| Phenomenon | Key Statistic | Note |
|------|---------|------|
| Hypersexualization (NSFW % increase, PickScore) | Females +19% vs. Males +7% (approx. 2.7×) | PickScore shows highest increase; females disproportionately affected |
| Skin Exposure Increase (PickScore) | Females approx. 2.3× higher than males | SE is also significantly higher for females |
| Max NSFW Increase (Single Pair) | 25% (PickScore + PixArt-α DMD) | Most severe combination of base model and RM |
| Non-White → White Shift ($\lambda=+1$) | ImageReward 76.1% / HPS 89.2% / CLIP 36.2% | Non-White images are frequently reclassified as White after optimization |
| Gender Flipping ($\lambda=+1$) | >39% (ImageReward), >26% (CLIP) | Optimization pushes female subjects towards male |

### Part II: Scoring-Level Bias

| Analysis | Key Findings | Note |
|------|---------|------|
| Regression (Occupation prompts) | Avg. 27.4/30 top gender gaps were in occupation prompts | Occupations trigger the strongest demographic effects |
| Correlation with Labor Stats | Higher scores for males in male-dominated jobs; females in female-dominated jobs | Scoring mirrors real-world employment ratios |
| Ranking (Race) | HPS, ImageReward rank White subjects highest, Asian lowest | Structural demographic prior persists even with negative prompts |

### Key Findings
- **Causal Chain**: Scoring bias leads to optimization drift. Because White subjects receive higher scores regardless of prompt valence, gradient ascent pushes noise toward the "White output" latent regions.
- **Model Variation**: PickScore is worst for hypersexualization, while ImageReward/HPS are strongest in racial convergence. No RM is "safe."
- **Incompression Baseline**: Confirmed that shifts stem from RM data bias rather than side effects of generator optimization.

## Highlights & Insights
- **"Evaluation-as-Optimization" is a powerful lens**: Scoring biases hidden in scalars are amplified into visible, quantifiable generative distortions when gradients drive the noise.
- **Behavioral + Mechanistic Closed Loop**: Part I provides evidence of "bad behavior," while Part II provides the "why" via scoring-level metrics, linking the two causally.
- **Conformity vs. Quality**: By correlating RM scores with labor statistics, the abstract argument that "RMs encode frequency priors" is turned into a falsifiable empirical fact.

## Limitations & Future Work
- **Imperfection of "Counterfactuals"**: The images are not strictly counterfactual; race changes often introduce non-demographic variations (earrings, backgrounds) that might confound scores.
- **Categorical Simplification**: Using discrete labels like "female" simplifies complex identities and may miss nuances of lived experiences.
- **Optimization Dependency**: Results depend on the ReNO framework; whether other noise optimization or selection techniques (e.g., Best-of-N) yield identical conclusion remains for future work.
- **OOD Risks**: Some prompts may fall outside the RM training distribution, necessitating caution in absolute effect size interpretation.

## Related Work & Insights
- **Vs. Concept2Concept**: While prior work examined dataset content (e.g., CSAM), this paper audits the behavioral bias of RMs as scoring/optimization functions.
- **Vs. Known RM Failures**: Unlike studies focusing on reward hacking or mode collapse, this work specifically targets demographic fairness, signaling it as a critical dimension of RM robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic demographic audit of T2I RMs using "evaluators as optimizers" is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 5 RMs, 3 base models, and multiple counterfactual sets.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with a well-defined causal chain.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the assumed reliability of RMs, impacting the entire T2I pipeline from filtering to evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RevINN: An End-to-End Invertible Neural Network for Reversible Adversarial Examples Generation](revinn_an_end-to-end_invertible_neural_network_for_reversible_adversarial_exampl.md)
- [\[CVPR 2026\] VisiLock: Authorizing Instruction-based Image editing with Dual Score Distillation](visilock_authorizing_instruction-based_image_editing_with_dual_score_distillatio.md)
- [\[CVPR 2026\] DSO: Direct Steering Optimization for Bias Mitigation](dso_direct_steering_optimization_for_bias_mitigation.md)
- [\[CVPR 2026\] Decoupling Bias, Aligning Distributions: Synergistic Fairness Optimization for Deepfake Detection](decoupling_bias_aligning_distributions_synergistic_fairness_optimization_for_dee.md)
- [\[CVPR 2026\] Mitigating Simplicity Bias in OOD Detection through Object Co-occurrence Analysis](mitigating_simplicity_bias_in_ood_detection_through_object_co-occurrence_analysi.md)

</div>

<!-- RELATED:END -->
