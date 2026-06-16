---
title: >-
  [Paper Note] Bias at the End of the Score: Demographic Biases in Reward Models for T2I
description: >-
  [CVPR 2026][Image Generation][Text-to-Image] Large-scale demographic bias audit of widely used reward models (PickScore, ImageReward, HPS, etc.) in text-to-image generation reveals that reward-guided optimization disproportionately sexualizes female images, causes demographic convergence toward White individuals, and reward scores correlate with real-world demogr
tags:
  - CVPR 2026
  - Image Generation
  - Text-to-Image
date: 2026-05-08
content_hash: 6b6db04d21fa50bb
---
# Bias at the End of the Score: Demographic Biases in Reward Models for T2I

**Conference**: CVPR 2026  
**arXiv**: [2604.13305](https://arxiv.org/abs/2604.13305)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Reward Models, Text-to-Image, Demographic Bias, Hypersexualization, Fairness

## TL;DR

Large-scale demographic bias audit of widely used reward models (PickScore, ImageReward, HPS, etc.) in text-to-image generation reveals that reward-guided optimization disproportionately sexualizes female images, causes demographic convergence toward White individuals, and reward scores correlate with real-world demographic frequency priors.

## Background & Motivation

**Background**: Reward Models (RM) are ubiquitous in T2I pipelines—functioning as dataset filters, evaluation metrics, supervision signals for parameter optimization, and post-generation filters. Models such as PickScore, ImageReward, and HPS are trained on human preference data.

**Limitations of Prior Work**: RMs are designed and deployed as "quality metrics," but their robustness and fairness regarding demographic bias remain largely unstudied. Data used for RM training, human preferences, and model inductive biases may all inject bias.

**Key Challenge**: RMs are widely used as proxies for "quality," but they may implicitly encode demographic biases, causing these biases to be amplified exponentially through various stages of the T2I pipeline.

**Goal**: Systematically audit the demographic bias behavior of RMs during fine-tuning and evaluation.

**Key Insight**: Utilize the ReNO framework for reward-guided optimization to observe demographic changes in images before and after optimization; use counterfactual datasets to analyze bias at the scoring level.

**Core Idea**: RMs do not merely evaluate image quality; they implicitly reward images that conform to dominant demographic characteristics present in their training data.

## Method

### Overall Architecture

This paper does not propose a new model but performs a "physical exam" on reward models (RM) typically treated as "neutral quality rulers." It investigates what kinds of people these RMs quietly reward when guiding T2I generation. The analysis follows two paths. The first is the **optimization-side** (Part I): RMs are treated as differentiable optimization objectives to observe whether the gradient ascent process systematically alters the race, gender, and sexualization of individuals in images. If a ruler is neutral, optimizing for high scores should not turn Black individuals White or make women more scantily clad. The second is the **scoring-side** (Part II): RMs are directly fed "counterfactual" images—identical in all aspects except demographic attributes—to determine if the scores themselves vary systematically based on skin color or gender. Furthermore, this "score bias" is correlated with real-world demographic frequency priors to identify the source of the bias.

### Key Designs

**1. Reward-guided optimization experiments (Part I): Forcing the RM to reveal its preferences**

Since RMs cannot be directly questioned for fairness, the authors utilize the ReNO framework to push RMs to the extreme. Given a fixed generator $G_\theta$, the initial noise vector is optimized to maximize the reward:

$$\varepsilon^* = \arg\max_\varepsilon\; R\big(G_\theta(\varepsilon, p), p\big),$$

The changes in the same image before and after optimization are then compared. The metrics are not aesthetic scores but demographic signals: NSFW classification rates, skin exposure area, and demographic classifier outputs. Prompts are divided into "with demographic identifiers" and "without" to distinguish whether bias is explicit in the prompt or hallucinated by the RM. The logic is rigorous: a truly neutral ruler should not change an individual's race or gender, nor should it unilaterally increase sexualized content for women when climbing toward higher scores.

**2. Counterfactual scoring analysis (Part II): Isolating "scoring bias" from confounding factors**

Optimization experiments show amplified bias, but this could originate from the generator rather than the RM. To attribute the bias to the RM, the authors construct "paired images" using three counterfactual datasets (CausalFace, SocialCounterfactuals, PAIRS) where everything (pose, composition, lighting) remains consistent except race $\rho_I$ and gender $\gamma_I$. OLS regression is performed on the RM scores $s^R_{I,p}$:

$$s^R_{I,p} \approx \beta_0 + \beta_1 \rho_I + \beta_2 \gamma_I + \beta_3(\rho_I \times \gamma_I) + \epsilon_I,$$

Statistical significance in $\beta_1, \beta_2$, or the interaction term $\beta_3$ indicates that the RM assigns different scores based solely on demographic attributes. Additionally, a ranking analysis identifies relative preferences for different skin tones. The value of the counterfactual design lies in holding "image quality" constant, ensuring the remaining score variance is attributable to demographic attributes.

**3. Real-world frequency correlation analysis: Testing if RMs mistake "population distribution" for "quality"**

This design investigates the source of the bias. The authors correlate RM scores for various occupation prompts with actual female employment ratios reported by the U.S. Bureau of Labor Statistics. The reasoning is direct: if an RM evaluates pure image quality, its scores should not correlate with social statistics like "which occupations have more women." A significant correlation suggests the RM is rewarding images that match dominant demographic distributions in its training data, effectively treating population frequency as a proxy for quality.

### Loss & Training

This is an audit/analysis paper and does not train new models. Optimization experiments follow the default hyperparameters of ReNO. To allow for cross-comparison between RMs with different scales like PickScore, ImageReward, and HPS, all scores are normalized to zero mean and unit variance before regression and ranking.

## Key Experimental Results

### Main Results

| Finding | RM | Effect Size |
|------|-----|-------|
| Hypersexualization Amplification | PickScore | Female NSFW rate increased 19% vs Male 7% (2.7×) |
| Demographic Convergence | ImageReward/HPS | >80% of Black images classified as White after optimization |
| Gender Flipping | ImageReward | 39% of female images classified as male after optimization |
| Racial Scoring Bias | HPS/ImageReward | White images systematically receive the highest scores |
| VQAScore Inversion | VQAScore | Positive prompts prefer White; negative prompts prefer Black |

### Ablation Study

| RM | White Rank | Black Rank | Gap |
|----|---------|---------|------|
| HPS | 1.2 | 3.1 | Largest Bias |
| ImageReward | 1.4 | 2.8 | Significant Bias |
| CLIP | 2.5 | 3.5 | Black consistently lowest |
| PickScore | 1.8 | 2.3 | Moderate Bias |

### Key Findings

- PickScore exhibits the strongest hypersexualization effect: females are affected 2.7 times more than males.
- ImageReward and HPS cause the most severe demographic convergence: over 80% of Black images are classified as White post-optimization.
- RM scores significantly correlate with US occupational gender ratios, indicating RMs have learned real-world frequency priors.
- VQAScore demonstrates a "stereotype reinforcement" pattern: preferring White individuals for positive descriptions and Black individuals for negative ones.

## Highlights & Insights

- This is the most systematic fairness audit of T2I reward models to date, revealing that RMs are far from neutral quality metrics.
- The discovery of "demographic convergence" (where optimization drives diverse images toward White representation) is critical, showing that RMs can act as barriers to diversity.
- The conclusion that RMs encode "dominant demographic alignment" rather than "quality" has profound implications for the design and deployment of future reward models.

## Limitations & Future Work

- Only the ReNO optimization method was utilized; other optimization strategies may exhibit different behaviors.
- Dependence on automated classifiers for demographic attributes introduces measurement noise.
- The sources of bias (training data vs. annotator preference vs. architecture) require deeper analysis.
- There is a need to develop training methods for debiased RMs.

## Related Work & Insights

- **vs Concept2Concept**: C2C identified CSAM in the Pick-a-Pic dataset; this work focuses on systematic demographic biases within RMs.
- **vs T2I Fairness Studies**: Previous research focused on biases in the generative models themselves; this work reveals that RMs, as evaluation and optimization tools, carry equally severe biases.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic audit of demographic bias in T2I RMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 RMs × 3 counterfactual datasets × multiple analysis methods.
- Writing Quality: ⭐⭐⭐⭐ Findings are clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Significant warning for AI safety and fairness in the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[CVPR 2026\] SpeeDiff: Scalable Pixel-Anchored End-to-End Latent Diffusion Model](speediff_scalable_pixel-anchored_end-to-end_latent_diffusion_model.md)
- [\[CVPR 2026\] FailureAtlas: Mapping the Failure Landscape of T2I Models via Active Exploration](failureatlas_mapping_the_failure_landscape_of_t2i_models_via_active_exploration.md)

</div>

<!-- RELATED:END -->
