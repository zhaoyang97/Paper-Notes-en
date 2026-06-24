---
title: >-
  [Paper Note] Causal Evidence for the Primordiality of Colors in Trans-Neptunian Objects
description: >-
  [ICML 2025][Causal Inference][Causal Discovery] Using a model-agnostic causal discovery method (the FCI algorithm), this paper demonstrates with 98.7% confidence that the color of Trans-Neptunian Objects (TNOs) is the root cause of their orbital inclination distribution. This provides strong support for the "primordial" hypothesis of TNO colors—implying that color reflects the formation location rather than post-formation collisional evolution.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Causal Discovery"
  - "Trans-Neptunian Objects"
  - "FCI Algorithm"
  - "Latent Variable Causal Inference"
  - "Astronomical Data Analysis"
date: 2026-05-08
content_hash: 9ad31e88e8ac74ed
---

# Causal Evidence for the Primordiality of Colors in Trans-Neptunian Objects

**Conference**: ICML 2025  
**arXiv**: [2507.03760](https://arxiv.org/abs/2507.03760)  
**Code**: None (uses open-source causal-learn library)  
**Area**: Causal Inference  
**Keywords**: Causal Discovery, Trans-Neptunian Objects, FCI Algorithm, Latent Variable Causal Inference, Astronomical Data Analysis

## TL;DR

Using a model-agnostic causal discovery method (the FCI algorithm), this paper demonstrates with 98.7% confidence that the color of Trans-Neptunian Objects (TNOs) is the root cause of their orbital inclination distribution. This provides strong support for the "primordial" hypothesis of TNO colors—implying that color reflects the formation location rather than post-formation collisional evolution.

## Background & Motivation

### Problem Definition

Trans-Neptunian Objects (TNOs) serve as important probes of the history and evolution of the Solar System. There are significant correlations between the surface colors (spectral slopes) of TNOs and their orbital parameters (semi-major axis $a$, eccentricity $e$, inclination $i$). However, the causal direction of these correlations remains a long-standing, unsolved central question:

**Primordial Hypothesis**: The colors of TNOs reflect the compositional gradient in the protoplanetary disk and have remained unchanged since their formation (Nesvorný et al., 2020; Ali-Dib et al., 2021).

**Collisional Evolution Hypothesis**: Collisions expose fresh subsurface ice or organics, altering the albedo and spectral slope (Luu & Jewitt, 1996; Stern, 2002).

**Evaporation-Irradiation Hypothesis**: The initially diverse bulk compositions undergo selective volatile loss, followed by ultraviolet photolysis and particle irradiation that yield different surface chemistry (Brown et al., 2011; Wong & Brown, 2017).

### Key Challenge

Astronomical intervention experiments are impossible (one cannot actively manipulate celestial variables), making it necessary to infer causal relationships purely from observational data. Furthermore, it is impossible to measure all variables in the universe; latent variables are always present, which further complicates causal inference.

### Design Motivation

The authors propose: Can a purely data-driven, model-agnostic statistical causal discovery method resolve the causal direction of TNO colors without relying on any astrophysical prior assumptions?

## Method

### Overall Architecture

The methodology core of this study is a **Causal Discovery** framework, specifically employing the **FCI (Fast Causal Inference) algorithm**. The overall pipeline is as follows:

1. **Data Preparation**: Collect orbital parameters ($a, e, i$) and colors (spectral slopes) of 229 TNOs.
2. **Data Preprocessing**: Perform Gaussianization using the Yeo-Johnson transform.
3. **Conditional Independence Testing**: Execute conditional independence (CI) tests on variable pairs.
4. **Causal Graph Construction**: Output a Partial Ancestral Graph (PAG) based on the FCI algorithm.
5. **Robustness Verification**: Affirm consistency using multiple testing methods and thresholds.

### Key Designs

#### 1. Rationality of the FCI Algorithm Selection

The FCI algorithm (Spirtes et al., 1995; Zhang, 2008) is a constraint-based causal discovery algorithm, whose core advantages lie in:

- **Allows for latent variables**: Unlike methods like the PC algorithm that assume all relevant variables have been measured, FCI explicitly accounts for the impact of unobserved variables.
- **Outputs a PAG** (Partial Ancestral Graph): When the causal direction cannot be determined due to latent variables, the PAG explicitly displays this uncertainty rather than arbitrarily assigning a direction.
- **Theoretical guarantees**: Under the Causal Markov Condition and Faithfulness assumptions, FCI is proven to yield correct causal conclusions.

#### 2. Interpretation of Edge Types in PAG

Different edge types in a PAG convey different causal information:

| Edge Type | Symbol | Causal Meaning |
|-----------|--------|----------------|
| Directed edge | $X \rightarrow Y$ | $X$ is a cause of $Y$ |
| Semi-directed edge | $X \circ\!\!\rightarrow Y$ | $Y$ is not an ancestor of $X$ (i.e., $Y$ cannot directly or indirectly cause $X$) |
| Circle-circle edge | $X \circ\!\!-\!\!\circ Y$ | $X$ and $Y$ cannot be d-separated, suggesting they are causally adjacent or share a potential common cause |
| Bi-directed edge | $X \leftrightarrow Y$ | There is a latent common cause between $X$ and $Y$ |

#### 3. Conditional Independence Testing Methods

The authors adopt three independence testing methods to ensure robustness:

- **Linear Fisher-Z test** (primary method): Suitable for Gaussianized data, with threshold $\alpha = 0.013$ ($98.7\%$ confidence level).
- **Fisher-Z test without transformation**: Run directly on original data, yielding the same PAG at $\alpha = 0.02$.
- **Kernel Conditional Independence (KCI) test**: A non-linear method using potential polynomial kernels, replicating the same PAG at $\alpha = 0.09$.

#### 4. Data Preprocessing: Yeo-Johnson Transform

To satisfy the Gaussian distribution assumption of the linear Fisher-Z test, the Yeo-Johnson transform (Yeo & Johnson, 2000) is applied for Gaussianization. Compared to the Box-Cox transform, Yeo-Johnson can handle data containing zero and negative values.

### Loss & Training

This approach is a causal discovery method based on statistical testing and does not involve loss function optimization or model training. The core "training strategy" is manifested in:

1. **Threshold Selection**: $\alpha = 0.013$ guarantees that every conditional independence test passes at a $98.7\%$ confidence level.
2. **Multi-method Cross-validation**: Linear/non-linear tests, with/without data transformation, all yield consistent PAG structures.
3. **Sanity Check Design**: Verifying whether the causal model can reproduce known TNO dynamical paradigms (such as predicting the existence of Neptune) before applying it to address unknown questions.

## Key Experimental Results

### Main Results

The "experiment" in this study is the construction and validation of the causal graph, with core outcomes presented via the PAG structure:

| Test Method | Data Preprocessing | Threshold $\alpha$ | Confidence | PAG Consistency |
|-------------|-------------------|--------------------|------------|-----------------|
| Linear Fisher-Z | Yeo-Johnson Transform | 0.013 | 98.7% | Baseline PAG |
| Linear Fisher-Z | Untransformed | 0.02 | 98.0% | ✓ Consistent |
| KCI (Polynomial kernel) | Untransformed | 0.09 | 91.0% | ✓ Consistent |

**Key Causal Discoveries**:
- **Color $\rightarrow$ Inclination**: Color is the root cause of the inclination distribution, rather than inclination causing color changes.
- **Latent variable between $a$ and $e$ ($a \leftrightarrow e$)**: There is an unobserved common cause between the semi-major axis and eccentricity; the model "blindly" predicts the existence of an unknown perturber, i.e., Neptune.

### Ablation Study

| Configuration | Key Indicator | Description |
|---------------|---------------|-------------|
| Orbital parameters only ($a, e, i$) | PAG predicts latent variables | Successfully 'uncovers' the gravitational influence of Neptune |
| Incorporating color variable | Color $\rightarrow$ $i$ | Identifies color as the cause of inclination |
| Different TNO subgroups | Consistent PAG | Results remain consistent across classic, resonant, Centaur, and other subgroups |
| Removing Yeo-Johnson transform | Consistent under looser thresholds | Results do not depend on specific preprocessing |
| Non-linear test (KCI) | Consistent at $\alpha = 0.09$ | Results do not depend on linear assumptions |

### Key Findings

1. **The causal graph reproduces the astronomical consensus on TNO dynamics without physical priors**: Without any knowledge of Neptune's existence, the FCI algorithm automatically "predicts" an unobserved perturber (Neptune) via the bi-directed edge between $a$ and $e$.

2. **Color is the root cause of the inclination distribution**: Not vice-versa, which rules out the collisional evolution hypothesis (explaining color changes via late-stage collisions) with 98.7% confidence.

3. **The irradiation-induced color modification hypothesis is ruled out**: If colors were primarily determined by post-formation irradiation, one would not expect to observe the color $\rightarrow$ inclination causal relationship. Hence, the causal model excludes scenarios of major post-formation color modifications.

## Highlights & Insights

1. **Methodological Innovation**: For the first time, a rigorous causal discovery framework (FCI) is applied to central problems in planetary science, highlighting the great potential of machine learning causal inference in astronomy.

2. **"Blindly Discovering Neptune"**: As a sanity check, the causal model automatically infers the presence of an unobserved perturber without any input information about Neptune, greatly enhancing the credibility of the methodology.

3. **Model-Agnosticism**: Relying solely on the statistical properties of the data rather than orbital dynamics models or physical assumptions, it avoids the prior biases in traditional physical modeling.

4. **Multi-method Robustness**: Consistent results are yielded via three different conditional independence tests (linear/non-linear) and two data preprocessing schemes, ensuring robust conclusions.

5. **A Model of Interdisciplinary Integration**: By deeply coupling advanced methods in causal inference (the causal-learn library) with planetary science, this work offers a new research paradigm for both fields.

## Limitations & Future Work

1. **Limited Sample Size**: With only 229 TNOs, the non-linear test (KCI) is prone to overfitting, restricting the ability to discover more complex causal structures.

2. **Color as a Proxy for Formation Location**: This is a key assumption rather than a derived conclusion; if the mapping of color to formation location is non-monotonic, the interpretation of the causal conclusion might require revision.

3. **Limitations of the Linearity Assumption**: The primary method depends on the linear Fisher-Z test. Although KCI provides a complementary non-linear check, the statistical power of KCI with only 229 samples is limited.

4. **Obscuration of Temporal Evolution**: Causal graphs are static and cannot capture the temporal evolution path of TNOs from formation to the present; dynamic causal models might provide a more detailed picture.

5. **Scalable to More Variables**: Future studies can incorporate variables such as TNO size, albedo, and spectral details to build a more comprehensive causal network.

## Related Work & Insights

- **Foundations of Causal Discovery**: *Causation, Prediction, and Search* by Spirtes et al. (2001) and *Causality* by Pearl (2009) serve as the theoretical cornerstones of the methodology in this study.
- **causal-learn Library**: The Python causal discovery library developed by Zheng et al. (2024) provides high-quality implementations of algorithms including FCI.
- **Causal Inference in Astronomy**: Jin et al. (2025b) reviewed applications of causal discovery in astrophysics, and this study serves as a representative work of this emerging field.
- **Insights**: This purely data-driven causal analysis method can be extended to other astronomical studies (e.g., galaxy evolution, origins of exoplanetary atmospheric compositions) and serve as a template for causal analysis on observational data across other natural sciences.

## Rating

| Dimension | Rating (1-5) | Description |
|-----------|--------------|-------------|
| Novelty | ⭐⭐⭐⭐⭐ | First to resolve core disputes in planetary science using rigorous causal discovery |
| Methodological Rigor | ⭐⭐⭐⭐ | Backed by excellent cross-validation across multiple methods and sanity check design |
| Experimental Thoroughness | ⭐⭐⭐ | Standard evaluation is somewhat limited, constrained by a sample size of 229 |
| Writing Quality | ⭐⭐⭐⭐⭐ | This interdisciplinary paper is exceptionally clear and easy to understand |
| Value | ⭐⭐⭐⭐ | Opens new directions for the intersection of astronomy and causal inference |
| **Overall** | **⭐⭐⭐⭐** | Outstanding interdisciplinary innovation with solid methodology, though limited by sample size |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)
- [\[ICML 2025\] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)
- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](latent_variable_causal_discovery_under_selection_bias.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)
- [\[ICML 2025\] Estimating Causal Effects in Gaussian Linear SCMs with Finite Data](estimating_causal_effects_in_gaussian_linear_scms_with_finite_data.md)

</div>

<!-- RELATED:END -->
