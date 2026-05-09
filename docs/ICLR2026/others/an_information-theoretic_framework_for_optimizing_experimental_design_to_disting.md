---
title: >-
  [Paper Note] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes
description: >-
  [ICLR 2026][Information Gap] This paper proposes the **information gap**, an information-theoretic measure that quantifies the ability of a given experimental design to distinguish between likelihood coding and posterior coding hypotheses. By deriving closed-form expressions for the cross-entropy performance difference between decoders under each hypothesis—shown to be the KL divergence between the true posterior and a task-marginalized surrogate posterior—the framework enables theory-driven optimal experimental design by maximizing this measure over the stimulus prior distribution.
tags:
  - ICLR 2026
  - Information Gap
  - Probabilistic Coding Hypothesis
  - Likelihood Coding
  - Posterior Coding
  - Optimal Experimental Design
date: 2026-05-08
content_hash: 485c8a7297c9b415
---

# An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes

**Conference**: ICLR 2026
**arXiv**: [2603.01387](https://arxiv.org/abs/2603.01387)
**Code**: [https://github.com/walkerlab/information-gap-probabilistic-neural-codes](https://github.com/walkerlab/information-gap-probabilistic-neural-codes)
**Authors**: Po-Chen Kuo, Edgar Y. Walker (University of Washington)
**Area**: Computational Neuroscience — Probabilistic Neural Coding, Bayesian Perception, Experimental Design Optimization
**Keywords**: Information Gap, Probabilistic Coding Hypothesis, Likelihood Coding, Posterior Coding, Optimal Experimental Design

## TL;DR

This paper proposes the **information gap**, an information-theoretic measure that quantifies the ability of a given experimental design to distinguish between likelihood coding and posterior coding hypotheses. By deriving closed-form expressions for the cross-entropy performance difference between decoders under each hypothesis—shown to be the KL divergence between the true posterior and a task-marginalized surrogate posterior—the framework enables theory-driven optimal experimental design by maximizing this measure over the stimulus prior distribution.

## Background & Motivation

**State of the Field**: The Bayesian brain hypothesis is the dominant theoretical framework for understanding perceptual decision-making under uncertainty. Extensive psychophysical evidence suggests that the brain performs approximately Bayes-optimal computation in tasks such as multisensory integration, motion perception, and sensorimotor learning. However, the fundamental implementation question of how probability distributions are encoded in sensory neural populations remains unresolved.

**Limitations of Prior Work**: Two competing hypotheses exist: the **likelihood coding hypothesis** (exemplified by probabilistic population codes, PPC), which posits that primary sensory areas encode the likelihood function $p(x|\theta)$, and the **posterior coding hypothesis** (exemplified by neural sampling), which posits that primary sensory areas integrate prior information via feedback connections to directly encode the posterior $p(\theta|x)$. The key distinction between the two hypotheses is whether the stimulus prior $p(\theta)$ modulates neural responses in early sensory populations. Nevertheless, most existing neurophysiological experiments employ a single fixed stimulus context (uniform prior), rendering the predictions of the two hypotheses indistinguishable.

**Root Cause**: Distinguishing the two hypotheses requires manipulating stimulus prior distributions across different contexts, but the choice of prior involves a non-trivial trade-off: priors that are too dissimilar yield insufficient stimulus overlap across contexts (preventing comparison of responses to the same stimulus), while priors that are too similar produce negligible differences in the predictions of the two hypotheses. This trade-off cannot be resolved by intuition alone.

**Paper Goals**: ① How can one quantitatively measure the ability of a given experimental design (i.e., a choice of stimulus prior distributions) to distinguish between the two coding hypotheses? ② How can experimental parameters be systematically optimized to maximize discriminability?

**Starting Point**: The authors adopt a decoding framework: if a neural population encodes the likelihood function, a likelihood decoder should outperform a posterior decoder, and vice versa. By deriving the cross-entropy performance difference between optimal decoders at the theoretical limit, one can analytically quantify the discriminative power of an experimental design.

**Core Idea**: The problem of experimental design optimization is recast as maximizing the information gap—the performance difference between optimal decoders when decoding matched versus mismatched probabilistic content—providing a computable and optimizable theoretical framework for distinguishing probabilistic neural coding hypotheses from an information-theoretic perspective.

## Method

### Overall Architecture

The framework is built upon an experimental paradigm with two contexts $c \in \{A, B\}$: each context has a specific stimulus prior $p^c(\theta)$, the latent variable $\theta$ (e.g., orientation angle) is sampled from the prior and generates sensory observations $x$ via a generative model $p(x|\theta)$, and neural populations produce responses $\boldsymbol{r}$ to these observations. The core output is the **information gap** $\Delta^{\text{info}}$—the expected cross-entropy performance difference between the optimal likelihood decoder and the optimal posterior decoder under a given experimental design $(p(c), p^c(\theta))$. Closed-form expressions are derived for $\Delta_L^{\text{info}}$ under the likelihood coding hypothesis and $\Delta_P^{\text{info}}$ under the posterior coding hypothesis. The optimal experimental design is then identified by searching the task parameter space for configurations that simultaneously maximize both information gaps.

### Key Designs

1. **Information Gap for Likelihood Coding**:

    - **Function**: Quantifies the performance loss of a posterior decoder relative to a likelihood decoder when the neural population encodes the likelihood function.
    - **Mechanism**: A likelihood-coding population $\boldsymbol{r}_L \sim p(x|\theta)$ contains no prior information, so a posterior decoder cannot perfectly decode the posterior distribution. The optimal posterior decoder converges to a **task-marginalized surrogate posterior** $q_{P,i}^*(\theta) = \frac{[\sum_c p(c) p^c(\theta)] \cdot p(x_i|\theta)}{\sum_{\theta'} [\sum_c p(c) p^c(\theta')] \cdot p(x_i|\theta')}$, i.e., the true context prior is replaced by a mixture prior. The information gap $\Delta_L^{\text{info}}$ equals the expected KL divergence between the true posterior $p^c(\theta|x_i)$ and the surrogate posterior $q_{P,i}^*(\theta)$: $\Delta_L^{\text{info}} = \mathbb{E}_{p(x_i,c)}[D_{\text{KL}}(p^c(\theta|x_i) \| q_{P,i}^*(\theta))]$
    - **Design Motivation**: Every observation $x_i$ contributes a non-zero information gap as long as the two context priors differ, so the information gap under likelihood coding is generally large and the two hypotheses are relatively easy to discriminate experimentally.

2. **Information Gap for Posterior Coding**:

    - **Function**: Quantifies the performance loss of a likelihood decoder relative to a posterior decoder when the neural population encodes the posterior distribution.
    - **Mechanism**: In a posterior-coding population, different contexts may produce identical posterior distributions ($p^A(\theta|x_j) = p^B(\theta|x_k)$) corresponding to different likelihood functions ($p(x_j|\theta) \neq p(x_k|\theta)$). The optimal likelihood decoder converges to a Bayes-optimal likelihood estimate $\ell_{jk}^*(\theta)$, which requires fixed-point iteration to solve. The information gap $\Delta_P^{\text{info}}$ receives contributions only from observation pairs $(x_j, x_k)$ satisfying the posterior-matching condition.
    - **Design Motivation**: The magnitude of the information gap under posterior coding is typically an order of magnitude smaller than under likelihood coding (since only matching pairs contribute), revealing that distinguishing posterior-coding populations poses a greater experimental challenge and requires more deliberate experimental design.

3. **Task Optimization via Information Gap Landscape**:

    - **Function**: Systematically searches the task parameter space for the optimal experimental design.
    - **Mechanism**: For Gaussian context priors $p^c(\theta) = \mathcal{N}(\mu^c, \sigma^2)$, the task parameter space is defined by the prior mean separation $d = |\mu^A - \mu^B|$ and the shared standard deviation $\sigma$. The information gap under each hypothesis is computed over a grid of $(d, \sigma)$ values to obtain a two-dimensional landscape. Since $\Delta_P^{\text{info}}$ is smaller in magnitude and harder to optimize, the strategy prioritizes maximizing $\Delta_P^{\text{info}}$ while ensuring $\Delta_L^{\text{info}}$ remains sufficiently large. For example, under low-contrast stimuli, the optimal parameters are approximately $d \approx 30°$ and $\sigma \approx 20°$.
    - **Design Motivation**: The information gap landscape directly visualizes the difficulty of discriminating the two hypotheses across parameter settings, transforming experimental design from heuristic trial-and-error into a principled optimization problem. The analysis also reveals that heavy-tailed priors (t-distribution, Cauchy distribution) produce near-zero information gap under posterior coding across almost the entire parameter space, theoretically ruling out their use.

### Loss & Training

Decoders are implemented as deep neural networks (MLPs) trained with a cross-entropy loss objective. The likelihood decoder $g_L(\boldsymbol{r})$ outputs a discretized estimate of the likelihood function, and the posterior decoder $g_P(\boldsymbol{r})$ outputs a discretized estimate of the posterior distribution. Training follows standard supervised learning: $(\boldsymbol{r}, \text{target})$ pairs are generated from simulated likelihood-coding or posterior-coding populations, with targets being the discretized true likelihood or posterior distributions, respectively. The theoretical information gap serves as the reference upper bound to which the decoder performance difference is expected to converge.

## Key Experimental Results

### Main Results: Theoretical Predictions vs. Simulation Validation

The accuracy of information gap predictions is validated on Poisson neuron models and gain-modulated Poisson models across multiple task parameters and stimulus contrasts.

| Validation Dimension | Likelihood Coding $\Delta_L^{\text{info}}$ | Posterior Coding $\Delta_P^{\text{info}}$ | Key Findings |
|---|---|---|---|
| High-contrast stimuli | Theory–simulation closely matched | Theory–simulation closely matched | $\Delta_L$ exceeds $\Delta_P$ by an order of magnitude |
| Medium-contrast stimuli | Information gap increases | Information gap increases | Lower contrast amplifies prior influence |
| Low-contrast stimuli | Maximum information gap | Maximum information gap | Broadest effective parameter region |
| Gain-modulated Poisson model | Predictions accurate | Predictions accurate | Framework remains valid under more biologically realistic models |
| Convergence (trial count) | Converges at 30K trials | Converges at 30K trials | Decoder performance difference converges to theoretical value |
| Convergence (neuron count) | Converges at 500 neurons | Converges at 500 neurons | Sufficient population size is adequate |

### Real Data Validation: Allen Brain Observatory

| Dataset | Decoder Performance Difference | Theoretical Prediction | $p$-value | Conclusion |
|---|---|---|---|---|
| Allen Visual Coding (169 sessions, >300 trials each) | $0.0024 \pm 0.064$ | 0 | $p = 0.63$ | Not significant |

Under a single context (uniform prior), the theoretical information gap is predicted to be zero. The empirical result is highly consistent with this prediction, validating the necessity of multi-context prior manipulation.

### Key Findings

- **Asymmetry in information gap magnitude**: $\Delta_L^{\text{info}}$ exceeds $\Delta_P^{\text{info}}$ by up to an order of magnitude, because under likelihood coding every observation contributes a non-zero gap, whereas under posterior coding only observation pairs satisfying the posterior-matching condition contribute. This implies that distinguishing posterior-coding populations is experimentally more challenging.
- **Stimulus contrast affects discriminability**: Low-contrast stimuli (high sensory uncertainty) expand the effective parameter region, since the prior exerts greater influence on the posterior under such conditions. Experimental designs should be tailored to the specific stimulus properties (e.g., contrast).
- **Heavy-tailed priors are unsuitable for distinguishing coding hypotheses**: When Student's t-distribution or Cauchy distribution is used as the context prior, the information gap under posterior coding is effectively zero across nearly the entire parameter space. The theoretical explanation is that such distributions yield almost no observation pairs satisfying the posterior-matching condition.
- **Trade-off in optimal parameters**: The optimal parameter regions for the two hypotheses do not fully overlap, requiring a strategic choice of a "sweet spot"—prioritizing maximization of the more difficult-to-detect posterior coding information gap while ensuring the likelihood coding information gap remains sufficiently large.

## Highlights & Insights

1. **Unifying experimental design optimization within an information-theoretic framework**: The core contribution is not a new neural coding model or decoding algorithm, but the theoretical establishment of a paradigm in which "the discriminative power of an experimental design" can be analytically computed and optimized. This elevates experimental design in neuroscience from empirical trial-and-error to a theoretically grounded optimization problem.

2. **Elegant derivation of the surrogate posterior**: When a decoder is forced to extract probabilistic information from a mismatched encoding, the optimal output converges to a task-marginalized Bayes-optimal estimate (rather than random guessing). This result is both intuitively elegant and mathematically rigorous, forming the theoretical cornerstone of the entire framework.

3. **Practical insight from the magnitude asymmetry**: The finding that $\Delta_L^{\text{info}} \gg \Delta_P^{\text{info}}$ directly informs experimental design strategy—the distinguishability of posterior coding should be treated as the bottleneck when optimizing experimental parameters, rather than simply maximizing discriminability for likelihood coding.

4. **Theoretical value of negative results**: Demonstrating that heavy-tailed priors are unsuitable for distinguishing coding hypotheses, and that single-context experimental designs are uninformative, provides equally clear theoretical guidance on what should be avoided in experimental design.

## Limitations & Future Work

1. **Ideal decoder assumption**: The framework derivations are based on the theoretical limit of optimal decoders. Practical decoders (even deep networks) may underfit, causing empirically observed information gaps to fall below theoretical values. The paper mitigates this through extensive training and large datasets, but reliability under limited-data regimes is not thoroughly discussed.

2. **Dependence on prior knowledge of the generative model**: Computing the information gap requires a known generative model $p(x|\theta)$, which must be established through preliminary experiments in practice. This introduces the risk of model misspecification—if the assumed generative model deviates substantially from the true one, the optimized experimental design may be suboptimal.

3. **Binary hypothesis framing**: Although the paper discusses extensions to mixed coding hypotheses in the appendix, the core framework remains a binary likelihood-vs.-posterior dichotomy. Real neural systems may adopt intermediate strategies on a continuous spectrum, and the framework's sensitivity to such fine-grained distinctions remains to be validated.

4. **Predominantly simulation-based validation, lacking prospective real-data experiments**: The real-data analysis is used only to validate the negative conclusion that "a single context cannot distinguish the hypotheses." The core value of the framework—whether optimized experimental designs can effectively distinguish coding hypotheses in real neural recordings—still requires prospective experimental validation.

5. **Computational scalability**: Computing the information gap under posterior coding requires enumerating observation pairs satisfying the posterior-matching condition and solving fixed-point iterations, which may become computationally intractable as the dimensionality of the observation space increases.

## Related Work & Insights

| Method / Work | Core Idea | Relation to This Paper |
|---|---|---|
| PPC (Ma et al., 2006) | Poisson population codes naturally represent likelihood functions | Representative model of the likelihood coding hypothesis; one end of the framework |
| Neural Sampling (Hoyer & Hyvärinen, 2002) | Neural variability reflects sampling from the posterior | Representative model of the posterior coding hypothesis; the other end of the framework |
| Walker et al. (2020) | Decodes likelihood functions from V1 population responses to predict behavioral choices | Direct predecessor of the decoding framework used here; however, that experimental design cannot distinguish the two hypotheses |
| STRING (Lange et al., 2023) | Proposes that encoding and decoding represent two distinct perspectives on neural coding | Theoretically complementary; this paper focuses on experimental discrimination, while STRING focuses on the conceptual framework |
| Optimal stimulus design (Lewi et al., 2006/2011) | Information-theoretic optimization of electrophysiological stimuli | Methodological inspiration—extends information-theoretic optimization from single-neuron tuning curve estimation to population coding hypothesis testing |

**Directions for Inspiration**: The core idea of the information gap framework—quantifying the performance loss of mismatched decoding to distinguish encoding strategies—is transferable to representation learning analysis in machine learning. For example, analogous approaches could be used to assess whether pretrained models encode specific types of information (local features vs. global semantics).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical derivation of the information gap and the concept of the surrogate posterior are original; however, the experimental paradigm itself (multi-context prior manipulation) is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Simulation validation is comprehensive (two neuron models, multiple parameters, multiple contrast levels), but prospective validation with real multi-context experiments is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical progression from intuition to theory to experiment is exceptionally clear, with excellent figure design and consistent notation.
- **Value**: ⭐⭐⭐⭐ Provides an actionable theoretical tool for a fundamental debate in computational neuroscience, though impact is limited by the scale of the field.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[AAAI 2026\] DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design](../../AAAI2026/others/deeprwcap_neural-guided_random-walk_capacitance_solver_for_ic_design.md)
- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/others/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)

<!-- RELATED:END -->
