---
title: >-
  [Paper Note] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling
description: >-
  [CVPR 2026][Multimodal VLM][Conformal prediction] This paper proposes CFC (Conditional Factuality Control), a post-hoc conformal framework that learns feature-conditional acceptance thresholds via augmented quantile regr…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Conformal prediction"
  - "conditional coverage"
  - "LLM hallucination control"
  - "set-valued prediction"
  - "PAC guarantees"
date: 2026-05-08
content_hash: aca9ac09fb94b01d
---

# Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling

**Conference**: CVPR 2026
**arXiv**: [2603.27403](https://arxiv.org/abs/2603.27403)  
**Code**: Available  
**Area**: Multimodal VLM / LLM Reliability
**Keywords**: Conditional conformal prediction, hallucination control, inference-time sampling, PAC certificates, set-valued output

## TL;DR

This paper proposes CFC (Conditional Factuality Control), a post-hoc conformal framework that learns feature-conditional acceptance threshold functions via augmented quantile regression, providing conditional coverage guarantees (rather than merely marginal guarantees) for LLM sampled outputs. The authors further derive a PAC-style finite-sample certificate CFC-PAC, and validate the approach on synthetic data, reasoning/QA benchmarks, and VLM settings.

## Background & Motivation

LLMs exhibit strong capabilities on reasoning and generation tasks, yet hallucination renders their outputs unreliable. Inference-time multi-sample generation with reranking can improve accuracy, but lacks **formal reliability guarantees**. Conformal Prediction (CP) is a natural candidate—model-agnostic and distribution-free, it constructs set-valued predictions containing the correct answer under an exchangeability assumption.

**Core Problem: Heterogeneity of Marginal Guarantees**

Existing CP methods applied to LLMs rely on a **single global threshold**, providing only marginal coverage guarantees (holding on average across all prompts). This leads to:

**Under-coverage on hard prompts**: Difficult prompts such as long math problems and rare entities are systematically under-covered, with reliability left unguaranteed.

**Over-coverage on easy prompts**: Simple prompts receive unnecessarily excessive coverage, causing prediction sets to inflate and wasting computation.

**Compromise of the global threshold**: A single threshold must trade off between easy and hard regions of the feature space, resulting in subgroup calibration bias and poor sample efficiency.

**Motivation**: **Conditional coverage** is needed—guarantees that hold not only on average, but also conditionally on prompt features. Conditional coverage is strictly stronger than marginal coverage and directly addresses reliability for systematically difficult subgroups. At the same time, prediction sets should remain as compact as possible to preserve the computational efficiency of sampling-based inference.

## Method

### Overall Architecture

CFC is a post-processing layer placed on top of any LLM sampler. The pipeline is as follows:
1. Given a test prompt $X$, sample $M$ candidates $C(X) = \{Y_j\}_{j=1}^M$ from a base generator $\pi$
2. Score each candidate with a verifier $V(X, y) \in [0,1]$ (lower is better)
3. Learn a feature-conditional acceptance threshold $\hat{\lambda}_\alpha(X)$
4. Return the prediction set $\hat{C}_\alpha(X) = \{y \in C(X) : V(X,y) \leq \hat{\lambda}_\alpha(X)\}$

The core innovation is learning $\hat{\lambda}_\alpha(X)$ via **augmented quantile regression** rather than using a global threshold.

### Key Designs

1. **Latent Success Score**: Defined as the best verifier score among correct answers in the candidate set:

    $S(X) := \inf\{V(X,y) : y \in C(X),\; A(X,y) = 1\}$

   The prediction set contains at least one correct answer if and only if $S(X) \leq \lambda(X)$. CFC aims to learn a feature-conditional $\lambda(\cdot)$ such that $S(X) \leq \lambda(X)$ holds with high probability.

2. **Augmented Quantile Regression**: Building on the function-class conditional conformal framework of Gibbs et al., for a candidate score $s \in [0,1]$, the following is solved:

    $\beta_s = \arg\min_{\beta \in \mathbb{R}^d} \left[\frac{1}{N+1}\sum_{i=1}^N \rho_{1-\alpha}(S_i - \Phi(X_i)^\top \beta) + \frac{1}{N+1}\rho_{1-\alpha}(s - \Phi(X_{N+1})^\top \beta)\right]$

   where $\rho_{1-\alpha}$ is the pinball loss and $\Phi(X)$ is a feature map. The key step is to take the **largest fixed point** of the mapping $g_X(s) = \Phi(X)^\top \beta_s$ as the deployment threshold:

    $\hat{\lambda}_\alpha(X) = \sup\{s \in [0,1] : s \leq g_X(s)\}$

   This makes the threshold adaptive to prompt features (difficulty): hard prompts receive a more lenient threshold (allowing more candidates to pass), while easy prompts receive a stricter threshold.

3. **CFC-PAC High-Probability Certificate**: CFC's conditional coverage is an expectation-level guarantee. CFC-PAC further provides a PAC-style finite-sample certificate: Ridge regularization $\frac{\lambda}{2}\|\beta\|_2^2$ is added and the nominal risk level is shrunk:

    $\alpha_{\text{eff}} = \alpha - \varepsilon_N(\delta), \quad \varepsilon_N(\delta) = O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right)$

   With probability at least $1-\delta$, the deployed rule achieves coverage of at least $1-\alpha$.

4. **Efficiency Analysis**: Under mild assumptions (monotonicity and concavity of the score distribution), it is proved that the expected prediction set size of the oracle conditional rule is strictly smaller than that of marginal CP:

    $\mathbb{E}[G_X(\lambda^*(X))] \leq \mathbb{E}[G_X(\bar{\lambda}_\alpha)]$

   with the inequality being strict when $\mathbb{P}(q_\alpha(X) \neq \bar{\lambda}_\alpha) > 0$. CFC asymptotically inherits this efficiency when the quantile regression is consistent (Theorem 4.4).

### Loss & Training

- CFC is a purely post-hoc method that **does not fine-tune the base model**
- Only augmented quantile regression (pinball loss) is fitted on the calibration set
- At deployment, only a fixed-point threshold computation is required, with negligible computational overhead

## Key Experimental Results

### Main Results

Synthetic data ($\alpha = 0.10$, $N_{\text{cal}} = 10000$):

| Method | ECR | APSS↓ | GSC↑ | Note |
|--------|-----|-------|------|------|
| TopK | 90.6 | 16.00 | 58.2 | Fixed-size set |
| ICP (standard conformal) | 90.2 | 16.71 | 57.4 | Single global threshold |
| Learnt CP | 90.2 | 15.72 | 84.3 | Learned threshold without conformal correction |
| **CFC** | **90.3** | **15.53** | **88.7** | Conditional conformal |
| **CFC-PAC** | 90.8 | 15.87 | **89.1** | +PAC high-probability certificate |

CFC substantially improves worst-group coverage (GSC) from 57.4% (ICP) to 88.7%, while producing smaller prediction sets (15.53 vs. 16.71).

TriviaQA ($\alpha = 0.25$):

| Method | ECR | GSC↑ | APSS↓ |
|--------|-----|------|-------|
| TopK | 73.4 | 55.9 | 1.00 |
| ICP | 74.9 | 56.7 | 1.08 |
| Learnt CP | 74.7 | 74.0 | 1.22 |
| **CFC** | **72.7** | **65.2** | **1.03** |

### Ablation Study

| Configuration | GSC↑ | APSS↓ | Note |
|---------------|------|-------|------|
| ICP (global threshold) | 57.4 | 16.71 | Baseline: marginal guarantee |
| Learnt CP (learned threshold) | 84.3 | 15.72 | Learning helps but is insufficient |
| CFC (conditional conformal) | 88.7 | 15.53 | Conformal correction yields further gains |
| CFC-PAC | 89.1 | 15.87 | High-probability certificate; slightly larger sets |

### Key Findings

- **Learning better score thresholds alone is insufficient**: Learnt CP already substantially outperforms ICP (GSC 84.3 vs. 57.4), but the absence of conformal correction still falls short of CFC's subgroup reliability (88.7)
- **Visualization of conditional thresholds** confirms the intuition: easy prompts receive strict thresholds and hard prompts receive lenient ones—precisely the mechanism that corrects systematic under-coverage of difficult inputs by global thresholds
- Conditional rules empirically reduce average prediction set size, validating the efficiency theorem
- CFC transfers directly to VLMs (Flickr8k) without retraining the base model

## Highlights & Insights

- **Conditional conformal prediction for LLM hallucination control** is a natural and valuable research direction
- **Theoretically rigorous**: conditional coverage theorem (Thm 4.1) + PAC certificate (Thm 4.2) + efficiency analysis (Prop 4.3, Thm 4.4) form a cohesive theoretical contribution
- **Practical significance**: For safety-critical applications (medical QA, legal reasoning, etc.), conditional coverage is more meaningful than marginal coverage—systematic under-coverage of hard questions is unacceptable
- The fully post-hoc, model-agnostic design makes the framework broadly applicable without modifying any base model

## Limitations & Future Work

- The choice of feature map $\Phi(X)$ has a large impact on performance, yet no automated selection mechanism is provided
- The linear assumption in quantile regression may be limiting in high-dimensional feature spaces
- Experiments are relatively small-scale (TriviaQA + GSM8K + Flickr8k); scalability to large-scale LLM settings remains to be verified
- The PAC convergence rate $O(\sqrt{\log(1/\delta)/N})$ may be loose when the calibration set is small
- An external verifier is required for scoring, and verifier quality itself becomes a bottleneck

## Related Work & Insights

- Builds on the function-class conditional conformal framework of Gibbs et al., instantiating it for the LLM sampling setting
- Compared to existing LLM CP methods such as conformal factuality and TopK, the core contribution is conditionalization
- The efficiency analysis makes an independent theoretical contribution to the CP literature
- Transfer experiments to the VLM setting provide a new perspective on multimodal reliability

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of conditional conformal prediction and LLM hallucination control is novel, though the theoretical framework is largely inherited from Gibbs et al.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic + real + VLM multi-setting validation, though the scale is relatively small
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory is presented clearly and rigorously; the logical chain from motivation to method to experiments is highly coherent
- **Value**: ⭐⭐⭐⭐ — Offers direct theoretical and practical value for safe LLM deployment, though large-scale validation is still needed

---

# Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling

**Conference**: CVPR 2026
**arXiv**: [2603.27403](https://arxiv.org/abs/2603.27403)  
**Code**: [GitHub](https://github.com/) (mentioned in paper)  
**Area**: Multimodal VLM / LLM Reliability
**Keywords**: Conformal prediction, conditional coverage, LLM hallucination control, set-valued prediction, PAC guarantees

## TL;DR

This paper proposes CFC (Conditional Factuality Control), a post-hoc conformal framework that learns feature-conditional acceptance thresholds via augmented quantile regression, providing conditional coverage guarantees for LLM/VLM sampled outputs while maintaining compact prediction sets and significantly improving reliability on hard-prompt subgroups.

## Background & Motivation

Large language models have achieved remarkable progress on reasoning and generation tasks, yet hallucination remains the primary obstacle to reliability. Existing uncertainty control methods face a core problem:

1. **Marginal guarantees from CP are insufficient**: Standard CP uses a single global threshold, providing only marginal coverage—average coverage across all prompts may be met, but **hard prompts may be systematically under-covered while easy prompts are over-covered**
2. **Heterogeneity is masked**: Coverage on difficult prompts (e.g., long math problems, rare entities) may fall far below the target, while coverage on easy prompts is unnecessarily high, causing prediction set inflation
3. **Conditional coverage is what is actually needed**: Safety-critical applications require coverage guarantees to hold not only on average, but also within specific feature subgroups

The motivation for CFC is to replace the global threshold with a **feature-conditional threshold**, allowing the acceptance criterion to adapt to prompt difficulty—using a more lenient threshold for hard prompts and a stricter one for easy prompts.

## Method

### Overall Architecture

CFC is a purely post-hoc layer that requires no fine-tuning of the base generation model. The pipeline is:
1. Given prompt $X$, sample $M$ candidate answers from the base generator
2. Score each candidate with a verifier $V(X, y) \in [0,1]$ (lower is better)
3. Define the latent success score $S(X) = \inf\{V(X,y) : y \in C(X), A(X,y)=1\}$
4. Learn a conditional threshold $\hat{\lambda}_\alpha(X)$ via augmented quantile regression
5. Accept all candidates with scores below the threshold: $\hat{C}_\alpha(X) = \{y : V(X,y) \leq \hat{\lambda}_\alpha(X)\}$

### Key Designs

1. **Augmented Quantile Regression**:
    - Builds on the function-class conditional conformal framework of Gibbs et al.
    - For a candidate test score $s \in [0,1]$, the following parameter optimization is defined: $\beta_s = \arg\min_\beta [\frac{1}{N+1}\sum_{i=1}^N \rho_{1-\alpha}(S_i - \Phi(X_i)^\top\beta) + \frac{1}{N+1}\rho_{1-\alpha}(s - \Phi(X_{N+1})^\top\beta)]$
    - where $\rho_{1-\alpha}(u) = u(1-\alpha - \mathbb{1}\{u<0\})$ is the pinball loss
    - The deployment threshold is determined via the largest fixed point: $\hat{\lambda}_\alpha(X) = \sup\{s \in [0,1] : s \leq g_X(s)\}$
    - **Design Motivation**: The feature map $\Phi(X)$ allows the threshold to vary with prompt characteristics, automatically relaxing for hard prompts and tightening for easy ones

2. **CFC-PAC Variant**:
    - Adds Ridge regularization $\frac{\lambda}{2}\|\beta\|_2^2$ to the augmented quantile regression
    - Shrinks the nominal risk level: $\alpha_{eff} = \alpha - \varepsilon_N(\delta)$
    - Provides a finite-sample PAC certificate: with probability at least $1-\delta$, the deployed rule achieves coverage $\geq 1-\alpha$
    - The slack $\varepsilon_N(\delta) = O(\sqrt{\log(1/\delta)/N})$ shrinks as the calibration set grows
    - **Design Motivation**: Upgrades from an expectation-level guarantee to a high-probability guarantee, suitable for safety-critical scenarios

3. **Efficiency Analysis**:
    - Under mild assumptions (monotonicity and concavity of the score distribution), it is proved that the conditional rule's **expected prediction set size is strictly smaller** than that of the marginal CP rule
    - Core intuition: the conditional rule achieves tighter thresholds (accepting fewer candidates) for easy prompts and looser thresholds (preserving coverage) for hard prompts; Jensen's inequality guarantees overall efficiency
    - CFC asymptotically inherits oracle efficiency as quantile regression becomes consistent

### Loss & Training

- The core optimization objective is the pinball loss (quantile regression loss); no neural network training is involved
- Feature map $\Phi(X)$ design: GSM8K uses a quadratic basis $[1, T(X), T(X)^2]$ (where $T(X)$ is the mean verifier loss); TriviaQA uses a calibration-defined feature map based on answer distribution entropy and verifier loss
- Purely post-hoc; no model fine-tuning

## Key Experimental Results

### Main Results

**Synthetic data ($\alpha=0.10$):**

| Method | ECR | APSS↓ | GSC↑ |
|--------|-----|-------|------|
| TopK | 90.6 | 16.00 | 58.2 |
| ICP | 90.2 | 16.71 | 57.4 |
| Learnt CP | 90.2 | 15.72 | 84.3 |
| **CFC** | **90.3** | **15.53** | **88.7** |
| **CFC-PAC** | **90.8** | **15.87** | **89.1** |

**GSM8K ($\alpha=0.05$):**

| Method | ECR | APSS↓ | GSC↑ |
|--------|-----|-------|------|
| ICP | 95.09 | 4.73 | 79.85 |
| CFC | 94.82 | **2.35** | **88.48** |
| CFC-PAC | 95.24 | 4.59 | **88.79** |

**Flickr8k VLM ($\alpha=0.03$):**

| Method | ECR | APSS↓ | GSC↑ |
|--------|-----|-------|------|
| ICP | 95.58 | 1.84 | 85.21 |
| **CFC-PAC** | **97.27** | 1.42 | **95.21** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Learnt CP only (no conformal correction) | GSC 84.3 | Learning a good threshold helps but is insufficient |
| CFC + conformal correction | GSC 88.7 | Conformal correction further improves subgroup reliability |
| Entropy-linear Φ | GSC 45.1 (CFC) | Feature map choice has significant impact |
| Chosen Φ | GSC 62.8 (CFC) | A well-designed feature map is critical |
| N=5 vs. N=20 sampling | APSS 2.35 vs. 7.97 | Larger sampling budget inflates prediction sets with limited GSC gain |

### Key Findings

1. **Conditional thresholds effectively flatten subgroup coverage**: Across all datasets, CFC substantially alleviates under-coverage of the hardest subgroups
2. **Efficiency advantage**: CFC produces smaller prediction sets while maintaining coverage (APSS reduced from 4.73 to 2.35 on GSM8K)
3. **Transfer to VLMs**: The same post-hoc layer is applied directly to Qwen2-VL-7B-Instruct without modification
4. **Feature map design matters**: A well-chosen difficulty proxy (e.g., mean verifier loss) has a large effect on performance
5. **PAC variant is more conservative but more reliable**: CFC-PAC achieves coverage closer to the target at the cost of slightly larger prediction sets

## Highlights & Insights

- **Unified theory and practice**: Conditional coverage guarantees, PAC certificates, and efficiency analysis are integrated, with rigorous theory and thorough empirical validation
- **Minimalist design**: Post-hoc, training-free, and model-agnostic—directly applicable to any LLM/VLM sampling pipeline
- **Elegance of the efficiency analysis**: The convexity argument via Jensen's inequality proves that conditional rules are more efficient than marginal rules
- **Strong practical utility**: Five candidates and a simple quadratic feature map suffice to yield substantial improvements
- **Division of labor between CFC and CFC-PAC**: CFC minimizes set size; CFC-PAC most closely achieves the target coverage rate—users may select according to their needs

## Limitations & Future Work

1. **Feature maps require manual design**: The choice of $\Phi(X)$ depends on domain knowledge (e.g., using verifier loss as a difficulty proxy); automated feature selection is worth exploring
2. **Exchangeability assumption**: The calibration and test sets must satisfy exchangeability; the framework may fail under covariate shift
3. **Dependence on an external verifier**: Verifier quality directly affects CFC performance, yet the paper does not address uncertainty in the verifier itself
4. **Convergence rate of quantile regression in high-dimensional feature spaces**: The paper uses low-dimensional features (1–3 dimensions); sample complexity for high-dimensional feature maps remains unanalyzed
5. **Validated only on mid-sized LLMs**: Experiments use Llama-3-8B and Qwen2-VL-7B; behavior on larger models is unknown

## Related Work & Insights

- Builds on the function-class conditional conformal framework of Gibbs et al. (2023), adapting it to the LLM sampling setting
- Compatible with Best-of-N decoding and pass@N evaluation paradigms; CFC can be viewed as a reliability enhancement for these strategies
- The discussion of conditional vs. marginal coverage is relevant to all AI systems requiring uncertainty quantification
- PAC-Bayes-style finite-sample guarantees provide a tool for compliance auditing at deployment

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Adapting conditional conformal prediction to the LLM setting is innovative, though not an entirely new paradigm
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three-tier validation (synthetic + real + VLM) with comprehensive ablations
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear, figures are intuitive, and the logical flow from motivation to method to experiments is smooth
- **Value**: ⭐⭐⭐⭐ — Provides a practical, theoretically grounded tool for reliable LLM deployment, though large-scale validation remains needed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/multimodal_vlm/conditional_diffusion_sampling.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](pop_proof_of_perception_conformal_reasoning.md)
- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](towards_multimodal_domain_generalization_with_few_labels.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)

</div>

<!-- RELATED:END -->
