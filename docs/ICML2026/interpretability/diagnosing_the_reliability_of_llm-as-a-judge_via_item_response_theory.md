---
title: >-
  [Paper Note] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory
description: >-
  [ICML 2026][Interpretability][Item Response Theory] This paper applies the Graded Response Model (GRM) from Item Response Theory (IRT) in psychometrics to LLM-as-a-Judge. It decomposes "judgment scores" into judge attrib…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Item Response Theory"
  - "Graded Response Model"
  - "Judge Consistency"
  - "Human Alignment"
  - "Latent Quality"
date: 2026-05-08
content_hash: f4803a2268076e66
---

# Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory

**Conference**: ICML 2026  
**arXiv**: [2602.00521](https://arxiv.org/abs/2602.00521)  
**Code**: https://github.com/elu-lab/IRT-Judge  
**Area**: LLM Evaluation / LLM-as-a-Judge / Reliability Diagnosis  
**Keywords**: Item Response Theory, Graded Response Model, Judge Consistency, Human Alignment, Latent Quality  

## TL;DR
This paper applies the Graded Response Model (GRM) from Item Response Theory (IRT) in psychometrics to LLM-as-a-Judge. It decomposes "judgment scores" into judge attributes $(\alpha, \beta)$ and latent sample quality $\theta$. Four interpretable metrics are utilized in a two-stage diagnosis (intrinsic consistency + human alignment) to systematically evaluate whether seven mainstream LLMs serve as stable measurement instruments across 11 evaluation criteria.

## Background & Motivation
**Background**: LLM-as-a-Judge has become pervasive in scenarios such as summarization, dialogue evaluation, visual generation assessment, and RLHF reward modeling, providing a scalable and cost-effective alternative to human annotation. Reliability is typically verified through two paths: *intrinsic consistency* (whether the same sample receives the same score with prompt variations) and *human alignment* (consistency with human ratings).

**Limitations of Prior Work**: These two paths are usually treated separately and remain at the "output level." Consistency is measured via inter-rater agreement, multi-seed reruns, McDonald's $\omega$, or token probability uncertainty. However, these metrics fail to decouple the "measurement characteristics of the judge" from the "inherent variance in sample quality." Alignment metrics like Pearson/Spearman/Kendall correlations or Cohen's $\kappa$ only reflect result similarity without identifying whether a judge is systematically strict/lenient or lacks discriminative power.

**Key Challenge**: Existing methods conflate measurement error with inherent sample differences. Consequently, even when instability is detected, it is unclear whether it stems from prompt sensitivity, poor model discrimination, or the inherent difficulty of the evaluation dimension. A statistical framework is needed to separate the characteristics of the measurement instrument from those of the objects being measured.

**Goal**: To establish a unified framework that simultaneously addresses: (1) Is the LLM judge stable as a "measurement instrument"? (2) Is its "measurement result" aligned with humans? Furthermore, the diagnostic signals for these questions must be interpretable and factor-attributable.

**Key Insight**: The authors draw from Item Response Theory, accumulated over a century in psychometrics. Just as IRT is used in educational testing to evaluate if questions reliably measure student ability, it can evaluate if an LLM judge reliably measures sample quality. Specifically, the Graded Response Model (GRM) is selected because judgment scores are typically Likert-style ordered discrete categories.

**Core Idea**: Model "LLM Judgment Score = Judge Characteristics $(\alpha, \beta) \times$ Latent Sample Quality $\theta$" as a generative probabilistic model. Using Bayesian inference, the framework estimates all three components. Four diagnostic metrics are then extracted from the estimated $\theta$ distribution and $(\alpha, \beta)$ to attribute measurement instability to specific causes like "prompt sensitivity vs. lack of discrimination vs. systematic bias vs. range mismatch."

## Method

### Overall Architecture
The framework consists of three stages: **(1) Prompt perturbation generation**—generating three semantically invariant versions (typo / newline / paraphrase) for each original evaluation prompt, totaling four variants; **(2) GRM fitting**—feeding scores from 7 LLMs × each prompt variant × each sample into the GRM, using the NUTS sampler (4 chains, 1000 warmup + 1000 sampling, target acceptance 0.95) with PyMC + NumPyro backends to simultaneously estimate $(\alpha_p, \boldsymbol{\beta}_p)$ for each variant and a shared $\theta_j$ for each sample; **(3) Two-stage diagnosis**—Phase 1 utilizes $C_V$ and $\rho$ to judge "intrinsic consistency." Only judges that pass ( $C_V \le 0.10$ and $\rho \ge 0.70$ ) enter Phase 2, which uses $\theta_{\text{ratio}}$ and $D_W$ to compare latent quality distributions against humans.

### Key Designs

1. **GRM Decoupling "Prompt Effects" from "True Quality"**:
    - **Function**: For $K$ rating levels, the model defines the probability of judge $p$ scoring sample $j$ as $\ge k$ as $P(Y_{pj} \ge k \mid \theta_j) = \sigma(\alpha_p (\theta_j - \beta_{pk}))$, where $\theta_j$ is the latent quality of the sample (shared across prompt variants), $\alpha_p$ is the "discrimination" of the prompt (slope), and $\boldsymbol{\beta}_p$ is the sequence of "thresholds" between adjacent levels (enforced as monotonically increasing).
    - **Mechanism**: Priors are set as $\theta_j \sim \mathcal{N}(0,1)$, $\alpha_p \sim \text{LogNormal}(0, 0.5)$, and $\beta_{pk} \sim \mathcal{N}(0,1)$. NUTS samples all parameters simultaneously. Since four variants share the same $\theta_j$, score differences caused by wording sensitivity are absorbed into $(\alpha, \boldsymbol{\beta})$, while true quality variance is captured by $\theta$. This allows for direct comparison of $\theta$ distributions across models, avoiding issues where different models use different label ranges.
    - **Design Motivation**: To solve the fundamental pain point of inseparable measurement noise and signal. GRM is the standard model for ordered discrete responses; Ours avoids NRM (which ignores order) and PCM (which assumes a shared $\alpha$ across items, which is too restrictive for varying prompt sensitivities).

2. **Phase 1: Differential Diagnosis using $C_V$ + $\rho$**:
    - **Function**: $C_V$ measures "prompt consistency." For each prompt variant $p$, the mean variance of $\theta_j$ within the same score level $\bar V_p$ is calculated, and the coefficient of variation across prompts is derived as $C_V = \sigma_V / \mu_V$. $\rho$ measures "marginal reliability"—$\rho = \text{Var}(\hat\theta_j) / (\text{Var}(\hat\theta_j) + \mathbb{E}[\sigma_j^2])$, where the numerator represents true quality variance and the denominator includes expected posterior variance (measurement uncertainty).
    - **Mechanism**: Thresholds are borrowed from psychometric conventions: $C_V < 0.10$ (ensuring 75% of samples stay within $\pm 20\%$ of the mean) and $\rho > 0.70$ (Nunnally’s threshold). The combination allows for "differential diagnosis": High $C_V$ + High $\rho$ indicates prompt sensitivity; Low $\rho$ suggests insufficient model discrimination regardless of $C_V$.
    - **Design Motivation**: To decompose evaluation instability into two independent, orthogonal, and actionable dimensions. Phase 1 also acts as a "gatekeeper," preventing alignment calculations for inherently unstable judges.

3. **Phase 2: Alignment Attribution using $\theta_{\text{ratio}}$ + $D_W$**:
    - **Function**: $\theta_{\text{ratio}} = \theta_{\text{range}}^{(\text{LLM})} / \theta_{\text{range}}^{(\text{Human})}$, where $\theta_{\text{range}}$ is defined as the difference between the median $\theta$ of the highest and lowest score categories. $D_W = W_1(\hat\theta^{(\text{LLM})}, \hat\theta^{(\text{Human})})$ is the 1-Wasserstein distance between the LLM and human $\theta$ distributions.
    - **Mechanism**: Wasserstein distance is preferred over correlation or KL divergence because it captures both "location shift" and "distribution shape difference" while being well-defined for non-overlapping supports. $\theta_{\text{ratio}} < 1$ implies the LLM is "over-sensitive" (compressing the range), while $> 1$ implies the LLM is "numb" (exaggerating differences humans find negligible).
    - **Design Motivation**: Traditional metrics only answer "yes/no" regarding alignment. $(\theta_{\text{ratio}}, D_W)$ decomposes misalignment into "range mismatch" and "distribution drift," providing direct guidance for prompt and scale design.

### Loss & Training
There is no end-to-end training. The framework performs posterior inference. The GRM probabilistic model is implemented in PyMC, with Bayesian sampling conducted via NUTS (4 chains, 1000 warmup, 1000 sampling, target acceptance 0.95). For binary scores, a 2-PL logistic model replaces GRM. Prompt perturbations include character-level typos via AugLy on the top-5 attention tokens of Qwen3-8B, random insertions of three newlines, and paraphrase substitutions of five verbs/adjectives using NLTK POS tagging and GPT-4o-mini.

## Key Experimental Results

### Main Results
Evaluation of 7 models (Gemini 2.5 Flash, GPT-4o, GPT-4o-mini, Qwen3-30B-A3B, Qwen3-235B-A22B, Llama-4-Maverick, Llama-4-Scout; Qwen3-VL for vision) across 3 text benchmarks (SummEval, TopicalChat, HelpSteer-2) and 1 vision benchmark (VIEScore).

**Phase 1 Key Results** ($C_V \le 0.10$ and $\rho \ge 0.70$ as passing):

| Task / Model | $C_V$ | $\rho$ | Qualified? | Interpretation |
|------------|-------|--------|-------|------|
| SummEval Relevance / GPT-4o | 0.05 | 0.92 | ✓ | Most stable summarization judge |
| SummEval Consistency / GPT-4o-mini | 0.92 | 0.88 | ✗ | High $\rho$ but extreme $C_V$ → Prompt sensitive |
| TopicalChat Understandability / Qwen3-235B | 0.27 | 0.34 | ✗ | Very low $\rho$ → Poor discrimination |
| HelpSteer-2 Helpfulness / Gemini-2.5 | 0.03 | 0.86 | ✓ | Best performance |
| VIEScore CIG-SC / Gemini-2.5 | 1.32 | 0.94 | ✗ | Typical high $C_V$ prompt sensitivity |

**Phase 2 Key Results** ($\theta_{\text{ratio}}$, $D_W$):

| Task / Model | $\theta_{\text{ratio}}$ | $D_W$ | Interpretation |
|------------|------------------------|-------|------|
| SummEval Relevance / Gemini-2.5 | 0.96 | 0.30 | Matched range, mild drift |
| TopicalChat Understandability / GPT-4o | 2.59 | 0.33 | Severe "numbness," quality stretched 2.6× |
| VIEScore TIE-PQ / GPT-4o | 4.40 | 0.60 | Range magnified 4×, significant drift |
| HelpSteer-2 Coherence / Qwen3-30b | 1.03 | 0.16 | Rare "matched range + high alignment" |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Simple prompt → Detailed prompt | $C_V$ significantly decreases. Detailed instructions stabilize prompt consistency. |
| Detailed prompt + CoT | $C_V$ decreases further (e.g., GPT-4o Naturalness $C_V = 0.01$). |
| 3-point → 5-point scale | $\rho$ slightly increases (e.g., Naturalness 0.91 → 0.95). Increased granularity improves reliability. |
| 5-point → 7-point scale | $\rho$ may actually decrease. More points are not always better. |
| Impact on $\rho$ | Reliability $\rho$ is only marginally improved by prompts; detailed instructions primarily stabilize $C_V$. |

### Key Findings
- **No Free Lunch**: None of the 7 LLMs satisfy $C_V \le 0.10$ and $\rho \ge 0.70$ across all 11 criteria, suggesting a universally reliable LLM judge does not yet exist.
- **Vision vs. Text**: VIEScore (vision) $C_V$ values (0.16-1.32) are much higher than NLP (0.03-0.30), indicating extreme sensitivity to wording. However, vision $\rho$ (0.80-0.96) is often higher, meaning rankings are stable once the prompt is fixed.
- **Split Scaling Effects**: In NLP, larger models consistently yield lower $C_V$ and higher $\rho$. This does not hold for vision tasks, where scale does not guarantee stability.
- **Prevalent "Numbness"**: Almost all LLMs have $\theta_{\text{ratio}} > 1$, exaggerating quality differences compared to humans. This systematic over-separation is invisible to traditional correlation metrics.
- **Remedy Guidelines**: Detailed prompts and CoT address $C_V$; optimal scale selection (e.g., 5-point) addresses $\rho$.

## Highlights & Insights
- **Introducing IRT to LLM Evaluation**: This is the first systematic attempt to use GRM to diagnose LLM-as-a-Judge. IRT provides a generative probabilistic model, interpretable parameters, and established thresholds, offering advantages over simple correlation coefficients.
- **Gated Two-Stage Design**: By validating the "instrument" before measuring alignment, the framework avoids meaningless alignment scores on unstable judges—a common pitfall in existing literature.
- **Revealing Hidden Mismatches**: $\theta_{\text{ratio}}$ exposes systematic defects like "range mismatch" even when correlations look decent.
- **Generalizability**: The framework is model-agnostic and tasks-agnostic, provided the output is ordered and discrete. It provides a standardized reliability verification process for new evaluation criteria.

## Limitations & Future Work
- **GRM Assumptions**: Requires ordinal scoring and a shared latent trait $\theta$. It is inapplicable to categorical judgments (e.g., win/lose/tie without intrinsic order).
- **Computational Overhead**: NUTS sampling for large chains across many models/benchmarks requires engineering optimization to scale.
- **Limited Perturbation Scope**: Only surface-level perturbations (typo, newline, paraphrase) were tested. Structural changes (reordering rubrics) should be treated as "new instruments" for independent validation.
- **Human Baseline**: Human scores are treated as "ground truth" to solve for $\theta^{(\text{Human})}$, but inter-annotator disagreement in humans is not explicitly modeled.
- **Prescriptive Gap**: While the framework diagnoses unreliability (e.g., "low $\rho$"), it does not yet provide definitive prescriptions on whether to change the model, the prompt, or the task definition.

## Related Work & Insights
- **vs. Inter-rater Agreement / McDonald's $\omega$**: These only observe output consistency. Ours decouples prompt sensitivity from sample variance using the generative model.
- **vs. Uncertainty Quantification**: Probability-based uncertainty measures confidence for single outputs but misses the structural stability across prompts. $C_V$ aggregates this consistency.
- **vs. Correlation Coefficients / Cohen's $\kappa$**: These fail to detect systematic bias or range mismatch. $\theta_{\text{ratio}}$ and $D_W$ decompose alignment failure into specific, actionable components.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Solid interdisciplinary transfer of IRT/GRM to LLM evaluation with a complete framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Extensive coverage across NLP and Vision with clear ablation on prompts and scales.
- **Writing Quality**: ⭐⭐⭐⭐ Clear definitions and actionable diagnostic interpretations, though high terminology density for non-IRT readers.
- **Value**: ⭐⭐⭐⭐ Provides the first theoretically grounded, standardized calibration process for LLM judges with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)
- [\[ICML 2026\] Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems](prompt_optimization_is_a_coin_flip_diagnosing_when_it_helps_in_compound_ai_syste.md)
- [\[ICLR 2026\] PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions](../../ICLR2026/interpretability/posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)
- [\[AAAI 2026\] CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](../../AAAI2026/interpretability/crosscheck-bench_diagnosing_compositional_failures_in_multim.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)

</div>

<!-- RELATED:END -->
