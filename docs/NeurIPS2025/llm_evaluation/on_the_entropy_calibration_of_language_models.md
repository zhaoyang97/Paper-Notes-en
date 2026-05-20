---
title: >-
  [Paper Note] On the Entropy Calibration of Language Models
description: >-
  [NeurIPS 2025][LLM Evaluation][entropy calibration] This paper systematically investigates the entropy calibration of language models — whether the entropy of generated text matches the log loss on human text — and finds…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "entropy calibration"
  - "error accumulation"
  - "scaling laws"
  - "distribution truncation"
  - "diversity-quality tradeoff"
date: 2026-05-08
content_hash: caf746c33591f7bc
---

# On the Entropy Calibration of Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.11966](https://arxiv.org/abs/2511.11966)  
**Code**: [GitHub](https://github.com/stevenxcao/entropy-calibration)  
**Area**: LLM Evaluation
**Keywords**: entropy calibration, error accumulation, scaling laws, distribution truncation, diversity-quality tradeoff

## TL;DR
This paper systematically investigates the entropy calibration of language models — whether the entropy of generated text matches the log loss on human text — and finds that due to the power-law nature of data distributions ($\alpha \approx 1$), error accumulation improves extremely slowly with model scale (scaling exponent $\approx -0.05$). The paper further provides a theoretical proof that entropy can be calibrated in polynomial time without sacrificing diversity.

## Background & Motivation
**Background**: Autoregressive language models suffer from "error accumulation" during generation — the model feeds its own (slightly biased) outputs as subsequent inputs, causing per-step entropy of generated text to increase with length, whereas per-step entropy of human text remains approximately constant.

**Limitations of Prior Work**: In practice, distribution truncation (top-k/top-p/min-p sampling) is used to reduce entropy and improve quality, but at the cost of diversity (higher log loss). This is particularly limiting in settings that require aggregating multiple answers or generating synthetic training data.

**Key Challenge**: Truncation improves quality but hurts diversity. It remains unclear whether scaling model size can automatically resolve miscalibration, and whether it is theoretically possible to preserve both quality and diversity simultaneously.

**Goal**: (1) Quantify the rate at which miscalibration improves with model scale; (2) establish a theoretical framework explaining why improvement is slow; (3) demonstrate the theoretical feasibility of lossless calibration.

**Key Insight**: The relationship between the scaling exponent and the data distribution parameter $\alpha$ is established by analyzing the probability of generating singleton tokens under power-law distributions.

**Core Idea**: The power-law exponent $\alpha \approx 1$ in data distributions causes the probability of generating rare tokens to decay as $m^{1/\alpha - 1} \approx m^0$ with increasing data volume, resulting in extremely slow improvement of miscalibration.

## Method

### Overall Architecture
The paper is structured in three parts: (1) theoretical analysis — scaling of singleton mass in a simplified model; (2) empirical measurement — miscalibration scaling across models ranging from 0.5B to 70B parameters; (3) theoretical feasibility — proving the existence of a polynomial-time lossless calibration algorithm.

### Key Designs

1. **Singleton Mass Scaling Theory**:

    - Function: Analyzes how the probability of generating tokens seen only once during training scales with data volume.
    - Mechanism: For an $\alpha$-power-law distribution, the probability of generating a singleton is $\mathbb{E}[K_{m,1}/m] = C_\alpha m^{1/\alpha - 1}$. When $\alpha \approx 1$ (typical for text), the exponent $\approx 0$, meaning the probability barely decreases as data volume grows.
    - Design Motivation: Once a singleton token is generated and enters the context, it derails subsequent generation; its probability governs the rate of error accumulation.

2. **Scaling Law Measurement Across Model Families**:

    - Function: Measures calibration error vs. model parameter count across 4 model families (Qwen2.5, Llama3, Llama2, Pythia) and 3 datasets.
    - Mechanism: Fits $\log \text{EntCE} = \beta \cdot \log m + C$ and measures the scaling exponent $\beta$.
    - Design Motivation: Validates theoretical predictions and confirms that scaling is extremely slow for text datasets but comparatively faster for code datasets.

3. **Theoretical Feasibility of Lossless Calibration**:

    - Function: Proves that a polynomial-time algorithm exists that can calibrate entropy without increasing log loss.
    - Mechanism: Assuming access to a black-box regression model that predicts the "future entropy" of text continuations given a prefix, a calibration procedure can be designed that adjusts token probabilities according to the expected future entropy of each candidate token.
    - Design Motivation: Braverman et al. (2020) proved the feasibility of global temperature scaling but showed it to be computationally intractable; this paper proves the polynomial-time feasibility of local adjustment.

## Key Experimental Results

### Scaling Exponents (calibration error vs. model size)

| Dataset | Power-law exponent $\alpha$ | Theoretically predicted exponent | Llama2/Pythia (measured) | Qwen2.5/Llama3 (measured) |
|--------|-----------------|------------------|-------------------|---------------------|
| WikiText | 0.918 | +0.089 | ~0.0 | ~-0.13 |
| WritingPrompts | 1.114 | -0.10 | ~0.0 | ~-0.13 |
| CodeContests | 1.5 | -0.33 | ~-0.2 | ~-0.35 |

### Effect of Instruction Tuning

| Setting | Entropy | Log Loss | Calibration |
|------|-----|---------|---------|
| Base (temperature 1.0) | Too high | Baseline | Uncalibrated |
| Temperature 0.85 | Reduced | Increased ↑ | Partially calibrated (diversity sacrificed) |
| Instruction-tuned | Greatly reduced | Greatly increased ↑ | Over-calibrated |

### Key Findings
- The scaling exponent for text datasets is close to 0 (~-0.05), implying that reducing calibration error by 10× requires a $10^{10}$-fold increase in model size.
- Code datasets exhibit more favorable scaling (exponent ~-0.3), owing to $\alpha = 1.5$ (a steeper power-law tail).
- Both instruction tuning and truncation trade diversity for quality — this explains the "alignment tax" phenomenon.
- Newer model families (Qwen2.5, Llama3) perform slightly better than older ones, possibly due to mid-training phases in pretraining data mixtures.

## Highlights & Insights
- **Elegant connection to power-law theory**: The paper links the classical Zipf's law in NLP to the generation quality of LLMs, explaining why "larger models use similar truncation parameters."
- **Theoretical hope for lossless calibration**: Although not practically realizable, the paper proves that generation stability and diversity are theoretically compatible, pointing the way for future research.
- **Implications for synthetic data generation**: If truncation harms diversity, using truncated models to generate training data may lead to capability degradation.

## Limitations & Future Work
- **Singleton mass model is overly simplified**: In practice, error accumulation does not arise solely from singleton tokens.
- **The lossless calibration algorithm is not practically executable**: It requires a black box capable of predicting the "future entropy" of text.
- **Only base models are studied**: Fine-grained analysis of instruction-tuned models remains limited.

## Related Work & Insights
- **vs. Braverman et al. (2020)**: The original paper identifying entropy miscalibration; this work adds scaling analysis and a proof of lossless calibration feasibility.
- **vs. Hewitt et al. (2022) min-p**: min-p is a practical truncation method; this paper explains why truncation cannot perfectly resolve the problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to establish scaling laws for entropy calibration, with an elegant connection to power-law theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 model families × 3 datasets, spanning 0.5B to 70B parameters.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are naturally interwoven, with clear conclusions.
- Value: ⭐⭐⭐⭐⭐ Profound implications for understanding LLM generation quality and scaling behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Decoupled Entropy Minimization](decoupled_entropy_minimization.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)

</div>

<!-- RELATED:END -->
