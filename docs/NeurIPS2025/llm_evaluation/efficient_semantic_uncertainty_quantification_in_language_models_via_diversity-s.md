---
title: >-
  [Paper Note] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling
description: >-
  [NeurIPS 2025][LLM Evaluation][semantic uncertainty] This paper proposes a diversity-steered sampling framework that injects NLI-based semantic similarity penalties during decoding to encourage semantically diverse gener…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "semantic uncertainty"
  - "diversity sampling"
  - "importance weighting"
  - "NLI"
  - "language models"
date: 2026-05-08
content_hash: 9c732a7486a0a6e5
---

# Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2510.21310](https://arxiv.org/abs/2510.21310)  
**Code**: None  
**Area**: LLM Evaluation
**Keywords**: semantic uncertainty, diversity sampling, importance weighting, NLI, language models

## TL;DR
This paper proposes a diversity-steered sampling framework that injects NLI-based semantic similarity penalties during decoding to encourage semantically diverse generation, and corrects distributional bias via importance weighting with control variates to reduce variance. The method accurately estimates semantic entropy (aleatoric uncertainty) and mutual information (epistemic uncertainty) of LLMs using as few as 16 samples.

## Background & Motivation

**Background**: Uncertainty quantification for LLMs in open-ended question answering (QA) primarily relies on Semantic Entropy (SE)—clustering multiple generations by semantics and computing the entropy of the resulting cluster distribution. Recent work further computes mutual information (MI) via iterative prompting to measure epistemic uncertainty.

**Limitations of Prior Work**: These methods require a large number of IID samples for stable estimation, yet standard sampling yields many semantically redundant outputs (i.e., paraphrases of the same answer), wasting computational resources. Diversity heuristics such as temperature scaling and nucleus sampling are semantically agnostic and thus fail to effectively cover distinct semantic clusters.

**Key Challenge**: Accurate estimation requires coverage of as many semantic clusters as possible to obtain a reliable cluster distribution, but standard sampling concentrates on high-probability regions, making it difficult for a small sample to cover rare clusters. Increasing the sample count is a straightforward solution but incurs high inference costs.

**Goal**: To efficiently cover a greater number of semantic clusters with a small number of samples (e.g., 16), thereby accurately estimating both aleatoric and epistemic uncertainty in LLMs.

**Key Insight**: Semantic diversity penalties are injected directly into the decoding process—at each token step, tokens semantically similar to already-generated outputs are penalized, steering subsequent generation toward new semantic directions. The key innovation is fine-tuning an NLI model to support pairwise evaluation of incomplete sequences.

**Core Idea**: NLI entailment scores are used as continuous penalty terms injected into decoding logits to steer sampling away from existing semantic clusters, while importance weighting corrects the resulting distributional bias.

## Method

### Overall Architecture
The framework consists of three steps: (1) **Diversity-steered sampling**—modifying the token-level conditional distribution to penalize candidates semantically similar to existing generations; (2) **Importance-weighted bias correction**—since the sampling distribution $q \neq p$, self-normalized importance weights $w_i = p(s_i)/q(s_i)$ are used to correct cluster probability estimates; (3) **Control variate variance reduction**—leveraging the correlation between model log-probabilities and the target statistic to construct a control variate that further reduces estimation variance.

### Key Designs

1. **Injection of Semantic Similarity Penalties**:

    - Function: Adds a continuous penalty term at each token decoding step.
    - Mechanism: For each candidate token $y_t$, the bidirectional entailment score between the current partial sequence $y_{\leq t}$ and the most similar sample in the existing generation set $\mathcal{S}$ is computed as $E(y_{\leq t}, s) = \frac{1}{2}(\text{entailment}(y_{\leq t}, s) + \text{entailment}(s, y_{\leq t}))$, and the logits are modified as $\log \tilde{q}(y_t | y_{<t}) = \log p(y_t | y_{<t}) - \lambda \max_{s \in \mathcal{S}} E(y_{\leq t}, s)$.
    - Design Motivation: Max aggregation ensures that new generations diverge from the most similar existing sample; bidirectional entailment captures semantic equivalence more accurately than unidirectional entailment.

2. **NLI Model Fine-tuning for Incomplete Sequences**:

    - Function: Fine-tunes the NLI model to yield reliable entailment judgments over partially generated sequences (prefixes or masked sequences).
    - Mechanism: All parameters of a pretrained DeBERTa-large-MNLI are frozen; only a newly added [TRUNC] token embedding and classification head are trained (~3M parameters, 0.3% of total). Training data are constructed by randomly truncating one side of MNLI samples. For MDM, [MASK] token embeddings are analogously fine-tuned.
    - Design Motivation: Standard NLI models can only compare complete sentence pairs, whereas diversity steering requires evaluating incomplete prefixes during generation. Lightweight fine-tuning preserves the original NLI performance while enabling prefix handling.

3. **Importance-Weighted Bias Correction**:

    - Function: Corrects the sampling bias introduced by diversity steering.
    - Mechanism: For samples drawn from $q$, self-normalized importance weights $\tilde{w}_i = \frac{p(s_i)/q(s_i)}{\sum_j p(s_j)/q(s_j)}$ are computed and used to weight cluster probability estimates as $\hat{p}(c) \approx \sum_i \mathbf{1}[s_i \in c] \tilde{w}_i$.
    - Design Motivation: Diversity steering shifts the sampling distribution away from the model's true distribution; without correction, this produces biased uncertainty estimates.

4. **Control Variate Variance Reduction**:

    - Function: Reduces the variance of importance-weighted estimates by leveraging available information (model log-probabilities).
    - Mechanism: A control variate $X_i = -\log p(s_i)$ (model log-probability) is constructed, which is correlated with the target variable $Y_i = -\log \hat{p}(c(s_i))$ (cluster log-probability). An adaptive coefficient $\alpha$ minimizes variance: $\hat{H}_{cv} = \sum \tilde{w}_i Y_i - \alpha \sum \tilde{w}_i (X_i - \mu_X)$.
    - Design Motivation: $X_i$ is already available when computing importance weights, so no additional inference cost is incurred.

5. **Extension to Masked Diffusion Models (MDM)**:

    - Function: Applies the same diversity-steered framework to the iterative denoising process of MDMs.
    - Mechanism: At each denoising step, semantic similarity between candidate infillings and existing generation trajectories is computed and penalized. The NLI model is fine-tuned to handle sequences containing [MASK] tokens.
    - Design Motivation: MDMs are a powerful and increasingly prominent generative paradigm, yet have been entirely overlooked in uncertainty quantification. This work is the first to extend semantic uncertainty estimation to MDMs.

### Loss & Training
NLI fine-tuning uses standard cross-entropy loss, updating only the [TRUNC]/[MASK] embeddings and the classification head. The base LLM requires no modification or gradient access—the framework is fully modular.

## Key Experimental Results

### Main Results

AUROC for semantic entropy estimation with $N=16$ samples on 4 QA benchmarks (correctness threshold: ROUGE-L < 0.3):

| Method | CoQA (OPT-13B) | TriviaQA (OPT-13B) | TruthfulQA (OPT-13B) | CoQA (LLaDA 8B) |
|------|----------|-----------|-----------|----------|
| Vanilla (τ=1) | .81±.04 | .76±.06 | .67±.04 | .85±.04 |
| Temperature (τ=2) | .82±.04 | .80±.04 | .67±.03 | .89±.04 |
| DBS | .83±.04 | .85±.04 | .67±.03 | — |
| SDLG | .83±.03 | .81±.04 | .70±.04 | — |
| **Ours** | **.85±.04** | **.85±.04** | **.71±.04** | **.94±.02** |

### Ablation Study

| Configuration | Semantic Cluster Coverage | AUROC | ESS/N |
|------|-----------|-------|-------|
| Full method (diversity + IS + CV) | Highest | Best | ~0.4–0.6 |
| Without control variate | Highest | Slightly lower | ~0.4–0.6 |
| Without importance weighting | Highest | Biased decrease | N/A |
| Temperature scaling only | Moderate | Moderate | 1.0 |

### Key Findings
- **Significant improvement in semantic cluster coverage**: On CoQA, the proposed method covers 1.5–2× more semantic clusters on average than standard sampling (Figure 4), directly translating to more accurate uncertainty estimates.
- **Largest gains on high-ambiguity datasets**: Improvements are most pronounced on CoQA and AmbigQA (multi-answer questions), where the semantic space is broader.
- **Particularly strong results for MDMs**: AUROC on LLaDA 8B improves from .85 to .94 (+9%), providing the first demonstration of high-quality semantic uncertainty estimation for MDMs.
- **ESS/N > 0.4**: The effective sample size ratio of importance weights consistently remains high, indicating that diversity steering does not deviate excessively from the original distribution.
- **High-quality NLI prefix evaluation**: Entailment probabilities converge to their final judgments when sequences are only 30–50% unrolled (Figure 2), confirming that prefixes already carry sufficient semantic signal.

## Highlights & Insights
- **Alignment between generation and estimation**: The same NLI model is used both to steer diversity sampling and to define downstream semantic clustering, ensuring consistency between the sampling space and the estimation space. This "metric-driven diversity" approach is far more principled than semantically agnostic temperature scaling.
- **Extremely lightweight NLI adaptation**: Fine-tuning only 0.3% of parameters (one token embedding and a classification head) enables the NLI model to handle incomplete sequences—an elegant and practical engineering choice.
- **First unified framework for uncertainty estimation in ARM and MDM**: By handling [TRUNC] (truncation marker) and [MASK] (mask marker) respectively, the same framework applies to two fundamentally different generative paradigms.
- **Zero-cost variance reduction via control variates**: Model log-probabilities already obtained during importance weight computation are repurposed as control variates, reducing estimation variance without any additional inference overhead.

## Limitations & Future Work
- **Sequential generation overhead**: Each new sample requires computing semantic similarity against all previously generated samples, resulting in generation time that grows linearly with the number of samples $N$. Batch parallelization strategies are an important direction for future work.
- **Limitations of NLI models**: Entailment judgments are inherently noisy, particularly for long texts and complex reasoning. Hard clustering (bidirectional entailment = same cluster) may miss gradual semantic differences.
- **Robustness of a single penalty strength $\lambda$**: Although the paper proposes an adaptive $\lambda$ strategy, its generalization across different question types remains to be validated.
- **Tokenizer compatibility between NLI model and LLM**: When the two tokenizers differ, prefix handling requires additional decoding and re-encoding steps.

## Related Work & Insights
- **vs. Semantic Entropy [Kuhn et al.]**: SE is purely an estimation framework and does not modify the sampling strategy; the proposed method makes SE more accurate under low-sample regimes through diversity steering.
- **vs. SDLG [Aichberger et al.]**: SDLG achieves diversity by replacing key tokens but requires NLI gradients, does not support MDMs, and does not account for the diversity of the running sample set. The proposed method is gradient-free and unified across ARM and MDM.
- **vs. Conformal Prediction**: Conformal methods provide theoretical guarantees but require calibration datasets; the proposed method requires no additional data.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of diversity-steered sampling and importance-weighted bias correction is a natural yet carefully executed innovation; NLI prefix fine-tuning is a notable highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 QA benchmarks, 4 models (including MDM), and multiple sampling baselines; ablations could be more systematic.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical exposition is rigorous, motivation is clear, and algorithm pseudocode is complete.
- Value: ⭐⭐⭐⭐ The modular design requires no gradient access to the LLM, making it highly practical; pioneering coverage of uncertainty estimation for MDMs is significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Asymmetric Duos: Sidekicks Improve Uncertainty](asymmetric_duos_sidekicks_improve_uncertainty.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](on_the_entropy_calibration_of_language_models.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](../../ACL2026/llm_evaluation/revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)
- [\[NeurIPS 2025\] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
