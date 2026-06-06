---
title: >-
  [Paper Note] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning
description: >-
  [ICML 2026][Code Intelligence][SFT] RankTuner introduces the Relative Rank Indicator $I_t$, which uses "the actual rank of the ground-truth token $R_t$" compared to "the expected rank under the model distribution $\mathb…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "SFT"
  - "token reweighting"
  - "probability-entropy calibration"
  - "relative rank"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: 819ef38f0472f55b
---

# Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning

**Conference**: ICML 2026  
**arXiv**: [2602.01745](https://arxiv.org/abs/2602.01745)  
**Code**: https://github.com/LvAoAo/Ranktuner_VERL  
**Area**: LLM Efficiency / Supervised Fine-tuning / Token Reweighting  
**Keywords**: SFT, token reweighting, probability-entropy calibration, relative rank, mathematical reasoning

## TL;DR
RankTuner introduces the Relative Rank Indicator $I_t$, which uses "the actual rank of the ground-truth token $R_t$" compared to "the expected rank under the model distribution $\mathbb{E}[R_t]$" as a single scalar signal. By intertwining probability $p_t$ (task alignment) and entropy $H_t$ (intrinsic uncertainty) into a token-level weight, it consistently outperforms pure probability or pure entropy reweighting baselines in Pass@1 on mathematical reasoning SFT.

## Background & Motivation

**Background**: In LLM fine-tuning, standard SFT that treats "every token equally" has been improved by various token-level reweighting methods. These are mainly divided into two camps: **Prob-Dominant** using ground-truth probability $p_t$ (e.g., DFT, TALR, OverTone) and **Entropy-Dominant** using predicted entropy $H_t$ (e.g., EAFT), both aiming to concentrate gradients on "important" tokens.

**Limitations of Prior Work**: Both camps rely on **one-dimensional** signals. Entropy-Dominant methods misidentify fillers or replaceable words like "umm" and "essentially" as "high uncertainty = important," thereby strengthening noise. Prob-Dominant methods severely penalize all low $p_t$ positions, forcing the model to struggle with tokens that naturally have multiple reasonable synonyms, thus undermining the linguistic flexibility provided by pre-training. A diagnostic test injecting noisy tokens (Tab. 1) reveals that among the top-10% high-weight tokens, the Entropy camp recalls 55% of the noise and the Prob camp recalls 40%, while RankTuner recalls only 26%—confirming that one-dimensional signals cause "collateral damage."

**Key Challenge**: $p_t$ measures "downstream alignment" while $H_t$ measures "upstream pre-training prior difficulty." These are orthogonal dimensions; any approach looking at only one will conflate "difficult tokens that should not be forced" with "easy tokens that were learned incorrectly."

**Goal**: Construct a scalar token weight that **simultaneously** reflects $p_t$ and $H_t$, while ensuring it is comparable, interpretable, and training-stable.

**Key Insight**: Probability and entropy cannot be directly divided due to different units, but **rank** is a common dimension for both. The actual rank $R_t$ of the ground truth is bounded by $1/p_t$, while the expected rank $\mathbb{E}[R_t]$ under the model distribution is bounded by entropy $H_t$ (a classic conclusion from the Guessing Problem). By mapping both to the rank space, they can be evaluated in a single ratio.

**Core Idea**: Use $I_t = 2^{f(R_t)-f(\mathbb{E}[R_t])}$ (where $f(x)=1/\log_2(x+1)$) as a relative rank signal to characterize "how poorly you guessed given this difficulty level." Its reciprocal $S_t = I_t^{-1}$ is used as the token weight for SFT loss, concentrating updates on "genuinely under-learned" positions rather than "inherently high-entropy" ones.

## Method

### Overall Architecture
RankTuner does not modify the model architecture or introduce new parameters; it replaces the base weight $w_t$ in the weighted NLL loss $\mathcal{L} = -\mathbb{E}[\sum_t w_t \log p_t]$ with $\tilde{w}_t = w_t \cdot S_t$. The pipeline is: for each target token $y_t$, obtain the full vocabulary distribution $\pi_\theta(\cdot|y_{<t},x)$ during the forward pass → Calculate ground-truth rank $R_t$ and expected rank $\mathbb{E}[R_t]=\sum_{\hat i} \hat i \cdot p_{t,\hat i}$ → Derive the Relative Scale $S_t$ from both → Multiply it by the token loss. For mathematical tasks, $w_t=p_t$ (compatible with DFT-family); for general tasks, $w_t=1$.

### Key Designs

1. **Relative Rank Indicator $I_t$ (Core Signal)**:
    - **Function**: Simultaneously encodes "task alignment" and "intrinsic uncertainty" into a single scalar.
    - **Mechanism**: From the Guessing Problem perspective, $R_t$ is the "number of guesses required to find the ground truth by traversing the vocabulary in descending order," and $\mathbb{E}[R_t]$ is the "expected number of guesses when randomly guessing according to the model distribution." Defining $I_t = g(f(R_t)-f(\mathbb{E}[R_t]))$ with $f(x)=1/\log_2(x+1)$ (logarithmic compression of rank, common in NDCG) and $g(x)=2^x$ (normalizing zero difference to $I_t=1$). A larger $R_t$ (worse guess) decreases $I_t$, while a larger $\mathbb{E}[R_t]$ (harder position) increases $I_t$. For the same token error, the penalty is lighter at high-difficulty positions and heavier at low-difficulty ones. When both $R_t$ and $\mathbb{E}[R_t]$ are large, $I_t$ saturates near 1, naturally forming a "Noise Region" that neutralizes replaceable/noisy tokens with high entropy and low probability.
    - **Design Motivation**: Direct ratios of $p_t/H_t$ suffer from dimensionality and range issues. Using rank as an intermediate representation works because $R_t \le 1/p_t$ and $\mathbb{E}[R_t] \ge \tfrac{1}{4}2^{H_t}+1$ (for $H_t \ge 2$) provide tight bounds that bridge probability and entropy to the rank space, making the ratio naturally comparable.

2. **Relative Competence Template and CMVT Derivation**:
    - **Function**: Provides a "probabilistic interpretation" for $I_t$, showing it approximates a meaningful competence ratio rather than being a hand-tuned heuristic.
    - **Mechanism**: Define abstract token competence $C_t = \rho(p_t)/\kappa(H_t)$ (where $\rho$ is monotonically increasing with $p_t$ and $\kappa$ is monotonically decreasing with $H_t$), analogous to conditional probability $\Pr(A|U)$: treating $p_t$ as the "joint support of alignment and prior" and $H_t$ as "effective prior support." Using the Cauchy Mean Value Theorem (CMVT) to write $f(R_t)-f(\mathbb{E}[R_t])$ in logarithmic ratio form yields $I_t = (\mathbb{E}[R_t]/R_t)^{K(\xi_t)}$, where $K(\xi_t) \approx 0.5$ (typical for reasoning tokens). Substituting rank bounds results in $\hat\rho(p_t)=p_t^{K(\xi_t)}$ and $\hat\kappa(H_t)=s(H_t)^{-K(\xi_t)}$, leading to $I_t \gtrsim \hat C_t = (p_t \cdot s(H_t))^{K(\xi_t)}$.
    - **Design Motivation**: Many reweighting methods are empirical heuristics, difficult to interpret or tune. This bridging "downgrades" the choice of $(f,g)$ to a specific CMVT instance. The appendix proves that switching to other monotonic $(f,g)$ yields stable results, suggesting gains come from the "probability-entropy calibration principle" itself rather than a specific function pair.

3. **Relative Scale $S_t = I_t^{-1}$ and Training Integration**:
    - **Function**: Converts the indicator into a token weight that can be multiplied with the SFT loss while maintaining training stability.
    - **Mechanism**: $S_t = (p_t \cdot s(H_t))^{-K(\xi_t)}$. In practice, let $\xi_t = \max(R_t, s(H_t))$ and $K(\xi_t) = (\log_2(\xi_t+1))^{-2}$ (dropping the $\xi/(\xi+1)$ factor for stability), resulting in $\tilde w_t = w_t \cdot S_t$. Algorithmically, this requires one sorting operation and expected rank accumulation on the forward logits, requiring no extra networks and zero extra inference cost.
    - **Design Motivation**: Using $I_t$ directly as a reward would overweight "well-learned" tokens, leading to overfitting on easy positions. Taking the reciprocal is equivalent to "applying more gradient pressure where learning is deficient" and naturally down-weighting "mastered" tokens. It is also compatible with RL post-training like PPO/GRPO (left for future work).

### Loss & Training
The base loss follows weighted NLL. $w_t = p_t$ is used for mathematical reasoning SFT (same base as DFT), and $w_t = 1$ for other general tasks. Training was performed on the verl framework using 4×A800 GPUs on 10k NuminaMath-CoT samples, AdamW lr=5e-5, cosine schedule + 0.1 warmup, batch size 256, max length 2048, with temperature 1.0 and max generation length 4096.

## Key Experimental Results

### Main Results: Math Reasoning Pass@1 / Pass@16 (Qwen3-8B, Selected)

| Dataset | Metric | RankTuner | Strongest Baseline | $\Delta$Best | Original Model |
|---|---|---|---|---|---|
| MATH-OAI | P@1 | **72.38** | 70.92 (DFT) | +1.46 | 65.14 |
| MATH-OAI | P@16 | **90.20** | 90.20 (EAFT) | +0.00 | 87.40 |
| Minerva Math | P@1 | 38.26 | 40.46 (TALR) | -2.20 | 31.39 |
| Minerva Math | P@16 | **65.44** | 63.60 (EAFT) | +1.84 | 48.53 |
| OlympiadBench | P@1 | **36.25** | 35.07 (DFT) | +1.18 | 27.19 |
| OlympiadBench | P@16 | **64.00** | 60.00 (TALR) | +4.00 | 51.11 |
| AIME24 | P@1 | **10.21** | 8.75 (DFT) | +1.46 | 6.04 |
| AIME24 | P@16 | **26.67** | 26.67 (TALR) | +0.00 | 26.67 |
| AMC23 | P@1 | **46.56** | 45.78 (DFT) | +0.78 | 35.62 |
| AMC23 | P@16 | **85.00** | 80.00 (EAFT/TALR) | +5.00 | 75.00 |

Trends are consistent on Qwen2.5-Math-7B: best or tied for best in 6/10 items. On AIME24 P@16, it maintains the original model performance while gaining +0.83 in P@1, proving more stable than baselines where "P@1 rises while P@16 crashes."

### Noise Sensitivity Diagnosis (Tab. 1)

| Method | TOK PREC@10% ↓ | TOK REC@10% ↓ | SEQ HIT@10% ↓ |
|---|---|---|---|
| Entropy-Dominant | 4.54% | 55.33% | 77% |
| Prob-Dominant | 3.25% | 39.65% | 77% |
| **Ours (RankTuner)** | **2.16%** | **26.39%** | **9%** |

By manually injecting noisy tokens into SFT data and measuring how much noise is covered in the top-10% high weights: RankTuner's proportion of "mistaking noise for importance" is significantly lower than one-dimensional approaches. The sequence-level hit rate plummeted from 77% to 9%, providing direct evidence for suppressing "high entropy + low probability" tokens into the Noise Region.

### Key Findings
- **OOD Reasoning Transfer**: RankTuner remains optimal on non-math reasoning benchmarks like ARC-C and GPQA, showing the calibration signal is not tied to math. DFT's strategy of "reweighting already confident tokens" leads to over-sharpening and poor transfer.
- **Automatic Down-weighting of Replaceable Tokens**: Visualization on CoT shows pronouns like "them" and "all" consistently fall into the $I\approx 1$ neutral zone, while calculation-critical tokens like "frac", "0", and "{" fall into the deep red $I < 1$ zone—indicating the signal successfully separates "linguistic flexibility" from "computational correctness."
- **Empirical Match with Theoretical Bounds**: Scatter plots of $R$-$p$ for Qwen3-8B track the $R=1/p$ upper envelope, and $\mathbb{E}[R]$-$H$ plots track the entropy lower bound, validating that using rank as a proxy for probability/entropy is tight and not a theoretical abstraction.
- **Insensitivity to $(f,g)$**: Ablation in the appendix shows stable results across different monotonic $(f,g)$ pairs, proving the gain stems from the principle of "bridging probability and entropy via rank."

## Highlights & Insights
- **Right Dimensionality**: Translating the incomparable $p_t$ and $H_t$ into the common dimension of "guessing cost" and then taking a ratio is a highly reusable idea for multi-signal fusion (e.g., coupling reward and KL terms in RLHF).
- **Elegant Diagnostic Design**: Injecting noise tokens and observing precision/recall of top-k weights provides a quantitative metric for whether reweighting is truly selecting the right positions, which is more intrinsic than just Pass@k. This protocol can be applied to evaluate any token-level reweighting method.
- **Zero Inference Overhead + Framework Friendly**: It only requires one sorting and accumulation step after the forward pass, and the weight is simply multiplied by the loss. Implemented on verl, it adheres to the "engineering-ready / no architecture changes" philosophy, offering a much lower barrier to entry than methods requiring attention modification or sampling changes.

## Limitations & Future Work
- Calculating $\mathbb{E}[R_t]=\sum_{\hat i}\hat i \cdot p_{t,\hat i}$ requires sorting the full vocabulary distribution. For large vocabularies (>100k), while $O(|V|\log|V|)$ per token is acceptable, the cumulative overhead might not be negligible; training latency for long-context + large-vocab scenarios was not provided.
- Main results were obtained on 10k NuminaMath-CoT samples, and backbones are concentrated in the Qwen family. Cross-family or cross-task (e.g., code generation) experiments are relatively sparse.
- Integration with RL post-training (PPO / GRPO) is listed as future work. The appendix suggests a form for injecting $S_t$ into the token-level policy ratio but lacks empirical testing, which is the scenario reasoning models care about most.
- $K(\xi_t)$ uses an approximation that drops the $\xi/(\xi+1)$ factor. While justified by training stability, the impact of this approximation on gradient scaling in long sequences is not shown.

## Related Work & Insights
- **vs DFT / TALR / OverTone (Prob-Dominant)**: These determine weights based on monotonic functions of $p_t$. They often gain in Pass@1 but lose in Pass@16 (evident in AIME24/AMC23) and suffer in OOD generalization. Ours proves that "adding entropy-based contextualization" preserves both P@1 and P@16.
- **vs EAFT (Entropy-Dominant)**: Pure entropy overweights filler tokens and has the highest noise recall. Ours uses the Noise Region mechanism to "reverse" this—neutralizing high entropy tokens rather than weighting them up.
- **Insight**: Reversing ranking metrics (like log-decay in NDCG) from "evaluation metrics" to "training signals" is an interesting direction. Any token-level task with "ground truth + model distribution" (multi-label, retrieval distillation) could mimic this rank bridging for more stable reweighting.

## Rating
- Novelty: ⭐⭐⭐⭐ Rank bridging and the CMVT explanation provide a clear theoretical narrative for "probability-entropy fusion" for the first time, moving beyond assembled heuristics.
- Experimental Thoroughness: ⭐⭐⭐ 5 math benchmarks + 2 OOD + noise diagnosis are convincing, but backbones are limited, and hard data for code/general SFT is missing.
- Writing Quality: ⭐⭐⭐⭐ The chain of Motivation → Counter-example → Theory → Implementation → Experiment is clear. The four-quadrant diagram in Fig. 1 intuitively explains "why one dimension fails."
- Value: ⭐⭐⭐⭐ Zero architecture changes, zero inference overhead, and drop-in compatibility with existing SFT pipelines provide direct engineering value for reasoning fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](../../ACL2026/code_intelligence/learning_adaptive_parallel_execution_for_efficient_code_localization.md)

</div>

<!-- RELATED:END -->
