---
title: >-
  [Paper Note] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning
description: >-
  [ICML 2026][Code Intelligence][SFT] RankTuner proposes the Relative Rank Indicator $I_t$, which uses a single scalar signal comparing the "actual rank $R_t$ of the ground-truth token" against the "expected rank $\mathbb{E}[R_t]$ under the model distribution." By coupling probability $p_t$ (task alignment) and entropy $H_t$ (intrinsic uncertainty) into a
tags:
  - ICML 2026
  - Code Intelligence
  - SFT
  - token reweighting
date: 2026-05-08
content_hash: 8a728f44837989ae
---
# Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning

**Conference**: ICML 2026  
**arXiv**: [2602.01745](https://arxiv.org/abs/2602.01745)  
**Code**: https://github.com/LvAoAo/Ranktuner_VERL  
**Area**: LLM Efficiency / Supervised Fine-tuning / Token Reweighting  
**Keywords**: SFT, token reweighting, probability-entropy calibration, relative rank, mathematical reasoning

## TL;DR
RankTuner proposes the Relative Rank Indicator $I_t$, which uses a single scalar signal comparing the "actual rank $R_t$ of the ground-truth token" against the "expected rank $\mathbb{E}[R_t]$ under the model distribution." By coupling probability $p_t$ (task alignment) and entropy $H_t$ (intrinsic uncertainty) into a token-level weight, it consistently outperforms pure probability/entropy reweighting baselines in Pass@1 for mathematical reasoning SFT.

## Background & Motivation

**Background**: In LLM fine-tuning, standard SFT—which treats "every token equally"—has been improved by various token-level reweighting methods. These are primarily divided into two schools: **Prob-Dominant** methods using ground-truth probability $p_t$ (e.g., DFT, TALR, OverTone), and **Entropy-Dominant** methods using prediction entropy $H_t$ (e.g., EAFT). Both aim to concentrate gradients on "important" tokens.

**Limitations of Prior Work**: Both schools rely on **one-dimensional** signals. Entropy-Dominant methods mistakenly identify filler/replaceable words like "umm" or "essentially" as "high uncertainty = important," thereby strengthening noise. Conversely, Prob-Dominant methods severely penalize all low-$p_t$ positions, forcing the model to learn tokens that naturally have multiple reasonable synonyms as errors, which destroys the linguistic flexibility provided by pre-training. A diagnostic test involving "deliberate noise injection" (Tab. 1) shows that among the top-10% high-weighted tokens, Entropy methods recall 55% of the noise and Prob methods recall 40%, while RankTuner recalls only 26%—confirming that one-dimensional signals indeed cause "collateral damage."

**Key Challenge**: $p_t$ measures "downstream alignment," while $H_t$ measures "upstream pre-training prior difficulty." These are orthogonal dimensions; any approach looking at only one side will conflate "difficult but should not be forced" with "easy but learned incorrectly."

**Goal**: Construct a scalar token weight that **simultaneously** reflects $p_t$ and $H_t$, while ensuring it is comparable, interpretable, and training-stable.

**Key Insight**: Although probability and entropy have different units and cannot be divided directly, **rank** is a common dimension. The actual rank $R_t$ is constrained by the $1/p_t$ upper bound, and the expected rank $\mathbb{E}[R_t]$ under the model distribution is constrained by the $H_t$ lower bound (a classic conclusion from the Guessing Problem). By mapping to the ranking space, both sides can be placed into the same ratio.

**Core Idea**: Use $I_t = 2^{f(R_t)-f(\mathbb{E}[R_t])}$ (where $f(x)=1/\log_2(x{+}1)$) as a relative rank signal to characterize "how poorly you guessed given this difficulty." The reciprocal $S_t = I_t^{-1}$ is then used as the token weight for the SFT loss, concentrating updates on "truly under-learned" tokens rather than "inherently high-entropy" positions.

## Method

### Overall Architecture
RankTuner addresses the issue of "one-dimensional token weight collateral damage." It does not modify the model architecture or introduce new parameters; it only replaces the base weight $w_t$ with $\tilde{w}_t = w_t \cdot S_t$ in the weighted NLL loss of SFT: $\mathcal{L} = -\mathbb{E}[\sum_t w_t \log p_t]$. During the forward pass, each target token $y_t$ already has a full vocabulary distribution $\pi_\theta(\cdot|y_{<t},x)$. RankTuner calculates the actual rank $R_t$ and the expected rank $\mathbb{E}[R_t]$ to derive a scalar $S_t$ that encapsulates both probability and entropy, which is then multiplied back into the token loss. Mathematical tasks follow $w_t=p_t$ (compatible with the DFT family), while general tasks use $w_t=1$.

### Key Designs

**1. Relative Rank Indicator $I_t$: Translating Probability and Entropy to the "Guessing Cost" Dimension**

The pain point is that $p_t$ and $H_t$ have different units and ranges, making direct fusion or division meaningless. RankTuner breaks this through the Guessing Problem perspective: $R_t$ is the "number of guesses until the truth is found in a descending sorted vocabulary," and the expected rank $\mathbb{E}[R_t]=\sum_{\hat i} \hat i \cdot p_{t,\hat i}$ is the "expected number of guesses when sampling from the model distribution." Rank is precisely sandwiched by probability and entropy: $R_t \le 1/p_t$ and $\mathbb{E}[R_t] \ge \tfrac{1}{4}2^{H_t}{+}1$ (for $H_t\ge 2$). These two tight bounds bridge probability and entropy into the same rank space, making the ratio naturally comparable.

Based on this, $I_t = g\big(f(R_t)-f(\mathbb{E}[R_t])\big)$ is defined with $f(x)=1/\log_2(x{+}1)$ (log-compression used in NDCG) and $g(x)=2^x$ (normalizing zero difference to $I_t=1$). The intuition is clear: larger $R_t$ (worse guess) decreases $I_t$, while larger $\mathbb{E}[R_t]$ (inherent difficulty) increases $I_t$. For the same token error, penalties are lighter at high-difficulty positions and heavier at low-difficulty ones. When both $R_t$ and $\mathbb{E}[R_t]$ are large, $I_t$ saturates near 1, naturally forming a "Noise Region" that neutralizes high-entropy, low-probability replaceable/noisy tokens.

**2. Relative Competence Template and CMVT Derivation: A Probabilistic Interpretation of $I_t$**

To prove $I_t$ is not just a hand-crafted heuristic, the paper defines an abstract token competence score $C_t = \rho(p_t)/\kappa(H_t)$ (where $\rho$ increases with $p_t$ and $\kappa$ decreases with $H_t$), analogous to conditional probability $\Pr(A|U)$. Using the Cauchy Mean Value Theorem (CMVT), the difference $f(R_t)-f(\mathbb{E}[R_t])$ is rewritten into a log-ratio form, yielding $I_t = (\mathbb{E}[R_t]/R_t)^{K(\xi_t)}$, where $K(\xi_t) \approx 0.5$ for typical reasoning tokens. Substituting the rank bounds yields $\hat\rho(p_t)=p_t^{K(\xi_t)}$ and $\hat\kappa(H_t)=s(H_t)^{-K(\xi_t)}$, confirming $I_t \gtrsim \hat C_t = (p_t \cdot s(H_t))^{K(\xi_t)}$ as a "lower bound estimate of the competence ratio."

The value of this bridging lies in "downgrading" the choice of $(f,g)$ to a specific CMVT instance. The appendix proves that results remain stable with other monotonic $(f,g)$ functions, suggesting that the gain comes from the principle of "calibrating probability and entropy via rank" rather than specific functional forms.

**3. Relative Scale $S_t = I_t^{-1}$ and Training Integration: More Gradients for Under-learned Positions**

Finally, the indicator is transformed into a weight. The key is taking the reciprocal rather than using $I_t$ directly as a reward; if $I_t$ were used directly, "well-learned" tokens would be over-weighted, leading to overfitting on easy positions. With $S_t = (p_t \cdot s(H_t))^{-K(\xi_t)}$, more gradients are applied to under-learned positions. For stability, the implementation uses $\xi_t = \max(R_t, s(H_t))$ and $K(\xi_t) = (\log_2(\xi_t{+}1))^{-2}$. The calculation involves only one sort and expected rank accumulation on the existing logits, introducing zero inference overhead and compatibility with RL post-training.

### Loss & Training
The base loss is weighted NLL; $w_t = p_t$ for math reasoning SFT and $w_t = 1$ for other tasks. Training is conducted on the verl framework using 4×A800 GPUs with 10k NuminaMath-CoT samples, AdamW lr=5e-5, cosine scheduler with 0.1 warmup, batch size 256, max sequence length 2048, and generation temperature 1.0.

## Key Experimental Results

### Main Results: Mathematical Reasoning Pass@1 / Pass@16 (Qwen3-8B, Excerpt)

| Dataset | Metric | RankTuner | Strongest Baseline | $\Delta$Best | Original |
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

Trends are consistent on Qwen2.5-Math-7B: Best or tied for best in 6/10 items. RankTuner is noticeably more stable than baselines that "gain in P@1 but crash in P@16."

### Noise Sensitivity Diagnosis (Tab. 1)

| Method | TOK PREC@10% ↓ | TOK REC@10% ↓ | SEQ HIT@10% ↓ |
|---|---|---|---|
| Entropy-Dominant | 4.54% | 55.33% | 77% |
| Prob-Dominant | 3.25% | 39.65% | 77% |
| **RankTuner** | **2.16%** | **26.39%** | **9%** |

By injecting noise tokens into SFT data, it was found that RankTuner's proportion of "mistaking noise for importance" is significantly lower than one-dimensional schemes. The sequence-level hit rate dropped from 77% to 9%, direct evidence of pushing "high entropy + low probability" into the Noise Region.

### Key Findings
- **OOD Reasoning Transfer**: RankTuner is optimal on ARC-C / GPQA, indicating the calibration signal is not tied to math tasks. Strategies like DFT encounter over-sharpening and poor transfer.
- **Automatic De-weighting of Replaceable Tokens**: In CoT visualization, pronouns like "them" and "all" fall into the $I\approx 1$ neutral zone, while critical tokens like "frac", "0", and "{" fall into the $I < 1$ high-weight zone.
- **Theoretical Bound Alignment**: Empirical scatters of $R$-$p$ and $\mathbb{E}[R]$-$H$ on Qwen3-8B align closely with theoretical bounds, validating rank as a tight proxy.
- **Robustness to $(f,g)$**: Stability across different functional forms proves the gain comes from the "rank-bridging" principle.

## Highlights & Insights
- **Right Dimensions**: Translating incomparable $p_t$ and $H_t$ into the common "guessing cost" rank space is a reusable strategy for multi-signal fusion.
- **Elegant Diagnostic Design**: Quantifying precision/recall for noise tokens provides a metric superior to downstream benchmarks for assessing the core purpose of reweighting.
- **Framework Friendly**: Zero inference overhead and simple integration into SFT pipelines make it highly practical for the engineering community.

## Limitations & Future Work
- Computing $\mathbb{E}[R_t]$ requires sorting the full vocabulary distribution, which may be costly for extremely large vocabularies (>100k).
- Experiments are concentrated on the Qwen family and NuminaMath-CoT; broader cross-family and non-math tasks (e.g., code) are needed.
- Integration with RL (PPO/GRPO) remains as future work.
- The use of $K(\xi_t)$ approximations for training stability lacks a detailed analysis of its impact on gradient scales over long sequences.

## Related Work & Insights
- **vs. DFT / TALR / OverTone**: These rely on monotonic functions of $p_t$, often gaining in Pass@1 but losing in Pass@16 and OOD scenarios. RankTuner maintains both by adding entropy-based contextualization.
- **vs. EAFT**: Pure entropy rewards filler tokens. RankTuner uses the mechanism to "reverse" this—neutralizing high entropy instead of rewarding it.
- **Insight**: Reversing ranking metrics (like log-decay in NDCG) as training signals is an intriguing direction for any token-level task requiring ground-truth alignment with model priors.

## Rating
- Novelty: ⭐⭐⭐⭐ Rank-bridging + CMVT provides the first clear theoretical narrative for prob-entropy fusion.
- Experimental Thoroughness: ⭐⭐⭐ Convincing math and OOD results, but backbone diversity is limited.
- Writing Quality: ⭐⭐⭐⭐ The logic chain from motivation to theory to experiment is very clear.
- Value: ⭐⭐⭐⭐ Plug-and-play for existing SFT pipelines with zero inference cost.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](../../AAAI2026/code_intelligence/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
