---
title: >-
  [Paper Note] Resolution Diagnostics for Paired LLM Evaluation
description: >-
  [ICML 2026][LLM Evaluation][Paired testing] This paper treats the ranking difference of "A vs B" on LLM leaderboards as a paired hypothesis testing problem. By inverting level-$\alpha$ / power-$(1-\beta)$ tests…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Paired testing"
  - "McNemar"
  - "Resolution Ratio"
  - "Minimum Detectable Effect"
  - "Leaderboard Multiple Comparisons"
date: 2026-05-08
content_hash: 35811eda039e3efa
---

# Resolution Diagnostics for Paired LLM Evaluation

**Conference**: ICML 2026  
**arXiv**: [2605.30315](https://arxiv.org/abs/2605.30315)  
**Code**: https://github.com/akotawala10/llm-power  
**Area**: LLM Evaluation / Hypothesis Testing / Leaderboard Statistics  
**Keywords**: Paired testing, McNemar, Resolution Ratio, Minimum Detectable Effect, Leaderboard Multiple Comparisons

## TL;DR
This paper treats the ranking difference of "A vs B" on LLM leaderboards as a paired hypothesis testing problem. By inverting level-$\alpha$ / power-$(1-\beta)$ tests, it defines the "Resolution Ratio" $q=N/N^\star$. It proves that the common calculator shortcut of multiplying the single-arm Cohen-$h$ formula by $(1-\rho)$ systematically underestimates the required sample size by half under small effects. Empirical findings show that 11/40 pairs on Open LLM Leaderboard v1 and 4/9 adjacent pairs in the MMLU-Pro top-10 are entirely "unresolvable" at $(\alpha,1-\beta)=(0.05,0.8)$, a number that increases to 6/9 after accounting for multiple comparisons, subject clustering, and anytime-validity constraints.

## Background & Motivation

**Background**: Modern LLM leaderboards translate tiny differences, such as "Model A scores 78.3% and Model B scores 77.5%" (a 0.8 pp gap), directly into headlines and product decisions. The relevant statistical tools for this are classic: binary accuracy uses the McNemar paired test with the Connor (1987) sample size formula, while continuous scoring uses paired $t$-tests or paired bootstrap.

**Limitations of Prior Work**: The closed-form required-$N$ formula provided by Miller (2024) is based on unpaired Gaussian assumptions. NLP/LLM works often either adopt this directly or use unpaired sample sizes (from Cohen 1988, G*Power, or R's `pwr` package) and multiply by $(1-\rho)$ as a paired correction. Consequently, many adjacent rankings claimed as "significant" fail to reach the target 0.8 power resolution under a true paired design.

**Key Challenge**: The true variance of a paired design is $\sigma_D^2 = p_A q_A + p_B q_B - 2\rho\sqrt{p_A q_A p_B q_B}$, which is the variance of the difference $X^A-X^B$ involving both arms. The "universal paired adjustment" of $(1-\rho)$ only scales the single-arm variance $p(1-p)$. Multiplying the single-arm result by $(1-\rho)$ misses a factor of 2 introduced by the summation $\text{Var}(X^A)+\text{Var}(X^B)$.

**Goal**: (1) Provide a standardized "Resolution Report" protocol for leaderboard evaluation; (2) Strictly characterize the bias of the aforementioned shortcut; (3) Audit real public leaderboards to identify unsupported rankings.

**Key Insight**: Treat "benchmark size + observed gap" as a hypothesis testing design parameter. Invert the classic Wald/McNemar-Connor formulas to derive three metrics: MDE (Minimum Detectable Effect at current $N$), $N^\star$ (required paired sample size for a target effect), and Resolution Ratio $q=N/N^\star$. A conclusion is supported only if $q\ge 1$.

**Core Idea**: Use $q=N/N^\star$ as a one-sentence diagnostic for every pair comparison on a leaderboard. Provide a sharp expansion lemma for small effects and a pip package `llm-power` to transform "paired sample size estimation" from empirical jargon into a verifiable standard.

## Method

### Overall Architecture
The framework accepts a per-question score matrix $\{(X_i^A, X_i^B)\}_{i=1}^N$ of two models on any paired benchmark. It outputs: (i) the minimum detectable effect $\delta_{\mathrm{MDE}}$ for the current $N$; (ii) the minimum paired sample size $N^\star$ for a target effect $\delta$; (iii) the ratio $q=N/N^\star(\hat\delta)$ for the observed gap $\hat\delta$. All quantities are derived by inverting two-sided level-$\alpha$, power-$(1-\beta)$ paired Wald tests (McNemar-Connor for binary, paired bootstrap for scores). The pipeline also incorporates Bonferroni/Holm/BH corrections, design effect clustering adjustments, and anytime-valid e-processes as stress tests to provide an end-to-end verdict table for leaderboards.

### Key Designs

1. **Resolution Ratio $q=N/N^\star$ as a Unified Diagnostic**:

    - **Function**: Compresses the reliability of a ranking into a dimensionless number. $q\ge 1$ indicates the benchmark is large enough to achieve $(\alpha,1-\beta)$ resolution, while $q<1$ indicates it is not.
    - **Mechanism**: Derived by inverting the Wald formula $N^\star(\delta;\alpha,\beta) = ((z_{1-\alpha/2}+z_{1-\beta})\sigma_D/|\delta|)^2$. For a single pair, $q$ corresponds directly to the squared Wald statistic (where $q\ge 1 \Leftrightarrow |T_N| \ge z_{1-\alpha/2}+z_{1-\beta}\approx 2.80$). The value-add lies in the aggregation layer—multiple comparisons, clustering, and anytime-validity are more naturally combined on the $q$ scale than the $p$-value scale.
    - **Design Motivation**: $q$ explicitly states "the benchmark lacks statistical power to resolve this gap," avoiding the "power at observed effect" trap criticized by Hoenig & Heisey (2001). It provides maintainers with a one-liner for each row.

2. **Small Effect Expansion Lemma (Lemma 1) — Quantifying Shortcut Bias**:

    - **Function**: Provides an explicit second-order constant $C(p,\rho)$ for the ratio $n_h/N^\star - 1/2$ near $p$, quantifying when the "single-arm $\times (1-\rho)$" shortcut is too inaccurate to use.
    - **Mechanism**: Taylor expands $h^2$ (Cohen's $h$ arcsine difference) and $\sigma_D^2$ at the midpoint $p_A=p+\delta/2, p_B=p-\delta/2$. Linear terms cancel due to symmetry, and combining $O(\delta^2)$ corrections yields $n_h/N^\star = 1/2 - (\delta^2/2)[(1+\rho)(1-2p)^2/(16(1-\rho)u^2) - 1/(6u)] + O(\delta^4)$, where $u=p(1-p)$. At $p=1/2$, the $(1-2p)^2$ term vanishes, leaving $C(1/2,\rho)=1/3$ independent of $\rho$. Corollary 1 provides a threshold $\delta^\star(p,\rho,\epsilon)=\sqrt{\epsilon/C(p,\rho)}$ for valid use.
    - **Design Motivation**: Industrial software (Cohen 1988, G*Power 3.1, R `pwr`) often returns single-arm $K/h^2$ sample sizes. This lemma formalizes the "doubling" factor ($N^\star \approx 2 \times n_{\text{shortcut}}$) as an $O(\delta^2)$ bias. For close comparisons ($|\hat\delta| \le 5$ pp), the shortcut underestimates $N^\star$ by half, potentially flipping a verdict from "unresolvable" to "significant."

3. **Triple Stress Tests: Multiple Comparison / Clustering / Anytime-Valid**:

    - **Function**: Simultaneously reports "unresolved pairs" under fixed-$n$, Bonferroni/Holm, clustering, and anytime-valid paradigms to see how verdicts tighten.
    - **Mechanism**: Multiple comparisons scale $N^\star$ by $((z_{1-\alpha'/2}+z_{1-\beta})/(z_{1-\alpha/2}+z_{1-\beta}))^2$; clustering scales IID $N^\star$ by the design effect $\mathrm{DE}=1+(\bar m-1)\mathrm{ICC}(D)$; anytime-validity replaces $z_{1-\alpha/2}$ with time-uniform boundaries $u(n)$ (Howard et al., 2021).
    - **Design Motivation**: Fixed-$n$ is a necessary but insufficient condition. Real leaderboards face family-wise error rates, subject clustering, and sequential updates. Presenting these side-by-side demonstrates the true statistical vulnerability of LLM rankings.

### Loss & Training
No training is involved. Values are derived from lm-evaluation-harness dumps and OLL detail repositories. Evaluation is performed on OLL v1 (40 pairs) and MMLU-Pro top-10 ($N=12{,}032$, 9 adjacent pairs). Calibration experiments were conducted on synthetic paired Bernoulli data ($p\in\{0.5,0.7,0.9\}\times \rho_z\in\{0,0.4,0.8\}$) with $M=1500$ trials/cell.

## Key Experimental Results

### Main Results

Unresolved proportion on OLL v1 (40 pairs) binned by $|\delta|$:

| $\|\delta\|$ Bin | Pairs | Unresolved | Median $r=N^\star/N$ | Worst |
|---|---|---|---|---|
| $\le 1\%$ | 3 | 3 (100%) | 94 | 1,892 |
| 1–2% | 4 | 4 (100%) | 4.2 | 6.8 |
| 2–5% | 10 | 4 (40%) | 0.75 | 2.8 |
| 5–15% | 17 | 0 (0%) | 0.15 | 0.65 |
| >15% | 6 | 0 (0%) | 0.03 | 0.07 |
| **all** | **40** | **11 (28%)** | **0.16** | **1,892** |

The resolution boundary falls near $|\delta|\approx 5\%$. Paired McNemar shows a median efficiency gain of 2.15× over the unpaired formula (Miller, 2024), matching the $1/(1-\rho)$ prediction. Prospective validation on 3 pairs with $|\delta|\in[6.3, 10.1]$ pp shows empirical power of 0.796–0.827, hitting the 0.80 target within $\pm 2.7$ pp.

### Ablation Study

Unresolved count for 9 adjacent pairs on MMLU-Pro top-10 under 4 paradigms:

| Paradigm | OLL v1 (40 pairs) | MMLU-Pro (9 pairs) | Description |
|---|---|---|---|
| Fixed-$n$ | 11/40 | 4/9 | Baseline verdict, IID assumption |
| Bonferroni / Holm | 14/40 | 4/9 | Family-wise error, $N^\star$ inflation ~2.11× |
| Anytime-valid | 14/40 | 5/9 | Uniform e-process, threshold inflation ~2.15× |
| Clustering | n/a | 6/9 | DE calculated via 14 MMLU-Pro subjects |

In the clustering column, the rank 4 vs 5 comparison jumps from IID $N^\star=432$ to cluster $N^\star=13{,}621$ (exceeding $N$), as model differences associate with specific subjects ($\mathrm{ICC}(D)=0.036$, $\mathrm{DE}=31.5$).

### Key Findings
- **Lemma 1 accuracy**: On OLL v1, the $n_h/N^\star$ median is 0.5002 (IQR [0.4999, 0.5035]). For close comparisons ($|\hat\delta|\le 2\%$), it hits 0.5000 precisely. The shortcut consistently underestimates $N^\star$ by half.
- **HellaSwag boundary**: gemma-7B vs Llama-3-8B shows $\hat\delta=+0.46$ pp ($N=10{,}042$). The asymptotic $\chi^2_1$ yields $p=0.049$ (significant), but the exact conditional binomial yields $p=0.054$ (not significant); $q\approx 1/2$. This highlights that "significant" is not the same as "resolvable."
- **Clustering robustness**: The clustering verdict is robust to different granularities. Leave-one-subject-out (LOSO) maintains 6/9 unresolved in 11/14 cases.

## Highlights & Insights
- **Formalizing shortcut bias**: What was previously "common knowledge" is written as a strict Lemma. The explicit $C(p,\rho)$ and the $p=1/2$ independence are elegant additions to statistical reporting.
- **Engineering value of $q=N/N^\star$**: $q$ provides a superior abstraction for aggregation compared to $p$-values. Family-wise, design effect, and anytime-valid adjustments all translate directly into $N^\star$ multipliers, making the audit more intuitive.
- **Synergy of stress tests**: The side-by-side display of stress test results provides a powerful "reporting protocol" for reproducibility and benchmark design, showing how easily tiny gaps vanish under realistic statistical assumptions.

## Limitations & Future Work
- Authors acknowledge the empirical focus on binary accuracy; continuous scoring and pairwise preferences (Chatbot Arena style) were treated via paired bootstrap but lack the same depth of empirical audit.
- MMLU-Pro clustering is limited to $K=14$ subjects; finer granularity (task templates) might further inflate the design effect.
- The work separates "resolution" from "construct validity"; thus, $q\ge 1$ is a necessary but not sufficient condition for a meaningful benchmark comparison.
- Future work: Generalize Lemma 1 to multi-arm power and integrate Bradley–Terry e-processes for Elo-style leaderboards.

## Related Work & Insights
- **vs Miller (2024)**: Miller uses unpaired Gaussian formulas; this work quantifies the paired efficiency gain at 2.15× and provides a per-pair reporting protocol.
- **vs Card et al. (2020) / Dror et al. (2018)**: While previous work identified underpowered NLP tests, this paper provides the first systematic audit of paired $N^\star$ across multiple modern leaderboards.
- **vs Madaan et al. (2024) / Jo & Wilson (2025)**: These works estimate benchmark variance; this paper uses those estimates to derive testing power, showing calibration and design effect corrections are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Lemma 1 provides a new explicit constant; the $q$ framework is a systematic integration of known tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad coverage of public leaderboards, multiple statistical calculators, and robust calibration.
- Writing Quality: ⭐⭐⭐⭐ Transparent explanation of subtle statistical points; excellent comparative tables.
- Value: ⭐⭐⭐⭐⭐ Provides a practical pip package and protocol for the increasingly crowded LLM evaluation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](../../ACL2026/llm_evaluation/beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ICML 2026\] Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](../../AAAI2026/llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)

</div>

<!-- RELATED:END -->
