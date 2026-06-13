---
title: >-
  [Paper Note] CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting
description: >-
  [ICML 2026][LLM Evaluation][LLM Benchmarking] CapBencher injects randomness into each test item (generating multiple logically correct answers and randomly selecting one as the gold standard) to cap the Bayes accuracy of…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM Benchmarking"
  - "Data Contamination Detection"
  - "Bayes Accuracy"
  - "Test-set Overfitting"
  - "Leaderboard Hacking"
date: 2026-05-08
content_hash: 3622508c1573b8ea
---

# CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting

**Conference**: ICML 2026  
**arXiv**: [2505.18102](https://arxiv.org/abs/2505.18102)  
**Code**: https://github.com/TakashiIshida/capbencher  
**Area**: LLM Evaluation  
**Keywords**: LLM Benchmarking, Data Contamination Detection, Bayes Accuracy, Test-set Overfitting, Leaderboard Hacking  

## TL;DR
CapBencher injects randomness into each test item (generating multiple logically correct answers and randomly selecting one as the gold standard) to cap the Bayes accuracy of the benchmark at a controllable level (e.g., 50%). This enables black-box statistical detection of data contamination in publicly released benchmarks—any model with accuracy significantly exceeding the Bayes upper bound is flagged as contaminated.

## Background & Motivation

**Background**: Constructing modern LLM benchmarks has become increasingly expensive—FrontierMath required over 60 professional mathematicians (including IMO gold medalists and Fields Medalists), and Humanity's Last Exam aggregated contributions from 1,000+ experts across 50 countries. Once these high-investment benchmarks release their answers on the internet, they are liable to be intentionally or unintentionally included in LLM training data.

**Limitations of Prior Work**: The mainstream strategy is "private evaluation": keeping the benchmark private and requiring participants to submit models or predictions to an evaluation server. However, this approach still fails to defend against "leaderboard hacking" via repeated queries and lacks effective statistical detection methods. Existing contamination detection methods (e.g., Min-k%/Min-k%++) either require access to model logits (unavailable for closed-source models) or use canary strings (which can be maliciously removed) and do not support rigorous statistical testing.

**Key Challenge**: There is a fundamental contradiction between "openness" and "evaluation security"—publicly releasing a benchmark facilitates open community evaluation, but public answers lead to data contamination and test-set overfitting.

**Goal**: Design a benchmark release protocol that retains open evaluation capabilities while providing a built-in contamination detection mechanism, without fully exposing the true answers.

**Key Insight**: The authors observe that if the Bayes accuracy upper bound of a benchmark can be controlled, any model performance exceeding that bound constitutes statistical evidence of contamination. The key insight is that benchmark designers can proactively lower the Bayes accuracy.

**Core Idea**: By preparing multiple logically correct answers for each question and randomly selecting one as the ground truth, the Bayes accuracy is "capped." This allows the benchmark to function as both an evaluation tool and a contamination alarm.

## Method

### Overall Architecture
The CapBencher pipeline consists of two stages: the **Release Stage**, where randomness is injected into each item of the original benchmark to generate and publicly release a "capped" version (with ground truth answers obfuscated); and the **Detection Stage**, where a one-sided binomial test is used to determine if a model's accuracy significantly exceeds the Bayes upper bound. The input is the original benchmark $(q, a)$ pairs, and the output is the capped benchmark $(q', a')$ pairs along with the contamination detection conclusion.

### Key Designs

1.  **Answer Randomization (Capping)**:
    - **Function**: Lower the Bayes accuracy to a target level by injecting controllable randomness to obfuscate the ground truth answers.
    - **Mechanism**: For each question $x$, a set $F(x)$ containing $L$ logically correct answers is constructed, and one is selected uniformly at random as the gold standard $Y'$. For instance, "3×6=?" could be modified to "randomly select an integer from [1,5] and add it to the answer," resulting in answers like 17 or 19 (instead of 18), which reduces the Bayes accuracy to $1/L$. Two strategies are provided: **Obfuscation** (true answers are hidden, suitable for direct response and multiple-choice) and **Disclosure allowed** (true answers are visible but with random labels, suitable for benign threat models).
    - **Design Motivation**: By controlling $L$, benchmark designers can balance contamination detection sensitivity (higher $L$ is more sensitive) and ranking precision (higher $L$ increases variance).

2.  **Statistical Detection based on Binomial Test**:
    - **Function**: Provide a rigorous statistical framework to judge whether a model is contaminated.
    - **Mechanism**: The null hypothesis assumes the model's true performance does not exceed the Bayes accuracy $\alpha$. Under this hypothesis, the number of correct predictions follows $\text{Binom}(n, \alpha)$. The p-value is calculated as $\sum_{i=k}^{n}\binom{n}{i}\alpha^{i}(1-\alpha)^{n-i}$, where $k$ is the number of correct responses. Based on the Karlin-Rubin theorem, this is a Uniformly Most Powerful (UMP) test. In practice, the upper bound $1/L$ from Corollary 1 can substitute for the unknown exact Bayes accuracy.
    - **Design Motivation**: Unlike Min-k%, where the AUC threshold requires validation set tuning, the binomial test provides exact p-values, remains effective for small samples, and requires no access to model internals.

3.  **Unbiased Estimator of Original Scores**:
    - **Function**: Recover original benchmark scores from capped scores to maintain model rankings.
    - **Mechanism**: It is theoretically proven that an affine relationship exists between capped and original scores: $s_{\text{capped}}(X) = (\frac{1}{L} - \frac{L-1}{L(K-1)})s_{\text{orig}}(X) + \frac{L-1}{L(K-1)}$, where $K$ is the number of options and $L$ is the number of randomized answers. From this, an unbiased estimator $\hat{A}$ is derived, with variance only modestly higher than the original estimator (e.g., standard deviation on MMLU increases from 0.003 to 0.012).
    - **Design Motivation**: Ensure that the capped benchmark can still reliably evaluate and rank models while obfuscating answers, with experimental Kendall's $\tau = 0.92$.

## Key Experimental Results

### Main Results: Contamination Detection

Contamination experiments were conducted via continual pre-training on 12 benchmarks using several model families (Llama, Qwen, DeepSeek):

| Model | Benchmark | Uncontaminated Acc | Contaminated Acc | Bayes Bound | Detection Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen 2.5-14B | GSM8K | ~40% | >50% | 50% | ✅ Detected (fewer epochs) |
| Qwen 2.5-3B | GSM8K | ~35% | >50% | 50% | ✅ Detected (more epochs) |
| Llama 3.2-3B-Instruct | GSM8K (α=50%) | — | >50% | 50% | ✅ Detected at epoch 6 |
| Llama 3.2-3B-Instruct | GSM8K (α=25%) | — | >25% | 25% | ✅ Detected at epoch 5 |
| Llama 3.2-3B-Instruct | GSM8K (α=10%) | — | >10% | 10% | ✅ Detected at epoch 4 |

### Comparison with baseline methods

| Detection Method | Internal Access Req. | Bypassable | Statistical Test | Full Benchmark Success |
| :--- | :---: | :---: | :---: | :---: |
| **CapBencher** | ❌ | Partially (RE) | ✅ Exact p-value | ✅ All datasets |
| Canary String | ✅ (log prob) | ✅ (removal) | ❌ Approximate | ✅ Inconsistent |
| Min-k% | ✅ (logits) | — | ❌ No threshold theory | ❌ (Failed on GPQA) |
| Min-k%++ | ✅ (logits) | — | ❌ No threshold theory | ❌ (Failed on GPQA) |

### Leaderboard Hacking Detection (Model Merging)

| Model | Accuracy (%) | Detected |
| :--- | :--- | :--- |
| Qwen 2.5-7B-Instruct | 39.87 ± 2.32 | — |
| DeepSeek-R1-Distill-Qwen-7B | 40.02 ± 2.01 | — |
| Qwen 2.5-Math-7B-Instruct | 41.02 ± 2.12 | — |
| Merged Model (1+2+3) | 56.52 ± 2.04 | ✅ Significantly > 50% |

### Key Findings
- Larger models are detected as contaminated in fewer epochs (consistent with Carlini et al. 2023), indicating that CapBencher is more sensitive to larger models.
- Lowering the Bayes accuracy (e.g., from 50% to 10%) allows for earlier detection of contamination but increases the variance of ranking estimates, representing a sensitivity vs. precision tradeoff.
- In cross-lingual experiments on MMLU-ProX, contamination was detectable in all languages. For GSM8K (without chain-of-thought), it was only detectable in European languages, suggesting that the inclusion of reasoning processes significantly reduces the detectability of cross-lingual contamination.
- Reverse Engineering (RE) experiments show that even when models are provided with obfuscated answers and randomization rules, current frontier models cannot reliably recover the ground truth (e.g., RE accuracy on HLE-MC is far below 50%).

## Highlights & Insights
- **Proactive Benchmark Defense**: Unlike passive contamination detection, CapBencher allows designers to embed a detection mechanism at the time of release—"installing a built-in alarm for the benchmark." This proactive mindset can be transferred to other data leakage scenarios, such as watermarking for private datasets.
- **Leveraging Classical Statistics**: The method utilizes the classical concept of the Bayes error rate but innovatively flips "estimating Bayes error" to "controlling Bayes accuracy," turning a passive theoretical tool into an active design parameter.
- **Ranking Preservation via Affine Transformation**: The affine relationship between capped and original scores not only supports unbiased recovery but also naturally ensures the monotonicity of model rankings. This theoretical guarantee significantly enhances the practical value of the method.

## Limitations & Future Work
- Randomized answers may still leak weak signals (e.g., an answer of 19 implies the true value might be 18 or 20); future stronger models might exploit these clues via reverse engineering.
- For open-ended generation tasks (e.g., free text, long-form generation), constructing a set of logically correct answers $F(x)$ is difficult, which limits the scope of application.
- The variance of capped scores increases with $L$. On small-scale benchmarks (e.g., GPQA-diamond, $n=198$), ranking correlation decreases significantly ($\tau$ drops to 0.67).
- Future directions include stronger obfuscation strategies, cryptographic extensions like Zero-Knowledge Proofs, and robustness evaluations against more sophisticated RE attackers.

## Related Work & Insights
- **Data Contamination Detection**: Existing works like Min-k%/Min-k%++ (require logits) or canary strings (removable); CapBencher unifies "defense" and "detection."
- **Dynamic Benchmarking**: Systems like LiveBench and DynaBench avoid contamination by continuously generating new items but are restricted to domains where solutions can be automatically generated. CapBencher is applicable to expert-level problems without efficient algorithmic solutions.
- **Bayes Error Rate Estimation**: Classical statistical work on estimating Bayes error rates (Fukunaga & Hostetler 1975, etc.) provides a theoretical basis, but the uniqueness of this work lies in "designing" rather than "estimating" Bayes accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](../../ACL2026/llm_evaluation/multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)
- [\[NeurIPS 2025\] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](../../NeurIPS2025/llm_evaluation/your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](../../ACL2026/llm_evaluation/how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ICML 2026\] Resolution Diagnostics for Paired LLM Evaluation](resolution_diagnostics_for_paired_llm_evaluation.md)

</div>

<!-- RELATED:END -->
