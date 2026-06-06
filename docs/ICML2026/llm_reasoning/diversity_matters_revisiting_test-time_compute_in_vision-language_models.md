---
title: >-
  [Paper Note] Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models
description: >-
  [ICML2026][LLM Reasoning][Test-time Compute] This paper systematically investigates the effectiveness of test-time compute (TTC) strategies in Vision-Language Models (VLMs). It theoretically demonstrates that the gains o…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Test-time Compute"
  - "Majority Voting"
  - "Entropy Selection"
  - "VLM Ensemble"
  - "Prediction Diversity"
date: 2026-05-08
content_hash: a355e142d496d3df
---

# Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models

**Conference**: ICML2026  
**arXiv**: [2605.30713](https://arxiv.org/abs/2605.30713)  
**Code**: https://github.com/nanfang-wuyu/Diversity-Matters  
**Area**: LLM Inference / Multimodal VLM  
**Keywords**: Test-time Compute, Majority Voting, Entropy Selection, VLM Ensemble, Prediction Diversity

## TL;DR
This paper systematically investigates the effectiveness of test-time compute (TTC) strategies in Vision-Language Models (VLMs). It theoretically demonstrates that the gains of majority voting are limited by prediction diversity and proposes ETTC, which selects the most confident model based on prediction entropy. This allows smaller models to enhance larger ones, achieving an average gain of +2.8% over voting across 7 VLMs and 6 benchmarks, outperforming the strongest single model.

## Background & Motivation

**Background**: In LLMs, TTC (test-time compute) has been proven to significantly improve inference quality without modifying parameters. Mainstream approaches fall into two categories: feature-based Best-of-N (scoring using heuristics like pivot words, response length, and lexical diversity) and confidence-based aggregation (self-consistency / majority voting). These methods are considered standard for "lightweight performance boosts," yet few have systematically verified their effectiveness on VLMs.

**Limitations of Prior Work**: Directly applying LLM TTC to VLMs carries three risks: (1) Visual perception inherently has higher error rates and significant variance across models; (2) Imperfect cross-modal alignment leads to subtle inconsistencies; (3) Textual cues used to judge reasoning quality in LLMs (e.g., pivot words like "alternatively" or "let me check," CoT length) do not reflect the correctness of visual understanding—no matter how elegant the reasoning chain, it cannot recover from failed perception.

**Key Challenge**: The essence of voting is to amplify correct signals using "diversity + average accuracy > 1/K." However, VLM outputs tend to be highly convergent during sampling, leading to insufficient diversity. Furthermore, while multi-model ensembles are naturally diverse, standard voting weights all models equally, allowing weak models to drag down strong ones, sometimes resulting in worse performance than the strongest single model.

**Goal**: The paper addresses three sub-questions: (i) When exactly is TTC effective for VLMs? (ii) What is the quantitative relationship between voting gain and prediction diversity? (iii) Can an aggregation strategy be designed for multi-model ensembles to "automatically trust the strongest expert," allowing small models to enhance large ones?

**Key Insight**: The authors start from a simple observation: "If the same model answers incorrectly 16 times in the same way, voting is useless; if different models fail in different ways, voting is the cure." This attributes voting effectiveness to "statistical dependency between predictions," quantified via NMI and correlation coefficient $\rho$. Furthermore, in multi-model scenarios, "the most confident model is the most likely to be correct," using normalized prediction entropy as a selection signal.

**Core Idea**: Replace "counting votes" with "selecting the model with the lowest entropy as the answer." In single-model scenarios, this degrades to majority voting; in multi-model scenarios, it allows strong models to dominate while allowing weak models to surpass the large model when they are highly confident.

## Method

### Overall Architecture
The paper presents a complete pipeline of "Diagnosis → Theory → Improvement." Input: A multiple-choice visual reasoning problem, $K$ candidate answers, $U$ predictions (either from one model sampled $U$ times or $M$ different VLMs × multiple samples). Output: The aggregated final option. The process involves three phases: (1) Systematically running feature heuristics (Pivot Word / CoT Length / Feature-All) and majority voting across 7 VLMs and 6 benchmarks, confirming that feature-based methods fail and voting is only slightly effective when using CoT (§3); (2) Using Information Theory metrics (NMI and correlation coefficient $\rho$) to characterize prediction dependency, proving that voting gain $\Delta A_{MV}(U)$ monotonically decreases with respect to $\rho$ and NMI (§4, Theorem 1); (3) Proposing ETTC, which selects models by prediction entropy, theoretically proving it strictly outperforms voting under weak assumptions (§5, Theorem 2).

### Key Designs

1.  **Prediction Dependency Metrics (NMI + $\rho$)**:
    - **Function**: To quantify "diversity" among $U$ predictions in a model-agnostic way without requiring ground truth, thereby predicting whether voting is worthwhile.
    - **Mechanism**: For the answer options, the average Normalized Mutual Information $\mathrm{NMI}(X;X') = I(X;X') / \min\{H(X), H(X')\}$ is calculated across all $C(U,2)$ pairs. For binary correctness indicators $Z_u = \mathbb{I}\{X_u = Y\}$, the average correlation coefficient $\rho(Z,Z') = (E[ZZ'] - p^2) / (p(1-p))$ is used. Theorem 1 proves: If all pairs have the same dependency level, $\Delta A_{MV}(U)$ monotonically decreases with both $\rho$ and NMI; at $\rho=1$, the gain is zero; at $\rho=0$ and $p > 1/K$, voting accuracy approaches 1 as $U \to \infty$.
    - **Design Motivation**: To turn the intuition of "when voting is useful" into a measurable criterion, providing practitioners with a label-free pre-screening tool—if a large model has high $\rho$, do not waste compute on voting; if a small model has low $\rho$, voting is cost-effective.

2.  **Entropy-based TTC (ETTC) Selection Rule**:
    - **Function**: To replace majority voting in multi-model ensemble scenarios, allowing models to "speak based on confidence" and preventing groups of weak models from outvoting the strong model's correct answer.
    - **Mechanism**: For each model $u$, the normalized entropy $\tilde{H}_u = -\frac{1}{\log K}\sum_k p_u(k)\log p_u(k) \in [0,1]$ and top-1 prediction $\hat{y}_u = \arg\max_k p_u(k)$ are calculated from its prediction distribution $p_u(\cdot)$ over $K$ options. The final answer is $\hat{y}_{u^*}$ where $u^* = \arg\min_u \tilde{H}_u$. In single-model multi-round scenarios, averaging the distributions and taking the argmax causes ETTC to degrade into majority voting, maintaining backward compatibility. In multi-model scenarios, strong models dictate when confident, while weak models can surpass them when they are occasionally more certain.
    - **Design Motivation**: Voting treats all models as equal weight voters, which fails in VLM ensembles when "weak but correlated models" cluster; given that "low entropy $\Rightarrow$ high accuracy" (Assumption 1: Entropy-Accuracy Monotonicity) generally holds in practice, entropy is used as a cheap proxy for "expert status."

3.  **Theoretical Guarantees for ETTC (Theorem 2)**:
    - **Function**: To provide a strict inequality for ETTC over voting, explaining why the ETTC advantage grows with higher correlation.
    - **Mechanism**: A coupling model is established where, with probability $\lambda$, all non-optimal models cluster to give the same correlated error prediction $W$ (accuracy $\bar{c}$), and with probability $1-\lambda$, they are independent. Let $c^*$ be the accuracy of the strongest model and $A_{MV}(0)$ be the baseline voting accuracy under complete independence. This yields $A_{MV}(\lambda) = \lambda \bar{c} + (1-\lambda) A_{MV}(0)$, while ETTC accuracy $A_{\min H} \ge c^*$. The difference $A_{\min H} - A_{MV}(\lambda) = \lambda(c^* - \bar{c}) + (1-\lambda)(A_{\min H} - A_{MV}(0))$ is strictly greater than 0 as long as $\lambda>0$ and $\bar{c} < c^*$.
    - **Design Motivation**: Since VLMs share large pre-training corpora and architectures, errors are naturally correlated ($\lambda \neq 0$). This theory characterizes the core advantage of ETTC in real-world scenarios—it structurally plugs the "collective bias" loophole in voting.

### Loss & Training
This is a pure inference-time method that requires no model training or reward models. All experiments use stochastic decoding (HuggingFace default sampling) + zero-shot one-stage prompting. Both CoT and Direct Answer templates are evaluated. For single-model setups, the sample count is $U = 16$ (based on §4.2 showing NMI and $\rho$ converge at $U \approx 12$). Multi-model ensembles use 4 models, each sampled multiple times.

## Key Experimental Results

### Main Results

| Dataset | Metric | Vanilla strongest single model | Majority Voting | ETTC | Gain (ETTC vs Voting) |
|--------|------|--------------------|----------|------|-----------------------|
| MathVista (cross-family) | Acc% | 72.08 (Qwen-7B) | 68.33 | **75.93** | +7.60 |
| MathVision (cross-family) | Acc% | 31.84 (Gemma) | 32.05 | **35.57** | +3.52 |
| TQA (cross-family) | Acc% | 78.86 (Gemma) | 83.65 | **83.90** | +0.25 |
| MMMU (cross-family) | Acc% | 52.49 (Gemma) | 53.66 | **58.63** | +4.97 |
| Average (cross-family, 6 datasets) | Acc% | 61.30 (Qwen-7B) | 63.75 | **66.56** | +2.81 |
| Average (same-family Qwen 3B/7B/32B/72B) | Acc% | 69.90 (Qwen-72B) | 68.84 | **71.68** | +2.84 |

**Highlights**: In the same-family Qwen series, voting (68.84) was even lower than the Qwen-72B single model (69.90), validating the theoretical prediction that "voting dilutes strong models." ETTC (71.68) outperformed the strongest model by 1.78 points, indicating that Qwen-3B/7B/32B can occasionally surpass 72B when they are highly certain.

### Ablation Study

| Configuration / Observation | Key Metric | Description |
|-------------|----------|------|
| Direct Answer + Majority Voting | <1% Gain | Without CoT, VLM outputs across 16 samples are nearly identical; zero diversity makes voting ineffective. |
| CoT + Pivot Word / CoT Length / Feature-All | ≈ 0% Gain | Textual heuristics fail completely on VLMs because perception bottlenecks decouple text style from correctness. |
| CoT + Majority Voting | 2–4% Gain | Voting shows a small, consistent boost only under CoT, but is limited by prediction dependency. |
| $\Delta A_{MV}(16)$ vs NMI / $\rho$ | Significant Negative Correlation (Fig. 3) | Validates Theorem 1 across 7 models × 6 datasets: higher dependency leads to lower voting gain. |
| Convergence of NMI/$\rho$ (U=2…16) | Stabilizes after $U \approx 12$ | Provides a practical upper limit of 16 samples. |
| Model Size vs Diversity | Qwen-3B/LLaMA: High diversity, high voting gain; Qwen-72B/Pixtral: Convergent output, near-zero gain. | Practical principle: Use voting for small models. |

### Key Findings
- The fundamental reason majority voting is "more decorative than useful" in VLMs is the high correlation of sampled outputs. This conclusion provides a quantifiable and predictive criterion using NMI and $\rho$.
- ETTC not only beats voting but can also outperform the strongest single model—meaning small models can be more confident than large models on problems they "truly know." Selecting these minority opinions is key to "smaller models enhance larger ones."
- Textual heuristics (pivot words, length, lexical diversity) failed entirely on VLMs, suggesting that VLM reasoning quality is primarily determined by visual perception, and surface text features can no longer gauge perception success.
- Entropy-Accuracy Monotonicity (Assumption 1) largely holds in measurements (§C.2), providing the empirical basis for ETTC's cross-architecture generalization.

## Highlights & Insights
- **Linking Voting Gains to Statistical Dependency**: While voting was previously treated as an engineering trick, this paper provides a theoretical criterion and empirical fit (via NMI and $\rho$) for "whether voting will yield gains." This acts as a budget estimator for TTC that requires no labels and works across tasks and models.
- **Elegance of ETTC's "Single Model Degradation"**: In single-model scenarios, ETTC is mathematically equivalent to majority voting, allowing it to seamlessly replace existing self-consistency pipelines without regression risk. It only triggers the "strongest expert dictatorship" mode in multi-model ensembles.
- **Anti-intuitive "Small Models Enhancing Large Ones" Conclusion**: Traditional ensembles assume large models should dominate. This paper proves that Qwen-3B is occasionally more confident and correct than 72B, showing that "per-instance expert selection" is more fine-grained than "per-model expert selection," offering a new direction for low-cost ensembles.
- **Tight Coupling of Theory and Empirics**: Theorem 1 explains "why voting fails," and Theorem 2 explains "why ETTC is better." Both are supported by visual experimental results in §4–§5. This paradigm of "quantifying failure before designing solutions" can be transferred to diagnosing other inference-time methods like RAG, self-refine, or Best-of-N scorers.

## Limitations & Future Work
- Only validated in Multiple Choice Question (MCQ) scenarios, where $K$ options allow entropy to be directly normalized to $[0,1]$. Defining "answer distribution entropy" for open-ended generation (free-form, code, long-form QA) is non-obvious and may require clustering or semantic equivalence judgment.
- Assumption 1 (low entropy $\Rightarrow$ high accuracy) holds in an "aggregate" sense; the paper admits it may not hold strictly for single instances. For "confidently wrong" models (poorly calibrated VLMs), ETTC might be misled, requiring calibration correction or temperature adjustment.
- Ensemble costs are somewhat downplayed—the total inference overhead for multiple models + multiple samples per model is dozens of times that of a single model. The paper lacks a fair comparison against "applying the same compute to training a larger model or longer CoT."
- The possibility of combining ETTC with more complex TTC methods like process reward models, self-refinement, or tree search was not explored. It could serve as a unified entropy-based selection layer in those pipelines.

## Related Work & Insights
- **vs Self-Consistency / Majority Voting (Wang et al., 2023)**: Standard TTC baseline for LLMs. This paper shows it only yields 2-4% gains on VLMs and depends on CoT. ETTC is mathematically equivalent in single-model cases but strictly superior for ensembles.
- **vs Feature/Heuristic Best-of-N (Chang et al., 2025; Fu et al., 2023; Jin et al., 2024)**: These methods score reasoning traces using pivot words or length. This paper proves they fail on VLMs because visual perception bottlenecks make surface text cues non-discriminative—a strong counter-example for reward-model-free methods.
- **vs Learned Reward Models / Verifiers**: These require additional training of scorers, whereas ETTC is training-free and model-agnostic, offering lower deployment costs but potentially sacrificing task-specific signals learned by reward models.
- **Insight**: The idea of "using prediction entropy as a model quality proxy" can be extended to RAG (selecting among multiple retrieved-augmented answers), agent multi-path reasoning (selecting trajectories), and speculative decoding (switching between draft/verifier based on entropy). Any inference-time scenario with multiple candidate outputs is worth testing with this formula.

## Rating
- Novelty: ⭐⭐⭐⭐ Turns "voting diversity" into measurable theorems and designs the backward-compatible ETTC. Clear logic, though the single-point innovation is incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 VLMs × 6 benchmarks × two ensemble configurations. Tight coupling of theory and empirics, though lacks open-ended tasks and compute trade-off analysis.
- Writing Quality: ⭐⭐⭐⭐ Smooth "why → when → how" narrative; theorems, intuitive explanations, and charts support each other well.
- Value: ⭐⭐⭐⭐ The engineering conclusion "don't use voting on large models; use entropy selection in multi-model ensembles" is immediately actionable for VLM inference services and low-cost ensembling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](../../ACL2026/llm_reasoning/scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
