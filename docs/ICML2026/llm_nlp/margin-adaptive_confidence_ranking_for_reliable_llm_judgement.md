---
title: >-
  [Paper Note] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement
description: >-
  [ICML 2026][LLM/NLP][LLM-as-a-judge] Addressing the frequently violated monotonicity assumption in LLM-as-a-judge—where "high confidence implies reliability"—this work proposes mapping multiple in-context predictive prob…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "LLM-as-a-judge"
  - "confidence estimation"
  - "margin ranking"
  - "PAC-Bayesian bound"
  - "selective evaluation"
date: 2026-05-08
content_hash: 60e41a7c952b7d39
---

# Margin-Adaptive Confidence Ranking for Reliable LLM Judgement

**Conference**: ICML 2026  
**arXiv**: [2605.15416](https://arxiv.org/abs/2605.15416)  
**Code**: None currently available  
**Area**: LLM Evaluation / Selective Prediction / PAC-Bayes Generalization  
**Keywords**: LLM-as-a-judge, confidence estimation, margin ranking, PAC-Bayesian bound, selective evaluation

## TL;DR
Addressing the frequently violated monotonicity assumption in LLM-as-a-judge—where "high confidence implies reliability"—this work proposes mapping multiple in-context predictive probabilities to confidence using a small MLP. Through a margin-based ranking loss and a margin-adaptive training strategy derived from PAC-Bayes generalization bounds, the learned confidence achieves lower ranking loss and higher AUROC across four datasets and six judge models, significantly improving the success rate of target consistency in fixed-sequence testing.

## Background & Motivation

**Background**: LLM-as-a-judge is currently the mainstream method for evaluating open-ended generation tasks (e.g., AlpacaEval, Chatbot Arena). Its reliability typically depends on a "selective evaluation" pipeline: first estimating the confidence $C_{LM}(x)$ for each sample, and then using fixed-sequence testing to find a threshold $\widehat{\lambda}$ such that the human-machine agreement rate for high-confidence samples reaches a target $1-\alpha$ with high probability. Jung et al. (2025) provided a formal PAC risk upper bound within this framework.

**Limitations of Prior Work**: All such methods rely on an implicit "monotonicity assumption"—that higher confidence implies lower disagreement risk with human judgment. Empirical results in Figures 1 and 3 of this paper demonstrate that the relationship between confidence and human-machine agreement is often non-monotonic for both predictive probabilities and simulated annotators (especially on AlpacaEval and Chatbot Arena). Furthermore, existing theories only provide guarantees for risk **under calibration set conditions**, lacking analysis of the **out-of-sample generalization** of the confidence estimator itself.

**Key Challenge**: Treating heuristic signals (predictive probability, verbalized confidence) as reliable confidence essentially conflates "classification correctness" with "ranking consistency." The former only requires correct discrimination near the threshold, while the latter requires the **global ordinal relationship** to align monotonically with the human-machine agreement rate, which are two entirely different learning objectives.

**Goal**: To model confidence as a learnable ranking function $C_\theta$ and establish PAC-Bayes generalization bounds for its "misranking probability," thereby theoretically controlling selective risk monotonicity.

**Key Insight**: The authors represent each sample $x$ as a high-dimensional feature vector $s = (\mathbb{P}_{LM}(r_1\mid x; t_1),\dots,\mathbb{P}_{LM}(r_1\mid x; t_{|\mathcal{T}|}))$, where each $t_i$ is a different set of in-context annotation examples. In this way, simulated annotators are no longer manually aggregated into a max-mean scalar but are fed as multi-view raw features into a small MLP.

**Core Idea**: Train $C_\theta$ using a margin-based pairwise ranking loss and derive a trade-off through PAC-Bayes: "larger margin $\rightarrow$ harder to minimize empirical loss, but lower generalization complexity." Based on this, $\theta$ and the margin $\gamma$ are **jointly optimized**.

## Method

### Overall Architecture
Given a judge model $f_{LM}$ and a small-scale calibration set $S_{cal}$ with human preference labels:

1. For each sample $x$, use $N=5$ simulated annotators with $K=5$ examples each, enumerating all $1\sim K$-shot subsets $\mathcal{T}$ to obtain the LLM's predictive probabilities for candidate $r_1$ under each subset, concatenated into a feature vector $s \in \mathbb{R}^{|\mathcal{T}|}$.
2. Partition $S_{cal}$ into agreement/disagreement sets based on whether $f_{LM}(x)$ equals the human label $y$. Randomly construct 5,000 pairs $(x_i, x_j)$ satisfying $a(x_i) > a(x_j)$ to form the ranking training set $S_{pair}$.
3. Use a 3-layer MLP (64→32→16, ReLU, finishing with sigmoid) $C_\theta: \mathbb{R}^{|\mathcal{T}|} \to [0,1]$ to fit the ordinal relationship "agreement samples should be ranked above disagreement samples."
4. After training, embed $C_\theta$ into the fixed-sequence testing framework of Jung et al. (2025). Select the threshold as $\widehat{\lambda} = \inf\{\lambda: \widehat{R}^+(\lambda') \le \alpha,\ \forall \lambda' \ge \lambda\}$ to achieve selective evaluation with high-probability consistency guarantees.

### Key Designs

1. **Multi-perspective in-context features + ranking-based confidence**:
    - **Function**: Replaces manual aggregation of simulated annotators (taking the max average) with the preservation of all $|\mathcal{T}|$-dimensional raw probabilities, allowing the MLP to learn how to aggregate autonomously.
    - **Mechanism**: Defines agreement indicator $a(x) = \mathbb{1}\{f_{LM}(x) = y\}$ for sample $(x,y)$, and defines the margin ranking loss $\ell_\gamma(\theta; x_i, x_j) = \mathbb{1}(C_\theta(s_i) < C_\theta(s_j) + \gamma)$ over all ordered pairs satisfying $a(x_i) > a(x_j)$. During training, softmax is used as a differentiable proxy via $\log(1+e^{-(C_\theta(s_i)-C_\theta(s_j)-\gamma)/0.1})$.
    - **Design Motivation**: Monotonicity is a property **concerning ordinal relationships**; therefore, the training objective must directly penalize errors where "agreement samples are ranked lower than disagreement samples," rather than doing so indirectly via classification accuracy.

2. **PAC-Bayes margin-based generalization bound**:
    - **Function**: Explicitly characterizes the "misranking probability" of confidence as the empirical margin loss plus a margin-dependent complexity term.
    - **Mechanism**: First, the PAC-Bayes framework (McAllester, 1999; Neyshabur et al., 2017) is used to obtain the expected ranking risk upper bound for a randomized estimator $C_{\theta+\mathbf{u}}$. Then, via a sharpness constraint $\mathbb{P}_{\mathbf{u}}(\max_s |C_{\theta+\mathbf{u}}(s) - C_\theta(s)| < \gamma/4) \ge 1/2$, the bound is converted for a deterministic estimator, resulting in $\mathcal{RK}( \theta) \le \widehat{\mathcal{RK}}_\gamma(\theta) + \mathcal{O}(\sqrt{(\Phi(C_\theta) + \ln(3m_p/\delta'))/(\gamma^2 (m_p - 1))})$, where $\Phi(C_\theta) = n^2 h \ln(nh) \prod_l \|W_l\|_2^2 \sum_l \|W_l\|_F^2/\|W_l\|_2^2$.
    - **Design Motivation**: Upgrades the justification of "why this confidence is trustworthy" from heuristic explanation to quantifiable generalization guarantee; meanwhile, the $1/\gamma^2$ dependence in the complexity term indicates that the margin $\gamma$ is the key knob for controlling generalization.

3. **Margin-adaptive joint optimization**:
    - **Function**: Replaces a fixed margin with automatic calibration based on data noise levels, mitigating the conflict between "clean data needing large margins" and "noisy data needing small margins."
    - **Mechanism**: Simplifies the complexity term in the bound to $\mathcal{C}_\gamma(\theta) = \sqrt{\sum_l \|W_l\|_F^2}/\gamma$ (using Frobenius norm as a differentiable proxy for spectral norm) and optimizes $\min_{\theta,\gamma}\widehat{\mathcal{RK}^s_\gamma}(\theta) + \beta\,\mathcal{C}_\gamma(\theta)$. Since joint optimization of $\gamma$ can be unstable due to non-smoothness, the authors use decoupled alternating updates: fix $\gamma$ to update $\theta$ with SGD, then fix $\theta$ to select $\gamma$ that minimizes the objective.
    - **Design Motivation**: Theoretical analysis shows no universal optimal margin—clean data can seek higher separation, while noisy data requires compromise. Letting $\gamma$ adapt to data characteristics is key to turning the PAC-Bayes bound from an "analytical tool" into a "training objective."

### Loss & Training
- **Empirical Objective**: $\min_{\theta} \min_{\gamma}\, \widehat{\mathcal{RK}^s_\gamma}(\theta) + \beta\,\mathcal{C}_\gamma(\theta)$, with $\beta = 10^{-4}$.
- **Hyperparameters**: 3-layer MLP (64-32-16), 30 epochs, learning rate $10^{-3}$, weight decay $10^{-4}$; ~3,000 training samples per dataset with 5,000 randomly sampled pairs.
- **Inference**: After obtaining $C_\theta$, it replaces $C_{LM}$ in the fixed-sequence testing framework of Jung et al., using the binomial exact $(1-\delta)$ upper bound $\widehat{R}^+(\lambda)$ to find the minimum acceptable threshold.

## Key Experimental Results

### Main Results: Ranking Loss & AUROC (Selected 3 Judges × 4 Datasets)

| Judge | Dataset | Metric | Predictive Prob. | Simulated Annot. | Learning (Vanilla) | **Ours** |
|-------|---------|--------|------------------|------------------|--------------------|----------|
| Mistral-7B | AlpacaEval | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.421 / 0.580 | 0.418 / 0.582 | 0.387 / 0.618 | **0.339 / 0.667** |
| Mistral-7B | Chatbot Arena | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.332 / 0.665 | 0.323 / 0.677 | 0.282 / 0.703 | **0.274 / 0.713** |
| Llama3-70B | AlpacaEval | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.402 / 0.599 | 0.384 / 0.616 | 0.324 / 0.673 | **0.278 / 0.705** |
| Llama3-70B | Chatbot Arena | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.255 / 0.746 | 0.265 / 0.735 | 0.249 / 0.752 | **0.217 / 0.787** |
| Qwen2.5-72B | HH-RLHF | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.441 / 0.554 | 0.368 / 0.647 | 0.348 / 0.658 | **0.281 / 0.713** |
| Qwen2.5-72B | TL;DR | $\mathcal{RK}\downarrow$ / AUROC$\uparrow$ | 0.372 / 0.627 | 0.361 / 0.631 | 0.338 / 0.658 | **0.279 / 0.702** |

Across all 4 datasets × 6 judge model combinations, the proposed method achieves the lowest ranking loss and highest AUROC.

### Ablation Study: Vanilla (standard ranking loss only) vs. Ours (margin-adaptive)

| Configuration | AlpacaEval $\mathcal{RK}\downarrow$ | HH-RLHF $\mathcal{RK}\downarrow$ | Chatbot Arena $\mathcal{RK}\downarrow$ | TL;DR $\mathcal{RK}\downarrow$ | Description |
|---------------|-------------------------------------|----------------------------------|----------------------------------------|--------------------------------|-------------|
| Predictive Probability | 0.402 | 0.448 | 0.255 | 0.416 | Heuristic baseline |
| Simulated Annotators | 0.384 | 0.357 | 0.265 | 0.392 | Manual aggregation by Jung et al. |
| Learning Confidence (Vanilla) | 0.324 | 0.360 | 0.249 | 0.358 | Same MLP + fixed margin ranking loss |
| **Learning Confidence (Ours)** | **0.278** | **0.309** | **0.217** | **0.314** | + margin-adaptive joint optimization |

(Comparison on Llama3-70B judge) The improvement from Vanilla to Ours (~0.03–0.05 ranking loss) is specifically attributed to margin-adaptive training, indicating that the PAC-Bayes derived strategy is not merely decorative.

### Key Findings
- **Restoration of Monotonicity**: Figure 3 shows that on Llama3-8B + Chatbot Arena, the confidence-agreement curve of Simulated Annotators is significantly non-monotonic, whereas the proposed method pulls the curve back to monotonicity, making fixed-sequence testing truly viable.
- **Bernoulli Simulation (Figure 2)**: In 10,000 synthetic experiments, ranking loss and monotonicity violation rate rise in tandem, empirically supporting the inference that "lowering ranking loss reduces monotonicity violations."
- **Downstream Gains**: In cascaded selective evaluation (L→Q→O, M→L→O) with a target agreement $1-\alpha = 0.85$, the guarantee success rate of traditional heuristic selection is nearly 0%, while the learned confidence significantly boosts the success rate, bridging "theoretical improvement" and "real-world deployment."

## Highlights & Insights
- **Shifting LLM-as-a-judge from "heuristic confidence" to "learnable ranking functions"**: Previous confidence estimation either took softmax probabilities or manually aggregated simulated annotators. This work is the first to explicitly model it as a **Learning to Rank** problem, allowing a small MLP to learn judge- and dataset-specific aggregation methods.
- **PAC-Bayes theory truly "feeds back" into the training objective**: The complexity term $\sqrt{\sum_l \|W_l\|_F^2}/\gamma$ appears directly in the loss. The margin is no longer an empirical "magic" hyperparameter but an optimizable quantity derived from generalization bounds, a rare "theory-to-practice" closed loop in LLM evaluation.
- **Transferable Trick**: Packaging predictive probabilities under multiple in-context example sets into a feature vector for a small model is a "sub-prompt ensemble as features" approach applicable to any downstream task requiring reliable signals from LLMs (e.g., calibration, selective abstention, reward modeling).

## Limitations & Future Work
- Still relies on a **small calibration set with human preference labels** (~3,000 samples per dataset); the method cannot train $C_\theta$ in zero-annotation scenarios.
- Replacing the spectral norm with the Frobenius norm in the complexity term, while differentiable, may loosen the bound; a "proxy gap" remains between theory and the training objective.
- Alternating updates for $\gamma$ converge stably in practice but remain sensitive to the choice of learning rate and $\beta$; the paper lacks discussion on fallbacks for failed joint optimization.
- Experiments are based entirely on "two-candidate preference judgment" formats; complex judge scenarios like multi-candidate or open-ended scoring are not covered.
- The dimensionality of the feature vector $|\mathcal{T}|$ grows exponentially with $K, N$ ($\sum_{k=1}^K \binom{K}{k} \cdot N$), causing inference overhead to increase rapidly with the number of in-context examples.

## Related Work & Insights
- **vs Jung et al. (2025) Simulated Annotators**: Jung used fixed max-mean aggregation + heuristic confidence. This work treats the same in-context signals as raw features for a learnable MLP trained under a ranking objective. Both are compatible with the fixed-sequence testing framework, with this work serving as a plug-in upgrade for the confidence module.
- **vs Mohri & Hashimoto (2024) Conformal Prediction**: Conformal methods provide marginal correctness guarantees but assume confidence is trustworthy. This work, conversely, directly guarantees the **ranking consistency** of confidence, offering an orthogonal and complementary perspective.
- **vs Neyshabur et al. (2017) PAC-Bayes for Classification**: While Neyshabur analyzed discrete classification loss, this work extends it to $[0,1]$ continuous output + ranking error, deriving margin-dependent complexity via sharpness constraints—providing a clear template for "continuous-output + ranking" combinations.
- **vs Verbalized Confidence**: Asking an LLM to report its own confidence is the cheapest solution, but Table 1 shows its ranking loss and AUROC are the worst, confirming that "prompted confidence" is almost never directly reliable.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizes LLM-judge confidence as a PAC-Bayes margin ranking problem for the first time, with margin-adaptive optimization "reverse-engineered" from theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 datasets × 6 judge models + cascaded evaluation validation; however, lacks public code and coverage of multi-candidate/open scoring.
- Writing Quality: ⭐⭐⭐⭐ Strong chain of motivation—theory—method—experiment; PAC-Bayes derivations are dense but explained with remarks at each step.
- Value: ⭐⭐⭐⭐ Provides a reusable "confidence upgrade" for any work relying on LLM-as-a-judge + selective evaluation (evaluation platforms, RLHF reward model selection).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] YAQA: End-to-End KL Minimization LLM Adaptive Weight Quantization](model-preserving_adaptive_rounding.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICML 2026\] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)

</div>

<!-- RELATED:END -->
