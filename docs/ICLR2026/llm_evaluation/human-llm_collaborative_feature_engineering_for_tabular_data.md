---
title: >-
  [Paper Note] Human-LLM Collaborative Feature Engineering for Tabular Learning
description: >-
  [ICLR 2026][LLM Evaluation][Feature Engineering] This paper proposes a human-LLM collaborative feature engineering framework that decouples the proposal and selection of feature operations. A Bayesian neural network models operation utility and uncertainty to guide selection, with selective human preference feedback incorporated when appropriate. The framework achieves 8.96%–11.23% average error rate reduction across 18 tabular datasets.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Feature Engineering
  - Human-AI Collaboration
  - Bayesian Optimization
  - LLM
  - Tabular Data
date: 2026-05-08
content_hash: 6f2accff329c03fc
---

# Human-LLM Collaborative Feature Engineering for Tabular Learning

**Conference**: ICLR 2026
**arXiv**: [2601.21060](https://arxiv.org/abs/2601.21060)
**Code**: None
**Area**: AutoML / Tabular Learning
**Keywords**: Feature Engineering, Human-AI Collaboration, Bayesian Optimization, LLM, Tabular Data

## TL;DR

This paper proposes a human-LLM collaborative feature engineering framework that decouples the proposal and selection of feature operations. A Bayesian neural network models operation utility and uncertainty to guide selection, with selective human preference feedback incorporated when appropriate. The framework achieves 8.96%–11.23% average error rate reduction across 18 tabular datasets.

## Background & Motivation

**Background**: LLMs are widely adopted in tabular learning for automated feature engineering, leveraging semantic understanding to generate meaningful feature transformation operations (e.g., CAAFE, OCTree).

**Limitations of Prior Work**: Existing methods use LLMs simultaneously as both proposers and selectors of feature operations, relying entirely on LLM-internal heuristics without calibrated estimates of operation utility and uncertainty. This leads to repeated exploration of low-gain operations and poor performance under limited iteration budgets.

**Key Challenge**: LLMs excel at generating diverse candidate feature transformations but are ill-suited for making optimal selections among them — a fundamental tension between strong proposal ability and weak selection ability.

**Goal**: To decouple LLM-based operation proposal from selection, and to effectively integrate human expert knowledge into the selection process to improve feature engineering efficiency.

**Key Insight**: Drawing on Bayesian optimization, an explicit surrogate model replaces the LLM's implicit selection, and a selective human feedback mechanism is designed to control the cost of expert involvement.

**Core Idea**: The LLM is responsible solely for proposing candidate feature operations; selection is guided by a UCB strategy over a Bayesian neural network, with human preference feedback selectively queried under high uncertainty.

## Method

### Overall Architecture

In each feature engineering round: (1) the LLM generates $N$ candidate feature transformation operations based on task descriptions, feature semantics, and historical performance; (2) a Bayesian neural network (BNN) surrogate estimates the expected utility $\mu_t(e)$ and uncertainty $\sigma_t^2(e)$ for each operation; (3) a UCB strategy selects operations and selectively queries human preference feedback when conditions are met; (4) the actual utility of the selected operation is evaluated and the surrogate model is updated.

### Key Designs

1. **Feature Operation Encoding and BNN Surrogate**:
    - **Function**: Maps LLM-generated natural language feature operations to vector representations and estimates utility via a Bayesian neural network.
    - **Mechanism**: Operation embeddings are formed by concatenating semantic embeddings $\phi_{\text{embedding}}(e)$ (text-embedding-3-small) and column usage encodings $\phi_{\text{column}}(e) \in \{0,1\}^d$. The BNN learns a variational posterior $q_t(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{M}_t, \boldsymbol{\Sigma}_t)$ via variational inference, yielding predictive mean $\mu_t(e)$ and variance $\sigma_t^2(e)$.
    - **Design Motivation**: Gaussian processes scale poorly in the high-dimensional language-derived feature space and are less suited for modeling non-stationarity; BNNs address this limitation. Column usage encodings resolve ambiguity when multiple columns have similar semantic descriptions.

2. **Selective Human Preference Feedback Mechanism**:
    - **Function**: After UCB selects the best candidate $e_t^a$, determines whether to query a human expert for preference feedback.
    - **Mechanism**: Querying is triggered only when two conditions are simultaneously satisfied — (C1) confidence interval overlap: $\text{UCB}_t(e_t^b) > \text{LCB}_t(e_t^a)$, ensuring non-trivial uncertainty; (C2) sufficiently large uncertainty: $\sqrt{\beta_t}(\sigma_t(e_t^a) + \sigma_t(e_t^b)) \geq \gamma_\kappa$, ensuring potential gain exceeds query cost.
    - **Design Motivation**: Indiscriminate querying imposes unnecessary cognitive burden; human intervention is solicited only when feedback can yield a significant utility gain.

3. **Posterior Update via Preference Feedback**:
    - **Function**: Incorporates human preference feedback $Z_t$ into the surrogate model's posterior distribution.
    - **Mechanism**: Preference feedback is modeled via a probit likelihood $\mathcal{P}(Z_t | \boldsymbol{\theta}, e_t^a, e_t^b) = \Phi(\eta Z_t [\hat{g}(\phi(e_t^a); \boldsymbol{\theta}) - \hat{g}(\phi(e_t^b); \boldsymbol{\theta})])$; the variational posterior $q_t'(\boldsymbol{\theta})$ is updated and the updated UCB values are used for final selection.
    - **Design Motivation**: Probabilistic treatment of human feedback is more robust than direct adoption, smoothing the effect of noisy responses.

### Loss & Training

The BNN is trained by minimizing the ELBO: $\text{KL}(q_t(\boldsymbol{\theta}) \| \mathcal{P}(\boldsymbol{\theta})) - \mathbb{E}_{q_t(\boldsymbol{\theta})}[\log \mathcal{P}(H_t | \boldsymbol{\theta})]$. The UCB exploration coefficient is $\beta_t = 2\log(|\mathcal{S}_t|\pi^2 t^2 / 3\delta)$ with $\delta=0.1$. The human query cost threshold is $\gamma_\kappa=4$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (w/o human) | Ours (w/ human) | Best Baseline | Gain (w/ human) |
|--------|------|------|------|------|------|
| 13 classification datasets (MLP) | AUROC (%) | 85.3 | 85.5 | 84.7 (OCTree) | Error rate ↓ 8.96% |
| 13 classification datasets (XGBoost) | AUROC (%) | 87.4 | 87.4 | 86.7 (OCTree) | Error rate ↓ 11.23% |
| flight (MLP) | AUROC (%) | 96.9 | 97.3 | 94.8 (OCTree) | Error rate ↓ +48.1% |
| conversion (XGBoost) | AUROC (%) | 93.5 | 93.9 | 92.4 (OCTree) | Error rate ↓ +11.5% |

### Ablation Study

| Configuration | Metric | Notes |
|------|------|------|
| Different LLM backbone (GPT-5) | MLP avg. 85.9→86.5 | Ours (w/ human) achieves best performance under GPT-5 backbone |
| Different LLM backbone (GPT-3.5) | MLP avg. 84.6→85.1 | Advantage is maintained even with a weaker backbone |
| User study (ALG vs. Control) | Performance: p=0.011 | ALG framework significantly improves user performance |
| User study (ALG vs. Control) | Completion time: p<0.001 | ALG framework significantly reduces completion time |

### Key Findings

- LLM-based methods consistently outperform traditional AutoML approaches (OpenFE, AutoGluon), validating the value of semantic understanding for feature engineering.
- Explicitly modeling utility and uncertainty outperforms relying purely on LLM heuristics by 7.24% and 9.02% error rate reduction, respectively.
- Human preference feedback consistently provides additional gains, while the computational overhead of BNN+UCB accounts for only 2.2% of total runtime.

## Highlights & Insights

- Introducing Bayesian optimization principles into LLM-driven feature engineering and decoupling proposal from selection is an elegant system design. The theoretical guarantees of UCB for exploration–exploitation trade-offs render the selection process transparent and principled.
- The two conditions of the selective querying mechanism (confidence interval overlap + uncertainty gating) are grounded in solid theoretical analysis (Lemma 3.1–3.2), achieving an optimal trade-off between human cognitive cost and information gain.

## Limitations & Future Work

- Human feedback in experiments is simulated by GPT-4o; the actual user study is conducted on only a single dataset, limiting generalizability.
- BNN surrogate calibration quality may be poor in early rounds with sparse data; the cold-start problem is not sufficiently discussed.
- The framework models only single-operation utility and does not capture interaction effects among combinations of multiple operations.

## Related Work & Insights

- **vs. CAAFE**: CAAFE uses the LLM for both proposing and selecting feature operations, making it prone to local optima; the proposed decoupling enables continuous discovery of high-value operations.
- **vs. OCTree**: OCTree uses decision tree feedback to guide the LLM but still relies on LLM-internal heuristics for selection; this work employs a BNN to provide better-calibrated utility estimates.
- **vs. Traditional Bayesian Optimization**: Conventional BO uses Gaussian processes as surrogate models, which are effective in low-dimensional spaces; this work applies BNNs to handle high-dimensional language embedding spaces.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework design of decoupling proposal and selection with integrated human feedback is novel, supported by complete theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across 18 datasets with user studies and computational scalability analysis from multiple perspectives.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear, theoretical derivations are rigorous, and experimental presentation is comprehensive.
- **Value**: ⭐⭐⭐ The practical application scenario is well-defined, though LLM API costs are required; the method demonstrates good generality.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_with_the_hu.md)
- [\[AAAI 2026\] MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](../../AAAI2026/llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)

<!-- RELATED:END -->
