---
title: >-
  [Paper Note] LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems
description: >-
  [ICML 2026][Self-Supervised Learning][Selective prediction] Addressing the long-standing problem in LLM selective prediction where "UCB risk bounds are too conservative and yield few usable thresholds…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Selective prediction"
  - "risk control"
  - "conformal prediction"
  - "model routing"
  - "uncertainty quantification"
date: 2026-05-08
content_hash: 6f2c04bcc2710f84
---

# LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems

**Conference**: ICML 2026  
**arXiv**: [2512.01556](https://arxiv.org/abs/2512.01556)  
**Code**: The caption for Figure 1 in the paper states "Code is available here" (open-source link not provided in the main text)  
**Area**: AI Safety / Selective Prediction / Uncertainty Quantification  
**Keywords**: Selective prediction, risk control, conformal prediction, model routing, uncertainty quantification

## TL;DR
Addressing the long-standing problem in LLM selective prediction where "UCB risk bounds are too conservative and yield few usable thresholds," the authors reformulate the objective "post-selection error rate $\le \alpha$" into a **linear expectation constraint** involving two 0-1 indicator functions for selection and error. This derivation leads to a finite-sample sufficient condition (Eq. 5) that depends only on the calibration set. It maintains rigorous finite-sample guarantees while being significantly tighter than UCB. The framework naturally extends to two-model routing systems by jointly calibrating two thresholds, showing universal gains in power on CommonsenseQA / TriviaQA / ScienceQA / MM-Vet v2, and accepting 9.5% more samples than Clopper-Pearson UCB on TriviaQA.

## Background & Motivation

**Background**: LLMs/LVLMs are increasingly embedded into decision-making pipelines. However, they can hallucinate and express high confidence in incorrect answers. Thus, statistical guarantees for "accept / abstain / escalate" behaviors are required. Split conformal prediction (SCP) can transform heuristic uncertainty scores into prediction sets with coverage guarantees, but **set-valued outputs** are often not directly actionable for downstream decisions—they frequently contain unreliable candidates, leading to biased decision-making.

**Limitations of Prior Work**: Researchers have turned to the "point prediction + selective acceptance" paradigm: accept only when uncertainty $u \le \lambda$. The challenge lies in calibrating $\lambda$ to guarantee that the "error rate of accepted samples is $\le \alpha$." Current state-of-the-art methods rely on confidence interval-based UCB approaches—COIN uses the Hoeffding-based UCB-HFD, while Trust of Escalate uses the Clopper-Pearson exact UCB (UCB-CLP). These methods are **statistically valid but extremely conservative**: they apply worst-case tail control to empirical risk, causing actual acceptance rates to fall far below what the risk budget allows, sometimes failing to find any feasible threshold at low $\alpha$ (e.g., 0.05).

**Key Challenge**: UCB-based methods control the "upper bound of empirical risk," whereas the true objective is to control the "selection-conditioned empirical risk $\mathrm{SCER}(\lambda) = \Pr(\mathrm{err}=1 \mid S(\lambda)=1)$," which is a ratio. Forcing a "ratio constraint" into an "upper bound constraint" inevitably introduces excessive conservative padding.

**Goal**: (1) Identify a threshold calibration formula that preserves finite-sample guarantees while being tighter than UCB; (2) Generalize these guarantees from single models to two-model routing (primary $\to$ secondary $\to$ abstain) systems, achieving **system-level** rather than separate risk control.

**Key Insight**: The authors' key observation is that the ratio constraint $\mathbb{E}[Z]/\mathbb{E}[S] \le \alpha$ (where $Z = S \cdot \mathrm{err}$ is the joint indicator for "accepted and incorrect" and $S$ is the acceptance indicator) is equivalent to a **linear constraint** $\mathbb{E}[Z - \alpha S] \le 0$, provided $\mathbb{E}[S] > 0$. The advantage of this linear constraint is that it only requires the expectation of a single random variable $Z-\alpha S$ to be non-positive, eliminating the need for separate tail control on $Z$ and $S$. It is inherently tighter than "dividing the UCB of risk by the acceptance rate."

**Core Idea**: Frame selective prediction as "solving for a threshold under linear expectation constraints" rather than "sorting by uncertainty." Using a leave-one-out correction under the exchangeability assumption, the authors derive a clean "sum of differences $\le -1$" finite-sample sufficient condition. This single inequality provides calibration rules for both single-model and routing systems.

## Method

### Overall Architecture
Single-model LEC follows 4 steps: (1) Model $\mathcal{G}^{(a)}$ computes uncertainty $u_i$ and error indicators $\mathrm{err}_i$ on a calibration set $\mathcal{D}_{\mathrm{cal}}=\{(u_i^{(a)},\mathrm{err}_i^{(a)})\}_{i=1}^n$; (2) The acceptance count $k(\lambda)=\#\{i: u_i \le \lambda\}$ for a candidate threshold $\lambda$ is substituted into the finite-sample sufficient condition $\sum_{j=1}^{k(\lambda)}(\mathrm{err}_{(j)} - \alpha) \le -1$ (sorted by $u_i$ ascending); (3) The **largest** feasible $\hat{\lambda}$ is selected to maximize the test-time acceptance rate; (4) During testing, a new sample is accepted if $u_{n+1} \le \hat{\lambda}$, otherwise the system abstains. Two-model routing generalizes this principle to a joint search for $(\lambda^{(a)}, \lambda^{(b)})$, selecting the optimal pair by maximizing system-level acceptance.

### Key Designs

1.  **From Ratio Constraints to Linear Expectation Constraints**:
    - **Function**: Equivalent reformulation of the objective "$\Pr(\mathrm{err}=1 \mid S(\lambda)=1) \le \alpha$" (a conditional probability constraint) into a linear expectation inequality to facilitate the construction of tight finite-sample conditions.
    - **Mechanism**: Define the joint indicator $Z(\lambda) = S(\lambda) \cdot \mathrm{err}$ (set to 1 only if accepted and incorrect). Then $\mathrm{SCER}(\lambda) = \mathbb{E}[Z(\lambda)] / \mathbb{E}[S(\lambda)]$. Given $\mathbb{E}[S(\lambda)] > 0$, $\mathrm{SCER}(\lambda) \le \alpha \Leftrightarrow \mathbb{E}[Z(\lambda) - \alpha S(\lambda)] \le 0$. Intuitively, $Z - \alpha S$ represents the "marginal contribution of a sample to the number of accepted errors minus $\alpha$ times the marginal acceptance."
    - **Design Motivation**: UCB-CLP / UCB-HFD separately bound $\mathbb{E}[Z]$ and $\mathbb{E}[S]$, stacking two conservative bounds. By evaluating the non-positivity of the expectation of the combined term $Z - \alpha S$, this method requires only a single "sum of differences" correction, eliminating dual conservatism at its source.

2.  **Finite-Sample Sufficient Condition + Feasible Threshold Set**:
    - **Function**: Translates "$\mathbb{E}[Z - \alpha S] \le 0$" into a finite-sample criterion verifiable using only the calibration set.
    - **Mechanism**: Sort $u_i$ to get $u_{(1)} \le \dots \le u_{(n)}$ and corresponding $\mathrm{err}_{(j)}$. For a candidate $\lambda$, define $k(\lambda) = \#\{i: u_i \le \lambda\}$. Using the standard leave-one-out correction in distribution-free calibration, the authors prove (Appendix A.1) that under exchangeability, the following is a sufficient condition: $\sum_{j=1}^{k(\lambda)} (\mathrm{err}_{(j)} - \alpha) \le -1$. The feasible set is $\Lambda_\alpha = \{\lambda: \text{inequality holds}\}$, and the calibrated threshold is $\hat{\lambda} = \sup \Lambda_\alpha$. If $\Lambda_\alpha = \varnothing$, the $\alpha$ is declared infeasible. Theorem 3.1 proves that with $\hat{\lambda}$, a new sample satisfies $\Pr(\mathrm{err}_{n+1}=1 \mid u_{n+1} \le \hat{\lambda}) \le \alpha$ (marginal guarantee over calibration and test randomness).
    - **Design Motivation**: While "$\sum (\mathrm{err}_{(j)} - \alpha) \le -1$" appears simple, it differs fundamentally from UCB. It directly utilizes the summation of $Z - \alpha S$ on the calibration set and uses $-1$ as a leave-one-out correction (replacing worst-case tail bounds), preserving finite-sample rigor while avoiding the over-conservatism of Hoeffding/Clopper-Pearson.

3.  **Joint Threshold Calibration for Two-Model Routing**:
    - **Function**: When a single model is infeasible or its acceptance rate is too low for a given $\alpha$, uncertain inputs are escalated to a second model while ensuring **system-level** (not separate) SCER $\le \alpha$.
    - **Mechanism**: Define $S^{(b)}(\lambda^{(a)}, \lambda^{(b)}) = \mathbf{1}\{u^{(a)} > \lambda^{(a)} \land u^{(b)} \le \lambda^{(b)}\}$. System-level acceptance is $S = S^{(a)} + S^{(b)} \in \{0,1\}$, and system-level error is $Z = S^{(a)} \mathrm{err}^{(a)} + S^{(b)} \mathrm{err}^{(b)}$. The same linear equivalence yield the system constraint $\mathbb{E}[Z - \alpha S] \le 0$, with the finite-sample condition $\sum_{i=1}^n (Z_i(\lambda^{(a)}, \lambda^{(b)}) - \alpha S_i(\lambda^{(a)}, \lambda^{(b)})) \le -1$. The pair $(\hat{\lambda}^{(a)}, \hat{\lambda}^{(b)})$ is chosen from the feasible set $\Lambda^{(a,b)}_\alpha$ to maximize the empirical acceptance rate $\frac{1}{n}\sum S_i$. Theorem 3.2 proves system-level SCER $\le \alpha$. The framework extends naturally to $K$-model chains.
    - **Design Motivation**: Independent calibration of $\lambda^{(a)}$ and $\lambda^{(b)}$ ("naive LEC") fails because the secondary model sees a sub-population rejected by the primary model, violating the exchangeability assumption. Joint calibration is the **only correct path to valid system-level guarantees** for routing systems.

### Loss & Training
LEC is a **purely post-processing calibration method** and involves no gradient training. It requires: (1) A pre-trained model $\mathcal{G}$; (2) A scalar uncertainty function $\mathcal{U}$ (predictive entropy PE for closed QA/VQA, black-box semantic entropy SE for open-ended, or others like EigV / Deg / Ecc / SELF); (3) A labeled calibration set; (4) An admission function $A$ (defaulting to a sentence similarity threshold of 0.6). Computational overhead consists primarily of scanning $\lambda$ candidates ($\mathcal{O}(n)$ per candidate).

## Key Experimental Results

### Main Results

Comparison of Power (proportion of accepted correct samples, higher is better) across various risk levels $\alpha$ for 8 LLMs on the TriviaQA dataset. LEC consistently matches or outperforms UCB-CLP, with the most significant advantages in "low risk budget" scenarios like $\alpha=0.05 / 0.1$ (mean over 500 splits):

| α | OpenChat-3.5 UCB-CLP | OpenChat-3.5 LEC | Qwen2.5-14B UCB-CLP | Qwen2.5-14B LEC | LLaMA-3.1-8B UCB-CLP | LLaMA-3.1-8B LEC | LLaMA-3.1-70B UCB-CLP | LLaMA-3.1-70B LEC |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
| 0.05 | 0.6684 | **0.7230** | 0.6240 | **0.7193** | 0.7143 | **0.7538** | 0.9935 | **0.9996** |
| 0.10 | 0.9294 | **0.9521** | 0.9987 | **1.0000** | 0.9396 | **0.9612** | 1.0 | 1.0 |
| 0.15 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

UCB-HFD (Hoeffding variant) often returns "no feasible threshold" at $\alpha=0.05$ for several models, highlighting UCB's fragility in low $\alpha$ regions.

### Ablation Study

Comparison of "number of correct samples" accepted by a two-model routing system (Qwen2.5-3B as primary, LLaMA-3.1-8B as secondary) on CommonsenseQA:

| α | Qwen2.5-3B Single | LLaMA-3.1-8B Single | LEC-Routing (Qwen2.5-3B & LLaMA-3.1-8B) |
|------|--------|---------|----------|
| 0.05 | 965  | 1579 | **1610** |
| 0.10 | 2569 | 2357 | **2663** |
| 0.15 | 3174 | 2890 | 3174+ (System risk remains valid) |

At $\alpha=0.05$, using Qwen2.5-3B alone results in a 20.3% acceptance rate, whereas LEC-Routing increases this to 33.9% (17.44% by primary, 16.46% by secondary) for a 13.6% absolute gain. Figure 6 shows LEC-Routing's empirical SCER stays close to but under $\alpha$, while "naive LEC" (independent calibration) violates $\alpha$, proving the **necessity of joint calibration**.

### Key Findings
- **Statistical Validity**: Across 8 LLMs, multiple $\alpha$, and 500 random splits, the mean empirical SCER remains strictly below but near $\alpha$ (e.g., 0.0497 for $\alpha=0.05$), confirming Theorem 3.1 holds even in finite samples.
- **Tighter but Valid**: Compared to UCB-CLP, LEC utilizes the "risk budget" closer to the $\alpha$ boundary, accepting more samples without violating safety constraints.
- **Gain for TriviaQA + Qwen2.5-14B**: LEC accepts 9.5% more samples than the already-tightest UCB-CLP at $\alpha=0.05$.
- **Joint vs. Naive Routing**: Independent calibration causes SCER overflows, whereas LEC-Routing joint calibration remains valid while boosting coverage.
- **Robustness**: Performance gains are consistent across different UQ methods (SE/EigV/Ecc), calibration-test split ratios, and sampling counts.

## Highlights & Insights
- **The "Ratio $\to$ Linear" shift is the soul of the paper**: This seemingly simple reformulation collapses two conservative bounds into a single difference inequality, which is the root of LEC's tightness compared to UCB.
- **The condition "$\sum (\mathrm{err} - \alpha) \le -1$" is elegant and actionable**: Searching for a feasible $k$ has $\mathcal{O}(n)$ complexity, making it simpler to implement than quantile algorithms or Clopper-Pearson inversions.
- **Unified Routing Control**: The single-model and routing logic share the same mathematical structure. Extending the "linear expectation constraint + leave-one-out correction" to $K$-model chains is seamless.
- **Black-box Friendly**: LEC requires only uncertainty scores and correctness labels, making it applicable to closed-source APIs like GPT-4 or Gemini with minimal barriers to deployment.

## Limitations & Future Work
- **Exchangeability Dependency**: Guarantees fail if the distribution shifts (e.g., user queries change over time). Future work could integrate weighted or online conformal methods.
- **Admission Function Noise**: Using sentence similarity thresholds as "ground truth" labels for open-ended tasks is noisy. LEC controls SCER relative to the noisy admission function, not necessarily the absolute semantic truth.
- **Calibration Size vs. $\alpha$**: At very low $\alpha$ (e.g., 0.01), the condition is difficult to satisfy, necessitating larger calibration sets. Theoretical characterization of minimum sample sizes is missing.
- **Routing Search Complexity**: Joint threshold search is $\mathcal{O}(n^2)$ for two models and $\mathcal{O}(n^K)$ for $K$ models; long chains require approximation or pruning.

## Related Work & Insights
- **vs. COIN (UCB-HFD)**: COIN uses Hoeffding inequalities which are shown here to be excessively conservative for low $\alpha$ budgets.
- **vs. UCB-CLP / Trust of Escalate**: Even though Clopper-Pearson is the tightest UCB, it still applies conservative bounds in two steps. LEC's single sum-of-differences inequality is more efficient.
- **vs. Conformal Alignment / Labeling**: Those frameworks focus on false discovery control in multiple-testing, whereas LEC addresses marginal risk in single-point acceptance—targeting similar goals through different mathematical forms.
- **Inspiration**: The linear expectation constraint approach can be transferred to other "ratio constraint" scenarios, such as demographic parity in fairness or tool-calling error rates in agent systems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of the "ratio $\to$ linear" reformulation and leave-one-out correction is a clear paradigm upgrade over UCB methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive coverage across 12 models (LLM/LVLM), 6 UQ methods, multiple benchmarks, and 500 splits.
- **Writing Quality**: ⭐⭐⭐⭐ The progression from single-model to routing is clear and mathematically sound.
- **Value**: ⭐⭐⭐⭐⭐ As a post-processing, black-box-ready method with rigorous guarantees, it has high practical utility for LLM deployment decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[ICML 2026\] NITP: Next Implicit Token Prediction for LLM Pre-training](nitp_next_implicit_token_prediction_for_llm_pre-training.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2026/self_supervised/representation_learning_for_spatiotemporal_physica.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)

</div>

<!-- RELATED:END -->
