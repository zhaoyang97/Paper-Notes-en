---
title: >-
  [Paper Note] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition
description: >-
  [ICML 2026][Image Generation][Offline MARL] OMSD replaces the "independent marginal regression" behavior constraint in traditional offline MARL with a "sequential conditional decomposition + per-agent conditional diffusi…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Offline MARL"
  - "Behavior Policy Multimodality"
  - "Sequential Decomposition"
  - "Diffusion Score"
  - "CTDE"
date: 2026-05-08
content_hash: 1236619c0ff66325
---

# Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition

**Conference**: ICML 2026  
**arXiv**: [2505.05968](https://arxiv.org/abs/2505.05968)  
**Code**: https://github.com/qiaodan-cuhk/OMSD  
**Area**: Multi-Agent Reinforcement Learning / Offline RL / Diffusion Models  
**Keywords**: Offline MARL, Behavior Policy Multimodality, Sequential Decomposition, Diffusion Score, CTDE  

## TL;DR
OMSD replaces the "independent marginal regression" behavior constraint in traditional offline MARL with a "sequential conditional decomposition + per-agent conditional diffusion model." By regularizing each agent's policy conditioned on the actions already selected by prefix agents, it avoids OOD mismatch caused by "marginal alignment but joint misalignment" under multimodal joint behavior distributions. It improves average rewards by +33% to +74% over SOTA on MPE and MaMuJoCo datasets.

## Background & Motivation
**Background**: Online MARL typically converges to a **single** Nash equilibrium through repeated interactions and coupled updates, resulting in low-entropy unimodal joint behavior policies. Prevailing offline MARL approaches either follow IGM with conservative value estimation (CQL/OMAR/CFCQL/OMIGA) or utilize "independent behavior regularization" and "centralized planners/world models" (AlberDICE, MOMA-PPO, MADiff).

**Limitations of Prior Work**: Offline datasets are collected from a mixture of **multiple** expert or suboptimal strategies, making the joint behavior distribution $\mu(\bm{a}|s)$ inherently **highly multimodal** (e.g., the dual-apple cooperative foraging task in Fig. 1). However, existing methods almost all assume $\mu(\bm{a}|s)=\prod_i \mu_i(a_i|s)$, regularizing each agent toward its own marginal distribution.

**Key Challenge**: The authors present a minimalist counterexample in Proposition 3.1 (Combinatorial Mode Shift, CMS): for an $n$-agent cooperative task with two modes $\bm{a}_1=(1,\dots,1)$ and $\bm{a}_2=(0,\dots,0)$, if each $\hat\mu_i$ is learned independently, the marginals converge to $\text{Uniform}(\{0,1\})$. The reconstructed joint distribution spreads across $2^n$ modes, leading to a TV distance from the true distribution of $\delta_{TV}=1-2^{1-n}\to 1$. In other words, "marginal alignment" under multimodality results in joint policies that are **almost entirely misaligned** with OOD joint actions never seen in the data.

**Goal**: To provide a behavior regularization direction for each agent that is **compatible with the joint multimodal structure** without explicitly modeling the entire joint policy or relying on centralized planners.

**Key Insight**: Utilize the probability chain rule to perform an **exact** decomposition: $\mu(\bm{a}|s)=\prod_{i=1}^n \mu_i(a_i|s, a_{<i})$. This subjects each agent's constraint to the **actions already selected by prefix agents**, representing an unbiased decomposition rather than an independent approximation.

**Core Idea**: Distill conditional scores $\nabla_{a_i}\log\mu_i(a_i|s,a_{<i})$ using **one conditional diffusion model per agent** as policy gradient regularization. This is combined with the action gradient of a centralized $Q^{tot}$ to achieve "sequential conditional + mode-aware" coordinated regularization.

## Method

### Overall Architecture
OMSD is an actor-critic framework under CTDE, where the behavior constraint of the actor is shifted from "marginal KL" to "sequential conditional score regularization," trained offline in two stages:

1. **Pretraining Stage**: A centralized IQL is used to learn a joint state-action value $Q^{tot}(s,\bm{a})$ on the offline data $\mathcal{D}$. Simultaneously, a conditional diffusion model $\epsilon_i^*(a_i|s, a_{<i}, t)$ is trained **in parallel** for each agent $i$, where the low-noise limit approximates the conditional score $-\epsilon_i^*/\sigma_t \approx \nabla_{a_i}\log\mu_i(a_i|s,a_{<i})$.
2. **Policy Optimization Stage**: The deterministic actor $\pi_{\theta_i}(s)$ of each agent is updated in a round-robin fashion. When updating agent $i$, the prefix actions $a_{<i}^{\mathrm{new}}$ are sampled from the currently updated $\{\pi_{j}^{\mathrm{new}}\}_{j<i}$ and used as conditions for the diffusion model. Suffix actions are used only for estimating the $Q^{tot}$ gradient without backpropagation.
3. **Execution Stage**: All agents execute $\pi_{\theta_i}(o_i)$ **concurrently and independently**. Sequential conditions are used only for training regularization and **do not violate** the decentralized execution assumption.

The key to the entire process is replacing "independent diffusion + independent regularization" (as in DOM2) with "conditional diffusion along the chain decomposition + conditional regularization," at the cost of only one additional prefix sampling during training.

### Key Designs

1. **Sequential Score Decomposition of Joint Behavior Policy**:
    - **Function**: Replaces the traditional marginal decomposition assumption $\mu(\bm{a}|s)=\prod_i\mu_i(a_i|s)$ with the **exact** probability chain decomposition $\mu(\bm{a}|s)=\prod_{i=1}^n \mu_i(a_i|s,a_{<i})$.
    - **Mechanism**: Models multi-agent coordination as a serial conditional distribution where "agent $i$ decides after seeing the choices of agents $1\ldots i-1$." Thus, the reference distribution in the KL regularization $D_{\mathrm{KL}}[\pi_{\theta_i}(\cdot|s)\|\mu_i(\cdot|s,a_{<i})]$ is naturally the "conditional distribution under the prefix mode" rather than a marginal that averages all modes, thereby eliminating CMS.
    - **Design Motivation**: Proposition 3.1 proves that independent decomposition causes the TV distance between the true and reconstructed distributions to tend toward 1 for a two-mode task with $n$ agents. Corollary 3.2 extends this to $K$ modes $\to K^n$ spurious modes. Chained decomposition is the **most cost-effective unbiased fix**, introducing only the "training-time sequence" without needing to model the full $\mu(\bm{a}|s)$.

2. **Per-Agent Conditional Diffusion Score Model + Low-Noise Score Distillation (SRPO style)**:
    - **Function**: Implicitly models complex $\mu_i(a_i|s,a_{<i})$ using diffusion models without using them for generative sampling, utilizing the denoising network as a **score estimator**.
    - **Mechanism**: Each agent trains a DDPM-style network $\epsilon_i(a_i^k, k| s, a_{<i})$ with the standard denoising loss $\mathcal{L}_{\text{denoise}}=\mathbb{E}\|\epsilon-\epsilon_i(a_i^k,k|s,a_{<i})\|^2$. During policy updates, the $t\to 0$ limit is taken, where $-\epsilon_i^*/\sigma_t$ approximates the conditional score $\nabla_{a_i}\log\mu_i$, which is added directly to the actor gradient (see the OMSD gradient formula in Eq. 6).
    - **Design Motivation**: (a) Under multimodal distributions, lightweight density estimators like GMM or normalizing flows suffer from mode collapse or poor mode coverage (see Appendix D.7.5). (b) Methods like Diff-QL/MADiff that use "diffusion models as actors" require dozens of denoising steps during execution, leading to high inference costs and accumulated sampling errors on low-quality data. OMSD uses diffusion models only as "gradient direction providers"; one forward pass equals one score, with no diffusion calls during inference.

3. **Centralized $Q^{tot}$ Gradient + Sequential Prefix Conditioning + Deterministic Actor**:
    - **Function**: Implements the method as a specific policy gradient—see Eq. (6): $\nabla_{\theta_i}\mathcal{L}^i = \mathbb{E}[\nabla_{a_i}Q_\phi(s,\bm{a}) + \frac{1}{\beta}\cdot(-\epsilon_i^*/\sigma_t)]\nabla_{\theta_i}\pi_{\theta_i}(s)$, where the prefix $a_{<i}^{\mathrm{new}}$ is sampled from recently updated policies, and suffixes follow old policies without gradient backpropagation.
    - **Mechanism**: The centralized $Q^{tot}$ learned via IQL provides the "direction for higher returns," while the conditional diffusion score provides the "mode support within the data." Their summation yields a **hill-climbing direction within a mode**. Policy updates roll through agent indices, with subsequent agents conditioned on the recently updated actions of preceding agents, pushing "coordination" into training. At execution, all agents act concurrently and independently.
    - **Design Motivation**: Why use deterministic rather than stochastic actors? Prefix sampling variance accumulates along the chain; stochastic actors would cause $a_{<i}$ to fluctuate, leading to score estimation noise explosion. Deterministic policies (DiLac) maintain expressivity while stabilizing prefix signals (Appendix D.7.1/D.7.4 show insensitivity to update order, indicating the sequential structure is a "training coordination mechanism" rather than a strong inductive bias).

### Loss & Training
- **Pretraining**: `centralized IQL` learns $Q^{tot}$; each agent trains $\epsilon_i$ independently via conditional denoising loss (Eq. 1). All diffusion models are **fully parallelizable**, decoupled from the number of agents $n$, and scalable to large teams (evaluated up to 6-HalfCheetah).
- **Policy Update**: Policies $\pi_{\theta_i}$ are updated cyclically according to agent index $i=1\to n$ using the gradient in Eq. (6). The regularization coefficient $\beta$ is a key hyperparameter—strong constraints (0.001) for expert data and weak constraints (0.3) for random data. Stability is observed over a wide range of $\beta$ (Fig. 4(b)).
- **Execution**: Each agent calls $\pi_{\theta_i}(o_i)$ independently **without calling** the diffusion model or relying on centralized communication.

## Key Experimental Results

### Main Results

| Dataset (MPE) | Metric (Normalized Score) | Prev. SOTA | OMSD | Gain |
|---|---|---|---|---|
| Cooperative Navigation - Random | Normalized | 62.2 (CFCQL) | **69.8** | +12.1% |
| Predator Prey - Medium | Normalized | 83.9 (DoF-P) | **137.1** | +63.0% |
| Predator Prey - Random | Normalized | 78.5 (CFCQL) | **133.9** | +70.6% |
| World - Random | Normalized | 68 (CFCQL) | **141.1** | +107.5% |
| MPE Average | Normalized | 87.3 (CFCQL) | **126.7** | +33.2% |

| Dataset (MaMuJoCo / OMIGA) | Prev. SOTA | OMSD | Gain |
|---|---|---|---|
| 3-Hopper - Expert | 859.6 (OMIGA) | **3595** | +329% |
| 3-Hopper - Medium-Expert | 709.0 (OMIGA) | **3568** | +403% |
| 3-Hopper - Medium | 1189.3 (OMIGA) | **3360** | +183% |
| 6-HalfCheetah - Medium-Replay | 2504.7 (OMIGA) | **4582** | +83% |
| OMIGA Average | 1954.7 | **3400** | +73.9% |

**Key Finding**: The more severe the multimodality in the dataset (Medium / Medium-Replay / Random), the larger the improvement of OMSD. On Expert data (closer to unimodal), the gain is relatively smaller, which supports the claim that **CMS is the core problem to be solved.**

### Ablation Study

| Configuration | MPE Average / Observation | Description |
|---|---|---|
| BRPO-IND (Indep. Learning + Indep. KL) | Often trapped in $[1,-1]$/$[-1,1]$ OOD joint actions in 2-agent bandit; score $0\pm 1$. | CMS fails even on minimalist examples. |
| BRPO-CTDE (Central Critic + Indep. KL) | Similarly $0\pm 1$. | Centralized critic cannot save marginal regularization. |
| BRPO-JAL (Joint Action Learning, Oracle) | $1\pm 0$. | Performance upper bound. |
| **OMSD** | **$1\pm 0$** (Comprehensive SOTA on high-dim tasks). | Single-agent chained conditioning matches the JAL upper bound. |
| OMSD w/ different update orders (Hopper) | No significant performance difference. | Sequential structure is a training mechanism, not a strong inductive bias. |
| Score Estimator: GMM / Flow vs. Diffusion | Diffusion significantly outperforms lightweight estimators (App. D.7.5). | Diffusion mode coverage is necessary under multimodality. |
| $\beta$ Sweep (Fig. 4(b)) | Expert prefers $\beta=0.001$, Random prefers $\beta=0.3$. | Behavior constraint strength couples with data quality. |

### Key Findings
- **The exponential scaling of CMS severity with agent count is empirically visible**: BRPO-IND fails on a simple bandit; on high-dimensional tasks, independent diffusion actors (DOM2 approach) are significantly outperformed by OMSD on medium/random datasets due to OOD joint actions caused by $K^n$ spurious modes.
- **OMSD shows the most significant gains on low-quality data** (+107.5% for World Random, +183~403% for OMIGA Hopper), as multimodality is more pronounced in low-quality data (Fig. 1d), making CMS costlier and its fix more valuable.
- **Honest failure case diagnosis**: The authors state that Ours is slightly outperformed by MADiff on 2-Ant Good / 4-Ant Good, not due to decomposition errors, but because the **pretrained critic is insufficient**—while the diffusion model aligns modes, the critic fails to provide strong enough improvement directions.
- **t-SNE Visualizations (Fig. 4(c))** show that OMSD-learned $(s,a)$ pairs evolve within high-reward regions supported by the data, whereas independent regularization methods drift into sparse data regions.
- **Scalability**: Diffusion model pretraining is fully agent-parallel. Sequential dependency only appears during policy updates via prefix actions, and **execution remains concurrent**, as evidenced by results on 6-agent HalfCheetah.

## Highlights & Insights
- **Challenging the implicitly accepted "independent decomposition" assumption** with a closed-form counterexample (Prop 3.1). The paper's greatest contribution is not just the algorithm, but identifying the root cause of why offline MARL is harder than online as "online symmetry breaking vs. offline multi-source multimodal data." This perspective facilitates diagnosing tasks like offline multi-agent IL or coordination.
- **Diffusion models as score estimators rather than generators** is a highly reusable engineering paradigm. Inference costs a single forward pass, bypassing slow inference and error accumulation in diffusion actors. This is suitable for any scenario requiring complex distribution modeling without the generative overhead.
- **Sequential decomposition during training, concurrent execution**: Coordination is fully handled during the training phase, maintaining the purity of decentralized execution in CTDE. This decoupling can be applied to coordinated planning or multi-agent imitation.
- **Inverse correlation between $\beta$ and data quality**: (Strong constraints for experts, weak for random data) provides simple, actionable tuning intuition applicable to all BRPO-like methods.

## Limitations & Future Work
- **Robustness to update order was only verified on small-scale tasks** (OMIGA Hopper). Whether "order doesn't matter" holds when the number of agents is much larger than 6 requires further large-scale experiments; in extreme cases, incorrect ordering might amplify prefix sampling variance.
- **Reliance on deterministic actors (DiLac)**: The paper admits that stochastic policies might cause prefix noise to accumulate along the chain, potentially limiting performance in tasks requiring exploratory stochastic policies (e.g., POMDPs).
- **Dependency on IQL critic quality**: Sec. 4.2 points out that underperforming against MADiff on some Good datasets is due to a weak critic—meaning OMSD maximizes mode alignment, but improvement depends on the critic component.
- **Implicit "role asymmetry"**: Chained conditioning implies that prefix agents take on a stronger "prior setter" role. The theoretical properties of this asymmetry are not yet fully analyzed.
- **Future Directions**: (a) Importance-weighted multi-order ensembles to remove explicit sequential structures; (b) Replacing $Q^{tot}$ with a diffusion critic for joint direction estimation; (c) Extension to POMDP or asynchronous execution scenarios.

## Related Work & Insights
- **vs. AlberDICE / OMIGA / CFCQL** (Independent Regularization + Conservative Values): This paper fundamentally argues they suffer from CMS—no matter how conservative the critic is, it cannot fix the "marginal alignment $\to$ joint misalignment" directional error. OMSD achieves massive gains by merely changing the reference distribution.
- **vs. MOMA-PPO / MADiff-C** (Centralized Planners / World Models): These require modeling joint policies or environments and running planners during inference, which accumulates errors on low-quality data. OMSD avoids joint models and planning entirely.
- **vs. DOM2 / MADiff-D** (Independent Diffusion Actors): These assume independent decomposition + diffusion marginals, remaining victims of CMS. OMSD uses the same diffusion route but corrects the decomposition to sequential conditioning.
- **vs. SRPO (single-agent)**: OMSD can be seen as a multi-agent extension of SRPO—inheriting the "diffusion as score estimator + SRPO-style policy gradient" and extending single-agent scores to multi-agent chained scores.

## Rating
- **Novelty**: ⭐⭐⭐⭐ "Fixing CMS with chained decomposition" is a clean and effective insight. Consolidating problem diagnosis, counterexamples, and diffusion score applications into a single method is a refreshing perspective in an area dominated by independent decomposition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers MPE, MaMuJoCo, bandit toys, order robustness, density estimators, $\beta$ sensitivity, and t-SNE. Reporting negative results (Expert MaMuJoCo) and diagnosing them is commendable.
- **Writing Quality**: ⭐⭐⭐⭐ The closed-form CMS in Prop 3.1 is highly educational. Figures 1 and 2 clarify the pitfalls of independent regularization. The derivation in Sec 3.3 is somewhat dense and requires the appendix for full clarity.
- **Value**: ⭐⭐⭐⭐ Provides a "theoretical diagnosis + plug-and-play fix" paradigm for offline MARL. Open-sourcing the code makes it a strong baseline for any work currently using independent behavior regularization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2026\] Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)
- [\[ICML 2026\] CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)
- [\[ICML 2026\] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)

</div>

<!-- RELATED:END -->
