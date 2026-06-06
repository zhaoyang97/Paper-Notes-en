---
title: >-
  [Paper Note] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling
description: >-
  [ICML 2026][LLM Alignment][Reward modeling] This paper rewrites the Bradley–Terry reward model as a generative process based on Bayesian Non-negative Factor Analysis (NFA). By simultaneously modeling locally sparse insta…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Reward modeling"
  - "Reward hacking"
  - "Bayesian Non-negative Factor Analysis"
  - "Variational Inference"
  - "Weibull Distribution"
date: 2026-05-08
content_hash: 76765cc0e3182a12
---

# Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.10623](https://arxiv.org/abs/2602.10623)  
**Code**: https://github.com/GuoweiRong/Bayesian-Non-negative-Reward-Model (Available)  
**Area**: Alignment RLHF  
**Keywords**: Reward modeling, Reward hacking, Bayesian Non-negative Factor Analysis, Variational Inference, Weibull Distribution

## TL;DR
This paper rewrites the Bradley–Terry reward model as a generative process based on Bayesian Non-negative Factor Analysis (NFA). By simultaneously modeling locally sparse instance latent variables $\bm{\theta}$ and a globally sparse reward dictionary $\Phi$, the framework suppresses reward hacking caused by shortcut features (e.g., length, style) through a "disentanglement-then-debiasing" approach. Amortized variational inference with Weibull reparameterization allows the framework to be integrated into modern LLM backbones, consistently outperforming strong baselines like BT, Ensemble, and InfoRM on Unified-Feedback, RewardBench, HHH, and MT-Bench.

## Background & Motivation

**Background**: RLHF has become the mainstream paradigm for LLM alignment. Its core involves distilling human pairwise preferences into a differentiable Reward Model (RM), which is then used to optimize policies via RL algorithms like PPO. The standard practice involves adding a linear head $W_{\text{bt}}$ after an LLM backbone and training it with the Bradley–Terry ranking loss to output a deterministic scalar score for a response.

**Limitations of Prior Work**: Such RMs are highly susceptible to "hacking" in practice—the policy learns to optimize proxy rewards rather than true human goals. The most common shortcuts include surface cues like response length, phrasing style, and templated expressions. The root cause is reward misgeneralization: RMs extrapolate poorly outside the training distribution, while deep networks inherently favor shortcut features.

**Key Challenge**: Existing mitigation methods have various side effects. Ensemble methods (e.g., BT-Ensemble, ENS) require maintaining multiple large models, doubling computational costs. Information bottleneck methods (e.g., InfoRM) can only implicitly suppress irrelevant features and fail to explicitly separate "semantic intent" from "style noise." Supervised correction methods for specific biases like length only work in narrow, predefined scenarios. The problem lies in the scalar reward $r=\bm{z}^{\top}W_{\text{bt}}$ itself, which is dense, black-box, and deterministic, leaving no structural room for "uncertainty" or "sparse factors."

**Goal**: (1) Introduce explicit uncertainty modeling to the RM, covering both aleatoric uncertainty from human labeling and epistemic uncertainty of reward parameters; (2) Enable reward representations with a sparse, interpretable structure to mechanistically suppress reliance on spurious correlations; (3) Scale the solution to 8B-level LLMs without relying on multi-model ensembles.

**Key Insight**: The authors revisit sparse-aware Bayesian models (SBMs), particularly Non-negative Factor Analysis (NFA, such as PFA and LDA). NFA naturally possesses three RM-friendly properties: probabilistic latent variables with inherent uncertainty, non-negative sparse priors acting as implicit regularization, and non-negative basis vectors providing parts-based interpretable representations.

**Core Idea**: The "reward $r=\bm{z}^{\top}W_{\text{bt}}$" is reformulated as $r=\bm{\theta}^{\top}\Phi$, where both $\bm{\theta}$ (instance-level local factors) and $\Phi$ (global reward dictionary) are assigned Gamma priors to enforce non-negative sparsity. Local sparsity facilitates instance-level disentanglement, while global sparsity handles population-level debiasing. Together, they make the RM "insensitive to shortcuts" at the source.

## Method

### Overall Architecture
BNRM reformulates the standard BT reward model as a hierarchical Bayesian generative process and utilizes amortized variational inference + Weibull reparameterization for end-to-end training. The overall pipeline consists of four steps:

1.  **Input**: A preference triplet $(\bm{x},\bm{y}_1,\bm{y}_2)$, where $\bm{y}_1$ is chosen and $\bm{y}_2$ is rejected.
2.  **Feature Encoding**: An LLM backbone $f$ (e.g., Gemma-2B-it / Skywork-Reward-Llama-3.1-8B) encodes $(\bm{x},\bm{y})$ into $\bm{z}=f(\bm{x},\bm{y})\in\mathbb{R}^{d_{\text{model}}}$.
3.  **Variational Inference**: A linear head $W_{\text{vi}}\in\mathbb{R}^{d_{\text{model}}\times 2K}$ maps $\bm{z}$ to the shape and scale parameters $(\bm{k},\bm{\lambda})$ of a Weibull distribution. Samples of non-negative sparse local factors $\bm{\theta}\in\mathbb{R}^{K}_+$ are drawn. The global reward dictionary $\Phi\in\mathbb{R}^{K}_+$ is similarly modeled using a Weibull posterior $q(\Phi)$.
4.  **Reward Generation + BT Likelihood**: The scalar reward is generated as $r=\bm{\theta}^{\top}\Phi$ (non-negative, sparse, and decomposable into the contributions of $K$ semantic factors), which is then used in the BT model: $p(\bm{y}_1\succ\bm{y}_2)=\sigma(r_1-r_2)$.

During training, the ELBO is maximized: the reconstruction term ensures factors explain observed preferences, while two KL terms pull $q(\bm{\theta})$ and $q(\Phi)$ toward Gamma priors, enforcing sparsity and controlling model complexity. This process does not require training multiple models; a single forward pass provides the "reward mean + uncertainty + factor decomposition."

### Key Designs

1.  **NFA Reshaping the Reward Generation Process (disentanglement-then-debiasing)**:
    - **Function**: Replaces the dense linear head $W_{\text{bt}}$ with non-negative sparse local latent variables $\bm{\theta}$ and a global dictionary $\Phi$, rewriting the reward as $r(\bm{x},\bm{y})=\bm{\theta}^{\top}\Phi$.
    - **Mechanism**: Gamma priors $\mathrm{Gamma}(\alpha_0,\beta_0)$ and $\mathrm{Gamma}(\gamma_0,\delta_0)$ are applied to $\bm{\theta}$ and $\Phi$, respectively, forcing both to be non-negative and highly sparse. Local sparsity implies that each prompt-response pair activates only a few semantic factors, achieving instance-level disentanglement. Global sparsity ensures that only a few stable reward dimensions are retained in the dictionary, suppressing systematic biases (like length or templates) that recur across samples as "non-invariant features."
    - **Design Motivation**: Dense black-box rewards are a breeding ground for reward hacking because they blend all relevant and irrelevant features. Non-negative sparse factors act as a structural prior for the RM, forcing "semantic intent" and "style noise" into separate factors, then using global sparsity to zero out the weights of noise factors.

2.  **Two-Layer Uncertainty + Weibull Variational Posterior**:
    - **Function**: Simultaneously models both aleatoric (labeling noise) and epistemic (model parameters) uncertainty using differentiable sampling for both.
    - **Mechanism**: The BT model is generalized by integrating over the three latent variables $\bm{\theta}_1, \bm{\theta}_2, \Phi$: $p(\bm{y}_1\succ\bm{y}_2|\bm{x},\bm{y}_1,\bm{y}_2)=\int p(\bm{y}_1\succ\bm{y}_2|\bm{\theta}_1,\bm{\theta}_2,\Phi) q(\bm{\theta}_1)q(\bm{\theta}_2)q(\Phi)\,d\bm{\theta}_1 d\bm{\theta}_2 d\Phi$. The variational posteriors are defined as Weibull distributions $q(\bm{\theta}|\bm{x},\bm{y})=\mathrm{Weibull}(\bm{k},\bm{\lambda})$. The shape $\bm{k}$ uses Softplus for differentiable stability, and the scale $\bm{\lambda}$ uses ReLU to empirically encourage sample sparsity. The Weibull distribution supports standard backpropagation via reparameterization.
    - **Design Motivation**: Gamma posteriors are difficult to reparameterize and expensive to sample. Weibull and Gamma distributions exhibit similar tail behavior, but Weibull has an analytical reparameterization, allowing classical sparse models like NFA to be easily embedded into modern LLM training flows. Additionally, the two-layer uncertainty provides "confidence-aware" reward signals for subsequent BoN/PPO, penalizing uncertain regions of the RM to mitigate over-optimization.

3.  **Amortized Variational Inference + ELBO End-to-End Training**:
    - **Function**: Reuses the LLM backbone $f$ as a variational inference network (encoder), allowing $W_{\text{llm}}, W_{\text{vi}},$ and $\Phi$ to be optimized end-to-end on preference data.
    - **Mechanism**: The training objective is $\mathcal{L}(\mathcal{D})=\mathbb{E}_{q(\bm{\theta})q(\Phi)}[\log p(\mathcal{D}|\bm{\theta},\Phi)] - \eta\,\mathrm{KL}(q(\bm{\theta})\|p(\bm{\theta})) - \eta\,\mathrm{KL}(q(\Phi)\|p(\Phi))$. The first term ensures the factors explain preferences, while the two KL terms pull the posteriors toward the sparse priors. The hyperparameter $\eta$ balances the likelihood and KL terms, equivalent to controlling "sparse regularization strength."
    - **Design Motivation**: Traditional NFA requires per-document Gibbs sampling or SVI, which is incompatible with large-batch LLM training. Using the LLM's own dense representation $\bm{z}$ to amortize the posterior inference of $\bm{\theta}$ produces $(\bm{k}, \bm{\lambda})$ in a single forward pass, making the Bayesian framework efficient enough to run within a LoRA + 2 epoch / 8B full-parameter fine-tuning budget.

### Loss & Training
The ELBO is as formulated above, with $\eta$ as the key hyperparameter for sparsity strength (sensitivity analysis is provided in the appendix). Training settings: Gemma-2B-it / Gemma2-2B-it were trained for 2 epochs using LoRA; Skywork-Reward-Llama-3.1-8B was fine-tuned with full parameters for 1 epoch on Skywork-Preference-v0.2. In the RL stage, BNRM served as the proxy reward to train Llama3.1-8B-Instruct and OpenRLHF-Llama3-8B-SFT using PPO + LoRA for 1 epoch. Best-of-N tests only utilized the two Gemma models.

## Key Experimental Results

### Main Results: ID + OOD Reward Modeling Accuracy (Gemma2-2B-it, LoRA, UF Training)

| Data Scale | Method | UF (ID) | HHH | MT | RewardBench Avg |
|------------|--------|---------|-----|-----|-----------------|
| 40K        | BT     | 74.5    | 84.2| 73.3| 75.7            |
| 40K        | BT-Ensemble | 75.1 | 84.9 | 74.3 | 77.8          |
| 40K        | GRM-SFT | 75.8   | 85.5| 74.2| 77.3            |
| 40K        | InfoRM | 73.9    | 83.9| 74.6| 79.2            |
| 40K        | **BT-BNRM** | **77.2 ↑2.7** | **87.8 ↑3.6** | **76.8 ↑3.5** | **79.7 ↑4.0** |
| 400K       | BT     | 76.6    | 86.4| 75.2| 77.5            |
| 400K       | BT-Ensemble | 76.9 | 83.9 | 76.3 | 78.2          |
| 400K       | InfoRM | 77.3    | 85.4| 76.3| 80.7            |
| 400K       | **BT-BNRM** | **78.8 ↑2.2** | **88.2 ↑1.8** | **78.2 ↑3.0** | **79.5 ↑2.0** |

At the 40K scale, BT-BNRM improves RewardBench by 4.0 over BT and 1.9 over the much more expensive BT-Ensemble. It maintains a steady lead of ≥2 points at the 400K scale, indicating that sparse regularization is not only vital in low-resource scenarios but also remains effective with large datasets.

### Horizontal Comparison with Commercial/Open-source Large RMs (RewardBench)

| Category | Method | Average | Chat | Chat-Hard | Safety | Reasoning |
|----------|--------|---------|------|-----------|--------|-----------|
| Generative | GPT-4o | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
| Generative | Gemini-1.5 | 86.8 | 94.1 | 77.0 | 85.8 | 90.2 |
| Discriminative | Nemotron-340B-Reward | 92.2 | 95.8 | 87.1 | 92.2 | 93.6 |
| Discriminative | ArmoRM-Llama3-8B | 90.8 | 96.9 | 76.8 | 92.2 | 97.3 |
| Discriminative | Skywork-1-1BT-RM-8B | 91.8 | — | — | — | — |

The paper demonstrates that BNRM remains competitive against discriminative RMs in the 8B-70B range (specifically in Chat-Hard and Safety). Being only 2B/8B and not requiring ensembles, its cost-performance is significantly higher than large-scale models like Nemotron-340B or ArmoRM.

### Key Findings
- **Highest Gains on Small Data**: The improvement in RewardBench on 40K UF (+4.0) is significantly greater than on 400K (+2.0), confirming that sparse priors act as strong regularizers when data is scarce. Continued advantages on large data suggest that non-negative sparse factors are more than just a substitute for missing data.
- **Strongest Growth in HHH and Chat-Hard**: BT-BNRM gains 13 points on HHH relative to BT (40K setting) and frequently gains 5–8 points on RewardBench Chat-Hard. These subsets rely heavily on semantic discrimination and are most susceptible to style shortcuts, aligning with the hypothesis that sparse factors suppress spurious correlations.
- **Comparison with InfoRM**: InfoRM implicitly suppresses irrelevant features via an information bottleneck, whereas BNRM uses sparse factors for explicit decomposition. While they perform similarly on In-Distribution (InD) data, BNRM is more stable on Out-of-Distribution (OOD) data (HHH, MT), suggesting that explicit disentanglement-then-debiasing is more robust under distribution shift.
- **Interpretability Byproduct**: Each non-negative column of $\Phi$ corresponds to a "reward atom." Clustering associated responses provides human-readable semantic factors (case studies in Figure), a capability dense RMs lack.

## Highlights & Insights
- **Structural Anti-hacking**: While most work patches "symptoms" (removing length bias, adding ensembles, adding KL penalties), this work is among the few to intervene at the "representation form" level—changing the reward's mathematical form from $\bm{z}^{\top}W$ to $\bm{\theta}^{\top}\Phi$ to make sparsity and non-negativity intrinsic invariants. This approach of "changing representation rather than stacking tricks" is worth emulating.
- **Reusable NFA × LLM Paradigm**: Using Weibull reparameterization and LLM amortized encoders to integrate classical sparse Bayesian models into modern deep networks is a mechanism that can be transferred to DPO/KTO reward modeling, reasoning scorers, or even similarity heads in contrastive learning.
- **Uncertainty ≠ Ensemble**: The authors prove that a single model can achieve robustness comparable to or better than an ensemble, provided the model has a structure capable of carrying uncertainty (Weibull distribution + global parameter posterior), rather than simply averaging $N$ models.

## Limitations & Future Work
- **Sparsity Strength Sensitivity**: $\eta$ and Gamma prior hyperparameters $(\alpha_0,\beta_0,\gamma_0,\delta_0)$ directly affect sparsity. Sensitivity analyses are provided, but no universal recommendation is given; tuning is required for new datasets.
- **Tension Between Factor Dimension K and Interpretability**: If K is too small, factors are forced into entanglement; if K is too large, excessive sparsity might lose semantic meaning. The paper does not fully discuss adaptive K selection (e.g., automatic order determination in nonparametric Bayes).
- **Limited PPO Phase Evaluation**: While the core benefit is a more robust RM, downstream PPO/BoN evaluation is limited to 2B/8B models and single SFT bases. Cross-scale comparability for 70B+ and multiple bases remains to be explored.
- **Future Directions**: Integrating global sparsity of $\Phi$ with reward editing or interpretability tools to allow humans to "pick factors and adjust weights," moving from passive debiasing to active controllable alignment.

## Related Work & Insights
- **vs BT-Ensemble (Coste et al., 2024)**: Ensembles use $N$ RMs to approximate epistemic uncertainty via mean/variance at $N\times$ memory and compute costs. BNRM captures the same signal with a single model and a global Weibull posterior at lower cost with equivalent or better results.
- **vs InfoRM (Miao et al., 2024)**: InfoRM uses a variational information bottleneck to implicitly suppress spurious features. BNRM uses NFA to explicitly apply sparse priors to each factor. BNRM is more stable on OOD/HHH and offers better interpretability.
- **vs GRM (Yang et al., 2024)**: GRM uses auxiliary SFT/DPO heads for regularization. BNRM directly modifies the probabilistic structure of the reward. These are orthogonal; the paper shows that a combination (GRM-BNRM) continues to improve performance by layering sparse priors on other regularizations.
- **vs BT-Margin / Label Smoothing**: These tricks equate to adding small perturbations to the BT loss and are effective for single shortcuts like length. BNRM fixes the issue through structure rather than loss correction, generalizing better to unknown shortcuts.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying classical NFA tools to RMs via modern variational techniques is theoretically sound and provides a fresh perspective, though NFA + deep network combinations exist in topic modeling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers ID/OOD/BoN/PPO and compares against strong baselines including BT families, Ensemble, InfoRM, and GRM across two scales (40K/400K) and three bases.
- **Writing Quality**: ⭐⭐⭐⭐ Balanced use of formulas and diagrams. The derivation chain from BT → Bayesian BT → BNRM is logical and easy to follow.
- **Value**: ⭐⭐⭐⭐ Provides a structured, interpretable, and low-cost solution to reward hacking, offering direct utility for production-grade RLHF and alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[ACL 2026\] AgentV-RL: Scaling Reward Modeling with Agentic Verifier](../../ACL2026/llm_alignment/agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](../../ACL2026/llm_alignment/aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)

</div>

<!-- RELATED:END -->
