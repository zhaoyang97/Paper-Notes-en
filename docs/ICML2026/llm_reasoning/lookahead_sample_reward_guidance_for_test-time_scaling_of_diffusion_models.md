---
title: >-
  [Paper Note] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models
description: >-
  [ICML 2026][LLM Reasoning][Diffusion Models] LiDAR rewrites the Expected Future Reward (EFR) using pre-generated lookahead samples and forward perturbation kernels…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Diffusion Models"
  - "test-time scaling"
  - "reward guidance"
  - "lookahead sampling"
  - "closed-form Stein score"
date: 2026-05-08
content_hash: b2d09983cf20d24a
---

# Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2602.03211](https://arxiv.org/abs/2602.03211)  
**Code**: https://github.com/aailab-kaist/Diffusion-LiDAR-Sampling  
**Area**: Diffusion Models / Test-Time Scaling / Reward Guidance  
**Keywords**: Diffusion Models, test-time scaling, reward guidance, lookahead sampling, closed-form Stein score

## TL;DR
LiDAR rewrites the Expected Future Reward (EFR) using pre-generated lookahead samples and forward perturbation kernels, converting reward guidance into closed-form softmax weights without neural backpropagation. It matches DATE's performance on SDXL/GenEval while being 9.5× faster.

## Background & Motivation
**Background**: T2I diffusion models often generate samples that do not align with human intent. Two main alignment paths exist: fine-tuning (DPO, RLHF-like) and test-time scaling. The latter swaps computation for performance without training. The core involves pushing the distribution $p_\theta(\mathbf{x}_0\mid\mathbf{c})$ toward a reward-tilted target $p_\theta^r(\mathbf{x}_0\mid\mathbf{c}) \propto p_\theta(\mathbf{x}_0\mid\mathbf{c})\exp(\lambda r(\mathbf{x}_0,\mathbf{c}))$, requiring the estimation of the **Expected Future Reward (EFR)** $r_t^\lambda(\mathbf{x}_t,\mathbf{c}) = \log\mathbb{E}_{p_\theta(\mathbf{x}_0\mid\mathbf{x}_t,\mathbf{c})}[\exp(\lambda r(\mathbf{x}_0,\mathbf{c}))]$ for intermediate particles $\mathbf{x}_t$.

**Limitations of Prior Work**: Existing EFR estimation paths have significant drawbacks:

- **Backward rollout** (averaging multiple rollouts to $\mathbf{x}_0$): Requires full reverse diffusion at every timestep, incurring nearly unacceptable overhead.
- **Tweedie first-order Taylor approximation**: Replaces samples with $\bar{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$. Errors expand linearly as $\lambda$ increases, causing distortion under strong reward signals.
- **Gradient guidance** (UG / DATE): Requires three-stage neural backpropagation ($\mathbf{x}_t \to \mathbf{s}_\theta \to$ decoder $\to r$), necessitates differentiable rewards, and often leads to OOM on 2.6B models like SDXL.
- **SMC-based methods**: Use importance resampling to avoid backpropagation, but particles quickly collapse to a single high-reward sample in high-dimensional pixel space, significantly reducing diversity and performance depending heavily on particle count $N$.

**Key Challenge**: The EFR expression inherently forces $\mathbf{x}_t$ to serve as both a "neural network input" and a "gradient variable," which is the root cause of backpropagation necessity, approximation inaccuracies, and SMC instability.

**Goal**: Find an EFR rewriting method where $\mathbf{x}_t$ no longer enters any neural network as an input, while still accurately characterizing the Stein score of the reward-tilted distribution.

**Key Insight**: Noting $p_\theta(\mathbf{x}_0\mid\mathbf{x}_t,\mathbf{c}) \propto p(\mathbf{x}_t\mid\mathbf{x}_0)p_\theta(\mathbf{x}_0\mid\mathbf{c})$, can the expectation base be changed from "posterior conditioned on $\mathbf{x}_t$" to "prior $p_\theta(\mathbf{x}_0\mid\mathbf{c})$ weighted by forward kernel $p(\mathbf{x}_t\mid\mathbf{x}_0)$"? This way, $\mathbf{x}_t$ only appears in a Gaussian kernel with a known analytical form, decoupling neural dependencies.

**Core Idea**: Rewrite EFR using *future marginal samples + forward perturbation kernels* (Theorem 3.1) and perform cheap **lookahead sampling** using few-step ODE solvers (DPM-3/5/8, LCM-4, DMD-1) to generate these marginal samples. Then, prove that the Stein score in this form has a closed-form softmax solution (Theorem 3.3), resulting in a reward guidance sampler without neural backpropagation and with costs comparable to vanilla versions—**LiDAR**.

## Method

### Overall Architecture
LiDAR decouples test-time reward guidance into two phases, corresponding to Algorithm 1/2 in the paper:

- **Phase 1 (One-time budget)**: Given prompt $\mathbf{c}$, use a $\delta$-step fast solver $q(\mathbf{x}_0\mid\mathbf{c})$ (DPM-Solver, LCM, DMD, etc.) to generate $n$ lookahead samples $\{\hat{\mathbf{x}}_0^i\}_{i=1}^n$ in batch, labeled by the reward model $r$ to obtain $\{(\hat{\mathbf{x}}_0^i, r_i)\}$. This step is independent of $\mathbf{x}_t$ and calculated once per prompt.
- **Phase 2 (Target sampling)**: Iterate backward from $\mathbf{x}_T\sim p(\mathbf{x}_T)$ using SDE/ODE, replacing the standard Stein score with the LiDAR closed-form formula $\mathbf{s}_\theta(\mathbf{x}_t,t,\mathbf{c}) + s\cdot\nabla_{\mathbf{x}_t}\hat r_t^\lambda$. The gradient term is a softmax-weighted difference of lookahead samples. The process requires no backpropagation, and the reward model can be non-differentiable (e.g., ring counts in molecules).

Physical intuition is shown in Figure 1(b): Pulling $\mathbf{x}_t$ toward high-reward $\hat{\mathbf{x}}_0^i$ and pushing it away from low-reward ones, with "gravitational" strength proportional to the reward.

### Key Designs

1. **EFR Rewriting via Forward Rollout (Theorem 3.1)**:
    - **Function**: Rewrites EFR to depend only on prior samples $\mathbf{x}_0\sim p_\theta(\mathbf{x}_0\mid\mathbf{c})$ and the forward kernel $p(\mathbf{x}_t\mid\mathbf{x}_0)$, ensuring $\mathbf{x}_t$ is no longer fed into any neural network.
    - **Mechanism**: Uses Bayes' rule to rewrite the conditional expectation $\mathbb{E}_{p_\theta(\mathbf{x}_0\mid\mathbf{x}_t,\mathbf{c})}[\exp(\lambda r)]$ as $\mathbb{E}_{p_\theta(\mathbf{x}_0\mid\mathbf{c})}\big[\tfrac{p(\mathbf{x}_t\mid\mathbf{x}_0)}{\mathbb{E}[p(\mathbf{x}_t\mid\mathbf{x}_0)]}\exp(\lambda r)\big]$. The base shifts from posterior to prior, with $\mathbf{x}_t$ appearing only in the analytical Gaussian kernel.
    - **Design Motivation**: This is the "key" to the paper. Previous methods were trapped by backpropagation or Taylor approximations because $\mathbf{x}_t$ entered neural networks and required derivation. By changing the base, $\mathbf{x}_t$ becomes a variable in a Gaussian density with an analytical derivative, and pre-generated samples can be reused for any $\mathbf{x}_t$, eliminating the need to rerun rollouts at every step.

2. **Few-step Lookahead Sampling + Weak-to-Strong Interpretation**:
    - **Function**: Uses a cheap "weak" sampler $q(\mathbf{x}_0\mid\mathbf{c})$ (e.g., DPM-3/5, LCM-4, DMD-1) to approximate the expensive $p_\theta(\mathbf{x}_0\mid\mathbf{c})$ to generate marginal samples, making pre-generation costs negligible.
    - **Mechanism**: Substituting $q$ into Eq. 11 yields the lookahead reward $\tilde r_t^\lambda$ (Definition 3.2), where the guidance term is equivalent to $s\cdot\nabla_{\mathbf{x}_t}\log\tfrac{q^r(\mathbf{x}_t\mid\mathbf{c})}{q(\mathbf{x}_t\mid\mathbf{c})}$. This transfers the "density change under reward" of the weak sampler as a guidance signal to the strong sampler—a standard form of weak-to-strong generalization with a flexible weight $s$.
    - **Design Motivation**: Generating $n$ samples using full $p_\theta$ would still be slow. Lookahead transforms weak analytical power (few-step solvers) into "probes" for reward signals, using the strong sampler (full 50/100 step reverse) as the signal "executor," preserving high-quality distributions while amortizing costs into one-time, cacheable preprocessing.

3. **Derivative-Free Closed-Form Softmax Guidance (Theorem 3.3)**:
    - **Function**: Expresses the gradient of the target Stein score $\nabla_{\mathbf{x}_t}\hat r_t^\lambda$ as a pure algebraic expression, eliminating neural backpropagation and requirements for reward differentiability.
    - **Mechanism**: Direct derivation on the finite sample estimate of Eq. 11 yields $\sum_{i=1}^n (w_i^r - w_i)\hat{\mathbf{x}}_0^i / \sigma_t^2$, where $w_i^r = \mathrm{Softmax}_i(\lambda r_i - \|\mathbf{x}_t-\hat{\mathbf{x}}_0^i\|^2/2\sigma_t^2)$ and $w_i = \mathrm{Softmax}_i(-\|\mathbf{x}_t-\hat{\mathbf{x}}_0^i\|^2/2\sigma_t^2)$. The first softmax considers both reward and distance to $\mathbf{x}_t$; the second considers only distance. Their difference represents "how much more the reward should bias me toward a lookahead sample." When $r_i$ is high, $w_i^r > w_i$, pushing $\mathbf{x}_t$ toward $\hat{\mathbf{x}}_0^i$.
    - **Design Motivation**: This perfectly satisfies the four attributes in Table 1: Efficient-Rollout, Finite i.i.d., No-Taylor, and No-BackPropagation. This is the key to LiDAR's practicality—9.5× acceleration, no additional VRAM, and compatibility with black-box rewards.

### Loss & Training
LiDAR is a completely *training-free* test-time method. It introduces no loss or parameter updates. Key hyperparameters include: lookahead solver steps $\delta$, number of samples $n$, reward temperature $\lambda$, guidance scale $s$, and total target sampling steps $\tau$. The paper provides two scaling laws: $D_{TV}\le O(1/\sqrt{\delta})$ as $\delta$ increases (Theorem 3.4), and finite sample error converging to the lookahead target at $1/\sqrt n$ as $n$ increases (Theorem 3.5). In practice, $n=50$ is sufficient.

## Key Experimental Results

### Main Results
All methods used ImageReward as guidance on SD v1.5 / SDXL, comparing generation quality and single inference cost on GenEval prompts (4 images per prompt) using a single A100:

| Backbone (sampler) | Method | IR ↑ | GenEval ↑ | Time(s) ↓ | Mem(GiB) ↓ |
|----------------|------|------|-----------|-----------|------------|
| SD v1.5 (DDPM-100) | Vanilla | -0.001 | 0.426 | 7.07 | 8.90 |
| SD v1.5 (DDPM-100) | UG (Bansal'24) | 0.326 | 0.355 | 58.36 | 28.16 |
| SD v1.5 (DDPM-100) | DATE (Na'25) | 0.364 | 0.438 | 32.89 | 24.71 |
| SD v1.5 (DDPM-100) | **LiDAR (DPM-5,n=50)** | **0.384** | **0.478** | **13.41** | **8.90** |
| SDXL (DDPM-100) | Vanilla | 0.722 | 0.545 | 42.0 | 33.84 |
| SDXL (DDPM-100) | UG | 0.749 | 0.541 | 334.4 | OOM* |
| SDXL (DDPM-100) | DATE | 0.960 | 0.570 | 272.3 | OOM* |
| SDXL (DDPM-100) | **LiDAR (DPM-8,n=50)** | 0.994 | 0.585 | **97.99** | **33.84** |
| SDXL (DDPM-100) | **LiDAR (DMD-1,n=100)** | **1.006** | **0.598** | **78.67** | **33.84** |

LiDAR achieves GenEval scores on par with DATE (0.585 vs 0.570) on SDXL in ~30% of the time without OOM-inducing backpropagation memory overhead.

### Ablation Study

| Configuration | IR | GenEval | Time(s) | Description |
|------|----|---------|---------|------|
| Vanilla SD v1.5 | -0.001 | 0.426 | 7.07 | No guidance baseline |
| DPM-3, n=3 | 0.109 | 0.439 | 7.44 | Extremely weak lookahead, effective almost for free |
| DPM-5, n=3 | 0.172 | 0.449 | 7.54 | Upgrading lookahead solver precision only |
| DPM-5, n=9 | 0.211 | 0.453 | 8.27 | Increasing $n$ |
| DPM-5, n=50 | 0.384 | 0.478 | 13.41 | Full configuration |
| DPO Fine-tuned + DPM-5, n=50 | 0.445 | 0.489 | 13.41* | Orthogonal stacking with training-side methods |

### Key Findings
- **Lookahead precision $\delta$ and sample count $n$ are both monotonically beneficial**, following the theoretical $O(1/\sqrt{\delta})$ and $O(1/\sqrt n)$ scaling (Figure 3), allowing users to adjust based on budget.
- **Speed comes from "no backpropagation"**: The bottleneck for UG/DATE is backpropagation on the 2.6B SDXL model. LiDAR's closed-form score returns memory usage to vanilla levels.
- **Orthogonal to DPO fine-tuning** (IR 0.384 → 0.445), proving LiDAR is an additive benefit rather than just a reward-hacking substitute.
- **Adapts to non-differentiable rewards**: Using "ring count" as reward on UDLM discrete diffusion + QM9 molecules, the number of novel molecules increased from 130 to 257 (Table 4). FLUX flow matching IR also improved from 1.019 to 1.198 (Table 3).
- **No CLIP/HPS degradation** indicates guidance does not sacrifice prompt alignment (mitigates reward hacking), whereas UG's HPS dropped from 0.263 to 0.236.

## Highlights & Insights
- **"Base Change" solves all limitations at once**: Rewriting EFR from a $p_\theta(\mathbf{x}_0\mid\mathbf{x}_t)$ base to a $p_\theta(\mathbf{x}_0)$ base simultaneously enables efficient-rollout, finite i.i.d., no-Taylor, and no-backprop—the true "aha" moment of the paper.
- **Closed-form softmax formula is highly interpretable**: $w_i^r - w_i$ is a function of both reward and distance differences, essentially a softened version of SMC's hard sampling or UG's gradient.
- **Lookahead as a continuous generalization of "Weak-to-Strong"**: Decoupling weak solvers as reward probes and strong solvers as executors can be transferred to any guided generation requiring expensive rollouts (video, 3D, etc.).
- **Pre-generated samples are cacheable**: For online services, $\{(\hat{\mathbf{x}}_0^i, r_i)\}$ for a prompt is a one-time asset, amortizing costs across multiple users or seeds.

## Limitations & Future Work
- **Quality ceiling is capped by the weak sampler**: If the prompt falls into a region where $q$ fails (e.g., extremely long prompts), LiDAR's guidance signal may be distorted.
- **Steep $n$ vs. memory curve on SDXL**: $n=100$ approaches VRAM limits, and GenEval gains saturate on FLUX beyond $n=100$ (0.667 vs 0.668).
- **Reward combination strategies unexplored**: Table 6 only tests simple weighted IR and CLIP; there is room for multi-reward Pareto fronts or engineering combinations against reward hacking.
- **Theoretical scaling laws are upper bounds**: The optimal trade-off between $\delta$ and $n$ still requires empirical tuning.
- **Small-scale validation on discrete diffusion and flow matching**: The QM9/FLUX transfers are more like PoC; large-scale validation on industrial text/video diffusion is missing.

## Related Work & Insights
- **vs UG (Bansal 2024) / DATE (Na 2025)**: Both use Tweedie first-order Taylor for reward calculation on $\bar{\mathbf{x}}_0$ then backpropagate to $\mathbf{x}_t$, constrained by approximation errors at high $\lambda$ and differentiability needs. LiDAR provides a closed-form EFR without approximation or backpropagation, 9.5× faster.
- **vs SMC Series (Singhal 2025, Li 2025)**: SMC also avoids backpropagation but suffers from particle collapse in high-dimensional space and relies heavily on $N$. LiDAR's finite i.i.d. property decouples particle count from generation quality.
- **vs Backward rollout (Holderrieth 2026, Potaptchik 2025)**: Rollout is conceptually correct but stuck at "rerunning every $t$"; LiDAR amortizes this using forward kernels and one-time marginal samples.
- **vs DPO/ReFL/DRaFT (Fine-tuning)**: Training methods need gradients and compute. LiDAR is test-time and can be **orthogonally stacked** with DPO for deployment-stage tuning.
- **Insight**: Any guidance/control problem involving "intermediate variable $\to$ neural network $\to$ backpropagation" (e.g., classifier guidance, controllable molecule/video generation) can attempt this "base change + analytical kernel + closed-form softmax" paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Base change + forward kernel EFR rewrite" is a genuine conceptual breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbones + ablation + scaling laws are complete, but industrial-scale long prompts/video are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Table 1 and the narrative flow are exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ 9.5× faster with no VRAM increase and orthogonal to fine-tuning; immediate engineering utility for T2I services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)

</div>

<!-- RELATED:END -->
