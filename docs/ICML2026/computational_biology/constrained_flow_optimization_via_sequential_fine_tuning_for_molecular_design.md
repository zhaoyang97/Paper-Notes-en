---
title: >-
  [Paper Note] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design
description: >-
  [ICML 2026][Computational Biology][Flow Matching Fine-tuning] This paper addresses the critical scenario of "maximizing rewards (e.g., binding affinity, dipole moment) while satisfying domain hard constraints (e.g.…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Flow Matching Fine-tuning"
  - "Augmented Lagrangian"
  - "Constrained Generative Optimization"
  - "Molecular Design"
  - "KL Regularization"
date: 2026-05-08
content_hash: b1d935a0b02876d1
---

# Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design

**Conference**: ICML 2026  
**arXiv**: [2605.30610](https://arxiv.org/abs/2605.30610)  
**Code**: TBD  
**Area**: Scientific Computing / Molecular Design / Generative Optimization  
**Keywords**: Flow Matching Fine-tuning, Augmented Lagrangian, Constrained Generative Optimization, Molecular Design, KL Regularization

## TL;DR
This paper addresses the critical scenario of "maximizing rewards (e.g., binding affinity, dipole moment) while satisfying domain hard constraints (e.g., synthetic accessibility, energy upper bounds)." It proposes the CFO algorithm, which uses the Augmented Lagrangian to decompose constrained generative optimization into a sequence of standard KL-regularized fine-tuning subproblems. By adaptively updating the penalty factor $\rho_k$ and dual variable $\lambda_k$, CFO achieves provable convergence and significant Pareto improvements in reward-constraint trade-offs across both synthetic low-dimensional scenarios and FlowMol molecular design tasks.

## Background & Motivation

**Background**: Diffusion and flow matching models have become the de facto standard generators for scientific discoveries involving molecules, proteins, and DNA. To utilize them for real-world discovery tasks, the mainstream approach is to perform reward-driven fine-tuning on pre-trained flow models under KL regularization (e.g., Adjoint Matching, DiffusionNFT, Flow-GRPO) using RL or optimal control, maximizing learnable rewards such as binding affinity or QED.

**Limitations of Prior Work**: Many "hard" constraints in scientific discovery—such as synthetic accessibility, toxicity upper bounds, xTB energy, and physical plausibility of docking poses—are either not encoded in pre-training data or only weakly learned. While current reward-driven fine-tuning uses KL to anchor distributional drift, it **cannot prove that any hard constraints are satisfied** (Uehara et al., 2024a). The naive approach of "treating constraints as another weighted reward" is highly unstable in practice: the weight $\mu$ drifts between tasks and training phases, requiring exhaustive trial-and-error tuning; furthermore, when exploration enters high-reward regions, the same $\mu$ is often overwhelmed by the reward, resulting in "high-reward but invalid" molecules.

**Key Challenge**: There is a trade-off between reward maximization and constraint satisfaction, and a **fixed-weight Lagrangian** cannot reliably characterize this trade-off—it neither guarantees feasibility (unless $\mu$ exceeds an unknown threshold) nor provides a monotonic mapping from $\mu$ to the violation amount, and it lacks the dynamics to "tighten" penalties during training.

**Goal**: (i) To provide a rigorous optimization formulation for constrained generative optimization (reward + KL regularization + expected constraints) and unify the subproblem of "constrained generation" when the reward is constant; (ii) To design an algorithm that automatically and provably balances rewards and constraints while staying close to the pre-trained model in terms of KL divergence.

**Key Insight**: The authors observe that classical constrained optimization already offers highly mature, hyperparameter-insensitive methods—specifically, the Augmented Lagrangian (AL). The core benefit of AL is that the penalty factor $\rho_k$ and dual variable $\lambda_k$ adaptively adjust based on "constraint violation," eliminating the need for manual weight enumeration. By applying AL to flow models, each iteration reduces to a standard KL-regularized fine-tuning subproblem with a "new augmented reward," which can be solved using existing solvers like Adjoint Matching or DiffusionNFT.

**Core Idea**: Transform constrained generative optimization into a sequence of regular fine-tuning subproblems with adaptive augmented rewards. Use AL dual updates to automatically tighten or loosen penalties, avoiding manual $\mu$ tuning while providing provable feasibility and optimality guarantees.

## Method

### Overall Architecture

The input consists of a pre-trained flow model $\pi^{\text{pre}}$ (treated as a feedback policy in RL where the velocity field is the action), a scalar reward $r(x)$, a scalar constraint $c(x)$, and a violation upper bound $B$. The goal is to find a new policy $\pi^{*}$ whose terminal distribution $p_1^{\pi}$ satisfies:

$$\max_{\pi} \mathbb{E}_{x \sim p_1^{\pi}}[r(x)] - \alpha D_{KL}(p_1^{\pi}\,||\,p_1^{\text{pre}}) \quad \text{s.t.} \quad \mathbb{E}_{x \sim p_1^{\pi}}[c(x)] \le B$$

If $r \equiv 0$, the problem reduces to constrained generation: "maintaining proximity to the pre-trained model under expected constraints."

CFO maintains two dual variables in the outer loop—the penalty factor $\rho_k$ and the Lagrange multiplier $\lambda_k$. In the inner loop, each iteration converts the problem into a standard KL-regularized fine-tuning by "replacing the original $r$ with an augmented reward $f_k$." There are $K$ outer rounds, each consisting of five steps: constructing the augmented reward → invoking the inner solver → estimating the constraint gap → updating $\lambda$ via projection → deciding whether to increase $\rho$ based on the "contraction statistic" $V_k$.

### Key Designs

1. **Augmented Reward $f_k$ (Embedding constraints into rewards with online adaptive weights)**:
    - **Function**: Each round packages "reward $-$ quadratic penalty" into a new reward for any KL-regularized fine-tuning solver to use directly; dual variables determine the penalty intensity and onset point.
    - **Mechanism**: Define $f_k(x) = r(x) - \frac{\rho_k}{2}\bigl[\max(0,\, c(x) - B - \frac{\lambda_k}{\rho_k})\bigr]^2$. The offset term $\lambda_k/\rho_k \le 0$ shifts the "penalty onset" threshold towards a stricter direction, forcing the algorithm to act even before the violation exceeds $B$; the quadratic term ensures smooth KKT conditions after solving, making it easier to optimize than hard truncation.
    - **Design Motivation**: Naive Lagrangians with fixed $\mu$ (Eq. 7) cannot adapt. The quadratic penalty with an offset in AL preserves the intuition that "greater violations lead to heavier penalties" and equivalently implements proximal point updates for dual variables via outer updates for $\rho$ and $\lambda$, thereby achieving provable convergence.

2. **Dual Update $\lambda_{k+1}$ (Projected gradient-style Lagrangian multiplier adjustment)**:
    - **Function**: Dynamically adjust $\lambda$ based on the "empirical constraint gap under the current policy," avoiding manual enumeration.
    - **Mechanism**: First, use Monte Carlo to estimate $G_k = \mathbb{E}_{x \sim p_1^{\pi_k}}[c(x)] - B$. If $G_k > 0$ (violation), push $\lambda_{k+1}$ to be more negative to increase the penalty; if $G_k < 0$ (satisfaction), pull $\lambda$ back toward 0 to loosen the penalty. Finally, project onto the interval $[\lambda_{\min}, 0]$ using $\lambda_{k+1} \leftarrow \max\{\lambda_{\min}, \min\{0, \lambda_k - \rho_k G_k\}\}$.
    - **Design Motivation**: Drive the reward-constraint trade-off with data rather than trial-and-error. The upper bound $0$ ensures the penalty remains a penalty rather than a reward, while the lower bound $\lambda_{\min}$ prevents a single violation from pushing the penalty to infinity.

3. **Trigger-based Increase of Penalty Factor $\rho_{k+1}$ via "Contraction Statistic"**:
    - **Function**: If the algorithm identifies that no adjustment of $\lambda$ can suppress the violation, it multiplies $\rho$ by $\eta \ge 1$ to raise the overall penalty intensity; otherwise, it remains unchanged to avoid unnecessary stiffness.
    - **Mechanism**: Define $V_k = \min\{G_k,\, -\lambda_k/\rho_k\}$ as the progress statistic for "approaching the feasible region." If and only if $V_k > \tau V_{k-1}$ (i.e., failing to contract by ratio $\tau \in (0,1)$), the current penalty factor is deemed insufficient, and $\rho_{k+1} = \eta \rho_k$; otherwise, $\rho_{k+1} = \rho_k$.
    - **Design Motivation**: This is a standard heuristic in the AL framework for "judging whether to climb the penalty ladder" (Birgin & Martínez, 2014). it allows $\rho$ to stay at a "just sufficient" level, avoiding numerical ill-conditioning in subproblems while escalating strength when necessary.

### Loss & Training

The inner loop invokes an arbitrary KL-regularized fine-tuning solver ("FineTuningSolver") to solve $\pi_k \in \arg\max_{\pi} \mathbb{E}_{x \sim p_1^{\pi}}[f_k(x)] - \alpha D_{KL}(p_1^{\pi}\,||\,p_1^{\text{pre}})$. The paper validates CFO's decoupling from solvers using Adjoint Matching (AM, first-order, requires differentiable $r, c$) and DiffusionNFT (NFT, zero-order, handles non-differentiable targets). For fair comparison, the paper fixes the "total gradient step budget" $M = K \cdot N$, comparing CFO and baselines under equivalent computation; typical settings are $K = 6, N = 10$ (molecular tasks) or $K = 20$ (low-dimensional tasks). Theoretically, as long as the inner loop produces an approximately optimal solution with error $\epsilon_k$ in each round (Assumption 5.1), the feasibility of CFO is guaranteed (Theorem 5.2 + Corollary 5.3); global optimality is achieved if $\epsilon_k \to 0$ (Theorem 5.4).

## Key Experimental Results

### Main Results

**Low-dimensional visualization tasks** (reward is negative squared distance to a white cross, constraint is linearly increasing outside a red triangle) and **FlowMol molecular design on GEOM Drugs** (reward = dipole moment in Debye, constraint = total xTB energy $\le -80$ Ha).

| Task / Solver | Method | Reward $\mathbb{E}[r] \uparrow$ | Constraint $\mathbb{E}[c]$ | Satisfied? |
|---|---|---|---|---|
| 2D Toy, AM Inner | PRE | $-7.62 \pm 0.03$ | $0.58 \pm 0.07$ | No |
| 2D Toy, AM Inner | AM (Unconstrained) | $-2.93 \pm 0.03$ | $2.47 \pm 0.11$ | No (4.3x Worse) |
| 2D Toy, AM Inner | **CFO$_{\text{AM}}$** | $-4.75 \pm 0.04$ | $\mathbf{0.12 \pm 0.06}$ | Yes |
| 2D Toy, NFT Inner | DiffusionNFT | $-3.59 \pm 0.10$ | $1.76 \pm 0.05$ | No |
| 2D Toy, NFT Inner | **CFO$_{\text{NFT}}$** | $-5.28 \pm 0.14$ | $\mathbf{0.06 \pm 0.01}$ | Yes |
| Molecular Design, AM | PRE / AM / **CFO$_{\text{AM}}$** | $6.55 / 8.37 / \mathbf{8.39}$ D | $-77.86 / -78.31 / \mathbf{-82.28}$ Ha | No / No / **Yes** |
| Molecular Design, NFT | DiffusionNFT / **CFO$_{\text{NFT}}$** | $8.30 / \mathbf{8.27}$ D | $-78.67 / \mathbf{-80.72}$ Ha | No / **Yes** |

Key Point: CFO achieves strict constraint satisfaction with almost no reward loss, holding true for both first-order and zero-order solvers.

### Ablation Study

| Configuration | Reward $\mathbb{E}[r]$ | Constraint $\mathbb{E}[c]$ ($\le B = -80$) | Description |
|---|---|---|---|
| Fixed $\mu = 0.01$ (Manual Lagrangian) | $8.34$ D | $-78.94$ Ha | High reward but violates constraint |
| Fixed $\mu = 50.0$ | $6.69$ D | Satisfied | Reward degradation, barely better than PRE |
| 18 $\mu \in [10^{-6}, 10^{6}]$ Enum. | — | — | Only 2 $\mu$ values reached CFO-level reward while satisfying constraints; cost $\approx 18\times$ CFO |
| **CFO** ($K=6, N=10$) | $8.39$ D | $-82.28$ Ha | Succeeded in one run; $\approx 44$ min |
| Equal compute AM$_{\mu=0.5}$ | $8.38$ D | Satisfied | But requires 727 min enumeration overhead |
| Budget $M=6000, K=3, N=2000$ | High Reward | $0.40$ (Violated) | Too few outer updates |
| Budget $M=6000, K=20, N=300$ | $-4.75$ | $0.12$ | Balanced point |
| Budget $M=6000, K=100, N=60$ | $-5.91$ | $0.10$ | Inner solver too weak, reward loss |

### Key Findings

- **CFO is entirely insensitive to the "penalty weight" hyperparameter $\mu$** because $\rho_k, \lambda_k$ adapt online; meanwhile, manual enumeration of 18 $\mu$ values for the naive Lagrangian only yielded 2 viable candidates, validating the motivation that "manual $\mu$ tuning is unreliable."
- **Decoupling from inner solvers**: Replacing AM with DiffusionNFT allows CFO to push molecular energy from $-78.67$ Ha to $-80.72$ Ha (compliant) with only a minor sacrifice in dipole moment, proving CFO is a lightweight shell that can wrap around any KL-regularized fine-tuner.
- **Minimal chemical statistical side effects**: While strictly meeting energy constraints, CFO's QED ($0.37$) and Lipinski score ($77\%$) are significantly closer to PRE ($0.45, 88\%$) than AM ($0.34, 71\%$), indicating that CFO does not trade chemical plausibility for constraint satisfaction. Per-sample feasibility rate: CFO $61.4\%$ vs AM $40.6\%$.
- **Computation extrapolation**: Fixing the total gradient steps $M = K \cdot N$, too few $K$ rounds prevent the dual update from catching up with violations, while too many $K$ disadvantage the inner solver's reward optimization. A moderate $K$ is the sweet spot, providing clear guidance for computational resource allocation in practical deployments.

## Highlights & Insights

- **"Constrained Generative Optimization" as a Unified Concept**: The authors place "constrained generation" and "constrained + reward fine-tuning" into the same KL-regularized constrained optimization framework, where constant rewards reduce to the former. This unification allows one algorithm (CFO) to solve both types of tasks rather than stacking independent modules.
- **Clever Adaptation of AL to Generative Fine-tuning**: Fine-tuning diffusion/flow models is inherently modeled as KL-regularized optimal control. The "unconstrained inner optimization + adaptive outer dual/penalty" of AL is almost custom-made for this structure. Applying a proven optimization paradigm to a new scenario is more robust than inventing a new loss.
- **Valuable Solver Abstraction**: CFO is not bound to Adjoint Matching, differentiable rewards, or continuous spaces. Theoretical guarantees only depend on Assumption 5.1 (bounded error of the approximate solver). This means as new solvers (Flow-GRPO, DRAKES, SEPO) emerge, CFO can be used as a plug-and-play component for discrete design scenarios like proteins or peptides.
- **"$V_k$ Contraction Criterion + Conditional Penalty Upgrade"**: This is an often-overlooked engineering detail in AL algorithms—it allows $\rho$ to escalate only when "necessary," avoiding numerical instability in subproblems. This is the true source of eliminating manual $\mu$ scheduling for engineers.

## Limitations & Future Work

- The authors acknowledge that while KL is anchored, the molecular "validity" metric drops from $34\%$ (PRE) to $9\%$ (CFO) (though still higher than AM's $4\%$); this occurs because optimization enters underrepresented regions of the chemical space, and validity was not an explicit constraint. A natural extension is to include validity as an element of $c$ to handle multi-constraint scenarios.
- The theoretical optimality results (Theorem 5.4) require $\epsilon_k \to 0$, which is almost impossible in practice. While using approximate solvers like AM works well empirically, the quantitative relationship between "how close the approximation is" and the "final suboptimality gap" remains unquantified. This is a common legacy issue for the AL + neural solver combination.
- If the inner loop involves expensive diffusion fine-tuning, the $K$ outer rounds significantly amplify computation. While the $M = K \cdot N$ budget strategy identifies a sweet spot for $K$, it remains to be seen if this trade-off curve remains smooth for higher-dimensional problems like all-atom proteins.
- Currently, all constraints are approximated by GNN predictors (e.g., for xTB energy). Predictor errors can propagate into dual updates and pollute $\lambda$. The authors leave this error analysis to the appendix without providing a front-and-center quantitative relationship between "predictor inaccuracy $\Rightarrow$ feasibility degradation."

## Related Work & Insights

- **vs. Adjoint Matching (Domingo-Enrich et al., 2024)**: AM is a purely reward-driven first-order fine-tuning solver with no concept of constraints. CFO treats it as an inner loop, wrapping AL around it to manage constraints. Essentially, CFO = AM + Adaptive Augmented Reward + Outer Dual Updates.
- **vs. Fixed-weight Lagrangian / KL-shielded RL (Chamon et al., 2024; Zhang et al., 2025b)**: These works require manual specification of $\mu$. The paper's experiments show that only 2 out of 18 enumerated $\mu$ values were usable. CFO replaces "finding $\mu$" with "automatically adjusting $\rho, \lambda$," roughly $18\times$ more efficient.
- **vs. DiffOpt (Kong et al., 2024) / Khalafi et al. (2024)**: These are "inference-time" constrained generation methods—they don't modify model weights but use guidance or reweighting during sampling. The advantage is no retraining; the disadvantage is additional inference cost per sample (DiffOpt reports $44$–$55\times$ base sampling cost). CFO is a "fine-tuning time" approach: constraints are baked into the weights, making inference as fast as the base model.
- **vs. DRAKES / SEPO (Discrete Generative Constraints)**: CFO's solver abstraction is naturally compatible with discrete domains. Although experiments were not conducted, the paper notes this path opens up constrained optimization for discrete designs of proteins and peptides.
- **Transferable Engineering Insight**: (i) Any generative task with "rewards + hard constraints" (multimodal RLHF, constrained image generation, safe text generation) can utilize the same AL wrapper; (ii) Naive weighted objectives are common in LLM RLHF, and CFO provides a drop-in alternative especially suited for "safety constraints must be met, utility should be as high as possible" scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Rigorously migrating the mature AL framework to flow model fine-tuning with provable convergence is a clean "theory + engineering" combination, though AL itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both visualization toys and FlowMol molecular tasks, including first/zero-order solvers and $18\times \mu$ enumeration comparisons, though limited to a single molecular task.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation and algorithmic steps are clear; theorems align well with assumptions, and the logical chain from Eq. 5 → Alg. 1 → Theorem 5.2/5.4 is highly coherent.
- Value: ⭐⭐⭐⭐ Provides a standard component for "constrained + rewarded" molecular/protein design without the need for weight tuning, offering high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](../../NeurIPS2025/computational_biology/flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](../../ICLR2026/computational_biology/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](../../ICLR2026/computational_biology/verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)

</div>

<!-- RELATED:END -->
