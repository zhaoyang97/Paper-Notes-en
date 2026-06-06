---
title: >-
  [Paper Note] Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation
description: >-
  [ICML 2026][Image Generation][reward-guided generation] By projecting reward gradients onto a "white Gaussian noise feasible set" characterized by DFT block-wise $\ell_1/\ell_2$ norms…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "reward-guided generation"
  - "one-step generative models"
  - "white Gaussian noise constraint"
  - "gradient preconditioning"
  - "spectral domain projection"
date: 2026-05-08
content_hash: dcf3f1db415e5518
---

# Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation

**Conference**: ICML 2026  
**arXiv**: [2602.08646](https://arxiv.org/abs/2602.08646)  
**Code**: TBD  
**Area**: Image Generation / Diffusion Models / Test-time Optimization  
**Keywords**: reward-guided generation, one-step generative models, white Gaussian noise constraint, gradient preconditioning, spectral domain projection

## TL;DR
By projecting reward gradients onto a "white Gaussian noise feasible set" characterized by DFT block-wise $\ell_1/\ell_2$ norms, the authors make test-time latent optimization for one-step generative models both fast and stable: on FLUX, it matches the Aesthetic Score of the SOTA regularization method MPGR in only 30% of the wall-clock time while completely avoiding reward hacking.

## Background & Motivation

**Background**: With distillation techniques like shortcut/consistency enabling "one-step generation" for diffusion and flow models, performing gradient ascent directly on the latent noise $\bm{x} \in \mathbb{R}^N$ during inference to maximize a reward $r(\mathcal{M}(\bm{x}))$ has become a popular direction (e.g., ReNO, MPGR, ORIGEN). It requires no retraining and allows plug-and-play reward switching, making it the most lightweight route for reward-guided generation today.

**Limitations of Prior Work**: This test-time latent optimization faces two critical issues in practice. First is **reward hacking**—as the latent deviates from the white Gaussian prior during gradient ascent, the model starts generating artifacts or even collapsed images, despite reward values being inflated. Second is **slowness**—even with a one-step model, a single image often requires over a hundred gradient updates, costing tens of seconds to minutes.

**Key Challenge**: Existing methods (ReNO, PRNO, MPGR) follow a **soft regularization** approach, adding a term $-\lambda \mathcal{L}_{\text{reg}}(\bm{x})$ to the objective to encourage the latent to maintain Gaussian properties (e.g., $\ell_2$ norm, spectral block-wise $\ell_1$). However, soft regularization has three flaws: (i) it does not guarantee that the latent remains in the noise-like region, only "encourages" it; (ii) it requires manual tuning of $\lambda$, where the weight and learning rate are coupled; (iii) soft penalties cannot stop the optimizer once it finds a shortcut (e.g., blowing up a specific frequency component).

**Goal**: To upgrade "maintaining white Gaussian properties" from a soft constraint to a **hard constraint** without sacrificing speed (projection must be performed at every step, requiring a closed-form and at most $\mathcal{O}(N \log N)$ complexity).

**Key Insight**: The authors notice that MPGR's spectral block-wise $\ell_1$ penalty already characterizes white noise spectral flatness well. Can this be upgraded to a "hard set" with gradient projection? Direct projection onto original DFT coefficients $\hat{\bm{x}} = \bm{F}\bm{x}$ is infeasible—DFT of real signals satisfies Hermitian symmetry, causing blocks to be coupled and coefficients to alternate between real and complex without simple closed-form solutions. However, by **stripping Hermitian redundancy** and reorganizing independent degrees of freedom into a compact complex vector $\bm{y} \in \mathbb{C}^{N/2}$, the problem decouples into $P$ independent projections onto the "intersection of $\ell_1$ and $\ell_2$ balls," which has a known closed-form solution.

**Core Idea**: Use a **bijection $\mathcal{F}: \mathbb{R}^N \to \mathbb{C}^{N/2}$** to map the white Gaussian prior to a compact spectral domain. There, a feasible set $\mathcal{G}_{\mathbb{C}}$ is defined by requiring "the $\ell_1$ and $\ell_2$ norms of each size-$B$ block to equal their expectations under $\mathcal{CN}(0,1)$." Every reward gradient is then projected back to $\mathcal{G}_{\mathbb{R}} = \mathcal{F}^{-1}(\mathcal{G}_{\mathbb{C}})$ to obtain a noise-aligned update direction.

## Method

### Overall Architecture
The input is a text prompt and a one-step generative model $\mathcal{M}: \mathbb{R}^N \to \mathbb{F}$ (the paper uses FLUX-schnell, $N = 65{,}536$). The output is the optimized latent $\bm{x}^* \in \mathbb{R}^N$ and the corresponding image $\mathcal{M}(\bm{x}^*)$. The process is a concise loop (Algorithm 1):

```python
repeat:
    J  ←  r(M(x))
    g  ←  ∇_x J
    g' ←  Proj_G(g)              # ← Core of this work
    x  ←  Adam(x, g')
```

Compared to soft regularization $\max_{\bm{x}} r(\mathcal{M}(\bm{x})) - \lambda \mathcal{L}_{\text{reg}}(\bm{x})$, there is **no $\lambda$**, and **$\bm{x}$ is not required to lie within $\mathcal{G}$** itself—it suffices that the "update direction" of each step lies within $\mathcal{G}$. This avoids reward hacking while focusing the search on subspaces compatible with white noise.

The difficulty lies entirely in the design and efficient implementation of the projection operator $\text{Proj}_{\mathcal{G}}$, addressed by three key designs.

### Key Designs

1. **Compact spectral domain bijection $\mathcal{F}: \mathbb{R}^N \to \mathbb{C}^{N/2}$**:

    - **Function**: Completely strips the redundancy caused by Hermitian symmetry in real-signal DFT, structurally decoupling the subsequent projection problem.
    - **Mechanism**: For even dimension $N$, only $\hat{x}_0, \hat{x}_{N/2}$ in DFT coefficients $\hat{\bm{x}}$ are real, and $\hat{x}_k = \overline{\hat{x}_{N-k}}$, meaning there are only $N/2$ complex independent degrees of freedom. The authors define $y_0 = \tfrac{\hat{x}_0}{\sqrt 2} + \tfrac{\hat{x}_{N/2}}{\sqrt 2} i$ and $y_k = \hat{x}_k$ ($k = 1, \dots, N/2-1$), forming $\bm{y} = \mathcal{F}(\bm{x})$. Proposition 4.1 proves $\mathcal{F}$ is a bijection $\mathbb{R}^N \leftrightarrow \mathbb{C}^{N/2}$, and $\bm{z} \sim \mathcal{CN}(\bm 0, \bm I_{N/2})$ iff $\mathcal{F}^{-1}(\bm{z}) \sim \mathcal{N}(\bm 0, \bm I_N)$. Proposition 4.2 further provides $\|\mathcal{F}^{-1}(\bm z)\|_2^2 = 2\|\bm z\|_2^2$. These translate "spatial domain white Gaussian noise constraints" into "compact spectral domain complex Gaussian constraints."
    - **Design Motivation**: Applying hard constraints directly on $\hat{\bm{x}}$ results in entanglement between blocks due to Hermitian coupling, precluding closed-form projections. Stripping redundancy naturally decouples constraints by block for efficient projection.

2. **Block-wise $\ell_1/\ell_2$ dual-norm feasible set $\mathcal{G}_{\mathbb{C}}$**:

    - **Function**: Defines a compact set on $\mathbb{C}^{N/2}$ to **precisely** characterize the statistical properties of white Gaussian noise, being tighter than a single $\ell_2$ or $\ell_1$ constraint.
    - **Mechanism**: Partition $\bm{y}$ into $P = N/(2B)$ blocks of size-$B$ (the paper uses $B = 16$). Simultaneously enforce the $\ell_1$ and $\ell_2$ norms of each block to equal their theoretical expectations under $\mathcal{CN}(0,1)$: $\|\bm{y}^{(p)}\|_1 = \tfrac{\sqrt\pi}{2}B$ and $\|\bm{y}^{(p)}\|_2^2 = B$. The spatial feasible set is defined as $\mathcal{G}_{\mathbb{R}} = \mathcal{F}^{-1}(\mathcal{G}_{\mathbb{C}})$. Thus, the $\ell_2$ constraint ensures the total energy matches the mode of the $\chi_N$ distribution $\|\bm{x}\|_2^2 = N$ (almost coinciding with the minima of $\ell_2$ norm regularization $\mathcal{L}_{\text{norm}}$), while the $\ell_1$ constraint prevents any single frequency component from dominating the spectrum (theoretically, the maximum $|y_j|^2$ within a block is only about $7.18$, whereas the total budget $N/2 \gg 10^4$), corresponding to the "no dominant frequency" property of white noise. Using 1.1M Gaussian samples, the authors verify that the minimum cosine similarity between $\bm{x} \sim \mathcal{N}(\bm 0, \bm I_N)$ and its projection onto $\mathcal{G}_{\mathbb{R}}$ is $0.988$, showing that real white noise naturally adheres to this feasible set.
    - **Design Motivation**: Compared to matching only total $\ell_1$ and $\ell_2$ (two global equalities), the feasible set formed by $2P$ block-wise equalities is strictly smaller and provides a tighter characterization of the white noise distribution. Unlike MPGR's soft regularization which only penalizes $\ell_1$ deviations, the hard set additionally locks the $\ell_2$ energy, completely excluding "shortcut solutions."

3. **$\mathcal{O}(N\log N)$ closed-form projection operator $\text{Proj}_{\mathcal{G}}$**:

    - **Function**: Given any $\bm{x} \in \mathbb{R}^N$ (the reward gradient), find its nearest point in $\mathcal{G}_{\mathbb{R}}$ within each optimization step with complexity close to FFT.
    - **Mechanism**: First calculate $\bm{y} = \mathcal{F}(\bm{x})$ via FFT. Due to the isometric relationship in Proposition 4.2 $\|\mathcal{F}^{-1}(\bm y) - \mathcal{F}^{-1}(\tilde{\bm y})\|_2^2 = 2\|\bm{y} - \tilde{\bm y}\|_2^2$, the spatial minimization problem has the same optimal solution in the compact spectral domain. Since the feasible set is block-independent, the problem decomposes into $P$ independent projections onto the "intersection of $\ell_1$ and $\ell_2$ balls," each solved in $\mathcal{O}(B\log B)$ using the algorithm by Liu et al. (2020). Specifically: sort $|y_j|$ within a block in descending order as $\bm{w}$, calculate prefix sums $S_{d,k} = \sum_{l=0}^k w_l^d$ ($d = 1, 2$), find the unique $k^*$ satisfying $w_{k^*+1} \le \lambda^{(k^*)} < w_{k^*}$, where $\lambda^{(k^*)} = \tfrac{S_{1,k^*}}{k^*+1} - \tfrac{\sqrt\pi}{2}\tfrac{\sqrt B}{k^*+1}\sqrt{\tfrac{(k^*+1)S_{2,k^*} - S_{1,k^*}^2}{k^*+1 - \tfrac{\pi}{4}B}}$, and use a ReLU soft-thresholding expression $\dot y_j = \tfrac{\sqrt\pi}{2}B \cdot \tfrac{\text{ReLU}(|y_j| - \lambda^{(k^*)})}{S_{1,k^*} - (k^*+1)\lambda^{(k^*)}} \cdot \tfrac{y_j}{|y_j|}$ to directly obtain the projection. Finally, inverse FFT back to the spatial domain. The total complexity is $\mathcal{O}(N\log N)$, taking only **0.04%** of the wall-clock time per iteration on FLUX.
    - **Design Motivation**: Speed is the bottom line as projection occurs every step. The combination of FFT and block-wise closed-form solutions makes the "hard constraint" practically free. In contrast, projecting onto the minimal set of $\mathcal{L}_{\text{power}}$ would require hundreds of inner-loop gradient descent steps (as shown in MPGR's Figure 1), which is slow and only approximately optimal.

### Loss & Training
There is no explicit loss term, only the reward $r$ and projection constraints. The optimizer is Adam (learning rate 0.02 for FLUX / 0.1 for SDXL-Turbo), with gradient clipping at 0.03. FLUX is run for 200 steps, and SDXL-Turbo for 50 steps. All experiments were conducted on a single A6000. Both the gradient and the latent itself are projected at each step (Appendix H provides an ablation decoupling these two).

## Key Experimental Results

### Main Results
Evaluation follows MPGR: use one of Aesthetic Score / PickScore / HPSv2 / ImageReward as the "optimized reward" and the others as "held-out" rewards to detect reward hacking. The prompt set consists of the animal dataset and T2I-CompBench++; the base one-step model is primarily FLUX-schnell.

| Method | Iterations | Aesthetic Score (target) ↑ | PickScore (held-out) | Wall-clock (s) ↓ |
|------|-------|---------------------------:|---------------------:|-----------------:|
| No Opt. | 0 | 5.99 | 0.219 | — |
| ReNO | 200 | 7.06 | 0.219 | 232.0 |
| PRNO | 200 | 7.02 | 0.218 | 255.4 |
| MPGR (Prev. SOTA) | 200 | 7.13 | 0.220 | 235.5 |
| **Ours (60 iters)** | 60 | **7.12** | 0.220 | **69.7** |
| **Ours (200 iters)** | 200 | **8.91** | 0.220 | 232.2 |

Two numbers are most notable: (1) 60 steps, 69.7s matches MPGR's 200-step, 235.5s performance—**reaching SOTA in 30% of the wall-clock time**; (2) at 200 steps, the reward for this method reaches 8.91 while baselines stall around 7.13, a gap of 1.8 points. Figure 2 shows that for all combinations of four rewards and three held-out metrics, this method forms the top-right Pareto frontier.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| No Reg. | Aesthetic spikes but image collapses | Standard reward hacking |
| $\mathcal{L}_{\text{norm}}$ ($\ell_2$ reg) | Cosine similarity 0.222 | Only constrains total energy; spatial correlation lost |
| MPGR (Spectral $\ell_1$ soft reg) | Cosine similarity 0.548; slow inner projection | Soft penalty + Slow |
| **Ours** (Hard set + closed-form) | **High similarity maintained; no hacking; single-step projection** | Latent remains aligned with initial noise (Figure 1) |

Diversity ablation (1,125 images, under Aesthetic optimization): Ours IS = 21.10 / Vendi = 6.97, consistent with the unoptimized baseline (IS 21.57–22.33, Vendi 6.42–6.61) within variance, indicating **no mode collapse**.

### Key Findings
- **Projection overhead is negligible**: The total $\mathcal{O}(N\log N)$ complexity from FFT and block-wise closed-form solutions accounts for only 0.04% of the wall-clock time.
- **"Accuracy per step" is more important than "number of steps"**: Updating within the noise-aligned subspace results in higher per-step reward gains, allowing 60 steps to suffice without hacking.
- **Hard constraints > soft regularization**: Qualitative results in Figure 3 show PRNO/MPGR occasionally generate images with artifacts or inconsistent prompts; Ours adheres to prompts.
- **The feasible set closely approximates real white noise**: A minimum cosine similarity of 0.988 for 1.1M Gaussian samples indicates the hard constraint does not "stifle" the prior but rather prunes along its natural contours.

## Highlights & Insights
- **Elegant "stripping redundancy" trick**: Using $\mathcal{F}$ to eliminate the "bad coupling" of Hermitian symmetry allows decomposing the spectral projection into independent small problems—the most elegant step, enabling closed-form projection.
- **Shift from "soft regularization" to "gradient preconditioning"**: While traditional regularization puts noise-ness in the objective, this work puts it in the **optimization geometry**. The optimizer still performs reward ascent on the original problem, but the direction is forced into a noise-aligned subspace. This paradigm can migrate to any latent gradient optimization scenario (e.g., RLHF, inference-time scaling).
- **0 hyperparameters + 0.04% overhead**: Unlike the $\lambda$ required by all baselines, this method introduces no new hyperparameters and the projection is nearly free—an attractive "subtractive contribution" for engineering.
- **Validating feasible set design via cosine similarity of real samples**: The 0.988 minimum cosine similarity for 1.1M samples is a clever experiment, proving the hard constraint does not distort the prior, which is more convincing than simple ablation.

## Limitations & Future Work
- Only one-step models were verified (FLUX-schnell, SDXL-Turbo, SANA-Sprint, SD-Turbo); application to intermediate states of multi-step diffusion/flow models remains unanswered.
- The feasible set uses theoretical expectations of $\mathcal{CN}(0,1)$, but real samples have variance. For small dimensions $N$ or few blocks $P$, strictly matching the expectation might be too restrictive.
- Assumes an isotropic Gaussian prior; no direct path for non-Gaussian priors (categorical tokens, VQ indices, etc.).
- Block size $B=16$ follows MPGR; Appendix G discusses its impact but lacks a practical guide for adaptive selection across models.
- Experiments focused solely on image generation; high-dimensional latent scenarios (video, 3D) require further validation.

## Related Work & Insights
- **vs ReNO (Eyring et al., NeurIPS 2024)**: ReNO founded this test-time route using $\ell_2$ soft regularization. This paper proves $\ell_2$ soft regularization fails to constrain spatial correlation and is easily bypassed.
- **vs PRNO (Tang et al., 2024)**: PRNO penalizes block means/covariances in the spatial domain to look "like i.i.d. Gaussian." This work moves to the spectral domain and uses hard constraints.
- **vs MPGR (Hwang et al., NeurIPS 2025)**: MPGR proposed spectral block-wise $\ell_1$ soft penalties. This work adopts the spectral block-wise idea but adds $\ell_2$ constraints, resolves Hermitian coupling, and uses closed-form projection, achieving a significant "Gain" in speed and stability.
- **vs RL fine-tuning (DDPO/RLAIF/Flow-GRPO)**: Those require retraining for each reward. This test-time method is zero-training and plug-and-play.
- **Insight**: Writing "desired statistical properties" as hard constraints + closed-form projections may be more robust than naive regularization—a paradigm useful for RLHF and Constrained Decoding.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](../../CVPR2026/image_generation/egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/image_generation/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[ICML 2026\] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)

</div>

<!-- RELATED:END -->
