---
title: >-
  [Paper Note] IDLM: Inverse-distilled Diffusion Language Models
description: >-
  [ICML 2026][Model Compression][Diffusion Language Models] This paper extends "Inverse Distillation" from continuous diffusion to discrete text diffusion models. By proving that the unique optimal solution for the IDLM lo…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Diffusion Language Models"
  - "Inverse Distillation"
  - "Discrete Diffusion"
  - "Few-step Sampling"
  - "MDLM/Duo"
date: 2026-05-08
content_hash: 97883d7850796773
---

# IDLM: Inverse-distilled Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.19066](https://arxiv.org/abs/2602.19066)  
**Code**: https://david-cripto.com/idlm (Available)  
**Area**: LLM Pre-training / Diffusion Language Models / Distillation Acceleration  
**Keywords**: Diffusion Language Models, Inverse Distillation, Discrete Diffusion, Few-step Sampling, MDLM/Duo

## TL;DR
This paper extends "Inverse Distillation" from continuous diffusion to discrete text diffusion models. By proving that the unique optimal solution for the IDLM loss under SEDD/MDLM/Duo is the true data distribution—combined with simplex relaxation and Gaussian reparameterization to resolve discrete backpropagation instability—the authors compress a 1024-step teacher DLM to 16 or even 4 steps while maintaining GenPPL/Entropy and MAUVE with almost no degradation.

## Background & Motivation
**Background**: Diffusion Language Models (DLMs, such as SEDD / MDLM / UDLM / Duo) have recently approached the quality of autoregressive LMs in text generation. They work by designing a forward corruption process for discrete tokens (masking/absorbing or uniform processes) and training a denoiser for step-by-step reverse recovery. However, reverse sampling naturally requires hundreds to thousands of steps, resulting in inference latency much higher than the throughput of an autoregressive model with a single forward pass and KV-cache, hindering industrial adoption.

**Limitations of Prior Work**: While accelerating diffusion in continuous domains has been extensively studied (DDIM, progressive distillation, consistency models, DMD, etc.), porting these directly to discrete domains faces two major obstacles: (1) backpropagation must pass through categorical sampling, where hard Gumbel-Softmax is prone to instability; (2) distillation targets often cannot guarantee that the optimal solution uniquely corresponds to $p^*$. Current mainstream discrete methods like SDTT or Duo-DCD follow a consistency style, essentially teaching the student to "skip teacher trajectory segments," but they retain the teacher's position-independent decomposition, making it difficult to characterize the joint distribution between tokens in the few-step limit, often collapsing to high-frequency modes.

**Key Challenge**: The denoiser $f^*$ of a DLM is "uniquely defined" by $p^*$ (diffusion training is $f^*=\arg\min_f \mathcal{L}(f,p^*)$). Conversely, "inferring $p_\theta$ from $f^*$" in the discrete domain lacks both a uniqueness theory and a stable gradient path.

**Goal**: Generalize Inverse Distillation (from the IBMD/UID/RSD lineage) to discrete DLMs to obtain a few-step generation framework where (a) the global optimal solution is uniquely $p^*$, (b) gradients can be stably backpropagated, and (c) quality matches the 1024-step teacher at 4–16 steps.

**Key Insight**: View distillation from a different perspective—instead of forcing the student to imitate a specific teacher trajectory or marginal at a given time, ask: "If I have a distribution $p_\theta$, would running diffusion training on it recover the known teacher $f^*$?" That is, using $f^*=\arg\min_f \mathcal{L}(f, p_\theta)$ as the optimality condition for $p_\theta$.

**Core Idea**: Use the IDLM loss $\mathcal{L}_{\text{IDLM}}(\theta)=\mathcal{L}(f^*,p_\theta)-\min_f \mathcal{L}(f,p_\theta)$ as the training target for the student distribution. The authors prove that this gap is actually equal to the **KL divergence over the entire diffusion trajectory** between the student and the real data. Thus, the few-step generator uniquely recovers $p^*$ by minimizing this gap to zero.

## Method

### Overall Architecture
IDLM maintains three networks: a frozen **Teacher** $f^*$ (a multi-step DLM pre-trained on $p^*$), a learnable **Pseudo-teacher** $f$ (a denoiser refitted on the student's current distribution $p_\theta$), and a **Student Generator** $G_\theta$. The student takes input $\epsilon = (x_t, t)$ and outputs simplex vectors $G_\theta(\epsilon) \in \Delta$ as predictions for clean tokens. By sharing the same $\epsilon$ across all positions, a sequence-level mixture distribution is obtained: $p_\theta(x_0^{1:L}) = \mathbb{E}_{\epsilon}[\prod_l \text{Cat}(x_0^l; G_\theta^l(\epsilon))]$. Training alternates: **(a) fixing $\theta$ and updating $f$ with $\mathcal{L}(f, p_\theta)$ to fit the student distribution**; **(b) fixing $f$ and updating $\theta$ with the IDLM gap $\mathcal{L}(f^*,p_\theta)-\mathcal{L}(f,p_\theta)$**. Inference follows the same reverse sampler as the teacher but with a grid of only 4–32 steps.

### Key Designs

1.  **IDLM Inverse Distillation Objective + Uniqueness Theorem**:
    *   **Function**: Transforms the inverse proposition—"for which distribution is the fixed teacher $f^*$ still optimal"—into an optimizable loss.
    *   **Mechanism**: Defines $\mathcal{L}_{\text{IDLM}}(\theta) = \mathcal{L}(f^*, p_\theta) - \min_f \mathcal{L}(f, p_\theta)$. The first term is the denoising loss of the teacher on student samples; the second is the lower bound achievable by an "optimally trained denoiser on student samples." The difference measures "how optimal the teacher remains." Theorem 3.1 proves: For SEDD / MDLM / Duo (in the $\tau\to 0^+$ limit), $\mathcal{L}_{\text{IDLM}}(\theta) \geq \mathcal{D}_{\text{KL}}(\mathbb{P}^\theta \| \mathbb{P}^*) \geq 0$, where equality holds if and only if $p_\theta = p^*$. The proof path expresses this gap as the KL divergence of trajectory distributions: $\mathcal{L}_{\text{IDLM}}(\theta)=\mathcal{D}_{\text{KL}}(\mathbb{P}^\theta \| \mathbb{P}^*)$, controlled by the data processing inequality at the terminal marginal.
    *   **Design Motivation**: Direct KL matching of marginals ($\mathcal{L}_{\text{DMD}}=\int w(t) D_{\text{KL}}(p_t^\theta \| p_t^*) dt$) only looks at "slices" of time. In processes like uniform diffusion where tokens can be repeatedly modified, this loses trajectory coupling signals. IDLM matches the entire path distribution, allowing few-step students to learn the joint structure between tokens rather than simple position-independent conditions.

2.  **Simplex Relaxation + Modality-specific Differentiability Tricks**:
    *   **Function**: Solves the difficulty of backpropagating through categorical sampling in the discrete domain.
    *   **Mechanism**: Relaxes the generator output range from the one-hot set $\mathcal{V}$ to the probability simplex $\Delta$, making the cross-entropy term a soft-label loss while keeping the forward corruption $q_t(\cdot \mid G_\theta(\epsilon))$ a valid categorical distribution. For MDLM, it leverages the property of subs-parameterization: unmasked positions $f^*(x_t,t)=f(x_t,t)=x_t$ cancel out, and the IDLM gap is non-zero only when $x_t=m$. Thus, generator updates simplify to $-\mathbb{E}_{\epsilon,t}[(1-\alpha_t)\lambda_t \langle G_\theta(\epsilon), \log f(m,t)\rangle]$, and the sampled token $x_t$ vanishes from the gradient path of $\theta$ (implicit stop-gradient). For Duo, Gaussian reparameterization $x_t = \text{softmax}((\tilde{\alpha}_t G_\theta(\epsilon)+\sqrt{1-\tilde{\alpha}_t^2}\xi)/\tau)$ is used to make $G_\theta(\epsilon) \mapsto x_t$ differentiable.
    *   **Design Motivation**: Hard Gumbel-Softmax exhibits high gradient variance in multi-step training. The mask-only simplification in MDLM turns "gradients only flowing through simplex outputs" into a naturally stable path. For Duo, the denoiser already trained on the simplex works with Gaussian relaxation to replace non-differentiable sampling with a continuous differentiable approximation.

3.  **Sequence-level mixture + Alternating Optimization**:
    *   **Function**: Fits sequence-level joint distributions into a few-step generator using a low-dimensional latent variable while stably approximating the inner $\min_f$.
    *   **Mechanism**: Parameterizing $p_\theta$ directly on $\mathcal{V}^L$ would require $N^L$ probabilities. The authors use a mixture: sample $\epsilon\sim p_\mathcal{E}$ first; given $\epsilon$, each position is independent on $\Delta$, but $\epsilon$ is shared across positions $\to$ each mixture component is equivalent to a sentence-level selection. Training follows the DMD lineage using alternating updates: each step either fits the pseudo-teacher using $\mathcal{L}_f=\mathcal{L}(f,p_\theta)$ or updates the student using $\mathcal{L}_{\text{IDLM}}$. Inference takes $\epsilon=(x_t,t)$ from partially noisy real data and reuses the teacher's reverse sampler for 4–32 steps.
    *   **Design Motivation**: Moving from pure noise to a clean sentence in one step is extremely difficult. Multi-step parameterization allows the student to handle intermediate states, significantly reducing one-step optimization difficulty. The mixture structure balances "position-wise differentiability" with "sequence-level joint modeling."

### Loss & Training
The general token-level objective is $\mathcal{L}(f,p)=\mathbb{E}_{p(x_0),t,q_t}[g(x_t,x_0,f(x_t,t))]$, summed over all positions for the sequence level. The final IDLM gradient signal can be interpreted as a token advantage vector $a_t = \log f^*(m,t)-\log f(m,t)$: the student does not just chase the teacher's highest probability token but pushes tokens that the "teacher prefers more than the pseudo-teacher." When $f$ catches up to $f^*$, the advantage disappears, leading to natural convergence. Both the student and pseudo-teacher are initialized with teacher weights.

## Key Experimental Results

### Main Results
Unconditional generation based on OpenWebText (OWT), comparing GenPPL ↓ / MAUVE ↑ / Entropy ↑ / GM ↓ under equivalent sampling steps.

| Setting | Steps | GenPPL ↓ | MAUVE ↑ | Entropy ↑ | Notes |
|------|------|---------|---------|----------|-----|
| MDLM Teacher | 1024 | 41.29 | 0.89 | 5.28 | Teacher Upper Bound |
| SDTT | 16 | 61.34 | 0.88 | 5.36 | Consistency Distillation |
| SDTT+Di4C2 | 16 | 44.82 | 0.90 | 5.34 | + Latent Mixture |
| DiDi-Instruct | 16 | 38.19 | – | 5.21 | DMD-style |
| **IDLM-MDLM (Ours)** | **16** | **32.75** | **0.93** | 5.42 | 64× Speedup |
| Duo Teacher (greedy) | 1024 | 71.72 | 0.90 | 5.22 | Teacher |
| Duo-DCDg | 4 | 96.24 | 0.69 | 4.93 | Consistency |
| **IDLM-DCDg (Ours)** | **4** | **77.47** | **0.89** | **5.28** | 256× Speedup |

On MDLM, 16 steps compress the 1024-step teacher's GenPPL from 41.29 to 32.75 while MAUVE improves to 0.93. For the Duo route, IDLM-DCDg significantly outperforms Duo-DCDg at both the 8-step and 4-step limits, pulling MAUVE back to 0.89 from 0.69 at 4 steps.

### Key Experimental Results (GSM8K / TinyGSM)
| Model | Steps | Accuracy (%) | Speedup |
|------|------|--------------|--------|
| MDLM Teacher | 1024 | 18.0 | 1× |
| **Ours-MDLM** | **128** | **19.86** | 8× |
| Duo Teacher | 1024 | 17.2 | 1× |
| **Ours-Duo** | **64** | **19.03** | 16× |
| Autoregressive (greedy) | 512 | 63.3 | – |

The few-step student **slightly exceeds** teacher accuracy even with a much smaller step budget, indicating that IDLM's inverse distillation does not sacrifice sequence-level correctness.

### Key Findings
- **64× Speedup with no loss in quality**: The 16-step MDLM student matches or exceeds the 1024-step teacher across GenPPL, MAUVE, and Entropy, proving more stable than SDTT/Di4C2/DiDi-Instruct at low step counts.
- **Superiority at the low-step limit**: The gap between IDLM and consistency methods is largest at 4–8 steps, validating the argument regarding "trajectory-level KL vs. marginal-level KL."
- **GenPPL–Entropy tradeoff**: Focusing only on GenPPL can be deceptive (e.g., FLM/ELF). IDLM reduces GenPPL while maintaining entropy close to the teacher, representing genuine improvement rather than mode collapse.
- **Ablation**: On MDLM, the original mask-only update outperforms variants with Gaussian noise. On Duo, full IDLM is more stable than the stop-gradient version (≈DMD), showing trajectory matching is crucial in uniform diffusion.
- **Scalability**: The same patterns hold on the 0.9B MDLM provided by SDTT.

## Highlights & Insights
- **The Elegance of the Inverse Perspective**: Replacing "student imitates teacher" with "student distribution makes the teacher still optimal" allows the distillation loss to be written naturally as $\mathcal{L}(f^*,p_\theta)-\min_f \mathcal{L}(f,p_\theta)$. This connects directly to trajectory KL through theorems, unifying discrete diffusion with the continuous IBMD/UID/RSD lineage.
- **Beauty of MDLM's Mask-only Simplification**: Subs-parameterization causes already "revealed" tokens to cancel out in the IDLM gap. Gradients only flow through simplex outputs, acting as a natural stop-gradient and single-token loss. This trick is simple, stable, and transferable to any "copy-forward" discrete diffusion variant.
- **Reusable Sequence-level Mixture Concept**: Using a shared latent to upgrade position-independent decomposition to a mixture is a lightweight solution for capturing joint distributions in few-step discrete diffusion. While VADD/Di4C2 use similar paths, IDLM unifies this with trajectory KL.

## Limitations & Future Work
- The authors acknowledge that direct one-step distillation failed; multi-step parameterization $\epsilon=(x_t,t)$ is necessary. IDLM is not a true one-step generator but rather "compresses 1024 steps into 4–16."
- The uniqueness result in Theorem 3.1 for Duo requires the $\tau\to 0^+$ limit; the gap between actual training with finite $\tau$ and the optima is not explicitly characterized.
- Evaluation focuses on OWT unconditional generation and GSM8K. More complex dialogue, code, or long-context scenarios remain unverified. The gap with autoregressive LMs in instruction following remains large (19.86% vs 63.3% on GSM8K).
- Training cost: Alternating updates for $f$ and $\theta$ is essentially co-training two models. Engineering overhead is comparable to the DMD lineage.

## Related Work & Insights
- **vs SDTT / Duo-DCD (consistency-style)**: They train students as "jump-step" approximators of the teacher's trajectory, retaining the teacher's position-independent decomposition. IDLM uses sequence-level mixtures and trajectory KL, which theoretically recovers $p^*$ and outperforms them at 4–16 steps.
- **vs DiDi-Instruct / D-MMD / Di[M]O (DMD-style)**: They match KL at each time step $\int w(t) D_{\text{KL}}(p_t^\theta\|p_t^*) dt$, a special case of IDLM with stop-gradients. While similar for MDLM, IDLM's full trajectory KL proves advantageous for processes like Duo where tokens change repeatedly.
- **vs IBMD / UID / RSD (continuous domain inverse distillation)**: Shares the core philosophy. IDLM is the first rigorous extension to discrete DLM, solving discrete backpropagation and uniqueness obstacles.
- **vs FLM / ELF (continuous-space language flows)**: They shift the state space to one-hot or embedding flows, achieving low GenPPL at the cost of Entropy. IDLM stays in the native discrete space, reducing PPL while preserving entropy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rigorous extension of Inverse Distillation to discrete DLMs with uniqueness theorems and trajectory KL interpretations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered MDLM/Duo/SEDD/0.9B + OWT + GSM8K + 4–32 step sweeps, though lacking larger models and dialogue benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain: theory → simplification → implementation → pseudo-code → experiments. Advantage vectors and mixture explanations are intuitive.
- Value: ⭐⭐⭐⭐⭐ Compressing DLM inference to 4–16 steps without quality loss is a key piece in making diffusion language models practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery](diverdiving_deeper_into_distilled_data_via_expressive_semantic_recovery.md)
- [\[ICLR 2026\] ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping](../../ICLR2026/model_compression/es-dllm_efficient_inference_for_diffusion_large_language_models_by_early-skippin.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
