---
title: >-
  [Paper Note] Celo2: Towards Learned Optimization Free Lunch
description: >-
  [ICLR 2026][Optimization][learned optimizer] This paper proposes Celo2—a learned optimizer meta-trained in only 4.5 GPU hours—that achieves stable generalization to models up to 1 billion parameters (GPT-3 XL, 1.3B)…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "learned optimizer"
  - "meta-learning"
  - "meta-generalization"
  - "normalized update rule"
  - "AdamW alternative"
date: 2026-05-08
content_hash: 1e587a3026df1cd1
---

# Celo2: Towards Learned Optimization Free Lunch

**Conference**: ICLR 2026
**arXiv**: [2602.19142](https://arxiv.org/abs/2602.19142)  
**Code**: [https://github.com/amoudgl/celo2](https://github.com/amoudgl/celo2)  
**Area**: Optimization
**Keywords**: learned optimizer, meta-learning, meta-generalization, normalized update rule, AdamW alternative

## TL;DR

This paper proposes Celo2—a learned optimizer meta-trained in only 4.5 GPU hours—that achieves stable generalization to models up to 1 billion parameters (GPT-3 XL, 1.3B), which is 6 orders of magnitude beyond the meta-training distribution, via simple recipes including a normalized MLP update rule and task augmentation. Celo2 outperforms the prior VeLO optimizer (which required 4,000 TPU-months of meta-training) and carefully tuned AdamW baselines.

## Background & Motivation

Foundation model pretraining dominates modern computational workloads, and optimizer choice—typically Adam or AdamW—directly impacts training efficiency. Learned optimizers (LOs) aim to discover update rules through meta-learning that surpass hand-designed counterparts. However, this direction faces three core challenges:

**Meta-generalization difficulty**: Optimizers meta-trained on small-scale tasks often fail to generalize to large-scale ones. VeLO, the strongest prior learned optimizer, failed to generalize beyond 600M parameters despite 4,000 TPU-months of meta-training (roughly 10× the compute of GPT-3 training).

**High meta-training cost**: VeLO's compute budget makes iterative research on learned optimizers extremely slow.

**Instability**: Learned optimizers tend to exhibit unstable training dynamics when deployed outside their training distribution, limiting practical adoption.

Root Cause: How can strong meta-generalization be achieved at minimal meta-training cost?

This paper offers a surprising answer: through careful design of a simple normalized optimizer architecture combined with augmented meta-training strategies, a high-performing general-purpose learned update rule can be meta-trained in just 4.5 GPU hours, stably scaling to tasks 6 orders of magnitude larger than the meta-training distribution (GPT-3 XL, 1.3B parameters).

Core Idea: Rather than learning the step size and scheduler (decoupled to user-side tuning), Celo2 learns only the normalized update direction. This decoupling yields update rules with stronger task invariance and scale generalizability.

## Method

### Overall Architecture

Celo2 is a drop-in Optax optimizer transformation that can replace AdamW with a single line of code:
- **Meta-training phase**: A small MLP update rule is trained on four 8×8 image classification MLP tasks, requiring only 4.5 GPU hours.
- **Deployment phase**: The learned update rule is inserted as an Optax transformation into the standard training pipeline, used alongside a learning rate schedule and weight decay.
- Supports modern optimization techniques: orthogonalization (Newton-Schulz), separate update rules for 1D/2D parameters, and decoupled weight decay.

### Key Designs

1. **Normalized Learned Update**:
   This is Celo2's most central design innovation. Prior learned optimizers use raw MLP outputs directly as update steps; Celo2 instead applies **RMS normalization** to the MLP output:
   $$\Delta\mathbf{p}_t = \frac{\text{MLP}(\mathbf{F})}{\text{RMS}(\text{MLP}(\mathbf{F}))}$$

   This seemingly simple change yields multiple benefits:
   - Forces the MLP to learn **task-invariant update directions** rather than task-dependent raw step magnitudes during meta-training.
   - Produces training dynamics consistent with AdamW (weight norm curves align, as shown in Figure 2).
   - Prevents gradient explosion or vanishing when deployed to larger-scale tasks.

   The authors also compare alternative normalization schemes (rolling RMS, clipped normalization, etc.) and find that simple per-step RMS normalization performs best (Table 2).

2. **Tunable Step-size Decoupling**:
   Unlike VeLO and Celo, Celo2 **does not learn a learning rate scheduler**, leaving step-size adjustment to the user. This requires tuning one additional hyperparameter (the learning rate) but enables reliable generalization to large-scale tasks. This trade-off is critical: the predecessor Celo, by learning the scheduler, actually failed to generalize to large-scale tasks.

3. **Simple MLP Architecture**:
   Celo2 uses a 2-layer MLP with 8 hidden units and ReLU activations, totaling fewer than 200 parameters. Per-parameter input features include:
   - Three momentum accumulators ($\beta_1, \beta_2, \beta_3 = 0.9, 0.99, 0.999$)
   - One RMS gradient accumulator ($\beta_4 = 0.95$)
   - Adafactor row/column features

   The MLP outputs only the direction $\mathbf{d}$ (not the magnitude $\mathbf{m}$), which is the optimal configuration identified through ablation (Table 1e).

4. **Orthogonalization Compatibility**:
   Celo2 is highly compatible with the Newton-Schulz orthogonalization used in the Muon optimizer. Applying orthogonalization to the learned MLP update (rather than standard momentum) further improves performance. Figure 4 shows the combined effect: Celo2-base + orthogonalization + Adam for 1D parameters yields progressive improvement when stacked.

5. **Task Augmentation**:
   During meta-training, parameters of the inner-loop network are randomly scaled ($\alpha \sim \text{LogUniform}(0.001, 1000)$) to simulate a broader range of optimization landscapes. This technique is essential for strong generalization (ablation Table 1c: removing task augmentation raises loss from 3.812 to 4.417).

### Loss & Training

**Meta-training setup:**
- Tasks: Four 8×8 image classification MLPs (MNIST, Fashion-MNIST, CIFAR-10, SVHN)
- Meta-optimization: Persistent Evolution Strategies (PES), avoiding gradient bias from long unrolls
- Inner-loop steps: $K=50$; unroll length sampled log-uniformly from [100, 2000]
- Meta-objective: Average loss over the unroll
- Total compute: 100K outer-loop iterations, 8 parallel tasks, on an Nvidia L40S GPU
- Total: approximately **4.5 GPU hours**

**Deployment setup:**
- Learning rate search: 7 values, log-uniformly distributed over $[10^{-5}, 10^{-3}]$
- Weight decay: 0.0, 0.1, 10.0
- Scheduler: cosine decay with linear warmup (5%)
- Precision: float32 by default (bfloat16 also stable on ImageNet)

## Key Experimental Results

### Main Results

**Language modeling (out-of-distribution generalization):**

| Task | Parameters | Scale ratio | Celo2 | AdamW | VeLO |
|------|-----------|------------|-------|-------|------|
| LM-30M | 30M | 30,000× | Competitive | Baseline | Competitive |
| GPT-2 | 124M | 124,000× | Slightly better | Baseline | Competitive |
| GPT-3 XL | 1.3B | 1,000,000× | **Competitive** | Baseline | **Fails to generalize** |

This is the first time a learned optimizer has successfully generalized to billion-parameter-scale pretraining tasks. GPT-3 XL lies **6 orders of magnitude** outside the meta-training distribution.

**ImageNet ViT classification (long-horizon generalization, 50K steps = 25× meta-training unroll length):**

| Metric | Celo2 | AdamW | VeLO |
|--------|-------|-------|------|
| Steps to reach VeLO's final loss | ~50% | Slower | 100% |
| Final validation accuracy | ~66% | ~66% | ~66% |
| Training stability | High (consistent with AdamW) | High | Atypical dynamics |

Celo2 reaches VeLO's final loss in approximately 50% of the steps required by VeLO.

**Reinforcement learning (Atari PPO, generalization under high-variance gradients):**

| Environment | Celo2 | AdamW | VeLO |
|-------------|-------|-------|------|
| Asterix | On par with AdamW | Baseline | Significantly behind / stalled |
| Freeway | On par with AdamW | Baseline | Significantly behind / stalled |
| SpaceInvaders | On par with AdamW | Baseline | Significantly behind / stalled |

VeLO stalls on all RL tasks (consistent with VeLO's original paper, Figure 11), while Celo2 remains stable throughout.

### Ablation Study

| Configuration | Validation loss (LM-30M) | Notes |
|--------------|--------------------------|-------|
| Hidden size = 8 (default) | **3.812** | Optimal |
| Hidden size = 4 | 4.128 | Too small |
| Hidden size = 16 | 3.857 | Larger hurts |
| RMS decay $\beta=0.95$ (default) | **3.812** | Optimal |
| $\beta=0.999$ | 3.893 | Classic Adam setting |
| With task augmentation (default) | **3.812** | Critical component |
| Without task augmentation | 4.417 | Severe degradation |
| With normalization (default) | **3.812** | Critical component |
| Without normalization | 3.961 | Clear degradation |
| Output direction $\mathbf{d}$ only (default) | **3.812** | Optimal |
| Output both $\mathbf{d}$ and $\mathbf{m}$ | 3.900 | Magnitude output is harmful |

### Key Findings

- **Normalization is key to generalization**: RMS-normalizing the MLP output aligns training dynamics with AdamW and serves as the core mechanism enabling cross-scale generalization.
- **Task augmentation is indispensable**: Removing it raises loss from 3.812 to 4.417 (+16%), underscoring the importance of gradient landscape diversity for meta-generalization.
- **Celo2 is competitive with Muon**: On GPT-2, Celo2 (3.35588 or 3.36785) is nearly identical to Muon (3.35636) (Figure 7); the only difference between the two is the update rule—Muon uses momentum while Celo2 uses a learned MLP.
- **Runtime and memory overhead**: Celo2-base matches Adam in wall-clock time; memory overhead is approximately 5× (3 momentum buffers + 1 RMS + Adafactor features, vs. Adam's 3×); adding orthogonalization increases wall-clock time to 1.3×.

## Highlights & Insights

- **A surprising "free lunch"**: Only 4.5 GPU hours of meta-training yields a practical general-purpose optimizer—a compute efficiency improvement of 5–6 orders of magnitude compared to VeLO's 4,000 TPU-months.
- **A philosophical shift in design**: From "learn everything" (VeLO learns update rules + scheduler + step size) to "learn only the update direction"—the less is learned, the better the generalization.
- **The power of normalization**: A simple RMS normalization elevates learned optimizers from "toy" to "practical" status.
- **Training a GPT-3 optimizer on 8×8 image classification**: The stark contrast between the simplicity of the meta-training tasks and the complexity of deployment tasks highlights the fundamental universality of the learned update rule.
- **Complementarity with Muon**: Celo2 extends the Muon orthogonalization framework from hand-crafted momentum orthogonalization to learned update rule orthogonalization.

## Limitations & Future Work

- **Requires learning rate tuning**: Unlike VeLO's self-tuning mode, Celo2 requires a learning rate search (7 candidate values), which, though modest in scope, raises the barrier to use.
- **Higher memory overhead**: The 5× parameter memory overhead exceeds Adam's 3×, which may be problematic in memory-constrained settings.
- **Homogeneous meta-training tasks**: Meta-training on only four 8×8 image classification MLPs is limiting; a more diverse task distribution could yield better generalization.
- **Insufficient testing under mixed precision**: The authors acknowledge float32 as the default; bfloat16 has only been preliminarily evaluated on ImageNet.
- **Scheduler not learned**: Although decoupling the step size is critical for generalization, how to safely incorporate scheduler learning remains an open question.
- **Limited comparison with newer optimizers**: Comparisons are restricted to AdamW, VeLO, and Muon; newer methods such as SOAP and AdaMuon are not evaluated.

## Related Work & Insights

- **VeLO (Metz et al., 2022)**: The strongest prior learned optimizer, requiring 4,000 TPU-months of compute, with a generalization ceiling of approximately 600M parameters.
- **Celo (Moudgil et al., 2025)**: The predecessor to Celo2, achieving better compute efficiency than VeLO in 24 GPU-hours but at a performance cost.
- **Muon (Jordan et al., 2024)**: A hand-designed orthogonalization-based optimizer that excels in the NanoGPT speed competition; Celo2 is highly compatible with it.
- **SOAP (Vyas et al., 2024)**: Explores optimization within module norms; complementary in direction to Celo2.
- **Insight**: Celo2's success suggests the existence of a low-dimensional "universal update rule space" that a sub-200-parameter MLP can capture, offering a new perspective for understanding the essence of optimization algorithms.

## Rating

- Novelty: ⭐⭐⭐⭐ — The design decisions of normalization and step-size decoupling are simple yet surprisingly effective, with thorough ablation support.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers language modeling (30M→1.3B), ImageNet ViT, and Atari RL, spanning multiple domains and scales.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear, ablations are well-organized, and the appendix provides complete code and background.
- Value: ⭐⭐⭐⭐⭐ — A milestone contribution to the learned optimizer field; the first to achieve practical generalization to billion-parameter scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](../../NeurIPS2025/optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](../../ICML2026/optimization/ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] RACO: Reward-free Alignment for Conflicting Objectives](../../ICML2026/optimization/reward-free_alignment_for_conflicting_objectives.md)

</div>

<!-- RELATED:END -->
