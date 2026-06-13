---
title: >-
  [Paper Note] ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning
description: >-
  [ICLR 2026][Reinforcement Learning][Mixture-of-LoRAs] ReMix identifies a severe routing weight collapse problem in existing Mixture-of-LoRAs models (even when $k>1$ LoRAs are activated…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Mixture-of-LoRAs"
  - "routing weight collapse"
  - "reinforcement learning routing"
  - "RLOO"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: 3db7145a5725193d
---

# ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning

**Conference**: ICLR 2026
**arXiv**: [2603.10160](https://arxiv.org/abs/2603.10160)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Mixture-of-LoRAs, routing weight collapse, reinforcement learning routing, RLOO, parameter-efficient fine-tuning

## TL;DR

ReMix identifies a severe routing weight collapse problem in existing Mixture-of-LoRAs models (even when $k>1$ LoRAs are activated, the effective LoRA count rapidly degenerates to 1), proposes non-learnable constant routing weights to ensure equal contribution from all activated LoRAs, and trains the router using the RLOO reinforcement learning gradient estimator, significantly outperforming state-of-the-art PEFT methods.

## Background & Motivation

**Background**: LoRA is the most popular parameter-efficient fine-tuning method. Mixture-of-LoRAs extends model capacity by maintaining multiple LoRAs per layer and using a router to select a subset. Existing methods (e.g., MixLoRA, HydraLoRA) employ learnable routing weights computed via softmax.

**Limitations of Prior Work**: The authors identify a fundamental and severe flaw in existing Mixture-of-LoRAs routers — **routing weight collapse**. Even when $k>1$ LoRAs are designated for activation, softmax routing weights rapidly concentrate on a single LoRA during training (effective support size ESS drops to 1), while the weights of all other LoRAs approach zero. This renders the computation of the additional $k-1$ LoRAs entirely wasteful.

**Key Challenge**: Learnable routing weights permit end-to-end training but inherently tend toward imbalance — Theorem 1 proves that under Gaussian initialization, the effective LoRA count is extremely small with high probability (e.g., among 8 LoRAs, there is an 84% probability that only $\leq 2$ are effective). This imbalance further intensifies during training.

**Goal**: (1) Theoretically and empirically expose the routing weight collapse problem; (2) design a router that does not collapse; (3) address the non-differentiability introduced by non-learnable weights.

**Key Insight**: Fundamentally rethink router design — abandon learnable weights and instead use constant weights to ensure equal contribution from all activated LoRAs. The resulting non-differentiability is resolved by reformulating the problem as a reinforcement learning task.

**Core Idea**: Constant routing weights $\omega$ eliminate collapse (ensuring $ESS = k$); the RLOO gradient estimator trains the router for LoRA selection; top-$k$ selection is used at inference (theoretically proven to be optimal when the router is sufficiently trained).

## Method

### Overall Architecture

Each layer maintains $n$ LoRAs and one router. The router produces a probability distribution $\mathbf{q}^{(l)} = \text{softmax}(\mathbf{P}^{(l)}\mathbf{x}^{(l)})$. During training, $k$ LoRAs are sampled without replacement from this distribution and assigned a constant weight $\omega$, yielding $\mathbf{y}^{(l)} = \mathbf{W}^{(l)}\mathbf{x}^{(l)} + \omega \sum_{j=1}^{k} \mathbf{B}_{i_j}^{(l)}\mathbf{A}_{i_j}^{(l)}\mathbf{x}^{(l)}$. Router gradients are estimated via RLOO. At inference, the $k$ LoRAs with the highest probabilities are selected deterministically.

### Key Designs

1. **Non-Learnable Constant Routing Weights**:

    - Function: Fundamentally eliminates routing weight collapse, ensuring equal contribution from all activated LoRAs.
    - Mechanism: The routing weight is replaced from a learnable softmax output to a fixed constant $\omega$; activated LoRAs receive weight $\omega$ while inactive ones receive 0. $\omega$ can follow the LoRA scaling $2/(kr)$ or rsLoRA scaling $2/\sqrt{kr}$. This guarantees $ESS(\boldsymbol{\pi}^{(l)}) = k$, meaning the effective LoRA count permanently equals the activation count, fundamentally preventing collapse.
    - Design Motivation: With fixed constant weights, the problem transforms from "how to allocate weights" to "how to select a LoRA subset." Every selected LoRA must contribute fully, leaving no possibility for any LoRA to be marginalized.

2. **RLOO Reinforcement Learning Gradient Estimator**:

    - Function: Provides an unbiased gradient estimate for the non-differentiable discrete LoRA selection.
    - Mechanism: Router training is framed as an RL problem — the SFT loss $\mathcal{L}(\mathfrak{I})$ serves as the negative reward, and the routing distribution $\mathbf{q}^{(l)}$ serves as the policy. $M$ selections $\mathfrak{J}_1, \ldots, \mathfrak{J}_M$ are sampled independently, each containing LoRA selections across all layers. The RLOO gradient estimator is $\hat{\mathbf{G}}_{\mathbf{P}^{(l)}} = \frac{1}{M-1}\sum_{m}(\mathcal{L}(\mathfrak{I}_m) - \bar{\mathcal{L}})\nabla_{\mathbf{P}^{(l)}}\log Q(\mathfrak{J}_m)$, where $\bar{\mathcal{L}}$ is a mean baseline for variance reduction. This estimator is unbiased.
    - Design Motivation: Standard REINFORCE suffers from excessive variance. RLOO computes the baseline in a leave-one-out manner — using the average loss of all other samples — effectively reducing variance without requiring an additional value network.

3. **Top-$k$ Inference Selection with Theoretical Guarantees**:

    - Function: Deterministically selects the optimal LoRA subset at inference time.
    - Mechanism: Theorem 2 proves that as long as the router is sufficiently trained (the optimal subset is sampled with probability $> 50\%$), top-$k$ selection is guaranteed to recover the optimal subset. Intuitively, if the optimal subset $\mathcal{I}^*$ is sampled with the highest probability, each LoRA in $\mathcal{I}^*$ also has the highest marginal probability, so top-$k$ selection recovers $\mathcal{I}^*$.
    - Design Motivation: While stochastic sampling is necessary during training for exploration, it introduces unnecessary randomness at inference. Top-$k$ is the theoretically optimal deterministic strategy.

### Loss & Training

LoRA parameters are updated with standard SFT gradients $\nabla_{\mathbf{A},\mathbf{B}}\mathcal{L}(\mathfrak{I})$. Router parameters are updated with the RLOO gradient estimator. Training computation can be scaled by increasing the number of samples $M$ — a unique advantage of ReMix, since baseline methods have fixed training computation. Llama 3 8B is used as the base model, trained with LLaMA-Factory.

## Key Experimental Results

### Main Results

| Method | GSM8K | HumanEval Pass@1 | ARC-c | Average | Parameters |
|--------|-------|------------------|-------|---------|------------|
| LoRA | 59.21 | 26.83 | 83.05 | 56.36 | 0.112B |
| rsLoRA | 62.47 | 28.66 | 82.71 | 57.95 | 0.028B |
| MixLoRA | 61.87 | 28.05 | 82.37 | 57.43 | 0.101B |
| HydraLoRA | 62.47 | 20.12 | 82.71 | 55.10 | 0.084B |
| **ReMix** | **65.66** | **32.93** | **83.73** | **60.77** | **0.070B** |

ReMix consistently surpasses all baselines across three benchmarks, with an average gain of 2.82 accuracy points.

### Ablation Study

| Configuration | GSM8K Accuracy | Notes |
|---------------|---------------|-------|
| Full ReMix | Highest | RLOO + top-$k$ |
| w/o RLOO | Significant drop | Insufficient router training |
| w/o top-$k$ (random sampling at inference) | Drop | Unnecessary randomness introduced |
| Rank-$kr$ LoRA ($k=4$, $r=8$) | 59.21 | Single high-rank LoRA |
| $k$ Rank-$r$ LoRAs (ReMix) | 64.22 | Confirms diverse subset activation |
| Training compute $M=2\to32$ | 56.03→58.83 | Consistent improvement |

### Key Findings

- **Routing weight collapse** genuinely occurs in MixLoRA and deteriorates rapidly — ESS drops from ~4 at initialization to 1 within 1,000 steps and never recovers.
- ReMix ($k=4$, $r=8$) substantially outperforms a Rank-32 LoRA (64.22 vs. 59.21), confirming that ReMix activates diverse LoRA subsets rather than repeatedly selecting the same one.
- ReMix's training computation is scalable (increasing $M$ from 2 to 32 yields consistent improvement), a unique advantage absent in baseline methods.
- A 10% increase in training time yields a 15.97% relative accuracy improvement, demonstrating high efficiency.

## Highlights & Insights

- **Theoretical characterization of routing weight collapse** is among the paper's most important contributions — Theorem 1 derives a probabilistic upper bound on ESS, rigorously formalizing a pervasive yet overlooked problem. This finding carries cautionary implications for all MoE architectures employing softmax routing.
- **Training the router with RL** is an elegant design choice — constant weights transform the router's task from "weighting" LoRAs to "selecting" LoRAs, which is precisely a discrete decision problem naturally suited to RL. The adoption of RLOO simultaneously addresses gradient estimation and variance control.
- **Scalable training computation** is a distinctive advantage — for performance-critical scenarios, one can directly increase $M$ to improve results without altering the model architecture.

## Limitations & Future Work

- The additional $M$ forward passes increase training cost (though only by ~10% per step).
- Theoretical analysis is grounded in Gaussian initialization; the collapse mechanism in later stages of training may be more complex.
- Validation is limited to Llama 3 8B; generalizability to larger models and more diverse tasks remains to be confirmed.
- Whether constant routing weights are the only viable solution is an open question; progressive weight balancing (e.g., introducing a load balancing loss) may also be effective.

## Related Work & Insights

- **vs. MixLoRA (Li et al., 2024)**: MixLoRA uses standard learnable routing weights, which the paper demonstrates are subject to collapse; ReMix eliminates collapse via constant weights and RL.
- **vs. load balancing in MoE**: Methods such as Switch Transformer apply auxiliary losses to balance expert utilization across samples; the present work focuses on balancing routing weights across LoRAs within a single sample.
- **vs. VB-LoRA (Li et al., 2024)**: VB-LoRA shares LoRA parameters via vector quantization, achieving higher parameter efficiency at the cost of lower performance; ReMix strikes a better balance between parameter efficiency and performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The theoretical discovery of routing weight collapse combined with the RL-based routing solution are both highly insightful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmarks, detailed ablations, and analyses of efficiency and scalability.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem motivation is exceptionally strong; theory and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐⭐ — Delivers a fundamental improvement to the MoE/MoLoRA paradigm with practical plug-and-play utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Routing, Cascades, and User Choice for LLMs](routing_cascades_and_user_choice_for_llms.md)
- [\[ICLR 2026\] Trinity: An Evolved LLM Coordinator](trinity_an_evolved_llm_coordinator.md)
- [\[ICLR 2026\] References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
