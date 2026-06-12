---
title: >-
  [Paper Note] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models
description: >-
  [CVPR2026][LLM Alignment][Continual Learning] This paper proposes $\varphi$-DPO, which adopts DPO as a continual learning paradigm (using the previous-step model as the reference policy) and introduces a fairness modulat…
tags:
  - "CVPR2026"
  - "LLM Alignment"
  - "Continual Learning"
  - "DPO"
  - "Fairness"
  - "Catastrophic Forgetting"
  - "Large Multimodal Model"
  - "Focal Loss"
date: 2026-05-08
content_hash: 7e44e99dcb319f11
---

# $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models

**Conference**: CVPR2026
**arXiv**: [2602.22601](https://arxiv.org/abs/2602.22601)  
**Code**: TBD  
**Area**: LLM Alignment
**Keywords**: Continual Learning, DPO, Fairness, Catastrophic Forgetting, Large Multimodal Model, Focal Loss

## TL;DR
This paper proposes $\varphi$-DPO, which adopts DPO as a continual learning paradigm (using the previous-step model as the reference policy) and introduces a fairness modulation factor $(1-p)^\gamma$ inspired by focal loss to balance gradient contributions across data groups. The authors theoretically prove that the gradient bias approaches zero as $\gamma \to \infty$, and achieve state-of-the-art performance on the CoIN and MLLM-CL benchmarks.

## Background & Motivation
Large multimodal models (LMMs) must continuously learn new tasks in real-world deployments, making continual learning (CL) a critical capability. However, CL for LMMs faces two distinct challenges:

### Challenge 1: Catastrophic Forgetting
This is the classical problem in continual learning—performance on old tasks degrades when learning new ones. Existing mitigation strategies include:
- **Experience Replay**: Stores data from previous tasks for rehearsal, but incurs high storage overhead and may violate privacy constraints.
- **Regularization Methods (EWC, LwF, etc.)**: Constrain parameter updates to prevent overwriting prior knowledge, but overly strict constraints hinder new task learning.
- **Knowledge Distillation**: Uses the outputs of the old model as soft labels to supervise the new model, but requires additional forward pass overhead.

### Challenge 2: Fairness Degradation
This paper identifies a previously overlooked problem—**fairness degradation caused by data imbalance in continual learning**:

1. **Large disparity in group sizes**: Data volumes across CL stages vary dramatically (e.g., 100K samples in stage one vs. 10K in stage two), causing replay buffers to contain far more old-task data than new-task data.
2. **Gradient domination**: Groups with more data contribute disproportionately larger gradients, drowning out smaller groups and leading to poor model performance on them.
3. **Group fairness**: Performance disparities across different user populations or data sources constitute a potential fairness risk.

Traditional CL methods largely ignore fairness, while fairness-oriented methods (e.g., DRO, FairBatch) do not account for forgetting. The motivation behind $\varphi$-DPO is to **address both problems simultaneously**.

### Core Insight: DPO Is Naturally Suited for Continual Learning
Standard DPO relies on a **reference policy** $\pi_{\text{ref}}$ to prevent the optimized policy from deviating too far from it. The authors observe that if $\pi_{\text{ref}}$ is set to the **model from the previous CL step** $\pi_{t-1}$, DPO implicitly performs knowledge distillation—the KL divergence constraint naturally limits the deviation of the new model from the old one, thereby alleviating forgetting.

## Core Problem
How can DPO be reformulated into a unified framework that simultaneously addresses catastrophic forgetting and fairness degradation in continual learning?

## Method

### DPO as a Continual Learning Paradigm

At step $t$ of continual learning, the model is updated from $\pi_{t-1}$ to $\pi_t$. The standard DPO loss is:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{t-1}) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)\right]$$

where $y_w$ is the preferred response, $y_l$ is the rejected response, and $\beta$ is the temperature parameter. Setting the reference policy to $\pi_{t-1}$ means DPO implicitly penalizes the new policy for deviating too far from the old one.

#### Theoretical Connection: DPO and Knowledge Distillation
In Lemmas 1–2, the authors derive upper and lower bounds relating the DPO loss to KL divergence:

$$c_1 \cdot D_{\text{KL}}(\pi_{t-1} \| \pi_\theta) \leq \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{t-1}) \leq c_2 \cdot D_{\text{KL}}(\pi_{t-1} \| \pi_\theta) + C$$

where $c_1, c_2, C$ are constants depending on $\beta$. This shows that **minimizing the DPO loss is equivalent to implicitly minimizing the KL divergence between the new and old models**, i.e., performing knowledge distillation, providing a theoretical foundation for the claim that DPO is naturally suited for CL.

### $\varphi$-DPO: Fairness Modulation

While DPO mitigates forgetting, it does not address fairness issues arising from data imbalance. Inspired by focal loss, $\varphi$-DPO introduces a modulation factor:

$$\mathcal{L}_{\varphi\text{-DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[(1-p_{w,l})^\gamma \cdot \log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)\right]$$

where $p_{w,l} = \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)$ denotes the model's confidence on the current preference pair.

#### Intuition Behind the Modulation
- When the model is **already confident** on a preference pair ($p_{w,l}$ close to 1), $(1-p_{w,l})^\gamma$ approaches 0, downweighting the gradient contribution—the model no longer wastes gradient on well-learned samples.
- When the model is **uncertain** about a preference pair ($p_{w,l}$ close to 0), $(1-p_{w,l})^\gamma$ approaches 1, preserving the gradient—the model focuses on harder samples.
- Larger $\gamma$ induces more aggressive gradient redistribution.

#### Theoretical Fairness Guarantee (Lemma 3)
Let $g \in \{1, \ldots, G\}$ denote different data groups, and define the gradient bias across groups as:

$$B_\gamma(\theta) = \max_{g_1, g_2} \left|\frac{\nabla_\theta \mathcal{L}_{\varphi}^{g_1}}{\nabla_\theta \mathcal{L}_{\varphi}^{g_2}}\right|$$

The authors prove that as $\gamma \to \infty$, $B_\gamma(\theta) \to 0$—regardless of how imbalanced the data distribution is, a sufficiently large $\gamma$ equalizes the gradient contributions from all groups. Intuitively, large $\gamma$ causes the model to focus only on the hardest samples within each group, and the number of such hard samples is balanced across groups.

### Preference Pair Construction
Preference pairs are constructed for the CoIN and MLLM-CL continual learning benchmarks as follows:

1. **Preferred response $y_w$**: Human-annotated ground truth responses.
2. **Rejected response $y_l$**:
    - Generated by an LLM (e.g., GPT-4) based on the ground truth to produce plausible but incorrect responses (e.g., factual errors, subtle deviations).
    - Manually verified to ensure the rejected responses are indeed inferior to the preferred ones.
3. Each $(x, y_w, y_l)$ triplet is annotated with a group label $g$ for computing fairness metrics.

### Compatibility with Other CL Methods
$\varphi$-DPO is naturally compatible with experience replay: old and new data in the replay buffer belong to different groups, and the fairness modulation factor automatically balances their gradient contributions.

## Key Experimental Results

### CoIN Benchmark (8 Task Stages)

| Method | Final Avg Acc ↑ | Forgetting ↓ | Fairness (Worst-group Gap) ↓ |
|--------|----------------|--------------|------------------------------|
| Sequential FT | 34.2 | 42.1 | 18.3 |
| EWC | 48.7 | 28.5 | 14.2 |
| LwF | 51.3 | 25.2 | 13.8 |
| Experience Replay | 55.8 | 20.1 | 11.5 |
| DPO (as CL) | 58.2 | 16.4 | 9.7 |
| **$\varphi$-DPO** | **63.1** | **12.3** | **4.2** |

### MLLM-CL Benchmark

| Method | Domain Avg ↑ | Ability Avg ↑ | Backward Transfer ↑ | Worst-group Acc ↑ |
|--------|-------------|--------------|---------------------|-------------------|
| Sequential FT | 41.5 | 38.2 | -15.3 | 22.1 |
| LwF | 52.1 | 49.8 | -8.7 | 35.4 |
| Experience Replay | 56.3 | 53.1 | -5.2 | 40.8 |
| DPO (as CL) | 59.7 | 56.8 | -3.1 | 45.2 |
| **$\varphi$-DPO** | **65.2** | **62.4** | **-1.4** | **55.6** |

### Ablation Study
- **Effect of $\gamma$**: Fairness metrics improve monotonically from $\gamma=0$ (degenerates to standard DPO) through $\gamma=1$, $\gamma=2$, to $\gamma=5$; performance saturates at $\gamma \geq 5$.
- **DPO vs. SFT as CL paradigm**: DPO achieves 4.1% lower forgetting than SFT + KD, validating the implicit distillation effect of DPO.
- **Reference policy choice**: Using $\pi_{t-1}$ outperforms using $\pi_0$ (the initial model) by 5.2% lower forgetting, as it better preserves recently acquired knowledge.
- **Sensitivity to $\beta$**: Performance is stable for $\beta \in [0.05, 0.2]$, with $\beta = 0.1$ being optimal.

## Highlights & Insights
- **A new perspective on continual learning**: This work is the first to adopt DPO as a CL paradigm, proving that DPO inherently exhibits a knowledge distillation effect through elegant theoretical derivation.
- **Unified solution for dual problems**: A single framework simultaneously addresses forgetting and fairness, rather than stitching together two separate methods.
- **Theoretical guarantee for fairness**: Lemma 3 provides a rigorous proof that gradient bias approaches zero as $\gamma \to \infty$, rather than relying solely on empirical evidence.
- **Elegant transfer of the focal loss idea**: The focal loss mechanism, originally designed for class imbalance in object detection, is naturally and effectively transferred to the inter-group imbalance problem in continual learning.
- **Lightweight modification**: Compared to standard DPO, the only addition is a single modulation factor $(1-p)^\gamma$, introducing virtually zero additional computational cost.

## Limitations & Future Work
1. **Adaptive selection of $\gamma$**: Currently $\gamma$ is a manually tuned hyperparameter; ideally it should adapt automatically based on the degree of imbalance across groups.
2. **Dependence on preference pair quality**: Rejected responses are generated by an LLM and verified by humans, limiting scalability due to annotation costs.
3. **Insufficient validation for long CL sequences**: Experiments cover at most 8 CL stages; the cumulative drift of the $\pi_{t-1}$ reference policy under much longer sequences (e.g., 50+ stages) remains unexplored.
4. **Single $\gamma$ shared across all groups**: All groups share the same $\gamma$, whereas different groups may in practice require different degrees of modulation.
5. **Combination with parameter-efficient fine-tuning**: Current experiments use full fine-tuning; whether the implicit distillation effect of DPO holds when combined with PEFT methods such as LoRA remains to be verified.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — DPO as a CL paradigm combined with focal-inspired fairness modulation; both innovations are theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two CL benchmarks with comprehensive ablations, though the number of CL steps is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and motivation is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new "DPO for CL" research direction with a unique fairness perspective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/llm_alignment/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](../../ICLR2026/llm_alignment/token-importance_guided_direct_preference_optimization.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)

</div>

<!-- RELATED:END -->
