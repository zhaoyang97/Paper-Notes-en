---
title: >-
  [Paper Note] MESA: Improving MoE Safety Alignment via Decentralized Expertise
description: >-
  [ICML 2026][LLM Alignment][MoE Safety] MESA reformulates MoE safety alignment as a resource allocation problem of "decentralizing safety responsibility across experts." It employs KL-regularized Sinkhorn Optimal Transpor…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "MoE Safety"
  - "Safety Sparsity"
  - "Optimal Transport"
  - "Routing Alignment"
  - "Expert Selection"
date: 2026-05-08
content_hash: 709c96c6a9feabb5
---

# MESA: Improving MoE Safety Alignment via Decentralized Expertise

**Conference**: ICML 2026  
**arXiv**: [2606.00651](https://arxiv.org/abs/2606.00651)  
**Code**: https://github.com/lorraine021/MESA (Available)  
**Area**: Alignment RLHF / LLM Safety / MoE  
**Keywords**: MoE Safety, Safety Sparsity, Optimal Transport, Routing Alignment, Expert Selection

## TL;DR
MESA reformulates MoE safety alignment as a resource allocation problem of "decentralizing safety responsibility across experts." It employs KL-regularized Sinkhorn Optimal Transport (OT) to identify the lowest-cost subset of experts from the "shoulder region" for SFT. Simultaneously, an OT-constrained routing loss directs safety tokens toward these experts. This approach pushes Strata safety scores to 95+% on DeepSeek-V2-Lite / Qwen3-30B-A3B while maintaining GSM8K reasoning performance near original levels.

## Background & Motivation

**Background**: MoE has become the mainstream architecture for scaling LLM capacity (DeepSeek-V2, Qwen3-30B-A3B, Gemini series). Relying on a router to select Top-k experts per token leads to functional specialization (language, knowledge, tasks) among experts.

**Limitations of Prior Work**: This functional specialization introduces a unique vulnerability—**Safety Sparsity**: safety capabilities are highly concentrated in a few "safety experts." If an attacker constructs adversarial prompts to redirect the router to other experts (e.g., F-SOUR, PAIR, PAP), the safety guardrails are bypassed. Furthermore, directly applying dense-model alignment methods (SFT/GRPO/DPO) to MoE creates a dilemma: (1) Full-parameter fine-tuning erodes expert specialization, causing reasoning tasks like GSM8K to drop from 56% to 15% (as seen in Stair-DPO); (2) Forced modification of routing distributions disrupts load balancing and creates new unaligned pathways.

**Key Challenge**: There is a structural trade-off in MoE between **Safety** (requiring broad coverage of safety capabilities) and **General Capability** (requiring stability in expert specialization)—the former demands dispersion, while the latter demands minimal intervention.

**Goal**: (1) Select a subset of experts best suited to carry safety responsibility without undermining existing specialization; (2) Train a router that stably directs safety traffic to these newly aligned experts without perturbing the original routing patterns of general traffic.

**Key Insight**: Empirical and theoretical analyses reveal an asymmetric distribution of experts across two dimensions: "safety affinity" (routing inertia) and "general sensitivity" (Hessian fragility). Pure safety head experts are already saturated; pure general tail experts are too fragile to be modified due to high curvature. The "shoulder region" experts are truly suitable for safety adaptation, as they reside neither in the saturated head nor the fragile tail.

**Core Idea**: Use KL-regularized Optimal Transport to **redistribute** safety responsibility from original safety experts to these shoulder experts. An online OT constraint is applied during training to synchronize "which experts to select" with "how the router uses them."

## Method

### Overall Architecture
MESA takes a pre-trained MoE model (DeepSeek-V2-Lite or Qwen3-30B-A3B), a safety dataset $\mathcal{D}_{safe}$ (SafeRLHF, 15k), a general dataset $\mathcal{D}_{gen}$ (UltraFeedback, 15k), and a small subset $\mathcal{D}_{stat}$ (1,000 samples) for expert activation statistics.

The pipeline consists of two stages:

1.  **Offline Expert Selection (Sec. 3.1)**: Forward passes are run on $\mathcal{D}_{stat}$ to compute activation frequencies for safety, general, and mixed data (rankings $R_{safe}, R_{gen}, R_{mix}$). A Beta-Rational Cost Function defines each expert's "adaptation cost" as a function of its $R_{mix}$ rank. A global transmission plan $\pi^*$ is solved via KL-regularized Sinkhorn OT, and the Top-$w$ experts $\mathcal{E}_{select}$ are chosen based on $\pi^*$ values.
2.  **Online Joint Training (Sec. 3.2)**: Only parameters $\theta_{\mathcal{E}_{select}}$ within the selected experts and router parameters $\phi$ are updated. The objective includes a standard safety SFT loss $\mathcal{L}_{SFT}$ and an OT-constrained routing loss $\mathcal{L}_{OT}$ (specialized for safety and general flows). The total objective is $\mathcal{L}_{total} = \mathcal{L}_{SFT} + \gamma \mathcal{L}_{OT}$.

The output is an MoE model with broader safety coverage and a more robust router.

### Key Designs

1.  **Beta-Rational Cost Function + Two Principles**:
    *   **Function**: A scalar cost $C(f)$ informs the OT solver about the cost of assigning safety responsibility to an expert at rank $f$.
    *   **Mechanism**: The cost shape is derived from two principles. **Principle 1 (Safety Affinity)**: Saturated head experts offer marginal gains. Hibernating tail experts require significant router shifts; the parameter perturbation lower bound is $\|\Delta\phi\|_2 \geq \Omega(p_i^{-1/2})$ (Theorem 3.1). **Principle 2 (General Stability)**: General head experts have robust structures (flat loss landscapes), while tail experts reside in "sharp minima" where the Hessian spectral norm $\Lambda_i \sim \bar{p}_i^{-\gamma}$ ($\gamma>1$). Small updates drastically degrade general capabilities (Theorem 3.2: $\mathbb{E}_x[\Delta\mathcal{L}_g] \leq \frac{1}{2}\|\Delta\theta_i\|_2^2 \cdot \bar{p}_i \Lambda_i$). Consequently, optimal experts are in the shoulder region of $R_{mix}$, forming an "asymmetric U-shape." This is modeled via a maximum entropy Beta distribution with $\alpha=2, \beta=3$, yielding the cost $C(f) = 1/[(f+\alpha_{shift})(100-f)^2]$.
    *   **Design Motivation**: Selecting experts simply by $R_{safe}$ or $R_{gen}$ falls into the head-saturation or tail-fragility traps. This cost function compresses these principles into a closed-form expression, avoiding heuristic thresholds and softening head constraints via $\alpha_{shift}$.

2.  **KL-regularized Sinkhorn OT for Expert Selection**:
    *   **Function**: Outputs a globally optimal transmission plan $\pi^*$ given a cost matrix $\mathbf{C}$ and empirical activation distribution $\mathbf{P}_{emp}$.
    *   **Mechanism**: Greedy sorting by $\mathbf{C}$ ignores the routing topology and may cause mode collapse. The objective is $\pi^* = \arg\min_{\pi \in \mathcal{U}(\mathbf{r},\mathbf{c})} (\langle \pi, \mathbf{C} \rangle + \epsilon D_{KL}(\pi | \mathbf{P}_{emp}))$. The KL term ensures $\pi$ stays near the original router manifold. The strictly convex problem is solved via the Gibbs kernel $\mathbf{K} = \mathbf{P}_{emp} \odot \exp(-\mathbf{C}/\epsilon)$ using Sinkhorn-Knopp iterations.
    *   **Design Motivation**: Formulating selection as OT under manifold constraints identifies the lowest-cost experts without breaking the routing topology. Ablations show $E_{OT}$ significantly outperforms $E_{C_{max}}$ and $E_{C_{mid}}$.

3.  **Dynamic Routing Refinement (Online OT Constraints)**:
    *   **Function**: Teaches the router to use $\mathcal{E}_{select}$ for safety tokens while preserving original routing for general tokens.
    *   **Mechanism**: The OT framework is applied online: $\pi^*(x) = \arg\min_{\pi} (\langle \pi, \mathbf{C}(x) \rangle + \epsilon D_{KL}(\pi | \mathbf{P}_{ref}(x)))$, where $\mathbf{P}_{ref}(x)$ is the frozen base model distribution. **Safety Flow**: $\mathbf{C}(x)$ is the global cost matrix; $\pi^*_{safe}$ shifts mass to $\mathcal{E}_{select}$, and $\mathcal{L}_{OT} = \mathbb{E}_{x \sim \mathcal{D}_{safe}}[D_{KL}(\pi^*_{safe}(x) | \mathbf{P}_\phi(x))]$. **General Flow**: $\mathbf{C}(x) = 0$; the solution is $\mathbf{P}_{ref}(x)$, resulting in a conservative loss $\mathbb{E}_{x \sim \mathcal{D}_{gen}}[D_{KL}(\mathbf{P}_{ref}(x) | \mathbf{P}_\phi(x))]$.
    *   **Design Motivation**: Updating experts without the router limits WildJB scores to 76% instead of 90%. This design ensures synchronized adaptation for safety tokens and preservation for general tokens.

### Loss & Training
The total objective is $\mathcal{L}_{total} = \mathcal{L}_{SFT}(\mathcal{D}_{safe}; \theta_{\mathcal{E}_{select}}, \phi) + \gamma \cdot \mathcal{L}_{OT}(\phi)$. Experts outside $\mathcal{E}_{select}$ are frozen. Only selected experts and the router are updated.

## Key Experimental Results

### Main Results

Results for DeepSeek-V2-Lite (Safety vs. General trade-off):

| Method | Strata (Safety↑) | WildJB (Safety↑) | GSM8K (General↑) | HumanEval (General↑) |
|------|---------------|---------------|--------------|-------------------|
| Base | 70.50 | 43.40 | 55.95 | 42.07 |
| SFT | 92.00 | 77.70 | **16.15** (Collapse) | 31.10 |
| Stair-DPO (Prev. SOTA) | 93.00 | 83.60 | **15.54** (Collapse) | 26.22 |
| SafeX (MoE-specific) | 81.00 | 64.00 | 63.46 | 35.98 |
| **MESA (Ours)** | **95.00** | **90.90** | **66.11** | **42.07** |

On Qwen3-30B-A3B, Ours pushes Strata to 99.00 and WildJB to 97.65 while maintaining Math500=91.00 and HumanEval=94.51 near base levels.

### Ablation Study (DeepSeek-V2-Lite)

| Config | WildJB | Strata | GSM8K | Description |
|------|--------|--------|-------|------|
| Base | 43.40 | 70.50 | 55.95 | Starting point |
| Router only | 60.20 | 86.00 | 52.90 | Limited gain without new knowledge |
| $E_{ALL}$ (Full MoE SFT) | 83.00 | 93.00 | **8.33** | General capability collapse |
| $E_{OT}$ (OT Selection only) | 76.15 | 88.50 | 51.48 | Low safety without router alignment |
| **Full MESA** | **90.90** | **95.00** | **66.11** | Synergistic OT Selection + Routing Refinement |

### Key Findings
- **OT Expert Selection** is the primary contributor: Switching from $E_{ALL}$ to $E_{OT}$ restores GSM8K from 8.33 to 51.48, validating that the shoulder region avoids Hessian fragility.
- **Routing Refinement** is essential for high safety (88.50 to 95.00): Knowledge in experts requires the router to utilize them effectively.
- **Architectural Sensitivity**: DeepSeek-V2-Lite is extremely sensitive to full fine-tuning; MESA's advantage is most pronounced here (Ours 66.11 vs. Stair-DPO 15.54 on GSM8K).
- **Routing Attack Resistance**: Against F-SOUR, MESA achieves 0.00% ASR on JailbreakBench, whereas GRPO reaches 22.73%, proving topological expansion is superior to localized patching.

## Highlights & Insights
- **Repurposing Alignment as OT**: MESA provides a principled solution considering global routing topology as a hard constraint, unifying expert selection and router training under one OT formula.
- **The Shoulder Hypothesis**: An empirical insight that for scaling any new capability without interference (style, domain, safety), the middle-tier experts are more effective than the head or tail.
- **Theoretic Grounding**: Theorems 3.1 and 3.2 utilize statistical manifolds and Hessian spectra to provide a unified explanation for the "small update, large disaster" phenomenon in MoE fine-tuning.

## Limitations & Future Work
- **Limitations**: Evaluations were limited to DeepSeek-V2-Lite and Qwen3-30B-A3B; performance on larger models (DeepSeek-V3) and the quadratic scaling of OT with number of experts $N$ remain to be tested.
- **Future Directions**: (1) Making Beta distribution parameters $\alpha, \beta$ learnable; (2) Designing token-level cost matrices $\mathbf{C}(x_t)$ for mixed queries; (3) Generalizing the OT framework to multi-task continual learning.

## Related Work & Insights
- **vs SafeX**: SafeX uses localized additive merging, which lacks topological expansion and is easily bypassed by adversarial routing. MESA's decentralized responsibility is more robust.
- **vs Stair-DPO**: Stair-DPO is a SOTA content-level alignment tool that treats MoE as a dense model, leading to catastrophic forgetting. MESA proves architecture-aware alignment is essential for MoE.
- **Insight**: Optimal Transport serves as a general tool for parameter subset selection under structural constraints, applicable beyond safety to PEFT and model editing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reformulating MoE safety alignment as an OT problem with the "shoulder region" theoretical support is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Strong coverage across architectures and benchmarks; lacks sensitivity analysis for $|\mathcal{D}_{stat}|$.
- **Writing Quality**: ⭐⭐⭐⭐ Logical derivation of the Beta cost function; however, some implementation details (e.g., $w$) are relegated to the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Essential for MoE deployment; provides a transferable framework for capability adaptation without degradation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICML 2026\] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](../../ICLR2026/llm_alignment/superficial_safety_alignment_hypothesis.md)

</div>

<!-- RELATED:END -->
