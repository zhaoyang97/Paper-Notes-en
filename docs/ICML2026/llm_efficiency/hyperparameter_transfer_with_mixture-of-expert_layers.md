---
title: >-
  [Paper Note] Hyperparameter Transfer with Mixture-of-Experts Layers
description: >-
  [ICML 2026][LLM Efficiency][μP] This paper extends the maximal update parametrization (μP/CompleteP) to sparse MoE Transformers. It defines initialization and learning rate (LR) scaling rules for routers, expert up/down projections, and expert biases when model width, depth, number of experts, and expert width are simultaneously scaled. Using a three-level Mean-Field Dynamical Mean Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant li…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "μP"
  - "CompleteP"
  - "MoE scaling"
  - "DMFT"
  - "Zero-shot hyperparameter transfer"
date: 2026-05-08
content_hash: b07c9fa539b571bd
---

# Hyperparameter Transfer with Mixture-of-Experts Layers

**Conference**: ICML 2026  
**arXiv**: [2601.20205](https://arxiv.org/abs/2601.20205)  
**Code**: None  
**Area**: LLM Efficiency / MoE / Hyperparameter Transfer  
**Keywords**: μP, CompleteP, MoE scaling, DMFT, Zero-shot hyperparameter transfer

## TL;DR
This paper extends the maximal update parametrization (μP/CompleteP) to sparse MoE Transformers. It defines initialization and learning rate (LR) scaling rules for routers, expert up/down projections, and expert biases when model width, depth, number of experts, and expert width are simultaneously scaled. Using a three-level Mean-Field Dynamical Mean Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (at fixed activation sparsity $\kappa$). Optimal LRs and initializations can be directly reused from 38M active parameter base models to 2B parameter MoEs. MoEs trained with zero-shot hyperparameters achieve performance comparable to or better than dense GPT2 speedrun models at equivalent active parameter counts.

## Background & Motivation
**Background**: μP and subsequent parametrizations such as CompleteP and depth-μP allow for the direct transfer of critical hyperparameters (HP) like LR and initialization from small to large dense Transformers as width and depth scale. MoE is currently the dominant method for expanding parameter counts, but HP transfer research has largely remained focused on dense models.

**Limitations of Prior Work**: Directly applying dense μP to MoE is problematic for two reasons. First, MoE introduces new parameter groups such as router weights and expert biases, and it is non-obvious how their optimal LR/init should scale with $n_{\text{embd}}$. Second, MoE introduces two new scaling axes: the number of experts $n_{\text{exp}}$ and expert width $\alpha_{\text{ffn}}n_{\text{embd}}$. Whether hyperparameters require systematic retuning along these axes has not been verified. Heuristic μP approaches (counting dimensions to ensure $\Theta(1)$ updates) cannot guarantee stability across $n_{\text{exp}}$, nor can they explain why changes in $\alpha_{\text{ffn}}$ should not impact optimal HPs.

**Key Challenge**: There is complex coupling between the router, sparse top-$k$ routing, and internal expert MLPs. Using μP heuristics on a per-parameter-group basis cannot ensure that these couplings converge to well-defined training dynamics at the limit. In other words, to achieve HP transfer, one must first prove the existence of a mean-field limit that is independent of specific scale variables.

**Goal**: The authors address three sub-problems: (1) determining the $n_{\text{embd}}$ and $\alpha_{\text{ffn}}$ scaling indices for MoE initialization and LR; (2) deciding whether to fix $n_{\text{act}}$ or $\kappa = n_{\text{act}}/n_{\text{exp}}$ as $n_{\text{exp}} \to \infty$; and (3) verifying whether these rules correspond to a convergent training limit.

**Key Insight**: The authors adopt CompleteP rules for width and depth and focus on the MoE module. They propose scaling $n_{\text{exp}}$ while fixing sparsity $\kappa$, ensuring that the proportion of tokens seen by each expert ($\kappa B$) remains constant. This corresponds to a constant probability event in the mean-field measure, which is natural for both theory and hardware deployment.

**Core Idea**: A "strengthened μP" condition is proposed where each component (router, expert, and bias) individually satisfies the maximal update condition $\Delta W \, \partial z/\partial W = \Theta(1)$, resulting in the parametrization in Table 1. Training dynamics are rigorously proven to have a well-defined limit via three-level mean-field DMFT as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L$ diverge simultaneously.

## Method

### Overall Architecture
The model is a pre-LayerNorm decoder-only Transformer where all FFNs are replaced by MoE modules: $f_{\text{MoE}}(h) = \frac{1}{n_{\text{act}}} \sum_{i \in A(h)} g_i(h) E_i(h)$, where $g_i(h) = \sigma(W_{\text{router}}^{(i)\top}h)$ represents sigmoid routing weights. The set $A(h)$ consists of top-$n_{\text{act}}$ experts determined via routing with trainable biases $b_i$. Experts are single-hidden-layer MLPs: $E_i(h) = W_{\text{down}}^{(i)} \phi(W_{\text{up}}^{(i)\top}h)$. Residual blocks include a $1/L$ multiplier (CompleteP style) to ensure depth transfer. Load balancing is handled without auxiliary losses by updating biases as $b_i \leftarrow b_i - \eta_{\text{bias}}(\text{Load}_i - \kappa)$. The scaling axes are $L$ (depth), $n_{\text{embd}}$ (residual width), $\alpha_{\text{ffn}}$ (expert hidden width multiplier), and $n_{\text{exp}}$ (number of experts), while $\kappa = n_{\text{act}}/n_{\text{exp}}$ remains constant.

### Key Designs

**1. MoE Parametrization (Table 1): Scaling Rules for Router, Expert Projections, and Bias**

Standard dense μP heuristics do not specify how new MoE parameter groups should scale. The authors refine the "entry-wise update $\Theta(1)$" principle into a stronger component-wise condition: the mixing coefficient $g_i$, expert output $E_i$, and hidden activation $h_{\text{up}}$ must each satisfy $\eta_W \overline{\nabla W} \, \partial z/\partial W = \Theta(1)$. Using SignGD as an approximation for Adam ($\Delta w \approx \eta \, \text{sgn}(\partial\mathcal{L}/\partial w)$) and Law of Large Numbers (LLN) alignment assumptions, the rules are derived as follows: router uses $\eta \in \Theta(1/n_{\text{embd}})$ and init $\Theta(n_{\text{embd}}^{-\gamma})$; expert up uses $\sigma_{\text{init}} = n_{\text{embd}}^{-1/2}$ and $\eta = n_{\text{embd}}^{-1}$; expert down includes an additional $\alpha_{\text{ffn}}^{-1}$ factor to account for the scaling from $h_{\text{up}}$ to $E$, resulting in $\sigma_{\text{init}} = \alpha_{\text{ffn}}^{-1}n_{\text{embd}}^{-1/2}$ and $\eta = \alpha_{\text{ffn}}^{-1}n_{\text{embd}}^{-1}$. Expert bias uses $\Theta(1)$ LR and zero initialization.

The most counter-intuitive rule is the $\alpha_{\text{ffn}}^{-1}$ factor for $W_{\text{down}}$, which differs from standard fan-in initialization. Treating $W_{\text{down}}$ as an intermediate width layer in a mean-field MLP ensures that $\alpha_{\text{ffn}}$ does not shift the optimal HP. Similarly, the router LR is suppressed to $1/n_{\text{embd}}$ to cancel out the $\sqrt{n_{\text{embd}}} \cdot \sqrt{n_{\text{embd}}}$ factor arising from LLN alignment in $h^\top \Delta W_{\text{router}}^{(i)}$.

**2. Three-Level Mean-Field DMFT Limit: Theoretical Validation**

While μP heuristics suggest $\Theta(1)$ behavior, they do not guarantee stability as $n_{\text{exp}} \to \infty$. The authors extend the DMFT framework to residual networks containing MoE modules, proving that training dynamics follow closed equations under the limit $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (at fixed $\kappa$). The expansion follows three nested levels: mean-field across neurons in the residual stream, mean-field across experts within a layer, and mean-field across hidden neurons within an expert. Hard routing is incorporated via a quantile threshold $q_\star(\kappa)$. This proof elevates HP transfer from an empirical observation to a consequence of a deterministic evolution equation.

**3. Fixed Sparsity $\kappa$ vs. Fixed $n_{\text{act}}$ Expansion Strategy**

There are two ways to scale $n_{\text{exp}}$: keeping $n_{\text{act}}$ constant (e.g., Switch Transformer), where $\kappa \to 0$, or keeping $\kappa$ constant, where $n_{\text{act}}$ scales with $n_{\text{exp}}$. The authors choose the latter to maintain the effective "data efficiency" of experts relative to other modules. Maintaining a constant $\kappa$ is also a theoretical requirement for a stable DMFT limit, as it represents a constant probability slice in the mean-field measure.

### Loss & Training
The foundation is standard Adam and AdamW. The router uses sigmoid routing with auxiliary-loss-free bias updates. The LR scheduler consists of 1000 linear warmup steps followed by a constant LR for 1B tokens. For longer horizons, cosine decay is used. In addition to exponential scaling with $n_{\text{embd}}$ and $\alpha_{\text{ffn}}$, each parameter group requires an individually tuned $\Theta(1)$ constant multiplier to ensure stable load balancing and training dynamics. Experiments use $\kappa=1/4$ on FineWeb and $\kappa=1/12$ on C4.

## Key Experimental Results

### Main Results

| Experimental Setup | Key Observation | Description |
|:---|:---|:---|
| FineWeb, $\kappa=1/4$, 1B tokens, 38M→1.8B scaling width, depth, $n_{\text{exp}}$, and $\alpha_{\text{ffn}}$ | Optimal LR and init std curves coincide across different sizes | Validates zero-shot HP transfer (Figure 2). |
| C4, $\kappa=1/12$, 1B tokens, 4 scale axes | Consistent with FineWeb results | Covers sparser routing and different corpora (Figure 4). |
| GPT2-small (124M) active parameters, FineWeb, 10B tokens | Val loss matches or exceeds dense GPT2 speedrun | Zero-shot HP transferred from 38M base model (Figure 1). |
| Early loss curve collapse | Loss curves perfectly overlap for the first several steps as scale increases | Matches DMFT scale-invariance predictions (Figure 3). |

### Ablation Study

| Configuration | Key Metric | Description |
|:---|:---|:---|
| Full parametrization + tuned constant multipliers | Optimal loss and stable load balancing | Main method. |
| No tuning of constant multipliers | Instability in load-balancing loss near optimal HP | MoE is more sensitive to these constants than dense models (Appendix D.1). |
| Standard fan-in init for $W_{\text{down}}$ (no $\alpha_{\text{ffn}}^{-1}$) | Optimal LR drifts with $\alpha_{\text{ffn}}$ | Justifies the $\alpha_{\text{ffn}}^{-1}$ factor (Figure 11). |
| More small experts vs. fewer large experts | More small experts are monotonically better | Confirms previous findings without needing to retune HP (Figure 5). |

### Key Findings
- LR and init transfer successfully across all four axes (width, depth, $n_{\text{exp}}$, $\alpha_{\text{ffn}}$) provided $\kappa$ is fixed. Transfer across $\alpha_{\text{ffn}}$ is a unique prediction of DMFT that is confirmed by experiment.
- MoE stability is significantly more sensitive to $\Theta(1)$ constant multipliers than dense models, requiring these to be tuned once per parameter group.
- Uniform expert load emerges naturally with the proposed parametrization, eliminating the need for auxiliary losses.
- The advantage of using "more but smaller experts" is verified to be independent of hyperparameter tuning artifacts.

## Highlights & Insights
- Decoupling the derivation of LRs for the router, expert, and bias using component-wise conditions is the key step in bringing sparse routing into the μP framework.
- The three-level mean-field nesting provides a clear paradigm for analyzing other sparse architectures like MoA or shared-expert models.
- The use of the $\alpha_{\text{ffn}}^{-1}$ factor for $W_{\text{down}}$ init is a critical insight derived from mean-field theory that enables zero-shot transfer across expert widths.
- By fixing $\kappa$, the authors establish that $\kappa \to 0$ and $\kappa = \text{const}$ are fundamentally different scaling limits, providing a formal basis for MoE scaling law research.

## Limitations & Future Work
- The DMFT analysis does not fully integrate the self-attention module; while the authors believe there are no technical barriers, the coupling effects have not been fully verified.
- Only base LR and init std are transferred; other constant multipliers (weight decay, router temperature) still require manual tuning.
- Experimental scale is limited to 2B total parameters and 10B tokens, leaving open questions about long-horizon behavior at industrial scales.
- Only two levels of sparsity ($\kappa=1/4, 1/12$) were tested; the relationship between HPs across different $\kappa$ remains an empirical unknown.
- The study uses sigmoid routing and does not cover other variants such as softmax-top-$k$ or expert-choice routing.

## Related Work & Insights
- **vs μP (Yang & Hu 2022) / CompleteP (Dey et al. 2025)**: These works handle dense Transformers; this paper supplements them with rules for the MoE module and adds $n_{\text{exp}}$ and $\alpha_{\text{ffn}}$ axes.
- **vs Bordelon & Pehlevan DMFT**: This paper extends their mean-field approach for attention heads to the hierarchical structure of sparse experts.
- **vs Malasnicki et al. 2025**: A concurrent work focusing mainly on LR transfer across width; this paper covers both LR and init across four axes with a rigorous DMFT proof.
- **vs Krajewski et al. 2024**: This paper confirms their "more small experts are better" conclusion using zero-shot hyperparameters, proving it is not an artifact of HP tuning.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICML 2026\] SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](softmoe_soft_differentiable_routing_for_mixture-of-experts_in_llms.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)

</div>

<!-- RELATED:END -->
