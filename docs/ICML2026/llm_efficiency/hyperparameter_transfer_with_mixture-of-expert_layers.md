---
title: >-
  [Paper Note] Hyperparameter Transfer with Mixture-of-Experts Layers
description: >-
  [ICML 2026][LLM Efficiency][μP] This paper extends the maximal update parametrization (mUP/CompleteP) to sparse MoE Transformers. It introduces initialization and learning rate (LR) scaling rules for the router…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "μP"
  - "CompleteP"
  - "MoE scaling"
  - "DMFT"
  - "Zero-shot HP Transfer"
date: 2026-05-08
content_hash: 4f8504dd568866dd
---

# Hyperparameter Transfer with Mixture-of-Experts Layers

**Conference**: ICML 2026  
**arXiv**: [2601.20205](https://arxiv.org/abs/2601.20205)  
**Code**: None  
**Area**: LLM Efficiency / MoE / Hyperparameter Transfer  
**Keywords**: μP, CompleteP, MoE scaling, DMFT, Zero-shot HP Transfer

## TL;DR
This paper extends the maximal update parametrization (mUP/CompleteP) to sparse MoE Transformers. It introduces initialization and learning rate (LR) scaling rules for the router, expert up/down projections, and expert biases when width, depth, number of experts, and expert width are simultaneously scaled. Using a triple-layer mean-field Dynamical Mean-Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (under fixed active sparsity $\kappa$). Optimal LR and initialization can be directly reused from a 38M active parameter base model up to a 2B total parameter MoE. Zero-shot HP-tuned MoEs match or outperform dense GPT2 speedrun results at equivalent active parameter counts.

## Background & Motivation
**Background**: Parametrizations like μP and subsequent CompleteP/depth-μP allow for the direct transfer of critical HPs (LR, initialization) from small to large dense Transformers as width and depth scale. MoE is currently the mainstream method for parameter expansion, but most HP transfer research remains focused on dense models.

**Limitations of Prior Work**: Directly applying dense μP to MoE is problematic for two reasons. First, MoE introduces new parameter groups like router weights and expert biases, and it is not obvious how their optimal LR/initialization should scale with $n_{\text{embd}}$. Second, MoE introduces two new scaling axes: the number of experts $n_{\text{exp}}$ and the expert width $\alpha_{\text{ffn}}n_{\text{embd}}$. There has been no systematic verification of whether HPs need retuning along these axes. Heuristic μP approaches for dense models (judging $\Theta(1)$ updates by counting dimensions) fail to address stability across $n_{\text{exp}}$ or explain why changes in $\alpha_{\text{ffn}}$ should not affect optimal HPs.

**Key Challenge**: There is a coupling between the MoE router, sparse top-$k$ routing, and the internal expert MLPs. Heuristically applying μP per parameter group cannot guarantee that these couplings converge to well-defined training dynamics in the infinite limit. In other words: to achieve HP transfer, one must first prove the existence of a mean-field limit that is independent of specific scale variables.

**Goal**: Address three sub-problems: (1) determine the $n_{\text{embd}}, \alpha_{\text{ffn}}$ exponents for MoE parameter initialization and LR; (2) determine whether to fix $n_{\text{act}}$ or $\kappa = n_{\text{act}}/n_{\text{exp}}$ as $n_{\text{exp}} \to \infty$; and (3) verify if these rules correspond to a convergent training limit.

**Key Insight**: The authors maintain the CompleteP rules for width and depth while focusing on the MoE module. They propose a scaling perspective of fixing sparsity $\kappa$ while expanding $n_{\text{exp}}$. This ensures that the proportion of tokens seen by each expert ($\kappa B$) remains constant, which corresponds to a constant probability event in the mean-field measure, making it natural for both theory and hardware deployment.

**Core Idea**: An "enhanced μP" is derived by requiring each component (router, expert, bias) to *individually* satisfy the maximal update condition $\Delta W \, \partial z / \partial W = \Theta(1)$ (Table 1). A triple-layer mean-field DMFT is used to rigorously prove that training dynamics have a well-defined limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L$ diverge simultaneously.

## Method

### Overall Architecture
The model is a pre-LayerNorm decoder-only Transformer with FFNs replaced by MoE modules: each layer is $f_{\text{MoE}}(h) = \frac{1}{n_{\text{act}}} \sum_{i \in A(h)} g_i(h) E_i(h)$, where $g_i(h) = \sigma(W_{\text{router}}^{(i)\top}h)$ is the sigmoid routing weight, $A(h)$ is the set of top-$n_{\text{act}}$ hard-routed experts with trainable biases $b_i$, and $E_i(h) = W_{\text{down}}^{(i)} \phi(W_{\text{up}}^{(i)\top}h)$ represents single-hidden-layer MLP experts. Residual blocks include a $1/L$ multiplier (CompleteP style) for depth transfer. Load balancing is auxiliary-loss-free: only the bias is updated as $b_i \leftarrow b_i - \eta_{\text{bias}}(\text{Load}_i - \kappa)$, leaving other parameters untouched.

The scaling axes are defined as: $L$ (depth), $n_{\text{embd}}$ (residual stream width), $\alpha_{\text{ffn}}$ (expert hidden width multiplier), and $n_{\text{exp}}$ (number of experts), while keeping $\kappa = n_{\text{act}}/n_{\text{exp}}$ constant. Given optimal HPs for a base model, the parametrization automatically extrapolates the LR and initialization for the router, experts, and bias to larger models using rules dependent only on the scale axes.

### Key Designs

1.  **MoE Parametrization (Table 1: Scaling Rules for Router/Expert/Bias Init and LR)**:
    *   **Function**: Refines the "entry-wise update $\Theta(1)$" principle of μP for each MoE parameter group, specifying how the initialization std and Adam LR for router matrices, expert up/down projections, and expert biases scale with $n_{\text{embd}}$ and $\alpha_{\text{ffn}}$.
    *   **Mechanism**: The authors require that the mixing coefficient $g_i$, expert output $E_i$, and hidden activation $h_{\text{up}}$ *each* satisfy $\eta_W \overline{\nabla W} \partial z / \partial W = \Theta(1)$. This is a stricter per-component condition than dense μP. Approximating Adam via SignGD ($\Delta w \approx \eta \, \text{sgn}(\partial \mathcal{L} / \partial w)$) and assuming LLN alignment $\cos(v, w) \in \Theta(1)$ for $h$ and $\Delta W$, the groups follow: router $\eta \in \Theta(1/n_{\text{embd}})$, init $\Theta(n_{\text{embd}}^{-\gamma})$ (used $\gamma=1$); expert up $\sigma_{\text{init}} = n_{\text{embd}}^{-1/2}, \eta = n_{\text{embd}}^{-1}$. Since expert down handles the secondary scaling from $h_{\text{up}}$ to $E$, it includes an extra $\alpha_{\text{ffn}}^{-1}$, specifically $\sigma_{\text{init}} = \alpha_{\text{ffn}}^{-1} n_{\text{embd}}^{-1/2}, \eta = \alpha_{\text{ffn}}^{-1} n_{\text{embd}}^{-1}$. Expert bias uses $\Theta(1)$ LR and zero initialization to satisfy both the $\Theta(n_{\text{act}})$ per-step change in the activation set and step-0 load balancing.
    *   **Design Motivation**: Standard fan-in initialization causes a mismatch in the dependence of $W_{\text{down}}$ on $\alpha_{\text{ffn}}$, leading the optimal LR to drift as $\alpha_{\text{ffn}}$ changes. Treating $W_{\text{down}}$ as the "intermediate width" layer of a mean-field two-layer MLP for initialization allows $\alpha_{\text{ffn}}$ to effectively disappear from the optimal HP tuning (explaining the zero-transfer phenomenon in Figure 2, column 4). Suppressing the router LR to $1/n_{\text{embd}}$ rather than $\Theta(1)$ accounts for the $h^\top \Delta W_{\text{router}}^{(i)}$ term naturally carrying a $\sqrt{n_{\text{embd}}} \cdot \sqrt{n_{\text{embd}}}$ factor under LLN alignment.

2.  **Triple-Layer Mean-Field DMFT Limit (Theoretical Foundation)**:
    *   **Function**: Generalizes the DMFT framework of Bordelon & Pehlevan to deep residual networks with sparse MoE, providing closed training dynamic equations for $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (while fixing $\kappa$ and keeping $n_{\text{embd}} / (n_{\text{exp}} n_{\text{hid}} L)$ bounded).
    *   **Mechanism**: The analysis focuses on residual networks composed of MoE modules $h^{(\ell+1)} = h^{(\ell)} + L^{-1} f_{\text{MoE}}^{\ell}(h^{(\ell)})$. Dynamics unfold in three nested mean-field layers: outer (between residual stream neurons), middle (between experts in a layer), and inner (between neurons within an expert). Hard routing appears via a quantile threshold $q_\star(\kappa)$ (satisfying $\mathbb{E}[\mathbf{1}_{q \ge q_\star}] = \kappa$), necessitating fixed $\kappa$. The equations show that limit dynamics are *independent* of $\alpha_{\text{ffn}}$ (consistent with dense results, explaining $\alpha_{\text{ffn}}$ transfer). When $n_{\text{embd}}, n_{\text{exp}}$ diverge, as long as $\alpha_\star = \lim n_{\text{embd}} / (n_{\text{hid}} n_{\text{exp}} L) = 0$, all joint scalings yield the same limit. The depth limit degrades to a neural ODE if $\alpha_\star = 0$ and a neural SDE if $\alpha_\star > 0$.
    *   **Design Motivation**: While μP heuristics suggest $\Theta(1)$ behavior, they cannot confirm if $n_{\text{exp}}$ can be infinite or if $n_{\text{hid}}$ can remain non-divergent. DMFT *proves* these parametrizations correspond to a deterministic set of evolution equations, elevating HP transfer from empirical observation to theoretical guarantee.

3.  **Scaling Strategy: Fixing Sparsity $\kappa$ over $n_{\text{act}}$**:
    *   **Function**: Scales $n_{\text{act}}$ proportionally with $n_{\text{exp}}$ to keep the token proportion $\kappa$ seen by each expert constant, contrasting with the Switch Transformer view (fixed $n_{\text{act}}=1, \kappa \to 0$).
    *   **Mechanism**: Under perfect balance, each expert sees $\kappa B$ tokens while the self-attention/router see $B$ tokens. If $\kappa$ varied with scale, the data efficiency of experts would mismatch other modules, preventing HP transfer. Fixed $\kappa$ also corresponds to a constant probability slice in mean-field measure, which is required for the existence and stability of $q_\star(\kappa)$ in DMFT.
    *   **Design Motivation**: This is both an engineering consideration (communication bandwidth limits $n_{\text{act}}$) and a theoretical necessity (preventing mean-field measure degradation).

### Loss & Training
The framework uses Adam/AdamW. The router utilizes sigmoid + auxiliary-loss-free bias updates $b_i \leftarrow b_i - \eta_{\text{bias}}(\text{Load}_i - \kappa)$. For fixed token budget experiments, a linear warmup (1000 steps) + constant LR (1000 steps) schedule is used (2000 steps / 1B tokens / batch 500K / seq 1024). For longer horizons, a cosine decay to zero is applied. In addition to exponential scaling of $n_{\text{embd}}, \alpha_{\text{ffn}}$, each parameter group requires a *separate* $\Theta(1)$ constant multiplier tuning (Appendix D.1: without this, training dynamics like load-balancing loss become unstable near optimal HP). Sparsity configurations: $\kappa=1/4$ (FineWeb) and $\kappa=1/12$ (C4).

## Key Experimental Results

### Main Results

| Experimental Setup | Key Observation | Explanation |
| :--- | :--- | :--- |
| FineWeb, $\kappa=1/4$, 1B tokens, 38M→1.8B swept along width/depth/$n_{\text{exp}}$/$\alpha_{\text{ffn}}$ | Optimal LR and init std curves for 4 scaling axes are almost co-located across sizes | Verifies zero-shot HP transfer (Figure 2). |
| C4, $\kappa=1/12$, 1B tokens, 4 scaling axes | Consistent with FineWeb findings | Covers sparser routing and different corpora (Figure 4). |
| GPT2-small (124M) active config, FineWeb, 10B tokens, zero-shot HP (migrated from 38M base) | Val loss comparable/better than dense GPT2 speedrun (AdamW/Muon) with more total params | Figure 1 |
| GPT2-medium (355M) active, 7.5B tokens | Stable training with zero-shot HP | Figure 16 |
| Early loss curve collapse | When using zero-shot optimal HP, loss curves for different scales overlap exactly for initial steps | Matches DMFT scale-invariance predictions (Figure 3). |

### Ablation Study

| Configuration | Key Metric | Explanation |
| :--- | :--- | :--- |
| Full Parametrization + Constant Multiplier Tuning | Optimal loss, stable load-balancing | Main pipeline |
| No constant multiplier tuning | Unstable load-balancing loss near optimal HP | Appendix D.1; MoE is more sensitive to constants than dense models. |
| $W_{\text{down}}$ with standard fan-in init (no $\alpha_{\text{ffn}}^{-1}$) | Optimal LR drifts with $\alpha_{\text{ffn}}$ | Highlights the necessity of the $\alpha_{\text{ffn}}^{-1}$ factor (Figure 11). |
| Mixing coefficient (sigmoid vs softmax) | Similar performance across scales | Figure 18 |
| Scaling $n_{\text{act}}$ vs expert size (inverse) | More, smaller experts are monotonically better at both 1B and 5B horizons | Replicates Krajewski et al. but *without* HP retuning (Figure 5/D.3). |

### Key Findings
- Under constant $\kappa$, LR and initialization transfer directly across width, depth, number of experts, and expert width. The *transfer across $\alpha_{\text{ffn}}$* is a unique prediction of DMFT that cannot be derived from μP heuristics, and it is validated experimentally.
- MoE stability is significantly more sensitive to $\Theta(1)$ constant multipliers than dense models. Constants from dense models cannot be blindly reused; they must be tuned per parameter group.
- Using this parametrization naturally results in uniform expert load (Figure 17) without explicit auxiliary losses, due to the synergy of sigmoid routing, bias updates, and LR scaling.
- For 5B–10B token horizons, while stable LRs converge, adding cosine decay to zero significantly further reduces validation loss.
- The phenomenon "more, smaller experts are better" is confirmed without the artifact of HP tuning, proving it is a fundamental property.

## Highlights & Insights
- Defining components of each expert to satisfy $\Delta z = \Theta(1)$ is the critical step in incorporating sparse routing into the μP framework. It decouples the LR derivation for the router, expert, and bias into mechanical rules.
- The triple-layer mean-field nesting (residual neurons / experts / intra-expert neurons) generalizes the multi-head self-attention analysis of "measure-within-measure" to MoE, providing a paradigm for other sparse architectures.
- The $\alpha_{\text{ffn}}^{-1}$ factor in $W_{\text{down}}$ initialization is counter-intuitive (relative to fan-in) but essential for zero-shot $\alpha_{\text{ffn}}$ transfer. This "mean-field intermediate width" perspective is a trick applicable to other modules with hidden multipliers.
- By fixing $\kappa$, the authors provide a formal justification for MoE scaling laws, establishing that $\kappa \to 0$ and $\kappa = \text{const}$ are fundamentally different limits.

## Limitations & Future Work
- Self-attention modules were not fully integrated into the DMFT proof. No technical barriers are expected, but the complete version is missing.
- Zero-shot HP only transfers base LR and init std; constant multipliers for weight decay, router temperature, and bias LR still need tuning per scale.
- The scale is limited to 2B total parameters and 10B tokens; whether corrections are needed for trillion-parameter models or trillion-token horizons remains an open question.
- Only $\kappa = 1/4$ and $1/12$ were verified; the relationship between HPs across different $\kappa$ values remains empirical.
- Primarily focused on sigmoid + bias routing; it does not explicitly cover softmax-top-$k$ or expert-choice routing variants.

## Related Work & Insights
- **vs. μP (Yang & Hu 2022) / CompleteP (Dey et al. 2025)**: These address width/depth transfer for dense models. This work maintains those rules but adds specific rules for MoE modules across $n_{\text{exp}}$ and $\alpha_{\text{ffn}}$.
- **vs. Bordelon & Pehlevan DMFT**: Extends their work on dense components to a three-layer nested mean-field for sparse experts.
- **vs. Malasnicki et al. 2025**: A concurrent work that only studies LR transfer across MoE width. This paper covers both LR and initialization across four axes with rigorous DMFT proof.
- **vs. Krajewski et al. 2024 / Boix-Adsera & Rigollet 2025**: Confirms their "more smaller experts" findings without the confounder of HP tuning bias.

## Rating
- Novelty: ⭐⭐⭐⭐ (Extending μP+DMFT to MoE is natural, but the details of $\alpha_{\text{ffn}}$/$n_{\text{exp}}$ axes and the triple-layer proof are significant new contributions).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers 38M to 2B scales, multiple corporate data, and long horizons, though lacks industrial-scale verification).
- Writing Quality: ⭐⭐⭐⭐ (Clearly distinguishes heuristic limitations from DMFT necessity; well-structured appendix).
- Value: ⭐⭐⭐⭐ (Provides a direct lookup table in Table 1 for scaling MoE and formalizes the theoretical basis for constant sparsity scaling).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)

</div>

<!-- RELATED:END -->
