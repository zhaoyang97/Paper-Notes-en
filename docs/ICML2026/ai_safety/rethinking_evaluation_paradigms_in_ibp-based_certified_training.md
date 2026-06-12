---
title: >-
  [Paper Note] Rethinking Evaluation Paradigms in IBP-based Certified Training
description: >-
  [ICML 2026][Interval Bound Propagation (IBP)] The authors point out that comparing IBP-based certified training methods using single "biased configurations" is fundamentally unfair. They propose using multi-objective Bay…
tags:
  - "ICML 2026"
  - "Interval Bound Propagation (IBP)"
  - "Certified Training"
  - "Pareto Front"
  - "Multi-objective Bayesian Optimization"
  - "Robust-Accuracy Trade-off"
date: 2026-05-08
content_hash: cc17747f43e2ed01
---

# Rethinking Evaluation Paradigms in IBP-based Certified Training

**Conference**: ICML 2026  
**arXiv**: [2606.02134](https://arxiv.org/abs/2606.02134)  
**Code**: https://github.com/ada-research/CTRAIN  
**Area**: AI Safety / Certified Robust Training / Multi-objective Hyperparameter Optimization  
**Keywords**: Interval Bound Propagation (IBP), Certified Training, Pareto Front, Multi-objective Bayesian Optimization, Robust-Accuracy Trade-off  

## TL;DR
The authors point out that comparing IBP-based certified training methods using single "biased configurations" is fundamentally unfair. They propose using multi-objective Bayesian hyperparameter optimization to plot the Pareto front for each method, proving that existing SOTA methods are generally under-tuned—CROWN-IBP's clean accuracy can be increased by approximately $6\%$, and MTL-IBP on Tiny ImageNet achieves a simultaneous gain of $\sim 2\%$ in both clean and certified accuracy.

## Background & Motivation

**Background**: Under the $\ell_\infty$ threat model, certified training uses incomplete verifiers (IBP / CROWN-IBP / SABR / MTL-IBP) to upper-bound the worst-case loss during training, enabling networks to obtain formal robustness certificates using complete verifiers (e.g., $\alpha\beta$-CROWN) post-hoc. These methods naturally involve a trade-off parameter ($\kappa$, $\tau$, or $\alpha$) to balance "clean accuracy vs. certified accuracy."

**Limitations of Prior Work**: From Gowal 2019 to De Palma 2024b, almost all papers report performance at a single biased point on the trade-off curve. Although the recent CTBench (Mao 2025) performed grid searches, it still treated the problem as single-objective, tending to suppress certified accuracy. Consequently, points reported in different papers are not on the same scale, and "who is SOTA" depends entirely on which side of the trade-off was prioritized.

**Key Challenge**: When objectives are inherently conflicting, picking one configuration for comparison is equivalent to "choosing a stance before choosing evidence." The true capability of IBP-based methods is reflected in the entire Pareto front, which the community has failed to map systematically. This failure masks both the complementarity between methods and significant room for tuning.

**Goal**: To upgrade the evaluation of certified training from single-point comparisons to Pareto front comparisons and provide a reusable, computationally affordable multi-objective hyperparameter search protocol.

**Key Insight**: The authors utilize multi-objective Bayesian optimization with Expected Hypervolume Improvement (EHVI) to directly search for the Pareto front. To keep the search within a budget similar to single-point tuning, they substitute "expensive complete verification rates" with "cheap incomplete verification rates" as proxy objectives and reduce the verification timeout from $1000\,\text{s}$ to $100\,\text{s}$, finally refining candidate points with complete verification.

**Core Idea**: Use constrained multi-objective Bayesian optimization to search for the Pareto set in the 2D space of "clean accuracy / certified accuracy." After deduplication via clustering, only representative points undergo expensive complete verification—allocating the same search budget across four methods to produce a method-agnostic, reproducible "true SOTA" map.

## Method

### Overall Architecture
The evaluation protocol consists of four components. **First** is a unified search space: each method (IBP / CROWN-IBP / SABR / MTL-IBP) is searched over a set of "general + method-specific" hyperparameters, including learning rate, $\ell_1$ regularization weight, Shi 2021 regularization weight, warm-up/ramp-up epochs, training $\epsilon$ scaling factor, plus method-specific parameters like $\kappa_{\text{start}} \ge \kappa_{\text{end}}$, $\beta$, $\tau$, $\alpha$, and PGD attack steps/step sizes. **Second** is a constrained multi-objective Bayesian optimizer: each objective (clean accuracy, incomplete certified accuracy) is modeled with an independent Gaussian Process. The acquisition function is EHVI, with the search region constrained to intervals of interest (e.g., for CIFAR-10 $\epsilon=2/255$, requiring clean $\ge 60\%$ and certified $\ge 40\%$). Three random seeds with 100 trials each are merged per method. **Third** is a cheap proxy objective: post-training, a cascaded incomplete verification (IBP $\to$ CROWN-IBP $\to$ CROWN) provides an underestimation of certified accuracy, leaving expensive complete verification for the end. **Fourth** is Pareto front refinement: single-linkage clustering ($d_{\min}=0.05$) merges adjacent points in the Pareto set. One representative configuration per cluster is randomly sampled for complete verification using $\alpha\beta$-CROWN (cutoff $1000\,\text{s}$), reconstructing the final front. The fronts of multiple methods are then merged into a "combined Pareto front" as a benchmark.

### Key Designs

1. **Multi-objective Bayesian Optimization + Constrained EHVI**:
    - **Function**: Simultaneously searches for a set of Pareto-optimal hyperparameters in the clean/certified dimensions rather than optimizing a weighted sum or single metric.
    - **Mechanism**: The objective vector is denoted as $\mathbf{f}(\boldsymbol{\theta}) = (\text{acc}_{\text{clean}}, \text{acc}_{\text{cert}})$, fitted with two independent GPs. Each step uses $\mathrm{EHVI}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{f}}\!\big[\max(0, \mathrm{HV}(P \cup \{\mathbf{f}\}) - \mathrm{HV}(P))\big]$ to capture non-dominated regions outside the discovered front $P$, with hard constraints to exclude degenerate regions of "approximate adversarial training." The union of Pareto fronts from three seeds eliminates local traps.
    - **Design Motivation**: Hyperparameters of IBP-based methods are highly interactive (e.g., $\kappa$ is coupled with warm-up length, $\tau$ with PGD step size). Any weighted scalarization would distort the true front. Multi-objective BO allows both objectives to "grow as they should," before being pruned by Pareto relationships to reveal the true reachable boundaries.

2. **Incomplete Verification as a Cheap Proxy for Certified Accuracy**:
    - **Function**: Reduces the evaluation cost of "certified rate" during the search phase from "minutes per sample" to "milliseconds per sample," making a 100-trial budget feasible.
    - **Mechanism**: For each network trained in a trial, incomplete verifiers are called in sequence: IBP $\to$ CROWN-IBP $\to$ CROWN, moving to a stronger method only if the previous one fails to prove robustness. The resulting certified rate is a verifiable lower bound of the true complete certified rate: $\widehat{\text{acc}}_{\text{cert}} \le \text{acc}_{\text{cert}}$. BO optimizes directly on $\widehat{\text{acc}}_{\text{cert}}$, with $\alpha\beta$-CROWN complete verification applied only once to representative points on the Pareto set.
    - **Design Motivation**: Complete verification is $\mathcal{NP}$-complete; running it for every trial is computationally impossible. However, monotonic proxies rarely change the Pareto order—lowering the cutoff from $1000\,\text{s}$ to $100\,\text{s}$ during idle verification still preserves the same front. For CIFAR-10 ($\epsilon=2/255$), total verification time for MTL-IBP was reduced from 1311 hours to 208 hours.

3. **Single-linkage Clustering + Complete Verification Refinement**:
    - **Function**: Avoids expensive complete verification for "nearly overlapping" configurations in the Pareto set while ensuring all points on the final curve are based on complete verification.
    - **Mechanism**: Hierarchical single-linkage clustering is performed using Euclidean distance in the 2D objective space. Hyperparameter points $i, j$ are merged if the distance is $\le d_{\min}=0.05$. Clustering is triggered if the Pareto set exceeds 5 points. One configuration per cluster is randomly selected for full $\alpha\beta$-CROWN, and the Pareto front is reconstructed using true certified accuracy.
    - **Design Motivation**: BO tends to sample densely near the front, leading to many "nearly identical performance" points with $<0.5\%$ difference. Without deduplication, the verification budget would be wasted on decorative details. Clustering keeps verification costs at the same order of magnitude as single-point tuning while ensuring reported figures are hard numbers under complete verification.

### Loss & Training
The training side follows the existing losses of each method: IBP's $\kappa \cdot \mathcal{L} + (1-\kappa) \cdot \mathcal{L}_{\text{ver}}$, CROWN-IBP's use of $\beta$ to transition between CROWN-IBP and IBP bounds, SABR's $\tau \epsilon$ sub-interval + ReLU shrinking, and MTL-IBP's $\alpha \cdot \mathcal{L}_{\text{ver}} + (1-\alpha) \cdot \mathcal{L}_{\text{adv}}$. The difference lies in the outer loop: the authors relax the $\kappa_{\text{start}} \ge \kappa_{\text{end}}$ bounds, allow up to 5 warm-up epochs (prior work often used 1), allow training $\epsilon$ to exceed evaluation $\epsilon$, and include both $\ell_1$ and Shi 2021 regularization in the search. This ensures the search space covers design regions previously obscured by default values. All experiments use the CNN7 architecture from Shi 2021, with EHVI run via BoTorch + Optuna on a budget of 3 seeds $\times$ 100 trials.

## Key Experimental Results

### Main Results
Four methods were compared using CNN7 on CIFAR-10 ($\epsilon \in \{2/255, 8/255\}$) and Tiny ImageNet ($\epsilon = 1/255$), with comparisons against original papers and CTBench.

| Dataset | $\epsilon$ | Method | Clean vs. Prev. SOTA | Certified vs. Prev. SOTA |
|--------|-----------|------|--------------------|--------------------|
| CIFAR-10 | $2/255$ | SABR | $\ge +1\%$ | $\ge +1\%$ |
| CIFAR-10 | $2/255$ | CROWN-IBP | $\sim +6\%$ | Comparable |
| CIFAR-10 | $8/255$ | IBP | Significant gain | Comparable |
| Tiny ImageNet | $1/255$ | MTL-IBP | $\sim +2\%$ | $\sim +2\%$ |
| Tiny ImageNet | $1/255$ | SABR | Slightly > MTL-IBP | Slightly < MTL-IBP |

By merging Pareto fronts, the authors found: on CIFAR-10 $2/255$, SABR and MTL-IBP are complementary, jointly forming the front; at $8/255$, all four methods contribute points; on Tiny ImageNet, SABR dominates the "high clean" end, while MTL-IBP dominates the "high certified" end. This redefines "who is SOTA" as "which method dominates in your specific trade-off interval of interest."

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Val-tuning vs. Test-tuning | Front strictly dominated | Prior work typically tuned on test sets, overestimating absolute values |
| Cutoff $1000\,\text{s} \to 100\,\text{s}$ | Front unchanged | Computational cost can be reduced by over an order of magnitude |
| BO trial count $100 \to 50$ | Front significantly degraded | Optimization budget is more sensitive than verification timeout |
| Removing $\kappa$ transition | IBP / CROWN-IBP fall off the front | $\kappa_{\text{start}}, \kappa_{\text{end}}$ are high-importance hyperparameters in all scenarios |

### Key Findings
- fANOVA importance analysis shows that the $\kappa$ transition is the primary control variable for the trade-off in IBP / CROWN-IBP. The community's habit of defaulting it to 0 is the main reason these older methods appeared obsolete.
- SABR's sub-selection dominates its front position more than $\tau$ or PGD attack parameters; MTL-IBP's $\alpha$ and training/attack $\epsilon$ scaling factors jointly determine the reachable region.
- Under large perturbation radii like $8/255$, the four methods converge to similar results—indicating that the true bottleneck in this regime is not loss design but the inherent relaxation of the IBP bound.
- "Tuning hyperparameters on the test set" is a default habit in the community, but Pareto fronts from validation-set tuning are strictly worse, implying that absolute figures in prior literature suffer from generalization overestimation.

## Highlights & Insights
- A seemingly purely "methodological" shift quantitatively rewrites the SOTA leaderboard of the past 5 years: CROWN-IBP, a method from 2020, was marginalized simply due to poor $\kappa$ tuning; "algorithmic progress" has been significantly overestimated.
- The three-stage pipeline (cheap proxy $\to$ clustering $\to$ terminal complete verification) is the key engineering maneuver for introducing multi-objective Bayesian optimization to certified training. This template is transferable to any benchmark characterized by "cheap training but expensive evaluation" (e.g., robustness, fairness).
- "Method complementarity" is quantified for the first time: practitioners should no longer ask "SABR or MTL-IBP," but rather "where is my target in the trade-off space?"

## Limitations & Future Work
- All experiments are limited to the $\ell_\infty$ threat model and CNN7 architecture; whether results hold for $\ell_2$, $\ell_1$, or Transformers remains an open question.
- The protocol itself is computationally intensive (3 seeds $\times$ 100 trials + complete verification per method). Despite the optimization via proxies and clustering, groups without large clusters may struggle to run it, raising the barrier for fair evaluation.
- The authors suggest that future work should shift toward "cheaply verifiable" training objectives rather than inflating complete verification timeouts to squeeze out a few more certificates—a subtle critique of current SABR / MTL-IBP practices, though no specific loss formulation for "verifiability" is provided.

## Related Work & Insights
- **vs. CTBench (Mao 2025)**: CTBench uses 250 grid search trials for single-objective comparison, concluding that "highest certified accuracy wins." This paper uses 300 BO trials for multi-objective comparison, revealing the true fronts undervalued by CTBench and emphasizing method complementarity.
- **vs. De Palma 2024b (MTL-IBP original)**: The original paper reported a single point biased toward certified accuracy. This paper proves MTL-IBP can achieve a previously unreported $\sim 2\%$ gain on the clean end for Tiny ImageNet; the original authors underestimated their own method.
- **vs. Müller 2023 (SABR original)**: The best point for SABR on CIFAR-10 $2/255$ from the original paper is outperformed by this work by $1\%$ in both clean and certified accuracy; it also shows SABR is not universally optimal at large $\epsilon$.

## Rating
- Novelty: ⭐⭐⭐⭐ Technically an application of mature MOBO, but using the Pareto front as the evaluation paradigm for certified training is a clear shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence from 4 methods, 3 benchmarks, and multiple ablations, including fANOVA and budget sensitivity.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation, though the discussion on making the protocol accessible for smaller labs is light.
- Value: ⭐⭐⭐⭐⭐ Rewrites the certified training leaderboard and provides the reusable open-source tool CTRAIN, with significant community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](../../AAAI2026/others/certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[ICML 2026\] Rethinking FID Through the Geometry of the Reference Dataset](rethinking_fid_through_the_geometry_of_the_reference_dataset.md)
- [\[ICML 2026\] CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)
- [\[CVPR 2026\] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](../../CVPR2026/others/rethinking_snn_online_training_and_deployment_grad.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)

</div>

<!-- RELATED:END -->
