---
title: >-
  [Paper Note] When Shift Happens - Confounding is to Blame
description: >-
  [ICLR 2026][learning_theory][Paper Note] Academic paper note for When Shift Happens - Confounding is to Blame.
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 3e4ab240f19eca0f
---
only XGB achieves a CI consistency of 0.92, corresponding to its significant lead in OOD performance (72.90% on subset A vs 62.75% for IRM). This perfectly matches the "maximize conditional information amount" target of Theorem 4.2.
- **Adding Covariates is Generally Beneficial, but Gains Diminish/Occasional Regression**: From C $\to$ AC $\to$ A, OOD accuracy increases for most methods (e.g., XGB 64.35 $\to$ 72.80 $\to$ 72.90), validating Proposition 4.1; however, MLP/GDRO show slight regression on A, indicating the side effect of amplified variation may manifest in some methods.
- **Invariance Methods Lose Out Significantly**: IRM's ID/OOD accuracy is lower across the board (OOD at only 61.14% on subset C), consistent with the theoretical judgment that pure invariance is sub-optimal under hidden confounding.
- **Synthetic Data Validation**: Under the known causal structure $U\to X, U\to Y, X\to Y, U\to X_I$, as the number of proxy variables $|X_I|$ increases, MSE decreases, conditional information amount and feature shift increase, and concept shift decreases (Fig. 4), consistent with the theory.

## Highlights & Insights
- **One Decomposition to Rule Them All**: Mapping methods like IRM / DANN / CDAN / GDRO as "manipulating a specific term in the predictive information decomposition" provides a highly reusable perspective — new OOD methods can be understood by asking which term they maximize or minimize.
- **The "Collapse" is the Strongest Step**: Using d-separation to make the six-term decomposition collapse precisely into two terms under hidden confounding turns the "what to learn" question from vague engineering intuition into a provable conclusion (learn environment-specific relationships, don't erase environment information), simultaneously explaining why ERM wins, MoE is sensible, and invariance methods fail.
- **Vindicating "Counter-intuitive Empirical Results"**: Changing "adding non-causal covariates is better" from a phenomenon that seems to violate causal intuition into an inevitable result of "proxies decreasing concept shift and raising conditional information amount" provides a theoretically grounded criterion for covariate selection (find proxies informative about $U$ or $Y$).
- **Transferable Thinking**: Using environment-specific statistics as proxies for hidden confounding to perform backdoor adjustment is transferable to any prediction task with unobserved confounding — one does not need to explicitly model $U$, but rather feed in enough environment/proxy information to recover the correct relationship.

## Limitations & Future Work
- **Explanation over Solution**: The authors explicitly state the target is explaining phenomena rather than providing a specific algorithm for hidden confounding shift; how to design new methods based on this remains an open question.
- **Theory Relies on Structural Assumptions**: The clean collapse in Theorem 4.2 depends on a clear unidirectional structure $X\to Y$ or $Y\to X$. In real data, $X\leftrightarrow Y$ is mixed (though the authors note $X\to Y$ dominates in 11 of 16 benchmarks); under more general entangled structures, the conclusion strength will be discounted.
- **Fragility of Mutual Information Estimation**: All conclusions are built on KSG mutual information estimation. In high-dimensional/small-sample settings, estimation bias might affect the reliability of measures like sign consistency; the paper does not deeply discuss the impact of estimation error.
- **Realism of Proxy Assumptions**: Proposition 4.1 requires $X_I$ to be an effective proxy for $U$/$Y$ and satisfy conditional independence, which is hard to verify in reality. The authors also list "handling entangled shifts without untestable proxy assumptions" as future work.
- **Potential Improvements**: Possible directions include quantifying the "cost of obtaining non-causal covariates vs accuracy gain," incorporating the side effects of amplified variation into explicit regularization, and modeling uncertainty regarding unobserved confounding into a new OOD-robust paradigm.

## Related Work & Insights
- **vs Nastl & Hardt (2024)**: They empirically found that "using all covariates is Pareto dominant for ID/OOD," but only provided the phenomenon without theory; this paper uses predictive information decomposition + Proposition 4.1 to explain why — adding covariates lowers concept shift and raises conditional information amount.
- **vs IRM / VREX / Group DRO**: These methods maximize/minimize specific terms in the decomposition (e.g., IRM suppresses variation, GDRO resists label shift); this paper proves these terms cancel out and are no longer the correct objective under hidden confounding, explaining why they are often outperformed by ERM.
- **vs Prashant et al. (2025)**: They proposed MoE (one expert per confounder value) for confounding shift; this paper proves MoE is equivalent to maximizing conditional information amount $I(\phi(X);Y\mid E)$, providing it with a theoretical basis while noting its assumptions of "confounder support overlap + discrete proxies" are strong, calling for more general methods.
- **vs anchor regression (Rothenhäusler et al., 2021) / Eastwood et al. (2023)**: The former performs a linear trade-off between "all covariates" and "purely causal covariates"; the latter argues unstable covariates can help when conditionally independent. This paper unifies these perspectives into a framework of "how adding proxies alters shift terms under hidden confounding."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies "why ERM wins" and "why non-causal covariates help" using a predictive information decomposition, offering a highly novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically validated with 8 real-world datasets + synthetic data + 5 method classes, though validation for estimation errors and general causal structures is slightly lacking.
- Writing Quality: ⭐⭐⭐⭐ Causal-information-theoretic reasoning is rigorous and motivations are clear; however, d-separation and multi-term decomposition present a hurdle for non-expert readers.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretically grounded guide for OOD generalization and covariate selection (learn environment-specific relations, collect informative proxies), with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

```mermaid
graph TD
    A[When Shift Happens - Confounding is to Blame (ICLR 2026)]
    B[Nastl & Hardt, 2024: All Covariates Often Pareto Dominate]
    C[Prashant et al., 2025: MoE for Hidden Confounding]
    D[Arjovsky et al., 2019: Invariant Risk Minimization]
    
    B -->|Empirical Foundation| A
    C -->|Specific Solution Explained| A
    D -->|Invariance Baseline| A
    A -->|Theoretical Explanation| B
```

- **Nastl, J., & Hardt, M. (2024).** Predictive models often perform better when trained on all available features, rather than just causal ones. *arXiv preprint*.
- **Prashant, K., et al. (2025).** Mixture of Experts for Handling Hidden Confounding in Out-of-Distribution Generalization. *ICLR 2025*.
- **Arjovsky, M., et al. (2019).** Invariant Risk Minimization. *arXiv preprint*.
</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] When Bias Meets Trainability: Connecting Theories of Initialization](when_bias_meets_trainability_connecting_theories_of_initialization.md)
- [\[ICLR 2026\] Neyman-Pearson Classification under Both Null and Alternative Distributions Shift](neyman-pearson_classification_under_both_null_and_alternative_distributions_shif.md)
- [\[ICLR 2026\] Know When to Abstain: Optimal Selective Classification with Likelihood Ratios](know_when_to_abstain_optimal_selective_classification_with_likelihood_ratios.md)
- [\[ICLR 2026\] Why Ask One When You Can Ask k? Learning-to-Defer to the Top-k Experts](why_ask_one_when_you_can_ask_k_learning-to-defer_to_the_top-k_experts.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](../../ICML2026/learning_theory/when_sample_selection_bias_precipitates_model_collapse.md)

</div>

<!-- RELATED:END -->
