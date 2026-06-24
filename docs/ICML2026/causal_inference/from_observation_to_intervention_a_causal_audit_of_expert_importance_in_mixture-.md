---
title: >-
  [Paper Note] From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models
description: >-
  [ICML2026][Causal Inference][MoE pruning] The authors use an interventional audit of "per-token ablation" to test the implicit assumption in MoE pruning that "observational routing statistics can predict which experts are deletable." On three high-redundancy MoE models, they obtain a clean "three-model null result": none of the 60 metric-layer combinations predict the causal importance of experts after multiple-comparison correction. This suggests that existing pruning method…
tags:
  - "ICML2026"
  - "Causal Inference"
  - "MoE pruning"
  - "Causal audit"
  - "Interventional evidence"
  - "Expert importance"
  - "Routing statistics"
date: 2026-05-08
content_hash: 32d3efb8038e33c3
---

# From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models

**Conference**: ICML2026  
**arXiv**: [2606.10703](https://arxiv.org/abs/2606.10703)  
**Code**: https://github.com/callmeloui/observational_metrics  
**Area**: Causal Inference / Mechanistic Interpretability  
**Keywords**: MoE pruning, Causal audit, Interventional evidence, Expert importance, Routing statistics

## TL;DR
The authors use an interventional audit of "per-token ablation" to test the implicit assumption in MoE pruning that "observational routing statistics can predict which experts are deletable." On three high-redundancy MoE models, they obtain a clean "three-model null result": none of the 60 metric-layer combinations predict the causal importance of experts after multiple-comparison correction. This suggests that existing pruning methods are effective not because metrics successfully identify "useless experts," but because redundancy in early and middle layers makes almost any selection criterion equally safe.

## Background & Motivation
**Background**: A common but rarely scrutinized reasoning pattern in interpretability is using "statistics calculated on observed model behavior" to predict "what will happen if a specific intervention is performed." Attention weights are read as explanations for single predictions, gradient saliency is read as "deleting these inputs will change the output," and routing statistics in MoE (utilization, activation norm, routing weight distribution) are read as "these experts are routed less / unimportant, and deleting them won't affect functionality."

**Limitations of Prior Work**: Following Pearl’s Causal Ladder, this involves using rung-1 (associational evidence) to draw rung-2 (interventional) conclusions. The validity of this leap from "observation" to "intervention" has almost never been directly tested; whenever it has been tested, it often fails—Jain & Wallace (2019) showed that attention weights fail to predict the effects of input perturbations, and Adebayo et al. (2020) showed that saliency maps "survive" even after randomizing model weights. MoE pruning literature is built on the same untested proxy assumption: ranking experts according to an observational criterion $\rightarrow$ deleting the lowest-ranked $\rightarrow$ fine-tuning to recover performance.

**Key Challenge**: The intuition that "experts routed more frequently/intensely = doing more important work" is a **population-level** average statistic, whereas the actual concern of pruning—whether deleting an expert will change the model's prediction on a specific token—is a **token-level** causal problem. The former does not guarantee the latter.

**Goal**: To ground this abstract "observation $\rightarrow$ intervention" gap in a concrete, actionable, and falsifiable instance—MoE expert pruning—and provide clear counterexamples.

**Key Insight**: Since the ability of observational metric rankings to predict which expert ablation actually changes behavior is a causal proposition, one should test it directly via **intervention** (ablation) rather than calculating another observational correlation.

**Core Idea**: Conduct paired ablation audits at the token level—for the same token, ablate the expert with the highest metric rank and the one with the lowest rank, then check if the difference in loss change is reliably positive. If population statistics are truly effective, this difference should be systematically greater than zero.

## Method

### Overall Architecture
This is not a methodology-proposing paper but an **audit** paper. The problem it addresses is "whether observational routing statistics are qualified to predict the causal consequences of expert ablation." The approach translates this question into a statistically testable per-token experiment, executed across three models spanning major contemporary MoE design dimensions.

The process is: for each "metric-layer-model" combination, sample $n=200$ token positions; at each position, identify the set of currently activated routed experts $\mathcal{A}_t$, rank them by the target metric, ablate the highest and lowest-ranked experts separately, and record the respective loss changes $\Delta\mathcal{L}_i^{(t)}=\mathcal{L}_t^{(-i)}-\mathcal{L}_t$ (where $\mathcal{L}_t=-\log p_\theta(x_{t+1}\mid x_{\le t})$); use paired $t$-tests (with Cohen's $d$ as effect size) cross-validated with Wilcoxon signed-rank tests; finally, apply Bonferroni multiple-comparison correction across all combinations. If all observational metrics fail, a "token-conditioned routing weight" is used as a control experiment to confirm that null results are not due to insufficient statistical power.

### Key Designs

**1. Per-token intervention audit protocol: Formulating "metric effectiveness" as a falsifiable causal proposition**

The pain point is that prior literature claiming "low-utilization experts are deletable" never transformed this statement into an assertion that could be experimental refuted. The authors provide Definition 2.1 (Causal Effectiveness of a Metric): An observational metric $m$ is **causally effective** at the token level if and only if, within the active expert set $\mathcal{A}_t$ at token position $t$, a higher $m(e)$ predicts greater functional importance $\Delta\mathcal{L}_e^{(t)}$. This refines "can we use $m$ to select experts for pruning" into "deleting experts with low $m$ should prioritize deleting those with low importance." Statistically, effectiveness is equivalent to the paired difference $\Delta\mathcal{L}_\text{high}-\Delta\mathcal{L}_\text{low}$ being reliably positive.

Two quantities are distinguished: **functional importance** $\Delta\mathcal{L}$ asks "did ablation change the prediction at this token"; **gap norm** $\delta_i^{(t)}=\lVert\mathbf{y}_t-\mathbf{y}_t^{(-i)}\rVert_2$ asks "did ablation change the residual flow, regardless of whether this change reached the output." The latter is used to diagnose anomalies in the final layers of OLMoE. Shared experts (present in Qwen, DeepSeek) are **never modified** during the audit; only routed experts are affected.

**2. Routing weight control experiment: Ruling out "null results due to insufficient power"**

The greatest threat to a clean null result is whether the experiment simply lacked the power to detect a weak signal. The authors use the **exact same machinery** (same $n$, same paired ablation, same tests) but replace observational metrics with **per-token routing weights** $g_i(\mathbf{x}_t)$. Note that this quantity is fundamentally different from the "population average routing weight" metric: it is conditioned on the current token and represents the upper bound of predictive signal that any routing-derived quantity could theoretically obtain.

If this machinery could not find even one signal, it would be a power issue; however, the experiment **did find** a Bonferroni-significant signal (OLMoE final layer), proving the protocol has sufficient power. The null results for observational metrics are "true zeros," not a lack of effort. This step is the most critical "negative control" in the paper's methodology.

**3. Progressive ablation: Directly confirming the "redundancy zone" as the reason for pruning success**

If observational metrics do not identify deletable experts, why do pruning methods reported in literature work? The authors provide a mechanistic explanation via progressive ablation: on OLMoE, they remove the $k$ highest-weighted active experts at a target layer and record cumulative loss changes ($n=500$ tokens per layer-$k$ cell, $k\in\{1,\dots,7\}$). Results show that in layers 0–9, even at $k=7$ (deleting seven out of eight active experts), the average loss change remains $<+0.083$ nats, while layer 15 "breaks" at $k=2$ ($\Delta\mathcal{L}=+0.155$ nats) and reaches $+0.431$ at $k=7$.

$$
\text{Early/Middle Layers:}\;\Delta\mathcal{L}(k{=}7)<0.083\ \text{nats}\quad\text{vs}\quad\text{Final Layer:}\;\Delta\mathcal{L}(k{=}2)=0.155\ \text{nats}
$$

The conclusion is robust: a **redundancy buffer** exists in early and middle layers, making almost any selection criterion appear "harmless" because almost all choices there are inherently harmless—the metrics are doing the same thing as a random baseline. This explains why the null results of observational metrics are **not contradictory** to the pruning gains reported in literature.

## Key Experimental Results

### Main Results
Three high-redundancy MoE language models were audited across 4 metrics × 5 representative layers = 20 combinations per model, totaling 60 combinations. Bonferroni correction: audit $\alpha_\text{adj}=0.05/20=0.0025$, control $\alpha_\text{adj}=0.05/5=0.01$. Corpus: WikiText-2 test set.

| Model | Architectural Highlights | Observational Metric Audit Results | Routing Weight Control |
|------|----------|------------------|--------------|
| OLMoE-1B-7B-0924 | 16L / 64E / top-8 / No shared | All 20 cells non-significant; only cell with $p_t{=}0.048$ (L11 std, $d{=}{+}0.141$) flipped sign in L15 ($d{=}{-}0.020$) | **L15 $d{=}{+}0.231,\ p{=}0.0013$, only significant result after correction in the entire experiment** |
| Qwen1.5-MoE-A2.7B | 24L / 60R+1S / top-4 | One cell passed uncorrected threshold but Wilcoxon failed ($p_W{=}0.036$), judged as noise | $\lvert d\rvert\le0.124$, no depth-based concentration |
| DeepSeek-V2-Lite | 27L / 64R+2S / top-6 | Three cells $p_t{<}0.05$, only one passed Wilcoxon (L20 utilization $d{=}{+}0.163$) | $\lvert d\rvert\le0.098$ |

Core Conclusion: **Not a single observational metric passed Bonferroni correction** across the 60 combinations. Effect sizes remained $<$ Cohen's $d=0.17$ throughout, with signs inconsistent across layers (a hallmark of a null distribution rather than a weak true signal). The "activation norm," rated as the strongest of 16 expert dropping criteria by Jaiswal et al. (2025), showed $\lvert d\rvert\le0.157$ in every model and layer, consistently non-significant.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|----------|------|
| Progressive Ablation L0–L9 | $\Delta\mathcal{L}<+0.083$ nats at $k{=}7$ | Functional tolerance when 7 out of 8 are deleted $\rightarrow$ Redundancy Buffer |
| Progressive Ablation L15 | Break at $k{=}2$, $+0.431$ nats at $k{=}7$ | No redundancy in final layer; experts are truly useful |
| gap norm vs. Depth | $0.0041$ at L0 $\rightarrow$ $0.1697$ at L15 | Residual flow displacement amplified $41\times$ |
| Qwen Control | Similar gap-norm amplification but no functional concentration | Rule out "residual flow growth" as a sufficient explanation |

### Key Findings
- **The three-model null result is universal**: Across contemporary MoE design dimensions (single/multi-objective load balancing, top-$k$ ratios of 6.7%–12.5%, train-from-scratch vs. dense upcycling, with/without shared experts), no metric-layer cell was significant after correction.
- **The OLMoE final-layer effect is narrow**: It appeared in only one model, one layer, and only under direct token-level conditioning; it is an existence proof of "a per-token predictor exists," **not** evidence that any deployable metric is effective (since $g_i(\mathbf{x}_t)$ requires a forward pass to that layer and token).
- Qwen exhibits equivalent gap-norm amplification without OLMoE's functional concentration, indicating that residual flow growth is insufficient to explain the final-layer effect; three architectures are insufficient to attribute this to any single factor, so the authors treat it as an empirical regularity requiring controlled follow-up.

## Highlights & Insights
- **Turning abstract methodological critique into falsifiable experiments**: The claim by Joshi et al. (2026) that "interpretability claims require interventional evidence" is a general statement; this paper grounds it in the specific scenario of MoE pruning, providing a "third counterexample" alongside attention and saliency.
- **Textbook negative control design**: Replacing observational metrics with per-token routing weights using the same machinery and $n$ both rules out insufficient power and happens to uncover the only true signal—one action simultaneously answers "Is the null result credible?" and "What kind of quantity actually has predictive power?"
- **The insight "metrics do the same thing as random baselines" is transferable**: Any work performing selective deletion on high-redundancy substructures (not just MoE, but also channel pruning, head pruning) should be wary—reported gains may come from the redundancy buffer rather than the selection criterion. A random baseline control is essential.

## Limitations & Future Work
- **Ours acknowledges**: Three architectures are insufficient to attribute the OLMoE final-layer effect to any single factor; controlled checkpoint tracking within an architecture is needed.
- The audit is token-level single-expert ablation, which is not directly equivalent to a deployment pipeline of "deleting a batch of experts then fine-tuning"—Ours explicitly states it **does not** prove that metric-guided pruning fails as a deployment process, only that when it succeeds, success is not due to the metric identifying token-level important experts.
- While per-token routing weights are effective predictors, they require a forward pass to that layer/token and cannot be pre-computed before inference. Thus, they cannot be used directly as deployable pruning metrics—a gap remains between "predictive signals exist" and "deployable metrics exist."
- Future work: Extending audits to multi-expert joint ablation, controlled comparisons across more architectures (especially upcycled models), and generalizing the same "observation $\rightarrow$ intervention" audit to other interpretability units like attention heads and neurons.

## Related Work & Insights
- **vs. Jain & Wallace (2019) / Adebayo et al. (2020)**: They proved observational statistics fail to predict interventional effects in attention and saliency, respectively; this paper adds an empirical instance in MoE routing statistics using the same methodological lineage (testing association with intervention).
- **vs. Jaiswal et al. (2025) et al.**: They rank experts using observational criteria like activation norm and report pruning gains; Ours does not deny the gains but points out they stem from early-layer redundancy rather than the metrics' causal identification power.
- **vs. Joshi et al. (2026)**: They proposed the general thesis that "interpretability claims need interventional evidence to generalize"; this paper realizes this thesis as a concrete, statistically corrected counterexample in MoE pruning.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new method but a new auditing paradigm, grounding abstract causal critique in falsifiable experiments with a solid perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 60 combinations across three models + negative control + progressive ablation, statistically rigorous; but limited to three architectures and single-expert ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ The "two findings, two scales" argumentative structure is clear, and the caveats for null results are restrained and honest.
- Value: ⭐⭐⭐⭐ Sounds an alarm for the MoE pruning / interpretability community—stop using population observational statistics to promise token-level interventional conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding](../../ICLR2026/causal_inference/causal_imitation_learning_under_expert-observable_and_expert-unobservable_confou.md)
- [\[ICML 2026\] Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal-jepa_learning_world_models_through_object-level_latent_masking.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ACL 2025\] Causal Graph based Event Reasoning using Semantic Relation Experts](../../ACL2025/causal_inference/causal_graph_based_event_reasoning_using_semantic_relation_experts.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/causal_inference/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)

</div>

<!-- RELATED:END -->
