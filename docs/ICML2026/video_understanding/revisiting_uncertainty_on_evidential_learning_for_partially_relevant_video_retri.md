---
title: >-
  [Paper Note] Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval
description: >-
  [ICML 2026][Video Understanding][PRVR] This paper addresses the query ambiguity and temporally sparse supervision in Partially Relevant Video Retrieval (PRVR) caused by "short queries vs. long videos." It proposes Holmes…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "PRVR"
  - "Evidential Learning"
  - "Dirichlet Distribution"
  - "Optimal Transport"
  - "Query Ambiguity"
date: 2026-05-08
content_hash: f0447586e7e3bcd2
---

# Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval

**Conference**: ICML 2026  
**arXiv**: [2605.06083](https://arxiv.org/abs/2605.06083)  
**Code**: https://github.com/ICML26-Holmes (available)  
**Area**: Video Understanding / Cross-modal Retrieval / Uncertainty Modeling  
**Keywords**: PRVR, Evidential Learning, Dirichlet Distribution, Optimal Transport, Query Ambiguity

## TL;DR
This paper addresses the query ambiguity and temporally sparse supervision in Partially Relevant Video Retrieval (PRVR) caused by "short queries vs. long videos." It proposes Holmes, a hierarchical evidential learning framework based on the Dirichlet distribution. Holmes distinguishes precise, polysemous, and under-determined queries across videos using a triple-principle and adaptively calibrates labels. Within videos, it achieves dense alignment via flexible optimal transport with a dustbin. The method achieves SOTA on ActivityNet, Charades, and TVR datasets.

## Background & Motivation

**Background**: The PRVR task requires retrieving untrimmed long videos using a single-sentence text query that only describes a local segment. Mainstream approaches, such as MS-SL and GMMFormer, adopt multi-instance learning (MIL), treating "the clip with the highest similarity to the query" as the positive sample for contrastive learning, and then ranking based on a deterministic similarity score.

**Limitations of Prior Work**: Figure 1 in the paper identifies two failure modes: (1) At the inter-video level, short queries and rich video content inevitably lead to "under-determined queries" (insufficient semantic information, low similarity to all candidates) and "polysemous queries" (ambiguous semantics, high similarity to multiple candidates). Training all queries as precise queries forces them incorrectly toward a single ground truth. (2) At the intra-video level, MIL supervises only the best clip, resulting in extreme imbalance between positive and negative clips. The model is easily misled by local noise in globally irrelevant videos, causing spurious spiky activations.

**Key Challenge**: Existing methods treat cross-modal similarity as a deterministic output and do not quantify "how reliable the score itself is." Recent methods like ARL recognize ambiguity but can only coarsely judge "whether a pair is ambiguous," without distinguishing between "insufficient signal" and "contradictory signal," leading to miscalibrated supervision.

**Goal**: (i) Explicitly quantify the uncertainty of each query across videos and distinguish query types; (ii) Break the sparse supervision bottleneck of MIL within videos, providing dense and noise-robust alignment signals.

**Key Insight**: Treat cross-modal similarity as "evidence" rather than a "score"—the perspective of Evidential Deep Learning (EDL). EDL, via the second-order probability of the Dirichlet distribution, can simultaneously provide epistemic (cognitive) and aleatoric (statistical) uncertainty, corresponding precisely to "insufficient signal" and "contradictory signal" failure modes.

**Core Idea**: Use Dirichlet evidential learning to jointly model inter-video query uncertainty and intra-video temporal supervision sparsity. Apply a triple-principle (epistemic uncertainty + label consistency + aleatoric uncertainty) to bucket queries and adaptively calibrate labels. Replace MIL's hard argmax with optimal transport with a dustbin.

## Method

### Overall Architecture
Input: An untrimmed video $V$ and a text query $T$. The query is encoded by RoBERTa + Transformer into $\bm{q}\in\mathbb{R}^d$. The video is processed in two branches: frame-scale extracts $M_f$ frame features $\bm{V}_f$, and clip-scale extracts $M_c$ clip features $\bm{V}_c$. Max-cosine is applied at each scale to obtain similarities $s^f$ and $s^c$. The pipeline has two layers: (1) **Inter-video evidential learning** maps the similarity vector of $K$ candidate videos in a batch to Dirichlet parameters, buckets queries into precise/polysemous/under-determined using the triple-principle, and applies soft label calibration to polysemous queries; (2) **Intra-video evidential learning** replaces single-point argmax with optimal transport with a dustbin, forming soft alignments between a query and multiple clips as intra-video evidence. Training uses least squares evidential loss; inference still ranks by $s=\alpha_f s^f + \alpha_c s^c$.

### Key Designs

1. **Uncertainty-Guided Query Identification (UGI) Based on Triple Principle**:

    - **Function**: Automatically classifies each query as precise, polysemous, or under-determined, avoiding uniform loss application.
    - **Mechanism**: Construct evidence from similarity $e_{ij}=\exp(\tanh(s_{ij}/\tau))$ and obtain Dirichlet parameters $\alpha_{ij}=e_{ij}+1$. Define three quantities: epistemic uncertainty $u_i=K/S_i$ ($S_i=\sum_j\alpha_{ij}$, reflecting total evidence deficiency), label consistency $c_i=\max(0, \bm{s}_i\cdot\bm{y}_i)$ (response strength for the GT video), and aleatoric uncertainty $\xi_i$ (Dirichlet expected entropy). Identification rules: large $u_i$ → under-determined (sparse evidence); small $u_i$ and large $c_i$ → preliminary precise; small $u_i$ and small $c_i$ → preliminary polysemous; then, use the median of $\xi_i$ to reassign high-entropy "pseudo-precise" samples to polysemous. Thresholds $\beta_u,\beta_p$ are dynamically determined by "currently correctly matched samples" in the batch, avoiding manual tuning. The two scales independently classify and are fused by "uncertainty-priority": the more uncertain side dominates ($\mathcal{S}_p\prec\mathcal{S}_n\prec\mathcal{S}_u$).
    - **Design Motivation**: Previous methods discard or downweight all samples where the GT is not ranked first, losing signals that truly reflect ambiguity. In EDL, $u$ alone cannot distinguish precise from polysemous (Theorem 3.2); $c$ and $\xi$ are also needed for unambiguous classification in three-dimensional space.

2. **Query-Adaptive Label Calibration + Dynamic Co-Evidence Aggregation (DCEA)**:

    - **Function**: Applies different supervision strengths to different query types and fuses evidential opinions from two scales.
    - **Mechanism**: Precise and under-determined queries retain one-hot labels (the former are reliable, the latter need to reinforce learning signals). Polysemous queries soften labels as $\hat{\bm{y}}_i=(1-\gamma)\bm{y}_i+\frac{\gamma}{2}(\sigma(s_i^f)+\sigma(s_i^c))$ ($\gamma=0.2$), allowing the model to distribute belief among multiple semantically relevant candidates and avoid over-penalization. The evidential opinions $\mathbb{M}^f,\mathbb{M}^c$ from two scales are fused using the Dempster–Shafer rule: $b_k^o=\frac{1}{1-\delta}(b_k^f b_k^c+b_k^f u^c+b_k^c u^f)$, where $\delta=\sum_{i\neq j}b_i^f b_j^c$ measures conflict.
    - **Design Motivation**: Hard labels treat semantically relevant candidates in polysemous queries as negatives, introducing supervision noise; fully soft labels dilute signals for precise queries. This "differentiated treatment after classification" aligns relevant candidates for polysemous queries and maintains discrimination for precise ones. DST fusion, rather than simple weighting, automatically reflects higher total uncertainty when the two branches conflict.

3. **Flexible Optimal Transport (FOT) with Dustbin for Accumulating Intra-video Evidence**:

    - **Function**: Replaces MIL's "supervise only the best clip" with dense soft alignment between query and clips, automatically suppressing noisy clips.
    - **Mechanism**: Treat a query and $M_c$ clips as source/target in OT, with an extra dustbin at the target to absorb irrelevant clips, yielding a flexible transport plan $\bm{\pi}\in\mathbb{R}^{1\times(M_c+1)}$. The first $M_c$ entries of $\bm{\pi}$ serve as intra-video soft assignment supervision; a larger dustbin mass indicates lower overall query-video relevance.
    - **Design Motivation**: MIL supervision is sparse and easily misled by local noise (Figure 1e); standard OT forces all mass onto clips, with no escape for noise. The dustbin allows the model to explicitly learn "which clips to ignore," achieving both dense supervision and noise robustness.

### Loss & Training
Derived from EDL's least squares Dirichlet loss: $L_U(\bm{\alpha}_i,\hat{\bm{y}}_i)=\sum_j(\hat y_{ij}-\alpha_{ij}/S_i)^2+\alpha_{ij}(S_i-\alpha_{ij})/(S_i^2(S_i+1))$. The total loss supervises frame, clip, and aggregated evidential opinions: $L_{\text{inter}}=L_U^f+L_U^c+L_U^o$. Intra-video soft labels from OT are also fed into $L_U$. Hyperparameters: $\tau=0.1$, $\gamma=0.2$, $\beta=0.3$; thresholds are dynamically updated during training, requiring no manual tuning.

## Key Experimental Results

### Main Results
On ActivityNet Captions, Charades-STA, and TVR standard PRVR datasets, R@1/5/10/100 + SumR are compared:

| Dataset | Metric | Holmes | Prev. SOTA | Gain |
|---------|--------|--------|------------|------|
| ActivityNet | SumR | Significantly highest (>148.3) | ARL 148.3 | $\approx$ +2 SumR |
| Charades-STA | SumR | Best | MamFusion 76.5 | Further improved |
| TVR | SumR | Best | ARL 185.9 | +1~3 SumR |

Holmes surpasses all PRVR baselines including ARL, MGAKD, ProtoPRVR, MamFusion, as well as strong VCMR baselines (CONQUER, JSG) and T2VR baselines (CLIP4Clip, Cap4Video) on SumR across all three datasets.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Full Holmes | Complete model, SOTA SumR on all datasets |
| w/o UGI (no bucketing) | Degrades to uniform hard label training; polysemous queries are over-penalized, SumR drops significantly |
| w/o Label Calibration | Bucketing retained but polysemous not softened; performance between w/o UGI and Full |
| w/o DCEA (no DST fusion) | Cannot explicitly model opinion conflicts; uncertainty estimation is biased low |
| w/o FOT (revert to MIL) | Intra-video supervision reverts to sparse single-point, affected by spurious local responses; largest SumR drop |
| w/o dustbin (standard OT) | Noisy clips are forced to receive probability, alignment is contaminated |

### Key Findings
- Among all modules, FOT + dustbin contributes most to dense intra-video supervision; removing it causes the largest performance drop, confirming that MIL's sparse supervision is the true bottleneck in PRVR.
- All three principles $u$, $c$, and $\xi$ are indispensable: Theorem 3.2 and Proposition 3.4 theoretically prove that no single metric can distinguish precise from polysemous; ablation shows removing $\xi$ leads to the most severe polysemous bucket contamination.
- Qualitative visualization shows Holmes correctly identifies "under-determined / polysemous" queries and significantly suppresses spiky activation on globally irrelevant videos in Charades.

## Highlights & Insights
- **Two-dimensional Decomposition of Uncertainty**: The long-standing "ambiguity" in "short query, long video" is cleanly decomposed into "signal sparsity (epistemic)" and "signal polysemy (aleatoric)"—EDL's second-order uncertainty precisely matches the two failure modes in retrieval, providing an "aha" perspective.
- **Provable Necessity of Triple Principle**: Theorem 3.2 / Proposition 3.4 demonstrate that $u$ and $c$ alone are insufficient; $\xi$ must be introduced, turning the previously heuristic "add more criteria" into a theoretically grounded rule.
- **Dustbin OT is Fully Transferable**: Any task involving alignment supervision but susceptible to noisy pairings (e.g., partial retrieval, weakly supervised localization, ITM) can adopt the same trick—add a virtual bucket to absorb irrelevant elements.

## Limitations & Future Work
- Although the triple-principle thresholds are adaptive, they still rely on statistics from "currently correctly matched" samples in the batch, which may yield unstable initial bucketing in cold-start or challenging datasets; EMA smoothing across batches could be considered.
- The paper only validates on three standard PRVR datasets, and frame/clip representations still depend on RoBERTa + CNN, without leveraging stronger CLIP video encoders; combining with CLIP-style large models could further enhance evidential calibration.
- DST fusion assumes independence between branches; when frame/clip are highly correlated, $u^o$ may be overestimated; future work could incorporate correlation modeling.
- Inference still uses weighted sum for ranking, rather than directly employing evidential opinion (e.g., belief-based scoring), which is a clear extension point.

## Related Work & Insights
- **vs ARL (Cho 2025)**: ARL only performs binary classification of "ambiguous/unambiguous," while Holmes further splits ambiguity into precise / polysemous / under-determined using the triple-principle and provides differentiated label strategies.
- **vs GMMFormer / RAL / MSC-PRVR**: These methods use Gaussian attention or probabilistic embeddings to implicitly model ambiguity; Holmes explicitly provides interpretable uncertainty via Dirichlet.
- **vs MS-SL / ProtoPRVR**: Both are MIL paradigms lacking dense intra-video supervision; Holmes' FOT-dustbin offers the first truly dense and noise-robust alignment scheme for PRVR.
- **Cross-task Insights**: The EDL+OT combination can be applied to video moment retrieval, grounded VLM training, and weakly supervised detection, benefiting any "long context, short query" scenario.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to finely map EDL's two types of uncertainty to PRVR's two failure modes, and proposes dustbin OT; highly original framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three mainstream PRVR datasets + multiple baselines + complete ablation, but lacks comparison with CLIP-based large models and real noise robustness studies.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly illustrates motivation; definitions/theorems are rigorous; downside is dense abbreviations, requiring frequent reference to Figure 2 on first read.
- Value: ⭐⭐⭐⭐ Achieves SOTA on PRVR; both uncertainty modeling and dustbin OT are highly transferable to related retrieval/alignment tasks.

## Related Papers

- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)
- [\[ICML 2026\] 把分子动力学知识蒸到非自回归离子输运预测器里](teaching_molecular_dynamics_to_a_non-autoregressive_ionic_transport_predictor.md)
- [\[ICML 2026\] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](../../CVPR2026/video_understanding/virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](../../NeurIPS2025/video_understanding/revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)

</div>

<!-- RELATED:END -->
