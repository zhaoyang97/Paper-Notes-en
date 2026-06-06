---
title: >-
  [Paper Note] Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval
description: >-
  [ICML 2026][Video Understanding][PRVR] Addressing the issues of query ambiguity and sparse temporal supervision caused by "short queries vs. long videos" in Partially Relevant Video Retrieval (PRVR)…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "PRVR"
  - "Evidential Learning"
  - "Dirichlet Distribution"
  - "Optimal Transport"
  - "Query Ambiguity"
date: 2026-05-08
content_hash: e0cd5b7dd581af14
---

# Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval

**Conference**: ICML 2026  
**arXiv**: [2605.06083](https://arxiv.org/abs/2605.06083)  
**Code**: https://github.com/ICML26-Holmes (Available)  
**Area**: Video Understanding / Cross-modal Retrieval / Uncertainty Modeling  
**Keywords**: PRVR, Evidential Learning, Dirichlet Distribution, Optimal Transport, Query Ambiguity

## TL;DR
Addressing the issues of query ambiguity and sparse temporal supervision caused by "short queries vs. long videos" in Partially Relevant Video Retrieval (PRVR), this paper proposes Holmes, a hierarchical evidential learning framework based on the Dirichlet distribution. At the inter-video level, it distinguishes precise, polysemous, and under-determined queries using triple principles and adaptively calibrates labels. At the intra-video level, it achieves dense alignment using flexible optimal transport with a "dustbin." Holmes achieves SOTA performance on ActivityNet, Charades, and TVR datasets.

## Background & Motivation

**Background**: The PRVR task requires retrieving untrimmed long videos using a single-sentence text query that describes only a partial segment of the video. Mainstream approaches, represented by MS-SL and GMMFormer, employ multi-instance learning (MIL), treating the "clip with the highest similarity to the query" as the positive sample for contrastive learning and ranking based on a deterministic similarity score.

**Limitations of Prior Work**: The authors identify two failure modes in Figure 1: (1) At the inter-video level, short text with rich video content inevitably produces "under-determined queries" (insufficient semantic information, yielding low similarity for all candidates) and "polysemous queries" (ambiguous semantics, yielding high similarity for multiple candidates). Training these as precise queries incorrectly "hard-pushes" them toward a single ground truth. (2) At the intra-video level, MIL supervises only a single best clip. This extreme imbalance between positive and negative clips makes the model susceptible to being deceived by "coincidentally similar" local noise in globally irrelevant videos, resulting in spurious spiky activations.

**Key Challenge**: Existing methods treat cross-modal similarity as a deterministic output without quantifying "how trustworthy this score itself is." Recent methods like ARL recognize the existence of ambiguity but can only perform coarse-grained judgment on whether a pair is ambiguous, failing to distinguish between "insufficient signal" and "conflicting signals," thus resulting in incorrect calibration directions.

**Goal**: (i) Explicitly quantify the uncertainty of each query at the inter-video level and distinguish query types; (ii) Break the sparse supervision bottleneck at the intra-video level to provide alignment signals that are both dense and noise-robust.

**Key Insight**: Treat cross-modal similarity as "evidence" rather than just a "score"—this is the perspective of Evidential Deep Learning (EDL). EDL utilizes the second-order probabilities of the Dirichlet distribution to simultaneously provide epistemic and aleatoric uncertainty, which correspond precisely to the "insufficient signal" and "conflicting signal" failure modes.

**Core Idea**: Utilize Dirichlet evidential learning to simultaneously model inter-video query uncertainty and intra-video temporal supervision sparsity. Queries are bucketed using triple principles (epistemic uncertainty + label consistency + aleatoric uncertainty) with adaptive label calibration. Furthermore, flexible optimal transport with a "dustbin" replaces the hard argmax of MIL.

## Method

### Overall Architecture
Input: An untrimmed video $V$ and a text query $T$. The query is encoded via RoBERTa + Transformer into $\bm{q}\in\mathbb{R}^d$. The video features are extracted via two branches: frame-scale features $\bm{V}_f$ for $M_f$ frames and clip-scale features $\bm{V}_c$ for $M_c$ clips. Max-cosine similarity is used to obtain scores $s^f$ and $s^c$ for both scales. The overall pipeline consists of two layers: (1) **Inter-video evidential learning** maps similarity vectors of $K$ candidate videos in a batch to Dirichlet parameters, categorizes queries into precise/polysemous/under-determined buckets via triple principles, and performs soft label calibration for the polysemous bucket. (2) **Intra-video evidential learning** replaces single-point argmax with flexible optimal transport with a "dustbin," treating the soft alignment between a query and multiple clips as intra-video evidence. The model is trained jointly using a least-squares evidential loss. During inference, ranking is performed using $s=\alpha_f s^f + \alpha_c s^c$.

### Key Designs

1.  **Uncertainty-Guided Query Identification (UGI)**:
    - **Function**: Automatically categorizes a query into precise, polysemous, or under-determined types to avoid rigid training with a uniform loss.
    - **Mechanism**: Evidence $e_{ij}=\exp(\tanh(s_{ij}/\tau))$ is constructed from similarity, yielding Dirichlet parameters $\alpha_{ij}=e_{ij}+1$. Three metrics are defined: epistemic uncertainty $u_i=K/S_i$ (where $S_i=\sum_j\alpha_{ij}$, reflecting total evidence deficiency), label consistency $c_i=\max(0, \bm{s}_i\cdot\bm{y}_i)$ (response strength of the GT video), and aleatoric uncertainty $\xi_i$ (expected entropy of the Dirichlet). The rules are: high $u_i$ $\rightarrow$ under-determined (thin evidence); low $u_i$ and high $c_i$ $\rightarrow$ preliminary precise; low $u_i$ and low $c_i$ $\rightarrow$ preliminary polysemous. The median of $\xi_i$ is used to reclassify high-entropy "pseudo-precise" samples back to polysemous. Thresholds $\beta_u,\beta_p$ are determined dynamically based on currently correctly matched samples to avoid manual tuning. After independent classification at both scales, they are fused using an "uncertainty-priority" rule, letting the more uncertain side dominate ($\mathcal{S}_p\prec\mathcal{S}_n\prec\mathcal{S}_u$).
    - **Design Motivation**: Previous methods discarded or down-weighted all samples where GT was not ranked first as noisy correspondence, losing signals that reflect true ambiguity. In EDL, $u$ alone cannot distinguish between precise and polysemous (Theorem 3.2); $c$ and $\xi$ must be added to achieve unambiguous classification in 3D space.

2.  **Query-Adaptive Label Calibration + Dynamic Co-Evidence Aggregation (DCEA)**:
    - **Function**: Applies varying degrees of supervision tightness for different query types and merges evidential opinions from two scales into a unified view.
    - **Mechanism**: Precise and under-determined queries retain one-hot labels (the former is already credible, the latter needs continued learning signals). For polysemous queries, labels are softened to $\hat{\bm{y}}_i=(1-\gamma)\bm{y}_i+\frac{\gamma}{2}(\sigma(s_i^f)+\sigma(s_i^c))$ ($\gamma=0.2$), allowing the model to assign belief to multiple semantically relevant candidates. Evidential opinions $\mathbb{M}^f,\mathbb{M}^c$ are fused via the parameter-free Dempster–Shafer combination rule: $b_k^o=\frac{1}{1-\delta}(b_k^f b_k^c+b_k^f u^c+b_k^c u^f)$, where $\delta=\sum_{i\neq j}b_i^f b_j^c$ measures conflict between branches.
    - **Design Motivation**: Hard labels treat semantically relevant candidates of polysemous queries as negatives, causing supervision noise, while complete softening dilutes signals for precise queries. This "differential processing after classification" aligns relevant candidates for polysemous queries while maintaining discriminative power for precise ones. Using DST rules instead of simple weighting allows conflicting opinions to naturally reflect higher total uncertainty.

3.  **Flexible Optimal Transport (FOT) with Dustbin for Accumulating Intra-video Evidence**:
    - **Function**: Replaces the "supervise only the best clip" approach of MIL by establishing dense soft alignment between query and clips while automatically suppressing noisy clips.
    - **Mechanism**: A query and $M_c$ clips are treated as sources/sinks for OT, but an additional "dustbin" sink is added to absorb irrelevant clips, resulting in a flexible transport plan $\bm{\pi}\in\mathbb{R}^{1\times(M_c+1)}$. The first $M_c$ items in $\bm{\pi}$ serve as intra-video soft assignment supervision. A larger mass in the dustbin indicates lower overall relevance between the query and the video.
    - **Design Motivation**: MIL supervision is sparse and easily deceived by local noise (Figure 1e), whereas standard OT forces all mass into clips with no exit for noise. The dustbin design allows the model to explicitly learn which clips should be ignored, satisfying both dense supervision and noise robustness.

### Loss & Training
The objective is derived from the least-squares Dirichlet loss of EDL: $L_U(\bm{\alpha}_i,\hat{\bm{y}}_i)=\sum_j(\hat y_{ij}-\alpha_{ij}/S_i)^2+\alpha_{ij}(S_i-\alpha_{ij})/(S_i^2(S_i+1))$. The total loss supervises frame, clip, and aggregated evidential opinions simultaneously: $L_{\text{inter}}=L_U^f+L_U^c+L_U^o$. At the intra-video level, soft labels obtained via OT are similarly fed into the $L_U$ objective. Hyperparameters: $\tau=0.1, \gamma=0.2, \beta=0.3$. Thresholds adapt dynamically during training.

## Key Experimental Results

### Main Results
Comparison of R@1/5/10/100 and SumR on ActivityNet Captions, Charades-STA, and TVR:

| Dataset | Metric | Holmes | Prev. SOTA | Gain |
|--------|------|--------|---------------|------|
| ActivityNet | SumR | **Highest** (>148.3) | ARL 148.3 | $\approx$ +2 SumR |
| Charades-STA | SumR | **Best** | MamFusion 76.5 | Further Improvement |
| TVR | SumR | **Best** | ARL 185.9 | +1~3 SumR |

Holmes outperforms all PRVR baselines, including ARL, MGAKD, ProtoPRVR, and MamFusion, across all datasets. It also surpasses strong VCMR-style baselines (CONQUER, JSG) and T2VR baselines (CLIP4Clip, Cap4Video).

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Full Holmes | Complete model, achieves SOTA SumR on all three datasets. |
| w/o UGI (No Bucketing) | Degenerates to uniform hard label training; polysemous queries are over-penalized, leading to significant SumR drop. |
| w/o Label Calibration | Retains bucketing but does not soften polysemous labels; performance is between w/o UGI and Full. |
| w/o DCEA (No DST Fusion) | Unable to explicitly model branch conflicts, leading to underestimated uncertainty. |
| w/o FOT (Revert to MIL) | Intra-video supervision return to sparse single-point; most significant drop due to spurious local responses. |
| w/o Dustbin (Standard OT) | Noisy clips are forced to receive probability mass, contaminating the alignment. |

### Key Findings
- Among all modules, FOT + dustbin contributes most to the dense intra-video supervision; its removal results in the largest performance drop, confirming that MIL's sparse supervision is a bottleneck in PRVR.
- All three components ($u, c, \xi$) of the triple principles are indispensable. Theorem 3.2 and Proposition 3.4 theoretically prove that any single metric cannot distinguish between precise and polysemous queries.
- Qualitative visualization shows that Holmes correctly identifies "under-determined / polysemous" queries and significantly suppresses spiky activations on globally irrelevant videos in Charades.

## Highlights & Insights
- **2D Decomposition of Uncertainty**: The long-standing issue of "ambiguity" in "short query, long video" scenarios is cleanly decomposed into two orthogonal dimensions: "signal sparsity (epistemic)" and "signal multiplicity (aleatoric)." Mapping EDL's second-order uncertainty to these failure modes is a major contribution.
- **Provable Necessity of Triple Principles**: The use of Theorem 3.2 and Proposition 3.4 to argue that $u$ and $c$ are insufficient without $\xi$ transforms a heuristic "set of criteria" into a theoretically grounded discrimination rule.
- **Transferable OT-Dustbin Concept**: The strategy of using OT with a dustbin to achieve dense supervision while avoiding noise contamination is highly transferable to other tasks like partial retrieval, weakly supervised localization, and ITM.

## Limitations & Future Work
- While thresholds are adaptive, they rely on statistics from "already correctly matched" samples within a batch, which might lead to unstable initial partitions during cold starts or on difficult datasets. EMA could be used for smoothing.
- The paper only validates three standard PRVR datasets and the frame/clip representations still rely on RoBERTa + CNN; integration with stronger CLIP video encoders remains to be explored.
- DST fusion assumes branch independence, which might overestimate $u^o$ when frame and clip features are highly correlated.
- Inference still relies on weighted sums; directly using evidential opinions for ranking (e.g., belief-based scoring) represents an obvious extension.

## Related Work & Insights
- **vs. ARL (Cho 2025)**: ARL only performs binary classification of "ambiguous/non-ambiguous," whereas Holmes uses triple principles to categorize queries into three types with differentiated label strategies.
- **vs. GMMFormer / RAL / MSC-PRVR**: These methods implicitly model ambiguity via Gaussian attention or probabilistic embeddings, while Holmes provides explicit, interpretable uncertainty using Dirichlet distributions.
- **vs. MS-SL / ProtoPRVR**: These MIL-based paradigms lack dense intra-video supervision. Holmes' FOT-dustbin provides the first truly dense yet noise-resilient alignment scheme for PRVR.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to precisely map the two types of EDL uncertainty to PRVR failure modes with a matching dustbin OT framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering three major PRVR datasets and complete ablations, though lacking comparison with CLIP-based backbones.
- Writing Quality: ⭐⭐⭐⭐ Intuitive motivation via Figure 1 and rigorous derivations; however, the density of acronyms requires frequent cross-referencing.
- Value: ⭐⭐⭐⭐ Achieves SOTA on PRVR; the uncertainty modeling and dustbin OT components are highly transferable to related retrieval/alignment tasks.

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
