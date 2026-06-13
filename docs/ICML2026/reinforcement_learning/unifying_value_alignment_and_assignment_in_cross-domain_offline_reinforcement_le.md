---
title: >-
  [Paper Note] Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline Reinforcement Learning] This paper reveals the "value misassignment" problem in heterogeneous cross-domain offline RL settings—where source data comes from multiple domains and…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline Reinforcement Learning"
  - "Cross-Domain Transfer"
  - "Value Alignment"
  - "Heterogeneous Datasets"
date: 2026-05-08
content_hash: d91adf1fd8639a50
---

# Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.24862](https://arxiv.org/abs/2605.24862)  
**Code**: https://github.com/zq2r/V2A.git  
**Area**: Reinforcement Learning / Cross-Domain Learning  
**Keywords**: Offline Reinforcement Learning, Cross-Domain Transfer, Value Alignment, Heterogeneous Datasets

## TL;DR
This paper reveals the "value misassignment" problem in heterogeneous cross-domain offline RL settings—where source data comes from multiple domains and policies, inaccurate advantage function estimation leads to the failure of data filtering. The proposed V2A framework addresses value alignment and assignment in a unified manner through temporally consistent modality representation learning and modality-aware advantage learning, improving performance by 21.4% over DVDF.

## Background & Motivation

**Background**: Cross-domain offline RL addresses data scarcity by combining sufficient source domain data with limited target domain data. Recent methods perform data filtering from the perspectives of dynamics alignment (IGDF/OTDF) or value alignment (DVDF).

**Limitations of Prior Work**: Existing methods assume source data originates from a single environment and single policy (unimodal). However, in real-world scenarios (e.g., robotics), source data often comprises multiple source domains and diverse behavioral policies (multimodal mixture), where existing methods suffer significant performance degradation.

**Key Challenge**: When source data is heterogeneous, DVDF uses a global advantage function $A_{\text{insrc}}^{\star}(s,a)$ to evaluate sample quality across sub-datasets. However, the same advantage value represents different relative qualities under different dynamics, leading to "value misassignment"—erroneously scoring low-quality samples as high-quality.

**Goal**: In the heterogeneous cross-domain offline RL setting, the objective is to align dynamics, align values, and correctly assign values simultaneously.

**Key Insight**: The critical insight is to differentiate distinct dynamics modalities within the source data through clustering and learn separate advantage functions for each modality to ensure accurate value estimation.

**Core Idea**: An end-to-end V2A framework is developed using EM methods to learn temporally consistent modality representations, followed by modality-aware advantage learning and subsequent data filtering.

## Method

### Overall Architecture
V2A consists of three stages: (1) extracting dynamics modalities from source data; (2) relabeling data based on modalities and learning accurate advantage functions; (3) filtering source samples using modality-aware scores.

### Key Designs

1. **Temporally Consistent Modality Representation Learning**:

    - **Function**: Extracts a latent dynamics modality representation $z$ for each trajectory from heterogeneous source data.
    - **Mechanism**: Standard ELBO encodes $z$ separately for each transition in the same trajectory, causing temporal inconsistency. V2A improves this with trajectory-level encoding and transition-level decoding, optimized alternatingly via EM—the E-step fixes the decoder to optimize the encoder, and the M-step fixes the encoder to optimize the decoder. The loss is $\mathsf{TC\text{-}ELBO} = \mathbb{E}_{\tau,z}[\sum_t \log p_\theta(s_{t+1}|s_t,a_t,z) - D_{KL}(q_\psi(\cdot|\tau),p(\cdot))]$.
    - **Design Motivation**: Ensures all transitions within the same trajectory share a single $z$ reflecting a unified dynamics environment, preventing modality identification failure due to random fluctuations.

2. **Modality-aware Advantage Learning**:

    - **Function**: Trains modality-conditional advantage functions $A(s,a,z)$ using relabeled data based on the learned modality representations.
    - **Mechanism**: Modality $z$ is input into Q and V functions. Sparse-QL is used to learn $Q(s,a,z)$ and $V(s,z)$, defining the advantage as $A(s,a,z) = Q(s,a,z) - V(s,z)$. The same $(s,a)$ pair receives different advantage values under different modalities, accurately reflecting relative quality.
    - **Design Motivation**: Corrects the fundamental flaw of DVDF—where a global advantage function cannot distinguish modality differences; the modality-aware version "scores" each dynamics environment individually.

3. **Modality-aware Data Filtering**:

    - **Function**: Performs data filtering based on modality-aware scores, selecting source samples in the top $\xi$ quantile.
    - **Mechanism**: The score $f(s,a,s',z) = \lambda \cdot h(s,a,s') + (1-\lambda) \cdot \text{Norm}(A(s,a,z))$ integrates dynamics alignment $h$ and modality-aware advantage $A$, with the weight $\lambda$ (fixed at 0.6 in experiments).
    - **Design Motivation**: Guarantees that selected samples are both dynamics-consistent with the target and high in quality; modality awareness prevents the filtering process from being "deceived."

## Key Experimental Results

### Main Results

| Method | IGDF | DVDF | V2A | OTDF | DVDF | V2A |
|------|------|------|-----|------|------|-----|
| Total Score | 1286.7 | 1374.7 | **1562.5** | 1319.5 | 1395.9 | **1612.9** |
| Gain | — | +6.8% | +21.4% | — | +5.5% | +22.2% |

Tests were conducted across 4 tasks (HalfCheetah / Hopper / Walker2d / Ant) × 6 source-target combinations. V2A outperformed IGDF in 20/24 tasks and OTDF in 21/24 tasks.

### Ablation Study

| Analysis Dimension | Result | Explanation |
|---------|------|------|
| Modality Quality | Fig 2(a) t-SNE | Trajectories from the same source domain cluster tightly, while different shift types are separated. |
| Advantage Distribution | Fig 2(b) Density | V2A advantage distribution is sharper; the flat distribution of DVDF indicates misestimated sample quality. |
| Hyperparameter $\lambda$ | Fig 3(a) | $\lambda=0.6$ is optimal; too small ignores dynamics alignment, too large ignores quality differences. |
| Selection Ratio $\xi$ | Fig 3(b) | $\xi \in [0.5, 0.75]$ performs well; performance drops if too small or too large. |

## Highlights & Insights
- **Precise Problem Definition**: Value misassignment is a genuine issue in multimodal heterogeneous data, formally characterized by Definition 4.6 in the paper.
- **Concise Framework Design**: Each of the three modules serves a specific purpose—EM for representation, modality-conditional advantage, and data filtering.
- **Strong Experimental Evidence**: Motivating experiments intuitively demonstrate the failure of DVDF; qualitative analysis (modality visualization, advantage distribution) verifies the internal logic.

## Limitations & Future Work
- Computational Cost: Iterative EM learning for modality representations increases training overhead.
- Theoretical Assumptions: Suboptimality bounds depend on "mild assumptions" that might not hold in highly non-stationary environments.
- Dynamics-centric Heterogeneity: The paper focuses on dynamics shifts but does not explore other source-specific differences like rewards or initial distributions.

## Related Work & Insights
- **vs. DVDF**: Both consider value alignment, but DVDF uses a global advantage function while V2A incorporates modality decomposition.
- **vs. IGDF/OTDF**: The former two only consider dynamics alignment, whereas V2A additionally accounts for quality.
- **Insights**: The modality decomposition approach can be generalized to other multi-source transfer tasks (e.g., multi-source domain adaptation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Heterogeneous data setting is a significant gap; the value misassignment problem is identified for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic task design plus comprehensive ablation and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with powerful motivating examples.
- Value: ⭐⭐⭐⭐⭐ Addresses practical challenges in offline RL; the framework is generalizable and can be integrated with multiple base algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](../../ICLR2026/reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICML 2026\] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning](latent_representation_alignment_for_offline_goal-conditioned_reinforcement_learn.md)
- [\[ICML 2026\] Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning](hista_and_numca_estimate_state_value_effectively_for_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](../../ICLR2026/reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)

</div>

<!-- RELATED:END -->
