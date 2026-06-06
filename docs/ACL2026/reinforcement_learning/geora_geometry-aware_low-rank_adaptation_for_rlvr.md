---
title: >-
  [Paper Note] GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR
description: >-
  [ACL 2026][Reinforcement Learning][Low-Rank Adaptation] This paper proposes GeoRA, a low-rank adaptation method specifically designed for Reinforcement Learning with Verifiable Rewards (RLVR). By constructing a geometric…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Low-Rank Adaptation"
  - "RLVR"
  - "Geometry-Aware"
  - "SVD Initialization"
  - "PEFT"
date: 2026-05-08
content_hash: e3125a6e75ddf702
---

# GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR

**Conference**: ACL 2026  
**arXiv**: [2601.09361](https://arxiv.org/abs/2601.09361)  
**Code**: None  
**Area**: Parameter-Efficient Fine-Tuning / Reinforcement Learning with Verifiable Rewards  
**Keywords**: Low-Rank Adaptation, RLVR, Geometry-Aware, SVD Initialization, PEFT

## TL;DR

This paper proposes GeoRA, a low-rank adaptation method specifically designed for Reinforcement Learning with Verifiable Rewards (RLVR). By constructing a geometrically constrained matrix (fusing spectral and Euclidean priors) to extract the principal directions of the RL update subspace for SVD initialization, and freezing the residual matrix as a structural anchor, GeoRA consistently outperforms baselines like LoRA, PiSSA, and MiLoRA on 1.5B-32B Qwen/Llama models across mathematical, medical, and code RLVR tasks. It demonstrates stronger out-of-distribution (OOD) generalization and reduced catastrophic forgetting.

## Background & Motivation

**Background**: RLVR has emerged as a core paradigm for enhancing the reasoning capabilities of large language models (e.g., OpenAI-o1, DeepSeek-R1). Unlike SFT, RLVR is essentially a constrained optimization process that amplifies latent reasoning behaviors through reward-induced sampling bias rather than injecting new knowledge. Consequently, RLVR is highly sensitive to update stability and the preservation of pre-trained representation geometry.

**Limitations of Prior Work**: (1) **Geometric mismatch between SFT-oriented low-rank methods and RLVR**: PiSSA allocates trainable parameters to the principal components of the weight matrix. While effective in SFT, this conflicts with the preferred update subspace of RLVR, which tends toward low-energy directions (orthogonal to pre-trained principal features). PiSSA's forced updates on principal directions lead to instability. (2) **Efficiency bottlenecks of sparse fine-tuning**: Although sparse methods (e.g., SparseFT) better align with RLVR update patterns, modern hardware support for unstructured sparsity is limited. Theoretically high parameter efficiency does not translate into actual speed gains, often introducing extra overhead (10.8% slower than FullFT).

**Key Challenge**: The effective update subspace of RLVR is anisotropic and compressible (concentrated in a few principal directions), but these do not align with the principal components of pre-trained weights. Existing low-rank methods either target the wrong subspace (PiSSA) or are computationally inefficient despite correct alignment (SparseFT).

**Goal**: Design a PEFT method that satisfies three conditions: (1) alignment with RLVR-specific update geometry, (2) maintenance of hardware efficiency through dense matrix computation, and (3) prevention of pre-trained representation disruption via structural anchors.

**Key Insight**: Analysis of actual RLVR update patterns reveals that the effective update subspace, while sparse, possesses a compressible low-rank structure. This subspace can be extracted via a geometric constraint mask and compressed into low-rank adapter initializations using SVD.

**Core Idea**: Instead of performing low-rank decomposition on the original weight $W$ (as in LoRA/PiSSA), SVD is performed on a geometrically constrained view $W_{Geo} = W \odot (M_{Spec} \cup M_{Euc})$. This view retains only parameters with low curvature (spectral prior) and high plasticity (Euclidean prior), which correspond to the preferred update regions for RLVR.

## Method

### Overall Architecture

GeoRA proceeds in two steps: (1) **Offline Preprocessing**: Construct the geometric constraint matrix $W_{Geo}$, perform SVD to extract the top-$r$ components for initializing adapters $A_{Geo}$ and $B_{Geo}$, and calculate the frozen residual matrix $W_{res}$. (2) **Online Training**: During forward passes, $h = W_{res} x + \frac{\alpha}{r} B_{Geo} A_{Geo} x$, where $W_{res}$ is frozen and only $A_{Geo}$ and $B_{Geo}$ are trained. The initialization ensures function preservation: $W_{res} + \frac{\alpha}{r} B_{Geo} A_{Geo} = W$.

### Key Designs

1.  **Geometric Prior Construction**:
    - **Function**: Extracts parameter subspaces from pre-trained weights suitable for RLVR updates.
    - **Mechanism**: Combines two complementary geometric priors. **Spectral Prior** $M_{Spec}$: Selects the $\rho$ quantile of parameters with the smallest absolute values in the rank-$r$ approximation $\hat{W}_r$, i.e., $(M_{Spec})_{i,j} = \mathbb{I}(|(\hat{W}_r)_{i,j}| \leq \tau_{Spec}(\rho))$, suppressing high-energy/high-curvature components to ensure spectral stability. **Euclidean Prior** $M_{Euc}$: Selects the $\rho$ quantile of the smallest absolute values in the original weights $(M_{Euc})_{i,j} = \mathbb{I}(|W_{i,j}| \leq \tau_{Euc}(\rho))$, capturing high-plasticity, near-zero parameters. Their union is taken: $W_{Geo} = W \odot (M_{Spec} \cup M_{Euc})$.
    - **Design Motivation**: Experiments show the intersection of these masks is only 4.55% (Jaccard 0.128), indicating they capture highly complementary parameter subsets. The spectral prior ensures stability of principal components, while the Euclidean prior maintains adaptation flexibility. Together, they define a stable yet expressive manifold for RLVR updates.

2.  **Geometry-Aware SVD Initialization**:
    - **Function**: Compresses the geometrically constrained subspace into efficient low-rank adapters.
    - **Mechanism**: Perform SVD on $W_{Geo}$: $W_{Geo} = U_{Geo} \Sigma_{Geo} V_{Geo}^\top$. Use the top-$r$ components to initialize the adapters: $A_{Geo} = \Sigma_{Geo[:r,:r]}^{1/2} V_{Geo[:,:r]}^\top$ and $B_{Geo} = U_{Geo[:,:r]} \Sigma_{Geo[:r,:r]}^{1/2}$, such that the initial $B_{Geo} A_{Geo}$ is the optimal rank-$r$ approximation of $W_{Geo}$. The residual matrix $W_{res} = W - \frac{\alpha}{r} B_{Geo} A_{Geo}$ is frozen during training.
    - **Design Motivation**: A critical difference from PiSSA (which takes principal components of the original $W$) is that GeoRA takes principal components of the constrained $W_{Geo}$, ensuring trainable directions align with RLVR update subspaces rather than pre-trained knowledge encoding directions.

3.  **Frozen Residual Matrix (Structural Anchor)**:
    - **Function**: Prevents the erosion of pre-trained principal components during training.
    - **Mechanism**: $W_{res}$ contains the original weight minus the geometric constraint subspace, preserving the core knowledge encoding of the pre-trained model. Since $W_{res}$ is frozen, the optimizer can only move along the geometrically aligned manifold parameterized by $A_{Geo}$ and $B_{Geo}$.
    - **Design Motivation**: Aggressive updates in RLVR can lead to behavioral collapse or capacity degradation (the "Reasoning Boundary Paradox"). Freezing the residual matrix provides a hard structural constraint equivalent to strategic updates within a geometrically aligned trust region.

### Loss & Training

RLVR training is conducted using the GRPO algorithm. A fixed rank $r=16$ and sparsity rate $\rho=0.2$ are used. Main experiments are trained on the DeepMath-103K dataset. SVD initialization is a one-time preprocessing overhead, negligible compared to RLVR training time.

## Key Experimental Results

### Main Results — Mathematical RLVR (Qwen3-8B)

| Method | AIME24 | AIME25 | MATH500 | OlymMATH | HumanEval(OOD) | MMLU(OOD) | IFEval(OOD) |
|------|--------|--------|---------|----------|---------------|-----------|-------------|
| Base | 13.33 | 11.67 | 71.20 | 9.75 | 76.83 | 71.94 | 54.32 |
| FullFT | 23.33 | 22.08 | 78.40 | 11.25 | 76.83 | 71.94 | 50.45 |
| LoRA | 19.58 | 19.58 | 75.60 | 10.75 | 81.10 | 75.65 | 52.13 |
| PiSSA | 22.50 | 20.42 | 74.40 | 11.75 | 71.95 | 73.89 | 48.74 |
| MiLoRA | 20.42 | 19.58 | 76.20 | 11.50 | 78.66 | 74.51 | 51.85 |
| **GeoRA** | **23.75** | **21.67** | **78.00** | **12.75** | **82.93** | **75.96** | **53.73** |

### Ablation Study (Qwen3-4B)

| Configuration | Reward | AIME24 | AIME25 | MATH500 | OlymMATH | Avg |
|------|--------|--------|--------|---------|----------|-----|
| GeoRA (Full) | 0.88 | 13.33 | 9.17 | 73.40 | 5.75 | 25.41 |
| Random-r Init | 0.85 | 12.50 | 8.50 | 72.10 | 5.25 | 24.60 |
| Tail-r Init | 0.82 | 11.67 | 7.50 | 70.80 | 4.50 | 23.40 |
| w/o $M_{Spec}$ | 0.86 | 12.50 | 8.33 | 72.00 | 4.75 | 24.40 |
| w/o $M_{Euc}$ | 0.83 | 13.33 | 8.75 | 72.80 | 5.50 | 25.10 |

### Key Findings

- GeoRA matches or exceeds FullFT on ID tasks while leading comprehensively on OOD tasks—HumanEval 82.93 (FullFT 76.83), MMLU 75.96 (FullFT 71.94). This indicates that geometrically aligned updates reduce interference with pre-trained capabilities.
- PiSSA performs worst on OOD tasks (IFEval 48.74), confirming that SFT-oriented principal component initialization is detrimental to RLVR.
- Spectral analysis confirms GeoRA's updates barely touch the principal component subspace ($\mathcal{S}_{Head} \leq 0.02$), unlike PiSSA which overlaps significantly ($\approx 0.98$).
- Significant efficiency advantages: with only 0.04B trainable parameters (0.5% of FullFT), training is 19.9% faster than FullFT and saves 28.5% VRAM.
- Strong hyperparameter robustness: GeoRA maintains high rewards across a wide range of learning rates, whereas PiSSA/MiLoRA performance drops sharply at high learning rates.
- Equally effective in medicine and code RLVR: GeoRA achieves 76.12 on MedQA (LoRA 74.23) and 81.60 on MBPP (LoRA 81.00).

## Highlights & Insights

- **Deep Core Insight**: The effective update subspace of RLVR is not isotropic random noise but possesses a compressible heavy-tailed spectral structure. This provides a theoretical foundation for applying low-rank methods to RLVR, contingent on identifying the correct subspace.
- **Complementarity of Geometric Priors**: The mere 4.55% parameter overlap (Jaccard 0.128) demonstrates that spectral stability and parameter plasticity indeed capture distinct informational dimensions.
- **Structural Anchor Paradigms**: The frozen residual matrix shifts the "additive residual" paradigm of LoRA to a "structural anchor" paradigm—not only maintaining initialization invariance but also enforcing optimization trajectories, which is critical for policy stability in RLVR.

## Limitations & Future Work

- SVD initialization, though a one-time cost, adds a preprocessing step that may be inconvenient for rapid iteration scenarios.
- Experiments focused primarily on reasoning-based RLVR tasks (math, medicine, code); efficacy in open-ended RL scenarios (e.g., dialogue preference optimization) remains unverified.
- Choice of sparsity rate $\rho=0.2$ and rank $r=16$ did not involve extensive search; optimal configurations may exist.
- Geometric prior construction relies on statistical properties of pre-trained weights; whether these hold after extensive post-training remains to be validated.
- Comparison with more LoRA variants (e.g., DoRA, AdaLoRA) was not conducted.

## Related Work & Insights

- **vs PiSSA**: PiSSA initializes adapters on pre-trained principal components, which suits SFT but harms RLVR. Its NSS is 0.395 (high structural disruption) and $\mathcal{S}_{Head} \approx 0.98$ (updates on principal components). GeoRA's NSS is only 0.092 and $\mathcal{S}_{Head} \leq 0.02$, precisely targeting the tail subspace.
- **vs MiLoRA**: MiLoRA selects minor components for initialization, moving closer to RLVR directions but without explicit geometric priors. GeoRA's dual-mask manifold definition consistently outperforms MiLoRA across all benchmarks.
- **vs SparseFT**: SparseFT aligns with RLVR update patterns but suffers from poor computational efficiency (10.8% slower than FullFT). GeoRA compresses the sparse subspace into dense low-rank operations, becoming 19.9% faster than FullFT.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First geometry-aware low-rank adaptation method specifically designed for RLVR, with tight integration of theory and design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-scale models (1.5B-32B) × three domains (math/med/code) × comprehensive ablation and mechanistic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and convincing spectral analysis, though heavy notation poses a slight entry barrier.
- **Value**: ⭐⭐⭐⭐⭐ Provides a new paradigm for parameter-efficient training in the RLVR era; geometry-aware concepts are generalizable to other RL fine-tuning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](../../ICLR2026/reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[NeurIPS 2025\] Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/shift_before_you_learn_enabling_low-rank_representations_in_reinforcement_learni.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)

</div>

<!-- RELATED:END -->
