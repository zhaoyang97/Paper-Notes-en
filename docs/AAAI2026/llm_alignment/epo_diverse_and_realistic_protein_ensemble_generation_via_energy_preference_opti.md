---
title: >-
  [Paper Note] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization
description: >-
  [AAAI 2026][LLM Alignment][Protein conformational ensemble] This paper proposes EPO (Energy Preference Optimization), which combines reverse SDE sampling with listwise energy-ranked preference optimization to align a pretrained protein generator with the target Boltzmann distribution using only energy signals. EPO achieves state-of-the-art performance across 9 metrics on three benchmarks (Tetrapeptides, ATLAS, and Fast-Folding), entirely eliminating the need for expensive molecular dynamics (MD) simulations.
tags:
  - AAAI 2026
  - LLM Alignment
  - Protein conformational ensemble
  - energy preference optimization
  - SDE sampling
  - Boltzmann distribution
  - MD-free generation
date: 2026-05-08
content_hash: f526b4269ed3ab27
---

# EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization

**Conference**: AAAI 2026
**arXiv**: [2511.10165](https://arxiv.org/abs/2511.10165)
**Code**: None
**Area**: Alignment & RLHF / Protein Generation
**Keywords**: Protein conformational ensemble, energy preference optimization, SDE sampling, Boltzmann distribution, MD-free generation

## TL;DR
This paper proposes EPO (Energy Preference Optimization), which combines reverse SDE sampling with listwise energy-ranked preference optimization to align a pretrained protein generator with the target Boltzmann distribution using only energy signals. EPO achieves state-of-the-art performance across 9 metrics on three benchmarks (Tetrapeptides, ATLAS, and Fast-Folding), entirely eliminating the need for expensive molecular dynamics (MD) simulations.

## Background & Motivation
**Background**: Protein function depends on conformational ensembles rather than single static structures, and understanding these ensembles is critical for drug design. Traditional approaches rely on computationally expensive MD simulations to generate conformations.

**Limitations of Prior Work**: (a) MD simulations are prohibitively costly, requiring days to weeks per protein; (b) pretrained generative models can produce conformations but do not follow the Boltzmann distribution; (c) pairwise DPO-based preference optimization tends to neglect high-energy yet functionally important metastable states.

**Key Challenge**: Generating diverse and physically realistic conformational ensembles is hampered by (a) the prohibitive cost of MD, (b) the thermodynamic inconsistency of generative models, and (c) the diversity-degrading effect of pairwise optimization.

**Goal**: Align a generative model to produce physically realistic and diverse protein conformational ensembles using only energy signals, without any MD trajectories.

**Key Insight**: (1) Converting ODE to SDE sampling introduces stochasticity to escape local minima; (2) listwise ranking replaces pairwise comparison to preserve ensemble diversity; (3) Jensen's inequality enables derivation of a tractable upper bound.

**Core Idea**: Listwise energy-ranked preference optimization combined with SDE stochasticity yields diverse and physically realistic conformational ensembles without MD simulations.

## Method

### Overall Architecture
Input: a pretrained ODE-based protein generator and an energy function. Output: a conformational ensemble generator aligned to the Boltzmann distribution.

### Key Designs

1. **ODE → SDE Conversion**:

    - Function: Transforms deterministic ODE sampling into stochastic SDE sampling.
    - Mechanism: $dx_t = v(x_t,t)dt + \frac{1}{2}w_t s(x_t,t)dt + \sqrt{w_t}d\bar{W}_t$, where $w_t$ controls the level of stochasticity.
    - Design Motivation: ODE sampling becomes trapped in local minima corresponding to single conformations; SDE stochasticity enables escape from energy barriers to explore multiple metastable states.

2. **Listwise Energy-Ranked Preference Optimization**:

    - Function: Replaces pairwise comparison with listwise ranking to treat different metastable states more equitably.
    - Mechanism: $\mathcal{L}_{LiPO} = -\mathbb{E}\sum_{k=1}^K \log\frac{\exp[s_\theta(\tau^{(k)})]}{\sum_{j=k}^K \exp[s_\theta(\tau^{(j)})]}$, based on the Plackett-Luce selection probability model.
    - Design Motivation: Pairwise DPO asymptotically ignores high-energy yet functionally important metastable states in multi-conformation settings; listwise optimization assigns equal gradient weight to all ranking positions.

3. **Tractable Upper Bound for Trajectory Likelihood**:

    - Function: Derives a practical upper bound for the intractable trajectory probability in continuous-time generative models.
    - Mechanism: Approximated via Jensen's inequality: $s_\theta(\tau^{(i)}) = \beta(\text{MSE}_t(y_0^{(i)}, y_1^{(i)}; \theta_{ref}) - \text{MSE}_t(y_0^{(i)}, y_1^{(i)}; \theta_{opt}))$.
    - Design Motivation: Exact trajectory probabilities are intractable in continuous generative models.

4. **Protein Structure Representation**:

    - SE(3)×R³ rotation-translation frames combined with 7 torsion angles ($\psi, \phi, \omega, \chi_1, \ldots, \chi_4$).
    - Per-residue token $\xi_t^j \in \mathbb{R}^{7+14}$.

### Loss & Training
Listwise LiPO loss with online iterative LoRA fine-tuning. Energies are computed via force fields without any MD simulations.

## Key Experimental Results

### Main Results

| Benchmark | Metric | EPO | MDGen (Baseline) | Note |
|-----------|--------|-----|-----------------|------|
| Tetrapeptides | Pairwise RMSD correlation | **SOTA** | 0.48 | Significant improvement |
| ATLAS | Global RMSF | **SOTA** | 0.50 | Global flexibility |
| Fast-Folding | Per-target RMSF | **SOTA** | 0.71 | Local flexibility |
| Overall | 9 metrics | **All SOTA** | — | Comprehensively leading |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| EPO-List (full) | **SOTA** | Listwise preference |
| EPO-Pair (pairwise) | Inferior | Neglects high-energy metastable states |
| ODE sampling | Single conformation | Trapped in local minima |
| SDE sampling | **Multiple conformations** | Escapes energy barriers |

### Key Findings
- **Energy signals alone can substitute MD trajectories**, substantially reducing the computational cost of conformational ensemble generation.
- **Listwise > Pairwise**: Pairwise DPO asymptotically neglects high-energy metastable states, whereas listwise optimization preserves diversity.
- **SDE stochasticity is essential**: Deterministic ODE sampling is confined to single conformations, while SDE effectively explores the energy landscape.
- EPO successfully captures key metastable states missed by the pretrained model (see Figure 3 visualizations).

## Highlights & Insights
- **Cross-domain transfer from RLHF to protein generation** — applying preference optimization from LLM alignment to molecular generation demonstrates the generality of the alignment paradigm.
- **Diversity preservation via listwise optimization** is broadly applicable to any scenario requiring ensemble diversity — not limited to proteins, but also relevant to molecular generation, materials design, and beyond.
- **The simple ODE → SDE conversion yields qualitative gains** — introducing controlled stochasticity into the energy landscape enables exploration of rare yet functionally important conformations.

## Limitations & Future Work
- Validation is limited to small proteins (Tetrapeptides and Fast-Folding); applicability to larger proteins remains unexplored.
- The accuracy of the energy function constitutes a hard ceiling — misspecified force fields lead to misaligned optimization targets.
- The stability of online LoRA updates may degrade over extended training.
- Future work could explore integrating EPO with structure prediction tools such as AlphaFold3.

## Related Work & Insights
- **vs. RLHF/DPO**: EPO transfers preference optimization from text alignment to molecular alignment, substituting energy for human preference. The key insight is that the mathematical framework of preference ranking naturally corresponds to energy ranking in physical systems.
- **vs. DistributionalGraphormer**: DistributionalGraphormer predicts equilibrium distributions but requires MD trajectories as training data; EPO directly aligns to the Boltzmann distribution using only energy functions.
- **vs. Traditional MD**: EPO entirely bypasses MD simulation — whereas traditional MD requires days to weeks to generate a conformational ensemble for a single protein, EPO generates ensembles in minutes.
- **vs. Boltzmann Generator**: Boltzmann Generator directly learns equilibrium distributions but is difficult to train; EPO achieves more stable alignment via preference optimization.
- **Insight**: The alignment paradigm is general — any generative task with a well-defined scoring function can benefit from preference optimization to improve generation quality. This principle extends from proteins to materials design, drug molecules, and even architectural structure optimization.
- **Methodological significance**: The diversity-preserving property of listwise preference optimization applies to all scenarios requiring ensemble diversity or sample diversity — not limited to protein generation.
- **Engineering insight from SDE stochasticity**: Introducing a small amount of controlled stochasticity into a deterministic model can substantially expand the exploration space — a broadly applicable cross-domain technique.
- **Computational cost comparison**: Traditional MD requires days to weeks to generate a conformational ensemble for one protein; EPO completes the same task in minutes, representing a 3–4 order-of-magnitude efficiency gain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First listwise preference optimization applied to protein ensemble generation, with theoretical contributions (upper bound derivation).
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, 9 metrics, ablations over pairwise vs. listwise and ODE vs. SDE.
- Writing Quality: ⭐⭐⭐⭐ Cross-domain motivation is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Direct impact on drug design and structural biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/llm_alignment/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](../../ICLR2026/llm_alignment/dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[AAAI 2026\] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)

</div>

<!-- RELATED:END -->
