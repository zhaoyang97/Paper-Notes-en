---
title: >-
  [Paper Note] How to Make the Most of Your Masked Language Model for Protein Engineering
description: >-
  [ICLR 2026][Medical Imaging][Protein Language Model] This work proposes a temperature-annealed stochastic beam search (SBS) sampling method for masked language models (MLMs)…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Protein Language Model"
  - "Masked Language Model"
  - "Stochastic Beam Search"
  - "Antibody Optimization"
  - "Multi-Objective Optimization"
date: 2026-05-08
content_hash: 486f9449a052610a
---

# How to Make the Most of Your Masked Language Model for Protein Engineering

**Conference**: ICLR 2026
**arXiv**: [2603.10302](https://arxiv.org/abs/2603.10302)
**Code**: None
**Area**: Protein Engineering / Antibody Design
**Keywords**: Protein Language Model, Masked Language Model, Stochastic Beam Search, Antibody Optimization, Multi-Objective Optimization

## TL;DR
This work proposes a temperature-annealed stochastic beam search (SBS) sampling method for masked language models (MLMs), leveraging a wild-type marginal approximation of pseudo-log-likelihood (PLL) for efficient full-sequence evaluation. In vitro experiments on real therapeutic antibody optimization demonstrate that the choice of sampling algorithm is at least as important as model selection; SBS with guidance achieves a 100% success rate.

## Background & Motivation

**Background**: A large number of protein language models (e.g., ESM-2, AbLang2) have been released and are widely used for iterative optimization of therapeutic antibodies—given a seed sequence, a model generates mutant candidates, which are experimentally tested before the next round. However, research on how to efficiently sample high-quality sequences from MLMs remains extremely limited.

**Limitations of Prior Work**: (1) The dominant MLM sampling paradigm is "mutation-centric" (e.g., denoising sampling, Gibbs sampling), masking and filling positions one at a time, with a computational cost of $O(EL^3)$, and empirically tending to produce low-quality sequences; (2) existing methods struggle to integrate non-differentiable scoring functions (e.g., OASis immunogenicity scores, isoelectric point); (3) systematic evaluation of MLM sampling algorithms—particularly with in vitro validation—is nearly nonexistent.

**Key Challenge**: The mutation-centric sampling paradigm restricts decisions to individual positions, preventing the use of full-sequence information for global assessment, while the $O(EL^3)$ computational complexity limits candidate diversity.

**Goal**: How can high-likelihood, diverse, and seed-proximal mutant sequences be efficiently sampled from an MLM? How can multiple scoring functions (differentiable or not) be flexibly integrated for multi-objective optimization? What combination of model and sampling strategy is most effective for practical antibody optimization?

**Key Insight**: Shifting MLM sampling from "mutation-centric" to "sequence-centric"—rather than having the model generate mutations position by position, the model evaluates the PLL of complete sequences, transforming the problem into a search problem. A key observation is that once the PLL of a sequence is computed, the PLLs of all its single-site neighbors are obtained almost for free.

**Core Idea**: By exploiting a wild-type marginal approximation, the approximate PLL of the entire 1-edit neighborhood is obtained at negligible additional cost from a single PLL computation, converting MLM sampling into an efficient stochastic beam search.

## Method

### Overall Architecture
Seed sequence → Compute PLL and approximate PLLs for all single-mutation neighbors → Stochastic beam search (SBS) over $E$ expansion steps → Temperature parameter balances likelihood and diversity → Optional: multi-objective optimization with scoring functions (affinity model, OASis, isoelectric point, etc.) via NDS or STS weighting → Output candidate sequence set → In vitro testing.

### Key Designs

1. **PLL-Based Efficient Full-Sequence Evaluation**:

    - Function: Obtain approximate PLLs for $20BEL$ candidate sequences at a cost of $O(BEL^3)$.
    - Mechanism: Computing the full PLL of a template sequence $\mathbf{x}$ requires $L$ forward passes to obtain exact conditional probabilities $\hat{p}(x_i|x_{j \neq i})$. For a sequence $\mathbf{x}'$ differing from the template at position $k$, the PLL is approximated as: $PLL(\mathbf{x}') \approx \sum_i \log(\text{softmax}_\tau(\tilde{Y}^{(i)}_{i,\text{index}(x'_i)}))$, where the exact conditional probability is used at position $k$ and the template's conditional probabilities are reused at all other positions (wild-type marginal approximation). A single template PLL computation thus covers all $20L$ single-mutation neighbors.
    - Design Motivation: Mutation-centric methods require $O(EL^3)$ per generated sequence, whereas the proposed method reduces the per-step expansion cost in beam search to $O(BL^3)$ (where $B$ is the beam width), achieving $20BEL$ candidates in $E$ steps at a total cost of $O(BEL^3)$—an exponential efficiency gain.

2. **Temperature-Annealed Stochastic Beam Search (SBS)**:

    - Function: Balance sequence likelihood and candidate diversity throughout the search.
    - Mechanism: The Gumbel-top-$k$ trick is employed: Gumbel noise is added to the PLL of each candidate sequence before ranking to select beam members. The softmax temperature $\tau$ scales the likelihood term but not the Gumbel term, so $\tau$ controls the trade-off between determinism (high likelihood) and stochasticity (diversity). Searching $E$ steps from the seed naturally enforces proximity constraints.
    - Design Motivation: Pure greedy selection (argmax) sacrifices diversity, while purely random sampling yields low-quality sequences. SBS achieves sampling without replacement via Gumbel noise while maintaining within-batch diversity.

3. **Flexible Multi-Objective Optimization (MOO) Guidance**:

    - Function: Seamlessly integrate arbitrary scoring functions (differentiable or not) for joint multi-objective optimization.
    - Mechanism: Because full-sequence evaluation is used, the MLM and all scoring functions are treated as black boxes—only requiring a complete sequence as input and returning a score. Two multi-objective scalarization strategies are supported: Pareto non-dominated sorting (NDS) for trade-offs among objectives, and smooth Tchebycheff scalarization (STS) for simultaneously maximizing all objectives. STS allows differential weighting of objectives.
    - Design Motivation: Mutation-centric methods must evaluate scoring functions on partially masked sequences, which many practical scoring functions (OASis, isoelectric point) cannot handle. The sequence-centric approach entirely eliminates this limitation.

### Loss & Training
The MLMs are pretrained models; no training is involved. Nine MLMs (ESM-2 35M/150M/650M, Sapiens, AbLang2, AMPLIFY 120M/350M, DiffAbOpt, in-house SAbDabMLM) and three CLMs (pIgGen, pIgGen-dev, CloneLM) are evaluated. In vitro experiments use a single FAb antibody seed, with each method generating ≥21 samples.

## Key Experimental Results

### Main Results (In Vitro)

| Method | Success Rate↑ | Notes |
|--------|--------------|-------|
| AbLang2 + Beam Search | ~65% | Among the best unsupervised methods |
| ESM2-650M + Beam Search | ~60% | Strong performance from a non-antibody-specific model |
| AbLang2 + Gibbs | ~40% | Same model; beam search substantially outperforms Gibbs |
| Sapiens + Gibbs | ~25% | Weak model + weak sampling |
| AbLang2 + Supervised Ranking | ~75% | Classifier trained on 729 samples used for ranking |
| AbLang2 + STS Guidance | **100%** | Multi-objective guided generation + ranking |
| AbLang2 + NDS Guidance | ~90% | Pareto ranking also yields significant improvement |

### Ablation Study

| Configuration | Key Observation | Notes |
|--------------|----------------|-------|
| Beam vs. Gibbs (same model) | Beam outperforms Gibbs across all 3 models | Sampling algorithm choice matters more than model choice |
| ESM2-650M (general-purpose protein model) | Comparable to antibody-specific AbLang2 | General-purpose models unexpectedly competitive for antibodies |
| Gibbs-argmax vs. Gibbs | Argmax yields higher success rate but lower diversity | Gibbs tends to produce low-quality sequences |
| With/without supervised guidance | Guidance eliminates weak binders | But guidance reduces humanness scores |

### Key Findings
- The choice of sampling algorithm is at least as important as model selection—switching samplers on the same model can result in a twofold difference in success rate.
- ESM2-650M, despite being trained on general protein data (not antibody-specific), performs competitively in antibody optimization, suggesting that capturing the general protein distribution is more important than antibody-specific specialization.
- Gibbs sampling tends to produce sequences that do not faithfully reflect the model's preferences—a systematic discrepancy exists between what the model "prefers" and what Gibbs "delivers."
- Supervised guidance eliminates the generation of weakly binding antibodies but introduces a reduction in humanness, necessitating the inclusion of humanness as an objective in multi-objective optimization.

## Highlights & Insights
- The key insight is the paradigm shift from "mutation-centric" to "sequence-centric" MLM sampling: exploiting the wild-type marginal approximation yields an exponential efficiency gain, making beam search feasible. This perspective reveals a fundamental inefficiency in prior MLM sampling methods—they discard a large amount of already-computed information.
- Rare in vitro validation: beyond in silico evaluation, a head-to-head comparison of 289 samples was conducted in a real therapeutic antibody project, with results directly applicable to industrial practice.

## Limitations & Future Work
- In vitro experiments are based on a single FAb antibody seed; generalizability requires validation across additional therapeutic programs.
- The accuracy of the wild-type marginal approximation for multi-site mutations decreases with edit distance; refreshing the template cache at each beam search expansion step mitigates this issue, but accumulated approximation error warrants further investigation.
- STS guidance achieves a 100% success rate at the cost of reduced humanness; trade-off management in multi-objective optimization requires further study.

## Related Work & Insights
- **vs. ESM-3 (Hayes et al., 2025)**: ESM-3 employs denoising sampling with derivative-free guidance for protein optimization, but requires scoring functions to accept partially masked inputs. The proposed method uses full-sequence evaluation and is compatible with any scoring function.
- **vs. DiffAbOpt/DiffAb+ (Raghu et al., 2025)**: Structure-based diffusion methods operate only on CDR regions. The proposed method can sample across arbitrary sequence positions, and DiffAb+ performs only modestly in the in vitro experiments.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The paradigm shift to sequence-centric sampling with PLL approximation is concise and compelling, though the underlying technical principles are relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rare large-scale in vitro validation; systematic comparison of 9 MLMs and 3 CLMs; comprehensive evaluation of supervised, unsupervised, and guided approaches.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; experimental findings are well-presented and thoroughly discussed.
- **Value**: ⭐⭐⭐⭐⭐ Directly actionable for antibody engineering practice; the finding that "sampling algorithm choice matters more than model choice" is a highly impactful and counterintuitive conclusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)
- [\[ICLR 2026\] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)

</div>

<!-- RELATED:END -->
