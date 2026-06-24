---
title: >-
  [Paper Note] GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance
description: >-
  [AAAI 2026][Computational Biology][molecular optimization] This paper proposes GP-MoLFormer-Sim, a training-free test-time molecular generation guidance method that leverages the contextual embeddings of a chemical language model (GP-MoLFormer) to estimate similarity to target molecules, dynamically adjusting logits during autoregressive decoding. Combined with a genetic algorithm (GP-MoLFormer-Sim+GA), the method achieves an average rank of 2nd across 23 tasks on the PMO ben…
tags:
  - "AAAI 2026"
  - "Computational Biology"
  - "molecular optimization"
  - "chemical language model"
  - "test-time guidance"
  - "genetic algorithm"
  - "SMILES"
  - "similarity-guided generation"
date: 2026-05-08
content_hash: 65f4bd8932191a7c
---

# GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance

**Conference**: AAAI 2026
**arXiv**: [2506.05628](https://arxiv.org/abs/2506.05628)  
**Code**: No public link  
**Area**: AI for Science / Drug Discovery
**Keywords**: molecular optimization, chemical language model, test-time guidance, genetic algorithm, SMILES, similarity-guided generation

## TL;DR
This paper proposes GP-MoLFormer-Sim, a training-free test-time molecular generation guidance method that leverages the contextual embeddings of a chemical language model (GP-MoLFormer) to estimate similarity to target molecules, dynamically adjusting logits during autoregressive decoding. Combined with a genetic algorithm (GP-MoLFormer-Sim+GA), the method achieves an average rank of 2nd across 23 tasks on the PMO benchmark and outperforms MOLLEO—which relies on GPT-4—under a strict black-box oracle setting.

## Background & Motivation

**Background**: Molecular optimization is a central problem in drug discovery and materials design, requiring the search for molecules satisfying specific property constraints within an enormous chemical space. Existing approaches include reinforcement learning, deep generative models (VAEs, diffusion models), GFlowNets, and genetic algorithms. Recent work has demonstrated that classical genetic algorithms remain highly competitive in molecular optimization.

**Limitations of Prior Work**:
- Methods that integrate GA with deep learning typically require **retraining** the generative model for each specific optimization task.
- Methods such as MOLLEO rely on GPT-4 to generate candidate molecules; however, GPT-4 prompts contain oracle task information (e.g., target molecule names), which compromises black-box oracle fairness and incurs high API costs.
- Reinforcement learning methods require training reward models or policy networks.

**Key Challenge**: How can a pretrained chemical language model be leveraged for targeted molecular generation without retraining?

**Key Insight**: The method tilts decoding probabilities at test time by using the model's **own** embedding space to estimate the similarity between generated molecules and target molecules, requiring no additional training.

## Method

### Overall Architecture

**GP-MoLFormer-Sim** (guided generation module) + **Genetic Algorithm** (GA search loop) = **GP-MoLFormer-Sim+GA**

### GP-MoLFormer Backbone
- A GPT-style autoregressive decoder with linear attention and rotary positional encodings.
- Approximately 47M parameters, trained on ~650M canonicalized SMILES (ZINC + PubChem).
- Unconditional sampling enables broad exploration of chemical space.

### Core Algorithm: Similarity-Guided Generation (Algorithm 1)

At each autoregressive decoding step:

1. **Embed current candidates**: For each possible next token $i$ in the vocabulary, compute the GP-MoLFormer embedding $x_i$ of the concatenated sequence $s \oplus i$.
2. **Embed target molecules**: For each target molecule $m_j$, compute the embedding $y_j$ of the prefix $m_j[1:t]$.
3. **Compute cosine similarity**: $S_{ij} = \langle x_i, y_j \rangle$, then average over all targets to obtain $\bar{S}_i$.
4. **Mix logits**: $u \leftarrow \frac{1}{\tau}((1-\alpha)u + \alpha\bar{S})$

Here $\alpha \in [0,1]$ controls guidance strength (0 = unconditional generation, 1 = pure similarity sampling), and $\tau$ controls sampling temperature.

**Theoretical Justification**: This procedure is equivalent to solving an optimization problem—maximizing similarity to a KDE over the target molecules while constraining deviation from GP-MoLFormer's original distribution via KL divergence.

**Optional Enhancements**:
- Random Fourier Features (RFF): Replaces cosine similarity with a Gaussian kernel density estimate, adding local control.
- Computational efficiency: Guided generation is only ~4× slower per token than unconditional generation (0.049s vs. 0.013s).

### GP-MoLFormer-Sim+GA (Algorithm 2)

GA loop (see Figure 1):
1. **Select guidance molecules** (A): Sample top-$G$ high-scoring molecules from the current pool, plus diversity candidates.
2. **Guided generation** (B): Use GP-MoLFormer-Sim to generate neighbor candidates (mutations) for each guidance molecule.
3. **Pruning/augmentation** (C): Filter candidates below a Tanimoto similarity threshold; optionally apply graph-based crossover.
4. **Scoring**: Evaluate new candidates with the oracle and add to the molecular pool.
5. **Iterate** until the oracle budget is exhausted.

### Theoretical Interpretation

The guided probability distribution is the closed-form solution to the following optimization problem:
$$\max_{p \in \Delta_V} (1-\alpha) \sum_i p_i \pi_{KDE}^{target}(x_i|c) - KL(p \| \pi_{ref}^\alpha(\cdot|c))$$

This formulation maximizes similarity to the target molecule KDE while remaining close to the original GP-MoLFormer distribution.

## Key Experimental Results

### Experiment 1: Similarity-Guided Molecular Generation

Goal: Generate molecules with high Tanimoto similarity to 5 trypsin inhibitor target molecules.

| Method | top-1 Tsim | top-10 Tsim | top-100 Tsim | top-1000 Tsim |
|--------|-----------|------------|-------------|--------------|
| **GPMFS (Ours)** | **1.000** | **0.972** | **0.877** | **0.763** |
| S Model (RL-tuned) | 0.694 | 0.618 | 0.554 | 0.499 |
| Random Gen. | 0.438 | 0.391 | 0.348 | 0.290 |
| Random Search | 0.477 | 0.450 | 0.417 | 0.377 |

- GP-MoLFormer-Sim significantly outperforms all baselines at every $k$ value.
- top-1 achieves a perfect match (Tsim = 1.0); among the top-10,000 molecules, 132 have QED > 0.7.

### Experiment 2: PMO Benchmark (23 Molecular Optimization Tasks)

| Method | Avg. Rank | Avg. AUC | Requires Training? | Requires LLM Calls? |
|--------|---------|---------|-----------|-------------|
| MOLLEO (GPT-4) | **1** | **0.777** | No | Yes ($$$) |
| **GPMFS+GA** | **2** | **0.662** | **No** | **No** |
| Mol-GA | 3 | 0.639 | No | No |
| Graph-GA | 4 | 0.597 | No | No |
| STONED SELFIES | 5 | 0.566 | No | No |
| SynNet | 6 | 0.499 | No | No |

- GPMFS+GA **surpasses all baselines including MOLLEO** on 3 tasks (GSK3, JNK3, ranolazine_mpo).
- Ranks 2nd on an additional 9 tasks.

### Experiment 3: Black-Box Oracle Fairness Analysis (Core Contribution)

| Task | GPMFS+GA | MOLLEO (name withheld) | MOLLEO (name included) |
|------|----------|-------------------|-----------------|
| thiothixene rediscovery | **0.504** | 0.462 | 0.692 |
| mestranol similarity | **0.658** | 0.644 | 0.983 |

- When the target molecule name is withheld, MOLLEO's performance **drops ~33%**, falling below GPMFS+GA.
- This demonstrates that MOLLEO's high performance is partly attributable to GPT-4's memorization of molecular knowledge (e.g., GPT-4 knows the SMILES of thiothixene).
- Reverse experiment: providing GPMFS+GA with target SMILES raises its average rank from 2.7 to 1.7.

### Ablation Study

| Configuration | Avg. Rank | Avg. AUC |
|------|---------|---------|
| Guided Generation only (GG) | 5.0 | 0.603 |
| +Crossover (XO) | 3.4 | 0.672 |
| +RFF768 | 5.3 | 0.597 |
| +XO+DIV | 3.0 | 0.678 |
| +RFF768+XO | 2.5 | 0.682 |
| **+RFF768+XO+DIV** | **1.8** | **0.690** |

- Crossover (XO) contributes the most; diversity enhancement (DIV) provides additional gains.
- RFF alone is ineffective but yields improvement when combined with XO.

## Highlights & Insights

1. **Elegant test-time guidance design**: No reward model training, no policy gradients, no external LLM calls—only the model's own embedding space serves as the similarity signal.
2. **Theoretically grounded**: The algorithm is justified as the closed-form solution to a well-defined optimization problem, not merely a heuristic.
3. **Black-box fairness exposes MOLLEO's implicit advantage**: The analysis demonstrates that MOLLEO exploits GPT-4's memorization of known molecules, a finding with significant implications for benchmark evaluation standards in the field.
4. **Cost efficiency**: A 47M-parameter model outperforms GPT-4 API calls under a fair comparison.
5. **Domain agnosticism**: The framework is theoretically applicable to any autoregressive language model and any sequence-based optimization task.

## Limitations & Future Work

1. **Only one CLM tested**: Although the method claims model-agnostic applicability, it is validated solely with GP-MoLFormer.
2. **Single-objective optimization only**: The method has not been extended to multi-objective optimization (e.g., simultaneously optimizing binding affinity, solubility, and drug-likeness).
3. **Negative guidance unexplored**: Low-scoring molecules are not exploited as negative guidance to actively steer generation away from undesirable regions.
4. **Vocabulary-level iterative computation**: Each decoding step requires embedding all tokens in the vocabulary (2,362 tokens), incurring increasing overhead for longer sequences.
5. **Synthetic accessibility insufficiently discussed**: Whether generated molecules are practically synthesizable is not thoroughly analyzed (SA scores are reported only in the appendix).

## Related Work & Insights

- **Molecular optimization methods**: RL-based (REINVENT), VAE (JT-VAE), Bayesian Optimization, GFlowNets, diffusion models.
- **GA-based methods**: Graph-GA, STONED, Mol-GA, GEAM, MOLLEO (GPT-4-assisted GA).
- **Test-time guided LLM decoding**: Reward-guided Decoding, SASA (self-constrained sampling), Conditional Activation Steering.
- **Chemical language models**: GP-MoLFormer, MoLFormer, BioT5, MoleculeSTM.

## Rating ⭐⭐⭐⭐

The method is elegant and theoretically justified; the black-box fairness analysis constitutes an important contribution, and the approach achieves competitive molecular optimization without any training. However, it ranks only 2nd on the full PMO leaderboard, and validation across multiple objectives and multiple CLMs remains absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Adaptation without Source Data for Out-of-Domain Bioactivity Prediction](../../ICLR2026/computational_biology/test-time_adaptation_without_source_data_for_out-of-domain_bioactivity_predictio.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)
- [\[ICLR 2026\] Learning Collective Variables from BioEmu with Time-Lagged Generation](../../ICLR2026/computational_biology/learning_collective_variables_from_bioemu_with_time-lagged_generation.md)

</div>

<!-- RELATED:END -->
