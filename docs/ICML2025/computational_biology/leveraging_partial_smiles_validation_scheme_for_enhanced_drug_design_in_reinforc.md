---
title: >-
  [Paper Note] Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks
description: >-
  [ICML2025][Computational Biology][SMILES molecule generation] The PSV-PPO algorithm is proposed, introducing a Partial SMILES Validation (PSV) truth table at each step of autoregressive SMILES molecule generation to penalize invalid tokens in real-time, enhancing chemical space exploration capability while maintaining molecular validity.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "SMILES molecule generation"
  - "Reinforcement learning"
  - "PPO"
  - "Partial validation"
  - "Drug discovery"
  - "Catastrophic forgetting"
date: 2026-05-08
content_hash: 311187bc001d2c8d
---

# Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks

**Conference**: ICML2025  
**arXiv**: [2505.00530](https://arxiv.org/abs/2505.00530)  
**Code**: To be confirmed  
**Area**: Computational Biology  
**Keywords**: SMILES molecule generation, Reinforcement learning, PPO, Partial validation, Drug discovery, Catastrophic forgetting

## TL;DR

The PSV-PPO algorithm is proposed, introducing a Partial SMILES Validation (PSV) truth table at each step of autoregressive SMILES molecule generation to penalize invalid tokens in real-time, enhancing chemical space exploration capability while maintaining molecular validity.

## Background & Motivation

SMILES-based molecule generation has become a mainstream method in drug discovery. Although Large Language Models (LLMs) combined with Reinforcement Learning (RL) can fine-tune generative models to optimize target molecular properties, they face a key challenge—**catastrophic forgetting**: while the molecular validity rate can exceed 99% during the pre-training phase, it drops drastically after RL fine-tuning.

Limitations of Prior Work:

- **REINVENT**: Uses a prior model as an anchor to retain pre-training knowledge, which restricts exploration capability and leads to insufficient diversity.
- **SELFIES / Grammar-VAE**: Enforce validity constraints at the representation level, but studies show SMILES-based methods typically outperform them in molecular property optimization and diversity.
- **Standard PPO**: The entropy-driven exploration mechanism performs unstably in SMILES generation—excessively high entropy leads to gradient explosion, while excessively low entropy causes mode collapse.
- **Sparse reward problem**: Molecular validity and property scores can only be evaluated after the complete SMILES string is generated, leaving the model without intermediate feedback.

## Method

### Mechanism: PSV Truth Table

The core innovation of PSV-PPO is the **Partial SMILES Validation (PSV) truth table**. At each step of autoregressive generation, the PSV table systematically evaluates all candidate tokens: if a token would make the current partial SMILES string invalid, it is immediately flagged and penalized.

The PSV table performs three checks:
1. **Syntactic compliance**: Ensuring the SMILES string adheres to syntax specifications.
2. **Aromaticity handling**: Checking whether aromatic systems can be correctly Kekulized.
3. **Valency verification**: Ensuring the valency of each atom is within a chemically reasonable range.

### PSV-PPO Loss Function

Based on standard PPO, PSV-PPO introduces four new loss terms, with the total loss being a weighted combination of six terms:

$$Loss = L^{\text{CLIP}}(\theta) + L^{\text{Value}}(\theta) + L^{\text{ENTROPY}}_{PSV}(\theta) + L^{\text{HD}}_{PSV}(\theta) + L^{\text{TPC}}_{PSV}(\theta) + L^{\text{GPS}}_{PSV}(\theta)$$

#### 1. PSV-driven Entropy Loss

Entropy is calculated only over the set of valid tokens $D_{PSV}$ verified by the PSV table, and normalized by $\log(\text{len}(D_{PSV}))$ to prevent the model from biasing toward actions with larger valid token sets (such as non-aromatic carbon "C"):

$$L^{\text{ENTROPY}}_{PSV}(\theta) = -\beta \mathbb{E}_t \left[ \sum_{a \in D_{PSV}} \frac{\pi_\theta(a|s_t) \log \pi_\theta(a|s_t)}{\log(\text{len}(D_{PSV}))} \right]$$

#### 2. PSV-driven Hellinger Distance Loss

Since the traditional KL divergence cannot handle zero-probability cases that arise after PSV filtering, it is replaced by the Hellinger distance to measure the gap between the current policy and the prior policy after PSV filtering:

$$L^{\text{HD}}_{PSV}(\theta) = \mathbb{E}_t \left[ \text{HD} \left[ \pi_{\theta_{\text{old\_PSV}}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t) \right] \right]$$

#### 3. TPC Loss (Token Probability Control Loss)

Dynamically penalizes tokens with excessively high probabilities to prevent mode collapse.

#### 4. GPS Loss (Global Probability Stabilization Loss)

An additional regularization term that maintains generation diversity when the model fails to discover molecules with higher scores.

### Training Process

1. The prior model generates molecular structures and their probability distributions.
2. Calculate rewards and the PSV truth table in parallel (minimizing computational overhead).
3. Store the scored molecules into the experience replay buffer.
4. Sample from the replay buffer, and the current model regenerates probability distributions.
5. Calculate the six loss terms and perform backpropagation to update parameters.

## Key Experimental Results

### GuacaMol Benchmark

| Task | SMILES GA | SMILES LSTM | Reinvent | MolRL-MGPT | **PSV-PPO** |
|------|-----------|-------------|----------|------------|-------------|
| C11H24 | 0.829 | 0.993 | 0.999 | 1.000 | **1.000** |
| C9H10N2O2P2Cl | 0.889 | 0.879 | 0.877 | 0.939 | **1.000** |
| Osimertinib MPO | 0.886 | 0.907 | 0.889 | 0.977 | 0.951 |
| Fexofenadine MPO | 0.931 | 0.959 | 1.000 | 1.000 | **1.000** |
| Perindopril MPO | 0.661 | 0.808 | 0.764 | 0.810 | **0.849** |
| Amlodipine MPO | 0.722 | 0.894 | 0.888 | 0.906 | **0.908** |
| Valsartan SMARTS | 0.552 | 0.978 | 0.095 | 0.997 | **0.999** |

### PMO Benchmark (AUC-Top10)

| Task | REINVENT | LSTM HC | LSTM PPO | **LSTM PSV-PPO** |
|------|----------|---------|----------|------------------|
| albuterol_similarity | 0.882 | 0.719 | 0.527 | **0.761** |
| drd2 | 0.945 | 0.919 | 0.883 | **0.959** |
| gsk3b | 0.865 | 0.839 | 0.794 | **0.869** |
| isomers_c9h10n2o2pf2cl | 0.642 | 0.342 | 0.608 | **0.652** |
| celecoxib_rediscovery | 0.713 | 0.539 | 0.532 | **0.612** |
| amlodipine_mpo | 0.635 | 0.593 | 0.587 | **0.647** |

**Key Findings**: LSTM PSV-PPO consistently outperforms LSTM PPO across all tasks, demonstrating the effectiveness of the PSV validation mechanism.

### Ablation Study

- **Removing the PSV table** (PSV-PPO_WO_PSV): Molecular validity rate drops significantly, validating the necessity of PSV for maintaining validity.
- **Removing GPS/TPC loss** (PSV-PPO_WO_PL): Duplication rates in the experience replay buffer and generation phase increase, confirming the role of these two loss terms in preventing mode collapse.

### Molecular Docking Experiments

On the fa7 protein target, PSV-PPO demonstrates competitive performance across Top-1/10/100 and diversity metrics compared to HC and standard PPO.

## Highlights & Insights

- **Step-by-step validation vs. Post-generation validation**: PSV performs validation at each autoregressive step rather than checking after the complete molecule is generated, achieving immediate feedback and significantly reducing the generation of invalid molecules.
- **Hellinger distance replacing KL divergence**: Elegantly solves the problem of KL divergence becoming incomputable when zero-probabilities are introduced by PSV filtering.
- **Normalized entropy loss**: Normalization by $\log(\text{len}(D_{PSV}))$ prevents the model from biasing toward tokens with larger valid sets.
- **Framework scalability**: The PSV framework can be extended to inject other domain knowledge (such as synthetic accessibility, toxicity constraints, etc.), and is not limited to validity validation.
- **High compatibility**: Based on the standard LSTM + PPO architecture, it can be easily integrated into existing SMILES generation pipelines.

## Limitations & Future Work

- PSV validation only guarantees the local validity of partial SMILES and **cannot guarantee that the final complete molecule is valid** (though it significantly improves the validity rate).
- The experiments are primarily based on LSTM pre-trained models; the performance on stronger architectures like Transformers has not been validated.
- The calculation of the PSV table introduces extra overhead; though the paper claims parallel computation minimizes this, its scalability on ultra-long sequences is not extensively discussed.
- It does not outperform the best baselines on certain GuacaMol tasks (such as Median molecules 2, Sitagliptin MPO).
- Molecular docking experiments are validated only on a single protein target (fa7); the breadth of biological relevance needs further investigation.

## Related Work & Insights

- **REINVENT** (Blaschke et al., 2020): Uses prior models to anchor pre-trained knowledge, but restricts exploration.
- **SELFIES** (Krenn et al., 2022): Enforces validity at the representation level, but limits optimization capability.
- **PPO** (Schulman et al., 2017): The foundational RL algorithm upon which PSV-PPO is extended.
- **partialsmiles** (O'Boyle, 2024): A toolkit providing real-time SMILES syntax validation, serving as the basis for the PSV table.
- The "step-by-step validation + domain knowledge injection" idea in this paper can be generalized to constraint satisfaction problems in other sequence generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The design of the PSV truth table and multiple loss terms is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — PMO, GuacaMol, docking, and ablation are all covered, but with a single target.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete formulas, and rich tables.
- Value: ⭐⭐⭐⭐ — Catastrophic forgetting is a real pain point in RL molecular generation; the solution is practical and scalable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improved Off-policy Reinforcement Learning in Biological Sequence Design](improved_off-policy_reinforcement_learning_in_biological_sequence_design.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](piloting_structure-based_drug_design_via_modality-specific_optimal_schedule.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](../../NeurIPS2025/computational_biology/gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
