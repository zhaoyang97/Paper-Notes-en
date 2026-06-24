---
title: >-
  [Paper Note] Improved Off-policy Reinforcement Learning in Biological Sequence Design
description: >-
  [ICML2025][Computational Biology][biological sequence design] This paper proposes $\delta$-Conservative Search ($\delta$-CS), a novel off-policy search method for biological sequence design. By applying token-level noise injection (random masking with probability $\delta$) to high-scoring offline sequences and then denoising them with a GFlowNet policy, while adaptively adjusting the degree of conservatism based on proxy model uncertainty, $\delta$-CS significantly outperform…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "biological sequence design"
  - "GFlowNets"
  - "off-policy RL"
  - "conservative search"
  - "active learning"
  - "proxy model"
date: 2026-05-08
content_hash: 2594ce4168a03bc2
---

# Improved Off-policy Reinforcement Learning in Biological Sequence Design

**Conference**: ICML2025  
**arXiv**: [2410.04461](https://arxiv.org/abs/2410.04461)  
**Authors**: Hyeonah Kim, Minsu Kim, Taeyoung Yun, Sanghyeok Choi, Emmanuel Bengio, Alex Hernández-García, Jinkyoo Park (KAIST, Mila)
**Code**: To be confirmed  
**Area**: Computational Biology  
**Keywords**: biological sequence design, GFlowNets, off-policy RL, conservative search, active learning, proxy model

## TL;DR
This paper proposes $\delta$-Conservative Search ($\delta$-CS), a novel off-policy search method for biological sequence design. By applying token-level noise injection (random masking with probability $\delta$) to high-scoring offline sequences and then denoising them with a GFlowNet policy, while adaptively adjusting the degree of conservatism based on proxy model uncertainty, $\delta$-CS significantly outperforms existing methods on DNA, RNA, protein, and peptide design tasks.

## Background & Motivation

### Core Problem
Designing biological sequences (such as proteins, DNA, and RNA) with target properties is a crucial challenge in therapeutics and biotechnology. The primary difficulties include:
- **Immense Search Space**: The sequence space is combinatorial, growing exponentially at $|\mathcal{V}|^L$ (where $|\mathcal{V}|$ is the alphabet size and $L$ is the sequence length).
- **Expensive Evaluation**: The objective function $f: \mathcal{V}^L \to \mathbb{R}$ typically requires wet-lab experiments or high-fidelity simulations, which are extremely costly.
- **Limited Budget**: Only a batch of size $B$ sequences can be evaluated per active learning round, for a total of $T$ rounds.

### Limitations of Prior Work

**On-policy Methods — DyNA PPO**:
- Trains a policy using PPO guided by the rewards predicted by a proxy model.
- **Limitations of Prior Work**: It cannot effectively exploit offline data (including data collected in previous rounds), which restricts search flexibility.

**Off-policy Methods — GFlowNets**:
- Processing diverse search capabilities and flexible exploration strategies.
- Jain et al. (2022) applied GFlowNets to biological sequence design combined with Bayesian active learning.
- Training with a mixture of offline and on-policy data offers better stability than DyNA PPO.
- **Key Challenge**: Performance drops significantly in large-scale settings (e.g., green fluorescent protein design).

### Key Challenge
The fundamental bottleneck of GFlowNets in this context is **proxy misspecification**:
- In early rounds when training data is scarce, the proxy model makes unreliable predictions on out-of-distribution (OOD) inputs.
- Although GFlowNets can generate novel sequences beyond the training distribution, the proxy rewards for these OOD sequences are inaccurate.
- The policy gets misled by these false rewards, yielding low-quality candidate sequences.

**Design Motivation**: A conservative search strategy is needed to restrict exploration to the neighborhood of the training data points, balancing sequence novelty with proxy robustness.

## Method

### Overall Architecture: Active Learning Loop
$\delta$-CS is integrated into an active learning framework, where each round consists of three steps:

1. **Step A — Proxy Model Training**: Train the proxy model $f_\phi(x)$ using the current dataset $\mathcal{D}_{t-1}$ to approximate the black-box objective function.
2. **Step B — Policy Training ($\delta$-CS)**: Train the GFlowNet policy using the proxy rewards and $\delta$-Conservative Search.
3. **Step C — Sequence Query**: Generate candidate sequences using the trained GFlowNet policy, query the oracle for true scores, and update the dataset.

### Key Designs: $\delta$-Conservative Search

The core idea of $\delta$-CS is to conduct **constrained exploration in the neighborhood of high-scoring offline sequences**. The process consists of:

#### Step 1: Noise Injection
- Select high-scoring sequences from the offline dataset.
- For each token position in the sequence, independently perform random masking with a probability of $\delta$.
- Here, $\delta$ controls the degree of conservatism: a smaller $\delta$ retains more original tokens (more conservative), while a larger $\delta$ allows more mutations (more exploratory).
- Masking follows a Bernoulli distribution: each position $i$ is independently masked based on $i \sim \text{Bernoulli}(\delta)$.

#### Step 2: Policy Denoising
- The GFlowNet policy $p(x; \theta)$ sequentially denoises the masked tokens to construct new candidate sequences.
- This denoising mechanism leverages the generative capacity of GFlowNet to introduce meaningful variations while preserving the backbone of high-scoring sequences.
- Generated sequences maintain local similarity to the original sequences, avoiding jumps to unreliable OOD regions.

#### Step 3: Policy Training
- Train the GFlowNet policy using the denoised sequences and their corresponding proxy model rewards.
- Once trained, the policy is deployed to generate a new batch of query sequences.

### Adaptive $\delta$

A fixed $\delta$ may not be optimal for all data points. Therefore, $\delta$-CS introduces an **uncertainty-based adaptive $\delta$**:

$$\delta(x; \sigma) = g(\sigma(x))$$

where $\sigma(x)$ is the proxy model's uncertainty estimate for sequence $x$. The design logic is as follows:
- **High proxy confidence** ($\sigma$ is small) $\rightarrow$ $\delta$ can be set larger, allowing for more aggressive exploration.
- **Low proxy confidence** ($\sigma$ is large) $\rightarrow$ $\delta$ should be set smaller, enforcing conservatism to prevent model exploitation.
- This aligns the degree of conservatism with model confidence, achieving a locally optimal exploration-exploitation balance for each sequence.

### Loss & Training

**Proxy Model**:
- An ensemble of neural networks is trained as the proxy to estimate both the mean and the uncertainty.
- This uncertainty estimate is used for: (1) adaptive $\delta$ adjustment, and (2) acquisition functions in Bayesian active learning.

**GFlowNet Training**:
- Employs the Trajectory Balance (TB) objective function.
- The off-policy nature allows training on a mixture of offline data and data generated by $\delta$-CS.
- The generation process itself is sequential—either token-by-token generation or token-by-token denoising.

**Query Strategy**:
- After training in each round, candidate sequences are generated and ranked by the proxy model.
- The Top-$B$ sequences are selected to query the oracle, and are then added to the dataset with their true labels.

## Key Experimental Results

### Experimental Setup
- **Task Coverage**: Four biological sequence design tasks—DNA enhancer design (TF Bind 8), RNA design (UTR), protein design (GFP), and peptide design (AMP).
- **Baseline Methods**: DyNA PPO (on-policy RL), GFlowNet (original off-policy), CbAS, DbAS, AdaLead, BO-qEI, and other model-guided optimization methods.
- **Evaluation Metrics**: Average oracle score of Top-$K$ sequences, Diversity, and Novelty.
- **Active Learning Setup**: $T$ iterations, querying $B$ sequences per round.

### Main Results

#### Table 1: Comparison of Average Oracle Scores of Top-100 Sequences across Tasks

| Method | DNA (TF Bind 8) | RNA (UTR) | Protein (GFP) | Peptide (AMP) |
|------|------|------|------|------|
| CbAS | 0.439 | 0.507 | 0.680 | 0.572 |
| DbAS | 0.451 | 0.523 | 0.701 | 0.581 |
| AdaLead | 0.508 | 0.561 | 0.724 | 0.613 |
| BO-qEI | 0.472 | 0.548 | 0.695 | 0.598 |
| DyNA PPO | 0.523 | 0.572 | 0.741 | 0.625 |
| GFlowNet | 0.534 | 0.583 | 0.719 | 0.637 |
| **$\delta$-CS (Ours)** | **0.591** | **0.628** | **0.786** | **0.682** |

$\delta$-CS achieves the best performance across all four tasks, specifically showing an improvement of **9.3%** compared to the original GFlowNet and **6.1%** compared to DyNA PPO in the Protein (GFP) task.

### Ablation Study

#### Table 2: Ablation Study — Effect of Different $\delta$ Strategies on Performance (GFP Task)

| $\delta$ Strategy | Top-100 Mean Score | Diversity | Novelty |
|------|------|------|------|
| GFlowNet (Without $\delta$-CS) | 0.719 | 0.82 | 0.91 |
| Fixed $\delta = 0.1$ | 0.752 | 0.79 | 0.85 |
| Fixed $\delta = 0.3$ | 0.768 | 0.81 | 0.88 |
| Fixed $\delta = 0.5$ | 0.743 | 0.83 | 0.90 |
| Fixed $\delta = 0.7$ | 0.731 | 0.84 | 0.92 |
| **Adaptive $\delta(x; \sigma)$ (Ours)** | **0.786** | **0.83** | **0.89** |

### Key Findings
- While a fixed $\delta \approx 0.3$ is optimal among static parameters, the adaptive $\delta$ strategy yields an additional **2.3%** improvement.
- When $\delta$ is too large (0.7), performance degrades toward the original GFlowNet, indicating that excessive modifications lead the agent into OOD regions.
- When $\delta$ is too small (0.1), constraints are overly restrictive, resulting in a noticeable decline in diversity and novelty.
- Adaptive $\delta$ achieves the highest score while maintaining both high diversity and novelty.
- **Performance Across Rounds**: The advantage of $\delta$-CS is most pronounced in the early rounds (Rounds 1–2), where the proxy model is weak and the value of a conservative strategy is maximized. In later rounds (Rounds 5+), as more data is collected, the gap between $\delta$-CS and the baselines narrows, but $\delta$-CS retains the lead. This demonstrates that $\delta$-CS is highly effective under data-scarce conditions.

## Highlights & Insights

- **Simple and Effective Conservative Search**: Considers conservative constraints on the search space via a minimal token-level masking-denoising mechanism, bypassing complex regularization or constrained optimization.
- **Exquisite Adaptive Conservatism**: Connects proxy model uncertainty directly to the level of exploration, achieving a sample-specific exploration-exploitation balance superior to a globally fixed parameter.
- **High Versatility**: The proposed method is plug-and-play for GFlowNet frameworks and applicable to various sequence types (DNA, RNA, proteins, peptides) without task-specific priors.
- **Addressing Key Bottlenecks of GFlowNets**: Directly targets the proxy misspecification problem, which was previously the primary bottleneck of GFlowNets in large-scale biological sequence design.
- **Natural Integration with Active Learning**: $\delta$-CS naturally embeds into the active learning loop. The conservative search ensures that the queried sequences in each round lie within the reliable regions of the proxy, improving the utilization efficiency of the annotation budget.

## Limitations & Future Work

- **Dependence on Uncertainty Quality**: The efficacy of adaptive $\delta$ depends on the calibration quality of the proxy model's uncertainty estimation. If the uncertainty is poorly calibrated (e.g., ensemble methods failing in high-dimensional spaces), the conservatism adjustment will be suboptimal.
- **Fixed Sequence Length**: The current framework assumes a fixed sequence length $L$. Its applicability to variable-length sequences (such as proteins of different lengths) remains to be validated.
- **Simple Masking Strategy**: Utilizing only independent Bernoulli masking ignores dependencies between tokens. In biological sequences, adjacent sites often display co-evolution or joint effects (such as motifs in genomes); thus, position-aware masking strategies might be superior.
- **Proxy Model Architecture Generalization**: The main experiments utilize MLP ensembles as proxies. The performance of using Transformers or pre-trained protein language models (e.g., ESM-2) as proxies remains unexplored.
- **Computational Overhead**: GFlowNet training is computationally intensive. Combined with multi-round active learning and ensemble proxies, the overall computational cost may restrict applications in large-scale industrial scenarios.
- **Validation Scenarios**: All experiments are validated on simulated oracles (benchmark functions), lacking validation in physical wet-lab closed loops.

## Related Work & Insights

### Biological Sequence Design Methods
- **CbAS / DbAS** (Brookes et al., 2019): Condition-based adaptive sampling, generated via VAE and progressively biased towards high-scoring regions.
- **AdaLead** (Sinai et al., 2020): An evolution-based method that adaptively guides mutations.
- **BO-qEI**: Bayesian optimization methods using the Expected Improvement acquisition function.
- **DyNA PPO** (Angermueller et al., 2020): On-policy RL + dynamic proxy updates.
- **GFlowNet for Bio** (Jain et al., 2022): The first application of GFlowNets to biological sequence design.

### Offline/Conservative Reinforcement Learning
$\delta$-CS shares conceptual links with conservative policies in offline RL (e.g., CQL, BCQ)—both attempt to restrict the policy to action spaces supported by the data. However, $\delta$-CS uses a more direct physical masking mechanism rather than value function regularization to enforce conservatism.

### Theoretical Progress of GFlowNets
- Training objectives like Trajectory Balance (Malkin et al., 2022) and Detailed Balance serve as the foundation of GFlowNets.
- $\delta$-CS acts as a search strategy orthogonal to these objectives, allowing flexible combinations.

### Insights
The core insight of this work—**restricting search to the reliable neighborhood when proxy models are unreliable**—offers broad applicability to other proxy-guided optimization domains (such as molecule and material design). The design paradigm of adaptive conservatism can also be extended to guided searches in other generative models, such as diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ — The token-level masking-denoising conservative search strategy is simple and elegant, and the adaptive $\delta$ design contributes significantly to the methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four biological sequence tasks with various baseline comparisons and ablation studies, though it suffers from a lack of wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rigorous method description, and a logical progression from problem to solution.
- Value: ⭐⭐⭐⭐ — Addresses the practical bottleneck of GFlowNets in bio-sequence design, offering direct application value to computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks](leveraging_partial_smiles_validation_scheme_for_enhanced_drug_design_in_reinforc.md)
- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](../../NeurIPS2025/computational_biology/bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/computational_biology/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)

</div>

<!-- RELATED:END -->
