---
title: >-
  [Paper Note] Aligning Protein Conformation Ensemble Generation with Physical Feedback
description: >-
  [ICML 2025][Computational Biology][Protein conformation generation] This work proposes Energy-based Alignment (EBA), which integrates energy feedback from physical force fields into the fine-tuning process of diffusion generative models. By aligning the generative distribution with the physical energy landscape via a Boltzmann factor-weighted classification objective, the method achieves state-of-the-art (SOTA) performance in protein conformation ensemble generation on the AT…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Protein conformation generation"
  - "diffusion models"
  - "physical alignment"
  - "Boltzmann distribution"
  - "preference optimization"
date: 2026-05-08
content_hash: 844f15599686ec2f
---

# Aligning Protein Conformation Ensemble Generation with Physical Feedback

**Conference**: ICML 2025  
**arXiv**: [2505.24203](https://arxiv.org/abs/2505.24203)  
**Code**: None  
**Area**: Computational Biology  
**Keywords**: Protein conformation generation, diffusion models, physical alignment, Boltzmann distribution, preference optimization

## TL;DR

This work proposes Energy-based Alignment (EBA), which integrates energy feedback from physical force fields into the fine-tuning process of diffusion generative models. By aligning the generative distribution with the physical energy landscape via a Boltzmann factor-weighted classification objective, the method achieves state-of-the-art (SOTA) performance in protein conformation ensemble generation on the ATLAS MD benchmark.

## Background & Motivation

Protein dynamics are crucial for understanding protein function and regulation, with protein structures transitioning across multiple conformational states over varying spatial and temporal scales. Although traditional molecular dynamics (MD) simulations can capture these dynamic behaviors, they are computationally expensive. Capturing biologically relevant transitions such as folding/unfolding typically requires microsecond to millisecond-scale simulations, which often demand hundreds to thousands of GPU days.

Recently, denoising diffusion models have been applied to protein conformation generation, reformulating it as a conditional generation task. However, these data-driven approaches face two core limitations:

**Lack of Thermodynamic Modeling**: Although purely data-driven methods can generate structurally plausible candidate conformations, they do not explicitly model thermodynamic properties, making it impossible to guarantee that the generated samples follow the Boltzmann distribution.

**Intractable Partition Function**: A more principled formulation—sampling equilibrium conformation ensembles from the Boltzmann distribution—is directly intractable for optimization because the partition function $Z = \sum_{\mathbf{x}} e^{-\beta E(\mathbf{x};\mathbf{c})}$ requires summing over all possible states in a high-dimensional space.

**Limitations of Prior Work**: Existing amortized sampling approaches (e.g., GFlowNet) scale poorly to protein structures containing thousands of atoms.

## Method

### Overall Architecture

The core idea of EBA is that instead of approximating the intractable partition function $Z$, it leverages the **Boltzmann factor**—the relationship between the ratio of probabilities of two states and their energy difference:

$$\frac{p_B(\mathbf{x}^i|\mathbf{c})}{p_B(\mathbf{x}^j|\mathbf{c})} = e^{-\beta \Delta E_{ij}}$$

where $\Delta E_{ij} = E(\mathbf{x}^i;\mathbf{c}) - E(\mathbf{x}^j;\mathbf{c})$. This form relying on energy differences is translation-invariant to absolute energy values, making it particularly suitable for generative model training since energy scales vary significantly with the number of protein atoms.

The entire training workflow consists of two stages:

- **Stage 1 — Supervised Fine-Tuning**: The pre-trained AlphaFold3 diffusion module is fine-tuned on the ATLAS MD trajectory data to allow the model to roughly adapt to the data distribution of the conformational space.
- **Stage 2 — Physical Alignment**: The diffusion model is trained for alignment using the EBA objective function based on force field energy feedback.

### Key Designs

#### 1. Derivation of EBA Objective Function

Assuming a learnable probability model $p_\theta(\mathbf{x}|\mathbf{c}) = e^{-\alpha E_\theta(\mathbf{x};\mathbf{c})}/Z$, minimizing the KL divergence (i.e., cross-entropy) with respect to the target Boltzmann distribution yields:

$$\mathbb{D}_{\text{KL}}(p_B \| p_\theta) = -\sum_i p_B(\mathbf{x}^i|\mathbf{c}) \log p_\theta(\mathbf{x}^i|\mathbf{c}) + \text{Const}$$

Because summing over all possible conformational states is intractable, EBA employs a **stochastic finite subset approximation**: sampling $K$ representative states $\{\mathbf{x}^i\}_{i=1}^K$ from a proposal distribution $p^*$, which yields the EBA objective:

$$\mathcal{L}_{\text{EBA}}(\theta) = -\mathbb{E} \left[ \sum_{i=1}^K \frac{e^{-\beta E(\mathbf{x}^i;\mathbf{c})}}{\sum_{j=1}^K e^{-\beta E(\mathbf{x}^j;\mathbf{c})}} \log \frac{e^{-\alpha E_\theta(\mathbf{x}^i;\mathbf{c})}}{\sum_{j=1}^K e^{-\alpha E_\theta(\mathbf{x}^j;\mathbf{c})}} \right]$$

This is an energy-weighted classification objective that guarantees Boltzmann factor invariance within the mini-batch.

#### 2. Adapting EBA to Diffusion Models

By defining the energy function as the sum of KL divergences along the diffusion chain and using Jensen's inequality (due to the convexity of the LSE function) to derive an upper bound, the KL divergence is ultimately replaced by the denoising error. This yields the diffusion version of the EBA objective:

$$\mathcal{L}_{\text{EBA-Diff}} = -\mathbb{E} \sum_{i=1}^K \frac{e^{-\beta E(\mathbf{x}^i;\mathbf{c})}}{\sum_j e^{-\beta E(\mathbf{x}^j;\mathbf{c})}} \log \frac{e^{-\alpha T \|\epsilon^i - \epsilon_\theta(\mathbf{x}_t^i,t,\mathbf{c})\|_2^2}}{\sum_j e^{-\alpha T \|\epsilon^j - \epsilon_\theta(\mathbf{x}_t^j,t,\mathbf{c})\|_2^2}}$$

#### 3. DPO as a Special Case of EBA

When $K=2$ and the temperature approaches zero ($\beta \to \infty$), EBA reduces to the standard DPO objective. This establishes a theoretical connection between EBA and the RLHF/DPO literature, while demonstrating that EBA is a more general formulation—it supports comparison across more than two states and preserves fine-grained energy difference information rather than just binary preferences.

#### 4. SE(3)-Invariant Loss Design

Standard MSE is suboptimal for protein conformation generation due to the SE(3)-invariance of the input conditions (the amino acid sequence). The paper designs two SE(3)-invariant losses:

- **Rigid-Body Aligned MSE**: First aligns the predicted coordinates to the ground truth using the Kabsch algorithm, and then computes the aligned MSE.
- **Smooth LDDT**: An auxiliary loss based on the pairwise distance matrix that captures inter-atomic geometric relations, with weighted evaluation for atomic pairs within 15Å.

#### 5. Energy Normalization

The vast differences in protein sizes lead to extreme variance in energy values. To address this, the authors introduce a sample-specific normalization factor $L^{0.5}$ (where $L$ is the number of residues) to scale $\beta$: $\beta \leftarrow \beta / L^{0.5}$, inspired by the empirical finding that folding times scale with the 0.5-power of the number of residues.

### Loss & Training

The final denoising training loss is:

$$L_{\text{total}} = \lambda_{\text{mse}} L_{\text{Aligned MSE}} + \lambda_{\text{lddt}} L_{\text{Smooth LDDT}}$$

In the EBA framework, this $L_{\text{total}}$ serves as the "energy" input for each candidate sample to be normalized via softmax:

$$\mathcal{L}_{\text{EBA-Diffusion}} = -\sum_{i=1}^K w(\mathbf{x}_0^i) \log \frac{e^{-L_{\text{total}}^i}}{\sum_{j=1}^K e^{-L_{\text{total}}^j}}$$

where $w(\mathbf{x}_0^i)$ is the Boltzmann weight computed from physical energy. The training utilizes Protenix (an open-source implementation of AlphaFold3), freezing the MSA Module and PairFormer, and only fine-tuning the diffusion module. Energy labeling is pre-computed via offline local minimization.

## Key Experimental Results

### Main Results

Evaluated on the ATLAS MD benchmark test set ($N=250$ protein targets), reporting median results:

| Metric Category | Metric | AlphaFlow-MD | MSA-sub(256) | MDGen | Pre-train | EBA-DPO | **EBA** |
|---------|------|-------------|-------------|-------|-----------|---------|---------|
| Flexibility | Pairwise RMSD r↑ | 0.48 | 0.15 | 0.48 | 0.43 | 0.59 | **0.62** |
| Flexibility | Global RMSF r↑ | 0.60 | 0.26 | 0.50 | 0.50 | 0.69 | **0.71** |
| Flexibility | Per-target RMSF r↑ | 0.85 | 0.55 | 0.71 | 0.72 | 0.90 | **0.90** |
| Distribution Accuracy | Root mean W₂↓ | 2.61 | 3.62 | 2.69 | 3.22 | 2.43 | **2.43** |
| Distribution Accuracy | MD PCA W₂↓ | 1.52 | 1.88 | 1.89 | 1.78 | 1.20 | **1.19** |
| Ensemble Observation | Weak contacts J↑ | 0.62 | 0.30 | 0.51 | 0.23 | 0.63 | **0.65** |
| Ensemble Observation | Exposed residue J↑ | 0.50 | 0.33 | 0.29 | 0.29 | 0.68 | **0.70** |
| Ensemble Observation | Exposed MI ρ↑ | 0.25 | 0.06 | - | 0.01 | 0.35 | **0.36** |

EBA achieves state-of-the-art or second-best performance across all 14 metrics. The runtime efficiency is 0.9 GPU-seconds/sample, which is significantly faster than AlphaFlow-MD's 70 seconds (~78x speedup).

### Ablation Study

Impact of different mini-batch sizes $K$ on EBA performance:

| Configuration | K=2 | K=3 | K=5 | Description |
|------|-----|-----|-----|------|
| Pairwise RMSD r↑ | 0.62 | 0.61 | 0.62 | Stable performance |
| Global RMSF r↑ | 0.71 | 0.71 | 0.72 | Slight improvement |
| Root mean W₂↓ | 2.43 | 2.42 | 2.40 | Slightly improved |
| MD PCA W₂↓ | 1.19 | 1.18 | **1.16** | Optimal at K=5 |
| Exposed MI ρ↑ | 0.36 | **0.37** | 0.34 | Optimal at K=3 |
| Iteration time (s) | 4.3 | 5.4 | 7.8 | Quasi-linear growth |
| GPU memory (GB) | 12.0 | 13.9 | 16.3 | Mild overhead |

### Key Findings

1. **EBA significantly outperforms DPO variants**: The improvements in Exposed residue J (0.70 vs 0.68) and Exposed MI $\rho$ (0.36 vs 0.35) indicate that preserving fine-grained energy difference information (rather than binary preference alone) is crucial for capturing long-range dynamics.
2. **Effectiveness of physical alignment**: The substantial gain from Pre-train $\to$ EBA (e.g., Pairwise RMSD r from 0.43 to 0.62) demonstrates that physical feedback effectively corrects the bias of purely data-driven models.
3. **Robustness to K values**: The performance variance across K=2, 3, and 5 is minimal, indicating that the mini-batch approximation is effective, and K=2 is sufficient to achieve good performance.
4. **Efficiency advantage**: Operating at 0.9 seconds/sample, it is approximately 78x faster than AlphaFlow-MD (70s), and while slightly slower than MDGen (0.2s), it offers vastly superior accuracy.

## Highlights & Insights

1. **Theoretical elegance**: Unifies the theoretical frameworks of RLHF/DPO and physical Boltzmann distribution alignment. Proving that DPO is a special case of EBA ($K=2$, $\beta \to \infty$) establishes a deep connection between these two historically independent research areas.
2. **Avoidance of partition function calculation**: By modeling via Boltzmann factors (relative weights between states) rather than absolute probabilities, the approach cleverly circumvents the intractable partition function.
3. **All-atom modeling**: Based on AlphaFold3's all-atom diffusion model, it captures fine-grained conformational changes more directly compared to methods relying on coarse-grained or internal coordinate representations.
4. **Energy normalization trick**: The $L^{0.5}$ normalization solves the practical issue of massive energy scale variations across proteins of different sizes, demonstrating a deep integration of physical intuition.

## Limitations & Future Work

1. **Limited to short timescale dynamics**: AlphaFold3 was originally designed for folding prediction; even after fine-tuning, it may not be suitable for modeling microsecond-to-millisecond long-timescale dynamics.
2. **Inadequate force field accuracy**: The accuracy of the utilized energy function is lower than quantum-level single-point energy calculation, which may limit the physical accuracy of the generated conformations.
3. **Limited to single-chain proteins**: The current work is restricted to single-chain protein ensemble generation and has not yet been extended to multi-chain complexes.
4. **Single generative model framework**: EBA is implemented and evaluated solely within the diffusion framework, leaving alternative generative paradigms like Flow Matching or VAEs unexplored.
5. **Dependence on MD simulation data**: Training still requires reference distributions from ATLAS MD trajectory data, indicating that the method has not fully bypassed the dependency on expensive simulations.

## Related Work & Insights

- **AlphaFlow** (Jing et al., 2024a): Recasts AlphaFold2 as a denoising network and serves as the main baseline. EBA further introduces physical feedback on top of it.
- **Diffusion-DPO** (Wallace et al., 2024): Extends DPO to diffusion models for text-to-image generation. This work generalizes it to multi-state comparisons and physical energy weighting.
- **ConfDiff** (Wang et al., 2024): Introduces energy and force guidance during reverse diffusion, which acts as inference-time guidance rather than training-time alignment.
- **Boltzmann Generator** (Noé et al., 2019): Uses Normalizing Flows to approximate the Boltzmann distribution but struggles to scale to large proteins.

**Insights**: The "physical feedback alignment" paradigm of EBA can be generalized to other physical systems (e.g., crystals, small molecules) where a target energy landscape is defined. Migrating RLHF concepts to scientific computing is a direction highly worthy of deeper exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The EBA framework unifies RLHF/DPO with the Boltzmann distribution, providing solid theoretical contributions; however, the core idea (using an energy-weighted softmax classification objective) is not entirely brand new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensively evaluated on the standard ATLAS benchmark, leading across all 14 metrics; ablation studies cover key design choices.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The mathematical derivations are clear and complete, the derivation of DPO as a special case is elegant, and the logistics across motivation, method, and experiments are coherent.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for protein dynamics modeling, though the target applications remain domain-specific.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/computational_biology/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](../../NeurIPS2025/computational_biology/confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](../../NeurIPS2025/computational_biology/energy_loss_functions_for_physical_systems.md)

</div>

<!-- RELATED:END -->
