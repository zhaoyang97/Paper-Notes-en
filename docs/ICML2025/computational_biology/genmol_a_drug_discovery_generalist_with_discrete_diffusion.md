---
title: >-
  [Paper Note] GenMol: A Drug Discovery Generalist with Discrete Diffusion
description: >-
  [ICML2025][Computational Biology][Discrete Diffusion Models] This work proposes GenMol, a generalist molecular generation framework based on Masked Discrete Diffusion, which generates SAFE sequences via non-autoregressive bidirectional parallel decoding. By introducing fragment remasking and Molecular Context Guidance (MCG), it covers four major drug discovery scenarios: de novo generation, fragment-constrained generation, target-directed hit generation, and lead optimization…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Discrete Diffusion Models"
  - "SAFE Molecular Representation"
  - "Fragment Remasking"
  - "Molecular Optimization"
  - "Drug Design"
date: 2026-05-08
content_hash: ddd7bdb8be4fb5d0
---

# GenMol: A Drug Discovery Generalist with Discrete Diffusion

**Conference**: ICML2025  
**arXiv**: [2501.06158](https://arxiv.org/abs/2501.06158)  
**Code**: [NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol)  
**Area**: Drug Discovery / Molecule Generation  
**Keywords**: Discrete Diffusion Models, SAFE Molecular Representation, Fragment Remasking, Molecular Optimization, Drug Design

## TL;DR
This work proposes GenMol, a generalist molecular generation framework based on Masked Discrete Diffusion, which generates SAFE sequences via non-autoregressive bidirectional parallel decoding. By introducing fragment remasking and Molecular Context Guidance (MCG), it covers four major drug discovery scenarios: de novo generation, fragment-constrained generation, target-directed hit generation, and lead optimization, using a **single model** and comprehensively outperforming previous state-of-the-art methods.

## Background & Motivation
Drug discovery involves multiple stages: de novo molecular generation, fragment-constrained generation (such as linker design, scaffold morphing, etc.), target-directed hit generation, and lead optimization. Existing molecular generation models typically cover only one or two of these scenarios, failing to act as generalist tools throughout the entire pipeline.

The most representative prior generalist method is **SAFE-GPT**, which represents molecules as SAFE (Sequential Attachment-based Fragment Embedding) sequences and utilizes GPT autoregressive decoding to complete multiple tasks. However, SAFE-GPT has three main limitations:

**Token order dependency**: SAFE itself is fragment-order independent, but GPT's left-to-right decoding contradicts this property.

**Low efficiency**: Autoregressive token-by-token generation prevents parallel decoding.

**Difficulty in guidance**: Autoregressive models struggle to introduce global guidance during the generation process; target-directed generation requires additional reinforcement learning fine-tuning.

The core motivation of GenMol is to **replace autoregression with discrete diffusion**, addressing these three pain points while maintaining the advantages of the SAFE representation.

## Method

### Overall Architecture
GenMol adopts the **BERT architecture** as its denoising network, with a training framework based on **MDLM** (Masked Discrete Language Model). The input consists of SAFE molecular sequences. The forward process gradually replaces tokens with [MASK], while the reverse process predicts masked tokens in parallel via bidirectional attention.

### Forward Masking Process
Interpolating each token $\boldsymbol{x}^l$ in the sequence independently:

$$q(\boldsymbol{z}_t^l | \boldsymbol{x}^l) = \text{Cat}(\boldsymbol{z}_t^l;\; \alpha_t \boldsymbol{x}^l + (1-\alpha_t)\mathbf{m})$$

where $\alpha_t$ is a monotonically decreasing masking rate schedule function. At $t=0$, all tokens are unmasked, and at $t=1$, all tokens are masked.

### Reverse Decoding Process
Unmasked tokens remain unchanged; for masked positions, the model predicts the denoising distribution:

$$p_\theta(\boldsymbol{z}_s^l | \boldsymbol{z}_t^l = \mathbf{m}) = \text{Cat}\!\left(\boldsymbol{z}_s^l;\; \frac{(1-\alpha_s)\mathbf{m} + (\alpha_s - \alpha_t)\boldsymbol{x}_\theta^l(\boldsymbol{z}_t, t)}{1-\alpha_t}\right)$$

### Training Loss
The NELBO loss is essentially a weighted average of MLM (cross-entropy) losses across different timesteps:

$$\mathcal{L}_{\text{NELBO}} = \mathbb{E}_q \int_0^1 \frac{\alpha_t'}{1-\alpha_t} \sum_l \log \langle \boldsymbol{x}_\theta^l(\boldsymbol{z}_t, t),\; \boldsymbol{x}^l \rangle \, dt$$

### Confidence Sampling
During each decoding step, the model predicts in parallel for all masked positions, revealing the top-$N$ tokens with the highest confidence. The quality-diversity trade-off is controlled via the softmax temperature $\tau$ and stochasticity $r$.

### Fragment Remasking
This is the core strategy of GenMol for **target-directed molecular optimization**, consisting of a three-step cycle:

1. **Fragment Scoring**: Disassembling the molecular collection into a fragment vocabulary, where each fragment is scored by the average target property of the molecules containing it: $y(\boldsymbol{f}_k) = \frac{1}{|\mathcal{S}(\boldsymbol{f}_k)|} \sum_{\boldsymbol{x} \in \mathcal{S}(\boldsymbol{f}_k)} y(\boldsymbol{x})$
2. **Fragment Assembly**: Randomly selecting two high-scoring fragments from the vocabulary and assembling them into an initial molecule.
3. **Fragment Remasking**: Randomly selecting a fragment from the initial molecule, replacing it with a [MASK] sequence, and regenerating a new fragment with GenMol.

This process can be interpreted as **fragment-level Gibbs sampling**—randomly walking in the neighborhood of a given molecule and dynamically updating the fragment vocabulary to explore chemical spaces beyond the initial dataset.

### Molecular Context Guidance (MCG)
Inspired by autoguidance, MCG interpolates the predictions of "good inputs" and "bad inputs" in the logit space:

$$\log \boldsymbol{x}_{\theta,i}^{(w),l} := w \log \boldsymbol{x}_{\theta,i}^l(\boldsymbol{z}_t, t) + (1-w) \log \boldsymbol{x}_{\theta,i}^l(\tilde{\boldsymbol{z}}_t, t)$$

where $\tilde{\boldsymbol{z}}_t$ is a corrupted input obtained by performing an additional mask of $\gamma \cdot 100\%$ on tokens in $\boldsymbol{z}_t$, and $w>1$ represents the guidance strength. This enables GenMol to better exploit molecular context information during fragment-constrained and target-directed generation.

## Key Experimental Results

### De Novo Generation

| Method | Validity(%) | Uniqueness(%) | Quality(%) | Diversity |
|------|------------|--------------|-----------|-----------|
| SAFE-GPT | 94.0 | 100.0 | 54.7 | 0.879 |
| GenMol (N=1, τ=0.5, r=0.5) | **100.0** | 99.7 | **84.6** | 0.818 |
| GenMol (N=3, τ=0.5, r=0.5) | 95.6 | 99.0 | 67.1 | 0.861 |

**Key Findings**: The Quality of GenMol improves from 54.7% of SAFE-GPT to 84.6% (+30pp), while achieving a 100% Validity. When N=3, the sampling speed is 2.5× faster than SAFE-GPT.

### Fragment-Constrained Generation (Average Quality)

| Method | Linker | Scaffold Morphing | Motif Extension | Scaffold Decoration | Superstructure |
|------|--------|-------------------|-----------------|--------------------|----|
| SAFE-GPT | 21.7 | 16.7 | 18.6 | 10.0 | 14.3 |
| GenMol | **21.9** | — | **30.1** | **31.8** | **34.8** |

GenMol consistently outperforms SAFE-GPT across all five subtasks.

### Target-Directed Hit Generation (PMO Benchmark, 23 Tasks)

| Method | Sum AUC Top-10 |
|------|---------------|
| **GenMol** | **18.362** |
| f-RAG | 16.928 |
| Genetic GFN | 16.213 |
| Mol GA | 14.708 |
| REINVENT | 14.196 |

GenMol achieves the **best performance in 19 out of 23 tasks**, with a total score of 18.362, significantly outperforming the runner-up f-RAG (+1.434).

### Lead Optimization
Across 30 tasks (5 target proteins × 3 seed molecules × 2 similarity thresholds), GenMol successfully optimizes the molecules in **26/30** tasks (while baselines fail extensively at $\delta=0.6$). This validates the effectiveness of the fragment remasking strategy in exploring chemical space.

## Highlights & Insights

1. **Unified Framework**: A single model and single checkpoint cover all four drug discovery scenarios without requiring task-specific fine-tuning.
2. **Fragment Remasking as Fragment-Level Gibbs Sampling**: Combines the remasking of discrete diffusion with chemical intuition (where fragments serve as functional units), outperforming token-level remasking.
3. **MCG Guidance Without Extra Training**: Directly guides the generation process through the contrast with corrupted inputs, requiring no conditional training or RL fine-tuning.
4. **Quality-Diversity Pareto Frontier**: The generation strategy can be continuously adjusted through the $(\tau, r)$ parameters, allowing users to flexibly balance quality and diversity according to their needs.
5. **Non-Autoregressive Parallel Decoding**: Naturally aligns with the fragment-order independence of SAFE while accelerating sampling.

## Limitations & Future Work

1. **Limited to 2D Molecular Graphs**: GenMol generates SAFE strings (2D) rather than 3D conformations directly, requiring post-processing for docking tasks that demand 3D structures.
2. **Docking Scores as Oracle**: Lead optimization relies on docking scores to evaluate binding affinity, which may require more accurate evaluation in real-world scenarios.
3. **Fragment Decomposition Dependent on BRICS Rules**: Predefined decomposition rules might overlook certain chemically meaningful substructures.
4. **Lack of Joint Protein-Ligand Modeling**: Currently, the model does not consider the 3D pocket information of target proteins, which limits structure-based drug design.
5. **Hyperparameters for MCG Guidance**: Selecting the guidance strength $w$ and additional masking ratio $\gamma$ requires task-specific hyperparameter tuning.

## Related Work & Insights

- **SAFE-GPT** (Noutahi et al., 2024): An autoregressive model also based on SAFE representation and the direct predecessor of GenMol. GenMol replaces GPT with discrete diffusion.
- **MDLM** (Sahoo et al., 2024): The training framework for masked discrete diffusion; GenMol directly adopts its loss function.
- **f-RAG** (Lee et al., 2024a): Fragment-level retrieval-augmented generation. GenMol inherits its fragment scoring equation from this work.
- **Mol GA** (Tripp &amp; Hernández-Lobato, 2023): Genetic algorithm-based molecular optimization. Fragment remasking can be viewed as the diffusion counterpart of its fragment-level mutation.
- **Autoguidance** (Karras et al., 2024): The theoretical foundation of MCG. GenMol generalizes it from continuous diffusion to masked discrete diffusion.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of discrete diffusion, SAFE, and fragment remasking is novel. MCG is the first to introduce autoguidance into masked discrete diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering four major tasks, 23+30 subtasks, and multiple baselines, alongside thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear framework, intuitive illustrations, and rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ — A unified framework achieving SOTA across all tasks, offering high practical utility and potential for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Compressing Biology: Evaluating the Stable Diffusion VAE for Phenotypic Drug Discovery](../../NeurIPS2025/computational_biology/compressing_biology_evaluating_the_stable_diffusion_vae_for_phenotypic_drug_disc.md)
- [\[ICLR 2026\] Refine Drugs, Don't Complete Them: Uniform-Source Discrete Flows for Fragment-Based Drug Discovery](../../ICLR2026/computational_biology/refine_drugs_dont_complete_them_uniform-source_discrete_flows_for_fragment-based.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](../../NeurIPS2025/computational_biology/split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](../../NeurIPS2025/computational_biology/why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)

</div>

<!-- RELATED:END -->
