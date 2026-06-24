---
title: >-
  [Paper Note] The Knowledge Microscope: Features as Better Analytical Lenses than Neurons
description: >-
  [ACL 2025][Interpretability][Sparse Autoencoders] This paper systematically validates through experiments that features decomposed by SAEs (Sparse Autoencoders) comprehensively outperform traditional neurons as analytical units in three dimensions: knowledge representation influence, interpretability, and monosemanticity. It proposes FeatureEdit, the first feature-based model editing method, which significantly outperforms neuron-based methods in private knowledge erasing tas…
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Knowledge Neurons"
  - "Feature Analysis"
  - "Monosemanticity"
  - "Private Knowledge Erasing"
date: 2026-05-08
content_hash: 388c08715cbe9485
---

# The Knowledge Microscope: Features as Better Analytical Lenses than Neurons

**Conference**: ACL 2025  
**arXiv**: [2502.12483](https://arxiv.org/abs/2502.12483)  
**Code**: None (The paper mentions "Code and dataset will be available")  
**Area**: Other  
**Keywords**: Sparse Autoencoders, Knowledge Neurons, Feature Analysis, Monosemanticity, Private Knowledge Erasing

## TL;DR

This paper systematically validates through experiments that features decomposed by SAEs (Sparse Autoencoders) comprehensively outperform traditional neurons as analytical units in three dimensions: knowledge representation influence, interpretability, and monosemanticity. It proposes FeatureEdit, the first feature-based model editing method, which significantly outperforms neuron-based methods in private knowledge erasing tasks.

## Background & Motivation

1. **Background**: Understanding the storage and representation mechanisms of factual knowledge in LLMs is a core problem in mechanistic interpretability. Mainstream methods use MLP neurons as the analytical unit, giving rise to the "knowledge neuron" theory—the idea that certain specific neurons are responsible for storing specific knowledge.

2. **Limitations of Prior Work**: Neurons suffer from severe **polysemanticity**—a single neuron responds to multiple unrelated facts, leading to: (1) **limited knowledge representation capability**: knowledge is distributed across a large number of neurons, with any single neuron making a minimal contribution; (2) **poor interpretability**: a neuron is coupled with multiple unrelated facts, making it difficult to accurately describe its function. For example, Gemma-2 2B has only about 230k neurons, but is trained on approximately 2 trillion tokens, making it inevitable that a single neuron will be associated with multiple facts.

3. **Key Challenge**: Utilizing polysemantic neurons to analyze the mechanism of a single piece of knowledge presents a fundamental mismatch in granularity.

4. **Goal**: (1) Can SAEs be used to decompose neurons into finer-grained features to serve as analytical units? (2) Can features resolve the issues of limited knowledge representation and poor interpretability associated with neurons? (3) Do features outperform neurons in downstream tasks?

5. **Key Insight**: SAEs map the low-dimensional neuron space to a high-dimensional feature space (similar to dimension elevation to make originally inseparable data separable), aligning different facts with different features to achieve the effect of a "knowledge microscope".

6. **Core Idea**: Use sparse autoencoders to decompose neurons into features, which comprehensively outperform neurons in knowledge representation, interpretability, and monosemanticity, and perform better in privacy protection tasks.

## Method

### Overall Architecture

The study is divided into four parts: (1) Preliminary experiments to verify that SAE is the best neuron-to-feature decomposition method (vs. PCA, ICA, random directions); (2) Comparing the knowledge representation influence of features and neurons (via change in predicted probability after ablation $\Delta Prob$); (3) Comparing interpretability (via the correlation of LLM-predicted activation values $IS$) and monosemanticity (via the separability of activation distributions); (4) Proposing the FeatureEdit method to demonstrate practical value in the task of private knowledge erasing.

### Key Designs

1. **SAE Feature Extraction and Evaluation Methodology**:

    - Function: Decomposes MLP activations into interpretable high-dimensional features.
    - Mechanism: Uses pretrained SAEs provided by Gemma Scope to obtain feature activations from MLP activations $\mathbf{h}$ via the encoder function $\mathbf{f}(\mathbf{h}) = \sigma(\mathbf{W}_{enc}\mathbf{h} + \mathbf{b}_{enc})$ (where $\sigma$ is JumpReLU). High-activation features are selected using a threshold $\tau_1$: $\mathbf{F_a} = \{(l,p) | f_{l,p}(\mathbf{h}) > \tau_1 \cdot \max f \}$. When ablating features (setting them to zero), the original activation is replaced by reconstructing the activation vector via the SAE decoder. Two core evaluation metrics: $\Delta Prob$ (the proportional drop in the target answer probability after ablation) and $IS$ (the interpretability score, which is the correlation between LLM-predicted activations and ground-truth activations).
    - Design Motivation: The high-dimensional sparse representation of SAEs is inherently suited for the ideal analytical paradigm of "one feature for one type of knowledge", resolving the fundamental "one-to-many" issue of neurons.

2. **Multi-Component Feature Comparative Analysis**:

    - Function: Identifies the most effective feature extraction locations.
    - Mechanism: Compares features from three Transformer components: post-attention residual, MLP activation, and post-MLP residual. It is found that post-MLP residual features perform best in both $\Delta Prob$ (around 0.85, with ablating a single feature reaching 0.6) and $IS$ (around 0.6, which is 4 times higher than that of neurons).
    - Design Motivation: Comprehensively comparing features at different positions helps understand the flow of knowledge in Transformers—the residual connections processed by the MLP after attention processing contain the richest factual knowledge information.

3. **FeatureEdit Knowledge Erasing Method**:

    - Function: The first feature-based model editing method for erasing privacy-sensitive knowledge.
    - Mechanism: For a feature $f_l^i$ corresponding to the knowledge to be erased, a one-hot probe vector $\mathbf{p}^i$ is constructed, and its contribution pattern in the MLP weight space $\mathbf{h}^i = \mathbf{W}_e^T \mathbf{p}^i$ is reconstructed using the SAE decoder. Positions in $\mathbf{h}^i$ where the absolute value exceeds a threshold $\tau_2$ are identified, and these corresponding positions are zeroed out in the second-layer MLP weights $\mathbf{W}_l^{(2)}$. Unlike neuron-based methods (which zero out entire columns), FeatureEdit selectively zeros out specific positions in the weight matrix, achieving finer granularity.
    - Design Motivation: The monosemanticity of features enables more precise erasing—only affecting the target knowledge without spilling over to unrelated facts. Neuron-based methods zeroing out entire columns have too broad an impact.

### Loss & Training

This paper does not involve training new models and utilizes pretrained SAEs from Gemma Scope. FeatureEdit is a training-free inference-time editing method. The privacy dataset PrivacyParaRel is injected into the model via incremental fine-tuning before conducting erasing experiments.

## Key Experimental Results

### Preliminary Experiments: SAE vs Other Decomposition Methods

On Gemma-2 9B:

| Method | $\Delta Prob$ | IS (Interpretability Score) |
|------|-----|-----|
| SAE | ~0.78 | ~0.64 |
| ICA | ~0.60 (×1.3) | ~0.32 (×2.0) |
| PCA | Low | Low |
| Random Direction | Lowest | Lowest |

### Main Results: Features vs Neurons

| Analytical Unit | $\Delta Prob$ (Knowledge Representation) | IS (Interpretability) |
|---------|---------|---------|
| Post-MLP Features | **~0.85** | **~0.6** |
| MLP Features | ~0.75 | ~0.6 |
| Post-Attention Features | ~0.55 | ~0.5 |
| Knowledge Neurons | ~0.45 (1.9× gap) | ~0.15 (4× gap) |

Fine-grained ablation (gradually removing features/neurons):

| Number Removed | Post-MLP Feature $\Delta Prob$ | Neuron $\Delta Prob$ |
|---------|---------|---------|
| 1 | ~0.6 | ~0.2 (3× gap) |
| 5 | ~0.8 | ~0.35 |
| 10 | ~0.85 | ~0.4 |

### Private Knowledge Erasing (FeatureEdit vs Neuron-based)

| Metric | FeatureEdit | Neuron-based Method |
|------|--------|--------|
| Reliability (Erasing Success Rate) | **~0.8** | ~0.65 |
| Generalization (Cross-prompt Generalization) | **~0.7** | ~0.25 |
| Locality (Unrelated Knowledge Preservation) | **~0.7** | ~0.2 |
| $\Delta$PPL (Perplexity Change ↓) | **~0.1** | ~0.3 |

### Key Findings

- **Ablating a single post-MLP feature has an impact equivalent to ablating around 3 knowledge neurons**, indicating that features are more precise units for knowledge localization.
- The IS interpretability score of features is 4 times that of neurons (~0.6 vs ~0.15), validating the causal chain of "polysemanticity $\rightarrow$ poor interpretability".
- In monosemanticity experiments, the activation distribution of features shows a clear bimodal separation as the ratio of relevant facts increases, whereas the distribution of neurons heavily overlaps, showing significant activations even with 0% relevant facts ($p < 0.001$, Cohen's $d > 0.8$).
- FeatureEdit has the largest advantage in Generalization (0.7 vs 0.25), indicating that the feature-based method effectively prevents jailbreak issues where "changing the phrasing leaks the knowledge".
- The feature distribution pattern remains consistent across different numbers of features $N$ (from $1\times$ to $8\times$ model dimensions), allowing $N=4\times d_{model}$ to be fixed without hyperparameter tuning.

## Highlights & Insights

- **The analogy of a "knowledge microscope" is highly fitting**: SAEs act like observing coarse-grained neurons through a higher-resolution microscope, revealing fine-grained knowledge structures that were previously invisible. This intuition can inspire future work on analyzing the internal mechanisms of models using SAEs.
- **Granularity scale advantage of FeatureEdit**: Neuron-based methods zero out entire columns of the weight matrix (affecting all downstream connections of that neuron), whereas FeatureEdit tracks the distribution of features in the weight space via the SAE decoder, only zeroing out relevant positions to perform "precision surgery" instead of "crude amputation".
- **The direct validation method for monosemanticity is elegant**: By controlling the ratio of relevant facts in the input (0% $\rightarrow$ 100%) and observing changes in the activation distributions of features/neurons, it intuitively demonstrates that features are "on when they should be on, off when they should be off", whereas neurons are "always partially active".
- **Post-MLP residual is the best analysis location**: This finding is highly valuable for subsequent research using SAEs to analyze LLMs.

## Limitations & Future Work

- The experiments were only conducted on Gemma-2 (2B/9B) and need to be validated on more architectures and larger scale models.
- The quality of the SAE's own training affects the analysis conclusions—it relies on Gemma Scope's pretrained SAEs; different SAE implementations might lead to different results.
- The threshold $\tau_2$ in FeatureEdit needs to be set manually, lacking an adaptive determination method.
- PrivacyParaRel uses synthetic private data, which may distribute differently from privacy information naturally acquired in real models.
- Downstream applications of features in more complex knowledge operations (e.g., knowledge updating, knowledge conflict resolution) were not explored.

## Related Work & Insights

- **vs Knowledge Neurons (Dai et al. 2022; Chen et al. 2024a)**: Knowledge neuron theory assumes that specific neurons store specific knowledge, but this paper demonstrates that this correspondence is imprecise and that features are a more appropriate level of granularity.
- **vs SAE Interpretability (Bricken et al. 2023)**: Bricken et al. first demonstrated that SAEs can decompose neurons into interpretable features, but primarily in the general text processing domain. This paper focuses on the factual knowledge domain and provides a systematic quantitative comparison.
- **vs ROME/MEMIT (Meng et al. 2022, 2023)**: These methods perform knowledge editing based on knowledge neuron theory, and FeatureEdit provides a feature-level alternative, which is theoretically more precise.
- The idea of using SAE features for model editing can be extended to more scenarios—such as knowledge conflict detection, factual verification, and harmful knowledge localization in safety alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically comparing SAE features against neurons for factual knowledge analysis represents a fresh perspective, and FeatureEdit is the first feature-based editing method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematically validated from multiple angles (knowledge representation, interpretability, monosemanticity, downstream tasks) with sufficient statistical tests and intuitive visualizations.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, progressing through research questions (Q1 $\rightarrow$ Q2 $\rightarrow$ Q3), though formulas are somewhat dense.
- Value: ⭐⭐⭐⭐⭐ Provides a superior analytical tool and methodology for LLM interpretability research, and FeatureEdit has practical application value (privacy protection).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](degenerate_knowledge_neurons.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](../../NeurIPS2025/interpretability/are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)
- [\[AAAI 2026\] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](../../AAAI2026/interpretability/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICML 2026\] Expand Neurons, Not Parameters](../../ICML2026/interpretability/expand_neurons_not_parameters.md)

</div>

<!-- RELATED:END -->
