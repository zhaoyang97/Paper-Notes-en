---
title: >-
  [Paper Note] Why Is Attention Sparse in Particle Transformer?
description: >-
  [NeurIPS 2025][Interpretability][Particle Transformer] This paper systematically analyzes the near-binary sparse attention phenomenon observed in Particle Transformer (ParT) after training on jet tagging tasks. Through c…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Particle Transformer"
  - "sparse attention"
  - "jet tagging"
  - "interaction matrix"
date: 2026-05-08
content_hash: 544657001c081183
---

# Why Is Attention Sparse in Particle Transformer?

**Conference**: NeurIPS 2025
**arXiv**: [2512.00210](https://arxiv.org/abs/2512.00210)  
**Code**: [Available](https://github.com/)  
**Area**: Interpretability / Physics AI / Transformer
**Keywords**: Particle Transformer, sparse attention, jet tagging, interaction matrix, interpretability

## TL;DR

This paper systematically analyzes the near-binary sparse attention phenomenon observed in Particle Transformer (ParT) after training on jet tagging tasks. Through cross-dataset comparisons and ablation studies, it demonstrates that the sparsity primarily originates from the attention mechanism itself rather than the physics-inspired interaction matrix. Nevertheless, the interaction matrix remains indispensable to final performance by influencing the argmax particle selection for the vast majority of tokens.

## Background & Motivation

**Background**: In high-energy physics experiments at CERN's Large Hadron Collider (LHC), jet tagging—classifying collimated streams of particles (jets) produced in high-energy collisions—is one of the central analysis tasks. Deep learning-based jet taggers have achieved remarkable progress in recent years, with classification accuracy continuously improving from graph neural network approaches such as ParticleNet to Transformer-based models. Among these, Particle Transformer (ParT) is one of the current state-of-the-art jet taggers. It augments standard multi-head attention with a physics-inspired pairwise particle interaction matrix, encoding kinematic relationships between particle pairs (e.g., angular distance $\Delta$, transverse momentum $k_T$, momentum fraction $z$, invariant mass $m^2$) as attention bias terms, thereby integrating domain prior knowledge with data-driven representation learning.

**Limitations of Prior Work**: Despite ParT achieving state-of-the-art performance on multiple benchmark datasets, researchers have observed a puzzling phenomenon in its post-training attention maps: the attention weights are nearly binary (approaching 0 or 1), with each particle attending almost exclusively to one other particle in each attention head. Such extreme sparsity patterns are exceedingly rare in Transformers applied to natural language processing or computer vision. Prior work (Wang et al.) had already observed this phenomenon and visualized attention maps in the $\eta$-$\phi$ plane, finding that ParT appears to capture physically meaningful jet substructure (e.g., leptons in semi-leptonic decays). However, a systematic explanation of the origin and mechanism of this sparsity was lacking.

**Key Challenge**: ParT's attention computation comprises two components: the conventional $QK^T/\sqrt{d_k}$ attention term and the physics-inspired interaction matrix $U$, which are summed prior to the softmax operation. This raises a fundamental question: which component dominates the extreme binary sparsity behavior? Is it learned spontaneously by the attention mechanism, or is it induced by the physical prior encoded in the interaction matrix? Furthermore, if the magnitude of the interaction matrix is orders of magnitude smaller than the attention term (subsequent experiments confirm a difference of $10^4$–$10^5$ times), how important is it to model performance? Can it be safely removed at inference time to reduce computational cost?

**Goal**: Specifically, this paper focuses on three progressively deeper questions: (1) What is the origin of sparse attention in ParT—the conventional attention term or the interaction matrix? (2) Is sparsity universally present across different datasets and feature configurations? (3) Although the interaction matrix has negligible magnitude, what is its actual impact on model inference?

**Key Insight**: The authors adopt a sophisticated analytical strategy: first, they compare the absolute magnitude ratios of the attention and interaction terms at the pre-softmax stage to directly quantify their relative contributions; then, through systematic comparisons across datasets (JetClass, Top Landscape, Quark-Gluon) and feature configurations (full features vs. kinematic-only), they eliminate potential confounds such as PID features; finally, through ablation experiments that zero out the interaction matrix, they precisely measure its effect on inference performance and introduce the novel metric of *interaction-dependent computation* to explain the underlying mechanism.

**Core Idea**: Through magnitude analysis, cross-dataset comparison, and fine-grained ablation, the paper demonstrates that ParT's sparse attention arises spontaneously from the attention mechanism itself, while the interaction matrix—despite its negligible magnitude—is indispensable because it alters the argmax particle selection for the vast majority of tokens.

## Method

### Overall Architecture

Rather than proposing a new model architecture, this work presents an in-depth analytical study of the internal mechanisms of Particle Transformer. The research framework consists of four progressively deeper analytical modules: (1) **attention distribution visualization**, plotting post-softmax attention weight histograms across multiple datasets to directly illustrate the degree of binarization; (2) **pre-softmax magnitude ratio analysis**, computing the ratio of conventional attention scores to interaction matrix values to identify the dominant source of sparsity; (3) **particle attention maps in the $\eta$-$\phi$ plane**, projecting attention relationships into physical coordinate space using jet substructure clustering (the $k_T$ algorithm) to assess whether the model captures meaningful physical correlations; and (4) **zeroing ablation and interaction-dependence analysis**, zeroing out the interaction matrix parameters and tracking changes in argmax selections to precisely quantify the inference contribution of the interaction matrix.

The central object of study is the Particle Multi-Head Attention (P-MHA) mechanism in ParT. Given representations $x \in \mathbb{R}^{N \times d}$ for $N$ particles in a jet, the P-MHA computation is:

$$\text{head}_i = \text{softmax}\left(\frac{xW_i^Q(xW_i^K)^\top}{\sqrt{d_k}} + U_i\right) xW_i^V$$

where the first term $A = xW_i^Q(xW_i^K)^\top / \sqrt{d_k}$ is the standard scaled dot-product attention, and the second term $U_i \in \mathbb{R}^{N \times N}$ is the interaction matrix learned via convolutional layers from pairwise kinematic features ($\ln\Delta, \ln k_T, \ln z, \ln m^2$). The entire analysis revolves around the relative contributions of $A$ and $U$.

### Key Designs

1. **Pre-softmax Magnitude Ratio Analysis**:

    - *Function*: Quantifies the relative magnitudes of the conventional attention scores $A$ and the interaction matrix $U$ at the pre-softmax stage, determining which component dominates the final sparse attention distribution.
    - *Mechanism*: For each particle pair $(i,j)$ in each attention head, the ratio $|A_{ij}| / |U_{ij}|$ is computed, and its distribution is aggregated over the entire test set. If this ratio is much greater than 1, the softmax input is almost entirely governed by the $A$ term and the contribution of $U$ is numerically negligible. The authors conduct this analysis separately on three datasets: on JetClass (both full features and kinematic-only), the ratio is almost always greater than 1, with a peak at the $10^4$–$10^5$ order of magnitude, indicating that the attention term numerically overwhelms the interaction matrix. On the Top Landscape dataset, however, the two terms are of comparable magnitude, suggesting that the interaction matrix plays a more equal role in that setting.
    - *Design Motivation*: The key insight is that softmax is a competitive function in which the largest value dominates the output. If the magnitude of $A$ far exceeds that of $U$, then $U$ can barely alter the distributional shape of the softmax output. Therefore, the binary sparsity pattern in attention must be driven by $A$. This analysis elegantly transforms a qualitative observation ("attention is sparse") into a quantitative conclusion ("sparsity originates from the attention mechanism itself, independent of the interaction matrix"), providing a theoretical foundation for the subsequent ablation experiments.

2. **Cross-Dataset and Cross-Feature Sparsity Comparison**:

    - *Function*: Compares the degree of attention sparsity across different datasets (JetClass, Top Landscape, Quark-Gluon) and feature configurations (full features with PID vs. kinematic-only), ruling out the hypothesis that PID features are the source of sparsity.
    - *Mechanism*: The authors plot post-softmax attention weight histograms for four configurations: (a) the Quark-Gluon dataset, (b) the $t \to bqq'$ class in Top Landscape, (c) the $t \to bqq'$ class in JetClass with full features, and (d) the $t \to bqq'$ class in JetClass with kinematic-only features. The results show that JetClass exhibits extreme binary distributions under both feature configurations, as does the Quark-Gluon dataset, whereas the Top Landscape attention distribution is relatively smooth without binary characteristics. The critical comparison is between JetClass kinematic-only and Top Landscape, which use identical types of input features (kinematic information only, no PID), yet the former exhibits binarization while the latter does not. This directly rules out the hypothesis that "PID features cause sparsity" and suggests that sparsity is more likely related to dataset scale and task complexity.
    - *Design Motivation*: Prior work (Wang et al.) reported the sparsity phenomenon without ruling out PID as a potential cause. The JetClass version of ParT uses 17-dimensional features (including PID), while Top Landscape uses only 7-dimensional kinematic features; a naive comparison might lead to the erroneous conclusion that "PID causes sparsity." By observing sparsity on JetClass even after removing PID features, the authors clearly establish a relationship between sparsity and dataset/task characteristics rather than feature type, reflecting a rigorous experimental design.

3. **Attention Visualization and Jet Substructure Analysis in the $\eta$-$\phi$ Plane**:

    - *Function*: Projects attention relationships into the $\eta$ (pseudorapidity)–$\phi$ (azimuthal angle) coordinate space, widely used in particle physics, and combines $k_T$ clustering to decompose jets into subjets, assessing whether ParT learns physically meaningful particle correlations.
    - *Mechanism*: For each jet, the $k_T$ algorithm in FastJet is first used to cluster particles into a fixed number of subjets (2 for semi-leptonic decay $t \to b\ell\nu$, 3 for hadronic decay $t \to bqq'$). Particle positions are plotted in the $\eta$-$\phi$ plane with distinct symbols for particle types (✖ for muons, ▲ for charged hadrons, ▼ for neutral hadrons, ⚫ for photons, ✚ for electrons), transparency proportional to transverse momentum $p_T$, and line thickness reflecting attention scores. A key innovation is that only pre-softmax attention values (excluding the interaction matrix) are used to construct the visualization, isolating the information-capturing capability of the attention mechanism itself. Results show that even without the interaction matrix, ParT's attention mechanism can still identify subjet structures and inter-subjet correlations. More impressively, even under the kinematic-only configuration (without PID information), the model can accurately identify the lepton in semi-leptonic decays.
    - *Design Motivation*: The $\eta$-$\phi$ visualization is a standard analytical tool in high-energy physics; a model that demonstrates physically meaningful attention patterns in this space earns greater trust from physicists. Observing correct substructure even after removing the interaction matrix has two important implications: first, it confirms that sparse attention is not merely numerical sparsity but captures genuine physical correlations; second, it demonstrates that the attention mechanism alone possesses sufficient representational capacity to encode jet topology.

### Ablation Study Design

This paper employs a distinctive ablation strategy—*zeroing ablation* (Zero $U$ Ablation)—distinct from the standard approach. Standard ablation trains a new model from scratch without the interaction matrix (ParT plain), whereas the zeroing ablation sets all parameters of the PairEmbed module in a fully trained ParT model to zero and evaluates it directly. The distinction is crucial:

The standard ablation (ParT plain) achieves an accuracy of 0.849, a modest gap from the full model's 0.861, which has previously been interpreted as indicating that "the interaction matrix is not very important." However, the zeroing ablation (ParT Zero $U$) causes accuracy to plummet to 0.405. This demonstrates that the model has incorporated information from the interaction matrix into the learning of its other parameters during training—the optimization of other weights proceeds under the condition that the interaction matrix is present, and sudden removal disrupts the model's finely calibrated internal balance.

To further explain why a magnitude-negligible interaction matrix can produce such a dramatic effect, the authors introduce the novel metric of **interaction-dependent computation**. Formally, for particle $j$, if $\text{argmax}(A_j + U_j) \neq \text{argmax}(A_j)$—i.e., adding the interaction matrix changes the highest-weight particle selection—that computation is labeled interaction-dependent. Statistical results show that while only 3.6% of token updates across all attention heads involve interaction-dependent computation, 85.4% of tokens experience at least one instance of interaction-dependent computation across the model's attention heads. This means that although the interaction matrix is small in magnitude, it influences the final representations of the vast majority of tokens through a butterfly-effect-like mechanism by altering critical argmax selections.

The authors also track the proportion of **non-binary computation**—defined as cases where the maximum post-softmax weight is below 0.8. The overall proportion of non-binary computation is 0.88%, with 42.1% of tokens experiencing it at least once. By computing the Pearson correlation coefficient between interaction-dependent and non-binary computations (PCC = 0.229), the authors demonstrate that the two are not strongly correlated. This implies that the primary role of the interaction matrix is not to soften the binary nature of attention, but rather to alter the competitive outcome between particles while preserving binarization—guiding the model to attend to the "correct" particles.

## Key Experimental Results

### Main Results

The core experiments span three benchmark datasets, comparing ParT's performance and attention characteristics under different configurations.

| Configuration | Dataset | Accuracy | AUC | Sparsity |
|---|---|---|---|---|
| ParT (full) | JetClass | 0.861 | — | Strong binarization |
| ParT (Zero $U$) | JetClass | 0.405 | 0.8974 | — |
| ParT (plain, reported) | JetClass | 0.849 | — | — |
| ParT | Top Landscape | — | — | Non-binary |
| ParT | Quark-Gluon | — | — | Binary |

The detailed per-class performance table for ParT (Zero $U$) reveals the degree of degradation across categories:

| Class | $\text{Rej}_{50\%}$ (ParT Zero $U$) | Notes |
|---|---|---|
| $H \to b\bar{b}$ | 15.0 | Significant degradation |
| $H \to c\bar{c}$ | 8.81 | Significant degradation |
| $H \to gg$ | 19.9 | Relatively preserved |
| $H \to 4q$ | 5.53 | Severe degradation |
| $H \to \ell\nu qq'$ | 3.03 ($\text{Rej}_{99\%}$) | Severe degradation |
| $t \to bqq'$ | 79.5 | Relatively preserved |
| $t \to b\ell\nu$ | 2.69 ($\text{Rej}_{99.5\%}$) | Severe degradation |
| $W \to qq'$ | 25.6 | Significant degradation |
| $Z \to q\bar{q}$ | 11.8 | Significant degradation |

The table reveals that zeroing the interaction matrix has highly heterogeneous effects across decay modes. The $\text{Rej}_{50\%}$ for $t \to bqq'$ remains at 79.5, preserving relatively more discriminative capacity, whereas the performance for $H \to 4q$ and semi-leptonic decay modes nearly collapses. This suggests that the interaction matrix is more critical for classes involving more complex substructures or requiring fine-grained particle discrimination.

### Ablation Study

| Metric | Overall Rate | Token Coverage | PCC |
|---|---|---|---|
| Interaction-Dependent Computation | 3.6% | 85.4% | 0.229 |
| Non-binary Computation | 0.88% | 42.1% | — |

This set of results reveals a profound phenomenon: at the level of individual computations, the proportion influenced by the interaction matrix is modest (only 3.6%), yet due to ParT's multiple attention heads and layers, the cumulative effect results in 85.4% of tokens being substantially affected by the interaction matrix at least once during inference. This explains why a magnitude-negligible interaction matrix causes catastrophic performance collapse upon zeroing—not because it alters many individual computation outcomes, but because it changes at least one critical selection for the vast majority of tokens.

### Key Findings from Magnitude Analysis

The magnitude ratio analysis between pre-softmax attention scores and interaction matrix values constitutes the most central quantitative finding of this paper. On the JetClass dataset (with or without PID features), the distribution of this ratio peaks at the $10^4$–$10^5$ order of magnitude, and cases where the ratio falls below 1 (i.e., the interaction matrix dominates) are extremely rare. This implies that for the vast majority of particle pairs, the softmax input is almost entirely determined by the conventional attention term, with the interaction matrix contributing negligibly in numerical terms.

On the Top Landscape dataset, however, the ratio distribution is more dispersed, with the interaction matrix and attention term of comparable magnitude. This is consistent with the observation that attention distributions on Top Landscape do not exhibit binarization—when two input terms are of similar magnitude, the softmax output is naturally "softer," as no single term can completely suppress the others.

### Key Findings

- **Confirmation of the source of sparsity**: Through magnitude ratio analysis, this paper definitively demonstrates that ParT's binary sparse attention is primarily generated by the attention mechanism itself, with the interaction matrix being numerically negligible. This refutes the intuitive conjecture that "sparsity might originate from physical priors."
- **PID features are not the cause of sparsity**: JetClass kinematic-only still exhibits binary distributions, while Top Landscape—which uses the same type of features—does not, indicating that sparsity is associated with dataset scale and task complexity rather than specific input features. JetClass has 100M training samples and 10 classes, while Top Landscape has only 1.2M samples and 2 classes; larger training scale and more complex classification tasks may drive the model to learn more extreme sparsity patterns.
- **The butterfly effect of the interaction matrix**: Although the absolute magnitude of the interaction matrix is far smaller than the attention term (by a factor of $10^{-4}$ to $10^{-5}$), it alters the argmax particle selection for at least one step in 85.4% of tokens. This reveals an interesting property of the softmax function: in highly sparse attention distributions, even a tiny bias can determine the "winner" at the margin of competition, since the attention differences among multiple candidates may be small.
- **Automatic discovery of physical structure**: ParT correctly identifies the lepton in semi-leptonic decays in the $\eta$-$\phi$ plane even without PID information, demonstrating that the model automatically learns an implicit representation of particle type from purely kinematic features. This is highly significant for physicists—the model is not only "getting it right" but also "getting it right for the right reasons."

## Highlights & Insights

- **The pre-softmax magnitude ratio analysis is an elegant interpretability tool.** In any architecture where multiple additive input terms pass through a softmax (e.g., various bias-augmented attention mechanisms), this type of magnitude ratio analysis can be directly applied to quickly assess the actual contribution of each input term. The method is simple, quantitative, and reproducible, and more reliable than directly interpreting attention maps. It can be straightforwardly borrowed for analyzing the practical impact of relative position encodings, ALiBi, and other attention biases in future work.
- **The interaction-dependent computation metric cleverly bridges the seemingly contradictory observations of "negligible magnitude" and "substantial impact."** By tracking whether the argmax changes rather than comparing numerical magnitudes, the authors identify the correct level of granularity for analysis. The design of this metric embodies a profound insight: in highly sparse attention distributions, what matters is not the numerical magnitude but the competitive ranking. This "small but critical" phenomenon may be pervasive in deep learning (e.g., the effects of small perturbations such as dropout and label smoothing on training) and warrants further attention.
- **The contrast between zeroing ablation and standard ablation reveals co-adaptation among model components.** Training ParT plain from scratch without the interaction matrix costs only 1.2% in accuracy, yet zeroing after training causes a 45.6% loss, demonstrating that the weights of different components have mutually adapted to one another. This has important implications for model compression and pruning: one cannot simply conclude that a component is safely removable because its weights are small in magnitude; retraining or fine-tuning is necessary.
- **The cross-dataset difference in sparsity is an intriguing finding.** JetClass (large-scale, multi-class) produces binary attention, while Top Landscape (small-scale, binary classification) does not. This hints that sparsity may be a spontaneous "information compression" strategy that models develop when confronted with complex multi-class tasks—simplifying computation and representation by having each particle attend to only the single most important partner. If this hypothesis holds, similar sparsification trends might be observable in large-scale multi-task Transformers in NLP and computer vision.

## Limitations & Future Work

- **Single model architecture**: All analyses are based on a single model (ParT) and a single set of pretrained weights (pretrained for JetClass, trained from scratch for Top Landscape and Quark-Gluon). It remains unverified whether the conclusions are robust across different model sizes and training hyperparameter settings. In particular, the evolution of sparsity across training epochs is not tracked—whether sparsity emerges early in training or gradually converges to an extreme state remains an open question.
- **Absence of causal explanation**: Although this paper successfully identifies the "source" of sparsity (the attention mechanism itself rather than the interaction), it does not explain the "cause"—what drives attention scores to converge toward extreme values during training. This may relate to the mathematical properties of the softmax function, the gradient dynamics of the training objective, or specific structures in the data distribution, and requires deeper theoretical analysis.
- **The fine-grained mechanism of the interaction matrix remains unrevealed**: This paper demonstrates that the interaction matrix "affects performance by changing the argmax," but does not deeply analyze "in what physical scenarios it changes the argmax" or "what types of particle pairs it guides the model to attend to." Future work could decompose interaction-dependent computation by particle type, $p_T$ range, subjet structure, and other dimensions to reveal what physical information the interaction matrix encodes.
- **Lack of empirical validation of top-k attention**: The authors mention in their outlook that sparse attention could be accelerated using a top-$k$ mechanism, but do not actually implement or test this. Given that attention is already nearly binary (each particle attends to essentially one partner), a top-1 attention approximation should theoretically approximate full softmax attention while substantially reducing computational cost—a high-value engineering direction.
- **Root cause of dataset differences remains unclear**: Why does JetClass produce binary attention while Top Landscape does not? Whether it is dataset scale (100M vs. 1.2M), number of classes (10 vs. 2), the complexity of the physical processes, or other factors is not answered by systematic controlled experiments.

## Related Work & Insights

- **vs ParticleNet (GNN-based approach)**: ParticleNet uses EdgeConv for local neighborhood aggregation, while ParT employs global attention for inter-particle correlation. Mokhtar et al. analyzed edge relevance in ParticleNet using layerwise relevance propagation and found that it can identify the trijet hadronic top decay structure. This paper analyzes ParT from the perspective of attention sparsity, forming an interesting parallel comparison between the two interpretability approaches—the former asks "which edges matter," the latter asks "why is attention sparse."
- **vs standard Vision/NLP Transformers**: Attention in standard Transformers is typically diffuse (especially in shallow layers), whereas ParT exhibits extreme binarization. This may reflect a fundamental difference in input data characteristics—tokens in natural language and images have rich semantic correlations, while correlations among particles in particle physics are more "hard" (a particle either originates from the same decay branch or it does not), naturally driving attention toward discretization. This perspective may transfer to other domains with discrete relational structures, such as molecular graphs and social networks.
- **vs Wang et al. (2024)**: Prior work observed sparse attention and performed $\eta$-$\phi$ visualization but did not systematically analyze its origin. This paper builds upon that foundation by adding magnitude ratio analysis, cross-dataset comparisons, and quantitative interaction-dependence analysis, elevating qualitative observations to quantitative conclusions. The present work can be regarded as a natural and important extension of Wang et al.
- **Relation to the attention-as-explanation debate**: The authors explicitly cite Jain & Wallace (2019)'s warning that "attention is not explanation," acknowledging that attention maps provide only a local view of model behavior. This honest self-positioning is commendable—the strength of this paper lies in analyzing not only what attention "looks like" but also "why it looks that way" and "the relative contributions of each component," going a step deeper than pure attention visualization.

## Rating

- **Novelty**: ⭐⭐⭐ The research question is novel (why is attention in ParT sparse), but the analytical methods are relatively standard (magnitude comparison, ablation, visualization); no new model or algorithm is proposed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The systematic comparison across three datasets and two feature configurations is fairly comprehensive, and the interaction-dependence metric is cleverly designed; however, controlled experiments explaining why sparsity differs across datasets and empirical validation of top-k attention are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and the logical chain is complete, progressing layer by layer from phenomenological observation to causal analysis to ablation validation; figure quality is high. However, the paper is somewhat brief overall, with insufficient elaboration on certain key details (e.g., a more thorough discussion of the Top Landscape vs. JetClass differences).
- **Value**: ⭐⭐⭐⭐ The paper has direct value for the high-energy physics ML community by clarifying ParT's internal mechanisms and pointing to a direction for architectural simplification (top-k attention). It also offers broader ML insights—the interaction-dependence analysis methodology and the finding that "small in magnitude but large in impact" have cross-domain transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)

</div>

<!-- RELATED:END -->
