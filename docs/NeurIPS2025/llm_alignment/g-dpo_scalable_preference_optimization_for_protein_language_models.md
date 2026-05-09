---
title: >-
  [Paper Note] g-DPO: Scalable Preference Optimization for Protein Language Models
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] To address the quadratic growth of preference pairs with respect to sample size when applying DPO to protein language models (PLMs)—which renders training intractable—this paper proposes g-DPO: (1) redundant preference pairs are pruned via union-mask-based clustering in sequence space, retaining more informative comparisons within local neighborhoods; (2) grouped likelihood amortization via shared union masks enables computation of log-likelihoods for all sequences within a group in a single forward pass. Across three protein engineering tasks, g-DPO achieves statistically indistinguishable in silico and in vitro performance compared to standard DPO, while delivering 1.7–5.4× training speedups.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - DPO
  - Protein Language Models
  - Preference Optimization
  - Scalability
  - Mutational Landscape
date: 2026-05-08
content_hash: bf310ded7c224f25
---

# g-DPO: Scalable Preference Optimization for Protein Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.19474](https://arxiv.org/abs/2510.19474)
**Code**: Available
**Area**: LLM Alignment / Protein Engineering
**Keywords**: DPO, Protein Language Models, Preference Optimization, Scalability, Mutational Landscape

## TL;DR
To address the quadratic growth of preference pairs with respect to sample size when applying DPO to protein language models (PLMs)—which renders training intractable—this paper proposes g-DPO: (1) redundant preference pairs are pruned via union-mask-based clustering in sequence space, retaining more informative comparisons within local neighborhoods; (2) grouped likelihood amortization via shared union masks enables computation of log-likelihoods for all sequences within a group in a single forward pass. Across three protein engineering tasks, g-DPO achieves statistically indistinguishable in silico and in vitro performance compared to standard DPO, while delivering 1.7–5.4× training speedups.

## Background & Motivation

**Background**: Protein language models (PLMs, e.g., ESM-2) are pretrained on large-scale sequence corpora via self-supervised learning, implicitly encoding signals about protein structure and function. Direct Preference Optimization (DPO), which eliminates the need for a separate reward model, is a natural fit for protein engineering—experimental data inherently define preference relations (e.g., higher thermostability is preferred, lower toxicity is preferred), enabling direct construction of preference pairs for DPO training.

**Limitations of Prior Work**: Protein DPO faces a scalability bottleneck absent in NLP. In NLP, preference structure arises naturally (multiple outputs from the same prompt, with annotators selecting the best), whereas protein datasets provide only scalar labels (e.g., thermostability values), necessitating exhaustive enumeration of all sequence pairs to construct preferences. $n$ sequences yield $O(n^2)$ preference pairs, rendering training time unacceptable even for moderately sized datasets (a few hundred sequences). Existing mitigation strategies include: (a) threshold-based partitioning (binary good/bad split, discarding fine-grained ranking information); (b) random sampling of preference pairs (risking the loss of informative training signal); and (c) rank-based sampling (preserving ordering but not reducing computational cost).

**Key Challenge**: The tension between the number of preference pairs and training efficiency—more pairs provide richer training signal but incur quadratic computational cost; naive reduction risks discarding comparisons that are genuinely informative in sequence space. A key insight is that the locality of protein mutations implies that a large fraction of preference pairs are in fact redundant.

**Goal**: To make DPO scalable to large-scale protein mutational landscape optimization without sacrificing optimization quality. This decomposes into: (a) how to identify and prune redundant preference pairs; (b) how to reduce the per-pair computational cost; and (c) how to preserve training effectiveness after pruning.

**Key Insight**: The unique structure of protein sequences provides a natural handle for optimization—mutations typically alter only a small number of positions, so sequences within the same cluster share the majority of tokens. This enables a shared union mask to compute the likelihoods of multiple sequences in a single forward pass—a strategy that does not hold in NLP (where different sentences share almost no tokens), but is highly applicable to proteins given the locality of mutations.

**Core Idea**: Exploiting the locality of protein mutations, g-DPO prunes redundant preference pairs via sequence-space clustering and amortizes likelihood computation through union-mask grouping, achieving up to 5.4× training speedup with no performance degradation.

## Method

### Overall Architecture
g-DPO training proceeds in three stages: (1) **Evo-tuning**—unsupervised fine-tuning of the PLM (ESM-2-650M) on evolutionarily related sequences of the wild-type protein to provide evolutionary context; (2) **Union mask clustering**—greedy agglomerative clustering of the experimental mutant dataset based on shared mutation positions; (3) **Grouped DPO training**—groups of size $g$ are uniformly sampled from each cluster, with a single forward pass computing the likelihoods of all sequences in the group and evaluating DPO losses over all intra-group preference pairs.

### Key Designs

1. **Union Mask Clustering**:

    - *Function*: Groups protein sequences by shared mutation positions such that sequences within a group differ only at a small number of sites.
    - *Mechanism*: The union mask is defined as $M(S) = \{i \in [L] : \exists j,k \text{ s.t. } s_i^{(j)} \neq s_i^{(k)}\}$, i.e., the set of all differing positions across a group of sequences. Greedy agglomerative clustering is applied, where the merge cost is $\phi(C_i, C_j) = m(C_i \cup C_j) - m(C_i)$ (the incremental increase in union mask size upon merging). Clusters with minimal additional divergence are merged preferentially. Termination occurs when $\min_{i \neq j} \phi(C_i, C_j) > \tau L$, where $\tau$ controls the maximum fraction of the sequence that the union mask may span.
    - *Design Motivation*: Unlike NLP, where outputs from different prompts differ almost entirely, protein mutants differ at only a few positions. Exploiting this locality brings sequence-space neighbors together—intra-cluster comparisons are more informative (capturing non-additive mutational effects), while inter-cluster comparisons, separated by many mutations, carry signal drowned out by excessive divergence.

2. **Grouped Likelihood Amortization (Group Sampling)**:

    - *Function*: Computes pseudo log-likelihoods for multiple sequences within a group using a single forward pass.
    - *Mechanism*: For $g$ sequences in a group, a shared union mask $D$ is constructed—masking all differing positions yields a common input $y_{\setminus D}$. A single forward pass produces logits for all positions, from which the log probability at differing positions is read for each sequence individually: $\log p(y_w) \approx \sum_{i \in D} \log p({y_w}_i | y_{\setminus D})$. Since non-differing positions have identical logits across sequences, only the differing positions need to be evaluated. At $g=4$, one forward pass yields signals for $\binom{4}{2}=6$ preference pairs, whereas standard DPO would require 6 forward passes.
    - *Design Motivation*: This constitutes a mean-field approximation—assuming that, conditioned on shared tokens, the differing positions are independent. When the number of differing positions is small relative to the total sequence length, the approximation error is negligible. Union mask clustering ensures this condition holds by enforcing that the intra-group fraction of differing positions does not exceed the threshold $\tau$.

3. **Synergistic Two-Stage Efficiency Gains**:

    - *Function*: Clustering reduces the number of preference pairs; grouping reduces the per-pair cost.
    - *Mechanism*: Clustering prunes cross-cluster preference pairs (which carry low information) by setting the threshold $\tau$, substantially reducing the $O(n^2)$ pair count; grouping via union mask sharing reduces the per-pair forward pass cost from 2 to $2/g$. The two effects multiply.
    - *Design Motivation*: Using grouping alone—without clustering—causes the mutation span to become too large, leading to approximation collapse (confirmed by ablation studies). Using clustering alone reduces the number of pairs but not the per-pair cost. Only their combination simultaneously ensures quality and efficiency.

### Loss & Training
- Standard DPO loss: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(y_w,y_l)} [\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$
- Backbone: ESM-2-650M, first evo-tuned then trained with g-DPO.
- All sequences appear in at least one group per epoch.
- Training conducted on a single NVIDIA A100 GPU.

## Key Experimental Results

### Main Results (Three Protein Engineering Tasks)

| Dataset | Functional Objective | # Sequences | Position Coverage | DPO Spearman ρ | g-DPO Spearman ρ | Speedup |
|--------|---------|--------|-----------|----------------|-------------------|--------|
| Anti-SARS-CoV-2 VHH | Thermostability | 462 | 47.1% | ~0.55 | ~0.55 | 1.7× |
| Trastuzumab scFv | Expression Level | 76 | 13.1% | ~0.50 | ~0.50 | — |
| DhaA Haloalkane Dehalogenase | Thermostability | 474 | 40.3% | ~0.52 | ~0.52 | 5.4× |

- KS tests confirm that the predicted property distributions of sequences generated by DPO and g-DPO are statistically indistinguishable (D statistic < 0.03).
- In vitro validation (DhaA thermostability + Trastuzumab expression level): experimentally measured distributions of sequences designed by both methods are consistent.

### Ablation Study

| Configuration | Spearman ρ | # Training Pairs | Convergence Speed | Notes |
|------|-----------|---------|---------|------|
| Standard DPO (g=2) | Baseline | $O(n^2)$ | Baseline | Exhaustive preference pairs |
| Clustering only (τ=0.3, g=2) | ≈ Baseline | Substantially reduced | Faster | Prunes redundant pairs, no quality loss |
| Grouping only (no cluster, g=4) | Degraded | Unchanged | Faster but lower quality | Large mutation span causes likelihood approximation collapse |
| Clustering + Grouping (τ=0.3, g=4) | ≈ Baseline | Substantially reduced | **Maximum speedup** | Best trade-off |
| Over-clustering (τ<0.3, g=2) | Degraded | Excessively reduced | Fast but lower quality | Discards valuable training signal |

### Key Findings
- **The clustering threshold τ≈0.3 is the critical point for quality preservation**: performance degrades significantly below this value, indicating that approximately 70% of preference pairs are redundant and can be safely pruned.
- **Grouping alone is not viable**: without clustering, g=4 directly degrades performance, as sharing a union mask across sequences that differ at many positions introduces excessive likelihood approximation error.
- **Speedup scales with dataset size**: the largest speedup is observed on the VHH L dataset (1,833 sequences), consistent with the theoretical expectation that redundant pairs grow with $n$.
- **In vitro results are consistent**: experimental validation on DhaA and Trastuzumab shows no significant difference between sequences designed by DPO and g-DPO—the speedup does not come at the expense of experimental efficacy.

## Highlights & Insights
- **Leveraging structural priors of protein sequences to accelerate training**—the core insight is that the locality of protein mutations renders a large fraction of preference pairs redundant, a property that does not hold in NLP. This paradigm of "data structure → computational optimization" generalizes to any sequence optimization problem with local structure (e.g., DNA sequence design, chemical reaction optimization).
- **Elegant design of union mask clustering**—using the size of the union of mutated positions as the clustering distance metric directly ties the clustering strategy to the error bound of the likelihood approximation (fewer differences → more accurate mean-field approximation), achieving theoretical coupling between the clustering and likelihood computation strategies.
- **Closed-loop in vitro validation**—beyond in silico metrics (Spearman correlation, KS tests), the paper validates DhaA thermostability and Trastuzumab expression level in actual laboratory experiments, representing a higher standard of validation than is typical in protein ML papers.

## Limitations & Future Work
- **Limited experimental scale**: all three datasets are of moderate size (76–474 sequences); while the theoretical speedup grows with dataset size, this has not been fully validated at large scale (>1,000 variants).
- **Restricted to masked language models**: the current method relies on pseudo log-likelihood (requiring per-position masking) and is not directly applicable to autoregressive PLMs.
- **Clustering hyperparameter τ requires tuning**: the optimal τ depends on the mutational landscape structure of the specific dataset, and no automated selection strategy is provided.
- **Listwise ranking objectives unexplored**: the method still uses a pairwise DPO loss despite full ranking information being available within each group—incorporating listwise ranking losses could further improve efficiency.
- **Unimodal limitation**: only protein sequences are considered; structural or functional modalities are not incorporated.

## Related Work & Insights
- **vs. Standard DPO (Rafailov et al., NeurIPS 2023)**: Standard DPO is designed for NLP, assuming preference data is provided by human annotators. g-DPO addresses the scalability challenge unique to protein applications—constructing preference pairs from scalar labels—which is a distinctive challenge of applying DPO to biology.
- **vs. CtrlProt (Liu et al., AAAI 2025)**: CtrlProt employs rank-wise preference objectives to improve controllability in multi-objective optimization but does not address computational efficiency. g-DPO focuses on efficiency; the two approaches are complementary—g-DPO's clustering strategy could be integrated with CtrlProt's multi-objective framework.
- **vs. Widatalla et al.**: Their finding that random sampling of preference pairs across rank levels yields comparable performance (with minimal difference across gap levels) implies redundancy. g-DPO approaches pruning from sequence space rather than rank space, which is more practically grounded.
- The work provides direct guidance for research employing DPO/RLHF in protein engineering—confirming that not all preference pairs are equally valuable, and that comparisons within local neighborhoods are more informative.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegantly combines the local mutation structure of protein sequences with DPO computational optimization; union mask clustering design is particularly well-crafted.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three-task in silico + two-task in vitro validation with comprehensive ablation studies; closed-loop experimental verification.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly derived, progressing logically from motivation to design to experiments.
- Value: ⭐⭐⭐⭐ Directly advances the practical application of DPO in protein engineering; accelerated training makes large-scale mutational landscape optimization tractable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[NeurIPS 2025\] On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)

<!-- RELATED:END -->
