---
title: >-
  [Paper Note] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts
description: >-
  [ACL 2025][LLM Efficiency][Mixture-of-Experts] This paper proposes DIVE, a method to reconstruct dense LLMs into MoE architectures. The core insight is that calibration datasets from different domains lead structured pruning to produce distinct pruning candidates, which can be leveraged to build domain-specific experts. Combined with an efficient two-stage retraining strategy (dense router training + sparse expert LoRA training), DIVE outperforms existing pruning and MoE reco…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Structured Pruning"
  - "Dense-to-MoE Conversion"
  - "Expert Diversity"
  - "Domain Affinity Mining"
date: 2026-05-08
content_hash: 5c7885d314128717
---

# DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts

**Conference**: ACL 2025  
**arXiv**: [2506.09351](https://arxiv.org/abs/2506.09351)  
**Code**: [https://github.com/yuchenblah/DIVE](https://github.com/yuchenblah/DIVE)  
**Area**: LLM Efficiency  
**Keywords**: Mixture-of-Experts, Structured Pruning, Dense-to-MoE Conversion, Expert Diversity, Domain Affinity Mining

## TL;DR
This paper proposes DIVE, a method to reconstruct dense LLMs into MoE architectures. The core insight is that calibration datasets from different domains lead structured pruning to produce distinct pruning candidates, which can be leveraged to build domain-specific experts. Combined with an efficient two-stage retraining strategy (dense router training + sparse expert LoRA training), DIVE outperforms existing pruning and MoE reconstruction methods while updating less than 1% of the parameters.

## Background & Motivation

**Background**: MoE architectures enable efficient inference via sparse activation (e.g., Mixtral 8x7B), but pre-training MoE models from scratch is extremely costly. Recently, dense-to-MoE conversion methods have gained interest, reconstructing MoE models from existing dense LLMs to save training budgets significantly.

**Limitations of Prior Work**:
   - Replication approaches (e.g., Sparse Upcycling): Directly clone existing FFNs as multiple experts, resulting in identical initial experts that lack diversity and require substantial retraining to differentiate.
   - Random splitting approaches (e.g., LLaMA-MoE): Randomly partition the intermediate dimensions of FFNs into multiple experts, which fails to guarantee meaningful functional specialization.
   - Both types of approaches neglect expert diversity, leading to redundancy issues.

**Key Challenge**: The key challenge of dense-to-MoE conversion lies in imparting initial diversity to the experts during the construction phase, rather than relying solely on subsequent fine-tuning to achieve specialization.

**Goal**:
   - How to leverage the intrinsic knowledge distribution of a dense LLM to construct MoE experts with initial diversity.
   - How to design an efficient retraining process to rapidly recover the performance of the reconstructed model.

**Key Insight**: The authors observe that structured pruning methods (such as FLAP) are highly sensitive to calibration datasets. A model pruned using math data performs well on math tasks but poorly on others, since different calibration domains produce completely different pruning masks. This phenomenon, typically considered a "drawback" of pruning, can be cleverly utilized to construct diverse experts.

**Core Idea**: Treat the sensitivity of structured pruning to calibration data as a "feature" rather than a "defect," and use calibration data from different domains to prune the same model separately, yielding domain-specific FFN subnetworks as MoE experts.

## Method

### Overall Architecture
Dense LLM $\mathcal{M}$ → **Domain Affinity Mining** (cross-evaluating pruning results across 24 datasets) → **Hierarchical Clustering** (grouping datasets into N domain clusters) → **Pruning-Based Expert Reconstruction** (calibrating with data from each cluster to prune FFNs separately, yielding N experts) → **MoE Assembly** (replacing FFNs with MoE layers + randomly initializing routers) → **Two-Stage Retraining** (Stage 1: dense training for routers, Stage 2: sparse activation LoRA training for experts + LayerNorm) → Final MoE LLM

### Key Designs

1. **Domain Affinity Mining**:

    - **Function**: Discover domain relationships between datasets to provide grouping criteria for expert construction.
    - **Mechanism**:
        - Perform pairwise evaluation on 24 datasets: prune the model with dataset $t_i$ as calibration, evaluate on dataset $t_j$ → obtain a 24×24 PPL matrix.
        - Normalize PPL via $\text{norm}(p)_{i,j} = \frac{\min(p)_{:,j}}{p_{i,j}}$ to eliminate scaling differences among evaluation sets.
        - Compute correlation between datasets using Pearson correlation coefficient.
        - Apply hierarchical clustering to group them into N clusters (e.g., 8 clusters for 8 experts).
    - **Design Motivation**: Calibration data from different domains causes FLAP pruning to preserve different FFN channels—e.g., mathematical data tends to retain math-related channels, while NLI data preserves reasoning-related channels. This sensitivity serves as a natural mechanism to construct diverse experts.
    - **Key Findings**:
        - Calibration sets within similar domains show highly correlated performance (MathQA ↔ GSM8K).
        - Domain affinity is more strongly influenced by data source than task type (QNLI ≈ SQuAD2 > QNLI ≈ ANLI).
        - C4 (general corpora) demonstrates broad generalization when used as a calibration set.

2. **Based on Pruning Expert Reconstruction**:

    - **Function**: Prune the dense FFN into multiple domain-specific, compact FFNs to serve as MoE experts.
    - **Mechanism**:
        - Leverage FLAP to evaluate channel importance based on the variance of hidden state fluctuations: $\mathbf{S}_{:,j}^\ell = \frac{1}{N-1}\sum_{n=1}^{N}(\mathbf{X}_{n,j}^\ell - \bar{\mathbf{X}}_{:,j}^\ell)^2 \cdot \|\mathbf{W}_{:,j}^\ell\|_2^2$
        - For each domain cluster, calibrate with data from that cluster and prune the intermediate dimensions of the FFN (e.g., by 50% or 75%).
        - 8 clusters → 8 pruned FFNs → assembled into MoE layers (equipped with randomly initialized noisy routers).
        - Embeddings, LM heads, and MHA modules remain unchanged except for the FFNs.
    - **Key Parameters**: Termed as $N/K$ models (e.g., 1/8 denotes activating 1 out of 8 experts, while 2/8 denotes activating 2 out of 8 experts).

3. **Efficient Two-Stage Retraining**:

    - **Function**: Recover the performance of the reconstructed model using the minimum number of trainable parameters.
    - **Mechanism**:
        - **Stage 1 (Dense Training)**: All experts are activated while only training the routers (utilizing softmax with a temperature parameter $t$) with a small data size (0.5B tokens).
        - **Stage 2 (Sparse Training)**: Only top-k experts are activated while training experts using LoRA + full tuning of LayerNorm (5B tokens).
        - MHA modules are kept frozen (ablation studies demonstrate retraining is unnecessary).
    - **Design Motivation**: Stage 1 lets the routers learn the assignment first, followed by Stage 2 where the experts are fine-tuned under sparse activation. Less than 1% of the parameters require updating.

### Loss & Training
- Standard language modeling objective (next token prediction).
- Stage 1: Router temperature coefficient is set to $t$=0.05 (for 1/8) or 0.5 (for 2/8) to make the softmax sharper, simulating sparse inference.
- Retraining data: SlimPajama (627B tokens), randomly sampled 5.5B tokens.
- LoRA is applied to the expert FFNs.

## Key Experimental Results

### Main Results (TinyLlama-1.1B, 50% FFN Activation)

| Method | Type | WikiText2 PPL ↓ | LAMBADA PPL ↓ | Average Downstream Acc ↑ |
|------|------|-----------------|---------------|-------------------|
| TinyLlama-1.1B | Dense 100% | — | — | 47.18 |
| LLM-Pruner | Pruning 50% | 17.59 | 56.66 | 39.57 |
| FLAP | Pruning 50% | 14.51 | 33.22 | 41.42 |
| LLaMA-MoE 1/8 | MoE 50%×1 | 19.57 | 87.27 | 39.34 |
| **DIVE 1/8** | **MoE 50%×1** | **13.52** | **24.84** | **42.17** |

DIVE 1/8 achieves 0.99 lower PPL on WikiText2 and 8.38 lower PPL on LAMBADA compared to the runner-up FLAP, while improving downstream task average accuracy by 0.75%.

### Ablation Study

| Configuration | WikiText2 PPL | LAMBADA PPL | ID Task Acc | OOD Task Acc |
|------|---------------|-------------|-----------|------------|
| DIVE 2/8 (Full) | 18.09 | 63.45 | 47.88 | 27.67 |
| w/o Domain Affinity Mining | 20.02 (+1.93) | 76.03 (+12.58) | 47.08 (-0.80) | 26.15 (-1.52) |

| MHA Training | DIVE 1/8 PPL Change | DIVE 2/8 PPL Change |
|---------|-----------------|-----------------|
| Included | baseline | baseline |
| Excluded | <0.29 degradation | 0.37 improvement instead |

### Key Findings
- **Domain Affinity Mining is crucial**: Excluding D.A.M. causes LAMBADA PPL to degenerate by 12.58, demonstrating the value of diverse expert initialization.
- **Highly efficient**: DIVE trained with only 1B tokens outperforms FLAP trained with 5B tokens.
- **Routing distribution validates expert diversity**: Math tokens are correctly routed to math experts, NLI tokens to NLI experts, and C4 tokens uniformly activate all experts.
- MHA modules require no retraining; training only router + expert LoRA + LayerNorm (<1% parameters) is sufficient.
- High-frequency splitting of token types corresponds strongly to domains, indicating that the router has successfully learned the domain assignment.

## Highlights & Insights
- **The "turning weakness into strength" philosophy is highly ingenious**: The sensitivity of structured pruning to calibration data is typically seen as a vulnerability, but DIVE utilizes this exact trait to construct diverse experts. This reverse-thinking approach is highly inspiring—suggesting that a method's limitations can be the solution to another problem.
- **Practical two-stage retraining design**: Training the routers first and then the experts decouples routing learning from expert adaptation. The temperature scaling trick in Stage 1 (enabling dense training to simulate sparse inference) is a valuable technical detail.
- **Systematic Domain Affinity Mining**: Quantifying dataset relationships using a 24×24 cross-evaluation matrix provides an analytical framework that can easily be transferred to multi-task learning or curriculum learning.

## Limitations & Future Work
- Validated only on TinyLlama-1.1B; the performance on larger models (7B+) remains unknown.
- Domain Affinity Mining requires a prefix cost of 24×24 pruning and evaluation runs, limiting immediate out-of-the-box deployment.
- The selection of 24 datasets and the count of 8 experts are hard-coded; how to automatically determine the optimal number of experts and domain partitioning is unresolved.
- Retraining still requires 5B tokens of general corpora, representing a barrier for resource-constrained scenarios.
- The integration of MoE with attention layers (e.g., Switch Attention) was not explored.

## Related Work & Insights
- **vs LLaMA-MoE**: LLaMA-MoE leverages random splitting to build experts, lacking diverse initialization. DIVE consistently outperforms it on both PPL and downstream tasks.
- **vs Sparse Upcycling**: Replication-based approaches start with identical experts, requiring extensive training to split. DIVE introduces diversity right from the start.
- **vs FLAP (pure pruning)**: FLAP cannot utilize the sparse activation advantage of MoE. DIVE combines pruning with MoE, achieving better performance with the same number of activated parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of utilizing pruning sensitivity to construct diverse experts is highly original; the overall framework is logically designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 11 downstream tasks with detailed ablations and routing distribution analyses, though constrained by focusing on a 1.1B model.
- Writing Quality: ⭐⭐⭐⭐ Outstanding figure designs (the heatmap clearly presents domain affinities) with complete algorithmic descriptions.
- Value: ⭐⭐⭐⭐ Offers a highly effective Dense-to-MoE conversion method, yielding an impressive efficiency with <1% of fine-tined parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)
- [\[ICLR 2026\] Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource](../../ICLR2026/llm_efficiency/mixture-of-experts_can_surpass_dense_llms_under_strictly_equal_resource.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)

</div>

<!-- RELATED:END -->
