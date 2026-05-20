---
title: >-
  [Paper Note] No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding
description: >-
  [AAAI 2026][LLM Pretraining][Game Theory] This paper proposes the Embedding CFR algorithm, which maps information sets in imperfect-information games to a continuous low-dimensional embedding space (rather than discrete…
tags:
  - "AAAI 2026"
  - "LLM Pretraining"
  - "Game Theory"
  - "CFR"
  - "Information Set Abstraction"
  - "Embedding Space"
  - "Poker AI"
date: 2026-05-08
content_hash: 46bdef9ea641617f
---

# No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding

**Conference**: AAAI 2026
**arXiv**: [2511.12083](https://arxiv.org/abs/2511.12083)  
**Code**: [https://github.com/PhilEnchan/EmbeddingCFR](https://github.com/PhilEnchan/EmbeddingCFR)  
**Area**: LLM Pre-training
**Keywords**: Game Theory, CFR, Information Set Abstraction, Embedding Space, Poker AI

## TL;DR

This paper proposes the Embedding CFR algorithm, which maps information sets in imperfect-information games to a continuous low-dimensional embedding space (rather than discrete clusters), achieving faster exploitability convergence and higher-quality strategy solving under the same space budget.

## Background & Motivation

### State of the Field

**Background**: In large-scale imperfect-information games such as No-Limit Texas Hold'em, the number of information sets is enormous ($\approx 10^{13}$), making it computationally infeasible to directly apply CFR (Counterfactual Regret Minimization) to find Nash equilibria. The dominant approach follows an "abstract–solve–translate" paradigm: similar information sets are clustered into discrete equivalence classes, the game is solved on the abstracted version, and the resulting strategy is mapped back to the original game.

**Limitations of Prior Work**: Hard clustering introduces a binary dilemma. For example, after the flop, both 9♠A♠ and T♠A♠ may form pairs, but the latter can additionally form straights and flushes while the former cannot. Placing them in the same cluster discards critical distinctions, while separating them forfeits the benefit of shared strategy learning. Such discrete classification irreversibly loses subtle differences between information sets.

**Key Insight**: Inspired by word embeddings in NLP—where continuous vectors capture semantic relationships among words—each information set can analogously be mapped to a probability distribution vector in a low-dimensional continuous space, preserving strategic correlations among similar information sets while retaining fine-grained distinctions.

## Method

### Overall Architecture

Embedding CFR proceeds in two stages: (1) pre-training an embedding network (HandEbdNet) that maps each hand to an $m$-dimensional probability distribution vector (embedding coordinates); and (2) running CFR-like regret accumulation and strategy updates in the embedding space, with the final strategy mapped back to the original space via the embedding matrix.

### Key Designs

1. **Extended Information Set Abstraction**

    - Introduces the concept of an "advisor": each info-block is associated with $m$ advisors, each maintaining a unified strategy.
    - The strategy for information set $I_q$ is given by a weighted aggregation of all advisor strategies: $\sigma(I_q, a) = \sum_p \phi_{p,q} \cdot \sigma(e_p, a)$
    - The weights $\phi_{p,q}$ are the embedding coordinates, forming the embedding matrix $\Phi$.
    - Space complexity is reduced from $O(n \cdot |A|)$ to $O(m \cdot |A|)$ (where $m \ll n$).
    - Unified relationship with traditional abstraction: when each information set trusts only one advisor ($\phi_{p,q} \in \{0,1\}$), the method degenerates to conventional discrete clustering.

2. **Regret-Driven Mechanism in Embedding Space**

    - Instantaneous counterfactual regrets in the original space are projected into the embedding space via $\Phi$: $r^T(\mathbf{E}, a) = \Phi \cdot r^T(\mathbf{J}, a)$
    - Advisor strategies are updated using regret matching in the embedding space.
    - Strategies are mapped back to the original space via $\Phi^\top$: $\sigma_i^T(\mathbf{J}, a) = \Phi^\top \sigma_i^T(\mathbf{E}, a)$
    - Sampling mechanism: only a subset of information sets is activated per iteration, maintaining an actual space complexity of $O(m)$.

3. **HandEbdNet Embedding Network**

    - Input: hand tensor (encoding suit, rank, and street).
    - Output: hand strength tensor (win-rate distributions across streets).
    - Architecture: hand feature encoder → $m$-dimensional vector → softmax → probability distribution as embedding coordinates.
    - Core Insight: hands with similar strengths naturally obtain similar embedding coordinates, allowing them to mutually influence each other's strategy updates.
    - Training: pre-trained with hand strength (win-rate distributions) as supervision; parameters are fixed after training and used as a query module during strategy solving.

### Loss & Training

HandEbdNet is trained using hand strength (win-rate distributions) as the supervision signal. No additional loss function is used during strategy solving; convergence is driven by the regret minimization mechanism. Theoretical analysis demonstrates that when individual advisors act independently, the cumulative regret in the embedding space satisfies a decreasing property.

## Key Experimental Results

### Main Results (Numeral211 Hold'em)

| Method | Exploitability after 1024 iterations (mb/g) | Characteristics |
|--------|---------------------------------------------|-----------------|
| EHS | ~Higher | Classic expected hand strength clustering |
| PaEmd (used in Libratus) | ~Higher | Potential-aware clustering, prior SOTA |
| KrwEmd | ~Higher | History-trajectory-based clustering |
| **Embedding CFR** | **Significantly lowest** | Strategy solving in continuous embedding space |

### Key Findings

- **Under the same space budget, Embedding CFR achieves significantly faster exploitability convergence than all clustering methods** (EHS, PaEmd, and KrwEmd perform comparably but are clearly inferior to Embedding CFR).
- Although the three clustering methods differ in design, they exhibit similar performance at the same abstraction granularity—indicating that the bottleneck lies in the discrete clustering paradigm itself, not in the specific clustering algorithm.
- The probabilistic representation of embedding coordinates allows strategies of similar hands to naturally align without requiring hard boundaries.
- After 1024 iterations, Embedding CFR achieves approximately 60–70% of the exploitability of clustering methods, with the gap continuing to widen as iterations increase.
- With the sampling-and-query mechanism at $m=50$ embedding dimensions, the actual space cost per iteration is only $O(50 \cdot |A|)$, equivalent to 50 buckets in clustering methods but with significantly higher strategy quality.

## Highlights & Insights

- **Transfer of NLP word embedding ideas to game theory**: Just as word embeddings resolve the problem of discrete word representations, Embedding CFR addresses the discrete clustering problem for information sets—an elegant analogy.
- **Sampling-and-query mechanism maintains low space complexity**: Rather than storing the full embedding matrix, the method dynamically generates embedding coordinates via a neural network, elegantly resolving the storage bottleneck in large-scale games.
- **Info-block partitioning by non-chance actions**: The perfect recall property of poker is exploited so that info-blocks sharing the same non-chance action sequence can reuse a single embedding matrix.

## Limitations & Future Work

- The convergence analysis only proves regret reduction when individual advisors act independently; convergence guarantees under mutual coupling of multiple advisors remain an open problem.
- Experiments are conducted only on Numeral211 Hold'em (a simplified variant) and have not been extended to full-scale No-Limit Texas Hold'em.
- The effect of the embedding dimension $m$ on performance has not been systematically ablated.
- HandEbdNet relies on domain knowledge (hand strength) as supervision; whether unsupervised embedding learning is feasible remains an open question.
- The method has only been validated in two-player zero-sum games; applicability to multiplayer non-zero-sum games is unknown.
- Whether the sparsity of embedding coordinates can be exploited, and an information-theoretic analysis of the optimal embedding dimension, are valuable directions for future work.

## Related Work & Insights

- **vs. PaEmd (Libratus)**: PaEmd is the prior SOTA clustering method used in Libratus, but is limited by information loss from discrete clustering; Embedding CFR achieves substantially higher strategy quality under equivalent resource constraints.
- **vs. Deep CFR (Brown 2019)**: Deep CFR uses neural networks to estimate regret values but requires interaction with the environment to learn; Embedding CFR leverages pre-trained embeddings and domain knowledge, making it more efficient in settings with a known model.
- **vs. RL-CFR**: RL-CFR applies reinforcement learning for online action abstraction; Embedding CFR performs information set abstraction—the two approaches are complementary.
- Insight: The embedding approach can be generalized to other game scenarios requiring abstraction, such as auction design and security games.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing embedding ideas into game-theoretic information set abstraction represents a genuinely new paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ Validation is limited to a simplified poker variant; extension to full-scale games is absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and illustrations are clear and intuitive.
- Value: ⭐⭐⭐⭐ Opens a new direction for solving large-scale imperfect-information games.

## Additional Notes
- The methodology and experimental design of this work provide useful reference for related fields.
- Future work can validate the generalizability and scalability of the method across more scenarios and at larger scales.
- Integration with recent related work (e.g., cross-disciplinary combinations with RL, MCTS, and multimodal methods) holds potential research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](../../ACL2026/llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](../../ICLR2026/llm_pretraining/deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)

</div>

<!-- RELATED:END -->
