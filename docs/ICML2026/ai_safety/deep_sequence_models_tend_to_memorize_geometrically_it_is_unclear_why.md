---
title: >-
  [Paper Note] Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why
description: >-
  [ICML 2026][AI Safety][Node2Vec] This paper demonstrates that when Transformer / Mamba models memorize graph edges, they do not simply degenerate into lookup tables (associative memory). Instead, they spontaneously organize node embeddings into a "geometric memory" that encodes multi-hop global structures. Through path-star experiments, the authors pr
tags:
  - ICML 2026
  - AI Safety
  - Node2Vec
  - Transformer
date: 2026-05-08
content_hash: 9142a442c21fe1c2
---
# Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why

**Conference**: ICML 2026  
**arXiv**: [2510.26745](https://arxiv.org/abs/2510.26745)  
**Code**: https://github.com/shahriarnz14/geometric_memory  
**Area**: Interpretability / Theory of Representation Learning  
**Keywords**: Geometric Memory, Associative Memory, Implicit Reasoning, Spectral Bias, Node2Vec, Transformer

## TL;DR
This paper demonstrates that when Transformer / Mamba models memorize graph edges, they do not simply degenerate into lookup tables (associative memory). Instead, they spontaneously organize node embeddings into a "geometric memory" that encodes multi-hop global structures. Through path-star experiments, the authors prove this geometry makes implicit reasoning abnormally easy, yet its emergence cannot be attributed to supervision, capacity, or optimization pressure, leaving a new "memorization puzzle."

## Background & Motivation
**Background**: The mainstream abstraction for interpreting Transformer parametric memory is "associative memory": each token receives a near-random/orthogonal embedding $\Phi(u)$, and facts $(u,v)$ are written into a weight matrix $W_{\text{assoc}}$, exposing logits $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$. Essentially, this is a transcription of the adjacency matrix under a set of random bases. This abstraction elegantly explains n-gram statistics and key-value memory slots and has been adopted by default in many recent works on theory and mechanistic interpretability.

**Limitations of Prior Work**: Associative memory only exposes "local" information. To perform $\ell$-hop reasoning, lookup operations must be compounded $\ell$ times, which theoretically requires $\exp(\ell)$ samples/computation to learn without intermediate supervision. However, increasing experimental evidence contradicts this narrative—multi-hop prediction on paths, two-hop grokking, and the "ripple effect" in knowledge editing do not behave like simple "table lookups."

**Key Challenge**: The story that "models are just lookup tables" and the story that "models can perform implicit multi-hop reasoning" cannot both be true. Either models are not performing real multi-hop reasoning (data has shortcuts), or models are not storing information as lookup tables.

**Goal**: (1) Construct an extremely clean scenario where implicit reasoning cannot be explained by any shortcuts to confirm "models indeed perform multi-hop reasoning"; (2) Provide a new memory abstraction capable of accommodating this behavior; (3) Explain, or at least honestly acknowledge, that current learning theory cannot explain why this abstraction emerges.

**Key Insight**: The authors move the reasoning task into the weights—the graph is not provided in-context but forced to be memorized into parameters. By calculating gradients only for the "first hop," they block all possible "cheating spaces" like implicit curricula, chained supervision, or train-test path overlaps.

**Core Idea**: Abstract parametric memory into two competing data structures—associative memory (direct storage of adjacency matrices) vs. geometric memory (low-rank spectral factorization of the adjacency matrix $\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$). Experiments repeatedly confirm that "even though associative memory is more efficient and easier to find, gradient descent consistently chooses geometric memory."

## Method

### Overall Architecture
The paper proposes a new memory interpretation framework rather than a new algorithm and validates it through controlled experiments. The authors use a path-star graph as a sandbox: $d$ disjoint chains of length $\ell$ originate from a root node. The training data mixes "edge memorization" (input a node to predict its neighbor, covering all edges) and "pathfinding" (input a leaf to output the entire root→leaf path, trained on 75% of leaves). Testing is done on the remaining 25% of end-to-end unseen leaves. The goal is to block all shortcuts that reduce multi-hop problems to single-hop and see if the model can still predict the "first step from the leaf"—a behavior that requires $\ell$ compound lookups in the associative memory view.

### Key Designs

**1. In-weights path-star sandbox: Eliminating "cheating space" to force true implicit reasoning**

Prior evidence of in-weights reasoning often involved small graphs (<200 nodes), only 2 hops, or overlaps between training and testing paths. The authors argue these are insufficient to disprove the "lookup table" hypothesis. Consequently, they remove all shortcuts: (i) burning the graph into weights rather than context; (ii) symmetrizing edges to avoid the reversal curse; (iii) unifying path lengths to $\ell$ to remove implicit curricula; (iv) using disjoint leaf paths for train/test to prevent substring stitching; (v) cutting all gradients except for the first token. Under this clean setting, even with $5\times10^4$ nodes, Transformer and Mamba still learn to "select the correct child for the first token." This behavior, requiring $\Omega(\exp(\ell))$ steps under random orthogonal embeddings $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$, creates a distinct puzzle.

**2. Geometric vs. Associative Memory: Explaining why "exponentially hard" becomes "1-step" via spectral factorization**

To explain how in-weights reasoning succeeds, the authors rewrite storage as $f(u)[v]=\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$, which is a (typically low-rank) factorization of the adjacency matrix $A$, where $\Phi_{\text{geom}}$ aligns with the top eigenvectors of the graph Laplacian $A-D$. In this abstraction, node embeddings on the same path converge toward a common direction $\mathbf{z}_i$. Thus, "finding the first hop from a leaf" reduces to "picking the neighbor of the root with the highest cosine similarity," collapsing an $\ell$-hop problem into 1 hop. Using cosine similarity heatmaps (showing clear diagonals for leaf×first-hop) and 3D embedding visualizations, they reproduce this geometric structure in Transformers, Mamba, and even a 3-layer MLP. This explains the contradiction: the model is not performing $\ell$ lookups but taking 1 step in a pre-organized geometric space.

**3. Memorization puzzle: Systematically refuting three classical explanations**

Since associative memory is more efficient and easier to find, why does gradient descent pick geometry? The authors block three common explanations: Supervision pressure: Removing pathfinding supervision (leaving only edge memorization) still results in geometric diagonals in heatmaps. Capacity/Regularization pressure: Even with very wide models (enough parameters for pure associative storage) and disabling dropout/weight decay, geometry still emerges; conversely, freezing embeddings allows the model to learn pure associative memory easily, showing the architecture doesn't forbid it. Optimization bias: Gradient descent finds an associative lookup table $\sum_{(u,v)\in E}\Phi(u)\otimes\Phi(v)$ in 2 steps, whereas geometry takes 100 steps—associative memory is "closer and easier to find." Furthermore, on sparse graphs, the bit/norm complexity of both is nearly equal, so "simplicity bias" cannot explain it. The authors honestly present this as an open problem, challenging future work to look deeper (flatness, spectral norms, implicit rank minimization).

### Loss & Training
Standard next-token cross-entropy is used throughout, without extra regularization, chain-of-thought supervision, or curricula. Ablations vary three factors: freezing embeddings, weight decay/dropout, and pathfinding supervision. Theoretical analysis is conducted on a 2-layer weight-tied Node2Vec with dynamics $\dot V(t)=\eta\, C(t)\, V(t)$. Empirical results show $V$ column vectors converge to Fiedler-like eigenvectors of the Laplacian, while the null space of $C$ absorbs these vectors, leading to "self-stabilization." This proves spectral bias can emerge naturally under cross-entropy with 1-hop supervision and no bottleneck, challenging traditional Node2Vec theories based on squared-error or explicit bottlenecks.

## Key Experimental Results

### Main Results
On the in-weights path-star task, both Transformer (GPT-mid) and Mamba architectures successfully predict the first token far above random baselines. When the same graph is presented as an in-context path task, they fail, highlighting that "geometry only grows when memorized into weights."

| Setting | Graph Scale / Path Length | First Token Accuracy | Remarks |
|------|-------------------|-------------------|------|
| In-weights, full path supervision | $5\times 10^4$ nodes, $\ell=6\sim10$ | ≈ 100% (far above $1/d$) | Disjoint train/test paths |
| In-weights, first token loss only | Same as above, 75% paths for training | ≈ 100% | Success despite zero intermediate supervision |
| In-context path-star (B&N'24) | Fails even on small graphs | Near $1/d$ | Counter-example control |
| Frozen embedding in-weights | Same as in-weights setting | Drops to baseline | Degenerates to associative; task fails |

### Ablation Study

| Configuration | Phenomenon | Meaning |
|------|------|------|
| Full Model (Edge Mem + Path Super.) | Strong diagonal heatmap + geometric embeddings | Default state of geometric memory |
| No path sup., only edge mem | Geometry still appears (Fig. 5), Mamba signal stronger | Supervision is not the cause of geometry |
| No dropout / weight decay, wide model | Geometry still appears | Explicit capacity/reg is not the cause |
| Freeze embedding, others same | Associative memory learned, reasoning fails | Architecture can express associative; geometry is "chosen" |
| Optimization timeline | Associative in 2 steps, Geometry in 100 steps | Associative is strictly easier-to-find |
| 2-layer weight-tied Node2Vec | Top Fiedler-like directions dominate rank | Spectral bias emerges without a bottleneck |

### Key Findings
- When models can express both associative and geometric memory, geometry eventually wins, even though associative memory forms one to two orders of magnitude faster. This refutes the standard "simplicity bias" explanation.
- Node2Vec learns much cleaner geometry than Transformers (Fig. 1). The authors propose a "spectral bias contaminated by associative memory" hypothesis: Transformer embeddings are spectral solutions "polluted" by local associative memory, suggesting that increasing geometricity is a practical direction for improvement.
- Weight-untied or multi-layer models exhibit "zigzag geometry"—adjacent node embeddings have opposite signs. This is interpreted as the model using negative directions to cancel self-loop logits on the diagonal; adding self-loops makes this disappear.
- The "self-stabilization" dynamics observed in 2-layer Node2Vec (embeddings shrinking toward Fiedler directions while $C$ swallows them in its null space) show that geometry doesn't require bottlenecks or regularization.
- For different graph topologies (path-star, cycle, grid, irregular), embeddings automatically fall onto the first 2-3 non-degenerate eigenvectors, demonstrating the universality of "low-rank bias" under cross-entropy loss.

## Highlights & Insights
- Shifting the default abstraction of "memorizing facts" from "lookup tables" to "embedding geometry" yields significant downstream implications: knowledge editing triggers "representation fractures," unlearning is hard to localize, and "hallucinated associations" may emerge. These are predictable under the geometric view but not the isolated key-value view.
- The experimental design is a major contribution: path-star with "first-token-only loss, disjoint paths, and fixed lengths" rigorously confirms implicit multi-hop reasoning. This provides a standard benchmark for "in-weights reasoning vs. substring stitching."
- The observation that "associative is strictly easier-to-find, yet the model chooses geometry" deconstructs the "simplicity bias" catch-all, forcing researchers to consider deeper mechanisms like flatness, spectral norms, or implicit rank minimization induced by depth.

## Limitations & Future Work
- Most conclusions are based on toy graphs (path-star, cycle, grid). The authors extrapolate by arguing that known geometries in language tasks (superposition, linear representations) are scaled-up versions of these toy geometries, but this hasn't been directly replicated in large LMs.
- The "self-stabilization" proof is closed-form only for 2-layer weight-tied Node2Vec. The competition between associative and geometric memory in deep models (timing, learning rate, weight decay effects) remains qualitative.
- While geometric memory allows stronger global reasoning, implying generative retrieval might outperform dual encoders, this was not verified. Similarly, the prediction that "stronger geometry leads to more fragile knowledge editing" lacks experimental validation.
- The "memorization puzzle" leaves a specific open question: under cross-entropy and depth, does a max-margin style implicit bias exist that derives Fiedler convergence? Even the convergence rate of 2-layer Node2Vec lacks a closed-form solution.

## Related Work & Insights
- **vs. Khona et al. 2024 / Wang et al. 2024**: Their path tasks used varying lengths (implicit curricula) and full path supervision. By removing these, this paper provides much stronger evidence for "geometry."
- **vs. Saxe et al. (Factorization in Deep Nets)**: Traditional theories use squared-error and assume bottlenecks. This work proves cross-entropy generates spectral bias naturally without those assumptions.
- **vs. Hopfield-style Associative Memory**: This paper doesn't deny the associative view's validity for disjoint facts but argues that when latent structures exist, the associative abstraction systematically misleads our intuition on scaling laws and knowledge editing.
- **vs. Huang et al. 2024 (Two-hop reasoning emergence)**: They attribute two-hop success to attention matrix factorization; this work generalizes it to a broader "geometric memory + spectral bias" mechanism applicable even to non-attention models (Mamba, MLP).
- **vs. Grokking (Nanda, Power et al.)**: Grokking is often "memorization then generalization." This paper reframes it as "associative storage then geometric reorganization," providing a visualizable sandbox for grokking.
- **vs. Platonic Representation Hypothesis (Huh et al.)**: Provides a microscopic mechanism: if training involves next-token cross-entropy, spectral bias pushes various models toward the same Laplacian characteristic subspace.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometrically Constrained Outlier Synthesis](geometrically_constrained_outlier_synthesis.md)
- [\[ICML 2026\] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](../../NeurIPS2025/ai_safety/factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)

</div>

<!-- RELATED:END -->
