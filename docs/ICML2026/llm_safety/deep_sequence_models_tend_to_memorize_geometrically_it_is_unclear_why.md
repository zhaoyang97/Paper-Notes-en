---
title: >-
  [Paper Note] Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why
description: >-
  [ICML 2026][LLM Safety][Geometric Memory] Ours points out that Transformers and Mamba do not merely degenerate into lookup tables (associative memory) when memorizing graph edges. Instead…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Geometric Memory"
  - "Associative Memory"
  - "Implicit Reasoning"
  - "Spectral Bias"
  - "Node2Vec"
  - "Transformer"
date: 2026-05-08
content_hash: 30b86a3ec506f523
---

# Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why

**Conference**: ICML 2026  
**arXiv**: [2510.26745](https://arxiv.org/abs/2510.26745)  
**Code**: https://github.com/shahriarnz14/geometric_memory  
**Area**: Interpretability / Theory of Representation Learning  
**Keywords**: Geometric Memory, Associative Memory, Implicit Reasoning, Spectral Bias, Node2Vec, Transformer

## TL;DR
Ours points out that Transformers and Mamba do not merely degenerate into lookup tables (associative memory) when memorizing graph edges. Instead, they spontaneously organize node embeddings into a "geometric memory" that encodes multi-hop global structures. Through path-star experiments, it is proven that this geometry makes implicit reasoning abnormally easy, yet its emergence cannot be attributed to supervision, capacity, or optimization pressure, leaving a new "memorization puzzle."

## Background & Motivation
**Background**: The current mainstream abstraction for explaining Transformer parametric memory is "associative memory": each token receives a nearly random/orthogonal embedding $\Phi(u)$, and facts $(u, v)$ are written into a weight matrix $W_{\text{assoc}}$, exposing logits $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$. This is essentially a transcription of the adjacency matrix under a set of random bases. This abstraction elegantly explains n-gram statistics and key-value memory slots and has been widely adopted in recent theoretical and mechanistic interpretability work.

**Limitations of Prior Work**: Associative memory only exposes "local" information. To perform $\ell$-hop reasoning, lookup operations must be compounded $\ell$ times, theoretically requiring $\exp(\ell)$ samples or computation to learn without intermediate supervision. However, an increasing number of experimental phenomena conflict with this narrative—multi-hop prediction on paths, two-hop grokking, and the "ripple effect" in knowledge editing do not resemble the behavior of a "lookup table."

**Key Challenge**: The story that "models are just lookup tables" and the story that "models can perform implicit multi-hop reasoning" cannot both be true. Either the models are not doing true multi-hop reasoning (data has shortcuts), or they are not storing a lookup table at all.

**Goal**: (1) Construct an extremely clean scenario where implicit reasoning cannot be explained by any shortcuts to confirm that "models indeed perform multi-hop reasoning"; (2) provide a new memory abstraction capable of accommodating this behavior; (3) explain or at least honestly admit that existing learning theories cannot explain why this abstraction emerges.

**Key Insight**: The authors move the reasoning task into the weights—the graph is not provided in-context but forces the model to memorize all edges within the parameters. Gradient computation is then limited to only the "first hop," blocking all possible "cheating spaces" such as implicit curricula, chained supervision, or train-test path overlaps.

**Core Idea**: Abstract parametric memory into two competing data structures: associative memory (direct storage of the adjacency matrix) vs. geometric memory (low-rank spectral factorization of the adjacency matrix $\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$). Experiments repeatedly confirm that "even though associative memory is more efficient and easier to find, gradient descent consistently picks geometric memory."

## Method
Ours does not propose a new algorithm but rather a new interpretive framework solidified by a series of controlled experiments. The methodology consists of three parts: cleaning the phenomenon, defining the abstraction, and systematically excluding alternative explanations for "why this happens."

### Overall Architecture
The authors use a path-star graph as a sandbox: $d$ disjoint chains of length $\ell$ originate from a root node. Training data mixes two types of samples: edge memorization (input a node, predict its neighbor, covering all edges) + pathfinding (input a leaf, output the full path from root to leaf, trained on only 75% of leaves). Testing is conducted on the remaining 25% of unseen end-to-end leaves. The key design is that even if all losses except for the first token are removed, train/test paths are disjoint, and the graph is scaled to $5\times10^4$ nodes, Transformers and Mamba still predict "which neighbor to move to from this leaf" with nearly 100% accuracy. This first-token prediction, under the associative memory view, would require $\ell$ composite lookups.

### Key Designs

1.  **In-weights path-star sandbox**:
    - **Function**: Construct a pure experimental field where "the model must rely on the graph memorized in parameters for implicit multi-hop reasoning," excluding all shortcuts that reduce multi-hop problems to single-hop ones.
    - **Mechanism**: (i) Burn graphs into weights instead of context; (ii) symmetrize edges to avoid the reversal curse; (iii) unify path lengths to $\ell$ to remove implicit curricula; (iv) ensure train/test leaf paths are disjoint to prevent substring stitching; (v) remove all gradients except for the first token. The result is that models still learn "which child to select as the first token" on massive graphs, a behavior requiring $\Omega(\exp(\ell))$ steps of search under nearly random orthogonal embeddings like $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$.
    - **Design Motivation**: Previous evidence for in-weights reasoning involved graphs that were too small (<200 nodes), limited to 2 hops, or had overlapping train/test paths. The authors argue these are insufficient to falsify the "lookup table" hypothesis. By removing all fallback options and seeing the model still succeed, the puzzle is firmly established.

2.  **Geometric Memory vs. Associative Memory Dual Abstraction**:
    - **Function**: Provide a unified explanatory abstraction for why in-weights reasoning succeeds and compare it with classic associative memory.
    - **Mechanism**: Rewrite the storage method as $f(u)[v]=\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$, i.e., a (typically low-rank) factorization of the adjacency matrix $A$, where $\Phi_{\text{geom}}$ aligns with the top eigenvectors of the graph Laplacian $A-D$. In this abstraction, embeddings of nodes on the same path converge to a common direction $\mathbf{z}_i$. Finding the first hop from a leaf degenerates into "picking the neighbor of the root with the largest cosine similarity," reducing an $\ell$-hop problem to 1 hop. The paper demonstrates this geometric structure in Transformer, Mamba, and a 3-layer MLP via cosine similarity heatmaps and 3D embedding visualizations.
    - **Design Motivation**: The authors argue that "models are not actually performing $\ell$ lookups, but taking 1 step in a pre-organized geometric space." Only this new abstraction allows "success" and "theoretical exponential difficulty" to coexist.

3.  **Memorization puzzle: Excluding three classes of classic explanations**:
    - **Function**: Honestly present "why geometric memory wins" as an open question by systematically debunking common explanations: supervision pressure, capacity/regularization pressure, and optimization bias.
    - **Mechanism**: (i) Supervision pressure: Removing pathfinding supervision and keeping only edge memorization still manifests geometric diagonals in heatmaps; (ii) Capacity pressure: In very wide models (with enough parameters for associative storage of $A$) and with dropout/weight decay disabled, geometry still emerges; conversely, freezing embeddings allows the model to easily learn pure associative memory, proving the architecture does not forbid it; (iii) Optimization bias: Experiments show gradient descent constructs an associative lookup table $\sum_{(u,v)\in E}\Phi(u)\otimes\Phi(v)$ in 2 steps, while geometry takes 100 steps—implying associative memory is "closer and easier to find." Furthermore, on sparse graphs, associative and geometric memory have nearly identical bit/norm complexity.
    - **Design Motivation**: Rather than adding another narrative, the authors block all candidate narratives, forcing subsequent work to look deeper (e.g., flatness, spectral norm, implicit rank minimization induced by depth).

### Loss & Training
Standard next-token cross-entropy is used throughout, without extra regularization, chain-of-thought supervision, or curricula. Controlled experiments only vary three things: whether embeddings are frozen, whether weight decay/dropout is kept, and whether pathfinding supervision is provided. A minimal theoretical analysis is performed on a 2-layer weight-tied Node2Vec: dynamics $\dot V(t)=\eta\, C(t)\, V(t)$. Ours empirically shows $V$ column vectors gradually converge to Laplacian Fiedler-like eigenvectors, while the null space of $C$ absorbs these vectors, causing gradient self-stabilitzation. This spectral bias emerges naturally under cross-entropy + no bottleneck + 1-hop supervision, challenging traditional Node2Vec theories based on squared-error or explicit bottlenecks.

## Key Experimental Results

### Main Results
On the in-weights path-star task, both architectures (Transformer GPT-mid, Mamba) complete first-token prediction far above random baselines. When the same graph is switched to an in-context path task, they both fail, highlighting that "geometry only grows when memorized into weights."

| Setup | Graph Scale / Path Length | First Token Accuracy | Remarks |
| :--- | :--- | :--- | :--- |
| In-weights, full path supervision | $5\times 10^4$ nodes, $\ell=6\sim10$ | ≈ 100% (far above $1/d$) | Disjoint train/test paths |
| In-weights, first token loss only | Same as above, 75% paths for training | ≈ 100% | Success despite zero intermediate supervision |
| In-context path-star (B&N'24) | Fails even on small graphs | Near $1/d$ | Counter-example control |
| Frozen embedding in-weights | Same as in-weights setup | Drops to baseline | Degenerates to associative; task fails |

### Ablation Study
| Configuration | Phenomenon | Meaning |
| :--- | :--- | :--- |
| Full model (Edge + Path supervision) | Strong diagonal heatmap + geometric embeddings | Default state of geometric memory |
| No path supervision, edge only | Geometry still emerges (Fig. 5); stronger in Mamba | Supervision pressure is not the cause |
| Wide model, no dropout / weight decay | Geometry still emerges | Explicit capacity/reg pressure is not the cause |
| Frozen embeddings, others constant | Associative memory learned, path reasoning fails | Architecture can express associative; geometry is "actively" chosen |
| Optimization timeline | Associative in 2 steps, geometry in 100 steps | Associative is strictly easier-to-find |
| 2-layer weight-tied Node2Vec | Top Fiedler-like directions dominate rank | Spectral bias emerges naturally without bottleneck |

### Key Findings
- When a model can express both associative and geometric memory, geometry eventually wins, even though associative memory forms one to two orders of magnitude faster. This directly debunk the "gradient descent prefers simpler solutions" explanation.
- Geometry learned by Node2Vec is much cleaner than that of Transformers (Fig. 1, col 3 vs col 2). The authors suggest a "spectral bias contaminated by associative memory" hypothesis: Transformer embeddings are versions of Node2Vec spectral solutions with some local lookup noise, suggesting improving geometricity is a practical direction for model improvement.
- Weight-untied or multi-layer models exhibit "zigzag geometry"—adjacent node embeddings have opposite signs, while two-hop neighbors are closer—corresponding to negative eigenvectors of the adjacency matrix. This is interpreted as the model "using negative directions to offset self-loop logits" and disappears when self-loops are added, aligning with recent phenomena regarding "adding identity to improve two-hop reasoning."
- "Self-stabilizing" dynamics observed in 2-layer Node2Vec (embeddings shrinking towards Fiedler directions while $C$'s null space absorbs them) suggest geometry does not require any bottleneck/regularization to emerge, conflicting directly with Levy & Goldberg’s traditional theory.
- Across different graph topologies (path-star / cycle / grid / irregular), embeddings automatically fall onto the first 2~3 non-degenerate eigenvectors, demonstrating the universality of "low-rank bias" under cross-entropy loss.

## Highlights & Insights
- Replacing the default abstraction of "model memorizing facts" from a "lookup table" to "embedding geometry" leads to significant downstream implications: knowledge editing triggers "representation fragmentation," unlearning is hard to localize, and "hallucinatory associations" can be generated out of nowhere. These side effects are predictable in a geometric view but not when viewing memory as isolated key-values.
- The experimental design is the real "killer move": path-star combined with "first token loss only, disjoint paths, fixed length" pins down the behavior of "implicit multi-hop reasoning" so firmly it can serve as a benchmark for future work distinguishing "in-weights reasoning vs. substring stitching."
- The observation that "associative is strictly easier-to-find, yet the model picks geometry" exposes the "simplicity bias" explanation as a catch-all that fails here, forcing the theory community to either refine complexity measures (flatness, spectral norm) or admit that depth and factorization are the true hidden drivers.

## Limitations & Future Work
- Most conclusions are based on toy graphs (path-star, cycle, grid). The authors argue through extrapolation that known geometries in language/arithmetic tasks (superposition, linear representations, world models) are scaled versions of this toy geometry, but no direct replication on large-scale LMs is provided.
- Proof of "self-stabilization" is closed-form only for 2-layer weight-tied Node2Vec. How associative and geometric memory compete in deep models (learning rates, weight decay, time windows) currently only has qualitative analysis.
- The prediction that "geometric strength leads to fragile knowledge editing" was not experimentally verified.
- The "memorization puzzle" leaves a specific open question: under cross-entropy + depth + factorization + no bottleneck, is there a max-margin style implicit bias that derives Fiedler convergence? Currently, even the convergence rate of 2-layer Node2Vec lacks a closed-form solution.

## Related Work & Insights
- **vs. Khona et al. 2024 / Wang et al. 2024 (Implicit reasoning toy experiments)**: Their path tasks used varying lengths (implicit curriculum), full path supervision, and overlapping train-test paths. Ours removes these, yet first-hop prediction still succeeds, significantly strengthening the evidence for "geometry."
- **vs. Saxe et al. (Factorization theory in deep networks)**: Traditional theory uses squared-error and implies bottlenecks/early stopping. Ours proves cross-entropy naturally produces spectral bias without bottlenecks, generalizing Node2Vec theory.
- **vs. Classic Hopfield-style associative memory (Bietti, Sukhbaatar, Geva, etc.)**: Ours does not negate the associative view for disjoint facts but argues that when facts have latent structure (even if not in supervision), the associative abstraction systematically misleads intuitions about capacity, scaling laws, and knowledge editing.
- **vs. Huang et al. 2024 (Emergence of two-hop reasoning)**: They attribute two-hop reasoning to the factorization of non-merged attention K/Q matrices. Ours generalizes this to a universal "geometric memory + spectral bias" mechanism, showing it emerges even in Mamba or pure MLPs.
- **vs. Grokking literature (Nanda, Power, etc.)**: Grokking is often "memorization then generalization." Ours rewrites this phase transition as "associative storage then geometric reorganization," providing a minimal, visualizable sandbox for grokking.
- **vs. Platonic Representation Hypothesis (Huh et al.)**: Ours identifies a potential micro-mechanism—as long as the objective includes next-token cross-entropy, spectral bias pushes models toward identical Laplacian eigenspaces.
- **vs. Nichani et al. 2024 (Associative memory capacity)**: They provide existence theorems for MLP capacity ($m^2$ associations in $m^2$ parameters). Ours acknowledges this but asks: "Given the capacity exists, why doesn't the model use it?", shifting the direction of capacity analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ICML 2026\] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)
- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](../../ACL2026/llm_safety/why_agents_compromise_safety_under_pressure.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](../../AAAI2026/llm_safety/radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](../../ACL2026/llm_safety/maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)

</div>

<!-- RELATED:END -->
