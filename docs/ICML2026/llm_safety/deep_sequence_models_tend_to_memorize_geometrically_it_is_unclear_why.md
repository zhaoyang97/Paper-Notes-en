---
title: >-
  [Paper Note] Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why
description: >-
  [ICML 2026][LLM Safety][Node2Vec] This paper demonstrates that when Transformers and Mamba models memorize graph edges, they do not merely degrade into lookup tables (associative memory). Instead, they spontaneously organize node embeddings into a "geometric memory" that encodes multi-hop global structures. Through path-star experiments, it is shown th
tags:
  - ICML 2026
  - LLM Safety
  - Node2Vec
  - Transformer
date: 2026-05-08
content_hash: 8e6d04c002752afd
---
# Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why

**Conference**: ICML 2026  
**arXiv**: [2510.26745](https://arxiv.org/abs/2510.26745)  
**Code**: https://github.com/shahriarnz14/geometric_memory  
**Area**: Interpretability / Representation Learning Theory  
**Keywords**: Geometric Memory, Associative Memory, Implicit Reasoning, Spectral Bias, Node2Vec, Transformer

## TL;DR
This paper demonstrates that when Transformers and Mamba models memorize graph edges, they do not merely degrade into lookup tables (associative memory). Instead, they spontaneously organize node embeddings into a "geometric memory" that encodes multi-hop global structures. Through path-star experiments, it is shown that this geometry makes implicit reasoning anomalously easy. However, its emergence cannot be attributed to supervision, capacity, or optimization pressure, leaving a new "memorization puzzle."

## Background & Motivation
**Background**: The mainstream abstraction for explaining Transformer parametric memory is "associative memory": each token receives a near-random/orthogonal embedding $\Phi(u)$, and facts $(u, v)$ are written into a weight matrix $W_{\text{assoc}}$, exposing logits $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$. This is essentially a transcription of the adjacency matrix under a set of random bases. This abstraction elegantly explains n-gram statistics and key-value memory slots and has been adopted by default in many recent works on theory and mechanistic interpretability.

**Limitations of Prior Work**: Associative memory only exposes "locality." To perform $\ell$-hop reasoning, lookup operations must be compounded $\ell$ times, which theoretically requires $\exp(\ell)$ samples or compute to learn without intermediate supervision. However, increasing experimental phenomena conflict with this narrative—multi-hop prediction on paths, two-hop grokking, and how knowledge editing affects the whole system are not behaviors characteristic of "lookup tables."

**Key Challenge**: The story that "models are just lookup tables" and the story that "models can perform implicit multi-hop reasoning" cannot both be true. Either the models are not doing true multi-hop reasoning (shortcuts in the data), or they are not storing lookup tables at all.

**Goal**: (1) Construct an extremely clean scenario where implicit reasoning cannot be explained by any shortcuts to definitively prove models perform multi-hop reasoning; (2) Provide a new memory abstraction capable of accommodating this behavior; (3) Explain, or at least honestly admit, that current learning theory cannot explain why this abstraction emerges.

**Key Insight**: The authors move the reasoning task "into the weights"—the graph is not provided in-context but forced to be memorized into parameters. Then, gradients are calculated only for the "first hop," blocking all possible "cheating spaces" such as implicit curricula, chain supervision, or train-test path overlap.

**Core Idea**: Parametric memory is abstracted into two competing data structures—Associative Memory (direct storage of the adjacency matrix) vs. Geometric Memory (low-rank spectral factorization of the adjacency matrix $\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$). Experiments repeatedly confirm that "even though associative memory is more efficient and easier to find, gradient descent consistently selects geometric memory."

## Method

### Overall Architecture
This paper does not propose a new algorithm but rather a new interpretive framework for memory, rigorously validated through controlled experiments. The authors use a path-star graph as a sandbox: $d$ disjoint chains of length $\ell$ originate from a root node. The training data mixes "edge memorization" (input a node to predict its neighbor, covering all edges) and "path finding" (input a leaf to output the entire root→leaf path, trained on 75% of leaves). Testing is done on the remaining 25% of leaves never seen end-to-end. The design aims to block all shortcuts that could reduce multi-hop problems to single hops, testing if the model can predict "which way to go for the first hop from a leaf"—an action that, in the associative memory view, should require $\ell$ compounded lookups.

### Key Designs

**1. In-weights path-star sandbox: Stripping away "cheating space" to force true implicit reasoning**

Prior evidence of in-weights reasoning often involves small graphs (<200 nodes), only 2 hops, or overlaps between training and test paths. The authors argue these fail to falsify the "lookup table" hypothesis. Thus, they remove all crutches: (i) baking the graph into weights rather than the context; (ii) symmetrizing edges to avoid the reversal curse; (iii) unifying all path lengths to $\ell$ to remove implicit curricula; (iv) making training/test leaf paths disjoint to prevent substring stitching; (v) cutting all gradients except for the first token. In this extremely clean setting, even with graphs scaled to $5\times10^4$ nodes, Transformers and Mamba still learn "which child to select as the first token." This behavior, under near-random orthogonal embeddings $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$, would require $\Omega(\exp(\ell))$ search steps—the model's success after removing all crutches establishes the puzzle.

**2. Geometric vs. Associative Memory Abstraction: Explaining why "exponentially hard" becomes "one-step" via spectral factorization**

To explain how in-weights reasoning succeeds, the authors rewrite storage as $f(u)[v]=\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$, which is a (typically low-rank) factorization of the adjacency matrix $A$, where $\Phi_{\text{geom}}$ aligns with the top eigenvectors of the Graph Laplacian $A-D$. Under this abstraction, embeddings of nodes on the same path converge toward a common direction $\mathbf{z}_i$. Consequently, "finding the first hop from a leaf" collapses into "picking the neighbor of the root with the highest cosine similarity," reducing an $\ell$-hop problem to 1 hop. Using cosine similarity heatmaps (clear diagonals for leaf × first-hop) and 3D embedding visualizations, this geometric structure is reproduced in Transformers, Mamba, and even a 3-layer MLP. This explains how "task success" and "theoretical exponential difficulty" can coexist: the model isn't performing $\ell$ lookups; it's taking 1 step in a pre-organized geometric space.

**3. Memorization puzzle: Debunking three classic explanations to leave an honest open question**

Since associative memory is more efficient and easier to find, why does gradient descent pick geometry? The authors systematically debunk three common explanations. Supervision pressure: Removing path-finding supervision and leaving only edge memorization still results in geometric diagonals in the heatmaps. Capacity/Regularization pressure: Making the model extremely wide (enough parameters for pure associative storage) and disabling dropout/weight decay still results in geometry; conversely, freezing embeddings allows pure associative memory, showing the architecture doesn't forbid it—geometry is "actively" chosen. Optimization bias: Experiments show gradient descent constructs an associative lookup table $\sum_{(u,v)\in E}\Phi(u)\otimes\Phi(v)$ in 2 steps, while geometry takes 100 steps—associative is "closer and easier to find." Furthermore, on sparse graphs, the bit/norm complexity is nearly equal, so "preference for simpler solutions" does not explain geometry. After this triple negation, the authors list "why geometry is chosen" as a genuine open question, prompting future work to look deeper into flatness, spectral norms, or implicit rank minimization induced by depth.

### Loss & Training
Standard next-token cross-entropy is used throughout, without extra regularization, chain-of-thought supervision, or curricula. Ablations vary only three things: frozen vs. trainable embeddings, presence of weight decay/dropout, and presence of path-finding supervision. Minimal theoretical analysis is performed on a 2-layer weight-tied Node2Vec: with dynamics $\dot V(t)=\eta\, C(t)\, V(t)$, empirical evidence shows $V$ column vectors converge to Fiedler-like eigenvectors of the Laplacian, while the null space of $C$ absorbs these vectors, leading to self-stabilization. This demonstrates that spectral bias emerges naturally under cross-entropy + no bottleneck + 1-hop supervision, challenging traditional Node2Vec theories based on squared-error or explicit bottlenecks (Levy & Goldberg).

## Key Experimental Results

### Main Results
On the in-weights path-star task, both architectures (Transformer GPT-mid, Mamba) complete first-token prediction far above random baselines. In contrast, they fail when the same graph is provided as an in-context path task, highlighting that "geometry only grows when memorized into weights."

| Setting | Graph Size / Path Length | 1st Token Accuracy | Notes |
|---------|-------------------------|--------------------|-------|
| In-weights, full path supervision | $5\times 10^4$ nodes, $\ell=6\sim10$ | ≈ 100% (Far > $1/d$ baseline) | Disjoint train/test paths |
| In-weights, 1st token loss only | Same as above, 75% paths for training | ≈ 100% | Success despite removal of intermediate supervision |
| In-context path-star (B&N'24) | Fails even on small graphs | Near $1/d$ | Provided as counter-example |
| Frozen-embedding in-weights | Same as in-weights setting | Drops to baseline | Associative memory learned, task fails |

### Ablation Study

| Configuration | Phenomenon | Implication |
|---------------|------------|-------------|
| Full model (Edge mem + Path sup) | Strong diagonal heatmap + Geometric embeddings | Default state of geometric memory |
| No path supervision (Edge only) | Geometry still emerges (Fig. 5), stronger in Mamba | Supervision is not the cause of geometry |
| Wide model, no dropout / weight decay | Geometry still emerges | Explicit capacity/regularization is not the cause |
| Frozen embedding, others same | Associative memory learnable, reasoning fails | Architecture can express association; geometry is "actively" selected |
| Optimization timeline | Association in 2 steps, Geometry in 100 steps | Association strictly easier-to-find; optimization bias is reversed |
| 2-layer weight-tied Node2Vec | Top Fiedler-like directions dominate rank | Spectral bias emerges naturally without bottlenecks |

### Key Findings
- When a model can express both association and geometry, geometry eventually wins, although associative memory forms one to two orders of magnitude faster. This debunk's the "simpler solution" explanation for gradient descent.
- The geometry learned by Node2Vec is cleaner than that of Transformers (Fig. 1). Ours proposes a "spectral bias contaminated by associative memory" hypothesis: Transformer embeddings are spectral solutions mixed with local associative noise, suggesting headroom for practical improvement by increasing geometricity.
- Weight-untied or multi-layer models display "zigzag geometry"—adjacent node embeddings have opposite signs—corresponding to negative eigenvectors of the adjacency matrix. This is interpreted as the model "using negative directions to cancel diagonal self-loop logits," which disappears when adding self-loops, aligning with recent findings that identity additions improve two-hop reasoning.
- "Self-stabilizing" dynamics observed in 2-layer Node2Vec (embeddings shrink toward Fiedler directions while the null space of $C$ absorbs them) show geometry emerges without bottlenecks/regularization, conflicting with traditional theories by Levy & Goldberg.
- Different graph topologies (path-star / cycle / grid / irregular) produce different Fiedler-like directions, but embeddings consistently fall on the first 2-3 non-degenerate eigenvectors, demonstrating the universality of "low-rank bias" under cross-entropy loss.

## Highlights & Insights
- Changing the default abstraction of "memorized facts" from "lookup tables" to "embedding geometry" seems like a simple coordinate shift, but it has rigid downstream implications: knowledge editing triggers "representation shattering," unlearning is hard to localize, and "hallucinated associations" can appear out of nowhere. These are unpredictable if memory is seen as isolated key-value pairs but are logical side effects in a geometric view.
- The experimental design is the real breakthrough: path-star plus "first-token loss, disjoint paths, fixed lengths" establishes "implicit multi-hop reasoning" so firmly it can serve as a standard benchmark for future work distinguishing "in-weights reasoning" from "substring stitching."
- The observation that "association is strictly easier-to-find, yet geometry is chosen" strips away the "simpler solution" cliché, forcing the theoretical community to seek answers in finer complexity measures (flatness, spectral norm) or to acknowledge depth and factorization as the true drivers.

## Limitations & Future Work
- Nearly all conclusions are based on toy graphs (path-star / cycle / grid). Ours uses the argument that known geometries in language/arithmetic tasks (superposition, linear representations, world models) are scaled versions of these toy geometries, but this is not directly replicated in large-scale LLMs.
- The proof of "self-stabilizing" spectral bias is only closed-form for 2-layer weight-tied Node2Vec. How association and geometry compete in deep models remains a qualitative analysis.
- The conclusion that "geometric memory grants Transformers stronger global reasoning" implies that generative retrieval might outperform dual encoders in retrieval-style applications, but this is not verified here; similarly, the prediction that "stronger geometry implies more fragile knowledge editing" lacks experimental data.
- The "memorization puzzle" leaves a specific open question: under cross-entropy + depth + factorization + no bottlenecks, is there a max-margin style implicit bias that derives Fiedler convergence? Currently, even the convergence rate for 2-layer Node2Vec lacks a closed-form solution.

## Related Work & Insights
- **vs. Khona et al. 2024 / Wang et al. 2024 (Implicit reasoning toy experiments)**: Their path tasks used path length diversity (curricula), full path supervision, and train-test overlaps. Ours removes these crutches and still succeeds, elevating the strength of evidence for "geometry."
- **vs. Saxe et al. (Factorization theory in deep networks)**: Traditional theory uses squared-error and assumes bottlenecks/early stopping. Ours proves cross-entropy naturally produces spectral bias without those conditions, generalizing Node2Vec theory.
- **vs. Classical Hopfield-style Associative Memory (Bietti, Sukhbaatar, Geva, etc.)**: Ours does not deny the utility of associative memory in explaining disjoint facts but argues that when latent structure exists between facts, the associative abstraction systematically misleads our intuition on capacity, scaling laws, unlearning, and editing.
- **vs. Huang et al. 2024 (Two-hop reasoning emergence)**: They attribute two-hop emergence to the factorization of attention K/Q matrices. Ours generalizes this to a universal "geometric memory + spectral bias" mechanism appearing even without attention (Mamba, MLPs).
- **vs. Grokking Literature (Nanda, Power, etc.)**: Grokking is often "memorization followed by generalization." Ours rewrites this phase transition as "associative storage followed by geometric reorganization," providing a minimal, visualizable, and analytical grokking sandbox.
- **vs. Platonic Representation Hypothesis (Huh et al.)**: Ours provides a potential microscopic mechanism for why representations across models look alike—spectral bias pushes models toward the same Laplacian eigenspaces under next-token cross-entropy.
- **vs. Nichani et al. 2024 (Associative memory capacity)**: They provide existence theorems for $m^2$ associations in $m^2$ parameters. Ours acknowledges this but asks "why models don't use it," shifting the direction of capacity analysis.

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
