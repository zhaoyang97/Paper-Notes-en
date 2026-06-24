---
title: >-
  [Paper Note] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks
description: >-
  [ICLR 2026][Learning Theory][fibring] This paper establishes a **precise correspondence** between fibring neural networks (where pre-activations of a parent network are fed into a fibring function to generate weights and inputs for a sub-network, which then injects its output back) and fibring modal logics for the first time. Based on this, it unifies GNNs, GATs, and Transformer encoders as fragments of fibred modal logic formulas and provides their **non-uniform logical expr…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Neuro-symbolic AI (Logical Expressivity)"
  - "fibring"
  - "modal logic"
  - "Kripke models"
  - "GNN"
  - "GAT"
  - "Transformer encoder"
  - "non-uniform expressivity"
  - "neuro-symbolic AI"
date: 2026-05-08
content_hash: a8543f774a1ab87a
---

# From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=P1iAEhonhY](https://openreview.net/forum?id=P1iAEhonhY)  
**Code**: To be confirmed  
**Area**: Learning Theory / Neuro-symbolic AI (Logical Expressivity)  
**Keywords**: fibring, modal logic, Kripke models, GNN, GAT, Transformer encoder, non-uniform expressivity, neuro-symbolic AI  

## TL;DR
This paper establishes a **precise correspondence** between fibring neural networks (where pre-activations of a parent network are fed into a fibring function to generate weights and inputs for a sub-network, which then injects its output back) and fibring modal logics for the first time. Based on this, it unifies GNNs, GATs, and Transformer encoders as fragments of fibred modal logic formulas and provides their **non-uniform logical expressivity** results.

## Background & Motivation
**Background**: A major theme in neuro-symbolic AI is mapping modern neural architectures to logical reasoning for improved interpretability and verifiability. Existing work characterizes the logical expressivity of GNNs (upper-bounded by the Weisfeiler-Leman test, uniform expressivity given by Presburger logic, and bounded GNNs corresponding to first-order logic fragments) and Transformer encoders (UHAT corresponds to AC⁰ fragments, AHAT corresponds to temporal logics like LTL(C,+) with counting modalities). Meanwhile, Bronstein et al. pointed out that a Transformer encoder is essentially a GNN acting on a complete graph with positional encodings and attention.

**Limitations of Prior Work**: Although GNNs and Transformers share similar elements like "counting modalities" in their expressivity characterizations, **no unified theory has covered both architectures simultaneously until now**. On the other hand, although fibring neural networks proposed in 2004 (where parameters of one network become functions of another) were inspired by fibring logic, **the precise correspondence between them and fibring modal logic has never been rigorously established**—this 20-year-old clue remained an informal analogy.

**Key Challenge**: Fibring logic naturally supports "combining a countable family of modal logics into a single fibred language with shared semantics," which is precisely the composition mechanism needed to unify multiple architectures. However, the lack of a formal bridge mapping neural computation to Kripke semantics has prevented this potential from being realized.

**Goal**: To bridge this gap—formalize "fibred models compatible with fibred neural networks," prove that the output of a fibred neural network is **input-wise consistent** with the truth value of a fibred logical formula, and use this to uniformly derive the non-uniform expressivity of GNN/GAT/Transformer. **Core Idea**: Treat neural networks as "compositions of underlying modal logics stacked layer by layer," using fibring as a unifying lens.

## Method

### Overall Architecture
The paper redefines 2004's fibring neural networks as a directed tree where nodes are neural architectures (fibring architecture). Each edge specifies a layer of the parent network and a set of "fibred neuron" positions. On the logical side, a fibred modal language is constructed by assigning a modal operator $\square_{v,\ell}$ to each (node, layer) in the tree, with "compatible fibred models" defined as a subclass forming a valid fibred logic $L_F$. The two sides are aligned via "compatibility conditions": the input-output behavior corresponds to reachability between Kripke worlds, and parent-to-subnetwork delegation corresponds to fibring jumps between worlds. Finally, the computation of GATs (including GNNs as a special case and Transformer encoders as GATs on complete graphs) is rewritten as fibred neural networks to derive their corresponding fibred logic formula families.

```mermaid
graph TD
    A[Parent network N computes to layer ℓ<br/>obtains pre-activations x_i] --> B[Fibring function f̃<br/>generates sub-network instance N_i + input y_i]
    B --> C[Sub-network<br/>fibred computation ⟨N_i,F_i,f̃_i⟩y_i]
    C --> D[Output injected back to parent<br/>replaces components at position set S_i]
    D --> E[Parent network continues to output layer N^ℓk↑]
    F[Logical side: Each v,ℓ assigned modal operator □_v,ℓ] -.Compatibility C0-C2.-> A
    F --> G[Fibred formula ψφ,F = ∧_i □_vi,ℓi ψφ,F_i]
    E -.Truth value consistency Thm 4.5.-> G
```

### Key Designs

**1. Redefinition of fibred neural networks: Organizing "networks generating networks" into a directed tree**. Original fibring only handled two networks; this paper generalizes it to arbitrary numbers and combinations. A fibring architecture is a directed tree where nodes are labeled with architectures $A_v$ and edges with (layer index $\ell$, position set $S$). A fibred network $\tilde N=\langle N,F,\tilde f\rangle$ assigns a fibring function $\tilde f_{(v,v')}$ to each edge, mapping vectors of dimension $\le d_\ell$ to "an instance of sub-architecture $A_{v'}$ + a valid input." Computation proceeds recursively from the root: when reaching a layer with outgoing edges, it pauses to invoke the fibring function, sends part of the current vector to the sub-network, and replaces the parent's components at $S_i$ with the sub-network's output. Formally, each stage produces a quadruple $(x_i,N_i,y_i,h_i)$, where $h_i$ replaces $x_i$ at $S_i$ with the result of the sub-fibred network on $y_i$. The final output is $\tilde N(x)=N^{\ell_k\uparrow}(h_k)$. If the root network outputs a scalar, it is interpreted as a true/false classifier based on whether it is $>0$. This redefinition allows fibring to directly interface with the layer-wise delegation structure of modern deep architectures.

**2. Consistent fibred models: Using admissibility to anchor reachability to network behavior**. The key on the logical side is the admissible mapping $\pi$ in Definition 4.1—it assigns a network input vector to each Kripke world, requiring $\pi$ to be injective and satisfy that world $w$ and $w'$ are reachable in model $m$ $\iff N(\pi(w))=N(\pi(w'))$. That is, **whether worlds are connected is determined entirely by whether the network yields equal outputs for the corresponding inputs**; reachability is no longer arbitrary but a mirror of network semantics. Based on this, Definition 4.2 provides compatibility conditions (C0)–(C2): (C0) the root model maps each world to a vector in $\{0,1\}^n$ where bits represent proposition truth values; (C1) each $\pi_{v,\ell}$ is admissible for the sub-network $N_v^{\ell\uparrow}$; (C2) fibring jumps $f_{v,\ell}$, $f_{v_i,1}$ are aligned one-to-one with the quadruples $(x_i,N_i,y_i,h_i)$ produced by network computation, and proposition sets remain consistent across models.

**3. Consistent model classes as valid fibred logic: Non-emptiness + Isomorphism closure**. To ensure "consistent models" form a usable logic $L_F$, it must be proven that $\mathrm{Comp}_F(v,\ell)$ (the projection of all consistent fibred models onto the $(v,\ell)$ component) satisfies the two conditions of a valid fibred logic (Proposition 4.3). **Non-emptiness** is shown by explicit construction: for every $\tilde N$ and $x\in\{0,1\}^n$, a consistent model is generated by creating a world for each collected vector, connecting them when $N_v^{\ell\uparrow}$ outputs are equal, and recursively defining proposition truth values along the root model. **Isomorphism closure**: if $m\in\mathrm{Comp}_F(v,\ell)$ originates from some consistent $M$ and $\pi:m\cong m'$, then replacing that component of $M$ with $m'$ and $\pi_{v,\ell}$ with $\pi_{v,\ell}\circ\pi^{-1}$ preserves reachability, truth values, and fibring jumps, meaning $m'$ remains consistent. These properties confirm $L_F$ is a reasonable model class that is non-empty and closed under Kripke isomorphism.

**4. Precise Correspondence Theorem: Network classification and formula truth consistent per input**. Definition 4.4 recursively lifts a propositional formula $\varphi$ along the fibring tree into a fibred formula $\psi(\varphi,F)=\bigwedge_i \square_{v_i,l_i}\psi(\varphi,F_i)$ (where $l_i$ is the maximum layer label leaving $v_i$). Theorem 4.5 then proves: for any network instance $N$ of the root architecture, there exists a propositional formula $\varphi$ (taking the characteristic formula of $N$: $\varphi:=\bigvee_{h:N(h)>0}(\bigwedge_{h_k=1}p_k\wedge\bigwedge_{h_k=0}\neg p_k)$), such that **for any input $x$, any fibring function matching $F$, and any consistent fibred model $M$**:
$$M,(\pi_{u,1})^{-1}(x)\models\psi(\varphi,F)\iff \langle N,F,\tilde f\rangle\text{ classifies }x\text{ as True}.$$
The proof uses induction on the depth of $F$: the leaf case is guaranteed by (C0) where $\varphi$ matches $N(h)>0$ on the root model; the induction step utilizes (C1)–(C2) and $\square$ semantics, making the evaluation of $\psi(\varphi,F)$ exactly equivalent to evaluating sub-trees along fibring jumps. Both sides follow the same recursive structure, ensuring the truth value matches the root output $>0$.

**5. Application: Non-uniform characterization of GAT/GNN/Transformer**. Under assumptions like truncated ReLU, local sum aggregation, Boolean inputs, and hard attention, the paper uses GAT as the primary thread (GNN is a GAT without attention; Transformer encoder is a GAT on a complete graph with positional encoding). Theorem 5.1 proves: for every triple $\tau=\langle G,\mathcal G,u\rangle$ (GAT, graph, node), there exists a fibring architecture $F_\tau$, a root network instance $N^\tau$, and a family of fibring functions $\tilde f_x^\tau$ varying with node features $x$, such that the computation of the fibred network $\langle N^\tau,F_\tau, \tilde f_x^\tau\rangle$ on $x_u$ equals $\mathcal G(G,x,u)$. Crucially, the tree structure of $F_\tau$ is exactly the **unraveling tree** of node $u$, which is isomorphic to the recursive aggregation of GAT along the tree depth. **The architecture $F_\tau$ is fixed for a given $(G,u)$, while only the fibring functions vary with $x$**. Theorem 5.2 then invokes the correspondence from 4.5 to obtain a formula $\tilde\varphi_\tau$ in the fibred logic $L_{F_τ}$ that **does not depend on node features**, such that $M,(\pi_{u_\tau,1})^{-1}(x_u)\models\tilde\varphi_\tau\iff\mathcal G(G,x,u)=\text{true}$. This family of formulas characterizes the network instance—yielding non-uniform expressivity. For Transformers, one simply replaces the root $\pi_{u,1}$ with a bijection mapping to $\{pos(t,s),1+pos(t,s)\}^n$ to incorporate positional encoding.

## Key Experimental Results

### Main Results

| Result | Content | Significance |
|------|------|------|
| Prop 4.3 | $\mathrm{Comp}_F(v,\ell)$ is non-empty and closed under Kripke isomorphism | $L_F$ is a valid fibred logic |
| Thm 4.5 | Fibred network True classification $\iff$ Fibred formula $\psi(\varphi,F)$ is true at the corresponding world (for all inputs/fibring functions/consistent models) | **Precise correspondence** between neural computation and logical semantics |
| Thm 5.1 | Input-wise computation of any GAT/GNN/Transformer can be reproduced by a fibred network (fixed architecture, input-dependent fibring functions) | Architecture $\to$ non-uniform description via fibred networks |
| Thm 5.2 | The above network instances can be characterized by a family of formulas in $L_{F_τ}$ (independent of node features) | **Non-uniform logical expressivity** of the three architectures |

### Key Findings
- **The unraveling tree is the geometric core of the correspondence**: The tree structure of the fibring architecture = the unraveling tree of the target node, naturally aligning the recursive aggregation of GAT with the recursive delegation of fibred networks.
- **Formulas depend only on the root network and architecture, not input features**: Consequently, different $x$ values under the same $(G,u)$ share the same formula skeleton, while only the fibring functions (semantic jumps) vary—this is the precise meaning of "non-uniform."
- **Unification of previously independent characterizations**: GNN, GAT, and Transformer encoder are characterized for the first time under the same fibred logic framework, rather than using distinct tools like WL tests, Presburger logic, AC⁰, or LTL.

## Highlights & Insights
- **Closing a 20-year open formalization gap**: The analogy between fibring neural networks and fibring logic from 2004 is finally proven as a precise correspondence. Admissibility and compatibility conditions (C0)–(C2) are the key technical bridges.
- **Unifying three architectures in one framework**: By viewing Transformer encoders as GATs on complete graphs with positional encoding and GNNs as GATs without attention, the unified characterization requires only one main thread with two degenerations.
- **A research agenda for uniform expressivity and interpretability**: The authors argue that the collection of vectors in various components of fibred networks is closely related to the "$\ell$-spectrum" of GNNs. Its finiteness is key to proving uniform expressivity (Presburger formulas), providing a concrete path from non-uniform to uniform unification. Furthermore, "reverse-engineering the fibred logic theory learned by a network" aligns with modular interpretability (e.g., circuit tracing).

## Limitations & Future Work
- **Non-uniform coverage only**: The core result provides input-wise formula families rather than a single formula characterizing the entire architecture instance. Collapsing these families into a single (e.g., Presburger) formula remains open for GATs and Transformer encoders (Transformers currently only have lower bounds).
- **Strong assumptions**: Technical results are limited to truncated ReLU, local sum aggregation, rational coefficients, Boolean inputs, and hard attention. Soft attention, general activations, and continuous features are not yet covered.
- **Purely theoretical**: The "application potential" of the fibring lens in interpretability and formal verification remains speculative, with no experimental validation of extracting readable logical rules from real networks.
- **Future Work**: Using the finiteness of the $\ell$-spectrum to unify fibred formula families into uniform formulas, applying fibring for cross-architecture formal verification (using modal/fibred logic satisfiability complexity results), and "reverse-engineering fibred formulas for typical inputs" to extract interpretable rules.

## Related Work & Insights
- **Fibring logic and neural networks**: Gabbay (1999) and Garcez & Gabbay (2004) are the sources; this paper generalizes the latter and establishes a precise correspondence.
- **GNN Expressivity**: Complements and unifies WL upper bounds (Barceló 2020), non-uniform circuit/descriptive complexity (Grohe 2023), Presburger uniform expressivity (Nunn/Benedikt 2024), and FO fragments for bounded GNNs (Cuenca Grau 2025).
- **Transformer Expressivity**: Incorporates Transformer encoders (UHAT $\to$ AC⁰ by Hao 2022, AHAT $\to$ LTL(C,+) by Barcelo 2024) into the same fibred framework.
- **Interpretability**: Shares the spirit of circuit tracing (Ameisen 2025) which decomposes network computation into modular reasoning steps, inspiring the "reverse-engineering of logical theories."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes a long-sought precise correspondence and unifies GNN/GAT/Transformer under one framework, filling a 20-year theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work; the theorem chain is self-consistent, but the practical value of fibring in interpretability/verification remains at the stage of argumentation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formalization and clear narrative of motivations. Compatibility conditions (C0)–(C2) and admissibility involve heavy notation, posing a barrier to non-logic experts.
- Value: ⭐⭐⭐⭐ Provides a unified logical expressivity lens for neuro-symbolic AI and suggests a clear research agenda toward uniform expressivity and interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)
- [\[ICLR 2026\] Proper Velocity Neural Networks](proper_velocity_neural_networks.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)

</div>

<!-- RELATED:END -->
