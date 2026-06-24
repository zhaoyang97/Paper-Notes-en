---
title: >-
  [Paper Note] On The Geometry and Topology of Representations: the Manifolds of Modular Addition
description: >-
  [ICLR 2026][Interpretability][Modular Addition] This paper adopts a perspective of "viewing a whole cluster of neurons with the same frequency as a manifold," proving that various networks previously thought to have learned "completely different circuits (Clock vs. Pizza)" actually learn the **same class of torus/vector-addition disk manifolds** in the first layer. This is statistically validated across hundreds of networks using closed-form formulas and Topological Data Anal…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Modular Addition"
  - "Mechanistic Interpretability"
  - "Universality Hypothesis"
  - "Topological Data Analysis"
  - "Representational Manifolds"
date: 2026-05-08
content_hash: 2d35b39b7c0d001b
---

# On The Geometry and Topology of Representations: the Manifolds of Modular Addition

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2olkCiSELH](https://openreview.net/forum?id=2olkCiSELH)  
**Code**: To be confirmed  
**Area**: Interpretability / Mechanistic Interpretability / Representational Geometry  
**Keywords**: Modular Addition, Mechanistic Interpretability, Universality Hypothesis, Topological Data Analysis, Representational Manifolds

## TL;DR
This paper adopts a perspective of "viewing a whole cluster of neurons with the same frequency as a manifold," proving that various networks previously thought to have learned "completely different circuits (Clock vs. Pizza)" actually learn the **same class of torus/vector-addition disk manifolds** in the first layer. This is statistically validated across hundreds of networks using closed-form formulas and Topological Data Analysis (TDA), thereby repairing the "Universality Hypothesis" previously challenged by Zhong et al. (2023).

## Background & Motivation
**Background**: Mechanistic interpretability aims to decompose neural networks into understandable "circuits," relying on two pillar hypotheses: the **Universality Hypothesis** (networks with similar structures and data learn similar circuits) and the **Manifold Hypothesis** (representation learning essentially finds low-dimensional manifolds for data). Modular addition $(a+b)\bmod n = c$ has become a standard testbed for toy interpretability due to its nature as cyclic group multiplication, linear non-separability, and existing deep research.

**Limitations of Prior Work**: Zhong et al. (2023) interpolated between "uniform attention $\leftrightarrow$ learnable attention" in the transformer from Nanda et al. (2023), claiming that networks learn two **disjoint** circuits—uniform attention learns Pizza (vector addition), while learnable attention learns Clock (angular summation)—and provided two metrics, distance irrelevance and gradient symmetricity, to distinguish them. This essentially provided a counterexample to the Universality Hypothesis: same data, same task, but different architectures learning circuits with "zero commonality."

**Key Challenge**: If this counterexample holds, the consequences are severe—it implies that large models might simultaneously learn a vast number of disjoint circuits for the same task within their weights, making the task of "identifying generalizable interpretable principles" combinatorially hopeless. The question is: Are Clock and Pizza truly two fundamentally different algorithms, or just different projections of the same structure artificially divided by a specific set of metrics?

**Goal**: (1) Use closed-form mathematics to characterize what manifolds these networks actually learn in the first layer; (2) prove that architectures like Clock, Pizza, and MLP are topologically and geometrically equivalent; (3) provide computational tools for statistical validation across thousands of networks.

**Key Insight**: Instead of explaining individual neurons or weights as in previous work, the authors **cluster all neurons belonging to the same learned representation (the same key frequency $f$) and treat them as a single collective entity**. The set of pre-activation points of this neuron cluster constitutes a **manifold**, allowing for the application of tools from topology (Betti numbers, persistent homology).

**Core Idea**: Under the "simple neuron" model validated by McCracken et al. (2025), the structure of the first-layer manifold is **determined solely by the joint distribution of two phases $(\phi_L, \phi_R)$**. Based on this, it can be proven that the manifold is almost surely a torus $T^2$ or its linear projection (the vector-addition disk = Pizza), whereas the second-order angular summation structure of the Clock would not naturally emerge under this hypothesis.

## Method

### Overall Architecture
The task is fixed as modular addition with $n=59$. All architectures first use a shared learnable embedding matrix to map $a, b$ to $E_a, E_b \in \mathbb{R}^{128}$, followed by different processing methods: MLP-Add feeds $E_a + E_b$ directly into the MLP; MLP-Concat feeds the concatenation $E_a \oplus E_b$ into the MLP; learnable attention (referred to as Clock/Attention 1.0) and uniform/constant attention (referred to as Pizza/Attention 0.0) pass through a self-attention layer before entering the MLP.

The analysis pipeline of this paper is: **First use the "simple neuron" model to give a closed-form for the first-layer pre-activation $\to$ use a symmetry theorem to decompose the pre-activation matrix of a whole neuron cluster into rank-2 (disk) or rank-4 (toroidal) factors to predict manifold geometry $\to$ then design two statistical tools (Phase Alignment Distribution PAD + Betti number distribution) to test 703 trained networks and verify that the predicted manifolds indeed appear universally**. Its framework is "theoretical characterization $\to$ large-scale statistical validation," which is not a traditional serial pipeline and is thus explained via theorems and formulas.

The key objects are the pre-activation manifold and logit manifold for a frequency cluster $f$ at layer $\ell$:

$$M^{\text{pre}}_{\ell,f} := \{ h^{\text{pre}}_{\ell,f}(a,b) : (a,b)\in\mathbb{Z}_n^2 \}, \qquad M^{\text{logit}}_{f} := \{ l_f(a,b) : (a,b)\in\mathbb{Z}_n^2 \}.$$

### Key Designs

**1. Simple neuron model + Treating a cluster of neurons as a manifold**

Previous works focused on individual neurons/weights, making it difficult to see commonalities across architectures. This paper leverages the fact validated by McCracken et al. (2025): most first-layer neurons are "simple neurons," whose pre-activation is a linear superposition of two sinusoids for $a$ and $b$:

$$N(a,b) = \cos(2\pi f a/n + \phi_L) + \cos(2\pi f b/n + \phi_R),$$

implying that all degrees of freedom for a simple neuron reside in the phase pair $(\phi_L, \phi_R)$. The authors cluster all neurons with the same "key frequency" $f$ (determined by a 2D Discrete Fourier Transform of the $n \times n$ pre-activation matrix of each neuron), flatten each neuron's pre-activation matrix, and stack them column-wise to obtain an $n^2 \times |\text{cluster } f|$ **neuron-cluster pre-activation matrix**. Thus, the object of study is elevated from a "single neuron" to a "point set spanned by a cluster of neurons = a manifold," enabling the use of geometric and topological tools.

**2. Phase distribution determines the manifold: The Torus/Disk Dichotomy Theorem**

Modular addition is commutative; intuitively, swapping $a$ and $b$ should not change the output, leading to an expectation of symmetry between the two phases. This intuition is formalized in Theorem 4.1: Let a frequency cluster have $m \geq 2$ neurons, with matrix $X \in \mathbb{R}^{p^2 \times m}$ having elements $X_{(a,b),i} = \cos(\theta_a + \phi_i^L) + \cos(\theta_b + \phi_i^R)$ ($\theta_a = 2\pi f a/p$), and assume $\phi_i^L, \phi_i^R$ are identically distributed with a joint distribution $\mu_i^{a,b}$ whose support has positive measure. Almost surely, two cases occur:

- **Perfect Phase Correlation** ($\phi_i^L \equiv \phi_i^R$): $X$ has a rank-2 decomposition $X = V^{\text{disc}}W$, where $V^{\text{disc}}_{(a,b)} = (\cos\theta_a + \cos\theta_b, \sin\theta_a + \sin\theta_b)^\top$, which is exactly the **vector-addition disk (Pizza)** shown in Fig 1;
- **Independent Phases**: $X$ has a rank-4 decomposition $X = V^{\text{torus}}W$, where $V^{\text{torus}}_{(a,b)} = (\cos\theta_a, \sin\theta_a, \cos\theta_b, \sin\theta_b)^\top$, which exactly encodes the **torus $T^2$**.

The horizontal insight is: the disk is merely a linear projection of the torus $(x_1, x_2, x_3, x_4) \mapsto (x_1 + x_3, x_2 + x_4)$. Thus, **the torus is the more general structure, and the pizza disk is its rank-reduced version**. A direct corollary (Remark 4.2) is that the Clock (angular summation requiring second-order interaction) proposed by Zhong et al. **cannot naturally emerge** under the assumptions of Theorem 4.1—it is theoretically possible but will not be learned naturally. This dismantles the argument that "Clock vs. Pizza are fundamentally different algorithms": the difference lies only in whether phases are perfectly correlated, not the circuit type.

**3. Statistical validation via Phase Alignment Distribution (PAD) + MMD/torus distance**

The theorem simplifies "manifold identification" into "identifying whether phases are aligned," which can be verified at scale by statistics. The authors define **PAD**: a distribution over $\mathbb{Z}_n \times \mathbb{Z}_n$ sampled by training a network with a random seed, uniformly sampling a neuron, and returning the input pair $(a,b)$ that maximizes its activation (phases can also be estimated via "activation centroids," which give qualitatively consistent PAD). PAD intuitively characterizes "how frequently learned phases fall on the $a=b$ diagonal," i.e., how aligned the phases are. To quantitatively compare PAD across architectures, the authors use **Maximum Mean Discrepancy (MMD)** and propose **torus distance**—the discrete graph distance on a torus from point $(a,b)$ to the $a=b$ line—using its histogram to distinguish models. These tools allow statistics for 703 one-hidden-layer networks rather than manual interpretation of a few.

**4. Betti number distribution for characterizing multi-layer network topology**

While PAD targets one-hidden-layer networks, for multi-layer networks, the authors turn to TDA, using persistent homology (Ripser library) to estimate the distribution of the **Betti number vector** $(\beta_0, \beta_1, \beta_2)$ for the neuron set of a frequency cluster at each layer (and logits): $\beta_0$ counts connected components, $\beta_1$ counts loops, and $\beta_2$ counts voids enclosed by surfaces. Reference values: Disk is $(1,0,0)$, Circle is $(1,1,0)$, 2-Torus is $(1,2,1)$. Using the Betti distribution allows for statistical inference of whether a layer structure resembles a disk, torus, or circle, verifying that "different architectures perform topologically equivalent computations" and "logit layers universally converge to an annulus." Note: "disks" occasionally detected at the logits are often artifacts of persistent homology failing to find small-radius voids.

## Key Experimental Results

Experiments were conducted on 703 trained one-hidden-layer networks spanning MLP-Add, Attention 0.0 (Pizza), Attention 1.0 (Clock), and MLP-Concat architectures.

### Main Results: First-layer representation vs. Reference manifold similarity (CKA / RSM)

| Reference Manifold | Metric | MLP-Concat | MLP-Add | Attn 0.0 | Attn 1.0 |
|----------|------|-----------|---------|----------|----------|
| Disk (Vector Add) | CKA | 0.707 | **0.998** | **0.988** | **0.974** |
| Disk (Vector Add) | RSM | 0.578 | **0.998** | **0.986** | **0.972** |
| Torus | CKA | **0.994** | 0.706 | 0.699 | 0.689 |
| Circle (Clock) | CKA | ~0 | ~0 | ~0 | 0.012 |

Conclusion: The first layers of MLP-Add, Pizza, and Clock almost perfectly align with the **disk** (CKA $\approx$ 0.97–0.998), while alignment with the torus reference is only 0.69–0.71; MLP-Concat, conversely, strongly aligns with the **torus** (CKA 0.994). Alignment for all architectures with the "Circle (Clock)" reference is near zero—meaning no network learns the angular summation structure in the first layer as claimed by Zhong et al.

### Logit layer vs. Reference manifold

| Reference Manifold | Metric | MLP-Concat | MLP-Add | Attn 0.0 | Attn 1.0 |
|----------|------|-----------|---------|----------|----------|
| Circle (Clock) | CKA | **0.986** | **0.926** | **0.940** | **0.941** |
| Disk | CKA | ~0 | 0.037 | 0.002 | 0.002 |

Logits for all four architectures align highly with the **circle/annulus** (CKA 0.93–0.99), indicating that regardless of whether the first layer is a disk or torus, they ultimately converge to the same logit manifold.

### PAD / MMD and Key Findings
- PAD histograms show that MLP-Add, Attention 0.0, and Attention 1.0 are highly concentrated on the $a=b$ diagonal, while MLP-Concat is spread almost uniformly off-diagonal.
- The PADs of Attention 0.0 and 1.0 are extremely close under MMD (**0.0237** and **0.0181** respectively), with statistical significance $p \approx 0$; MLP-Add is close to both, while MLP-Concat is strongly separated from all. This directly invalidates the claim that Pizza and Clock learn different circuits.
- Betti results: MLP-Add, Attention 0.0, and Attention 1.0 are topologically equivalent; MLP-Concat appears different but is actually **more efficient**—the torus already contains the "holes" needed to project the correct answer to logits, requiring only one nonlinearity. It is simply a lower-cost implementation of the same computation.
- Post-ReLU activations concentrate along the $a=b$ diagonal and decay smoothly with $|a-b|$. This "diagonal dependence" was previously treated as a **definitive feature** of Pizza by Zhong et al.; this paper finds it also appears in MLP-Add and Clock, further refuting the Clock/Pizza distinction.

## Highlights & Insights
- **Lifting "circuit explanation" from single neurons to the "manifold" level**: Clustering + 2D-DFT frequency anchoring + closed-form factorization allows for treating a whole cluster as a single entity. The perspective is transferable—transforming mechanistic interpretability from element-wise qualitative analysis into bulk statistical analysis using topological/geometric tools.
- **Unifying opposing explanations with "Disk is a linear projection of a Torus"**: The most "aha" moment is that the difference between Clock/Pizza/MLP is compressed into a single binary switch—whether phases are perfectly correlated—rather than two independent algorithms. The Universality Hypothesis is thus "saved."
- **Simplifying manifold discrimination into phase distribution discrimination**: Combined with computable statistics like MMD/Betti, this allows for validation across hundreds of networks rather than manual interpretation—a key methodological acceleration.
- The proposed torus distance and PAD are lightweight, reusable measures for "representational similarity" that can be generalized to other group operations or toy tasks for circuit comparison.

## Limitations & Future Work
- All experiments were fixed on the toy setting of $n=59$ cyclic group modular addition. Extrapolation to more general group operations or sub-tasks in real large models remains an open question.
- Core conclusions rely on the hypothesis of "simple neuron model + i.i.d. phases with positive measure support." While this holds empirically in the first layer, subsequent layers exhibit a mix of degree-1 and degree-2 sinusoids; Theorem 4.1 primarily characterizes the clean dichotomy of the first layer.
- Persistent homology is insensitive to **small-radius voids**, leading to "disk" misidentifications at logits, which the authors corrected via manual review—more robust handling of topological noise is needed for large-scale automation.
- The analysis primarily focuses on PAD for one-hidden-layer networks; multi-layer networks are only indirectly characterized via Betti distributions. The fine-grained process of "iterative rotation + linear projection" toward the logit annulus remains to be fully detailed.

## Related Work & Insights
- **vs. Zhong et al. (2023)**: They used distance irrelevance / gradient symmetricity labels to split networks into Clock and Pizza, asserting disjoint circuits. This paper uses closed-form characterization of phase distributions to prove both are the same disk manifold projection in the first layer, with PADs overlapping under MMD, refuting the "different circuits" counterexample.
- **vs. Nanda et al. (2023)**: They modeled neurons in all layers as degree-2 trigonometric polynomials to explain grokking. This paper follows the correction by McCracken et al. (2025)—the first layer is mainly degree-1 simple neurons, with degree-2 needed later—and derives exact closed-form manifold structures based on this.
- **vs. McCracken et al. (2025)**: They used abstract proofs to show MLP and transformers converge to a divide-and-conquer algorithm approximating the Chinese Remainder Theorem. This paper follows up by providing the precise closed-form of representation manifolds (torus/disk) and using TDA to quantitatively validate "representational universality" through statistics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies opposing Clock/Pizza explanations via "torus $\leftrightarrow$ linear projection" and identifies the existence of toroidal representations.
- Experimental Thoroughness: ⭐⭐⭐⭐ 703 networks + CKA/RSM/PAD/MMD/Betti cross-validation is solid, though limited to the $n=59$ toy task.
- Writing Quality: ⭐⭐⭐⭐ Theorems and statistical validation are clearly linked, though some topological/phase notation is dense for non-experts.
- Value: ⭐⭐⭐⭐⭐ Directly repairs a key case previously used as a counterexample against the Universality Hypothesis, providing strong methodological guidance for mechanistic interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability](temporal_geometry_of_deep_networks_hyperbolic_representations_of_training_dynami.md)
- [\[ICLR 2026\] From Data Statistics to Feature Geometry: How Correlations Shape Superposition](from_data_statistics_to_feature_geometry_how_correlations_shape_superposition.md)
- [\[ICLR 2026\] Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization](mixture_of_cognitive_reasoners_modular_reasoning_with_brain-like_specialization.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Diagnosing Generalization Failures from Representational Geometry Markers](diagnosing_generalization_failures_from_representational_geometry_markers.md)

</div>

<!-- RELATED:END -->
