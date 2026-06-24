---
title: >-
  [Paper Note] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks
description: >-
  [ICLR 2026][Learning Theory][Quantized Neural Networks] This paper systematically investigates the (parameterized) complexity of "training fully quantized ReLU networks" for the first time. It proves that training is NP-hard even under extremely simplified architectures such as binary quantization, single output, and no hidden layers. However, by treating the **input dimension $\alpha$** combined with network width (or more generally, treewidth) and either the output dimensio…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Parameterized Complexity"
  - "Quantized Neural Networks"
  - "Training Complexity"
  - "Fixed-Parameter Tractable"
  - "treewidth"
date: 2026-05-08
content_hash: 36e0ab7c3aed6a9d
---

# Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BAQNrsr987](https://openreview.net/forum?id=BAQNrsr987)  
**Code**: None (Pure theoretical paper)  
**Area**: Learning Theory / Parameterized Complexity  
**Keywords**: Quantized Neural Networks, Training Complexity, Parameterized Complexity, Fixed-Parameter Tractable, treewidth

## TL;DR
This paper systematically investigates the (parameterized) complexity of "training fully quantized ReLU networks" for the first time. It proves that training is NP-hard even under extremely simplified architectures such as binary quantization, single output, and no hidden layers. However, by treating the **input dimension $\alpha$** combined with network width (or more generally, treewidth) and either the output dimension $\omega$ or the error bound $\ell$ as parameters, the problem becomes Fixed-Parameter Tractable (FPT). The core conclusion is that "the hardness lies in high data dimensionality, rather than architectural complexity."

## Background & Motivation

**Background**: The complexity of neural network training has been studied repeatedly in a series of foundational papers in recent years, but almost all focus on **real-valued networks**. In the real-valued setting, training a network to optimality is $\exists\mathbb{R}$-complete (equivalent to the difficulty of "determining whether a set of real polynomial equations has real solutions"), belonging to a complexity class higher than NP.

**Limitations of Prior Work**: The actual trend in industry is **quantization**—restricting weights, biases, and data to a finite integer domain (e.g., 4-bit with a domain size of 16, or even binary $\{0,1\}$) to achieve massive improvements in speed and energy efficiency with almost no loss in accuracy. However, none of the existing complexity lower-bound and upper-bound algorithms **can be transferred to the quantized setting**. This is more than just a missing "bridge": real-valued training is $\exists\mathbb{R}$-complete, while quantized training clearly falls within NP (as the certificate is a linear number of integers in $\mathbb{Z}_d$), indicating that the two are fundamentally different.

**Key Challenge**: Quantization transforms a "continuous optimization / real algebraic geometry" problem into a "discrete search" problem, dropping the complexity class from $\exists\mathbb{R}$ to NP. Within NP, exactly how difficult is this problem? Does there exist some natural structural parameter that makes it tractable from NP-hard? This map was previously entirely blank.

**Goal**: To draw a fine-grained complexity map for the $d$-quantized ReLU network training problem (denoted as $d$-QNNT)—identifying the extremely restricted cases where it remains NP-hard (lower bounds) and the parameter combinations that yield FPT (upper bounds).

**Key Insight**: The authors introduce the **parameterized complexity** paradigm. Instead of simply asking if the problem is solvable in polynomial time, they assign a parameter $k$ to the instance (such as depth, width, input dimension $\alpha$, output dimension $\omega$, error bound $\ell$, or treewidth) and ask if it can be solved in $f(k)\cdot n^{O(1)}$ time (i.e., FPT). This allows for a precise identification of "which quantity is responsible for the hardness."

**Core Idea**: Trading "low dimensionality" for tractability—proving that the root cause of training difficulty is the **high dimensionality of input/output data**, rather than the complexity of the architecture. Thus, by fixing the input dimension $\alpha$ and augmenting it with either width/treewidth and $\omega$ or $\ell$, FPT can be achieved. The key technical pivot is a structural lemma: for every YES instance, there always exists a solution where the "non-zero in-degree is bounded by a function of the parameters."

## Method

### Overall Architecture

The decision problem $d$-QNNT studied in the paper is formalized as follows: Given an architecture $G$ (with $\alpha$ input neurons and $\omega$ output neurons), a $d$-quantized dataset $D$ (where each data point $(\vec{x},\vec{y})\in(\mathbb{Z}_d)^\alpha\times(\mathbb{Z}_d)^\omega$), and an error bound $\ell$, determine whether there exists a set of weights and biases **with values in $\mathbb{Z}_d$** such that the error of the network on $D$ does not exceed $\ell$. Here, the quantized integer domain is defined as $\mathbb{Z}_d=\{z\in\mathbb{Z}\mid -\lfloor\frac{d-1}{2}\rfloor\le z\le\lceil\frac{d-1}{2}\rceil\}$ (e.g., $\mathbb{Z}_2=\{0,1\}$), the activation function is $\mathrm{ReLU}_d$ which truncates ReLU into $\mathbb{Z}_d$, and the error is the simplest "count of mismatched data points."

The entire work proceeds along two tracks, with conclusions synthesized into a complexity map:

**Track One (Lower Bounds)**: Through a series of targeted reductions (starting from EXACT SET COVER, HITTING SET, SET COVER, and 3-SAT), the paper proves that $d$-QNNT remains NP-hard or W[2]-hard even under binary quantization $d=2$ and extremely restricted architectures (single output, no hidden layers, constant degree, $\alpha=2$, etc.). These reductions collectively reveal a phenomenon: the constructed hard instances have **extremely simple architectures, but high-dimensional data**. Therefore, no combination of architectural parameters like depth or width can save them.

**Track Two (Upper Bounds)**: A structural lemma (Lemma 1) is first proved to compress the "non-zero in-degree" to a function of the parameters. Based on this, a dynamic programming algorithm is run on tree decompositions (Lemma 3), resulting in a set of FPT algorithms parameterized by $\alpha+\text{tw}+(\omega\text{ or }\ell)$ (Theorems 5 and 6). Furthermore, it is shown that treewidth can be replaced by width (Corollaries 1 and 2).

This is a pure theory paper with no pipeline or trainable modules; the "Key Designs" correspond to four core theoretical components.

### Key Designs

**1. Formalization of d-QNNT: Reducing Training from $\exists\mathbb{R}$ to NP**

Addressing the pain point that real-valued training complexity conclusions cannot be transferred to quantization, the authors first formalize quantized training precisely as a **decision problem**. Weights, biases, and activations are all constrained within $\mathbb{Z}_d$. The value of a single neuron $v$ (with predecessors $z_1,\dots,z_q$) is given by:
$$f(v)=\mathrm{ReLU}_d\Big(\sum_{i\in[q]} f(z_i)\cdot \text{weight}(z_i v)\;-\;\text{bias}(v)\Big),$$
Error = number of data points where the output mismatches. This seemingly detailed setting has a fundamental consequence: because solutions can be verified using a linear number of $\mathbb{Z}_d$ integers as certificates, $d$-QNNT falls within **NP**, whereas the real-valued version is $\exists\mathbb{R}$-complete. This reduction is the prerequisite for the fine-grained analysis using parameterized complexity. The use of subtraction for the bias is intentional: in Boolean domains, subtraction allows the bias and weights to truly interact.

**2. High-Dimensional Data as the Root of Hardness: Four Extreme Lower Bounds**

The authors pinpoint "where the difficulty comes from" using four theorems. Theorem 1, reduced from EXACT SET COVER, proves that 2-QNNT remains NP-hard even if $\ell=0$ with a **single output neuron and no hidden layers**. This effectively rules out tractability for the combined parameter $\text{width}+\text{depth}+\ell+\omega$. However, since this reduction requires an arbitrarily large in-degree for the output neuron, Theorem 2 proves that it remains NP-hard under **constant degree** (single hidden layer, max out-degree 3, max in-degree 2, $|D|\le 4$), showing that architectural density is not the key. The real key is input dimensionality: Theorem 3 reduces from HITTING SET to prove that 2-QNNT is **W[2]-hard** when parameterized by input count $\alpha$. Theorem 4 uses a 3-SAT reduction to "compress" Theorem 2 to a case where $\alpha=2$, $\ell=0$, $|D|=3$, and depth 1, showing that width cannot be discarded or replaced by depth. Together, these communicate that **training difficulty stems from high-dimensional input/output data, not architectural complexity**.

**3. Bounded Non-zero In-degree Lemma: Pruning Solution Space via Steinitz Lemma**

The biggest technical obstacle for positive results is that an optimal solution might have $\Theta(n)$ non-zero weighted incoming edges for a hidden neuron, which DP on treewidth cannot efficiently search. The authors prove a structural lemma (**Lemma 1**): if a zero-error solution exists, there must exist a zero-error solution where the number of non-zero incoming neighbors (non-zero in-degree) for each neuron is at most $(dp)^{O(p)}$, where $p$ is the number of distinct input vectors. The proof uses the **Steinitz Lemma**: collecting the activations of neuron $v$ for various inputs into a small-norm vector $\vec{s}\in(\mathbb{Z}_d)^p$, which is a sum of several small-norm vectors. The Steinitz Lemma guarantees these vectors can be rearranged such that every prefix sum's norm is $\le d$. Thus, if there are many non-zero incoming edges, two prefix sums must be identical, implying the sum of the middle segment is zero—setting their corresponding weights to 0 does not change $v$'s activation. This "losslessly" compresses the solution's non-zero in-degree to a function of the parameters.

**4. DP on Tree Decomposition: From Bounded In-degree to FPT**

With the bounded in-degree from Lemma 1, the authors run DP on a tree decomposition (**Lemma 3**: $d$-QNNT is FPT w.r.t. treewidth and $\alpha$ when $\ell=0$). The intuition is that the DP must maintain "the state of pre-activation values brought to neuron $v$ by the left-side partial solution" at each small separator. If non-zero in-degree were unbounded, the left side could accumulate unbounded negative pre-activations, which could be cancelled by positive ones on the right, while the total sum remains in a small $\mathbb{Z}_d$ range—forcing the DP to maintain unbounded intermediate sums. Lemma 1 bounds the non-zero in-degree to $\Delta:=d^{O(\alpha d^\alpha)}$, ensuring bounded pre-activation sums and a controlled number of DP states. This is then extended from "zero error" to "non-zero error bound" (Theorem 6: FPT w.r.t. $\text{tw}+\alpha+\ell$) and combined with the output dimension parameter for Theorem 5 (FPT w.r.t. $\text{tw}+\alpha+\omega$). Finally, width is shown to be an upper bound for treewidth as long as there is at least one hidden layer.

### Loss & Training
This paper uses the simplest "mismatch count" error (number of misaligned data points), rather than $\ell_2^2$ losses commonly used in real-valued settings, to keep the complexity analysis clean. The authors note that most proofs can be directly transferred to other loss functions.

## Key Experimental Results

This is a pure complexity theory paper with **no empirical experiments**. Its "results" consist of a complete complexity map.

### Complexity Map (Lower vs. Upper Bounds)

| Parameter Combination | Complexity Conclusion | Theorem | Reduction Source / Key Constraint |
|----------|-----------|------|---------------------|
| $\text{width}+\text{depth}+\ell+\omega$ | NP-hard | Theorem 1 | EXACT SET COVER; $\ell=0$, single output, no hidden layer |
| $\ell+\Delta$ (max degree) | NP-hard | Theorem 2 | Constant degree (out-3 / in-2), 1 hidden layer, $|D|\le 4$ |
| $\alpha$ (input dim) | W[2]-hard | Theorem 3 | HITTING SET; no hidden layer |
| $\alpha=2$ + 1 hidden layer | NP-hard | Theorem 4 | 3-SAT; $\ell=0$, $|D|=3$, depth 1 |
| $\alpha+\text{tw}+\omega$ | **FPT** | Theorem 5 | Via Lemma 1 + Tree decomposition DP |
| $\alpha+\text{tw}+\ell$ | **FPT** | Theorem 6 | Same as above, extended to non-zero error |
| $\alpha+\text{width}+\ell$ | **FPT** | Corollary 1 | Derived from Theorem 6 |
| $\alpha+\text{width}+\omega$ | **FPT** | Corollary 2 | Derived from Theorem 5 |

### Key Findings
- **Hardness stems from data dimensionality, not architecture**: All NP/W-hard reductions use extremely simple architectures (even no hidden layers), but either the input dimension $\alpha$ or output dimension $\omega$ is large. Once $\alpha$ is fixed and supplemented with width/treewidth and either $\omega$ or $\ell$, the problem becomes immediately FPT.
- **Every parameter is essential**: Theorem 1 rules out using only width+depth+$\ell$+$\omega$; Theorem 3 rules out using only $\alpha$; Theorem 4 rules out substituting width with depth. All three parameters in the FPT results are necessary.
- **Treewidth is strictly stronger than width**: With hidden layers, tw never exceeds width and can be arbitrarily smaller. Thus, Theorems 5 and 6 directly imply the two corollaries parameterized by width.
- **Sharp contrast with real-valued settings**: There are no known non-trivial FPT parameterizations for real-valued training, whereas multiple exist for the quantized setting—quantization is not just an engineering optimization, it changes the problem's complexity structure.

## Highlights & Insights
- **The observation that "quantization drops $\exists\mathbb{R}$ to NP" is valuable**: It explains why real-valued conclusions do not transfer and clears the path for parameterized analysis within NP—a clean complexity class boundary.
- **Ingenious use of Steinitz Lemma in NN training**: Using a classic geometric vector rearrangement lemma to prove "the existence of low-in-degree optimal solutions" prunes the combinatorially exploding solution space to a parameter-controlled size. This is the technical hub for the upper bounds and could be adapted to other "discrete weight + bounded range" problems.
- **Implications for practice**: If a task's input dimension is naturally low (e.g., certain sensor or control scenarios), quantized network training may have provably efficient exact algorithms rather than relying solely on heuristics.

## Limitations & Future Work
- **Main open problem**: The complexity of $\alpha+\omega$ (fixing both input and output dimensions while allowing arbitrary architectural complexity) remains unknown—authors suspect solving it requires insights beyond current techniques.
- **Pure theory, no empirical validation**: The function $f(k)$ in the FPT algorithm is $d^{O(\alpha d^\alpha)}$, which is double-exponential for $\alpha$. It is far from being a practical algorithm. The paper acknowledges that deriving efficient empirical algorithms is a future direction.
- **Model simplification**: Error is measured by mismatch count, activation is ReLU, and quantization uses the clean $\mathbb{Z}_d$ model. While the authors claim results transfer to other encodings and losses, this is not formally proven.
- **Extensions**: Promoting results to the **distillation** setting and transforming the "tractability via low dimensionality" insight into practical exact solvers.

## Related Work & Insights
- **vs. Real-valued NNT complexity research (Abrahamsen 2021, Bertschinger 2023, etc.)**: They prove real-valued training is $\exists\mathbb{R}$-complete with no known FPT parameterization. Ours provides both NP-hard lower bounds and FPT fragments for quantization, highlighting that quantization changes the nature of the problem.
- **vs. NP-hardness of Boolean network training (Judd 1990, Schmitt 2004)**: While 2-QNNT's NP-hardness could be inferred from their reductions, Theorems 1–4 provide lower bounds under **additional input constraints** (necessary for parameterized analysis) and offer the first multivariate complexity analysis for quantized ReLU training.
- **vs. Classic DP on treewidth**: Standard treewidth-DP fails for solutions with high non-zero in-degree. Ours uses Lemma 1 for structural pruning before DP, serving as a paradigm for applying graph algorithm tools to neural network training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first systematic study of the parameterized complexity of fully quantized ReLU training. The observation that "quantization reduces $\exists\mathbb{R}$ to NP" and the use of the Steinitz Lemma are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ A pure theory paper; the complexity map covers lower/upper bounds and proves parameter necessity. No empirical results, but meets theory paper standards.
- Writing Quality: ⭐⭐⭐⭐⭐ Uses a clear logic to link all conclusions. The dual tracks of lower and upper bounds are clear, and the technical summary explains exactly "where the hardness lies."
- Value: ⭐⭐⭐⭐ Provides a foundational map and identifies clear open problems ($\alpha+\omega$). Opens a direction for the computability of quantized training, though far from practical algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Parameterized Hardness of Zonotope Containment and Neural Network Verification](parameterized_hardness_of_zonotope_containment_and_neural_network_verification.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](../../NeurIPS2025/learning_theory/the_parameterized_complexity_of_computing_the_vc-dimension.md)
- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)

</div>

<!-- RELATED:END -->
