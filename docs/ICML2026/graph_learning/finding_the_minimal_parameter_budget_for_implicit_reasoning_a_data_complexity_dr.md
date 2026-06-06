---
title: >-
  [Paper Note] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models
description: >-
  [ICML 2026][Graph Learning][Implicit reasoning] Starting from knowledge graph completion tasks, this paper proves and measures that the "minimal parameter amount required for implicit reasoning" satisfies a linear scalin…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Implicit reasoning"
  - "minimal parameter budget"
  - "graph search entropy"
  - "U-shaped scaling"
  - "knowledge graph completion"
date: 2026-05-08
content_hash: dc18725d38158a42
---

# Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models

**Conference**: ICML 2026  
**arXiv**: [2504.03635](https://arxiv.org/abs/2504.03635)  
**Code**: https://github.com/WANGXinyiLinda/reasoning-scaling-law (Available)  
**Area**: LLM Reasoning / Pre-training Scaling Law  
**Keywords**: Implicit reasoning, minimal parameter budget, graph search entropy, U-shaped scaling, knowledge graph completion

## TL;DR
Starting from knowledge graph completion tasks, this paper proves and measures that the "minimal parameter amount required for implicit reasoning" satisfies a linear scaling law with **graph search entropy** as the complexity metric. Each parameter supports approximately $0.008$ bits of reasoning information, overturning the naive intuition that "larger models always lead to stronger reasoning."

## Background & Motivation

**Background**: LM scaling laws proposed by Kaplan, Hoffmann, et al., are built on the classical assumption that "loss decreases monotonically with parameter count." Allen-Zhu & Li (2025) further proposed a memory capacity scaling law suggesting each parameter can store $2$ bits of knowledge. Most prevailing works assume: larger models $\rightarrow$ lower test loss $\rightarrow$ stronger capabilities.

**Limitations of Prior Work**: This paradigm describes "memory" but **fails to characterize the parameter budget required for "reasoning."** Methods like Chain-of-Thought (CoT) and RL post-training are "secondary processing" on top of pre-trained representations. The fundamental question is: what is the minimum model size required for reasoning to emerge during the pre-training phase? Existing scaling laws cannot answer this; even counter-examples like inverse scaling and broken scaling laws suggest that the monotonicity assumption is not universal.

**Key Challenge**: **Memory is "packing information into parameters," where more parameters are generally better; reasoning is "internalizing structures into functions," where excessive parameters may lead to overfitting specific triples and losing underlying rules.** The entanglement of these two capacities is the root cause of why current scaling laws lack precision.

**Goal**: Decomposition into two questions: (1) Is the concept of the "minimal model supporting optimal implicit reasoning" mathematically well-defined and identifiable? (2) If it exists, what data attributes determine this minimal size? Can we provide an extrapolatable scaling law for real-world data?

**Key Insight**: Abstract world knowledge into a **Knowledge Graph (KG)**, pre-training corpora into "triple streams on the graph," and reasoning into "completing unseen edges derivable by rules." Replace entity names with random IDs to strip lexical signals, reducing the problem to a pure mapping from "graph structure $\rightarrow$ parameter budget."

**Core Idea**: Use **graph search entropy** (the information rate of a maximum entropy random walk on the graph) as the data complexity metric. Prove that the optimal model size $N_\theta^*(G) = \Theta(H(G))$, and verify this on synthetic graphs and real-world graphs (FB15K-237).

## Method

### Overall Architecture

The method forms a closed loop: "Data $\rightarrow$ Training $\rightarrow$ Measurement $\rightarrow$ Theory $\rightarrow$ Verification." (1) **Data**: Use preferential attachment algorithms to generate controllable synthetic KGs $G$, sweeping parameters across dimensions like number of rules $N_h$, relations $N_r$, entities $N_e$, and derivable triple ratio $\gamma$. (2) **Training**: Use Llama architecture with a fixed batch size of $1024$ to pre-train LMs of various sizes from scratch. Inputs are triples (entities/relations replaced by random IDs and tokenized by character), aiming for next-token prediction loss $L(\theta) = \frac{1}{N} \sum_i -\log P_\theta(e_i^h, r_i, e_i^t)$. (3) **Evaluation**: Perform 1-of-10 multiple-choice questions on held-out derivable triples to record test loss and accuracy. (4) **Measurement**: Calculate graph search entropy $H(G)$ and optimal parameter budget $N_\theta^*(G)$ for each graph. (5) **Theory**: Prove $N_\theta^* = \Theta(H)$ and perform extrapolation on FB15K-237.

The most critical design of the pipeline is **eliminating all lexical signals using random IDs and character tokenization**. Only this allows the U-shaped scaling curve to emerge clearly without being contaminated by confounding variables like pre-trained tokenizers or word frequencies.

### Key Designs

1.  **U-shaped Scaling and Formalization of "Optimal Model Size"**:
    - **Function**: Distills the empirically observed phenomenon where "test loss first decreases and then increases with model size" into a mathematically identifiable and convergent target.
    - **Mechanism**: Define the "best achieved test loss within a budget of $t$ steps" as $\underline{\ell}_t(\theta, G) := \min_{0 \le s \le t} \ell(\theta_s, G)$ (i.e., optimal loss after early stopping). Then define the $\epsilon$-optimal model size $N_{\theta,t}^*(G) := \min\{N_\theta : \exists \theta, \underline{\ell}_t(\theta, G) \le \underline{\ell}_t^*(G) + \epsilon\}$. **Theorem 3** provides convergence: under a mild "gap condition" (where all models smaller than the optimal size are at least $\epsilon + \Delta$ worse), $N_{\theta,t}^*(G) \to N_\theta^*(G)$ as training steps $t \to \infty$. Thus, the concept of a "minimal sufficient model" is mathematically well-defined.
    - **Design Motivation**: Transforms the vague engineering "sweet spot" into a provable, measurable, and extrapolatable physical quantity, connecting the U-shaped curve with theoretical threads like benign overfitting and double descent—optimal reasoning occurs in the "smallest model that can just represent the task," not the largest.

2.  **Graph Search Entropy $H(G)$ as Data Complexity Metric**:
    - **Function**: Quantifies the "intrinsic information required for multi-hop reasoning on a graph" using a scalar determined purely by graph structure.
    - **Mechanism**: Consider a maximum entropy random walk on the graph. Let the principal eigenvalue of adjacency matrix $A$ be $\lambda$ and its eigenvector be $\psi$. The stationary distribution is $\rho_i = \psi_i^2 / \|\psi\|_2^2$, and the transition matrix is $S_{ij} = (A_{ij}/\lambda)(\psi_j/\psi_i)$. Merging entity-entity transitions into entity-relation transitions $S^r_{ij} = \sum_k \mathbb{1}[(i,j,k) \in G] S_{ik}$ yields the relation entropy rate $H^r(G) = -\sum_i \rho_i \sum_j S^r_{ij} \log S^r_{ij}$. The final graph search entropy is $H(G) = N_e \cdot (\log \lambda + H^r(G))$, characterizing uncertainties in both "which entity to reach" and "which relation to follow."
    - **Design Motivation**: Distinguishes this from the "knowledge entropy" of Allen-Zhu & Li—the latter measures "information in the generation process" (suitable for memory), while the former measures "complexity of graph traversal" (suitable for reasoning). This explains why the scaling coefficient ($0.008$ bits/parameter) is 250 times smaller than the memory capacity ($2$ bits/parameter)—reasoning is essentially more "parameter-hungry" than memory.

3.  **Theoretical Bridging and Empirical Verification of $N_\theta^*(G) = \Theta(H(G))$**:
    - **Function**: Locks graph complexity and parameter budget into a linear relationship, enabling the prediction of optimal model size just by looking at the data.
    - **Mechanism**: **Theorem 4** provides $N_\theta^*(G) = \Theta(H(G))$ under three assumptions: (i) random IDs prevent semantic sharing across entities; (ii) parameter capacity at finite precision is $O(N)$; (iii) the Bayes conditional distribution of each entity can be approximated by sparse coefficients $a_x$ on a shared basis $B$, with $\|a_x\|_0 \le \alpha H(Y|X=x) + \beta$. The core of the proof is that the total conditional complexity $C(G) := \sum_x H(Y|X=x)$ is of the same order as $H(G)$. Empirically, $(H(G), N_\theta^*)$ pairs from synthetic graphs fit a regression line with $R^2 = 0.85$. Substituting $H$ from the real-world FB15K-237 into the model yields a predicted $N_\theta^*$ that matches actual observations closely (green dots in Fig. 4 fall on the regression line).
    - **Design Motivation**: Synthetic graphs are sweepable but not "real," while real graphs are realistic but limited. This two-stage experimental design ensures rigor in variable control while proving the scaling law is not a synthetic artifact.

### Loss & Training

The training objective is standard next-token prediction (CE loss). All experiments fix batch size at $1024$ and train for $10$k steps (as a practical approximation of the large-budget limit in Theorem 3). Llama architecture is used, with model sizes swept by adjusting hidden dimensions and layers (see Appendix E). Evaluation scoring is only performed on held-out derivable triples using a 1-of-10 multiple-choice format to eliminate side effects from generated ID formats.

## Key Experimental Results

### Main Results

| Setting | Phenomenon | Key Numbers |
| :--- | :--- | :--- |
| FB15K-237 + Random IDs (Fig. 1, Row 3) | Test loss shows clear U-shape; train loss monotonically decreases | $N_e = 14,505$, $N_r = 237$, $N = 310,116$ |
| Synthetic Graph Sweep (Fig. 3 a-f) | Optimal size stable with training steps; increases with $N$ and $N_r$; insensitive to $N_h$ | 6D ablation coverage |
| Synthetic Graph $(H, N_\theta^*)$ Regression (Fig. 4) | Strong linear relationship | $R^2 = 0.85$ |
| FB15K-237 Extrapolation (Fig. 4, Green Dot) | Real graph falls within 95% CI of synthetic fit | Predicted vs. actual $N_\theta^*$ match closely |
| Reasoning Capacity Scaling Coefficient | $\approx 124$ parameters per $1$ bit of graph search entropy | $\approx 0.008$ bits / parameter |

### Ablation Study

| Changed Graph Property | Effect on Optimal Model Size | Effect on Reasoning Performance |
| :--- | :--- | :--- |
| Training steps $t$ ↑ | Decreases then stabilizes (consistent with Thm 3) | Gradual improvement until saturation |
| Triple count $N$ ↑ | Increases (classic scaling) | Improvement |
| Rule count $N_h$ ↑ | **Essentially no change** | Affects accuracy but not search complexity |
| Relation count $N_r$ ↑ | Increases | Improvement (decreased spurious correlations) |
| Derivable ratio $\gamma$ ↑ | Increases then saturates | Improves then saturates |
| Entity count $N_e$ ↑ | Increases | Decreases when rules/relations are sparse |

### Key Findings

- The fact that **$N_h$ does not affect optimal size** is counter-intuitive: more rules affect accuracy, but **the search complexity of the graph remains unchanged**, indicating that $H(G)$ captures the true determinant of parameter budget.
- **Reasoning capacity ($0.008$ bits/parameter)** is $\approx 250\times$ lower than **memory capacity ($2$ bits/parameter)** from Allen-Zhu & Li. Reasoning is more than two orders of magnitude more "expensive" in terms of parameters, suggesting that RL/CoT optimizations should target "small but precise" models rather than "large and generic" ones.
- **U-shaped curves only appear during "overtraining"**: Large models are not incapable of optimality; they are simply "unnecessary" and prone to overfitting. This aligns observations like U-shapes, inverse scaling, and broken scaling under the benign overfitting framework.

## Highlights & Insights

- **Splitting the "scaling law for reasoning" from the "scaling law for memorization"**: Slicing capacity by task type is a rare conceptual breakthrough in the last seven years of scaling research, far more meaningful than simply altering functional forms of fits.
- **Graph search entropy is an "extrapolatable" metric**: Real-world pre-training corpora can yield underlying graphs via automatic KG extraction (Zhong et al., 2023). Calculating $H$ and predicting optimal model size provides a principled path for "estimating data complexity before selecting model size," directly transferable to domains like code, theorem proving, and medical KGs.
- **Random IDs + character tokenization is a key trick for removing lexical confusion**: The "noise" in many scaling works comes from the tokenizer and word frequencies; this setup can be reused for other controlled scaling studies.

## Limitations & Future Work

- **Architecture Dependency**: The upper bound in Theorem 4 relies on Transformer's attention key-value memory (Assumption iii). Whether this holds for non-attention architectures like SSM/Mamba remains unverified, which the authors identify as future work.
- **Limited Training Duration**: Training for only $10$k steps as an approximation for "infinite training" means the localization of optimal size is still affected by discrete model sizes and early-stopping noise; this partly explains why $R^2 = 0.85$ instead of higher.
- **Limited Real-world Data**: Extrapolation was only verified on FB15K-237. Larger, more complex real KGs (Wikidata subgraphs, domain-specific KGs) have not been tested. A natural extension would be running automatic KG construction on text corpora $\rightarrow$ calculating $H$ $\rightarrow$ training LMs of various sizes to verify the prediction pipeline.
- **Narrow Definition of "Reasoning"**: The relationship between graph search entropy and more complex reasoning (CoT, multi-step arithmetic, theorem proving) is unknown. Exploring a more generalized "task search entropy" is a promising direction.

## Related Work & Insights

- **vs. Kaplan / Hoffmann classic scaling laws**: They characterize "loss monotonic decrease with parameters." This paper reveals that for reasoning tasks, loss is U-shaped, and the optimal size is determined by data complexity rather than compute budget, serving as a substantial supplement to classic scaling.
- **vs. Allen-Zhu & Li (2025) knowledge capacity scaling law**: Both perform sliced scaling. They focus on "memory" yielding $2$ bits/parameter, while this paper focuses on "reasoning" yielding $0.008$ bits/parameter; the two scaling laws are complementary.
- **vs. Wang et al. (2024b) random walk aggregation hypothesis**: Both use KG tasks as a testbed. They explain how LMs aggregate random walk paths to reason, while this paper quantifies the parameter budget required for this setup.
- **vs. Broken neural scaling law (Caballero et al., 2023) / Inverse scaling (Wei et al., 2023)**: Those works document phenomena; this paper provides a theoretical explanation for the U-shape (benign overfitting + gap condition) and locates the specific $N_\theta^*$.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to slice "reasoning" from scaling laws with a computable, extrapolatable complexity metric.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive 6D ablation on synthetic graphs and real graph extrapolation, though only one real graph and limited training steps were used.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear concepts, tight coupling between theory and experiments, well-explained motivations for key tricks (random IDs).
- **Value**: ⭐⭐⭐⭐⭐ Provides a principled framework for selecting model sizes for reasoning tasks and unifies scattered phenomena like U-shapes and inverse scaling under benign overfitting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[ACL 2026\] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](../../ACL2026/graph_learning/craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](../../ACL2026/graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)

</div>

<!-- RELATED:END -->
