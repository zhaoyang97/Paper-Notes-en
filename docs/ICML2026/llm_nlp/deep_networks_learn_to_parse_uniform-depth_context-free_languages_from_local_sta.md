---
title: >-
  [Paper Note] Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics
description: >-
  [ICML 2026][LLM/NLP][PCFG] The authors propose a "Varying-tree RHM" Probabilistic Context-Free Grammar (PCFG) with controllable ambiguity. They demonstrate that grammar rules can be recovered and CYK-style parsing perfor…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "PCFG"
  - "Syntactic Parsing"
  - "Sample Complexity"
  - "Hierarchical Representation"
  - "Local Statistics"
date: 2026-05-08
content_hash: acdad7900180bd3f
---

# Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics

**Conference**: ICML 2026  
**arXiv**: [2602.06065](https://arxiv.org/abs/2602.06065)  
**Code**: https://github.com/jackparley/learn_to_parse  
**Area**: NLP Theory / Interpretability / Learning Mechanisms of Language Models  
**Keywords**: PCFG, Syntactic Parsing, Sample Complexity, Hierarchical Representation, Local Statistics

## TL;DR
The authors propose a "Varying-tree RHM" Probabilistic Context-Free Grammar (PCFG) with controllable ambiguity. They demonstrate that grammar rules can be recovered and CYK-style parsing performed using only low-order moments (root-to-pair and root-to-triple) combined with layer-wise clustering. The derived sample complexity $P^\star \asymp v\, m_3\, m_2^{L-1} (p_2^2/2)^{1-L}$ is empirically validated by the power laws observed in CNN and Transformer experiments.

## Background & Motivation

**Background**: Probing experiments have repeatedly confirmed that Large Language Models (LLMs) can learn tree-like parsing behaviors without explicit syntactic supervision. On the theoretical front, PCFGs are commonly used as toy models. It is known that Transformers can approximate the "inside algorithm," and the sample complexity of deep networks under the Random Hierarchy Model (RHM) with fixed tree structures has been clearly characterized.

**Limitations of Prior Work**: Existing theories primarily focus on simplified "fixed-tree" scenarios where the shape of the parse tree is known in advance. In such cases, genuine "parsing" is unnecessary—hierarchical clustering suffices. This bypasses two critical challenges in real language learning: (A) the learner does not know which span corresponds to which latent non-terminal; (B) the same substring can be produced by different non-terminals (local ambiguity), causing low-order correlations to be "polluted."

**Key Challenge**: To maintain polynomial sample complexity, learners must rely on low-order statistics. However, PCFG ambiguity makes these statistics unreliable—a pair $(a,b)$ might be binary siblings, a prefix of a ternary triple $(a,b,c)$, or cross a span boundary.

**Goal**: (i) Construct a family of synthetic PCFGs with "adjustable ambiguity" controlled by a scalar $f$. (ii) Develop a rule inference algorithm using only low-order moments and clustering, proving its correctness and sample complexity. (iii) Empirically verify that the sample complexity of deep networks (CNNs/Transformers) strictly follows the theoretical predictions.

**Key Insight**: The authors observe that even with ambiguity, in the limit of vocabulary size $v \to \infty$, the root-to-pair covariance contributed by true binary siblings still dominates contributions from "fake siblings." In the ternary case, while true triples and "binary siblings + 1" contribute at similar scales, the latter can be explicitly subtracted using $C_2$.

**Core Idea**: Equivalent to "deep networks learning to parse" as "performing hierarchical clustering with denoising using low-order root-to-substring covariance," and providing a closed-form sample complexity through a signal-to-noise ratio (SNR) argument.

## Method

### Overall Architecture
The approach consists of three steps: (1) Defining the Varying-tree RHM dataset—a family of stochastic PCFGs allowing mixed binary/ternary rules and variable sentence lengths, with ambiguity controlled by $f_2, f_3$. (2) Iterative top-down Algorithm 1: at layer $\ell$, recover binary rules using root-to-pair covariance $C_2^{(\ell)}$ and ternary rules using root-to-triple $C_3^{(\ell)}$ after subtracting binary contamination; then construct indicator functions for the next layer's candidates. (3) Empirical measurements on CNNs/Transformers to verify that learning curves collapse to a master curve when rescaled by $P^\star$.

The input consists of $P$ pairs of (sentence, root label); intermediate products are layer-wise rule sets $\{\mathcal{R}_2^{(\ell)}, \mathcal{R}_3^{(\ell)}\}_{\ell=1}^L$; the output is a Bayes-optimal root classifier (which automatically yields a parse tree).

### Key Designs

1.  **Varying-tree RHM: Synthetic PCFG with Adjustable Ambiguity**:
    *   **Function**: Provides a family of solvable PCFGs that generate exponentially many parse tree topologies with explicit control parameters to study how NNs learn parsing.
    *   **Mechanism**: Each non-terminal $z$ has $m_2 = f_2 v$ binary rules and $m_3 = f_3 v^2$ ternary rules. Rule right-hand sides are sampled uniformly *without replacement*. This ensures individual rules are unambiguous, while their combinations create "local ambiguity" (e.g., $(a,b,c)$ could be derived via $z \to abc$ or $z' \to ab$ plus another derivation). Ambiguity is tuned via $f$; global ambiguity exhibits a phase transition (Fig. 2 bottom) categorized into low, intermediate (local), and high (global) regimes.
    *   **Design Motivation**: Unlike fixed-tree RHMs, this model makes "unknown span boundaries" and "local ambiguity" intrinsic properties, addressing whether NNs truly need to learn parsing.

2.  **InferBinary / InferTernary: Moment-Based Layer-wise Clustering for Rule Inference**:
    *   **Function**: Recovers the rule sets for all layers from $P$ samples without prior rule knowledge.
    *   **Mechanism**: For terminal pairs $(a,b)$, slices of the root-to-pair covariance tensor $u_{ab} = C_2^{(\ell)}((a,b), :) \in \mathbb{R}^v$ are analyzed. The norm $\|u_{ab}\|$ for true siblings is significantly higher than for others as $v \to \infty$. True siblings are filtered via threshold $\tau_2$, and normalized $\hat u_{ab}$ are clustered into $v$ classes, each representing a parent non-terminal (Prop. 3.1). For the ternary case, where $C_3$ includes both "true triples" and "binary + 1" contributions of similar magnitude, the authors subtract binary pollution: $w_{abc} = C_3((a,b,c), :) - \tfrac{1}{v} C_2((a,b),:) - \tfrac{1}{v} C_2((b,c),:)$. These are matched against cluster centers $c_z$ from InferBinary to assign parents (Prop. 3.2).
    *   **Design Motivation**: Explicitly decomposes "learning to parse" into "identifying boundaries" and "clustering rules." Using low-order moments with pollution subtraction handles local ambiguity without requiring high-order statistics, maintaining polynomial sample complexity.

3.  **Sample Complexity Formula and Conjecture 3.4: Transferring Algorithm Complexity to NNs**:
    *   **Function**: Provides a closed-form expression for the samples required to recover $C_2, C_3$ row vectors and posits that standard NNs of depth $\geq L$ exhibit the same complexity.
    *   **Mechanism**: Using the vector Bernstein inequality, the error is bounded by $E_s \leq \gamma_s(\sqrt{\log(2/\delta)/(v m_s P)} + \log(2/\delta)/P)$. Combined with the asymptotic row norm $\|u_{ab}\| \to \frac{1}{m_2 v}\sqrt{(p_2^2/2)^{\ell-1} / m_2^{\ell-1}}$, the samples needed for layer $\ell$ are $P_{\ell, s} \asymp (p_2^2/2)^{1-\ell} v m_s m_2^{\ell-1}$. Conjecture 3.4 states that NN sample complexity equals the hardest layer: $P_{\mathrm{NN}}^\star \asymp \max_{\ell, s} P_{\ell, s}$, which simplifies to $P_{\mathrm{NN}}^\star \asymp (p_2^2/2)^{1-L} v m_3 m_2^{L-1}$ in the asymptotic limit $m_3 \gg m_2$.
    *   **Design Motivation**: Quantifies the SNR intuition—the first path detected is the "easiest" (all binary for $L-1$ layers, ternary for the last). The formula also predicts that networks with depth $< L$ cannot achieve polynomial convergence.

## Key Experimental Results

### Main Results

| Experimental Setting | Predicted $P^\star$ Scaling | Observed Phenomenon | Conclusion |
|----------------------|-----------------------------|---------------------|------------|
| CNN, low ambig $f = 1/v$, $L=2$ | $\sim v^2$ | Curves collapse perfectly after rescaling | Correct scaling |
| CNN, low ambig $f = 1/v$, varying $L$ | $(p_2^2/2)^{1-L} v^2$ | $P^\star(v, L)$ matches theory and measured SNR | $(p_2^2/2)^{1-L}$ depth factor correct |
| CNN, intermediate $f=1/4$, $L=3$ | $\sim v^5$ | Learning curves collapse | Formula holds under local ambiguity |
| CNN, high $f=0.6, 0.8$, $L=2$ | $\sim v^2$ scaling | Loss saturates at analytic Bayes-optimal lower bound | NNs reach info-theoretic limit under high ambiguity |
| INN / CNN / Transformer ($L=2, 3$) | Same $P^\star \sim v^2$ | All architectures collapse with $v^2$ rescaling | Formula is architecture-agnostic |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|---------------|----------------|-------------|
| Full model (depth $= L$) | Curves collapse to $P^\star$ | Complete theory holds |
| Depth $< L$ networks | Failure to converge polynomially | Depth is a necessary condition |
| Removing ternary correction $\tfrac{1}{v}C_2$ | Ternary rules mixed with "fake siblings" | Correction term is mandatory |
| Using token frequency (no root condition) | Complete failure | Label information is necessary |

### Key Findings
- Learning curves of three architectures (INN, CNN, Transformer) collapse to a single master curve after rescaling, strongly supporting the conjecture that NNs and Algorithm 1 share the same mechanism.
- In the high global ambiguity regime, NN cross-entropy converges to the Bayes-optimal lower bound determined by the PCFG's intrinsic entropy.
- The factor $(p_2^2/2)^{1-L}$ indicates that while required samples grow exponentially with $L$, it remains polynomial in $v$, suggesting hierarchical representation is a feasible path for PCFG learning.

## Highlights & Insights
- **Explicit Decomposition of Local Ambiguity**: Unlike prior RHMs that fix tree structures to avoid parsing, this work uses co-existing binary/ternary rules to make local ambiguity natural and controllable.
- **Ternary Covariance Denoising**: The trick $w_{abc} = C_3 - \tfrac{1}{v}(C_2((a,b), :) + C_2((b,c), :))$ is a simple yet profound design—it removes "binary + 1" branches derived from the law of total covariance, allowing ternary rules to be recovered via clustering.
- **Easiest Path Dominance**: The insight that $P_{\mathrm{NN}}^\star$ is determined by the path most dominated by the signal provides a physical intuition for how networks prioritize learning from high-SNR data features.

## Limitations & Future Work
- The task is limited to root classification; the authors acknowledge that next-token prediction requires future work.
- Asymptotic analysis is rigorous as $v \to \infty$, but the neglect of constant terms in finite-vocabulary scenarios needs more exploration.
- Assumptions of uniform rule probabilities ($p_2=p_3=1/2$) and $f_2=f_3=f$ may not hold for heavy-tailed rule distributions in real languages.
- Direct comparison between "NN internal representations" and "explicitly recovered rules" was not performed.
- Transformer experiments were conducted on small $v, L$; scaling to LLM sizes with positional encodings and layer normalization remains unknown.

## Related Work & Insights
- **vs Cagnetta et al. 2024 (Fixed-tree RHM)**: Extends RHM from "learnable hierarchies" to "learnable parsing" by allowing stochastic tree shapes.
- **vs Allen-Zhu & Li 2025 (Transformer approximates Inside)**: Provides a forward theory for *how much* data is needed to reach the state where NNs approximate the inside algorithm.
- **vs Malach & Shalev-Shwartz 2018 (Clustering-based learnability)**: Quantifies the clustering intuition into a closed-form sample complexity aligned with real NN learning curves.
- **vs Sclocchi et al. 2025 (Belief Propagation)**: Algorithm 1 can be seen as an approximate BP for unknown tree structures, bridging BP literature with PCFG learnability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reduce "NN parsing" to provable low-order moment clustering with precise complexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across architectures and ambiguity regimes, though Transformer size is small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with a closed loop between theorems and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a foundational theoretical explanation for why LLMs are data-efficient in learning syntax.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](../../NeurIPS2025/llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](../../NeurIPS2025/llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[ACL 2026\] A Study of LLMs' Preferences for Libraries and Programming Languages](../../ACL2026/llm_nlp/a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)

</div>

<!-- RELATED:END -->
