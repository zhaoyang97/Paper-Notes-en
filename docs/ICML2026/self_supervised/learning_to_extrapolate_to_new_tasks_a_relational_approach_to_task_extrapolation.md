---
title: >-
  [Paper Note] Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation
description: >-
  [ICML 2026][Self-Supervised Learning][Task extrapolation] This paper proposes the Relational Task Extrapolator (RTE), which reinterprets "new tasks outside the training support" as a composition problem of "known anchor…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Task extrapolation"
  - "transductive learning"
  - "anchor-transformation decomposition"
  - "Task2Vec"
  - "relational operator"
date: 2026-05-08
content_hash: 941ffa75cb6fb9bf
---

# Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation

**Conference**: ICML 2026  
**arXiv**: [2605.30132](https://arxiv.org/abs/2605.30132)  
**Code**: The paper mentions a GitHub repository (specific link not provided in the text)  
**Area**: Meta-Learning / Task Extrapolation / Self-Supervised Representation  
**Keywords**: Task extrapolation, transductive learning, anchor-transformation decomposition, Task2Vec, relational operator

## TL;DR
This paper proposes the Relational Task Extrapolator (RTE), which reinterprets "new tasks outside the training support" as a composition problem of "known anchor tasks + seen inter-task transformations." It trains a relational operator $\Psi$ to assemble anchor-transformation pairs at test time to predict outputs for unknown tasks.

## Background & Motivation

**Background**: Modern learning systems are nearly omnipotent at "interpolation" (where test tasks fall within the support of the training distribution), primarily driven by the scale of data and models. The success of foundation models essentially stems from making the training distribution sufficiently vast.

**Limitations of Prior Work**: Once a target "task parameter" jumps out of the training support—for instance, if the training only covers projectile trajectories with initial velocity $v \in [30, 60]$ and the test requires $v = 65$—inductive models saturate at the boundaries. The output becomes static when extrapolating beyond the training support. This failure persists in large language models (Vafa et al. 2025 reports that LLMs learn "heuristics" that collapse when physical constants change).

**Key Challenge**: Inductive learning requires "test samples to be drawn from the training distribution," but many real-world problems require extrapolation. Pure inductive learning cannot identify the true mechanism outside the support (countless hypotheses can fit training data but diverge arbitrarily in the extrapolation zone). Thus, the problem is mathematically ill-posed and requires the injection of additional structural assumptions.

**Goal**: To find a structural assumption that renders extrapolation solvable from an ill-posed state. Simultaneously, this assumption must be general enough to cover three typical extrapolation regimes: parameter extrapolation (continuous), length extrapolation (recursive), and compositional extrapolation (combinatorial).

**Key Insight**: The authors draw inspiration from Vapnik’s transduction philosophy—instead of learning a global function $f$, learn an operator that "translates a known point $f(x')$ to $f(x)$." This paper lifts this concept from the input space to the task space: rather than learning a global solution for a single task, it learns "task-to-task transformations."

**Core Idea**: Any unseen task $f_{\theta^*}$ can be decomposed as $f_{\theta^*} = s_\phi(f_{\theta_{anc}})$, where $f_{\theta_{anc}}$ is an anchor task seen during training and $\phi$ is a relative transformation seen during training. In this way, the hard Out-of-Support (OOS) problem is downgraded to a easier Out-of-Combination (OOC) problem within the support.

## Method

### Overall Architecture

RTE splits task extrapolation into two phases. **Training Phase**: Task pairs $(f_i, f_j)$ are repeatedly sampled from a training task bank $\mathcal{F}_{train}$, and their relative transformation $\phi_{ij}$ is extracted. A relational operator $\Psi$ is trained such that $\Psi(x, f_i, \phi_{ij}) \approx f_j(x)$. **Inference Phase**: Given a small context $D_{target}$ of a target task, its proxy vector $\hat\theta_{target}$ in the task embedding space is first estimated. Then, an optimal anchor $f_{anc}^*$ and transformation $\phi^*$ are identified. Finally, predictions are generated using $\Psi(x_{query}, f_{anc}^*, \phi^*)$. The entire pipeline is applicable to both pure function prediction (with an MLP as $\Psi$) and sequence prediction (with an LLM as $\Psi$, fine-tuned via LoRA).

### Key Designs

1.  **Anchor-Transformation Decomposition**:
    - **Function**: Represents an out-of-support target task $f_{\theta^*}$ as $f_{\theta^*}(x) = \Psi(x, f_{anc}, \phi)$, where $f_{anc} \in \mathcal{F}_{train}$ and $\phi \in \Phi_{train}$.
    - **Mechanism**: Provides a unified interface for three extrapolation regimes: in continuous parameter extrapolation, $\phi = \Delta\theta = \theta_{target} - \theta_{anc}$ (difference operator); in recursive length extrapolation, $\phi$ is the expansion step from complexity $L-1$ to $L$ (e.g., adding a high-order coefficient $c_9$ to an 8th-order polynomial); in compositional extrapolation, $\phi$ is another primitive (e.g., the outer $\sin$ in $\sin\circ x^2$).
    - **Design Motivation**: Learning $f$ directly beyond the boundary is ill-posed. However, by assuming the "task manifold is connected by a family of structured transformations $\mathcal{S}$," the target task can be composed of two seen components within the support—downgrading OOS to OOC.

2.  **Relational Operator $\Psi$ Training**:
    - **Function**: Learns a parameterized operator that maps (query input, anchor task, transformation) to target predictions by minimizing $\mathbb{E}_{f_i, f_j}[\mathcal{L}(f_j(x), \Psi(x, f_i, \phi_{ij}))]$.
    - **Mechanism**: In continuous regimes, $\phi_{ij}$ is directly taken as the Task2Vec embedding difference $\hat\theta_j - \hat\theta_i$. In discrete (length/compositional) regimes, the authors assume ground-truth relational labels are available during training (e.g., if $f_j = f_i \circ g$, $g$ is used as $\phi_{ij}$). Thus, $\Psi$ only needs to learn "how to transform known task outputs based on the transformation" without simultaneously learning the task manifold and transformation mechanism (which is ill-posed).
    - **Design Motivation**: By explicitly modeling task relations as first-order objects, the model learns "mechanisms" rather than "heuristics." This is the key difference between RTE and meta-learning like MAML/Reptile, which assume new tasks fall within the local neighborhood of the training distribution.

3.  **Test-time Decomposition**:
    - **Function**: Given a sparse context $D_{target}$ of an unseen task, it solves for $(f_{anc}^*, \phi^*) = \arg\min_{f, \phi} \sum_{(x,y)\in D_{target}} \mathcal{L}(y, \Psi(x, f, \phi))$.
    - **Mechanism**: **Strategy A (Continuous)** uses Task2Vec to estimate $\hat\theta_{target} = \Gamma(D_{target})$, takes the nearest neighbor as $f_{anc}^*$, and computes $\phi^* = \hat\theta_{target} - \hat\theta_{anc}^*$ via a single lookup. **Strategy B (Discrete)** trains a decomposer $g_\psi$ to provide candidate top-$k$ anchor-transformation pairs, followed by a small-scale search/loss minimization over $\mathcal{C}_k$. **LLM Scenarios** use the negative log-likelihood of the model on $D_{target}$ as a scorer for brute-force search over candidates (referred to as "Neural RAG").
    - **Design Motivation**: The task manifold is geometrically well-behaved in continuous regimes, allowing nearest-neighbor search. In discrete regimes, the manifold is combinatorially explosive, requiring amortized inference to compress the search space to $O(k)$. For LLMs where proxy embeddings cannot be directly computed, using likelihood as a proxy score provides a unified approach without new modules.

### Loss & Training

In function prediction scenarios, $\Psi$ is an MLP and the loss is MSE. In LLM scenarios, $\Psi$ is a LoRA fine-tuned Qwen/Mistral. The (anchor demo, transformation description $\phi$, query input) are formatted into a prompt $P$, minimizing $\mathcal{L}_{SFT} = -\sum_t \log p_\theta(y_t | P, y_{<t})$. The transformation $\phi$ can be a natural language instruction, discrete control tokens, or learned embeddings, depending on the regime.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | Ours (RTE) | Prev. SOTA (Baseline) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Quadratic Parameter Extrap. (F2) | MSE | $\mathbf{7.33\times 10^2}$ | T2V Inductive $1.20\times 10^5$ | ~160× reduction |
| Tri-Trend Parameter Extrap. (F2) | MSE | $\mathbf{0.048}$ | Inductive 0.46 | ~10× reduction |
| 9th-order Poly. Length Extrap. | MSE | $\mathbf{0.371}$ | Naive Baseline 0.575 | -35% |
| Compositional Extrap. (Agg.) | MSE | $\mathbf{0.287}$ | Naive Baseline 0.389 | -26% |
| Sparse Parity $|S|=6$ (Qwen+LoRA) | Acc | $\mathbf{66.07\%}$ | Standard SFT 52.86% | +13.2pp |
| CodeIO Compositional Extrap. | Exact Match | $\mathbf{45.3\%}$ | Few-Shot 19.8% / CoT+Maj@16 30.2% | +25.5pp |

Inductive baselines saturate at the boundaries in all extrapolation regimes—failing to fit curvature outside the range in polynomials or predict correct frequencies for periodic functions. In CodeIO, even with CoT and majority voting, performance only reaches 30%. RTE consistently leads by structurally downgrading extrapolation to OOC.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| RTE (full) | Cubic MSE 1.53 | Full model with anchor selected by Task2Vec nearest neighbor |
| Inductive Oracle | Cubic MSE 3.24 | Even with ground-truth parameters, inductive models fail to extrapolate; the key is "relation" not "parameters" |
| Transductive Oracle | Cubic MSE 0.96 | Upper bound of RTE with ground-truth anchor and transformation |
| CodeIO Oracle | 76.0% | Upper bound for LLM with ground-truth primitive decomposition; ~31pp gap remaining |
| Sparse Parity Oracle | 100.0% | Perfect solution given known parent, suggesting the remaining gap is due to decomposition inference errors |

### Key Findings

- The "upper bound" of relational extrapolation is very high (oracles are near-perfect), but "search" is the bottleneck: in CodeIO, RTE reaches 45.3% vs. 76.0% for the oracle, with the gap stemming from the decomposer picking wrong primitives.
- Inductive learning failures are structural: In periodic functions (Sin-Trend / Tri-Trend), inductive models suffer from spectral bias and fail to fit frequencies, whereas RTE inherits pulse structures from the anchor and only learns a linear shift, bypassing the issue.
- Even with imperfect anchors, RTE provides significantly better predictions than baselines (e.g., in Composition, RTE reduced MSE from 0.39 to 0.29 despite wrong primitive selection), suggesting structural constraints act as strong priors.
- In LLM scenarios, RTE's advantage far exceeds CoT + majority voting, indicating that "explicitly providing anchor outputs + transformation parameters" is more effective than "letting the model reason on its own"—this suggests RTE's prompt template is a structured reasoning scaffold.

## Highlights & Insights

- Lifting transduction from "input space" to "task space" is a conceptually large jump from a simple idea. Single-point transduction (Netanyahu et al. 2023) only shifts within the same function; RTE shifts across functions. It requires the task manifold to be connected by structured transformations, but once satisfied, extrapolation becomes solvable.
- The "geometric shortcut" of Task2Vec + nearest-neighbor anchor + embedding difference for transformation is efficient and theoretically grounded in parameter regimes (Fisher Information embeddings are structure-preserving). This trick can be reused in any differentiable meta-learning scenario.
- Using "Likelihood as Score" for candidate ranking in LLMs essentially turns extrapolation into retrieval + verification, requiring no new training modules. This pattern can migrate to other tasks involving the assembly of known tools, such as tool-use agents or program synthesis.
- In Sparse Parity, RTE improves $|S|=6$ accuracy from 52.86% to 66.07%, showing that for logic tasks requiring recursive expansion, framing the reasoning as "known subtask + one structural expansion" is more reliable than one-shot answers.

## Limitations & Future Work

- **Strong Structural Assumption**: It must be assumed that the target task can be decomposed into an anchor and transformation seen during training; otherwise, the method is inapplicable. RTE is not a universal black box and fails on completely "alien" tasks.
- **Requirement for Relation Meta-labels in Discrete Regimes**: Training requires ground-truth task relations (e.g., $f_j = f_i \circ g$), which are often absent in real-world data. The appendix suggests self-labeling schemes but these are not yet fully validated.
- **Inference Computational Cost**: Strategy B involves candidate search, which is more expensive than a single forward pass. Multi-step extrapolation (chaining) would cause the search space and error to explode; the paper only demonstrates shallow chains.
- **Embeddings Sensitive to Pre-training**: Task2Vec relies on Fisher Information matrices, requiring high-quality pre-trained models. Stability of embedding estimation in few-shot ($k<10$) settings is a potential issue.
- **Future Directions**: Learning transformations $\phi$ as continuous differentiable latent variables, or introducing Bayesian optimization/evolutionary search instead of brute-force. SSL pretext tasks could self-label discrete relations to bypass the need for meta-labels.

## Related Work & Insights

- **vs MAML / Reptile**: These learn initializations for fast adaptation, assuming new tasks are in the local neighborhood of the training distribution. RTE explicitly models inter-task transformations for true extrapolation beyond the support.
- **vs Netanyahu et al. (2023)**: They perform transduction in the input space ("known point + offset" on the same function); RTE moves this to the task level, solving the "lack of natural task indexing" using Task2Vec.
- **vs Statistical Extrapolation (Pfister & Bühlmann 2024)**: Others rely on directional derivatives or causal mechanisms. RTE relies on relational structures and learned transformation operators.
- **vs Vafa et al. (2025) World Models**: These studies focus on whether models learn "mechanisms" vs. "heuristics." RTE explicitly encodes "mechanisms" into relational operators as a constructive world model design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Upgrading transduction to the task level and unifying three extrapolation regimes is highly distinct.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic functions, polynomials, parity, and CodeIO with clear oracle comparisons; however, real-world scientific data (e.g., physical extrapolation) is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual framework; well-presented formulas and algorithms. Some assumption explanations are slightly abstract.
- Value: ⭐⭐⭐⭐ Provides an actionable engineering path for OOD/extrapolation research; the prompt-as-decomposition approach is useful for LLM reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/self_supervised/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[NeurIPS 2025\] A Joint Learning Approach to Hardware Caching and Prefetching](../../NeurIPS2025/self_supervised/a_joint_learning_approach_to_hardware_caching_and_prefetching.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)

</div>

<!-- RELATED:END -->
