---
title: >-
  [Paper Note] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn
description: >-
  [ICML 2026][LLM Reasoning][many-shot ICL] This paper systematically reveals that the "rules of thumb" for many-shot ICL in non-reasoning tasks **completely fail** in CoT reasoning tasks—specifically…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "many-shot ICL"
  - "chain-of-thought"
  - "in-context test-time learning"
  - "demonstration ordering"
  - "curvature regularization"
date: 2026-05-08
content_hash: 816a680137f84265
---

# Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

**Conference**: ICML 2026  
**arXiv**: [2605.13511](https://arxiv.org/abs/2605.13511)  
**Code**: None  
**Area**: Large Model Reasoning / In-Context Learning / Chain-of-Thought  
**Keywords**: many-shot ICL, chain-of-thought, in-context test-time learning, demonstration ordering, curvature regularization

## TL;DR
This paper systematically reveals that the "rules of thumb" for many-shot ICL in non-reasoning tasks **completely fail** in CoT reasoning tasks—specifically, similarity retrieval becomes harmful and order sensitivity increases with the number of shots. By reinterpreting successful many-shot CoT as "in-context test-time learning," the authors propose the CDS method, which orders demonstrations by embedding trajectory curvature, achieving a 5.42 pp improvement on 64-shot geometry problems.

## Background & Motivation

**Background**: Long-context LLMs have enabled many-shot ICL. Prior works (Bertsch et al., Baek et al.) observed three patterns in non-reasoning tasks (classification, simple QA): (1) performance increases steadily with the number of shots; (2) sensitivity to demonstration order decreases as shots increase; (3) similarity retrieval (top-k most similar) enhances performance. Meanwhile, Chain-of-Thought (CoT) has become the standard for complex reasoning, but CoT-ICL is mostly studied in few-shot settings.

**Limitations of Prior Work**: Whether these three empirical laws hold when CoT is combined with many-shot (i.e., many-shot CoT-ICL) has not been systematically investigated. If the laws hold, prompt engineering can continue with traditional retrieval/stacking strategies; if they are violated, the entire prompt engineering paradigm requires rethinking. This is not just an engineering issue but concerns the fundamental debate of whether ICL is "scaled pattern matching" or "genuine learning."

**Key Challenge**: CoT demonstrations are significantly longer (e.g., a single CoT in geometry tasks is ~30× longer than in BANKING77), contain internal procedural reasoning chains, and demand higher-level understanding from the model. these properties suggest that traditional many-shot intuitions like "more is better" and "retrieval is correct" may not hold in CoT scenarios. If ICL truly involves "learning," then demonstrations serve as supervision and the order acts as a curriculum, requiring a gradual progression similar to teaching; whereas in a pattern-matching view, order should not matter.

**Goal**: (1) Systematically characterize the scaling, retrieval, and ordering behaviors of many-shot CoT-ICL; (2) Identify the root causes for the failure of empirical laws; (3) Propose a new perspective to unify these phenomena and guide demonstration selection/ordering.

**Key Insight**: many-shot CoT is viewed as **in-context test-time learning**: the long-context window is not merely a "retrieval cache" but an implicit curriculum, and the model's forward pass functions as a gradient-free adaptation. This perspective leads to two pedagogical principles: (P1) demonstrations must be **understandable to the model** to serve as effective supervision; (P2) demonstration sequences must provide **smooth transitions** to avoid abrupt conceptual jumps that disrupt the implicit learning trajectory.

**Core Idea**: Based on P2, the demonstration sequence is treated as a trajectory in the embedding space. The **total curvature** (the sum of angles between adjacent displacements) is used as a quantitative metric for "smoothness." Minimizing the total curvature yields a coherent in-context curriculum—referred to as Curvilinear Demonstration Selection (CDS).

## Method

### Overall Architecture
The paper first conducts extensive diagnostic experiments to expose the failure of the three empirical laws, then reconstructs the theory through the lens of in-context test-time learning, and finally implements CDS. **The diagnostic phase** involves 4 non-reasoning LLMs (LLaMA 3.1 8B / 3.3 70B / Qwen2.5 7B / 14B) and 4 reasoning-oriented LLMs (Qwen3 8B / 14B / QwQ 32B / DeepSeek-R1 685B), tested on classification tasks (SuperGLUE, NLU, TREC, BANKING77) and math/narrative reasoning tasks (GSM8K, MATH subdomains, DetectiveQA) across 1-128 shots using open-ended generation and exact match evaluation. **The CDS algorithm** finds a permutation $O = [\mathbf{d}_{\pi(1)}, \ldots, \mathbf{d}_{\pi(n)}]$ for $n$ demonstrations that minimizes the total curvature $\Theta(O) = \sum_{t=2}^{n-1} \arccos\!\left(\frac{\mathbf{v}_t \cdot \mathbf{v}_{t+1}}{\|\mathbf{v}_t\|\|\mathbf{v}_{t+1}\|}\right)$, where $\mathbf{v}_t = \tilde{\mathbf{e}}_t - \tilde{\mathbf{e}}_{t-1}$ represents the displacement vector of projected embeddings.

### Key Designs

1. **Diagnostic Experiments: Exposing the simultaneous failure of empirical laws in CoT reasoning**:
    - **Function**: Utilizes a comparative design to examine whether standard laws still hold.
    - **Mechanism**: (A) **Scaling**: Running reasoning tasks on non-reasoning LLMs shows that increasing shots results in unstable or even **declining performance** (e.g., LLaMA 3.3 70B showing negative gain in CoT-ICL); only reasoning-oriented LLMs (Qwen3, QwQ, R1) demonstrate monotonic positive scaling. (B) **Retrieval**: top-k similarity significantly outperforms bottom-k in BANKING77, but top-k is **consistently the worst** in geometry/number_theory/DetectiveQA—semantic similarity does not predict procedural compatibility. (C) **Ordering**: The standard deviation across random permutations decreases with shots in non-reasoning tasks but **increases with shots** in reasoning tasks, indicating strong and deepening path dependence.
    - **Design Motivation**: By contrasting many-shot "common sense" against CoT reasoning across three independent dimensions, the authors demonstrate that "CoT-ICL is a different beast" rather than a dataset-specific fluke.

2. **Direct Evidence for Procedure Absorption: Corrupted CoT Ablation**:
    - **Function**: Decouples the hypothesis that "the model only uses the final answer $y$" from "the model truly absorbs the intermediate reasoning $C$."
    - **Mechanism**: Two prompt versions are constructed for geometry tasks: a normal version $(x_i, C_i, y_i)$ and a **procedurally corrupted** version $(x_i, C_0, y_i)$, where all rationales are replaced by the chain from the first demonstration while keeping the questions and final answers intact. At $n=128$, the corrupted version causes a performance drop of 1.25 pp for Qwen3-8B and 2.51 pp for Qwen3-14B.
    - **Design Motivation**: This provides counterfactual evidence that the model "reads" the procedural content of demonstrations. The larger gap in long prompts suggests that procedure is the true signal for scaling.

3. **Curvilinear Demonstration Selection (CDS): Minimizing Total Curvature**:
    - **Function**: Finds an optimal permutation that makes the implicit learning trajectory as smooth as possible.
    - **Mechanism**: (i) Each demonstration $\mathbf{d}_i$ (question + CoT + answer) is encoded into $\mathbf{e}_i \in \mathbb{R}^d$ using Qwen3-Embedding-4B. (ii) All embeddings are projected into a low-dimensional subspace $\tilde{\mathbf{e}}_i \in \mathbb{R}^{d'}$ for stable curvature estimation. (iii) Local curvature $\theta_i$ is defined as the angle between adjacent displacement vectors, with the objective to minimize $\Theta(O) = \sum \theta_i$.
    - **Design Motivation**: The authors observed a significant negative correlation between ordering curvature and accuracy (e.g., $r = -0.545$ for geometry). To prove causality, they compared CDS against a "high-curvature" baseline (which flips the goal to maximize abrupt transitions) and found CDS still superior, confirming that **smooth transition itself** is the causal factor.

### Loss & Training
CDS is a **purely inference-time** algorithm with no training involved. The underlying embedding model is Qwen3-Embedding-4B (off-the-shelf). Evaluation models include LLaMA, Qwen2.5, Qwen3, QwQ, and DeepSeek-R1 series, with contexts up to 131K tokens and shots up to $n = 128$.

## Key Experimental Results

### Main Results
CDS improvements on the Qwen3 series:

| Task | Model | Configuration | Gain at n=64 |
|---|---|---|---|
| Geometry | Qwen3-14B | CDS vs. Random | **+5.42 pp** |
| Geometry | Qwen3-14B | n=128 + thinking on | 73.07% vs. 66.18% (n=16) |
| Geometry | Qwen3-14B | thinking on vs. off (n=128) | 73.07 vs. 65.76 |
| Number_theory | Qwen3-14B | thinking on vs. off (n=128) | 91.30 vs. 88.15 |
| DetectiveQA | Qwen3-8B | thinking on vs. off (n=128) | 69.48 vs. 66.88 |

### Ablation Study

| Configuration | Behavior | Description |
|---|---|---|
| CDS (low curvature) | Best | Full method |
| High-curvature baseline | Significantly worse | Same neighborhood, reversed curvature objective |
| Similarity top-k retrieval | Worse | Semantic similarity fails to predict procedural compatibility |
| Similarity bottom-k | Between top-k and original | Counter-intuitive result |
| Procedurally corrupted CoT (n=128) | Significantly worse (-1.25 to -2.51 pp) | Proves procedure plays a key role |
| Thinking mode disabled | Significantly worse | Reasoning prior is a necessity for scaling |
| Non-reasoning LLM + CoT-ICL | Unstable/Negative scaling | Model class determines ability to absorb CoT |

### Key Findings
- **CoT-ICL is not scaled pattern matching**: Similarity retrieval works for non-reasoning tasks but is **inverted** for reasoning tasks, nullifying the retrieval hypothesis in reasoning scenarios.
- **Order sensitivity increases with shots**: Unlike non-reasoning tasks, more demonstrations lead to a higher probability of "conceptual mutations" if ordered randomly, triggering procedural incoherence.
- **Self-generated CoT outperforms ground-truth CoT**: On weaker models, self-generated CoTs (even with wrong answers) perform better than dataset-provided CoTs. This advantage diminishes as models get stronger, validating P1 ("understandability first").
- **Gaps in scaling between reasoning and non-reasoning LLMs** stem from the thinking mode, which extracts procedural supervision rather than treating IO as a pattern.
- **Total curvature is significantly negatively correlated with accuracy**, making minimum curvature a justifiable quantitative target rather than an ad-hoc heuristic.

## Highlights & Insights
- **In-context test-time learning provides a unified anchor**: This perspective explains scaling failures, retrieval failures, and order sensitivity under one roof: long contexts are implicit curricula, not caches.
- **Self-generated CoT superiority** is a counter-intuitive finding: models "understand" their own generated rationales better, even if imperfect, benefiting from the procedural context.
- **Total curvature as an ordering objective**: Quantifying "smooth transitions" as the sum of displacement angles is geometrically intuitive and computationally feasible. The causal ablation using high-curvature baselines effectively rules out clustering as a confounding factor.
- **Encoding full demonstrations** is a crucial detail: question-only embeddings lose the procedural structure of the CoT.

## Limitations & Future Work
- "Smooth transition" relies on the embedding space's ability to represent procedural content; if the embedding model fails to encode CoT structures well, the curvature signal becomes noisy.
- Experiments focus on math and narrative reasoning; whether more complex types like programming or agentic planning satisfy the curvature-performance correlation remains unverified.
- The complexity of the CDS optimization algorithm for very large $n$ is not detailed.
- Future work could explore injecting curvature as a differentiable regularizer during training or combining it with chunk ordering in RAG.

## Related Work & Insights
- **vs. Bertsch et al. / Baek et al. (many-shot ICL)**: This work serves as a critical corrective, showing that their findings on scaling and retrieval do not transfer to CoT reasoning.
- **vs. Auto-CoT / Dr.ICL**: While they focus on few-shot selection, this work explores the entirely new dynamics of many-shot settings.
- **vs. Test-time scaling (Snell et al.)**: While test-time scaling usually refers to increased inference compute via sampling, this work treats many-shot CoT as a form of test-time scaling where demonstrations act as in-context supervision.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically disprove many-shot ICL rules for CoT and reconstruct a curvature-based framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive cross-model and cross-task evaluation, although CDS is primarily tested on the Qwen3 series.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from diagnosis to theory to algorithm.
- Value: ⭐⭐⭐⭐⭐ A wake-up call for prompt engineering in long-context scenarios; CDS offers a plug-and-play upgrade.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)
- [\[ICML 2026\] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning](latentchem_from_textual_cot_to_latent_thinking_in_chemical_reasoning.md)

</div>

<!-- RELATED:END -->
