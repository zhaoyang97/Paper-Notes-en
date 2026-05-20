---
title: >-
  [Paper Note] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn
description: >-
  [ICML 2026][LLM Reasoning][many-shot ICL] This paper systematically reveals that the "rules of thumb" for many-shot ICL in non-reasoning tasks **completely fail** in CoT reasoning tasks—similarity retrieval is actually h…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "many-shot ICL"
  - "chain-of-thought"
  - "in-context test-time learning"
  - "demonstration ordering"
  - "curvature regularization"
date: 2026-05-08
content_hash: b0a6d942e17c393a
---

# Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

**Conference**: ICML 2026  
**arXiv**: [2605.13511](https://arxiv.org/abs/2605.13511)  
**Code**: None  
**Area**: Large Model Reasoning / In-Context Learning / Chain-of-Thought  
**Keywords**: many-shot ICL, chain-of-thought, in-context test-time learning, demonstration ordering, curvature regularization

## TL;DR
This paper systematically reveals that the "rules of thumb" for many-shot ICL in non-reasoning tasks **completely fail** in CoT reasoning tasks—similarity retrieval is actually harmful, and order sensitivity increases with the number of shots. The paper reinterprets successful many-shot CoT as "in-context test-time learning," and proposes the CDS method, which sorts demonstrations by embedding trajectory curvature, achieving a 5.42 pp improvement on 64-shot geometry problems.

## Background & Motivation

**Background**: Long-context LLMs make many-shot ICL feasible. Prior work (Bertsch et al., Baek et al.) observed three rules in non-reasoning tasks (classification, simple QA): (1) performance steadily improves as the number of shots increases; (2) order sensitivity decreases with more shots; (3) similarity-based retrieval (top-k most similar) improves performance. Meanwhile, chain-of-thought (CoT) has become standard for complex reasoning, but CoT-ICL is mostly studied in few-shot settings.

**Limitations of Prior Work**: When CoT is combined with many-shot (i.e., many-shot CoT-ICL), do these three empirical rules still hold? This has not been systematically studied. If the rules still hold, engineering can continue with retrieval/shot stacking; if not, the entire prompt engineering paradigm must be reconsidered. This is not just an engineering issue, but also relates to the fundamental debate on whether ICL is "scalable pattern matching" or "true learning."

**Key Challenge**: CoT demonstrations are much longer (in geometry tasks, a single CoT is about 30× longer than BANKING77), contain internal procedural reasoning chains, and require higher model understanding. These properties mean that the traditional many-shot intuition of "more is better, retrieval is right" may not hold in CoT scenarios. If ICL is truly "learning," then demonstrations act as supervision and order is akin to curriculum, requiring gradual progression as in teaching; from a pattern-matching perspective, order should not matter.

**Goal**: (1) Systematically characterize the scaling, retrieval, and ordering behaviors of many-shot CoT-ICL; (2) Identify the root causes for the failure of empirical rules; (3) Propose a new perspective to unify the observed phenomena and guide demonstration selection/ordering.

**Key Insight**: Many-shot CoT should be viewed as **in-context test-time learning**: the long-context window is not just a "retrieval cache," but an implicit curriculum, and the model's forward pass is a form of gradient-free adaptation. This perspective naturally leads to two pedagogical principles: (P1) demonstrations must be **understandable to the model** to serve as effective supervision; (P2) demonstration order must **transition smoothly**, avoiding abrupt conceptual jumps that disrupt the implicit learning trajectory.

**Core Idea**: Based on P2, treat demonstration order as a trajectory in embedding space, where **total curvature** (sum of angles between adjacent displacements) quantifies the "smoothness" of the order; minimizing total curvature yields a coherent in-context curriculum—this is Curvilinear Demonstration Selection (CDS).

## Method

### Overall Architecture
The paper first conducts extensive diagnostic experiments to expose the failure of the three rules, then reconstructs the theory from the in-context test-time learning perspective, and finally implements CDS. **Diagnostic phase** uses 4 non-reasoning LLMs (LLaMA 3.1 8B / 3.3 70B / Qwen2.5 7B / 14B) and 4 reasoning LLMs (Qwen3 8B / 14B / QwQ 32B / DeepSeek-R1 685B), running 1-128 shots on classification tasks (SuperGLUE, NLU, TREC, BANKING77) and math/narrative reasoning tasks (GSM8K, MATH's geometry / number_theory / counting_and_probability, DetectiveQA), all evaluated with open-ended generation + exact match. **CDS algorithm** seeks a permutation $O = [\mathbf{d}_{\pi(1)}, \ldots, \mathbf{d}_{\pi(n)}]$ of $n$ demonstrations that minimizes total curvature $\Theta(O) = \sum_{t=2}^{n-1} \arccos\!\left(\frac{\mathbf{v}_t \cdot \mathbf{v}_{t+1}}{\|\mathbf{v}_t\|\|\mathbf{v}_{t+1}\|}\right)$, where $\mathbf{v}_t = \tilde{\mathbf{e}}_t - \tilde{\mathbf{e}}_{t-1}$ is the displacement vector between adjacent projected demonstration embeddings.

### Key Designs

1. **Diagnostic Experiments: Exposing the Simultaneous Failure of Three Empirical Rules in CoT Reasoning**:

    - Function: Uses controlled comparisons to rigorously test whether each rule still holds.
    - Mechanism: (A) **Scaling**: On reasoning tasks like geometry/number_theory, non-reasoning LLMs show **unstable or even declining** performance as shot count increases (e.g., LLaMA 3.3 70B exhibits negative gain in CoT-ICL); only reasoning-oriented LLMs (Qwen3, QwQ, R1) show monotonic positive scaling. Table 1 further shows that disabling thinking mode on Qwen3 drops geometry accuracy by 7 pp, proving reasoning prior is necessary for scaling. (B) **Retrieval**: Embedding cosine top-k (most similar) vs bottom-k (least similar); on BANKING77, top-k significantly outperforms bottom-k (validating retrieval hypothesis), but on geometry/number_theory/DetectiveQA, top-k performs **worst**—semantic similarity does not predict procedural compatibility. (C) **Ordering**: Standard deviation over 5 random permutations; in non-reasoning tasks, std decreases with more shots, but in reasoning tasks, std **increases** with more shots, indicating strong and deepening path dependence.
    - Design Motivation: By systematically testing each "common sense" of many-shot ICL in CoT reasoning, the simultaneous failure across three independent dimensions convincingly demonstrates that "CoT-ICL is fundamentally different," not just a dataset coincidence. This "triangulation" is a paradigm for diagnostic empirical work, more persuasive than a single phenomenon.

2. **Direct Evidence for Procedure Absorption: Corrupted CoT Ablation**:

    - Function: Separates the hypotheses "the model only uses the final answer $y$" and "the model truly absorbs the intermediate reasoning $C$."
    - Mechanism: On geometry, constructs two prompt sets—normal $(x_i, C_i, y_i)$ and **procedurally corrupted** $(x_i, C_0, y_i)$, where all rationales are replaced with the first demonstration's chain, but each question and final answer are retained. This controls for format, context length, and $x \to y$ mapping, only altering $C$. Table 2: At $n=16$, the two groups are nearly identical; at $n=128$, the corrupted version causes Qwen3-8B to drop 1.25 pp and Qwen3-14B to drop 2.51 pp.
    - Design Motivation: Provides direct counterfactual evidence for whether the model is truly reading the demonstration's procedure. Small differences with short prompts indicate the model can learn from both IO and CoT; large differences with long prompts indicate procedure is the true scaling signal—offering hard evidence for the "in-context test-time learning" perspective, much more convincing than philosophical arguments.

3. **Curvilinear Demonstration Selection (CDS): Minimizing Total Curvature Ordering**:

    - Function: Based on the smooth transition principle, finds an ordering of $n$ demonstrations that yields the smoothest implicit learning trajectory.
    - Mechanism: (i) Each demonstration $\mathbf{d}_i$ (question + CoT + answer) is encoded using Qwen3-Embedding-4B into $\mathbf{e}_i \in \mathbb{R}^d$, crucially using the **full demonstration** rather than just the question—since order effects depend on procedural content, question alone misses CoT structure. (ii) All prompt embeddings are projected to a low-dimensional subspace $\tilde{\mathbf{e}}_i \in \mathbb{R}^{d'}$ for stable curvature estimation. (iii) Local curvature $\theta_i = \arccos\!\left(\frac{(\tilde{\mathbf{e}}_i - \tilde{\mathbf{e}}_{i-1}) \cdot (\tilde{\mathbf{e}}_{i+1} - \tilde{\mathbf{e}}_i)}{\|\cdot\|\|\cdot\|}\right)$ is defined as the angle between adjacent displacements, and total curvature $\Theta(O) = \sum_{i=2}^{n-1}\theta_i$. (iv) Searches for a permutation minimizing $\Theta$ (algorithm details in Section 6).
    - Design Motivation: The authors observe that ordering curvature is significantly negatively correlated with accuracy (overall $r=-0.547$, geometry $-0.545$, counting $-0.628$), so minimizing curvature is a natural objective. To rule out "just clustering similar items," they introduce a high-curvature reverse baseline—preserving local neighborhoods but reversing the curvature objective to create abrupt transitions—and find CDS still outperforms, proving that **smooth transition itself** rather than clustering is the causal factor. This causal smoothness ablation is a methodological highlight.

### Loss & Training
CDS is a **pure inference-time** algorithm, with no training involved. The underlying embedding model is Qwen3-Embedding-4B (off-the-shelf). Evaluation models include LLaMA, Qwen2.5, Qwen3, QwQ, and DeepSeek-R1 series, with prompt context up to 131K tokens and shot numbers up to $n \leq 128$.

## Key Experimental Results

### Main Results
CDS improvements on Qwen3 series (geometry / number theory / DetectiveQA):

| Task | Model | Setting | n=64 Gain |
|---|---|---|---|
| Geometry | Qwen3-14B | CDS vs Random Order | **+5.42 pp** |
| Geometry | Qwen3-14B | n=128 + thinking on | 73.07% vs n=16 at 66.18% |
| Geometry | Qwen3-14B | thinking on vs off (n=128) | 73.07 vs 65.76 |
| Number_theory | Qwen3-14B | thinking on vs off (n=128) | 91.30 vs 88.15 |
| DetectiveQA | Qwen3-8B | thinking on vs off (n=128) | 69.48 vs 66.88 |

### Ablation Study

| Setting | Behavior | Notes |
|---|---|---|
| CDS (low curvature) | Best | Full method |
| High-curvature baseline | Significantly worse | Same embedding neighborhood, reversed curvature objective |
| Similarity top-k retrieval | Worse | Semantic similarity does not predict procedural compatibility |
| Similarity bottom-k | Between top-k and original | Counterintuitive |
| Procedurally corrupted CoT (n=128) | Significantly worse (-1.25 to -2.51 pp) | Shows procedure is key |
| Thinking mode disabled | Significantly worse | Reasoning prior is necessary for scaling |
| Non-reasoning LLM + CoT-ICL | Scaling unstable or negative | Model class determines ability to absorb CoT |

### Key Findings
- **CoT-ICL is not scalable pattern matching**: Similarity retrieval is effective on BANKING77 (non-reasoning) but **reversed** on geometry/number_theory/DetectiveQA (reasoning), refuting the retrieval hypothesis for reasoning tasks.
- **Order sensitivity increases with shot count** (opposite to non-reasoning tasks): Randomly ordering 100+ demonstrations leads to more "conceptual mutations," triggering procedural discontinuity.
- **Self-generated CoT outperforms ground-truth CoT**: On weaker models, self-generated CoT (even with wrong answers) outperforms dataset CoT; this advantage narrows as models strengthen, validating P1 ("understandability first").
- **Scaling gap between reasoning-oriented and non-reasoning LLMs** is rooted in the thinking token—it treats demonstrations as procedural supervision, not just IO pattern matching.
- **Total curvature is significantly negatively correlated with accuracy** (geometry $r=-0.545$, counting $r=-0.628$), so minimizing curvature is a quantifiable, non-ad-hoc objective.

## Highlights & Insights
- **In-context test-time learning is a unifying anchor**: From this perspective, scaling failure (P1 violation), similarity failure (procedure mismatches surface), and order sensitivity (P2 violation) are all explained—long context is an implicit curriculum, not a cache. This unified explanation provides clear design guidance for future prompt engineering.
- **Self-generated CoT outperforms ground-truth CoT** is a highly counterintuitive but reasonable finding: models "understand" their own CoT better, and even with wrong answers, benefit from procedural context. Incorporating this into prompt pipelines is a free engineering upgrade—let weak models train themselves with their own CoT.
- **Total curvature as an ordering objective**: Quantifies the abstract "smooth transition" as the sum of angles between adjacent displacements, both geometrically intuitive and computationally feasible; causal smoothness ablation with a high-curvature reverse baseline rules out "similarity clustering" confounds, demonstrating methodological rigor.
- **Using full demonstration for embedding** is a crucial detail: question-only embeddings miss CoT procedural structure; using question + CoT + answer allows curvature to reflect procedural transition difficulty.

## Limitations & Future Work
- The core "smooth transition" assumption of CDS depends on the embedding space's ability to represent procedural content; if the embedding model poorly encodes CoT internal structure (e.g., instruction-only models), curvature signals may be distorted and the method's effectiveness is not guaranteed.
- Experiments focus on math and narrative reasoning; it remains unverified whether more complex reasoning types (programming, theorem proving, agentic planning) also exhibit the curvature-performance negative correlation.
- The paper does not provide the optimization algorithm complexity or comparisons for CDS (a TSP-like ordering problem); minimization itself may become a bottleneck for large shot counts.
- The advantage of "self-generated CoT over ground-truth" narrows as models strengthen—but whether future strong models can entirely skip self-generation is not quantified.
- Future work could explore injecting the curvature term as a differentiable regularizer during training (curriculum learning fine-tuning), or combining with RAG chunk ordering for retrieval-aware curriculum.

## Related Work & Insights
- **vs Bertsch et al. / Baek et al. (many-shot ICL)**: They found scale + order robustness + effective retrieval in non-reasoning tasks; this paper demonstrates all three fail simultaneously in CoT reasoning, providing a key corrective.
- **vs Auto-CoT (Zhang et al.) / Dr.ICL (Luo et al.)**: These focus on CoT demonstration selection in few-shot settings; this paper addresses the new dynamics in the many-shot regime.
- **vs Test-time scaling (Snell et al.)**: Test-time scaling mainly increases inference computation via sample-and-revise; this paper views many-shot CoT as another form of test-time scaling, treating demonstrations as in-context supervision.
- **Insights**: (1) Any engineering relying on "long context for large-scale retrieval" (RAG, agent memory) should reconsider the impact of ordering; (2) Educational psychology's "zone of proximal development" and textbook curve concepts have concrete, quantifiable counterparts in prompt engineering, potentially spawning a new subfield of "pedagogical prompting."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically refute the transfer of many-shot ICL rules of thumb to CoT, reconstructs the perspective and implements CDS
- Experimental Thoroughness: ⭐⭐⭐⭐ 4+4 models × multi-task × multi-shot × multi-seed, covers three major dimensions with causal ablation; but CDS evaluation mainly on Qwen3
- Writing Quality: ⭐⭐⭐⭐⭐ Clear chain from diagnosis to theory to algorithm to validation, apt pedagogical analogies
- Value: ⭐⭐⭐⭐⭐ Wake-up call for all prompt engineering relying on long context, CDS is a plug-and-play engineering upgrade

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations](../../ACL2025/llm_reasoning/cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](../../NeurIPS2025/llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)

</div>

<!-- RELATED:END -->
