---
title: >-
  [Paper Note] MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier
description: >-
  [ICML 2026][Scientific Computing][hypothesis generation] MOOSE-Star decomposes the problem of "training an LLM to directly generate scientific hypotheses"—which originally requires searching a $\mathcal{O}(N^k)$ combinat…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "hypothesis generation"
  - "inspiration retrieval"
  - "hierarchical search"
  - "tractable training"
  - "TOMATO-Star"
date: 2026-05-08
content_hash: 50f4a6ebd62cb91d
---

# MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier

**Conference**: ICML 2026  
**arXiv**: [2603.03756](https://arxiv.org/abs/2603.03756)  
**Code**: https://github.com/ZonglinY/MOOSE-Star (available)  
**Area**: LLM reasoning / scientific discovery / decomposed training  
**Keywords**: hypothesis generation, inspiration retrieval, hierarchical search, tractable training, TOMATO-Star

## TL;DR
MOOSE-Star decomposes the problem of "training an LLM to directly generate scientific hypotheses"—which originally requires searching a $\mathcal{O}(N^k)$ combinatorial space—into two sequential subtasks: "inspiration retrieval + hypothesis composition." By further stacking hierarchical tree retrieval, bounded composition, and motivation planning, the optimal complexity is reduced from exponential to $\mathcal{O}(\log N)$. The authors also release the TOMATO-Star dataset with 108,717 decomposition-annotated papers.

## Background & Motivation
**Background**: Most work on LLMs for scientific discovery focuses on "how to use LLMs at inference" or "fine-tuning with external feedback" (e.g., reviewer feedback, rule-based scoring, reward aligned with data). Direct modeling and training of the core conditional distribution $P(\text{hypothesis}\mid\text{background})$ is almost unexplored.

**Limitations of Prior Work**: The authors theoretically point out that directly training $P(h\mid b)$ implicitly requires "finding the correct sequence of $k$ inspirations from a global scientific literature corpus of $N\approx 10^7$," resulting in a search space of $\mathcal{O}(N^k)$ (e.g., $N=10^7, k=3$ yields $\approx 10^{21}$). This "combinatorial complexity wall" makes end-to-end training mathematically ill-posed.

**Key Challenge**: One must either abandon direct modeling of $P(h\mid b)$ (the feedback-based route) or confront the combinatorial complexity head-on (which is infeasible). Both paths are problematic.

**Goal**: While retaining the ambition of directly modeling $P(h\mid b)$, the aim is to reduce training complexity to a level manageable by modern compute, and to provide reproducible datasets and open-source code.

**Key Insight**: Leveraging the probabilistic decomposition theorem from MOOSE-Chem, the authors approximate $P(h\mid b)\approx \prod_j P(i_j\mid b,h_{j-1},\mathcal{I})\cdot P(h_j\mid b,h_{j-1},i_j)$ as a sequence of "inspiration retrieval + incremental composition." Previously used only for inference, this decomposition is here upgraded to a trainable objective.

**Core Idea**: The intractable $\mathcal{O}(N^k)$ problem is reduced to tractable $\mathcal{O}(k\cdot N)$ sequential subtasks, and then hierarchical tree search, bounded composition, and motivation planning further compress the retrieval step from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$.

## Method

### Overall Architecture
The training pipeline has three layers:  
(1) **Data**: R1 / R1-distill-Qwen is used to decompose 108,717 open-access papers (2020–2025), yielding (research background $b$, hypothesis $h$, inspirations $\{i_j\}$) triplets. $h$ is further split into $\Delta h_1,\ldots,\Delta h_k$, each written in a three-level structure: Motivation, Mechanism, Methodology.  
(2) **Model**: $P(h\mid b)$ is decomposed into two RFT tasks: Inspiration Retrieval (IR) and Hypothesis Composition (HC). IR uses a pool of 1 positive + 14 hard negatives; HC employs a rubric-based evaluator for rejection sampling.  
(3) **Inference**: The full literature is organized as a semantic retrieval tree. The motivation variable dynamically prunes irrelevant subtrees, and HC is trained within a bounded tolerance radius $M$ to make composition robust to retrieval errors.

### Key Designs

1. **Decomposed Sequential Training (IR + HC)**:

    - **Function**: Replaces "end-to-end learning of $P(h\mid b)$" with "first learn retrieval, then incremental composition," repeated $k$ times.
    - **Mechanism**: By chain rule, $P(h\mid b)\approx \prod_{j=1}^{k} P(i_j\mid b,h_{j-1},\mathcal{I})\cdot P(h_j\mid b,h_{j-1},i_j)$. The IR task is "select the most relevant paper from 15 candidates" (input: title+abstract, output: CoT reasoning + selection). The HC task is "given ground-truth $i_j$, write the incremental hypothesis $\Delta h_j$." Both use teacher-based RFT. Overall complexity becomes $\mathcal{O}(k\cdot(N+1))$ instead of $\mathcal{O}(N^k)$.
    - **Design Motivation**: Replacing the exponential Cartesian product with $k$ linear summations is key to making the problem trainable. IR/HC are two clear, supervised, and evaluable tasks, more stable than scoring $h$ as a whole.

2. **Bounded Composition**:

    - **Function**: Makes the HC model robust to cases where the retrieved inspiration is not exactly the ground-truth $i^*$.
    - **Mechanism**: Defines a semantic tolerance neighborhood $\mathcal{I}_{i^*}\subset\mathcal{I}$ centered at $i^*$ with size $M$. During training, "approximate inspirations" are randomly sampled from this neighborhood and fed to HC, teaching the model to compose effective $\Delta h_j$ even with neighboring concepts. This relaxes the retrieval precision from "1/N exact match" to "1/(N/M) fuzzy match," further reducing IR's effective search space.
    - **Design Motivation**: Even with hierarchical tree retrieval at $\mathcal{O}(\log N)$, the final layer may not be precise. Bounded composition explicitly models "retrieval error" in the training distribution, akin to noise-aware training, ensuring pipeline robustness under real-world noise.

3. **Motivation-Guided Hierarchical Search**:

    - **Function**: Replaces "linear scan of $N$ papers" with "top-down traversal of a semantic tree in log N steps," using the motivation variable for pruning.
    - **Mechanism**: The full literature is clustered into a retrieval tree; at each step, the most relevant branch among current node's children is selected, ideally yielding search depth $\mathcal{O}(\log N)$. An explicit motivation variable $m$ (from the Motivation layer of $\Delta h$) is attached to the background, serving as the "generative root" to dynamically prune subtrees irrelevant to the current goal, reducing the searchable space from $N$ to $N_m\ll N$.
    - **Design Motivation**: A semantic tree alone only reduces retrieval steps, but "which subtree to search" remains open. The motivation variable provides a generative direction control signal, enabling true scaling at inference time.

### Loss & Training
Both IR and HC use Rejection Sampling Fine-Tuning (RFT) with CoT supervision: for each sample, N CoT traces are generated, low-quality ones are filtered by a rubric evaluator based on "selection correctness/composition quality," and high-quality traces are used for SFT. The HC rubric checks all three layers: Motivation, Mechanism, Methodology. The TOMATO-Star dataset is included only after passing four automated quality checks (necessity, sufficiency, exclusivity, non-redundancy).

## Key Experimental Results

### Main Results

| Dimension | Configuration | Highlights |
|-----------|--------------|------------|
| Data scale | 108,717 open-access papers, 38,400 GPU·hours | Training set: 2020-09/2025, Test set: 2025-10 (strict temporal split, no leakage) |
| Complexity (worst → best) | $\mathcal{O}(N^k)$ → $\mathcal{O}(\log N)$ | Stepwise compression via IR/HC decomposition, tree retrieval, and motivation pruning |
| Test-time scaling | brute-force vs. MOOSE-Star | Brute-force quickly hits the "complexity wall" on multi-inspiration tasks; MOOSE-Star's success rate rises with inference budget |
| Inspiration hit at inference | IR significantly outperforms random/nearest-neighbor baselines in 1 positive 14 negative pool | Demonstrates effectiveness of generative selection + CoT supervision (see § F for details) |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Remove Bounded Composition ($M=1$) | HC becomes sensitive to retrieval noise, overall task success drops | Validates necessity of robust composition under imperfect retrieval |
| Remove Motivation variable | Tree search path lengthens, pruning fails, success rate drops at same inference budget | Motivation is key to effective pruning |
| End-to-end training $P(h\mid b)$ (baseline) | Training fails to converge / composition trace cannot be distilled | § 7.1 shows direct distillation from $b\to h$ reasoning trace is infeasible |
| Brute-force test-time sampling | Hits "complexity wall" on multi-inspiration tasks | Contrasts with MOOSE-Star's scalable hierarchical search |

### Key Findings
- The fundamental reason direct training of $P(h\mid b)$ fails is the implicit combinatorial search space, not data scarcity or model size—this is a strong critique of the "feedback-driven discovery" paradigm.
- "Decomposition + hierarchy + tolerance + pruning" is a transferable paradigm: each step addresses a bottleneck of one complexity order, and only their combination reduces from $N^k$ to $\log N$.
- TOMATO-Star's (b, h, i) + (Motivation, Mechanism, Methodology) dual-layer structure is itself an upgrade in LLM-discovery dataset design, surpassing "summary-style hypotheses."
- Strict temporal split (test set only includes papers after 2025-10) ensures evaluation is not contaminated by pretraining—a practice worth emulating as LLMs grow larger.

## Highlights & Insights
- For the first time, the root cause of "why $P(h|b)$ is untrainable" is rigorously argued from a complexity perspective ($\mathcal{O}(N^k)$ vs $\mathcal{O}(\log N)$), elevating "scientific discovery LLMs" from engineering to theoretically grounded research.
- Reinterpreting a probabilistic decomposition theorem (previously used only at inference) as a training objective is the most critical leap since MOOSE-Chem, analogous to how RL decomposes the Bellman equation into TD-updates.
- Bounded Composition relaxes the requirement for perfect alignment between retrieval and composition, accommodating real-world retrieval noise—highly relevant to search/RAG practice.
- The release of a 108k structured decomposition dataset, full training/inference code, and models sets a new reproducibility standard in the historically weak area of "scientific discovery."

## Limitations & Future Work
- The current system still relies on the assumption "author citation = ground-truth inspiration," which biases toward explicitly acknowledged influences and lacks sensitivity to "unacknowledged but impactful" inspirations.
- The 1 positive 14 negative IR setup is still a constrained approximation; real literature pools are much larger, and hierarchical search lacks self-correction if the root is misselected.
- The tolerance radius $M$ in Bounded Composition is a hyperparameter; too small degenerates to exact match, too large causes HC output to lose generalization. The paper does not provide a systematic $M$ selection strategy.
- Validation is mainly in biology, chemistry, and medicine; whether this decomposition applies to ML/CS fields with denser citation structures and shorter inspiration chains remains to be tested.

## Related Work & Insights
- **vs MOOSE-Chem (Yang et al., 2025b)**: MOOSE-Chem only uses probabilistic decomposition at inference; this work upgrades the same decomposition to a training objective—a key shift from "inference tool" to "training paradigm."
- **vs feedback-driven training (Weng/Behzadifar/Goel et al.)**: These approaches fine-tune hypotheses using reviewer/data/rule feedback, without touching the core distribution $P(h\mid b)$; this is the first work to directly train this distribution.
- **vs O'Neill et al. (2025)**: Also attempts direct modeling of $P(h\mid b)$ via distillation, but § 7.1 here shows that distilling the reasoning trace from $b\to h$ is infeasible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rigorously attributes the root cause of "untrainable scientific discovery LLMs" to combinatorial complexity, and provides an executable path to compress it to $\log N$—a rare blend of theoretical and engineering novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Data scale and GPU investment (38,400 A800 hours) are substantial, but ablation studies are more qualitative, lacking hard comparisons on unified benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Complexity derivations are clear, causal chains between modules are smooth, and the rationale for each complexity reduction step is well articulated.
- Value: ⭐⭐⭐⭐⭐ Provides framework, dataset (TOMATO-Star 108k), code, and trained models, serving as a de facto baseline for the "LLM-for-discovery" direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](../../NeurIPS2025/llm_reasoning/the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)

</div>

<!-- RELATED:END -->
