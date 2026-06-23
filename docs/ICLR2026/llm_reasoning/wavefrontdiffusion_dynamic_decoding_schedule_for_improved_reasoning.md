---
title: >-
  [Paper Note] WavefrontDiffusion: Dynamic Decoding Schedule for Improved Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Diffusion Language Model] Addressing the scheduling problem of "which tokens to determine first" during Diffusion Language Model (DLM) decoding, this paper proposes **WavefrontDiffusion**—a training-free dynamic scheduling strategy. It allows finalized tokens to expand candidate regions like water waves, ensuring each token is finalized only wh
tags:
  - ICLR 2026
  - LLM Reasoning
  - Diffusion Language Model
date: 2026-05-08
content_hash: 4388e490183f41e2
---
# WavefrontDiffusion: Dynamic Decoding Schedule for Improved Reasoning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4smJ6zY7vy](https://openreview.net/forum?id=4smJ6zY7vy)  
**Code**: https://github.com/page link (Placeholder link provided in the paper, to be open-sourced)  
**Area**: LLM Reasoning / Diffusion Language Models  
**Keywords**: Diffusion Language Models, Decoding Schedule, Wavefront Expansion, Mathematical Reasoning, Code Generation

## TL;DR
Addressing the scheduling problem of "which tokens to determine first" during Diffusion Language Model (DLM) decoding, this paper proposes **WavefrontDiffusion**—a training-free dynamic scheduling strategy. It allows finalized tokens to expand candidate regions like water waves, ensuring each token is finalized only when sufficient context is available. Across five reasoning and code benchmarks, it consistently outperforms the current strongest BlockDiffusion using the **exact same compute budget**.

## Background & Motivation

**Background**: Diffusion Language Models (DLMs) model text generation as an iterative denoising process of discrete token sequences: starting from a full sequence of `[MASK]`, multiple masked positions are predicted in parallel at each step, gradually converging to a clean sequence. Compared to autoregressive models that finalize only one token at a time, DLMs allow for parallel updates and maintain global consistency, making them a competitive alternative paradigm. The output quality of DLMs depends heavily on the **denoising schedule**—the selection of which tokens to finalize from all masked positions at each step.

**Limitations of Prior Work**: The two mainstream schedules have structural flaws. **Standard Diffusion** performs global denoising without range constraints, selecting only the most confident tokens for finalization based on local confidence at each step. However, due to the lack of global structural constraints, the model often becomes overconfident in EOS (end-of-sequence) tokens and terminates sequences prematurely. Furthermore, early errors, once locked, cannot be corrected and are amplified through subsequent steps. **BlockDiffusion** segments the sequence into fixed-size blocks and updates them in a strict left-to-right order. While it offers better stability and represents the current SOTA among block-based schedules, fixed boundaries **artificially sever semantic units**. Naturally coherent structures like function signatures, formulas, and reasoning steps can be split by block boundaries, forcing the model to finalize tokens with incomplete context. Additionally, the fixed update order cannot adapt flexibly to context or confidence.

**Key Challenge**: There is a misalignment between fixed block boundaries and true semantic boundaries (which are variable-length, cross-block, and content-dependent). Semantic units vary in length, and dependencies often span multiple blocks; any fixed segmentation is suboptimal, serving as the root cause of early errors and cascade failures.

**Goal**: To design a new schedule that simultaneously satisfies three criteria: (1) **Adaptive Scheduling**: dynamically adjusting the denoising order based on the generated context rather than a fixed pattern; (2) **Contextual Integrity**: ensuring each token has a more complete local context when it is finalized; (3) **Constant Compute**: maintaining overhead parity with block-based methods, where quality improvements stem from better scheduling rather than increased computation.

**Key Insight**: Imagine generation as a **wave propagating outward**—maintaining a "wavefront" candidate set that gradually expands from finalized tokens into surrounding masked regions. A token is finalized only when it enters the wavefront and its local context is largely in place.

## Method

### Overall Architecture

WavefrontDiffusion is a **training-free** decoding schedule strategy that can be directly applied to existing DLM backbones (taking a sequence of all `[MASK]` + prompt as input and outputting a clean sequence). It modifies only "which positions to finalize at each step" without altering model weights. Its core abstraction is the **wavefront set** $W_t$: at step $t$, the wavefront contains all masked tokens within a radius $R$ of any finalized position, i.e.,
$$W_t = \{i \mid \mathrm{dist}(i, C_t) \le R\}$$
where $C_t$ is the set of currently finalized positions, $\mathrm{dist}(i,C_t)$ is the minimum distance from position $i$ to any finalized position, and $R$ is the user-defined expansion radius. Intuitively, the wavefront is the "candidate ring around the completed area where context is largely ready." Initially, $C_0$ contains only the prompt, and $W_0$ takes the first $F$ positions following the prompt ($F$ is the maximum wavefront capacity).

Each step follows a four-step cycle of **scoring → selection & finalization → expansion → pruning**, allowing the denoising front to advance from the finalized area like a water wave. This fits the natural extension of semantics while strictly capping the budget for each step, ensuring the total update volume equals that of block-based methods.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Full [MASK] sequence + prompt<br/>Initialize wavefront W₀"] --&gt; B["Scoring: One forward pass<br/>Calculate confidence sⱼ for all masked positions"]
    B --&gt; C["Wavefront expansion mechanism<br/>Select top-kₜ to finalize → Expand to neighbors within radius R"]
    C --&gt;|Wavefront exceeds F| D["Compute-alignment pruning<br/>Keep top-F by confidence"]
    D --&gt;|Steps < T and masks remain| B
    D --&gt;|All finalized or T steps exhausted| E["Clean sequence output"]
```

### Key Designs

**1. Wavefront Candidate Set: Growing denoising along semantic structures instead of fixed blocks**

This design directly addresses the two pain points of BlockDiffusion (severing semantic units) and Standard Diffusion (insufficient context). Instead of updating within pre-cut fixed blocks, WavefrontDiffusion maintains a dynamic front $W_t$ that expands from finalized tokens. Whenever a batch of positions is finalized (added to $C_t$), their neighbors within radius $R$ that remain `[MASK]` are included in the next wavefront:
$$W_t = \bigcup_{i \in C_t} \{\, j \mid \mathrm{dist}(j,i) \le R,\; x_j = [\text{MASK}] \,\}$$
Thus, the wavefront boundary drifts naturally with the generated content, focusing compute on the "periphery of the newly completed area" where the context is most relevant and complete. The authors provide theoretical support via the **Information Gradient Hypothesis**: the conditional entropy of a token increases monotonically with its distance from the finalized context. Therefore, a restricted search within the "distance-defined isentropic surface" contains a higher density of low-entropy (high-certainty) candidates than fixed blocks, minimizing semantic misalignment (proof provided in Appendix D). The key difference from fixed blocks is that the boundary is **content-driven and time-varying**.

**2. Confidence Scoring + Gradual Budget Selection: Finalizing only the tokens with highest current certainty**

Having a candidate region is insufficient; one must also decide which and how many tokens to finalize within that region. Each step begins with **one** forward pass to calculate the confidence for each masked position $j$ as the maximum softmax probability:
$$s_j = \max_{v \in V} p_\theta(x_j = v \mid x_t, c)$$
Then, positions with the top-$k_t$ confidence are selected from the current wavefront $W_{t-1}$, and their masks are replaced with predicted values to complete finalization. The number of finalized tokens per step $k_t$ is distributed across total length $N$ and total steps $T$:
$$k_t = k_{\text{base}} + \mathbb{1}[t \le \text{extra}], \quad k_{\text{base}} = \lfloor N/T \rfloor,\ \text{extra} = N \bmod T$$
This ensures tokens are "finalized only when there is certainty," reducing the risk of premature locking and cascade errors seen in Standard Diffusion, while keeping the per-step workload controllable and reproducible.

**3. Compute-Aligned Pruning: Decoupling quality gains from increased computation**

The wavefront expands as tokens are finalized. Without constraints, overhead would exceed block-based baselines, undermining the conclusion that "improvement comes from better scheduling." To prevent this, **pruning** is performed after expansion each step: if $|W_t| > F$, only the top-$F$ positions are retained according to cached confidence scores. Consequently, the total token update volume is strictly limited to $F \times T$, exactly equal to the budget of BlockDiffusion. The only difference between the two is "where to update" rather than "how much to update." This design is the cornerstone of the experimental validity: any precision gain compared over a fixed 1024-step forward pass can only be attributed to the **superior scheduling method** rather than more compute.

### Loss & Training
Ours is **training-free**, introducing no additional learnable parameters or training objectives. It only replaces the decoding schedule during the inference stage. The DLM backbone itself is trained in the standard manner using cross-entropy under the Variational Lower Bound (VLB) on masked positions. The two core hyperparameters are the maximum wavefront capacity $F$ and expansion radius $R$, with defaults $F=8$ and $R=2$.

## Key Experimental Results

### Main Results
Five benchmarks: GSM8K, MATH, BBH (Reasoning, reporting exact-match accuracy), HumanEval, MBPP (Code, reporting pass@1). Three backbones: LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B. All methods used a fixed 1024 forward steps, temperature 0.0, zero-shot without CoT, ensuring differences arise solely from scheduling.

| Backbone | Strategy | GSM8K | MATH | HumanEval | MBPP | BBH |
|------|------|-------|------|-----------|------|-----|
| LLaDA-8B-Instruct | Standard | 23.15 | 26.60 | 17.68 | 13.50 | 11.30 |
| LLaDA-8B-Instruct | Block (Prev. SOTA) | 80.74 | 40.62 | 45.73 | 41.17 | 43.23 |
| LLaDA-8B-Instruct | **Wavefront** | **82.03** | **41.04** | **47.56** | **42.40** | **44.30** |
| LLaDA-1.5 | Block | 82.33 | 41.64 | 46.34 | 44.04 | 44.56 |
| LLaDA-1.5 | **Wavefront** | **82.94** | **41.96** | **48.17** | **46.20** | **45.26** |
| Dream-7B | Block | 78.92 | 43.60 | 53.05 | 58.52 | 45.13 |
| Dream-7B | **Wavefront** | **80.66** | **44.00** | **54.27** | **59.03** | **46.91** |

WavefrontDiffusion is optimal across **all tasks and all three model families**. Relative to BlockDiffusion, the gains for LLaDA-8B are +1.27 (GSM8K) / +0.42 (MATH) / +1.83 (HumanEval) / +1.23 (MBPP) / +1.07 (BBH); Dream-7B also shows consistent gains such as +1.74 (GSM8K). While the magnitude of improvement is modest, it is stable across mathematical reasoning and code synthesis, and across model scales, all achieved within the same step count and wall-clock budget.

### Semantic Fidelity & Schedule Quality

| Metric (WikiText, BERTScore) | F1 | P | R |
|------|------|------|------|
| Standard | 0.7885 | 0.7664 | 0.7913 |
| Block | 0.7946 | 0.7663 | 0.8142 |
| **Wavefront** | **0.8094** | **0.7749** | **0.8236** |

The increase in Precision indicates fewer irrelevant tokens are inserted, while the increase in Recall suggests more complete sequence completion. Together, the higher F1 supports the idea that "finalizing when context is sufficient" indeed reduces block-based fragmentation. The paper also proposes the **MHCO (Masked Higher-Confidence Outside)** metric to quantify whether the schedule respects the confidence order:

$$\text{MHCO}_t = \frac{1}{|S_t|}\sum_{i \in S_t} \mathbb{1}\big[\exists j \in N_{\text{out}}: c_t(j) > c_t(i)\big]$$

Where $S_t$ is the set selected for finalization in the current step, $N_{\text{out}}$ are masked tokens within radius $R$ outside the front, and $c_t(\cdot)$ is the confidence. It counts the frequency of finalizing a low-confidence token while a higher-confidence token remains nearby—lower is better. Figure 2 shows that Wavefront's MHCO is lower than Block's across all datasets and scales, indicating it more consistently finalizes according to confidence priority, which correlates with the accuracy gains in Table 1.

### Ablation Study (Hyperparameter Sensitivity, LLaDA-8B)

| Configuration | MATH | GSM8K | HumanEval | Description |
|------|------|-------|-----------|------|
| F=4 (R=2) | 41.02 | 82.71 | 45.12 | Fewer candidates |
| **F=8 (R=2)** | 41.04 | 82.03 | **47.56** | Default configuration |
| F=16 (R=2) | 41.22 | 82.03 | 45.12 | Diminishing returns, redundant candidates |
| F=8, R=4 | 40.98 | 82.03 | 46.34 | Slightly larger radius, marginal gain |
| F=8, R=8 | 41.00 | 82.09 | 42.07 | Too large radius, HumanEval performance drops |

### Key Findings
- **Compute alignment is central to the credibility of conclusions**: All comparisons are completed under a fixed budget of 1024 forward steps and identical $F\times T$ update volumes. Accuracy gains can only be attributed to the schedule itself, excluding "scaling compute" as a confounding factor.
- **Gains from F=4→8, diminishing returns from 8→16**: A larger wavefront incorporates more candidates with sufficient context, but exceeding this threshold only adds redundant candidates without increasing information.
- **Radius R should not be too large**: Increasing $R$ from 2 to 4 yields marginal gains, and at $R=8$, HumanEval accuracy drops from 47.56 to 42.07. An overly broad front weakens local focus and introduces noise. Overall, the method is insensitive to hyperparameters, with $F=8,R=2$ being default and robust.

## Highlights & Insights
- **"Wavefront" is an apt physical metaphor**: Formalizing "context diffusion from known to unknown" as a distance-based growing candidate set provides more structure than Standard Diffusion's global updates and more flexibility than BlockDiffusion's fixed blocks—an elegant compromise.
- **Information Gradient Hypothesis provides a theoretical skeleton**: The assumption that "conditional entropy increases monotonically with distance from finalized context" explains why proximal tokens should be finalized first, arguing that dynamic boundaries contain a higher density of low-entropy candidates than fixed blocks.
- **Training-free, Plug-and-Play**: By only changing the decoding schedule and not the weights, it can be directly applied to any DLM backbone (LLaDA, Dream) with extremely low migration costs. This "inference-only scheduling optimization" could inspire other parallel decoding scenarios.
- **MHCO is a reusable diagnostic metric**: It quantifies whether a schedule violates confidence priorities and correlates with final accuracy, serving as a general probe for evaluating the rationality of parallel decoding strategies.

## Limitations & Future Work
- **Reliance on internal confidence, which may be inaccurate**: The method uses max softmax probability as a proxy for confidence to select tokens. However, confidence itself can be miscalibrated, especially in out-of-distribution scenarios, potentially misleading the finalization order.
- **Cannot fully avoid cascade errors**: If an error occurs early in a long reasoning chain, it can still bias subsequent steps; finalization is irreversible. The authors suggest exploring **deferred finalization / reversible decoding** in the future.
- **Modest improvement magnitude**: Gains relative to BlockDiffusion are mostly between +0.3 and +2 points, representing a robust incremental gain rather than a breakthrough. The advantage lies in getting these gains "for free" with zero additional compute.
- **Expansion directions**: Improving confidence calibration for enhanced robustness; extension to multimodal or structured domains (code, graphs).

## Related Work & Insights
- **vs Standard Diffusion**: It updates globally and in parallel without range limits, finalizing tokens based solely on local confidence. This easily leads to premature EOS locking and cascade errors. Ours limits the candidate range to ensure contextual integrity before finalization, correcting the issue of "finalizing with insufficient context."
- **vs BlockDiffusion**: It uses fixed-size blocks and a fixed order to control error propagation, representing the block-wise SOTA. However, fixed boundaries sever semantic units and the order is immutable. Ours replaces "fixed blocks" with "content-driven, dynamically expanding wavefronts," respecting semantic boundaries within the **exact same $F\times T$ compute budget**. The only difference lies in update locations rather than update volume.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling decoding schedule as "dynamic wavefront expansion" with the Information Gradient Hypothesis is novel and training-free.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 benchmarks × 3 backbones + strict compute alignment + custom MHCO metric + hyperparameter analysis; quite complete, though lacking horizontal comparisons with more parallel decoding variants.
- Writing Quality: ⭐⭐⭐⭐ Motivation-Theory-Algorithm-Experiment logic is clear; the physical metaphor is intuitive.
- Value: ⭐⭐⭐⭐ Provides stable improvements for DLM reasoning/code generation with zero extra compute; plug-and-play with low migration cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Early Exit in Reasoning Models](dynamic_early_exit_in_reasoning_models.md)
- [\[ICLR 2026\] Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs](sample_smart_not_hard_correctness-first_decoding_for_better_reasoning_in_llms.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[ACL 2026\] Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning](../../ACL2026/llm_reasoning/adapt_to_thrive_adaptive_power-mean_policy_optimization_for_improved_llm_reasoni.md)
- [\[ICML 2026\] DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling](../../ICML2026/llm_reasoning/dycon_dynamic_reasoning_control_via_evolving_difficulty_modeling.md)

</div>

<!-- RELATED:END -->
