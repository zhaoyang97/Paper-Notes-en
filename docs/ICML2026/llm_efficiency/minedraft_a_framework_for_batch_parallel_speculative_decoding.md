---
title: >-
  [Paper Note] MineDraft: A Framework for Batch Parallel Speculative Decoding
description: >-
  [ICML2026][LLM Efficiency][Speculative Decoding] MineDraft transforms the originally serial "draft-verify" pipeline of speculative decoding into batch parallel PSD by maintaining two batches of requests and allowing the…
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "Speculative Decoding"
  - "Parallel Speculative Decoding"
  - "Batch Parallel"
  - "vLLM"
  - "GPU Overlapping"
date: 2026-05-08
content_hash: 1ab9435d1d919bb5
---

# MineDraft: A Framework for Batch Parallel Speculative Decoding

**Conference**: ICML2026  
**arXiv**: [2603.18016](https://arxiv.org/abs/2603.18016)  
**Code**: Available (MineDraft GitHub repository, released as a vLLM plugin)  
**Area**: LLM Efficiency / Inference Acceleration  
**Keywords**: Speculative Decoding, Parallel Speculative Decoding, Batch Parallel, vLLM, GPU Overlapping

## TL;DR
MineDraft transforms the originally serial "draft-verify" pipeline of speculative decoding into batch parallel PSD by maintaining two batches of requests and allowing the drafting of one batch to **overlap execution** with the verification of the other on independent GPUs. At the cost of only one additional GPU, it increases throughput by up to 75% and reduces end-to-end latency by up to 39% compared to standard SD, and is implemented as a plug-and-play vLLM plugin.

## Background & Motivation

**Background**: Speculative Decoding (SD) is the mainstream solution for accelerating LLM inference—using a small draft model to autoregressively generate $k$ draft tokens, followed by a large target model for parallel verification. When most drafts are accepted, SD is significantly faster than vanilla autoregressive decoding.

**Limitations of Prior Work**: The effectiveness of SD highly depends on the verification success rate (VSR) of the drafter, and drafting and verification are **strictly serial**—verification begins only after drafting is complete, and the next draft step starts only after verification. Existing works (Medusa, EAGLE, TETRIS, etc.) focus on "improving VSR" or "tree-based/multi-branch drafting," but these methods often slow down the drafting phase (due to more complex drafters or higher sampling overhead), further pinning drafting to the critical path and hitting an acceleration ceiling.

**Key Challenge**: Since verification data-dependently relies on drafting output, direct parallelization is non-trivial. Existing parallelization attempts either require multiple GPUs/VRAM (Wang 2024, Timor 2025), require retraining draft models (Xiao 2024), or handle only single requests (PEARL/Liu 2025a). In batched multi-request scenarios, effectively hiding drafting behind verification using **limited additional resources** remains an open problem.

**Goal**: (i) Theoretically quantify "how much time PSD can save compared to SD"; (ii) Provide a batch parallel PSD framework that can be directly integrated into production inference stacks (vLLM + PagedAttention + continuous batching); (iii) Ensure orthogonality with existing drafting strategies (EAGLE, TETRIS, PEARL) for additive benefits.

**Key Insight**: The authors observe that since verification waits for the current batch's draft, **letting the drafter generate drafts for the "next batch" simultaneously** allows the drafter's work to be entirely hidden within the verifier's execution time. As long as the request pool is split into two and batches enter the verifier alternately, the drafter will never be idle.

**Core Idea**: Use "dual-batch rotation + independent GPUs on both sides" to hide drafting in the shadow of verification. While one batch is being verified, the other is being drafted. The two exchange tokens via direct GPU-to-GPU communication, keeping the verifier fully utilized.

## Method

### Overall Architecture

MineDraft is deployed such that the **target model runs on $N$ GPUs using tensor parallelism (where $N=4$ in the paper), while the drafter occupies 1 separate GPU**. The total cost is only one card more than standard SD. The framework consists of four components:

- **Batch Manager**: Splitting up to $2m$ concurrent requests into Batch 0 and Batch 1, maintaining `balance` and `skip_batch` states, and handling batch ID allocation for new requests and recycling for terminated ones.
- **Scheduler**: Managing request lifecycles and KV block allocation, patched in vLLM to solve the over-allocation problem of "drafted but not yet verified" requests.
- **Drafter**: Generating draft tokens for the *draft batch* on the draft GPU and broadcasting them to the verifier.
- **Verifier**: Performing parallel verification for the *target batch* (the previous step's draft batch) on the target GPUs and sending sampling results back to the drafter point-to-point.

The execution timing is shown in Fig. 2 (right): Before the first SD step, the drafter serially drafts for Batch 0 and broadcasts to the verifier, then drafts for Batch 1. In subsequent SD steps, while the verifier verifies the previous draft batch, the drafter is already drafting the next round for the other batch. **The execution times on both sides overlap almost completely on the timeline.** The moment the drafter sends output back to the Scheduler at the end of each SD step is the sync point, where `skip_batch` flips and the batch roles swap.

### Key Designs

1. **Dual-Batch Rotating Batch Manager (balance + skip_batch State Machine)**:
    - **Function**: Maintaining two approximately equal-sized request pools across $2m$ concurrent requests, ensuring the verifier receives a pre-drafted target batch at each step while the drafter immediately works on the other batch to enable "hidden drafting."
    - **Mechanism**: Tracking the size difference between the two batches using `balance = |Batch 1| - |Batch 0|`. During the first SD step, new requests are assigned to the smaller batch for **load balancing** based on the sign of `balance` (if `balance >= 0`, assign to Batch 0 and `balance--`; otherwise, assign to Batch 1 and `balance++`). After the first step, new requests are assigned to the current `skip_batch` (the batch currently being drafted but not yet verified), which maintains balance naturally without disrupting the verifier. Terminated requests trigger a reverse `balance` update via `recycle`.
    - **Design Motivation**: The core benefit of PSD comes from continuous overlapping. **An empty batch on either side causes the pipeline to fallback to standard SD.** The `balance/skip_batch` mechanism is a necessary state machine to avoid irreversible imbalance in real-world scenarios like preemption, abortion, or chunked prefill, turning a 37% theoretical gain into engineering reality.

2. **Cross-GPU Drafter–Verifier Parallel Pipeline (Independent GPUs + Direct Communication)**:
    - **Function**: Decoupling the compute, VRAM, and KV cache of the drafter and verifier, allowing execution times to be truly parallel rather than competing for resources on the same card.
    - **Mechanism**: The drafter occupies 1 GPU, while the target uses tensor parallel across the rest. Drafter $\rightarrow$ Verifier uses **broadcast** for draft tokens (magenta arrows), and Verifier $\rightarrow$ Drafter uses **point-to-point dispatch** for target sampler outputs (dark green arrows). At the sync point of each step, `skip_batch` flips, automatically swapping the draft and target batches. Theoretical analysis (Theorem 1) suggests that if $f(t) = 1 - e^{-\alpha t}$ describes the drafter's Pareto front (drafting time vs. VSR), then when $\alpha V \approx 1.68$, $T_{\text{SD}} \gtrsim 1.59 \, T_{\text{PSD}}$, meaning PSD saves at least 37% time; the ideal limit is 50% (drafting completely hidden).
    - **Design Motivation**: Existing parallel schemes either pack both models on one card (causing VRAM contention; Fig. 5 shows standard SD with Qwen3-8B as a drafter leads to OOM) or require doubled GPUs. MineDraft treats "1 dedicated card for draft" as the minimum hardware investment to achieve the engineering effect of drafting disappearing behind the verifier's timeline, and it is orthogonal to strategies like EAGLE/TETRIS.

3. **vLLM Scheduler Patch: Deferred KV Block Allocation (`has_deferred`)**:
    - **Function**: Avoiding over-allocation of KV blocks to requests that are "only being drafted and not yet verified" within the PagedAttention framework, maintaining compatibility while preventing memory waste.
    - **Mechanism**: Observing that the drafter only reads but does not write newly allocated KV blocks, whereas the verifier performs actual writes. The default vLLM scheduler assumes all running requests generate tokens at each step, allocating KV blocks for both batches simultaneously, which causes "touched but unwritten" blocks for the draft batch. The patch introduces a `has_deferred` set to track request IDs that have had allocation postponed. When both batches are non-empty, prefill requests allocate as usual, while decoding requests only allocate if their ID is not in `has_deferred` or belongs to the current draft batch, after which the ID is added to `has_deferred`. If either batch is empty (except the first step), it falls back to allocating for all requests. This ensures the first SD step target batch is not incorrectly skipped.
    - **Design Motivation**: Ensures MineDraft is not just theoretically parallel but acts as a **plug-and-play vLLM plugin** compatible with continuous batching and PagedAttention. This is the "last mile" for moving PSD from paper to production stack.

### Loss & Training
MineDraft is a **training-free** inference acceleration framework. it modifies neither the draft model nor the target model, only scheduling the execution pipeline and KV cache allocation. Thus, it can be applied to any existing SD/EAGLE/TETRIS setup.

## Key Experimental Results

### Main Results

Evaluated on **seven target–draft configurations**, target using tensor parallel = 4, drafter on 1 card; datasets: Arena, ShareGPT, Spec-Bench, Tough.

| Config (Target–Draft) | Dataset Example | MineDraft vs Best Baseline | MineDraft vs Standard SD (Δ) |
|---|---|---|---|
| Qwen3 32B–0.6B | Arena | +42.36% Throughput | +70.32% |
| Qwen3 32B–1.7B | Tough | +48.47% Throughput | +75.68% (Highest) |
| Qwen3 32B–4B | ShareGPT | +65.02% Throughput | +65.64% |
| Llama-3 70B–8B | ShareGPT | +30.81% Throughput | +37.06% |
| Vicuna 33B–EAGLE | ShareGPT | +3.95% Throughput | +22.09% |
| Qwen3 32B–1.7B (E2EL) | Tough | -28.97% Latency | -39.51% Latency |
| Qwen3 32B–8B | Arena | Standard SD OOM | MineDraft Operable |

**Normalization (Adjustment for 5 vs 4 GPUs)**: In Setting 2, MineDraft still improves per-GPU normalized throughput by up to 40.55% and reduces normalized latency by up to 24.38% compared to standard SD, only slightly trailing by 2.08% in normalized latency on Spec-Bench with $k=2$.

### Ablation Study

Four groups of ablations on the Arena dataset (corresponding to Fig. 8 in the paper):

| Configuration | Key Finding | Description |
|---|---|---|
| Different Draft Models | Drafter choice significantly impacts parallel gains | When drafter compute approaches verifier compute, the $t$ term in $\max(V, t)$ dominates, reducing gains. |
| Different Extra Tokens (with TETRIS) | MineDraft consistently outperforms standard SD across all $k$ | Orthogonal benefits when combined with "multi-sampling + selection" strategies. |
| Different #sequences per request $n$ | Benefits persist as $n$ increases | PSD remains robust under intra-batch multi-sampling. |
| Different Batch Size $m$ | Gains are stable as $m$ increases | The dual-batch rotating design scales well. |

### Key Findings
- **Drafter size is a double-edged sword**: Larger drafters improve VSR but lengthen $t$. When $t > V$, the $\max(V, t)$ term of PSD is dominated by $t$, causing acceleration ratios to drop sharply (Section C.6). The "sweet spot" is 1.7B/4B drafters for Qwen3-32B.
- **Overly long $k$ (draft steps) can backfire**: Too many drafts $\rightarrow$ massive verification pressure $\rightarrow$ drafting becomes the critical path. This aligns with observations in adaptive drafting and explains why pairing with methods like TETRIS for adaptive $k$ selection is necessary.
- **VRAM decoupling is a hidden benefit**: In Fig. 5, standard SD OOMs when using Qwen3-8B as a drafter, while MineDraft stays operational because the drafter is on an independent GPU and does not compete for KV cache; this also allows MineDraft to serve massive targets like Qwen3-235B.
- **EAGLE Degradation**: EAGLE's performance drops as $k$ increases in current vLLM implementations (under investigation by the vLLM team). Consequently, the benefits of combining MineDraft with EAGLE are partially offset, showing only +3.95%~7.51% throughput gain on Vicuna+EAGLE, far lower than the 30%+ gain with standard SD drafters.

## Highlights & Insights
- **Engineering Analogy of "Minecraft Chunk Loading"**: Mapping "pre-loading the next chunk in the background while the player interacts with the current chunk" to "drafter pre-generating the next batch while the verifier verifies the previous batch." This double-buffering idea, recurring in OS/game engines, is cleanly transposed to the SD pipeline.
- **Clear Theoretical Bounds**: The $\max(V, t)$ term succinctly explains all PSD behaviors—ideally, when $t \le V$, drafting is effectively "free," and the acceleration ratio approaches the 50% upper bound; as soon as $t > V$, gains diminish. All experimental phenomena can be predicted by this one-line formula.
- **Convincing ROI**: "Adding 1 card for +75% throughput" is highly persuasive for engineering teams. Compared to schemes requiring double the GPUs for parallelization, MineDraft minimizes the marginal hardware cost.
- **Portable Design**: The `balance + skip_batch + dual-batch rotation` state machine can be applied almost as-is to any two-stage pipeline with multiple requests (e.g., retrieve-then-rerank, prefill-vs-decode separation).

## Limitations & Future Work
- **Imbalance Degradation**: If chunked prefill results in fewer than $2m$ requests for the first step, or if preemption/abortion clears a batch, subsequent new requests might be assigned to the same batch, potentially leading to a "one batch empty, fallback to standard SD" state.
- **Lack of Adaptive Drafter Selection**: Experiments show drafter size hugely impacts gains, but MineDraft does not provide an online mechanism to select the optimal drafter.
- **Modest Gains on EAGLE**: Combining with EAGLE yielded only single-digit throughput gains, partly due to vLLM's implementation, but this also reveals that PSD gains depend on the drafting phase being relatively lightweight.
- **Improvement Ideas**: (i) Enabling the Batch Manager to actively schedule from the waiting queue during imbalance; (ii) Integrating adaptive $k_i$ selection (like TETRIS/AdaSpec) based on real-time $V$ vs. $t$ ratios; (iii) Exploring $\ge 3$ batch rotations to saturate the verifier in extreme cases where drafters are significantly smaller.

## Related Work & Insights
- **vs PEARL (Liu 2025a)**: PEARL uses pre-verify/post-verify to parallelize drafting and verification for a **single request**; MineDraft uses "dual-batch rotation" for **batched multi-request** scenarios.
- **vs Wang 2024 / Timor 2025 Parallel SD**: These rely on doubling GPU/VRAM to break the draft-verify dependency; MineDraft uses only one extra card (5 vs 4) and avoids VRAM contention via KV cache decoupling.
- **vs Xiao 2024**: Xiao 2024 requires specialized drafter training; MineDraft is training-free and works with off-the-shelf drafters.
- **vs EAGLE/TETRIS/Medusa**: These improve grafting quality/sampling, while MineDraft optimizes the **temporal relationship** between drafting and verification. They are orthogonal.
- **Insight**: Inference acceleration research can shift from "optimizing a single stage" to "reordering the timing of two stages." In systems with stage dependencies like RAG or prefill/decode separation, this double-buffering spirit is worth reusing.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Implementing double-buffering for batched SD with theoretical grounding; concepts are established, but the application and engineering are spot-on.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 7 configurations × 4 datasets, including comparisons with SD/PEARL/EAGLE/TETRIS, normalized fair comparisons, and end-to-end vLLM plugin validation.
- **Writing Quality**: ⭐⭐⭐⭐ Excellent Minecraft analogy; theoretical and systemic threads are clear.
- **Value**: ⭐⭐⭐⭐⭐ High ROI (1 GPU for 75% gain), implemented as a production-grade plugin, directly beneficial to LLM serving teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [\[ACL 2026\] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](../../ACL2026/llm_efficiency/racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)

</div>

<!-- RELATED:END -->
