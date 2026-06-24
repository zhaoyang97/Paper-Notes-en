---
title: >-
  [Paper Note] MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning
description: >-
  [CVPR2025][VLM Reasoning][MLLM Benchmark] MM-CondChain is the first MLLM benchmark for visually grounded deep compositional reasoning. By using a Verifiable Programmatic Intermediate Representation (VPIR), it automatically constructs multi-layer conditional chains and chain-style hard negatives. The strongest model achieves only a 53.33 Path F1, revealing that deep compositional reasoning remains a fundamental challenge.
tags:
  - "CVPR2025"
  - "VLM Reasoning"
  - "MLLM Benchmark"
  - "Compositional Reasoning"
  - "conditional chain"
  - "hard negative samples"
  - "programmatic verification"
date: 2026-05-08
content_hash: be6e7aafa6c846c2
---

# MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning

**Conference**: CVPR2025  
**arXiv**: [2603.12266](https://arxiv.org/abs/2603.12266)  
**Code**: [GitHub](https://github.com/Accio-Lab/MM-CondChain)  
**Area**: Multimodal VLM  
**Keywords**: MLLM Benchmark, Compositional Reasoning, conditional chain, hard negative samples, programmatic verification

## TL;DR
MM-CondChain is the first MLLM benchmark for visually grounded deep compositional reasoning. By using a Verifiable Programmatic Intermediate Representation (VPIR), it automatically constructs multi-layer conditional chains and chain-style hard negatives. The strongest model achieves only a 53.33 Path F1, revealing that deep compositional reasoning remains a fundamental challenge.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: 1. MLLMs are increasingly used in workflows requiring chain-style visual verification (e.g., GUI navigation), yet this capability lacks systematic evaluation.
2. Existing visual reasoning benchmarks only evaluate shallow, single-layer composition (e.g., "is the object red and large?") or independent constraints.
3. Instruction-following benchmarks focus on independent constraints rather than nested, layer-by-layer conditional reasoning.
4. Existing hard negative samples are typically limited to single-layer changes (e.g., replacing a single attribute) and lack chain-style hard negatives.
5. Most benchmarks rely on LLM-as-judge evaluation, which lacks certainty and reproducibility.
6. Directly prompting MLLMs to generate multi-layer reasoning chains often results in logical conflicts and unverifiable assertions.

## Method

### Overall Architecture
VPIR-based Agentic Benchmark Construction Pipeline: (1) Planner orchestrates reasoning chain construction layer by layer; (2) Each layer ensures mechanical verifiability of conditions through VPIR (Verifiable Programmatic Intermediate Representation); (3) Verifier performs two-stage quality control; (4) Composer compiles reasoning chains into True-path/False-path paired evaluation instances.

### Key Designs

**1. Layer-by-Layer VPIR Synthesis (4 steps)**
- **Step 1**: Select relation strategy $r_t$ (Deepening or Transition) to constrain subject selection.
- **Step 2**: Extract structured facts $F_t$ (JSON key-value mapping) from the visual input to ensure the subject can be uniquely localized.
- **Step 3**: Generate executable predicate pairs $(p_t, \tilde{p}_t)$ and verify them in a sandbox environment to ensure $\llbracket p_t \rrbracket(F_t) = 1$ and $\llbracket \tilde{p}_t \rrbracket(F_t) = 0$.
- **Step 4**: Render the verified logic into natural language conditions $(c_t, \tilde{c}_t)$.

**2. Two-Stage Verifier**
- Stage I: Fact and Subject Verification (visual localizability, non-repetition, relation compliance, pattern consistency).
- Stage II: Linguistic Realization Verification (semantic fidelity, unambiguous referencing, counterfactual quality).
- Stage-Aware Feedback: Stage I failure triggers fact regeneration; Stage II failure only triggers language re-rendering.

**3. Planner: Verification-Aware Chain Control**
- Hybrid depth control: Hard rules + MLLM strategy.
- Action space: EXTEND / FINISH / ROLLBACK.
- Verification-aware backtracking: ROLLBACK is triggered when repeated verification failures occur.

**4. Composer: Paired Path Instance Compilation**
- True-path: All conditions are satisfied, reaching the terminal layer to answer $q^{\text{fin}}$.
- False-path: Randomly selects a divergence layer $j$, substitutes $c_j \leftarrow \tilde{c}_j$, and prematurely terminates to answer $q_j^{\text{aux}}$.
- Subject de-leakage: Rewrite subject descriptions to prevent conditional answer leakage.
- Deterministic multiple-choice evaluation, eliminating the need for an LLM-as-judge.

### Three Visual Domains
- **Natural Images**: SAM + GQA, 398 images.
- **Data Charts**: ChartQA, 200 charts (bar/line/pie + structured annotations).
- **GUI Trajectories**: AITZ, 377 trajectories (3,421 screenshots).

## Key Experimental Results

### Overall Performance (Path F1, %)


### Main Results

| Model | Natural F1 | Chart F1 | GUI F1 | Avg F1 |
|------|-----------|---------|--------|--------|
| Gemini-3-Pro | 55.91 | 66.04 | 38.05 | **53.33** |
| GPT-5-0807 | 47.51 | 65.44 | 38.06 | 50.34 |
| Gemini-3-Flash | 47.19 | 61.96 | 35.78 | 48.31 |
| Qwen3-VL-235B-Thinking | 49.31 | 59.96 | 31.23 | 46.83 |
| Qwen3.5-397B-A17B | 38.97 | 58.55 | 40.19 | 45.90 |
| GPT-4o-1120 | 22.23 | 17.49 | 20.46 | 20.06 |

### True vs. False Path Analysis
- GPT-4o on Natural domain: True-path 83.92% vs. False-path 12.81%, showing a severe imbalance.
- Qwen3.5-4B on Natural domain: True 88.92% vs. False 15.37%.
- Gemini-2.5-Pro performs better on False-path (Natural 55.28%), but its True-path is only 38.94%.
- Small models tend to adopt an "all-pass" strategy, leading to extremely high True scores and extremely low False scores.

### Key Findings
- The strongest model, Gemini-3-Pro, achieves an Avg F1 of only 53.33, indicating that deep compositional reasoning is highly challenging.
- Severe imbalance between True and False paths: most models perform much worse on hard negative samples than on positive ones.
- The Chart domain has the highest overall F1, while the GUI trajectory domain is the hardest (requiring sequential reasoning across multiple frames).
- Performance decreases further as the reasoning depth and predicate complexity increase.
- High structural diversity of VPIR expressions: 128 templates cover 80%, while the top 20 templates cover only 50.07%.
- Deterministic evaluation (multiple-choice + programmatic verification) eliminates LLM-as-judge bias.

## Highlights & Insights
1. **VPIR Innovation**: Decouples logical construction from language rendering, using executable code to guarantee data quality instead of relying on LLM generation.
2. **Chain-Style Hard Negatives**: Flipping a single predicate changes the entire execution path, forcing the model to precisely verify each condition.
3. **Generality Across Three Domains**: A unified framework adaptable to natural images, charts, and GUIs, with domain-specific adaptation limited to the input preprocessing layer.
4. **Fully Deterministic Evaluation**: Multiple-choice questions combined with programmatically verified ground truth, eliminating LLM-as-judge bias.
5. **Revealing Fundamental Capability Gaps**: Demonstrates systemic weaknesses of MLLMs in deep conditional reasoning.

## Limitations & Future Work
1. Limited data scale (975 samples) may not fully reflect model performance on a wider distribution.
2. Subject de-leakage relies on MLLM rewriting, which might introduce imperfections.
3. Fact extraction relies on MLLM accuracy, so benchmark quality is constrained by the performance of the extraction model.
4. Only text outputs are evaluated, without considering performance in interactive execution.
5. Hard-coded rules for depth control might limit the naturalness of the chains.

## Related Work & Insights
- Difference from IFEval: IFEval uses code to check output format compliance, whereas MM-CondChain uses code to ensure the quality of constructed data.
- Difference from SugarCrepe/Winoground: The latter test single-layer composition, whereas MM-CondChain tests multi-layer nested reasoning.
- The decoupling concept of VPIR can be extended to other domains requiring reliable benchmark construction.
- Chain-style conditional reasoning capability is a core prerequisite for agentic AI, making this benchmark highly valuable as a reference.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (VPIR + chain-style hard negative benchmark construction paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Ten models, three domains, multi-dimensional analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Extremely clear system descriptions)
- Value: ⭐⭐⭐⭐⭐ (Reveals critical MLLM capability gaps, with broad impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](../../ICML2026/vlm_reasoning/ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ICLR 2026\] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](../../ICLR2026/vlm_reasoning/mmr-v_whats_left_unsaid_a_benchmark_for_multimodal_deep_reasoning_in_videos.md)
- [\[CVPR 2026\] DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models](../../CVPR2026/vlm_reasoning/deepscan_a_training-free_framework_for_visually_grounded_reasoning_in_large_visi.md)
- [\[NeurIPS 2025\] READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](../../NeurIPS2025/vlm_reasoning/enhancing_compositional_reasoning_in_clip_via_reconstruction.md)
- [\[ICLR 2026\] CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs](../../ICLR2026/vlm_reasoning/compodistill_attention_distillation_for_compositional_reasoning_in_multimodal_ll.md)

</div>

<!-- RELATED:END -->
