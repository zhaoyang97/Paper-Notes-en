---
title: >-
  [Paper Note] Ekka: Automated Diagnosis of Silent Errors in LLM Inference
description: >-
  [ICML 2026][LLM Efficiency][Silent Errors] Ekka reformulates the diagnosis of "silent errors"—where LLM serving frameworks produce degraded outputs without throwing errors—into a differential debugging task using referen…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Silent Errors"
  - "Differential Debugging"
  - "LLM Inference Frameworks"
  - "Agent Debugging"
  - "Activation Alignment"
date: 2026-05-08
content_hash: 3188deb9a0a0ddfa
---

# Ekka: Automated Diagnosis of Silent Errors in LLM Inference

**Conference**: ICML 2026  
**arXiv**: [2606.04594](https://arxiv.org/abs/2606.04594)  
**Code**: TBC  
**Area**: LLM Efficiency / LLM Serving Systems / Automated Debugging  
**Keywords**: Silent Errors, Differential Debugging, LLM Inference Frameworks, Agent Debugging, Activation Alignment

## TL;DR
Ekka reformulates the diagnosis of "silent errors"—where LLM serving frameworks produce degraded outputs without throwing errors—into a differential debugging task using reference implementations like HuggingFace as an oracle. By utilizing an agentic pipeline of "component mapping $\rightarrow$ activation alignment $\rightarrow$ change-point analysis," Ekka automatically locates specific faulty modules. It achieves 80% pass@1 and 88% pass@5 accuracy across 17 real-world vLLM/SGLang issues and discovered 4 new hidden bugs confirmed by developers.

## Background & Motivation

**Background**: Specialized LLM serving frameworks such as vLLM, SGLang, NanoFlow, and KTransformers have become industry standards. These systems are packed with deep optimizations like paged attention, radix attention, custom CUDA kernels, and CUDA graph compilation. The codebases are massive and iterate rapidly, with releases occurring every few days.

**Limitations of Prior Work**: This combination of high optimization and rapid iteration breeds a particularly troublesome type of bug: **silent errors**. In these cases, the framework neither crashes nor logs warnings, and it does not drop requests, yet the output quality silently degrades. A representative case cited is when vLLM Gemma 3 suddenly dropped nearly 30 points on HellaSwag; it took developers months to identify that sliding window attention was being used incorrectly. Among 90 real-world issues collected by the authors, 43.8% manifested as "accuracy regression," where outputs remain readable but areFactually incorrect.

**Key Challenge**: There is a massive semantic gap between the symptoms (end-to-end benchmark regression) and the root cause (implementation details of a specific kernel or module). Existing methods are inadequate:
- Traditional fault localization relies on pass/fail signals, which are absent in silent errors.
- Deep learning testing tools either treat models as black boxes or only compare API layers, failing to penetrate optimized serving engines.
- General agentic debuggers lack the specialized scaffolding for the LLM inference domain, leading to inefficient trial-and-error.

The empirical practice for developers is to use HuggingFace Transformers as a reference for **differential debugging** (used in approximately 50% of issues), but manually aligning intermediate tensors across frameworks is extremely laborious. For example, vLLM merges Q/K/V projections into a single `QKVProjection` class while HuggingFace uses three independent modules; writing the glue code just to align them is non-trivial.

**Goal**: To automatically provide a ranked report of "most likely buggy components" without requiring the oracle to provide pass/fail labels, allowing humans to review only the top-K candidates instead of dumping tensors layer-by-layer from scratch.

**Key Insight**: The authors observe that almost all mainstream models have a "slow but correct" reference implementation on HuggingFace. Thus, differential debugging is naturally viable for LLM serving; the missing piece is an LLM agent that can replace humans in "identifying corresponding modules + writing alignment code + determining the point of divergence."

**Core Idea**: Reframe silent error diagnosis as differential debugging between two implementations. Use an agent to automate component mapping and activation alignment, followed by a noise-robust error ratio and change-point detection to pinpoint the point of divergence.

## Method

### Overall Architecture
Ekka takes as input: a suspected target framework (vLLM or SGLang), a reference framework (HuggingFace Transformers), the model, and the prompt/configuration that triggers the bug. The output is a root-cause report ranked by suspicion.

The system is divided into two main stages:
1. **Diagnosis Information Collection**: Parses the code and model architectures of both frameworks, reproduces the bug, and records the execution trace (activations and calling sequences for each layer) to disk. This provides the agent with "comparable facts."
2. **Agent-based Bug Diagnosis**: A three-step pipeline—Component Mapping $\rightarrow$ Activation Alignment $\rightarrow$ Error Analysis. Ekka explicitly restricts the diagnosis scope to the **model stack layers (model implementation + kernel backend)** and avoids high-level orchestration (e.g., schedulers, async engines), as the latter is better suited for traditional logging/trace tools.

### Key Designs

1. **Component Mapping via Model Tree**:
    - **Function**: Identifies "semantically equivalent" sub-module pairs between frameworks with vastly different implementations.
    - **Mechanism**: Static analysis compresses each framework's `nn.Module` structure into a concise **Model Tree** (retaining hierarchy and naming while removing redundant wrappers). The agent then performs node matching across the two trees, allowing for one-to-many or many-to-one outputs (e.g., vLLM's fused QKV vs. HF's three independent Linears). For ambiguous names, the agent uses a `get_class_definition` tool to inspect source code. Unmapped modules (e.g., SGLang’s logit processor) are explicitly labeled with rationale.
    - **Design Motivation**: Class name matching almost always fails in LLM serving because modules are fused or split for performance. Abstracting structure into a tree leverages the LLM's strength in identifying variants and compositional logic while avoiding confusion from full codebase complexity.

2. **Activation Alignment via Agent-Generated Glue Code**:
    - **Function**: Unifies dumped tensors from mapped module pairs into the same shape, dtype, and memory layout for element-wise comparison.
    - **Mechanism**: Instead of a fixed rule set, the agent generates ad-hoc Python code for each mapping pair to handle differences like reordering paged KV-caches into dense tensors, casting BF16 back to FP32, slicing batch dimensions, or reordering tokens. The agent also generates self-checking code to verify shape consistency before comparison.
    - **Design Motivation**: Layout differences across frameworks are subject to combinatorial explosion (attention backend $\times$ dtype $\times$ KV layout $\times$ TP, etc.). Using an agent to generate "one-time translators" based on code definitions is extensible and turns alignment failures into explicit exceptions rather than silent errors in comparison results.

3. **Error Analysis: Robust Error Ratio and Change-Point Detection**:
    - **Function**: Determines whether a mapping pair contains a bug and locates where the bug first appeared in the sequence or layers.
    - **Mechanism**: Rather than using absolute difference or cosine similarity, it defines a **robust error ratio** indicator—a relative measure of abnormality that tolerates small drifts from BF16 accumulation but spikes significantly during real logical deviations. **Change-point analysis** is then applied along the layers or token sequences to locate the first significant jump, which is identified as the root cause. All mapped pairs are finally ranked by divergence intensity.
    - **Design Motivation**: Empirical study shows ~19.4% of symptom-level bugs are just floating-point noise. Common thresholds produce high false positives. However, true bugs always exhibit a "breakout" point during cross-layer propagation, which is the exact strength of change-point detection.

### Loss & Training
Ekka is an LLM agent-based system and does not train new models. It uses general closed-source LLMs as agents; the average cost per case is approximately \$30 (primarily token fees and reproduction execution).

## Key Experimental Results

### Main Results

Dataset: A self-constructed silent-error benchmark containing 90 real-world silent errors from vLLM and SGLang. 70 are fixed (for empirical study), and 20 are open (for evaluation). 4 undisclosed new bugs were used for discovery testing. HuggingFace Transformers served as the reference.

| Dataset | Metric | Ekka | Strongest Baseline | Gain |
|--------|------|------|---------------|------|
| 17 real vLLM/SGLang silent errors | pass@1 Diagnosis Accuracy | 80% | ~46-56% (SOTA Agentic Debugging) | +24%~+34% |
| 17 real vLLM/SGLang silent errors | pass@5 Diagnosis Accuracy | 88% | Below Ekka | Significant |
| In-the-wild discovery | New silent errors confirmed by devs | 4 | — | All New |
| Diagnosis cost | Average USD per case | ~\$30 | — | Economical for agents |

Main Conclusion: Given an oracle reference, differential debugging combined with agent automation improves diagnosis accuracy from ~50% to 80% (pass@1) and successfully identifies new bugs previously unnoticed by developers.

### Ablation Study

The qualitative trends of removing key Ekka capabilities:

| Configuration | Metric Trend | Description |
|------|---------------|------|
| Full Ekka | 80% pass@1 | Full three-step pipeline |
| w/o Model Tree mapping | Sharp Decrease | Failed to find corresponding modules for fused QKV, etc. |
| w/o Agent-generated alignment | Sharp Decrease | Layout/dtype mismatches caused crashes or false comparisons |
| w/o Robust error ratio (fixed threshold) | Significant Decrease | BF16 noise led to massive false positives |
| w/o Change-point analysis (max error point) | Decrease | Error accumulates in later layers; max point $\neq$ origin point |

### Key Findings
- **Deep root-cause distribution**: Empirical research shows only 30.6% of bugs stem from framework orchestration, while ~50% originate from model implementation/backend and 19.4% from pure numerical instability. This justifies the need to "open the model stack" and inspect activations.
- **Differential debugging is the natural paradigm**: Developers manually used HF for comparison in ~50% of real issues. Ekka automates existing expert workflows rather than inventing a new one.
- **Change-point analysis is crucial**: Since errors propagate and amplify, looking at the maximum error point is often misleading. Change-point analysis on the full sequence is necessary to catch the first jump.
- **\$30/case is a viable threshold**: Compared to weeks of manual developer effort, this represents an order-of-magnitude cost saving.

## Highlights & Insights
- **Problem formulation as a major contribution**: Reframing "LLM serving silent errors" from a manual experience-based task into an automated differential debugging task with an oracle opens a new design space. This paradigm can extend to compiler optimizations, quantization, and distributed training.
- **Model Tree as an "IR for LLMs"**: Directly feeding full code to an agent causes information overload. Structural compression plus tree alignment allows the agent to focus on semantic matching while retaining the ability to "double-click" into source code.
- **Decoupling Detection and Localization**: Using robust error ratios for "if" and change-point detection for "where" allows the system to utilize mature statistical techniques for each sub-problem respectively.
- **Real-world Value**: The discovery of 4 confirmed new bugs demonstrates that Ekka is not just a benchmark exercise but provides actual utility for industrial serving frameworks.

## Limitations & Future Work
- **Dependency on Reference Implementation**: Ekka fails for entirely new architectures not yet supported by HuggingFace or for custom fused kernels without equivalent HF implementations.
- **Model Stack Scope**: It does not handle silent errors in high-level orchestration (schedulers, etc.), which account for roughly 30.6% of the bugs studied.
- **Cost Scalability**: The \$30/case cost may scale linearly with model size or sequence length due to token usage and dump overhead.
- **Framework Generalization**: Mapping validation was focused on vLLM and SGLang; performance on closed-source or non-Python frameworks (e.g., TensorRT-LLM) is unverified.
- **Future Directions**: Integration with version bisecting (finding the specific commit), trace caching to share snapshots between issues, and introducing finer-grained oracles like PyTorch eager mode.

## Related Work & Insights
- **vs. Traditional Fault Localization**: Methods like SBFL require pass/fail labels. Ekka uses a reference implementation to bypass the need for explicit failure signals.
- **vs. Deep Learning Testing**: Tools like CRADLE treat frameworks as black boxes. Ekka improves the granularity from "operator equivalence" to "component-level root cause" inside the engine.
- **vs. General Agentic Debugging**: General agents lack domain-specific scaffolding for LLM inference (KV-cache, attention backends). Ekka provides "domain scaffolding" (Model Tree, alignment, etc.) to guide the agent.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[ICML 2026\] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)
- [\[NeurIPS 2025\] Silent Tokens, Loud Effects: Padding in LLMs](../../NeurIPS2025/llm_efficiency/silent_tokens_loud_effects_padding_in_llms.md)

</div>

<!-- RELATED:END -->
