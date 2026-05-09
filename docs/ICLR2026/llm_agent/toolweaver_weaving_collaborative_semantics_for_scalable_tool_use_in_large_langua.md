---
title: >-
  [Paper Note] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models
description: >-
  [ICLR 2026][LLM Agent][Tool use] ToolWeaver is proposed to represent each tool as a hierarchical discrete code sequence (rather than a single token) via collaboration-aware vector quantization, achieving logarithmic vocabulary scaling (47,000+ tools requiring only ~512 new tokens). It comprehensively outperforms the ToolGen baseline on ToolBench while reducing language model perplexity degradation from 16.5× to 4×.
tags:
  - ICLR 2026
  - LLM Agent
  - Tool use
  - vector quantization
  - collaborative semantics
  - vocabulary expansion
  - generative tool retrieval
date: 2026-05-08
content_hash: 9c976db802371046
---

# ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2601.21947](https://arxiv.org/abs/2601.21947)
**Code**: Available
**Area**: LLM Agent
**Keywords**: Tool use, vector quantization, collaborative semantics, vocabulary expansion, generative tool retrieval

## TL;DR
ToolWeaver is proposed to represent each tool as a hierarchical discrete code sequence (rather than a single token) via collaboration-aware vector quantization, achieving logarithmic vocabulary scaling (47,000+ tools requiring only ~512 new tokens). It comprehensively outperforms the ToolGen baseline on ToolBench while reducing language model perplexity degradation from 16.5× to 4×.

## Background & Motivation

**Background**: Generative tool use (e.g., ToolGen) represents tools as new tokens, allowing LLMs to directly "utter" tool names rather than retrieving from a candidate list. However, existing "one tool, one token" approaches face severe scalability challenges.

**Limitations of Prior Work**:
- **Vocabulary explosion**: 47,000 tools → 47,000 new tokens, causing linear growth in the embedding layer.
- **Language capability catastrophe**: Massive new tokens corrupt the language model; ToolGen raises perplexity from 6.34 to 104.54 (16.5×).
- **Semantic blindness**: A single token cannot encode inter-tool collaborative relationships; the model can only infer collaboration patterns from sparse tool ID co-occurrences.
- **Poor cross-domain generalization**: Single token IDs lack transferable semantic structure.

**Key Challenge**: A compact representation of large-scale tool libraries is required that simultaneously encodes intrinsic tool semantics (functionality) and extrinsic collaboration patterns (co-usage patterns) without degrading pretrained language capabilities.

**Goal**: Design a logarithmic vocabulary expansion scheme that jointly encodes semantic and collaborative information.

**Key Insight**: Inspired by VQ-VAE, tool embeddings are quantized into multi-level discrete codes via Residual Quantization (RQ)—$K^L$ tools require only $L \times K$ new tokens—with graph Laplacian regularization injecting collaborative signals.

**Core Idea**: Encode tools as hierarchical token sequences via collaboration-aware residual quantization, enabling logarithmic vocabulary scaling and collaborative semantic modeling.

## Method

### Overall Architecture
Stage 1: A text encoder obtains semantic embeddings for tools.
Stage 2: Collaboration-aware RQ-VAE quantizes embeddings into discrete code sequences.
Stage 3: Sinkhorn optimal transport resolves code conflicts.
Stage 4: Two-stage LLM fine-tuning (retrieval alignment + trajectory alignment).
Inference: Constrained beam search ensures only valid tool codes are generated.

### Key Designs

1. **Collaboration-Aware Residual Quantization (Stage 2)**

   - **Function**: Quantizes tool embeddings into multi-level discrete codes $[\iota_1, \iota_2, \ldots, \iota_L]$.
   - **Mechanism**: RQ-VAE quantizes residuals level by level: $\iota_{d,l} = \arg\min_k \|r_{d,l} - v_{l,k}\|^2$, $r_{d,l+1} = r_{d,l} - v_{l,\iota_{d,l}}$.
   - **Collaborative Regularization** (core innovation): A tool co-occurrence matrix is computed from usage trajectories as $A_{uv} = C_{uv}/\sqrt{C_{uu} \cdot C_{vv}}$, and a graph Laplacian loss $\mathcal{L}_{collab} = \sum_{u,v} A_{uv}\|\hat{z}_u - \hat{z}_v\|^2$ is added.
   - **Design Motivation**: Tools that are frequently used together are assigned similar codes, enabling the LLM to learn collaboration patterns from code-level co-occurrence rather than sparse tool ID co-occurrence.
   - **Vocabulary size**: $L$ levels × $K$ codebook entries = $L \times K$ new tokens. With $L=4, K=128$, 512 tokens cover 47,000+ tools.

2. **Conflict Mitigation via Optimal Transport (Stage 3)**

   - **Function**: Resolves conflicts where multiple tools map to the same code sequence.
   - **Mechanism**: A uniform distribution is enforced over the final-level codebook using the Sinkhorn-Knopp algorithm to solve the optimal transport problem.
   - **Constraints**: Each tool is fully assigned, and each codeword is used uniformly.
   - **Design Motivation**: Guarantees a unique identifier for each tool.

3. **Constrained Beam Search Inference**

   - **Function**: Ensures that only valid tool codes are produced during generation.
   - **Mechanism**: All valid code sequences are precomputed into a prefix tree (Trie); invalid next tokens are masked at inference time.
   - **Design Motivation**: Prevents the generation of non-existent tool codes.

### Loss & Training
- Quantization loss: $\mathcal{L}_{tokenize} = \mathcal{L}_{recon} + \mathcal{L}_{quant} + \lambda\mathcal{L}_{collab}$, with optimal $\lambda=1.0$.
- Retrieval alignment: $\mathcal{L}_{retrieval} = -\mathbb{E}[\log P(\boldsymbol{\iota}_d | q)]$ (489K query–tool pairs).
- Trajectory alignment: Standard SFT loss (183K trajectories).

## Key Experimental Results

### Main Results (ToolBench, 47,000+ tools)

| Method | I1 NDCG@1 | I3 NDCG@1 | I3 SoPR | WikiText-2 PPL |
|--------|-----------|-----------|---------|----------------|
| BM25 | 26.92 | 10.00 | - | - |
| ToolRetriever | 75.92 | 28.00 | - | - |
| ToolGen | 88.50 | 81.00 | 36.34% | 104.54 |
| **ToolWeaver** | **91.16** | **88.00** | **52.19%** | **25.36** |

ToolWeaver comprehensively outperforms all baselines on both retrieval and task completion, while reducing language model degradation from 16.5× to 4×.

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| $\lambda=0$ (no collaboration) | Significant I3 performance drop, validating that collaborative signals are critical for complex tasks. |
| $\lambda=1$ (optimal) | Best performance across all metrics. |
| $\lambda=10$ (over-constrained) | Performance degrades; collaborative signals override semantic information. |
| Removing semantic initialization | NDCG drops ~20; semantic grounding is an essential prerequisite. |
| Adding collaborative guidance | I1 improves by 1–2; I3 improves by 4–5; gains increase with task complexity. |
| Encoding strategy comparison | Atomic (ToolGen), Numerical, Hierarchical, and Semantic encodings all underperform ToolWeaver. |

### Key Findings
- **Logarithmic vs. linear scaling**: 512 new tokens vs. 47,000; token utilization improves by 16,400×.
- **Language capability preservation**: PPL 25.36 vs. 104.54; summarization F1 drops only 0.31% vs. 2.93%.
- **Collaborative signals are critical for complex tasks**: The largest improvements occur on I3 (cross-category multi-tool tasks) (+7 NDCG / +15.85 SoPR).
- **Semantic initialization is foundational**: The single largest contributing factor, demonstrating that high-quality tool representations are a prerequisite for all subsequent gains.
- **Structural additions alone are insufficient**: Hierarchical encoding and semantic name encoding both underperform ToolWeaver's learned collaborative encoding.

## Highlights & Insights
- **Logarithmic vocabulary scaling is a fundamental architectural improvement**: It transforms tool use from a "vocabulary expansion problem" into a "sequence generation problem," fundamentally resolving the scalability bottleneck of large-scale tool libraries. This paradigm generalizes to any scenario requiring large-scale discrete entity generation.
- **The collaborative regularization design is elegant**: Rather than simply encoding semantics, usage patterns (which tools are frequently co-used) are injected into the coding space, allowing the codes themselves to carry collaborative signals. This enables the LLM to learn tool combinations from code co-occurrence, which is far denser than tool ID co-occurrence.
- **Sinkhorn optimal transport resolves encoding conflicts**: Uniqueness is guaranteed through a mathematically principled approach that is more robust than hard-coded deduplication.

## Limitations & Future Work
- Collaborative signals depend on high-quality co-occurrence data; sparse or biased usage patterns may cause degradation.
- Evaluation is conducted solely on ToolBench; cross-domain transfer remains untested.
- Constrained beam search introduces additional inference overhead (not quantified).
- The hyperparameter $\lambda$ requires empirical tuning; no automatic configuration guidance is provided.

## Related Work & Insights
- **vs. ToolGen**: The one-tool-one-token baseline suffers from poor scalability (linear vocabulary growth + PPL catastrophe); ToolWeaver outperforms it on all metrics.
- **vs. ToolRetriever**: A retrieval-based method that is non-generative and exhibits limited recall in large-scale tool libraries.
- **vs. RQ-VAE in vision/recommendation**: ToolWeaver extends RQ-VAE from image/product quantization to tool semantic quantization; collaborative regularization is the key innovation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Logarithmic vocabulary scaling combined with collaboration-aware quantization constitutes an entirely new paradigm for tool representation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation at 47,000-tool scale, diverse ablations, and comprehensive language capability assessment.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear; ablation analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core scalability bottleneck of generative tool use with direct engineering value for agent systems.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](../../AAAI2026/llm_agent/autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](../../ACL2026/llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)

<!-- RELATED:END -->
