---
title: >-
  [Paper Note] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression
description: >-
  [ICML 2026][Model Compression][KV cache compression] This paper first uses the new benchmark system KVFundaBench to reveal a key asymmetry: "retrieval-type long-context tasks can be compressed…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "KV cache compression"
  - "high-density reasoning"
  - "few-shot semantic unit"
  - "prefill-decoding separation"
  - "long-context generation"
date: 2026-05-08
content_hash: a4fa629e3c90066e
---

# Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression

**Conference**: ICML 2026  
**arXiv**: [2502.01941](https://arxiv.org/abs/2502.01941)  
**Code**: None (no public link provided)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: KV cache compression, high-density reasoning, few-shot semantic unit, prefill-decoding separation, long-context generation

## TL;DR
This paper first uses the new benchmark system KVFundaBench to reveal a key asymmetry: "retrieval-type long-context tasks can be compressed, reasoning-type cannot." The root cause is attributed to KV compression breaking the integrity of few-shot examples as "semantic units." Based on this, ShotKV is proposed—preserving each shot as an indivisible unit during prefill, and applying dynamic token-level compression during decoding. This allows LG-GSM8K to improve from a baseline of 46.0 to 47.33 at a 40% compression rate, and reduces end-to-end latency by 11.3% under long-input settings.

## Background & Motivation

**Background**: Mainstream KV cache compression methods (H2O, SnapKV, StreamingLLM, PyramidKV, ChunkKV, Quest, etc.) are almost exclusively evaluated on "retrieval/localization" benchmarks like LongBench and NIAH, concluding that retaining only ~50% of tokens does not sacrifice accuracy.

**Limitations of Prior Work**: The authors observe a neglected workload—"High-Density Reasoning," where nearly every token in the prompt is critical for reasoning (e.g., CoT few-shot examples, multi-step arithmetic), rather than only a small "needle" segment being important. In such scenarios, arithmetic tasks suffer much greater performance drops than retrieval tasks at the same compression rate, and breaking a semantic chain in the reasoning process can cause catastrophic failure.

**Key Challenge**: Existing token-level KV compression scores and discards tokens individually based on attention scores, which fragments complete few-shot examples; chunk-level methods, while preserving blocks, apply the same strategy to both prefill and decoding, failing to balance "static instruction integrity" and "dynamic generation freshness."

**Goal**: (1) Provide a systematic benchmark, KVFundaBench, covering 5 fundamental abilities plus long-form generation; (2) Quantify which tasks are most sensitive to compression and which model types are most robust; (3) Operationalize "semantic integrity" as a compression principle, and construct a lightweight proof-of-concept, ShotKV, to validate the hypothesis.

**Key Insight**: Treat each shot in few-shot prompts as an "indivisible" Semantic Unit; during prefill, score and retain entire shots, while during decoding, independently apply token-level attention-top-k dynamic compression, thus explicitly separating the two types of information needs.

**Core Idea**: "Compression should be performed at the semantic unit level, and prefill and decoding must be handled separately"—this is the core conclusion from the benchmark, with ShotKV as the minimal viable implementation.

## Method

### Overall Architecture
Two parallel tracks: one is **KVFundaBench**, covering 5 task types (MMLU World Knowledge WK, CommonsenseQA CSR, GSM8K Arithmetic AR, HumanEval Code CG, JailBreakV Safety SA) plus LG-GSM8K long-form generation, systematically evaluated across four models (LLaMA-3.1-8B/Instruct, Mistral-7B-Instruct, DeepSeek-R1-Distill-Llama-8B) and six KV compression methods. Relative performance is defined as $\Delta P = (P_C - P_{\text{base}})/P_{\text{base}}$. The other is **ShotKV**: the prompt is split into $n$ shots $\{s_1,\dots,s_n\}$; for each layer $l$, shot importance is computed as $\text{Score}_{\text{prefill}}^l(s_i)=\frac{1}{k_i}\sum_{t\in s_i}\sum_h \alpha_{t,h}^l$, and shots are retained in descending order until the budget $r_p \cdot |KV_{\text{prefill}}|$ is filled; decoding uses an independent ratio $r_d$ for token-level attention top-k. The final $KV_{\text{total},l}=KV^C_{\text{prefill},l}\cup KV^C_{\text{decoding},l}$.

### Key Designs

1. **KVFundaBench Reveals Task-Dependent Degradation**:

    - **Function**: First systematic measurement of "differential degradation of KV compression across fundamental abilities."
    - **Mechanism**: Six empirical observations—(O1) WK/CSR are robust, AR/CG/SA collapse at compression rates < 20%; (O2) DeepSeek-R1 is more robust than instruct-tuned models; (O3) Short prompts are more fragile than long prompts (1-shot drops from 0.5 to 0.05 at 10% retention); (O4) Chunk-level methods (ChunkKV) are most robust for many-shot; (O5) Tasks with greater prompt gains are more sensitive (AR 0-shot→CoT improves 50.41% but is also most vulnerable); (O6) Long-context generation (LG-GSM8K) suffers random loss over 20%.
    - **Design Motivation**: Existing benchmarks focus reasoning ability on "sink token + retrieval head," masking the truly fragile "semantic chain"; attention heatmaps further confirm that arithmetic tasks have more diffuse non-sink attention (Fig. 3b), making token-level eviction more likely to cut critical chains.

2. **Shot-aware Prefill Retention (Semantic-Unit Preservation)**:

    - **Function**: Treats few-shot examples as atomic units, retaining them as a whole and preventing mid-example truncation.
    - **Mechanism**: First parses prompt boundaries to identify $n$ shots; for each layer $l$, computes average attention score per shot, independently selects top-K per layer until the total token count does not exceed the budget; selected shots are fully cached, with no mid-shot eviction allowed. After one compression, the prefill cache remains fixed during generation.
    - **Design Motivation**: Token-level methods like H2O/SnapKV may retain a shot's question but discard its answer, breaking the CoT causal chain; ChunkKV has shown contiguous chunks outperform discrete tokens, and this work further semanticizes "chunk" as "shot," allowing "different shots per layer" to leverage inter-layer attention specialization.

3. **Separate Compression for Prefill / Decoding Phases**:

    - **Function**: Allows static instructions and dynamic generation to use independent strategies.
    - **Mechanism**: Prefill uses the above shot-level retention (ratio $r_p$); decoding phase independently applies token-level TopK per layer based on $\text{Score}_{\text{decoding}}^l(t)=\sum_h \alpha_{t,h}^l$, with retention ratio $r_d$; the two sets of compressed results are merged per layer.
    - **Design Motivation**: Observation 6 shows that long-form generation (4k+ tokens) is especially ill-suited to unified compression strategies—ChunkKV/SnapKV lack dynamic eviction, causing decoding-side cache overflow; using dynamic strategies for prefill would repeatedly break in-context examples. Independent strategies for both sides is the natural trade-off solution.

### Loss & Training
ShotKV is a training-free inference-time method with no additional training; the only hyperparameters are $(r_p, r_d)$. Temperature is set to 0, $K=35, T=20$ (LG-GSM8K). For non-ICL document QA like HotpotQA, treating each sentence as a "shot" allows direct adaptation without retraining.

## Key Experimental Results

### Main Results

| Task / Method (Compression Rate) | FullKV | StreamingLLM | H2O | PyramidInfer | ChunkKV | SnapKV | **ShotKV** |
|---|---|---|---|---|---|---|---|
| LG-GSM8K @40% | 46.00 | 39.50 | 32.66 | 38.33 | — | — | **47.33** |
| LG-GSM8K @30% | 46.00 | 14.83 | 19.83 | 20.50 | — | — | **38.33** |
| LG-GSM8K @25% | 46.00 | 6.33 | 14.83 | 16.67 | — | — | **26.83** |
| Many-shot AR @10% | 82.35 | 74.32 | 51.27 | 70.37 | 79.32 | 68.27 | **80.37** |
| HotpotQA (LLaMA-3) @10% | 45.55 | 40.27 | 40.84 | 43.36 | 43.27 | — | **43.60** |

### Ablation Study

| Configuration | Many-shot AR @10% | Description |
|------|---------------------|------|
| ShotKV (full) | 80.37 | Complete method |
| Random Shot (same shot granularity but random selection) | 51.34 | Validates necessity of attention-based scoring, 29-point gap |
| Prefill shot-aware only (no decoding dynamic compression) | Rapid loss in long-form generation | Validates phase separation |
| ChunkKV (chunk but not shot boundary) | 79.32 | Shows shot semantic boundary outperforms generic chunk |

| Latency & Throughput | Input×Output | Latency (s) ↓ | Throughput (T/S) ↑ |
|------------|----------|----------------|---------------------|
| FullKV | 4096×4096 | 175.50 | 37.73 |
| ShotKV | 4096×4096 | 162.85 (**-7.2%**) | 41.12 (+9.0%) |
| FullKV | 8192×4096 | 183.42 | 55.93 |
| ShotKV | 8192×4096 | 162.78 (**-11.3%**) | 63.24 (+13.1%) |

### Key Findings
- Prompt-gain is strongly positively correlated with compression sensitivity: tasks with greater CoT improvement are more sensitive to KV compression (AR vs WK improvement +50.41 vs +6.20, with sensitivity gap magnified in the same direction), indicating that "tasks most reliant on in-context learning are most vulnerable to cache compression."
- DeepSeek-R1-Distill maintains ~0.60 accuracy at 10% compression, significantly higher than instruct-tuned LLaMA at 0.50; reasoning models' attention patterns are more robust, providing empirical support for deploying "reasoning models + aggressive compression."
- In non-ICL HotpotQA document QA, treating "sentences" as shots allows ShotKV to remain near-optimal at 10% compression; this demonstrates that the semantic unit concept can be smoothly transferred to any long text with natural segmentation boundaries.

## Highlights & Insights
- This is one of the rare works that "first conducts a rigorous benchmark, then proposes a minimal method based on the findings"; the authors explicitly state that ShotKV is "not an algorithmic innovation," but rather a hypothesis test that "semantic unit preservation > token preservation," showing commendable honesty. The paper's value lies more in the benchmark and insights.
- The "one-time prefill compression and freeze, dynamic scoring for decoding" two-stage structure can be directly reused by other KV compression methods—this is an essential adaptation for long-context generation, and is orthogonal and composable with KV quantization and cross-layer KV sharing.
- The strong correlation between prompt-gain and compression sensitivity is a highly practical deployment heuristic: online, one can estimate the safe compression threshold for a task based on its CoT sensitivity, without running the full benchmark for every task.

## Limitations & Future Work
- ShotKV requires direct access to the KV cache, so it is only applicable to self-hosted/open-source models (LLaMA, Mistral, DeepSeek, Qwen), and is ineffective for closed API models; it still relies on an attention-derived heuristic score, which the authors acknowledge is not a "principled semantic importance metric."
- In scenarios without few-shot structure or explicit sentence boundaries, such as "fully zero-shot long-document summarization," the shot concept fails; the authors only demonstrate sentence-level adaptation, and have not validated more complex structures like dialogue or code review.
- The benchmark covers only 5 fundamental abilities plus long-form generation, and does not include agentic tool use, multi-turn long dialogue, or RAG multi-document concatenation—real-world long-context workloads; ShotKV's performance in these scenarios remains to be validated.

## Related Work & Insights
- **vs ChunkKV (Liu et al., 2025)**: ChunkKV preserves contiguous blocks but uses a unified strategy; ShotKV semanticizes chunks as shot boundaries and adds prefill/decoding separation, effectively "ChunkKV + semantic boundary + phase separation."
- **vs SCOPE (Wu et al., 2025)**: SCOPE already proposed prefill/decoding separation, but did not combine it with the semantic unit concept; ShotKV merges both into a complete proof-of-concept.
- **vs H2O / SnapKV**: Both are token-level attention top-k; the Random Shot experiment indirectly proves that even with shot granularity, random selection is 29 points worse than attention-aware selection—"correct granularity + correct scoring" are both necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ The benchmark exposes the long-overlooked high-density reasoning dimension, which is more valuable than the method itself.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six observations × multiple models × multiple compression methods × multiple compression rates, with very comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear three-part narrative ("benchmark → insight → proof-of-concept"), with the authors proactively clarifying the method's simplicity to avoid over-marketing.
- Value: ⭐⭐⭐⭐ ShotKV is immediately usable and orthogonal to quantization and cross-layer sharing; the benchmark can serve as a de facto standard for future KV compression papers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](../../ACL2026/model_compression/fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](../../NeurIPS2025/model_compression/inference-time_hyper-scaling_with_kv_cache_compression.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)

</div>

<!-- RELATED:END -->
