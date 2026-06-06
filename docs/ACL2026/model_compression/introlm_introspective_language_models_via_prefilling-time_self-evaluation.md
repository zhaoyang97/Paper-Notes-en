---
title: >-
  [Paper Note] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation
description: >-
  [ACL 2026][Model Compression][Self-evaluation] IntroLM appends special `[CPX]` introspective tokens to the end of a prompt and uses "token-conditional LoRA" (active only for those tokens) to calculate "Can I answer this…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Self-evaluation"
  - "complexity prediction"
  - "prefilling"
  - "token-conditional LoRA"
  - "LLM routing"
date: 2026-05-08
content_hash: fb0737e1fc7dc557
---

# IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation

**Conference**: ACL 2026  
**arXiv**: [2601.03511](https://arxiv.org/abs/2601.03511)  
**Code**: https://github.com/hhosseini1377/LLM_routing (Available)  
**Area**: LLM Inference / Model Routing / Introspective Self-Evaluation  
**Keywords**: Self-evaluation, complexity prediction, prefilling, token-conditional LoRA, LLM routing

## TL;DR
IntroLM appends special `[CPX]` introspective tokens to the end of a prompt and uses "token-conditional LoRA" (active only for those tokens) to calculate "Can I answer this prompt correctly?" during the prefilling stage in one go. This self-evaluation does not enter the KV cache and does not affect generation. On long-context QA such as HotpotQA, the ROC-AUC is 14 points higher than DeBERTa-v3-Large. When used for model routing, it saves up to 50% of large model calls and 33% end-to-end latency.

## Background & Motivation
**Background**: LLM deployment currently commonly uses a "small model first, escalate to large for complex cases" routing strategy. The key is to predict whether a prompt exceeds the small model's capability before generation. Mainstream approaches train independent BERT-style classifiers (DeBERTa, MiniLM, etc.) for pre-routing.

**Limitations of Prior Work**: (a) BERT's context window is capped at 512 tokens, whereas modern RAG / long-document QA prefilling inputs easily range from thousands to hundreds of thousands of tokens, which BERT cannot "see" entirely; (b) using a stronger independent large model as an evaluator offsets the cost-saving purpose of routing itself; (c) posterior verification schemes (FrugalGPT, AutoMix) can only judge after generation, incurring high latency costs; (d) existing "confidence token" work (such as Chuang 2025's `<CN>/<UN>`) generates tokens at the end of decoding, which still requires generation before judgment and is slow.

**Key Challenge**: Complexity prediction requires "sufficiently strong semantic modeling + visibility of the complete long context," but running an additional large model evaluator brings the cost back to the starting point; meanwhile, modifying the main model itself for evaluation would disrupt its generation behavior.

**Goal**: To let the causal LLM **itself** predict its output quality during the prefilling phase, without introducing new models, affecting the KV cache, or altering the generation distribution.

**Key Insight**: Each token during prefilling aggregates information from the complete prompt via self-attention—by "attaching" a few special tokens at the end of the prompt to read the entire context and using a small classification head to read their hidden states, self-evaluation can be completed at zero extra inference cost.

**Core Idea**: Use `[CPX]` introspective tokens + token-conditional LoRA (low-rank adaptation that only takes effect for `[CPX]`) to decouple the "introspection path" from the "generation path," where the former is used for complexity estimation and the latter maintains the base model's behavior completely unchanged.

## Method

### Overall Architecture
Given a prompt $x$, IntroLM appends several `[CPX]` tokens and feeds them into the LLM for prefilling; each `[CPX]` reads the complete $x$ via unidirectional attention, and its hidden state is mapped by a lightweight linear classification head to $f_\theta(x)\in[0,1]$ (the probability that the small model answers correctly). Key engineering isolations:

- `[CPX]` tokens are not written to the KV cache: This ensures that subsequent tokens in the decoding phase cannot see `[CPX]`, making the generation distribution identical to the base model.
- Decoding starts from the "hidden state of the last token of the original prompt": It does not start from `[CPX]`, preventing prompt representations from being contaminated by the self-evaluation objective.
- Backbone parameters are fully frozen; trainable parameters = `[CPX]` token embeddings + classification head + token-conditional LoRA (< 1% of model size).

Downstream application: Compare $f_\theta(x)$ with a threshold $\alpha$ to decide whether to route to the small model $M_s$ or the large model $M_\ell$, and the prefilling state of $M_s$ can be directly reused for decoding without waste.

### Key Designs

1.  **`[CPX]` Introspective Tokens**:
    - **Function**: Insert special tokens at the end of the prompt as "dedicated reading slots for complexity estimation," aggregating the semantics of the entire prompt through self-attention during prefilling.
    - **Mechanism**: Since it is a decoder-only model and `[CPX]` is at the end, each `[CPX]` can attend to the complete $x$; however, by excluding `[CPX]` from the KV cache and not starting decoding from its hidden state, the generation trajectory remains completely independent. A light linear head reads the final hidden state of `[CPX]` to output binary classification logits.
    - **Design Motivation**: Using the hidden state of the last prompt token directly for classification ("Backbone only" in ablation) performs poorly because that token is optimized for "generating the next token," not for "judging complexity"; the newly introduced `[CPX]` can learn task-specific introspective representations.

2.  **Token-Conditional LoRA**:
    - **Function**: Let LoRA updates **take effect only for `[CPX]` tokens** while leaving the original prompt tokens completely unchanged, thereby enabling customized representations for introspection while maintaining zero distortion of generation behavior.
    - **Mechanism**: After calculating the standard output $HW$ and LoRA output $H\Delta W$ for any linear layer $W$, an element-wise multiplication is performed with a binary mask $M$ (`[CPX]` positions = 1, others = 0): $HW + (H\Delta W)\odot M$. Token-conditional LoRA is added only to query projections (`q_proj`), output projections (`o_proj`), and FFN gate/up/down; key and value projections are generated by prompt tokens and determine the overall attention pattern, so they remain frozen.
    - **Design Motivation**: Standard LoRA rewrites representations of all tokens, which inevitably interferes with generation; token-level masking is equivalent to "forking" a parallel computation path belonging only to `[CPX]` within the same set of weights, allowing the learning of classification-specific features with zero interference to the main task—this is the most clever engineering detail of the method.

3.  **Prefilling Reuse + Routing Strategy**:
    - **Function**: Embed the introspection result directly into routing; if $f_\theta(x)\ge\alpha$, use the small model $M_s$ and reuse its prefilling KV cache; otherwise, upgrade to the large model $M_\ell$ and restart.
    - **Mechanism**: Since prefilling has already run on $M_s$ (incidentally producing $f_\theta(x)$), if no upgrade is needed, the current KV can be used directly for decoding, saving a redundant prefilling step; the full cost of $M_\ell$ prefilling + decoding is only paid when an upgrade is truly necessary. Expected latency: $T^\alpha_{\text{IntroLM}}(L)=\mathrm{TTFT}_{M_s}+(1-c_\ell)(L-1)\mathrm{TPOT}_{M_s}+c_\ell T_{M_\ell}(L)$.
    - **Design Motivation**: Traditional BERT routers make decisions before $M_s$, saving cases where "neither small nor large model runs"; IntroLM makes decisions after the small model has already warmed up prefilling, so the "small model" path only leaves decoding, further compressing end-to-end latency.

### Loss & Training
- Loss: Class-weighted binary cross-entropy; labels are automatically generated by LLM-as-judge (LLaMA-3.1-8B-Instruct for General QA, Qwen2.5-32B-Instruct for Chat, scoring 0–10 with a threshold of 8).
- LoRA Configuration: rank 32, $\alpha=64$; batch size 64; context 2048; cosine LR + 10% warmup; max grad norm 0.3; weight decay 0.002; LR between $4\times 10^{-5}$ and $8\times 10^{-5}$.
- Data: General QA = MMLU + MMLU-Pro + GSM8K (136K questions, 21% complex samples); HotpotQA (97K, 14% complex); LMSYS-Chat-1M (100K English prompts, ~50% complex). 80/10/10 split.

## Key Experimental Results

### Main Results
Complexity prediction performance (ROC-AUC / PR-AUC, higher is better):

| Method | General QA ROC | General QA PR | HotpotQA ROC | HotpotQA PR | Chat ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeBERTa-v3-Base (184M) | 74.3 | 44.3 | 69.4 | 24.3 | 82.6 |
| DeBERTa-v3-Large (435M) | 75.8 | 45.5 | 71.8 | 26.8 | 86.3 |
| Matrix Factorization | 69.2 | 39.8 | 52.1 | 14.0 | 76.1 |
| **IntroLM (Qwen3-8B)** | **89.1** | **63.4** | **86.3** | **46.7** | **90.1** |

On HotpotQA, the ROC-AUC surpassed DeBERTa-Large by exactly 14.5 points (71.8 → 86.3), and PR-AUC nearly doubled (26.8 → 46.7)—this is precisely the long-context scenario where BERT's context window is a bottleneck.

End-to-end routing gains (vLLM + 2×H100, $M_s$=Qwen3-8B, $M_\ell$=Qwen3-32B):

| Dataset | Reduction in Large Model Calls | Reduction in Latency |
| :--- | :--- | :--- |
| General QA | Up to 50% (Avg 30%) | Up to 34% (Avg 15%) |
| HotpotQA | Up to 49% (Avg 41%) | Up to 30% (Avg 18%) |

### Ablation Study

Necessity of `[CPX]` tokens (HotpotQA):

| Configuration | ROC | PR | Description |
| :--- | :--- | :--- | :--- |
| Backbone only (Last prompt token hidden state) | 81.0 | 35.8 | Using generation-specific representation for self-evaluation |
| **IntroLM (Qwen3-8B, with `[CPX]`)** | **86.3** | **46.7** | +5.3 ROC, +10.9 PR |

Ablation of token-conditional LoRA targets (General QA, Qwen3-8B):

| LoRA Target | ROC | PR |
| :--- | :--- | :--- |
| No LoRA (Only train `[CPX]` embedding + head) | 85.7 | 56.4 |
| Attention only (q/o) | 88.5 | 61.0 |
| FFN only | 89.1 | 63.0 |
| Attention + FFN (Default) | 89.1 | 63.1 |

Backbone capacity ablation (General QA): IntroLM 84.2 vs. DeBERTa-Large 75.7 on Qwen3-1.7B; 89.1 vs. 75.8 on Qwen3-8B. Larger models provide stronger introspective signals.

### Key Findings
- Token-conditional LoRA is the key turning point: Removing it causes ROC-AUC to drop by 3.4 points and PR-AUC to drop by 6.7 points, proving that "soft prompt + classification head" alone is insufficient and a learnable representation path for `[CPX]` is necessary.
- FFN-only LoRA is sufficient: Adapting FFN alone (89.1) performs equally to attention+FFN full adaptation (89.1), allowing for parameter savings by using only FFN-LoRA.
- Long context is the Achilles' heel of BERT: On HotpotQA, IntroLM pulled PR-AUC from 26.8 to 46.7, nearly doubling it, validating that "prefilling with the full context" is a fundamental advantage over BERT.
- Introspective signals can be read from intermediate layers: Qwen3-8B can achieve 87.9 ROC (vs. 89.1 for the full model) using only the first 24/36 layers, suggesting that routing decisions can be made via early-exit during prefilling.
- Prefix truncation vs. Intermediate layers: Reading only the first 512 tokens on HotpotQA caused IntroLM to drop from 86.3 to 77.8 (still exceeding DeBERTa but with significant loss), proving that early decisions should follow "intermediate layers + full context" rather than "full layers + truncated context."
- Cross-model prediction is feasible: Using IntroLM from Qwen3-8B to predict Qwen3-1.7B success yields 83.8 ROC ≈ 84.2 from Qwen3-1.7B's own self-evaluation, and far exceeds DeBERTa's 75.7. This implies `[CPX]` representations have transferability to other candidate models and can scale to multi-model routing.

## Highlights & Insights
- **Introspection without side effects on generation**: The combination of "`[CPX]` not in KV cache + token-conditional LoRA" is the most exquisite engineering detail of the paper, achieving the best of both worlds—leveraging the full representation capability of the backbone for evaluation while ensuring the generation distribution is identical to the base. This "zero-distortion of the main task" constraint has strong transferability in LLM adaptation (e.g., alignment, safety monitoring).
- **A win-win in latency trade-offs**: Moving the routing decision from "pre-BERT" to "post-small model prefilling" seemingly just shifts the position, but actually opens the KV cache reuse channel—the small model path incurs almost no additional cost, which is the true source of the 33% latency reduction.
- **"Hierarchical Advancement" of complexity signals**: Reliability of introspective signals in intermediate layers means early-exit upgrades are possible halfway through prefilling, holding great potential for extremely long contexts (e.g., RAG over 100k tokens).
- **Prophecy of cross-model transfer**: Qwen3-8B's near-perfect prediction of Qwen3-1.7B suggests that `[CPX]` learns a more universal "prompt difficulty" feature rather than a model-specific "can I do this" signal. Attaching multiple heads could extend this to multi-model routing.

## Limitations & Future Work
- The authors acknowledge: (a) Experiments focus on QA and chat, validation on creative writing, multi-turn dialogue, or code generation is pending; (b) training costs are significantly higher than BERT classifiers (LoRA on an 8B model); (c) labels depend on LLM-as-judge, introducing judge model bias.
- Observation: (d) High ROC-AUC does not guarantee the routing won't be "confidently wrong"; the paper does not report calibration error or reliability diagrams; (e) hyper-parameters like the number and position of `[CPX]` were not systematically swept; (f) on the Chat set, the advantage over BERT narrowed from 14 points to 4 points, indicating that for short context prompts, "introspection" differentiation is quickly eroded.
- Future directions: Combine `[CPX]` into a "multi-head multi-model router" + "early-exit intermediate layer" combo, incorporating calibration loss to output reliability bounds for production-grade adaptive routers.

## Related Work & Insights
- **vs. FrugalGPT / AutoMix**: They judge quality after generation, paying the decoding cost; IntroLM provides the answer during the prefilling phase, saving an entire round of decoding.
- **vs. RouteLLM / HybridLLM / BEST-Route**: BERT-style encoder routing faces a hard ceiling of a 512 context window; IntroLM naturally supports arbitrary lengths using the LLM backbone.
- **vs. Chuang 2025 confidence tokens (`<CN>/<UN>`)**: They generate tokens at the end of decoding to report confidence, requiring decoding first; IntroLM moves confidence to the prefilling phase without affecting the generation trajectory, serving as a "pre-positioned + isolated" upgrade.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of "token-conditional LoRA + introspective tokens excluding KV cache" is a truly clean new design, providing a paradigm-level approach to self-evaluation without modifying generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid across three datasets × multiple backbones × various ablations (LoRA target / layers / prefix truncation / cross-model); however, it lacks calibration and production-grade workload testing.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas, intuitive figures 1-3, and well-documented engineering details.
- Value: ⭐⭐⭐⭐⭐ High direct application value—saving 33% latency and half of the large model calls is significant for any LLM serving, and the code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](../../ICLR2026/model_compression/beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)

</div>

<!-- RELATED:END -->
