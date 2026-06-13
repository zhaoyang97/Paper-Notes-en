---
title: >-
  [Paper Note] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing
description: >-
  [ACL2026][Code Intelligence][Code Editing] This paper treats the "output format" of LLM code editing itself as a training objective, proposing BlockDiff, FuncDiff, and an adaptive format selection strategy…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Code Editing"
  - "Structured diff"
  - "AdaEdit"
  - "AST"
  - "Low-latency generation"
date: 2026-05-08
content_hash: 7dec83cf6771a1db
---

# To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing

**Conference**: ACL2026 Findings  
**arXiv**: [2604.27296](https://arxiv.org/abs/2604.27296)  
**Code**: https://github.com/nju-websoft/AdaEdit  
**Area**: Code Intelligence / LLM Code Editing / Editing Format Learning  
**Keywords**: Code Editing, Structured diff, AdaEdit, AST, Low-latency generation

## TL;DR
This paper treats the "output format" of LLM code editing itself as a training objective, proposing BlockDiff, FuncDiff, and an adaptive format selection strategy, AdaEdit. It achieves accuracy close to full-code generation while reducing latency and output token costs by over 30% in long-code editing scenarios.

## Background & Motivation
**Background**: LLM-based code editing has become a foundational capability in IDEs, code assistants, and automated repair systems. Mainstream training and evaluation typically require models to either output the full modified code directly based on editing intent or generate patch formats like unified diff or search-replace within strong model prompts.

**Limitations of Prior Work**: Generating the full code is the most natural for models because pre-training corpora consist largely of complete files, but it is highly inefficient. Even for a single-line change, the entire file must be re-output, leading to high latency, API costs, and risks of unintended modifications. Traditional diffs appear to save tokens but are unnatural for LLMs: line-numbered formats require precise offsets, and content-addressed diffs shatter code into fragmented hunks, breaking syntactic structures.

**Key Challenge**: Code editing must satisfy two simultaneous goals: the output should be as short as possible for low latency and cost, yet sufficiently natural and patchable to ensure functional correctness. Full-code is natural but redundant, while line-level diff is efficient but fragile; a single fixed format rarely covers all editing scales.

**Goal**: The authors aim to answer a fundamental question: What output representation should LLM-based code editing use? If a diff format is to be learned by the model, which code structures should it preserve? If a diff is actually longer than the full-code in certain scenarios, can the model automatically switch formats?

**Key Insight**: The paper first systematically compares conventional diffs, proving that line-number offsets and fragmented hunks are primary sources of failure. It then extends text diffs to syntactically complete blocks or functions using AST. Finally, it allows the model to learn to select the representation with fewer tokens between diff and full-code through SFT.

**Core Idea**: Elevate local text modifications to syntax-block-level rewrites and use AdaEdit to let the model select between "structured diff or full-code" on a per-sample basis, rather than fixing a single editing format.

## Method
The methodology consists of two layers. The first layer defines structure-aware diff formats: BlockDiff and FuncDiff. These remain text-based search/replace patches, but the hunk anchors and rewrite content are no longer arbitrary line fragments; they are syntax blocks within the AST. The second layer is AdaEdit, which does not invent new patch syntax but modifies training labels: for each sample, it compares the token length of the full target code and its diff representation, using the shorter one as the training target to internalize format selection logic.

### Overall Architecture
Given source code, editing intent, and target code, the system first calculates the minimal text difference, MinUniDiff, to ensure all text changes are covered. It then uses tree-sitter to construct an AST block tree and maps each diff hunk to the smallest syntax node or set of contiguous nodes. Subsequently, anchor expansion ensures the segment to be replaced is uniquely locatable in the source code. Overlapping hunks or multiple modifications within the same fine-grained node are handled via bottom-up merging. During training, the model receives editing intent and source code as input, producing output in full-code, BlockDiff, FuncDiff, or a hybrid format selected by AdaEdit.

### Key Designs
1. **Structure-Aware Hunks in BlockDiff / FuncDiff**:
	- **Function**: Ensures the diff representation preserves the syntactic integrity of the code, preventing the model from generating arbitrary fragmented lines.
	- **Mechanism**: BlockDiff allows editing at fine-grained AST nodes (including branches, loops, context blocks, and functions); FuncDiff ignores control structures, favoring function-level rewrites. Both represent changes as "locating a unique anchor and replacing its structured content."
	- **Design Motivation**: LLMs are more proficient at generating complete, syntactically coherent code blocks rather than diff hunks with missing headers or footers. Structured hunks sacrifice some local minimality for a more natural generation distribution and higher patch usability.

2. **Unique Anchor Expansion and Text Patching**:
	- **Function**: Guarantees that the structured diff output by the model can be automatically and unambiguously applied back to the source code.
	- **Mechanism**: After an initial mapping to the smallest AST node, if the text content is not unique in the source, neighboring sibling nodes are gradually added. Expansion continues to parent nodes if necessary, using the entire file as an anchor in extreme cases. The patching phase does not rely on the AST; it performs text search/replace with fault tolerance for whitespace and blank lines.
	- **Design Motivation**: If a structured diff cannot be stably patched, it cannot enter real IDE workflows. Using AST only for format generation while keeping the final patch text-level allows coverage of comments, spaces, and syntactically incorrect fragments that AST tools might ignore.

3. **AdaEdit Adaptive Format Selection**:
	- **Function**: Allows the model to automatically choose the most token-efficient output format based on editing scale and code length.
	- **Mechanism**: For each training sample with source code $C_j$ and target code $C'_j$, the token counts for the full-code representation and the diff representation are calculated. The objective $E_j = argmin(|C'_j|, |Diff(C_j,C'_j)|)$ is then used for supervision. This allows the model to output the shorter format during inference without an explicit classifier.
	- **Design Motivation**: Diffs are not always more efficient. When changes are scattered or extensive, the overhead of anchors and multiple hunks may exceed that of the full code. AdaEdit avoids "diffing for the sake of diffing" by delegating efficiency choices to the token cost signals in the training data.

### Loss & Training
The training objective is standard token-level cross-entropy. Main experiments use OCEData for Python editing training, evaluating on EditEval, CanItEdit, HumanEvalFix, Aider-1, and Aider-2. Models include DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, and Qwen2.5-Coder-14B. All models undergo full-parameter SFT from their base versions to isolate the impact of "output format" on results. Evaluation spans two dimensions: effectiveness (pass@1, patch-apply success rate, and linter checks) and efficiency (first-rendered tokens, total output tokens, and latency).

## Key Experimental Results

### Main Results
The main table compares full-code, traditional ContentDiff, structured BlockDiff/FuncDiff, and versions with AdaEdit based on average pass@1.

| Base Model | FullCode | ContentDiff | BlockDiff | BlockDiff + AdaEdit | FuncDiff | FuncDiff + AdaEdit | Main Conclusion |
|----------|----------|-------------|-----------|---------------------|----------|--------------------|----------|
| DeepSeek-Coder-6.7B | 52.21 | 48.92 | 50.64 | 52.16 | 50.79 | 52.55 | AdaEdit reaches or exceeds FullCode |
| Qwen2.5-Coder-7B | 57.07 | 54.43 | 55.98 | 57.61 | 57.32 | 57.95 | FuncDiff + AdaEdit performs best |
| Qwen2.5-Coder-14B | 63.89 | 62.16 | 64.11 | 63.92 | 64.89 | 64.68 | Stronger models better utilize structured diffs |

Issues with traditional diffs were evident: on Qwen2.5-Coder-7B, MinUniDiff achieved only 14.07 average pass@1, UniDiff 33.15, and even with line-number assistance, it reached only 31.13 / 37.66. ContentDiff improved to 54.43 but remained lower than FullCode's 57.07. This indicates that "removing line numbers" is only the first step; preserving syntactic structure is key.

### Ablation Study
The paper analyzes AdaEdit through long-code efficiency, format selection accuracy, and cross-language generalization.

| Setting | Pass@1 | Output Cost (tokens) | Description |
|------|--------|-----------------|------|
| FullCode, CanItEdit >300 tokens | 39.75 | 648.30 | Accurate but redundant |
| ContentDiff | 33.75 | 612.85 | Minimal savings and low accuracy |
| ContentDiff + AdaEdit | 33.00 | 432.73 | Cost reduced but accuracy remains weak |
| BlockDiff | 38.69 | 570.26 | Closer to FullCode |
| BlockDiff + AdaEdit | 37.94 | 466.04 | Cost significantly reduced |
| FuncDiff | 40.75 | 546.77 | Accuracy exceeds FullCode |
| FuncDiff + AdaEdit | 40.69 | 481.63 | Accuracy maintained, cost reduced ~25.7% |

JavaScript cross-language experiments also support the generalization ability of structured formats.

| Format | HumanEvalFix-JavaScript pass@1 | Conclusion |
|------|---------------------------------|------|
| Base model | 63.48 | Not fine-tuned for editing formats |
| FullCode | 66.13 | Strong baseline |
| ContentDiff | 56.55 | Traditional diff degrades significantly |
| ContentDiff + AdaEdit | 64.97 | AdaEdit mitigates but is insufficient |
| BlockDiff + AdaEdit | 65.70 | Close to FullCode |
| FuncDiff + AdaEdit | 67.74 | Exceeds FullCode |

### Key Findings
- The advantage of structured diff becomes more pronounced as model capability increases. On Qwen2.5-Coder-14B, FuncDiff reached 64.89, exceeding FullCode's 63.89, suggesting stronger models better understand structured editing formats.
- AdaEdit's format selection accuracy exceeds 90%, and if a 20% token deviation is allowed, average accuracy exceeds 95%. Few-shot strong models do not naturally possess this cost-benefit judgment, making training necessary.
- For long code, FullCode output tokens grow approximately linearly with code length. FuncDiff involves anchor overhead for short code but becomes more efficient as code length increases. AdaEdit automatically selects the lower-cost format across different scales.
- Usability analysis shows ContentDiff is more prone to patch failures due to non-unique anchors generated by the model. While line-numbered formats appear to have higher patch success rates, linter checks reveal more code breakage.

## Highlights & Insights
- The core value of this paper is not just another code edit benchmark, but the elevation of "output format" to a component of model training design. Many code intelligence works treat full-code or diff as engineering details; this proves that format directly impacts accuracy, latency, and cost.
- The trade-off between BlockDiff and FuncDiff is clear: BlockDiff is finer and potentially saves more tokens; FuncDiff is coarser, more natural, and more stable. In experiments, FuncDiff often showed higher accuracy, suggesting that for LLMs, "writing a slightly larger complete structure" might be more cost-effective than extreme localization.
- AdaEdit resembles encoding inference-time routing into data labels. It requires no additional classification head or pre-output format decisions; instead, it makes format selection part of the generation distribution via the supervision target.
- Insights for IDE products: User experience is typically determined by latency, patch success rate, and unintended changes, not just pass@1. Structured diff provides a midpoint that is more controllable than full-code and more natural than traditional diffs.

## Limitations & Future Work
- The effectiveness of structured diff depends on the base model's strength. Weaker models might not surpass FullCode when they lack understanding of the new formats; the paper explicitly notes that performance gains scale with model size.
- Training data is primarily Python-based. Although JavaScript experiments were added, repository-level editing, cross-file changes, build system modifications, and large-scale refactorings are not yet fully covered.
- Any diff-based method carries risks of patch failure or accidental code damage. The paper mitigates this with unique anchors and linter checks, but a real IDE still requires test runs, type checking, and rollback mechanisms.
- AdaEdit currently uses SFT to learn format selection from token length labels. Future work could integrate functional correctness, test results, and token costs into a single verifiable reward, using RL to explore more flexible editing formats.

## Related Work & Insights
- **vs FullCode generation**: FullCode aligns best with the pre-training distribution and has high accuracy but significant redundancy. This paper's structured diff attempts to maintain sufficiently complete syntactic structures while avoiding full-file rewrites.
- **vs UniDiff / MinUniDiff**: Traditional unified diffs rely on line numbers and offsets, which LLMs struggle to generate stably. Experimental results show that even with line numbers in the source code, accuracy remains far below full-code.
- **vs ContentDiff / search-replace**: Content-addressed diffs remove line-number fragility, but hunks remain arbitrary line fragments, leading to unnatural generation or non-unique anchors. BlockDiff/FuncDiff use AST nodes to resolve these issues.
- **vs AST edit scripts**: Tools like GumTree are suitable for program analysis, but their output is a sequence of operations or DSLs poorly suited for token-by-token LLM generation, often ignoring whitespace and comments. This paper retains text patching and is thus better for precise refactoring of code text.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematizes code editing formats as a trainable design and proposes structured diff + adaptive selection; the problem approach is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, benchmarks, long code, usability, and JavaScript generalization; repository-level editing needs further validation.
- Writing Quality: ⭐⭐⭐⭐☆ Logical and clear, diagnosing conventional diffs before proposing methods, with experimental tables directly supporting the claims.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for engineering code assistants, IDE editing, and low-latency patch generation; serves as a reference for training code editing models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] The Path Not Taken: Duality in Reasoning about Program Execution](the_path_not_taken_duality_in_reasoning_about_program_execution.md)
- [\[ACL 2026\] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)

</div>

<!-- RELATED:END -->
