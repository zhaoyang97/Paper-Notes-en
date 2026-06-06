---
title: >-
  [Paper Note] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding
description: >-
  [ACL2026][Code Intelligence][Program Slicing] Sliceformer reformulates static program slicing as a seq2seq task for small code language models. It learns dependency relationships through dataflow-aware pretraining and em…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Program Slicing"
  - "Dataflow-Aware Pretraining"
  - "Constrained Decoding"
  - "CodeT5+"
  - "Static Analysis"
date: 2026-05-08
content_hash: 3ce029d612ed8e73
---

# Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding

**Conference**: ACL2026  
**arXiv**: [2604.26961](https://arxiv.org/abs/2604.26961)  
**Code**: https://anonymous.4open.science/r/staticsliceT5-4E22  
**Area**: Code Intelligence / Program Analysis  
**Keywords**: Program Slicing, Dataflow-Aware Pretraining, Constrained Decoding, CodeT5+, Static Analysis

## TL;DR
Sliceformer reformulates static program slicing as a seq2seq task for small code language models. It learns dependency relationships through dataflow-aware pretraining and employs lexical and syntactic constrained decoding to prevent hallucinations, significantly improving ExactMatch on Java and Python slicing benchmarks.

## Background & Motivation
**Background**: Static program slicing, which aims to identify code fragments related to a specific variable or statement, is a fundamental technique in debugging, vulnerability analysis, and program comprehension. Traditional methods rely on System Dependence Graphs (SDG) and graph reachability analysis, which are precise but engineering-intensive. Recently, learning-based approaches have attempted to use CodeBERT, GraphCodeBERT, or Large Language Models (LLMs) to automatically predict slices.

**Limitations of Prior Work**: Learning-based slicing faces two core issues. First, language models are prone to misjudging dependencies based on surface similarity or positional proximity, missing truly relevant statements or including irrelevant ones. Second, generative models may output tokens, variable names, or statements that do not exist in the original program, whereas a program slice must strictly be an exact subsequence of the input code.

**Key Challenge**: Code language models excel at generating natural code but do not naturally adhere to the hard constraints of program analysis tasks. Program slicing requires both an understanding of dataflow semantics and the output of a fully extractive, hallucination-free, and structurally valid result. Simple supervised fine-tuning (SFT) struggles to satisfy both simultaneously.

**Goal**: The authors aim to retain the advantages of language models in end-to-end modeling of full function contexts while explicitly injecting dataflow constraints from static analysis into the pretraining and decoding processes to enhance slicing accuracy and reliability.

**Key Insight**: The paper selects small encoder-decoder models like CodeT5+ rather than relying on large closed-source LLMs. The approach is divided into dataflow capability shaping during pretraining and hard-constrained decoding during inference, addressing dependency identification and hallucination suppression respectively.

**Core Idea**: Use Data Flow Graphs (DFG) to design pretraining tasks that teach the model "which statements are truly related," then use constrained decoding—allowing only input tokens and maintaining monotonic increases in AST similarity—to ensure the output is a valid slice.

## Method

### Overall Architecture
The input to Sliceformer consists of a sequence of function statements, the slicing variable, and the line number of the variable. The output is a backward slice ordered according to the original code. The model first undergoes dataflow-aware pretraining on Python/Java functions from CodeSearchNet, followed by supervised fine-tuning on labeled slicing data from CodeNet-Slice. Finally, it uses lexical and syntactic constraints during the decoding stage to filter out invalid candidates.

The key to this pipeline is decoupling the two task properties of program slicing: pretraining enhances "semantic dependency identification," while constrained decoding ensures "element preservation." The former informs the model where variable values originate and flow; the latter ensures the generative process cannot create tokens outside the original program or assemble fragments that deviate structurally from the input code.

### Key Designs
1.  **Dataflow-Preserving Statement Permutation Pretraining**:
    - **Function**: Teaches the model which statements have no data dependencies and which orders are semantically interchangeable.
    - **Mechanism**: Given the code and its DFG, identify pairs of statements within the same basic block that are not connected by data edges. Randomly swap them and train the model to generate dataflow-equivalent code variants. Unlike random sentence reshuffling in natural language, code has multiple legal orders that must respect def-use relationships.
    - **Design Motivation**: Slicing requires determining if a statement truly affects the slicing criterion. By learning "which statements are swappable," the model is forced to focus on data dependencies rather than just memorizing original positions.

2.  **Dataflow-Aware Span Corruption**:
    - **Function**: Enhances the model's ability to reconstruct dependency chains at the variable and statement levels.
    - **Mechanism**: Randomly select variable nodes in the DFG and identify their parents and children (sources and sinks of values). Masking is then applied at two granularities: masking the variables themselves or masking the statements containing them. The model must recover the masked def-use fragments based on context.
    - **Design Motivation**: Standard span corruption primarily learns local language patterns, and AST masking emphasizes syntax. Since program slicing relies most heavily on cross-statement dataflows, masking along the DFG is more aligned with the task's essence.

3.  **Lexical-Syntactic Constrained Beam Search**:
    - **Function**: Prevents the model from generating identifiers, expressions, or structurally incorrect statements not present in the original program.
    - **Mechanism**: At each decoding step, the vocabulary is restricted to tokens appearing in the input code. When a statement boundary is reached, the Tree Similarity Edit Distance (TSED) between the current partial slice and the input code's AST is calculated. If the TSED does not increase monotonically, the beam is considered structurally erroneous and is pruned.
    - **Design Motivation**: A valid slice should be a subsequence of the input code; thus, as statements are added, the generated fragment should increasingly resemble a structural subset of the input. Lexical constraints solve token hallucinations, while TSED monotonicity addresses cases where tokens are valid but their order or structure is not.

### Loss & Training
The pretraining phase is based on CodeT5+ 0.7B, using approximately 1.0M functions from the Python and Java subsets of CodeSearchNet. For each function, Tree-Sitter is used to extract the AST and variables, and the DFG is constructed in the style of GraphCodeBERT. Span corruption masks $25\%$ of tokens, statement permutation swaps a maximum of 3 statements per sample, the context length is 512, the batch size is 32, and training spans 100K steps.

The supervised fine-tuning phase uses the Python and Java subsets of CodeNet-Slice. Both input and output lengths are 512. The model is optimized using AdamW with a batch size of 32, a learning rate of 5e-5, and 1000 warmup steps over 10 epochs. The output format includes special control tokens for line numbers, code, criteria, and slices to help the model generate structured slices.

## Key Experimental Results

### Main Results
Sliceformer outperforms baselines across four metrics in both Java and Python, with particularly significant improvements in ExactMatch.

| Method | Java Acc-D | Java ExactMatch | Java CodeBLEU | Java TSED | Python Acc-D | Python ExactMatch | Python CodeBLEU | Python TSED |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5 + CoT | 60.27 | 14.00 | 71.35 | 63.81 | 56.94 | 13.00 | 68.56 | 61.27 |
| NS-slicer CodeBERT | 95.65 | 81.72 | 88.41 | 91.00 | 82.47 | 56.32 | 74.68 | 78.91 |
| NS-slicer GraphBERT | 96.51 | 85.77 | 89.26 | 90.35 | 84.92 | 61.25 | 76.84 | 80.12 |
| CodeT5+ SFT | 95.33 | 87.24 | 89.26 | 93.42 | 87.53 | 77.24 | 79.98 | 81.75 |
| Sliceformer | 98.78 | 92.20 | 93.23 | 97.68 | 90.85 | 83.15 | 85.35 | 89.74 |

Compared to the strongest previous baseline, NS-slicer GraphBERT, Sliceformer’s ExactMatch improved from 85.77 to 92.20 in Java (a 6.4 percentage point increase) and from 61.25 to 83.15 in Python (a 21.9 percentage point increase). Compared to direct SFT of CodeT5+, Java ExactMatch improved by approximately 5.0 percentage points, and Python by approximately 5.9 points.

In terms of efficiency, Sliceformer is an order of magnitude faster than 7B/8B SFT models while adding only minimal overhead compared to standard CodeT5+.

| Method | Model Scale | Single Task Runtime | Remarks |
| :--- | :--- | :--- | :--- |
| NS-slicer CodeBERT | 125M | 0.105s | Fastest but lower accuracy |
| NS-slicer GraphCodeBERT | 125M | 0.135s | Strong previous learning-based baseline |
| CodeT5+ | 770M | 0.289s | Direct SFT |
| Sliceformer | 770M | 0.296s | High accuracy with overhead close to CodeT5+ |
| CodeLlama-7B SFT | 7B | 5.75s | Significantly slower |
| Qwen3-8B SFT | 8B | 6.52s | Significantly slower |

### Ablation Study
The ablation results indicate that all four components are effective, with dataflow span corruption and lexical constraints having the largest impact. The appendix also provides a comparison of constrained decoding compatibility across different architectures.

| Configuration | Architecture | Pretraining | SFT | Constrained Decoding | Java ExactMatch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CodeT5 | Encoder-Decoder | No | Yes | No | 82.80 |
| CodeT5 + Sliceformer Comp. | Encoder-Decoder | Yes | Yes | Yes | 85.12 |
| CodeLlama-7B | Decoder-only | No | Yes | No | 75.27 |
| Qwen3-8B | Decoder-only | No | Yes | No | 80.55 |
| Qwen3-8B + Constrained Decoding | Decoder-only | No | Yes | Yes | 83.11 |

| Component | Ablation Phenomenon | Explanation |
| :--- | :--- | :--- |
| w/o span corruption | One of the largest performance drops | DFG-guided masking directly trains the model to recover def-use chains, critical for full slice identification. |
| w/o lexical constraints | One of the largest performance drops | Generative models easily produce out-of-input tokens; lexical hard constraints are vital for element preservation. |
| w/o statement permutation | Performance decrease | The model loses the ability to learn the interchangeability of dataflow-independent statements. |
| w/o TSED syntactic constraints | Small but stable performance decrease | Structural errors are less frequent than lexical hallucinations, but these constraints still filter incorrect beams in complex statements. |

### Key Findings
- Even with RAG or CoT, closed-source LLMs achieve very low ExactMatch, indicating that program slicing is a hard-constrained extraction task rather than general code QA.
- NS-slicer performs independent binary classification on each statement, making it difficult to distinguish identical statements in different control flow branches; Sliceformer generates slices within a function-level context, better handling positional and contextual differences.
- Constrained decoding adds almost no latency; Sliceformer's 0.296s is close to CodeT5+'s 0.289s, but with higher ExactMatch.
- Encoder-decoder architectures more naturally support dataflow span reconstruction; decoder-only models still benefit from constrained decoding but cannot directly reuse all pretraining objectives.

## Highlights & Insights
- The paper transforms the "hallucination problem" of code generation models into an element preservation constraint for program slicing and solves it using hard decoding constraints, which is far more reliable than simply prompting the model "not to hallucinate."
- The dataflow pretraining objectives are tailored to program analysis tasks: rather than general code understanding, they focus on learning def-use dependencies, the core of backward slicing.
- TSED monotonicity is an intuitive structural constraint: if the output is a subsequence of the input code, AST similarity should increase as valid statements are added.
- The small model approach is practical. A 770M CodeT5+ with task constraints far outperforms 7B/8B generative models and GPT prompting in precise program analysis tasks, showing that task-specific inductive biases are more critical than model scale.

## Limitations & Future Work
- The experiments only cover Java and Python; extending to C/C++, JavaScript, or Rust would require reapplying parsers, DFG construction, and slicing annotation tools.
- Dataflow pretraining is primarily designed for encoder-decoder models; designing equivalent pretraining objectives for decoder-only large models remains an open question.
- Evaluation depends on CodeNet-Slice and ground truth generated by tools; if the tools themselves contain noise, the model might learn specific annotation biases.
- TSED monotonic constraints target syntax and do not directly guarantee complete semantic dependency; complex control dependencies, exception handling, and cross-function calls may still be challenging.
- Future work could more tightly integrate traditional static analysis graph constraints with neural generation, such as by maintaining reachable dependency subgraphs explicitly during beam search.

## Related Work & Insights
- **vs JavaSlicer / Traditional Static Analysis**: Traditional tools rely on explicit dependency graphs and reachability rules; they are interpretable but costly to adapt to new languages. Sliceformer approximates slicing using a learned model while retaining program analysis biases via DFG constraints.
- **vs NS-slicer**: NS-slicer uses statement-level binary classification, which tends to lose function-level context. Sliceformer outputs full slices in a seq2seq format, handling identical statements and cross-positional dependencies more effectively.
- **vs GPT prompting**: Prompting can generate explanations, but its ExactMatch is poor and it is prone to hallucinations. Sliceformer’s hard constraints are better suited for program analysis tasks requiring precise extraction.
- **vs GraphCodeBERT**: GraphCodeBERT uses dataflow information for representation learning; this work further turns DFG into a generative pretraining objective tailored for slicing, combined with constrained decoding.
- **Insight**: For code intelligence tasks, the most effective LLM solutions are often not about simply scaling up the model, but about embedding the verifiable structure of the task into the training objectives and decoding process.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combined DFG pretraining and TSED monotonic constraints fit the task well; the logic is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results, efficiency, ablation, and architectural appendices are comprehensive, though the number of languages is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definitions and methodology explanations are complete; some algorithm layouts are long but do not hinder understanding.
- Value: ⭐⭐⭐⭐⭐ Highly relevant practical reference for program slicing, constrained decoding in code generation, and neural program analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](../../ICML2026/code_intelligence/locally_coherent_parallel_decoding_in_diffusion_language_models.md)
- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)

</div>

<!-- RELATED:END -->
