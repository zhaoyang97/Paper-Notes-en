---
title: >-
  [Paper Note] Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets
description: >-
  [ACL2026][LLM Safety][Code Data Protection] This paper proposes FunPoison, which injects inert weak-use fragments into real execution paths while ensuring Java code remains compilable, executable…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Code Data Protection"
  - "Data Poisoning"
  - "Functionality Preserving"
  - "CodeLLM"
  - "Unauthorized Fine-tuning"
date: 2026-05-08
content_hash: 12a80152d1647ece
---

# Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets

**Conference**: ACL2026  
**arXiv**: [2604.22291](https://arxiv.org/abs/2604.22291)  
**Code**: To be confirmed (The paper claims open-source, but the local cache does not provide the repository URL)  
**Area**: Code Large Language Models / Data Governance  
**Keywords**: Code Data Protection, Data Poisoning, Functionality Preserving, CodeLLM, Unauthorized Fine-tuning

## TL;DR
This paper proposes FunPoison, which injects inert weak-use fragments into real execution paths while ensuring Java code remains compilable, executable, and functionally equivalent. Poisoning only 10% of the data significantly weakens the gains of unauthorized CodeLLM fine-tuning and demonstrates strong robustness against formatting, rewriting, static analysis, and detection-based cleaning.

## Background & Motivation
**Background**: The capabilities of CodeLLMs largely stem from large-scale public code datasets, such as CodeSearchNet and The Stack. Many data authors have not authorized model training, but once data is scraped and used for fine-tuning, ex-post accountability, copyright litigation, or watermark attribution often involve high costs, long cycles, and unstable results.

**Limitations of Prior Work**: Data poisoning can serve as proactive protection to prevent unauthorized training from yielding benefits. However, code data has specific requirements: ordinary users still need to compile, run, test, and integrate this code. Existing methods like CoProtector either damage syntax or semantics, leading to near-total failure in compilability, or only modify comments, resulting in weak poisoning effects that often require 100% poisoning to be noticeable.

**Key Challenge**: Protective poisoning must make model training "learn poorly" without making human usage "work poorly." This requires poisoning fragments to generate distributional interference in the training token sequence while not altering observable program behavior, all while evading common cleaning and static analysis.

**Goal**: The authors aim to construct a functionality-preserving poisoning framework that suppresses unauthorized fine-tuning in a realistic partial poisoning setting, while ensuring that normal code quality, compilation success rates, and runtime behavior remain unaffected.

**Key Insight**: FunPoison does not insert dead code or obviously "bad" code. Instead, it places short, compilable, side-effect-free template fragments into the execution path and uses type-aware weak-use statements to ensure these fragments remain after static analysis and formatting. The key assumption is: fragments are inert at runtime but are not inert during autoregressive training because the model still must learn these tokens.

**Core Idea**: Transform "program-harmless" code fragments into "unauthorized-model-harmful" training signals, allowing functionality preservation and training disruption to coexist.

## Method

### Overall Architecture
The threat model of FunPoison assumes that data owners publish code for normal use but do not authorize large-scale training; attackers can collect this data, control the training process, and use methods like cleaning, formatting, static analysis, LLM rewriting, or supervised detection. The defense goal is not to make poisoning undetectable, but to make it difficult for attackers to clear enough poisoning signals under reasonable false positive rates, semantic preservation, and cost constraints, thereby preventing unauthorized fine-tuning from significantly exceeding the base model.

The method consists of two main stages. The first stage constructs a template pool: statement-level fragments are extracted from real code, processed through compilation repair, minimal context completion, type parsing, variable anonymization, and conflict metadata recording to retain portable templates. The second stage involves controlled injection: a subset of data is selected based on the poisoning ratio, templates undergo safety filtering, safe execution positions are identified within the host code, type-aware weak-uses are synthesized, and injection occurs after resolving naming conflicts. The resulting code remains compilable and behaviorally equivalent but exposes additional structural patterns to the model during training.

### Key Designs
1. **Compile-Driven Code Template Generation**:

	- **Function**: Obtain reusable, compilable, and naturally styled short code fragments from real code corpora to serve as injection material.
	- **Mechanism**: Systematically extracts statement-level fragments from method bodies rather than complete functions or type declarations. For incomplete fragments, RepairToCompile adds the minimum necessary context, such as importing standard library types, constructing lightweight stubs for non-JDK types, or rewriting isolated object constructions into forms with variable receivers, and validates them via Javac. Variables, method names, class names, and placeholder information are then recorded for handling naming conflicts during injection.
	- **Design Motivation**: Directly copying real code fragments often results in compilation failure due to context dependence; excessive normalization makes patterns repetitive and easy to detect. Compilation repair allows templates to maintain real code style while possessing cross-project portability.

2. **Functionality-Preserving Controlled Injection**:

	- **Function**: Insert templates into the host program without altering control flow, output, exceptions, I/O, or global state.
	- **Mechanism**: FunPoison uses two layers of safety filters to exclude risky templates: conceptually excluding patterns like control flow disruption, reflection dependence, and shared state; and at the program level, excluding side effects like I/O, concurrency, process control, container mutation, and non-local assignments. Injection positions are chosen only at syntactically stable and semantically inert locations within method bodies, avoiding return, throw, break, continue, boundary positions, and vicinity of observable side effects. Scope tracking and variable renaming are also performed during injection.
	- **Design Motivation**: Code data protection should not sacrifice the experience of normal developers. Only when compilation and runtime behavior are stable can poisoning be discussed as a viable data governance mechanism.

3. **Type-Aware Weak-use and Execution Path Supervision**:

	- **Function**: Ensure injected fragments are visible during training but are not easily deleted as dead code by compilers, formatters, or static analysis.
	- **Mechanism**: The system synthesizes weak-use statements based on variable types, performing inert consumption like identity, metadata, or security queries, avoiding changes in I/O, concurrency, and global state. These fragments are placed in the real execution path rather than in always-false branches. Mechanism analysis shows that weak-use and template signatures highly co-occur in failed generations; DeadBranchInsertion using the same template pool but placed in dead branches yields almost no performance degradation compared to FunPoison.
	- **Design Motivation**: The poisoning effect comes from the autoregressive model's learning interference with token distributions in the execution path, not simply from "seeing certain template text." This is the key distinction between FunPoison and dead code insertion or comment perturbation.

### Loss & Training
FunPoison itself does not involve training a model; thus, there is no new optimization loss. In experiments, attackers fine-tune CodeLLMs like DeepSeek-Coder, StarCoderBase, and CodeLlama, evaluating whether the poisoned dataset prevents the fine-tuned model from gaining benefits over the base model on HumanEval-X and MBPP. The primary metric is $Delta Pass@k$, the difference in Pass@k between the fine-tuned model and the base model; if clean fine-tuning provides significant improvement while FunPoison fine-tuning results in dissipated or negative gains, the defense is considered effective.

## Key Experimental Results

### Main Results

| Setting | Metric | Base | Clean FT | FunPoison | Conclusion |
|--------|------|------|------|------|------|
| DeepSeek-Coder-1.3B / HumanEval-X | Pass@1, T=0.0 | 0.31 | 0.38 | 0.20 (10% poisoning) | 10% poisoning turns FT gain into significant degradation |
| CodeLlama-7B / HumanEval-X | Pass@1, T=0.0 | 0.29 | 0.31 | 0.23 (10% poisoning) | Gains still suppressed at 7B scale |
| CodeLlama-7B-Instruct / HumanEval-X | Pass@1, T=0.0 | 0.30 | 0.38 | 0.30 (10% poisoning) | Clean FT gains essentially neutralized on instruct models |
| DeepSeek-1.3B / MBPP | Pass@1, T=0.0 | 0.31 | 0.41 | 0.16 (10% poisoning) | Strong degradation persists across benchmarks |

### Ablation Study

| Configuration | Poisoning Rate | Pass@1 | Description |
|------|------|------|------|
| Base | - | 0.31 | Unfrozen model |
| Clean fine-tuned | 0% | 0.38 | Normal fine-tuning yields gains |
| FunPoison | 10% | 0.20 | Weak-use fragments in execution path lead to significant degradation |
| DeadBranchInsertion | 1% | 0.37 | Same templates in dead branches; nearly equivalent to clean FT |
| DeadBranchInsertion | 10% | 0.38 | Indicates template exposure alone is not critical |
| DeadBranchInsertion | 50% | 0.34 | High ratio still much weaker than FunPoison |
| DeadBranchInsertion | 100% | 0.35 | Full dead branch insertion cannot replicate FunPoison effect |

| Functionality Preservation Metrics | Clean | FunPoison | Interpretation |
|------|------|------|------|
| Compilation success | 984/984 | 984/984 | Compilation success rate maintained at 100% |
| p95 time overhead | Baseline | mean 2.29%, p95 25% | Average time overhead is small |
| p95 memory overhead | Baseline | mean 0.09%, p95 2.41% | Memory impact is very low |
| Line coverage | 100% | 100% | Execution coverage remains unchanged |
| Execution jitter | 8.17% | 8.12% | Stability remains essentially consistent |
| Behavior consistency | Preserved | Preserved | Output, exceptions, and I/O behavior remain identical |

| Defense/Cleaning Method | Key Results | Implications for FunPoison |
|------|------|------|
| LLM rewriting / CodeLlama | ACC 0.07, CodeBLEU 0.70, avg 76.42s | Low rewriting success rate and high cost |
| LLM rewriting / GPT-4 | ACC 0.06, CodeBLEU 0.56, avg 70.07s | Even stronger models struggle to reliably clear it |
| CodeQL static analysis | Similar to clean, Rule 32: 4.3% | Standard rules cannot isolate poisoned samples |
| CodeBERT adaptive detector | FPR 100%, Accuracy 10.39% | Supervised detection tends to excessively misclassify benign code |
| clang-format | Remains lower than clean FT and base | Simply changing layout cannot remove training signals |

### Key Findings
- The most critical empirical result of FunPoison is its effectiveness in partial poisoning. While CoProtector's disruptive transformations usually only become evident at 100%, FunPoison significantly suppresses fine-tuning gains at 10%.
- The DeadBranchInsertion ablation is very convincing: same templates placed outside the execution path result in performance close to clean fine-tuning, indicating that training interference comes from execution path supervision rather than template text itself.
- Evidence for functionality preservation is comprehensive: all 984 tasks compiled and ran, and all 57,764 unit tests for Apache Commons Lang passed, showing the method does not succeed by destroying code quality.
- Robustness experiments cover detection, cleaning, LLM rewriting, static analysis, formatting, and adaptive supervised detection. While not proving it is unremovable, they show common low-cost cleaning strategies have limited effectiveness.

## Highlights & Insights
- The strongest point of the paper is optimizing the often-conflicting goals of "poisoning effectiveness" and "code usability" together. For code data governance, maintaining the normal user experience is a prerequisite for a method's adoption.
- The mechanism explanation regarding execution path supervision is important. Fragments are harmless at runtime but remain under token supervision during autoregressive training, thus affecting the code distribution learned by the model—a more nuanced observation than simply inserting "bad" code.
- The experiments look beyond Pass@1, incorporating dynamic analysis, real-world project testing, rewriting attacks, static analysis, and adaptive detector evaluations, making the paper resemble a complete defense system assessment.
- FunPoison also raises a broader question: if data owners want to allow human use but restrict model training, technical mechanisms, licensing governance, and transparent disclosure must be designed in tandem.

## Limitations & Future Work
- The paper systematically evaluates only Java; other languages require different parsers, compilers, weak-use designs, and side-effect filtering rules. Portability to Rust, Go, JavaScript, or C/C++ cannot yet be directly assumed.
- FunPoison depends on available insertion points. The paper notes that 80.3% of functions in CodeSearchNet Java have valid positions under full coverage settings, but highly compact or heavily optimized code may lack space.
- It is not theoretically unremovable. More aggressive training pipelines, pre-training from scratch, data deduplication, semantic normalization, strong human auditing, or RL-based adaptation might alter the effects.
- The method is clearly dual-use. Responsible deployment requires transparent disclosure in dataset cards, READMEs, or license addendums, and providing clean data for authorized training users; it is unsuitable for default deployment in open collaborative ecosystems.
- Current evaluations mainly focus on executable code generation. Whether code retrieval, completion, repair, test generation, or code understanding tasks are similarly affected requires task-level research.

## Related Work & Insights
- **vs CoProtector**: CoProtector disrupts code or comments through CC/CS/CR/CSR transformations, often sacrificing compilability or relying on 100% poisoning; FunPoison treats functionality preservation as a hard constraint and is effective at 10% partial poisoning.
- **vs Code Watermarking / Attribution**: Watermarking and attribution are typically used to prove data was used ex-post; FunPoison aims to reduce unauthorized fine-tuning gains ex-ante. The two are complementary.
- **vs Backdoor Attacks / General Data Poisoning**: Many poisoning studies pursue targeted misbehavior or label corruption; this work focuses on untargeted deterrence while requiring program behavior to remain unchanged.
- **vs Cleaning and Rewriting Defenses**: KillBadCode, DeCoMa, CodeQL, formatters, and LLM rewriting are tools attackers might use; this paper treats them as robustness tests rather than direct method comparisons.
- **Insights**: For datasets constrained by copyright and licensing, legal text alone is hard-pressed to stop model training. Future data usage governance may need to combine access control, data provenance, transparent disclosure, and technical perturbation into a finer-grained system.

## Rating
- Novelity: ⭐⭐⭐⭐☆ Functionality-preserving code poisoning and the execution path supervision mechanism are highly distinct from destructive code perturbations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of models, benchmarks, defenses, real projects, and dynamic analysis; ablations capture key mechanisms.
- Writing Quality: ⭐⭐⭐⭐☆ Structure is clear, with a complete description of the threat model and responsibility boundaries; some code link information in the cache is unclear.
- Value: ⭐⭐⭐⭐☆ Insightful for code data governance, protection against unauthorized fine-tuning, and CodeLLM training security, though deployment must be highly cautious.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](../../NeurIPS2025/llm_safety/imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)
- [\[ACL 2026\] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs](parasite_conditional_system_prompt_poisoning_to_hijack_llms.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](../../ICML2026/llm_safety/optimizing_token_choice_for_code_watermarking_an_rl_approach.md)

</div>

<!-- RELATED:END -->
