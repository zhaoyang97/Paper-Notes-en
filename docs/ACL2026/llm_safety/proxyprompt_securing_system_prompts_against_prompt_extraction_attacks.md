---
title: >-
  [Paper Note] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks
description: >-
  [ACL 2026][LLM Safety][System Prompt Protection] ProxyPrompt no longer requires the model to "not leak the system prompt." Instead, it replaces the original prompt with a functionally equivalent but semantically obfuscat…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "System Prompt Protection"
  - "Prompt Extraction"
  - "Soft Prompt"
  - "Semantic Leakage Detection"
  - "LLM Security"
date: 2026-05-08
content_hash: 1ce12e101ce6d443
---

# ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks

**Conference**: ACL 2026  
**arXiv**: [2505.11459](https://arxiv.org/abs/2505.11459)  
**Code**: [GitHub](https://github.com/boschresearch/proxyprompt)  
**Area**: LLM Security / Prompt Protection / System Prompt Privacy  
**Keywords**: System Prompt Protection, Prompt Extraction, Soft Prompt, Semantic Leakage Detection, LLM Security

## TL;DR
ProxyPrompt no longer requires the model to "not leak the system prompt." Instead, it replaces the original prompt with a functionally equivalent but semantically obfuscated proxy prompt. While maintaining task utility, it ensures that any extracted content makes it difficult to replicate the original task, achieving a 94.70% protection rate across 264 configurations, significantly higher than filtering and instruction-based defenses.

## Background & Motivation

**Background**: System prompts are core assets for many LLM applications, potentially containing task instructions, screening criteria, business strategies, tool-calling rules, or domain expertise. Compared to fine-tuning, system prompts are low-cost and iterate quickly, leading to their widespread use in GPT Store, HuggingChat assistants, and various applications.

**Limitations of Prior Work**: System prompts are easily induced out by users. Existing defenses generally fall into two categories: prompt-based methods, which ask the model not to leak or provide fake prompts; and filter-based methods, which detect if the output has n-gram overlap with the original prompt. Both categories are unstable: the former relies on the model obeying system instructions, while the latter easily misses semantically equivalent paraphrasing.

**Key Challenge**: Simply preventing output leakage is insufficient. Once the model outputs content semantically equivalent to the original prompt, the attacker can still reuse the task rules. A more fundamental goal is to ensure that "even if the model outputs a certain prompt," the content cannot recover the true semantics and task utility of the original system instruction.

**Goal**: Construct a proxy prompt that maintains the original task performance for normal user requests but presents content that is semantically unrelated to the original prompt and has low utility when extracted.

**Key Insight**: The authors leverage soft prompt / embedding-space optimization to replace the original system prompt with a proxy representation in a continuous space. Under normal input, the proxy produces similar answers to the original prompt; in extraction scenarios, the proxy is optimized to lean towards an unrelated target semantic.

**Core Idea**: Transform prompt protection from "output filtering" to "semantic obfuscation of the protected object itself," using semantic-level metrics to detect paraphrase-based leakage.

## Method

### Overall Architecture
ProxyPrompt assumes that the defender has access to the original system prompt and the model's embedding layer. The defender first prepares a set of benign queries representing normal usage and distills the functionality of the original prompt into a proxy prompt embedding. Simultaneously, the optimization goal requires the model to output a fixed target semantically different from the original prompt when requested to reveal system instructions. During deployment, the original system prompt is no longer placed directly in the context but replaced by the proxy prompt embedding.

### Key Designs

1.  **Joint Objective for Functionality Maintenance and Extraction Prevention**:
    *   **Function**: Ensure the proxy prompt remains useful for normal users while reducing the semantic value of extracted content.
    *   **Mechanism**: The first term minimizes the difference in answers between the original prompt and the proxy prompt on representative queries; the second term forces the proxy away from the original prompt's semantics and toward a defender-specified unrelated target semantic under extraction-type requests. Both terms are integrated into a single embedding-space optimization problem.
    *   **Design Motivation**: Pure soft prompts only optimize for task utility and may still leak task intent. By adding an extraction prevention term, the proxy is not just a functional approximation but also semantically deviates when extracted.

2.  **Continuous Space Proxy and Discrete Decoding Loss**:
    *   **Function**: Leverage the gap between continuous representation and discrete tokens to enhance protection.
    *   **Mechanism**: The proxy prompt is a continuous embedding that does not necessarily fall on the natural language token manifold. When the model attempts to "speak it out," it must pass through a lossy continuous-to-discrete mapping, which typically prevents the extracted text from retaining the original functionality.
    *   **Design Motivation**: Traditional prompts are readable text that can be migrated once repeated. Continuous prompts may lose significant task structure even if they generate some neighboring tokens.

3.  **Semantic-Level Leakage Metrics**:
    *   **Function**: Detect semantic paraphrase leakage that word-level overlap cannot see.
    *   **Mechanism**: In addition to Exact-Match and Approx-Match, the paper proposes Semantic-Match and Most-Similar to compare the semantic equivalence of the original prompt and extracted content at the sentence level. Semantic-Match focuses on whether a semantically substitutable sentence exists, while Most-Similar measures the similarity of the closest segment.
    *   **Design Motivation**: Security evaluation cannot only rely on string overlap. A paraphrase without shared n-grams can still leak actual task rules.

### Loss & Training
The training objective consists of two parts: the answer preservation loss on normal queries and the convergence loss toward an unrelated target semantic in extraction scenarios. In experiments, 100 representative queries are used for each victim-task configuration, and a validation split is kept to select the proxy with the lowest validation loss. This description maintains an academic high-level mechanism and does not involve executable attacks or bypass procedures.

## Key Experimental Results

### Main Results
Experiments cover three victim LLMs (Phi-3.5-mini-instruct, Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct) and five task types (GSM8K, Roles, CoLA, SST-2, QNLI), totaling 264 system prompt configurations.

| Defense Method | Task Utility (UR) | Approx-Match (AM) | Semantic-Match (SM) | Protection Conclusion |
|----------------|-------------------|-------------------|---------------------|-----------------------|
| No Defense     | UR ~ 1.00         | Most near 1.00    | Most near 1.00      | Original prompt easily replicated semantically |
| FILTER         | Utility drops significantly in some tasks | Still high leakage | 42.80% Protection Rate | String filtering harms utility and is unstable |
| FAKE/DIRECT/GUARD | UR mostly ~ 1.00 | Most still near 1.00 | Most still near 1.00 | Limited protection; relies on obedience or detectors |
| Ours†          | UR near 1.00      | AM mostly 0       | 81.06% Protection Rate | Single-target proxy is helpful but inferior to complete target |
| ProxyPrompt    | UR mostly 0.94-1.01 | AM=0 for all tasks | Only 14 SM leaks in 264 configs | 94.70% prompt protection |

### Ablation Study

| Analysis Item | Key Metric | Explanation |
|---------------|------------|-------------|
| L-70B + ProxyPrompt | SM=0 for GSM8K/Roles/CoLA/QNLI; SM=0.25 for SST-2 | Maintains low semantic leakage on large models |
| L-8B + ProxyPrompt | SM=0 for GSM8K/Roles; SM=0.10 for SST-2 | Classification tasks may still leak high-level intent |
| P-3.8B + ProxyPrompt| SM=0 for GSM8K/Roles/CoLA/QNLI; SM=0.25 for SST-2 | Small models also benefit |
| Continuous-to-discrete gap | Average cosine similarity to nearest token: ~0.11-0.12 | Proxy is far from natural token manifold; extracted utility drops |
| HuggingChat Case | UR 1.00, AM 0, SM 0, MS 0.45 | Protects sensitive instructions in real assistant-style prompts |

### Key Findings
- ProxyPrompt achieves AM=0 across all tasks and models, but more importantly, SM is significantly reduced, indicating it does not just avoid string matching.
- Successful leakage occurs mainly in classification tasks and consists mostly of high-level task intent rather than detailed system rules; these are semantics that may necessarily be retained to maintain task utility.
- Filter severely sacrifices utility on some tasks, showing that output-layer filtering struggles to balance security and usability.
- Using fewer representative queries can still maintain low AM and SM; increasing the number of queries primarily improves UR stability.
- ProxyPrompt can be concatenated with non-sensitive prompts, allowing protection of only sensitive parts while retaining the flexibility to extend system functionality.

## Highlights & Insights
- The direction of the paper is clever: instead of trying to make the model keep a secret forever, it transforms the secret itself into an unreadable, hard-to-reuse continuous representation.
- Semantic-level leakage metrics are a necessary supplement. If security evaluations only use n-grams, they overestimate filter-based defenses and are insensitive to paraphrase-based leakage.
- The combination of a 94.70% protection rate and UR near 1.0 indicates that embedding-space prompt protection has practical potential on open-source models.
- The paper clearly distinguishes between "leaking the original text" and "leaking reusable task functionality." The latter is closer to the true risk of system prompts as intellectual property assets.

## Limitations & Future Work
- ProxyPrompt requires access to the model's internal embeddings, so it cannot be directly used by regular application developers for closed-source models that only provide APIs, unless the model provider offers a similar interface.
- The set of representative queries affects utility maintenance; if the distribution of normal usage deviates significantly, the proxy may need re-optimization.
- High-level task intent must sometimes be retained to maintain utility, so limited semantic leakage may still occur in classification tasks.
- This method is not a formal security proof; adaptive adversaries, model updates, and more complex system prompt combinations still require ongoing evaluation.
- Soft prompts have weak interpretability, making debugging and auditing more difficult than natural language prompts.

## Related Work & Insights
- **vs prompt-based defense**: Telling the model not to leak is fragile. ProxyPrompt does not rely on the model continuing to obey natural language prohibitions under adversarial inputs.
- **vs filter-based defense**: Filters focus on output. ProxyPrompt protects the system prompt representation itself; meanwhile, semantic metrics can identify paraphrase leakage that filters miss.
- **vs soft prompt tuning**: Traditional soft prompt tuning pursues task performance. ProxyPrompt adds a semantic obfuscation goal for extraction scenarios, repurposing soft prompts for security protection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using proxy soft prompts to protect system prompts is a very novel perspective that captures the asset nature of system prompts.
- Experimental Thoroughness: ⭐⭐⭐⭐ 264 configurations, multiple models and tasks, plus real assistant and ALFWorld cases; however, experiments for closed-source API scenarios are still lacking.
- Writing Quality: ⭐⭐⭐⭐ Threat models, metrics, and experimental organization are clear, and security boundaries are well-explained.
- Value: ⭐⭐⭐⭐ Highly valuable for open-source models and platform-level prompt protection, though implementation depends on internal model access.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ACL 2026\] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs](parasite_conditional_system_prompt_poisoning_to_hijack_llms.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)

</div>

<!-- RELATED:END -->
