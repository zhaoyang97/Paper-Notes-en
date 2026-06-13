---
title: >-
  [Paper Note] PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts
description: >-
  [ICML 2026][LLM Agent][prompt obfuscation] PragLocker employs a two-stage strategy—"code-symbol initialization + noise injection under black-box feedback from the target model"—to encode agent system prompts into obfusca…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "prompt obfuscation"
  - "agent IP"
  - "non-portability"
  - "black-box optimization"
  - "random search"
date: 2026-05-08
content_hash: 13f2d924dd915a1f
---

# PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts

**Conference**: ICML 2026  
**arXiv**: [2605.05974](https://arxiv.org/abs/2605.05974)  
**Code**: None  
**Area**: Agent Security / Prompt Protection / LLM IP Defense  
**Keywords**: prompt obfuscation, agent IP, non-portability, black-box optimization, random search

## TL;DR
PragLocker employs a two-stage strategy—"code-symbol initialization + noise injection under black-box feedback from the target model"—to encode agent system prompts into obfuscated text. This text maintains utility only on the target LLM and fails on any other LLM, preventing attackers from reusing stolen prompts on their own models.

## Background & Motivation
**Background**: The core IP of commercial LLM Agents (e.g., Cursor, Manus, Zapier) lies in the system prompt. Even when using the same GPT-4o, different prompt designs create entirely different product experiences. Thus, prompts are high-value assets iteratively refined by experts.

**Limitations of Prior Work**: Agents are often deployed on user devices, third-party clouds, or multi-tenant infrastructure. Malicious end-users or cloud insiders can dump the prompt and replicate or even surpass the original Agent on more powerful LLMs. Existing solutions—prompt watermarking (post-hoc verification), encryption (must be decrypted to plaintext via API at runtime), emoji obfuscation (decodable by other LLMs), and representation-space obfuscation like Pape (requires white-box access)—fail to simultaneously satisfy the requirements of proactivity, runtime efficiency, usability, and non-portability.

**Key Challenge**: Constructing a prompt that maintains utility on a target LLM while failing on others fundamentally requires the prompt to preserve original semantics while over-fitting to the specific geometry of the target model's loss landscape, all while being restricted to API-level input-output and log-prob feedback.

**Goal**: (1) Formalize the four requirements for prompt protection (C1-C4); (2) provide an existence proof to guarantee theoretical feasibility; (3) design a black-box, API-only optimization algorithm; and (4) verify portability loss and utility maintenance across multiple models, agents, and tasks.

**Key Insight**: The authors leverage the "attention dilution" property of transformers—networks are insensitive to perturbations of certain tokens. Theoretically, an $\epsilon$-sphere stability region $S_{\bm{x}}$ exists where utility remains unchanged. Simultaneously, the geometry of stability regions differs across models, making it unlikely for target-specific perturbations to fall within the regions of other models.

**Core Idea**: Obfuscation is treated as "gradient-free discrete optimization over a target-LLM-specific loss landscape," using random search to optimize a joint loss involving utility, obfuscation, and non-portability.

## Method
PragLocker decomposes the abstract concept of "non-portable obfuscation" into a formal existence problem and an engineered two-stage pipeline.

### Overall Architecture
The input consists of a plaintext prompt $\bm{x}$, the target LLM API, and a task dataset $\mathcal{D}$. In Stage 1 (Initialization Transformation), the target LLM encodes $\bm{x}$ into a "code-symbol form" $\tilde{\bm{x}}_0$ that preserves semantics but is no longer natural language. In Stage 2 (Noise-Injected Obfuscation Optimization), random search is used to iteratively inject character-level noise. Each step is determined by a joint loss covering task performance, obfuscation distance, and non-language distribution. The final obfuscated prompt $\tilde{\bm{x}}$ is deployed in untrusted environments and used directly at runtime without deobfuscation.

### Key Designs

1.  **Theoretical Motivation: Functional Equivalence + Stability Region**:
    *   **Function**: Functional equivalence is defined such that embeddings $\tilde{\bm{h}}$ and $\bm{h}$ under query $\bm{q}_i$ are equivalent if they produce the same greedy decoding. By defining a correct-class margin $m(\tilde{\bm{h}}, \bm{q}_i, y_i) = f(\tilde{\bm{h}}, \bm{q}_i)_{y_i} - \max_{k \neq y_i} f(\tilde{\bm{h}}, \bm{q}_i)_k$, any point in the $\epsilon$-sphere $B_\epsilon(\bm{h})$ maintains utility as long as the margin $> 0$.
    *   **Mechanism**: The authors prove a theorem for the "Existence of obfuscated prompts"—by perturbing $k$ low-attention tokens, the cumulative embedding shift $\|\Delta\bm{h}\| \le \sum_{j} \|\bm{\delta}_j\|$ can be kept within $\epsilon$ to maintain utility, while the discrete distance $d(\tilde{\bm{x}}, \bm{x})$ grows large enough for obfuscation. Non-portability arises from "manifold mismatch," where $S_{\bm{x}}(\theta)$ and $S_{\bm{x}}(\theta')$ are nearly disjoint in high-dimensional space.
    *   **Design Motivation**: To move beyond ad-hoc claims, the existence proof via attention dilution and high-dimensional geometry provides a theoretical foundation.

2.  **Stage 1: Code-Symbol Initialization**:
    *   **Function**: The target LLM translates the original prompt into a code-symbol format as a warm start $\tilde{\bm{x}}_0$.
    *   **Mechanism**: Using the target LLM itself to generate a symbolic version acts as "target-conditioned" preliminary obfuscation. This representation introduces redundancy, creating space for subsequent noise injection.
    *   **Design Motivation**: Starting from scratch is unlikely to hit the stability region; using the LLM for semantic-preserving transformation sets the search starting point inside the manifold.

3.  **Stage 2: Random-Search Noise Injection + Joint Loss**:
    *   **Function**: At each step, character noise $\bm{n}_t$ is sampled and injected in-place. The candidate $\tilde{\bm{x}}'_{t+1}$ is accepted or rejected based on the joint loss $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{dist}} + \gamma \mathcal{L}_{\text{non-lang}}$.
    *   **Mechanism**: $\mathcal{L}_{\text{task}} = -\log p(\bm{y}|\bm{q}, \tilde{\bm{x}})$ preserves utility; $\mathcal{L}_{\text{dist}} = -\log \sigma(\mathrm{Dist}(\tilde{\bm{x}}, \bm{x}))$ uses Levenshtein distance to push the prompt away from the original; $\mathcal{L}_{\text{non-lang}} = -H(\tilde{\bm{x}})$ minimizes character Shannon entropy to deviate from natural language.
    *   **Design Motivation**: In a black-box setting, random search replaces gradients. The non-language term is a key trick; by diverging from natural language distributions (which are naturally portable), portability is broken.

### Loss & Training
The process follows a greedy random search (Algorithm 1) using mini-batches $(\bm{q}_t, \bm{y}_t)$. It requires only log-probs and text outputs from the target LLM API, with no white-box access needed.

## Key Experimental Results

### Main Results
Portability is measured by optimizing a prompt for a target LLM and then deploying it on other LLMs to observe task performance (e.g., LessonL agent on HumanEval/MBPP):

| Agent / Task | Target LLM | Original → GPT-4o | Original → Gemini2 | PragLocker → GPT-4o | PragLocker → Gemini2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LessonL / HumanEval | Gemini 2 | 93.90 | - | **0.61** | - |
| LessonL / HumanEval | DeepSeek | 93.90 | 98.78 | **0.61** | **2.44** |
| LessonL / MBPP | Gemini 2 | 91.89 | - | **0.51** | - |
| LessonL / MBPP | DeepSeek | 91.89 | 97.33 | **0.62** | **~0** |

PragLocker successfully reduces cross-model performance to near zero, whereas original prompts typically maintain over 90% performance.

### Ablation Study
The study compares PragLocker with variants: *PragLocker_tune* (task loss only) and *PragLocker_code* (initialization only).

| Configuration | Portability (Cross-model Perf) | Description |
| :--- | :--- | :--- |
| Original Prompt | 93-99 | Fully portable |
| PragLocker_code | 85-95 | Symbolization reduces portability slightly but remains usable |
| PragLocker_tune | 80-93 | Task tuning alone is insufficient to block portability |
| **PragLocker (Full)** | **0-3** | Portability vanishes after joint loss optimization |

### Key Findings
*   The `non-language loss` is the primary driver of non-portability.
*   Even between different precisions of the same model (FP16 vs. 4-bit), the obfuscated prompt fails to migrate, supporting the "manifold mismatch" hypothesis.
*   The target LLM itself cannot explain the semantics of the obfuscated prompt, indicating it has evolved into a "model-conditioned trigger" rather than recoverable text.

## Highlights & Insights
*   Framing prompt protection as "gradient-free discrete optimization" is highly effective for black-box environments.
*   The `non-language entropy regularizer` is a valuable insight: deviating from natural language distributions is a robust way to block semantic portability.
*   Using the target LLM's own symbolic transformation as a warm start significantly narrows the search space for random search.

## Limitations & Future Work
*   Assumes attackers cannot farm large query-output pairs to train their own deobfuscator.
*   Weighting for $\lambda, \gamma$ requires manual tuning per task.
*   Random search convergence is slow on long prompts; alternative optimizers (e.g., evolution strategies) are not compared.
*   The trade-off regarding increased token counts and latency is not fully quantified.

## Related Work & Insights
*   **vs. Prompt Watermarking**: Watermarking is reactive (detection only); PragLocker is proactive (prevents unauthorized reuse).
*   **vs. Encryption**: Encryption only protects data at rest; PragLocker protects the prompt during runtime execution.
*   **vs. Representation Obfuscation**: Approaches like Pape require white-box access, whereas PragLocker is strictly black-box.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)
- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)
- [\[ICML 2026\] A Minimal Agent for Automated Theorem Proving](a_minimal_agent_for_automated_theorem_proving.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)

</div>

<!-- RELATED:END -->
