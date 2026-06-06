---
title: >-
  [Paper Note] PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts
description: >-
  [ICML 2026][LLM Agent][prompt obfuscation] PragLocker employs a two-stage strategy of "code-symbol initialization + noise injection under black-box target model feedback" to encode the agent system prompt into an obfusca…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "prompt obfuscation"
  - "agent IP"
  - "non-portability"
  - "black-box optimization"
  - "random search"
date: 2026-05-08
content_hash: 317cfa86502cd5ab
---

# PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts

**Conference**: ICML 2026  
**arXiv**: [2605.05974](https://arxiv.org/abs/2605.05974)  
**Code**: None  
**Area**: Agent Security / Prompt Protection / LLM IP Defense  
**Keywords**: prompt obfuscation, agent IP, non-portability, black-box optimization, random search

## TL;DR
PragLocker employs a two-stage strategy of "code-symbol initialization + noise injection under black-box target model feedback" to encode the agent system prompt into an obfuscated text that only works on the target LLM and fails on any other LLM. Thus, even if the prompt is stolen from the deployment side, attackers cannot reuse it on their own LLMs.

## Background & Motivation
**Background**: For commercial LLM Agents such as Cursor, Manus, and Zapier, the core IP lies in the system prompt. Even when using the same GPT-4o, different prompt designs result in entirely different product experiences, making the prompt a high-value asset refined by expert iteration.

**Limitations of Prior Work**: Agents are often deployed on user devices, third-party clouds, or multi-tenant infrastructures, where malicious end users or internal cloud personnel can directly dump the prompt and replicate or even surpass the original Agent on any stronger LLM. Existing solutions—prompt watermarking (post-hoc verification), encryption (requires decryption to plaintext at runtime for API calls), emoji obfuscation (decodable by other LLMs), and representation-space obfuscation (requires white-box access, but GPT/Gemini are black-box)—cannot simultaneously satisfy proactivity, runtime protection, usability, and non-portability.

**Key Challenge**: The essential requirement is to construct a prompt that "retains utility on the target LLM but fails on others," meaning the prompt must preserve the original semantics while overfitting to the target model's unique loss landscape geometry, using only API-level input/output and log-prob feedback.

**Goal**: (1) Formalize the four requirements (C1–C4) for prompt protection; (2) Provide an existence proof for theoretical feasibility; (3) Design a purely black-box, API-only optimization algorithm to construct such prompts; (4) Validate portability loss and utility retention across multiple models, agents, and tasks.

**Key Insight**: The authors leverage the "attention dilution" property of transformer attention—networks are insensitive to perturbations of certain tokens, so there theoretically exists an $\epsilon$-ball stability region $S_{\bm{x}}$ where utility is preserved. The geometric shape of the stability region differs across models, making target-specific perturbations unlikely to fall within other models' regions.

**Core Idea**: Treat prompt obfuscation as "gradient-free discrete optimization over the target-LLM-specific loss landscape," using random search to optimize prompt token sequences under a joint loss for utility, obfuscation, and non-portability.

## Method
PragLocker decomposes the abstract "non-portable obfuscation" into a formal existence problem and an engineering two-stage pipeline, with each component directly corresponding to one of the C1–C4 requirements.

### Overall Architecture
Input: plaintext prompt $\bm{x}$ + target LLM API + task training set $\mathcal{D}$.  
Stage 1 (Initialization Transformation): The target LLM encodes $\bm{x}$ into a "code-symbol form" $\tilde{\bm{x}}_0$, preserving semantics but no longer in natural language.  
Stage 2 (Noise-Injected Obfuscation Optimization): Random search repeatedly injects character-level noise, with each step accepting or rejecting based on a joint objective of task loss, obfuscation loss, and non-language loss.  
The final output $\tilde{\bm{x}}$ is deployed in untrusted environments and used directly at runtime without deobfuscation.

### Key Designs

1. **Theoretical Motivation: Functional Equivalence + Stability Region**:

    - **Function**: The authors formally define functional equivalence—embeddings $\tilde{\bm{h}}$ and $\bm{h}$ are equivalent if they produce the same greedy decoding for query $\bm{q}_i$. The correct-class margin is defined as $m(\tilde{\bm{h}}, \bm{q}_i, y_i) = f(\tilde{\bm{h}}, \bm{q}_i)_{y_i} - \max_{k \neq y_i} f(\tilde{\bm{h}}, \bm{q}_i)_k$. As long as the margin $> 0$, there exists an $\epsilon$-ball $B_\epsilon(\bm{h})$ where all points maintain functional equivalence, defining the "stability region" $S_{\bm{x}}$.
    - **Mechanism**: The "Existence of obfuscated prompts" theorem is proven—by perturbing $k$ low-attention tokens, the cumulative embedding shift $\|\Delta\bm{h}\| \le \sum_{j} \|\bm{\delta}_j\|$ can be controlled within $\epsilon$ to preserve utility, while the discrete prompt distance $d(\tilde{\bm{x}}, \bm{x})$ increases with $k$ to ensure obfuscation. Non-portability arises from "manifold mismatch": the stability regions $S_{\bm{x}}(\theta)$ and $S_{\bm{x}}(\theta')$ for models $\theta$ and $\theta'$ are almost disjoint in high-dimensional space.
    - **Design Motivation**: The position-paper claim that "prompts can be obfuscated" is often criticized as ad hoc; the authors provide a theoretical foundation using attention dilution and high-dimensional sparsity. Non-portability is elevated from an empirical phenomenon to an interpretable geometric property.

2. **Stage 1: Code-Symbol Initialization**:

    - **Function**: The target LLM translates the original prompt into a code-symbol form as the warm start $\tilde{\bm{x}}_0$. This initialization preserves semantics and utility while shifting the representation from natural language to a "code + symbol" format that the target LLM can still interpret but is more compact.
    - **Mechanism**: The target LLM generates the symbolic version itself, serving as a "target-conditioned" preliminary obfuscation with inherent target bias. This representation introduces redundancy, providing room for subsequent noise injection during random search.
    - **Design Motivation**: Random search from scratch is unlikely to hit a functionally equivalent prompt; using the LLM to perform a semantic-preserving transformation places the search starting point within the stability region.

3. **Stage 2: Random-Search Noise Injection + Joint Loss**:

    - **Function**: At each step, a noise vector $\bm{n}_t$ (from common printable chars) is sampled and injected in-place into the current prompt $\tilde{\bm{x}}_t$, yielding candidate $\tilde{\bm{x}}'_{t+1}$. Acceptance is based on whether the loss decreases. The objective is $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{dist}} + \gamma \mathcal{L}_{\text{non-lang}}$, where $\mathcal{L}_{\text{task}} = -\log p(\bm{y}|\bm{q}, \tilde{\bm{x}})$ preserves utility, $\mathcal{L}_{\text{dist}} = -\log \sigma(\mathrm{Dist}(\tilde{\bm{x}}, \bm{x}))$ (Levenshtein distance) increases divergence from the original, and $\mathcal{L}_{\text{non-lang}} = -H(\tilde{\bm{x}})$ minimizes character Shannon entropy to push the result away from natural language distribution.
    - **Mechanism**: Random search is a classic gradient-free discrete optimization method (Rastrigin 1963), recently used in jailbreak suffix optimization. Each loss term addresses a requirement: task for C3, dist for C2, and non-lang for both C2 and C4—since natural language is inherently portable, pushing the prompt away from natural language distribution effectively destroys cross-model transferability.
    - **Design Motivation**: The black-box constraint precludes gradients, so only sampling and filtering are feasible. The non-language term is a key trick, making the final prompt resemble a random token salad that still works on the target LLM—equivalent to finding a "model-conditioned trigger" on the target's loss landscape.

### Loss & Training
Training follows Algorithm 1: at each step, sample a mini-batch $(\bm{q}_t, \bm{y}_t)$ and noise $\bm{n}_t$, compare the loss before and after injection, and update via greedy random search if the loss decreases. No gradients or white-box access are required; only the target LLM API's log-prob and text output are needed.

## Key Experimental Results

### Main Results
Portability is measured by optimizing a prompt for a target LLM, then running it as-is on other LLMs to assess task performance (e.g., LessonL agent + HumanEval/MBPP):

| Agent / Task | Target LLM | Original Prompt → GPT-4o | Original Prompt → Gemini2 | PragLocker → GPT-4o | PragLocker → Gemini2 |
|------|------|------|------|------|------|
| LessonL / HumanEval | Gemini 2 | 93.90 | - | 0.61 | - |
| LessonL / HumanEval | DeepSeek | 93.90 | 98.78 | 0.61 | 2.44 |
| LessonL / MBPP | Gemini 2 | 91.89 | - | 0.51 | - |
| LessonL / MBPP | DeepSeek | 91.89 | 97.33 | 0.62 | (near 0) |

PragLocker reduces cross-model transfer performance to nearly zero, while the original prompt's cross-model performance often exceeds 90, demonstrating extremely strong protection.

### Ablation Study
The paper presents two ablation variants—PragLocker_tune (task loss tuning only, no non-lang noise) and PragLocker_code (code-symbol initialization only, no noise optimization):

| Configuration | Cross-Model Transfer Performance | Notes |
|------|------|------|
| Original Prompt | 93-99 | Fully portable |
| PragLocker_code (init only) | 85-95 | Code-form slightly reduces portability, still usable |
| PragLocker_tune (task tune only) | 80-93 | Task loss alone insufficient to block cross-model transfer |
| **PragLocker (full)** | 0-3 | Joint loss + random search reduces portability to near zero |

### Key Findings
- The non-language loss is the key contributor to non-portability: task tuning or code-form alone cannot significantly reduce cross-model transfer; only by pushing the prompt outside the natural language distribution does it fail on other LLMs.
- Even for the same model with different precisions (FP16 vs 4-bit quantized), PragLocker's obfuscated prompt barely transfers, indicating the stability region is sensitive to model weight precision—strong empirical evidence for the "manifold mismatch" geometric hypothesis.
- The target LLM itself cannot interpret the obfuscated prompt's semantics (i.e., cannot deobfuscate to the original), indicating it degenerates into a "model-conditioned trigger" with no recoverable text-level information, enhancing robustness against adaptive attacks.

## Highlights & Insights
- Modeling prompt IP protection as "gradient-free discrete optimization" is an apt framing—gradients are unavailable (black-box), fluent generation is undesirable (need non-language), and random search with joint loss fits these constraints.
- The non-language entropy regularizer is the most transferable trick: any scenario aiming to block "semantic cross-model transfer" can leverage it—by pushing intermediate representations away from natural language, cross-model readability collapses.
- Using the target LLM for code-symbol initialization is an elegant warm start—essentially placing the search starting point within the target-specific manifold, greatly reducing the effective search space for random search.
- The existence theorem, explained via attention dilution and high-dimensional sparsity, provides a geometric foundation for previously "empirically effective but theoretically lacking" prompt obfuscation work, offering insights for other prompt attack/defense research.

## Limitations & Future Work
- Assumes attackers cannot access large numbers of query-output pairs to train their own deobfuscators, but in commercial deployments, attackers may continuously farm data; long-term attack risks are not fully discussed.
- The weights $\lambda, \gamma$ for the three loss terms require manual tuning and may need to be re-optimized for different agents/tasks.
- Random search converges slowly on long prompts, with potentially low acceptance rates; lacks comparison with alternative optimizers such as evolution strategies or GCG.
- Non-portability is mainly validated on GPT-4o, Gemini 2, and DeepSeek model families; robustness to unseen closed-source models (e.g., Claude) or minor fine-tunes within the same family remains unclear.
- The method indirectly increases token count, potentially raising API costs and latency, but the paper does not provide quantification.

## Related Work & Insights
- **vs prompt watermarking (PromptCARE / PromptCOS)**: Watermarking only enables post-hoc accountability and does not prevent misuse; PragLocker is proactive, preventing reuse after theft.
- **vs encryption-based (K8s secrets / TEE)**: Encryption only protects at rest; at runtime, prompts must be decrypted to plaintext for black-box LLM APIs. PragLocker ensures the runtime prompt is already obfuscated.
- **vs EmojiPrompt / Pape's representation obfuscation**: Emojis can still be decoded by other LLMs; representation methods require white-box access. PragLocker combines pure black-box, non-portability, and utility.
- **vs GCG-style jailbreak suffix optimization**: Technically also gradient-free discrete optimization, but PragLocker is defensive, with objectives including non-portability rather than attack success rate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic proposal for agent prompt IP protection and a "non-portable obfuscation" solution, novel in both theory and engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple agents, tasks, and model families, including adaptive attack evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear requirement breakdown, existence proof, algorithm description, and ablation study.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core IP pain point in commercial LLM Agent deployment; method is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Minimal Agent for Automated Theorem Proving](a_minimal_agent_for_automated_theorem_proving.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](../../ICLR2026/llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)

</div>

<!-- RELATED:END -->
