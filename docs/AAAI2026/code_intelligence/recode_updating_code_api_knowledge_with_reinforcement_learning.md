---
title: >-
  [Paper Note] ReCode: Updating Code API Knowledge with Reinforcement Learning
description: >-
  [AAAI 2026][Code Generation] This paper proposes ReCode, a framework that trains LLMs via rule-based reinforcement learning (rather than SFT) to correctly leverage API update documentation provided in the prompt for code version migration, enabling a 7B model to surpass 32B models on CodeUpdateArena.
tags:
  - AAAI 2026
  - Code Generation
  - API Update
  - Reinforcement Learning
  - GRPO
  - Version Migration
date: 2026-05-08
content_hash: 76f8608a684fb891
---

# ReCode: Updating Code API Knowledge with Reinforcement Learning

**Conference**: AAAI 2026
**arXiv**: [2506.20495](https://arxiv.org/abs/2506.20495)
**Code**: [https://github.com/zjunlp/ReCode](https://github.com/zjunlp/ReCode)
**Area**: Code Intelligence
**Keywords**: Code Generation, API Update, Reinforcement Learning, GRPO, Version Migration

## TL;DR

This paper proposes ReCode, a framework that trains LLMs via rule-based reinforcement learning (rather than SFT) to correctly leverage API update documentation provided in the prompt for code version migration, enabling a 7B model to surpass 32B models on CodeUpdateArena.

## Background & Motivation

LLMs have demonstrated strong code generation capabilities; however, external library APIs evolve rapidly (e.g., NumPy and PyTorch version iterations), while model parameters encode only the outdated API knowledge present at training time. When users operate in newer environments, model-generated code may invoke deprecated APIs and fail at runtime.

**Limitations of Prior Work**: (1) SFT directly fine-tunes models to memorize new API knowledge, but the high frequency of API updates makes continuous fine-tuning costly and prone to catastrophic forgetting; (2) placing update documentation in the prompt (akin to RAG) avoids parameter modification but yields limited improvements—models exhibit a "laziness" tendency, preferring to rely on internal parametric knowledge over external documentation when the two conflict.

**Root Cause**: The conflict between stale parametric knowledge and fresh knowledge provided in the prompt. Models inherently trust their own parameters, causing them to generate code using outdated APIs even when complete update documentation is supplied.

**Starting Point**: Since the problem is not that models are unaware of new APIs, but rather that they are unwilling to exploit new information in the prompt, reinforcement learning is used to cultivate the habit of "respecting external knowledge in the prompt." This mirrors the learning pattern of human programmers—first learning an older version, then migrating code to a newer version upon reading release notes.

## Method

### Overall Architecture

The core mechanism of ReCode is to construct a version migration training task and fine-tune the model with RL (rather than SFT), teaching it to migrate legacy code to a new version based on API update documentation provided in the prompt. At inference time, the model performs actual programming tasks grounded in update documentation.

**Training pipeline**: The input is `[dependency library, target version, update notes, legacy code]`; the output is the migrated code. Reward signals (string matching + AST syntax checking) guide the model to correctly interpret update documentation and perform migration.

**Evaluation pipeline**: On CodeUpdateArena, given a dependency library, API update documentation, a programming problem, and a function signature, the model generates a complete function using the new API, evaluated by Pass@k on test cases.

### Key Designs

1. **Training Data Construction (~2,000 instances)**

    - Real API update descriptions are extracted from release notes of mainstream libraries including NumPy, Pandas, PyTorch, and matplotlib.
    - GPT-4 is used to generate functionally equivalent legacy and updated code pairs for each update entry.
    - Manual review ensures code correctness, covering diverse change types such as API renaming, parameter addition, and behavioral modification.
    - The training set is fully isolated from the test set CodeUpdateArena (which is LLM-synthesized) to prevent data leakage.

2. **Reward Design (Format + Correctness)**

    - **Format reward**: The output must contain `<think>...</think><answer>...</answer>` tags; compliance yields +1, otherwise −1.
    - **Correctness reward**: A key innovation—rather than test-case pass rates (since migration tasks focus on correctness of migration rather than general functional correctness), string similarity is used.
    - ES* (Edit Similarity + AST syntax check) is proposed: the output is first parsed via AST; a syntax error yields −2.0; if syntactically valid, the score is mapped to $[-1.5, 2.0]$ via $ES^*(x) = ES(x) \times 3.5 - 1.5$.
    - Ablation experiments demonstrate that ES* outperforms EM*, as ES provides continuous reward values that prevent the zero-advantage problem in GRPO caused by uniform rewards within a group.

3. **RL Algorithm**

    - Both GRPO and DAPO policy gradient algorithms are supported.
    - DoRA (r=64, α=64) is used for parameter-efficient fine-tuning, trained for 5,000 steps with batch size 8 and learning rate $5 \times 10^{-5}$.

### Loss & Training

Total reward = format reward + correctness reward (ES*). A warm-up phase is applied for the first 150 steps, followed by cosine decay. Reward temporarily drops in early training (as instruction-tuned models lose instruction-following ability upon entering the exploration phase) before steadily increasing.

## Key Experimental Results

### Main Results

| Model | Method | CodeUpdateArena Pass@1 | Pass@5 | HumanEval+ (Δ) |
|------|------|----------------------|--------|----------------|
| Qwen2.5-Coder-32B-Instruct | No training | 75.7 | 84.3 | - |
| DeepSeek-R1-Distill-Qwen-32B | No training | 78.2 | 86.1 | - |
| Qwen2.5-Coder-7B-Instruct | No training | 67.3 | 74.0 | 84.1 |
| Qwen2.5-Coder-7B-Instruct | SFT | 69.4 (+2.1) | 78.2 (+4.1) | 70.2 (−11.7) |
| Qwen2.5-Coder-7B-Instruct | ReCode GRPO | 74.6 (+7.4) | 82.1 (+8.0) | 82.3 (−1.8) |
| **Qwen2.5-Coder-7B-Instruct** | **ReCode DAPO** | **78.7 (+11.3)** | **84.3 (+10.2)** | **81.7 (−2.4)** |
| DS-v1.5-Coder-7B-Instruct | ReCode DAPO | 63.6 (+4.5) | 78.2 (+5.6) | 68.9 (−2.4) |

Key finding: Qwen2.5-Coder-7B + ReCode DAPO achieves Pass@1 of 78.7, surpassing both the 32B instruction-tuned model (75.7) and the 32B reasoning model (78.2).

### Ablation Study (Reward Design)

| Correctness Reward | Pass@1 (Δ) | Pass@5 (Δ) |
|-----------|-----------|-----------|
| Format only | −2.3 | −3.0 |
| +EM | −1.2 | −3.2 |
| +ES | +5.4 | +5.2 |
| +EM* (+ AST) | +1.1 | +2.0 |
| +ES* (+ AST) | **+7.4** | **+8.0** |

### Key Findings

- **ES outperforms EM**: Continuous similarity rewards are better suited for GRPO than exact match, avoiding zero-advantage situations caused by uniform within-group rewards.
- **AST syntax checking is necessary**: Adding AST checking improves both EM* and ES* over their base variants; pure string matching may cause degradation in the model's task comprehension.
- **SFT severely degrades general capabilities**: SFT incurs a drop of 11.7 points on HumanEval+, whereas ReCode loses only 1.8–2.4 points.
- **Multi-API update scenario**: Across 20 test cases involving simultaneous updates to multiple APIs, ReCode improves the 7B model's Pass@1 from 35 to 60, approaching the 32B reasoning model's score of 65.

## Highlights & Insights

- **A novel approach to resolving knowledge conflicts via RL**: Rather than injecting new knowledge into model parameters, the approach trains models to "respect external information in the prompt." This is a generalizable and transferable solution with broad implications for all RAG-based settings.
- **Decoupling training and test tasks**: Training involves code version migration while evaluation involves documentation-grounded programming; the two tasks differ yet the learned capability transfers, indicating that RL acquires generalized ability rather than memorizing specific patterns.
- **Continuous rewards mitigate the zero-advantage problem in GRPO**: ES is more compatible with group-based RL algorithms than EM due to its discriminative reward signal.

## Limitations & Future Work

- Training data comprises only ~2,000 instances restricted to Python data science libraries; coverage is limited, and extensibility to other language ecosystems such as JavaScript or Rust remains unexplored.
- The model's capability ceiling is constrained by the pre-training base—RL cannot compensate for tasks requiring capabilities such as physical reasoning that are absent from the base model.
- Current rewards are based on string matching rather than execution testing, potentially penalizing functionally equivalent code with different surface forms.
- DAPO outperforms GRPO, but the underlying reasons are not thoroughly analyzed.

## Related Work & Insights

- **vs. SFT-based approaches (Liu et al. 2025c)**: SFT memorizes migration patterns through prompt-answer pairs, resulting in poor generalization and severe degradation of general capabilities. ReCode encourages genuine document comprehension via RL, yielding better generalization.
- **vs. RAG-based approaches**: Vanilla RAG is limited by the model's tendency to underutilize external information. ReCode fundamentally strengthens the model's ability to leverage prompt-provided information and can serve as an augmented training strategy for RAG systems.
- **vs. Versicode (Wu et al. 2024)**: Versicode covers only API renaming, whereas the dataset in this work encompasses a broader range of change types including parameter addition and behavioral modification.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying RL to the API update setting is a first exploration, though the method itself (GRPO + LoRA) combines existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are carefully designed, with coverage of multi-API scenarios and general capability evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; case studies are convincing.
- Value: ⭐⭐⭐⭐ The idea of using RL to resolve parametric-prompt knowledge conflicts has broad applicability and offers meaningful insights for RAG system optimization.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[NeurIPS 2025\] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](../../NeurIPS2025/code_intelligence/co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](../../ICLR2026/code_intelligence/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)

<!-- RELATED:END -->
