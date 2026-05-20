---
title: >-
  [Paper Note] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation
description: >-
  [NeurIPS 2025][Code Intelligence][Verilog code generation] This paper proposes QiMeng-SALV, a signal-aware learning method that extracts functionally correct signal-level code snippets from partially incorrect Verilog mo…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "Verilog code generation"
  - "signal-level optimization"
  - "DPO"
  - "AST"
  - "reinforcement learning"
date: 2026-05-08
content_hash: a3be034b39daf8f7
---

# QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.19296](https://arxiv.org/abs/2510.19296)  
**Code**: [GitHub](https://github.com/QiMeng-IPRC/QiMeng-SALV)  
**Area**: Code Generation / Hardware Design Automation
**Keywords**: Verilog code generation, signal-level optimization, DPO, AST, reinforcement learning

## TL;DR

This paper proposes QiMeng-SALV, a signal-aware learning method that extracts functionally correct signal-level code snippets from partially incorrect Verilog modules as reward signals for DPO training, elevating the optimization granularity from module level to signal level and achieving SOTA on VerilogEval and RTLLM.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: While LLMs have shown promise in Verilog code generation, RL-based preference optimization suffers from **insufficient functional reward** signals.

### State of the Field

**Background**: Existing RL methods rely on code structural similarity as rewards; however, a given functionality can be correctly implemented in multiple structurally distinct ways.

### Root Cause

**Key Challenge**: Using functional correctness as a reward is more principled, yet insufficient training data during the SFT stage makes it difficult for models to generate entirely correct modules.

### Solution

**Solution**: The key insight is that Verilog describes the structural interconnection of hardware gates and wires, making different output signals naturally independent. Even when an entire module is functionally incorrect, certain signal implementations may be correct—and these can provide effective functional correctness rewards.

## Method

### Overall Architecture

QiMeng-SALV consists of three stages:

1. **Signal-aware Verification**: Randomly generated test inputs are used to compare output signals between the generated module and a reference module, identifying correctly implemented signals.
2. **Signal-aware Code Extraction**: AST analysis is used to construct a signal dependency graph, from which code snippets related to target signals are extracted.
3. **Signal-aware DPO Training**: Token probabilities are computed only over code tokens associated with the contrasted signals, training the model to learn correct signal implementations.

### Key Designs

1. **AST-based Signal-aware Code Extraction**:
    - Function: Precisely extracts the complete code implementation corresponding to a specific output signal from a generated Verilog module.
    - Mechanism: Yosys parses the module code into an AST; dependency relationships among all output and intermediate signals are analyzed to form a topological graph; a reverse traversal from the target signal leaf node retrieves all dependent signals and their associated code.
    - Design Motivation: The signal independence of Verilog enables extraction of self-contained implementations of individual signals at the module level, providing process-level feedback for RL.

2. **Signal-level DPO Loss**:
    - Function: Extends standard DPO by computing token probabilities only over code tokens related to the contrasted signal.
    - Mechanism: Given a contrasted signal $c$, code snippets $S_w^c$ and $S_l^c$ are extracted from preferred ($y_w$) and dispreferred ($y_l$) samples respectively; the DPO loss is computed solely over these tokens.
    - Design Motivation: Standard DPO assumes the preferred sample is entirely correct, which does not hold in the Verilog setting; signal-level DPO avoids noise introduced by incorrectly implemented signals.

### Loss & Training

Signal-level DPO loss:

$$\mathcal{L}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}\left[\log\sigma\left(\beta\sum_{y_t\in S_w^c}\log\frac{\pi_\theta(y_t|y_{w,<t},x)}{\pi_{\text{ref}}(y_t|y_{w,<t},x)} - \beta\sum_{y_t\in S_l^c}\log\frac{\pi_\theta(y_t|y_{l,<t},x)}{\pi_{\text{ref}}(y_t|y_{l,<t},x)}\right)\right]$$

- SFT stage: Full-parameter fine-tuning on 135k Verilog samples for 2 epochs.
- DPO stage: LoRA fine-tuning for approximately 7,000 steps (~1 epoch), learning rate 5e-6.
- Base model: Qwen2.5 Coder Instruct 7B; 5 candidate outputs sampled per prompt.

## Key Experimental Results

### Main Results

| Model | Params | VerilogEval1.0 pass@1 | VerilogEval2.0 pass@1 | RTLLM v1.1 pass@1 |
|-------|--------|----------------------|----------------------|-------------------|
| GPT-4o | - | 60.1 | 62.5 | - |
| DeepSeek v3 | 671B | 70.7 | 68.8 | - |
| CodeV (Qwen2.5) | 7B | 57.9 | 44.8 | - |
| Origen | 7B | 54.4 | 49.3 | - |
| **QiMeng-SALV** | **7B** | **65.6** | **62.6** | **62.6** |

### Ablation Study

- Removing signal-aware verification (using module-level DPO): significant performance drop, confirming the necessity of signal-level rewards.
- Removing token filtering for contrasted signals (computing over all module tokens): noise from incorrectly implemented signals degrades DPO effectiveness.
- QiMeng-SALV 7B achieves pass@5 of 75.1% on RTLLM v1.1, matching DeepSeek v3 671B.

### Key Findings

- A 7B model matches the performance of DeepSeek-V3 671B on RTLLM.
- Signal-level optimization provides denser functional correctness rewards than module-level optimization, effectively expanding the usable training sample space.
- Modules containing correct signal implementations can be incorporated into training even when the overall module is incorrect.

## Highlights & Insights

- **Paradigm shift**: The first signal-level RL algorithm for Verilog generation, elevating optimization granularity from module level to signal level.
- Cleverly exploits the **hardware description characteristics** of Verilog (signal independence) to design the RL framework.
- AST-based signal extraction is generalizable and applicable to other hardware description languages.

## Limitations & Future Work

- Signal verification relies on random test inputs and may miss errors arising under specific boundary conditions.
- Validation is limited to Qwen2.5 Coder 7B; the effectiveness on larger-scale models remains unknown.
- AST parsing depends on the Yosys tool and does not support all Verilog syntax variants.
- Integration with more recent RL methods such as GRPO has not been explored.

## Related Work & Insights

- VeriPrefer applies module-level functional rewards for RL, whereas signal-level rewards offer greater density.
- CodeV focuses on dataset construction for the SFT stage; QiMeng-SALV provides complementary improvements at the RL stage.
- The signal-level extraction idea is generalizable to other code generation tasks with modular structure.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[NeurIPS 2025\] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [\[NeurIPS 2025\] MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)
- [\[NeurIPS 2025\] Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)

</div>

<!-- RELATED:END -->
