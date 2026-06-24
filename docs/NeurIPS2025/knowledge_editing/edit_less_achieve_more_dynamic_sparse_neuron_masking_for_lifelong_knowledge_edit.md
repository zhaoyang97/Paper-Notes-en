---
title: >-
  [Paper Note] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs
description: >-
  [NEURIPS2025][Knowledge Editing][lifelong learning] This paper proposes NMKE, a framework that identifies two categories of knowledge neurons—knowledge-general and knowledge-specific—via neuron-level attribution, and applies entropy-guided dynamic sparse masking to achieve precise neuron-level knowledge editing. NMKE maintains high edit success rates and general model capabilities after 5,000 consecutive edits.
tags:
  - "NEURIPS2025"
  - "Knowledge Editing"
  - "lifelong learning"
  - "sparse masking"
  - "neuron attribution"
  - "LLM"
date: 2026-05-08
content_hash: 09487472c1f3f4ea
---

# Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs

**Conference**: NEURIPS2025
**arXiv**: [2510.22139](https://arxiv.org/abs/2510.22139)  
**Code**: [LiuJinzhe-Keepgoing/NMKE](https://github.com/LiuJinzhe-Keepgoing/NMKE)  
**Area**: Knowledge Editing
**Keywords**: knowledge editing, lifelong learning, sparse masking, neuron attribution, LLM

## TL;DR
This paper proposes NMKE, a framework that identifies two categories of knowledge neurons—knowledge-general and knowledge-specific—via neuron-level attribution, and applies entropy-guided dynamic sparse masking to achieve precise neuron-level knowledge editing. NMKE maintains high edit success rates and general model capabilities after 5,000 consecutive edits.

## Background & Motivation
- Lifelong Knowledge Editing requires continuously updating outdated knowledge in LLMs without full retraining, yet existing methods suffer from accumulating errors as the number of edits grows.
- External-parameter methods (GRACE, WISE) incur increasing resource overhead and exhibit gradual accuracy degradation.
- Internal-parameter methods (ROME, MEMIT, AlphaEdit) modify parameters at the layer or parameter-block level, inevitably perturbing irrelevant neurons and causing forgetting and capability collapse.
- AlphaEdit, the prior SOTA, exhibits catastrophic forgetting after 1,500 edits: edit success rate drops by 0.67 on ZsRE and 0.78 on CounterFact.
- Root cause: coarse-grained parameter updates lead to cumulative damage to unrelated neurons under lifelong editing.
- The lack of fine-grained understanding of functionally distinct neurons in FFN layers prevents "editing only what should be edited."

## Method

### Neuron Attribution
- FFN layers are treated as key-value associative memories: the key of the $i$-th neuron is the $i$-th row of $W^{in}$, and the value is the $i$-th column of $W^{out}$.
- A perturbation-based method quantifies each neuron's contribution to target token prediction: the $i$-th neuron output $s^{(i)}$ is amplified, and the log-probability gain $\text{Imp}^{(i)}$ is computed.
- Three neuron types are identified: knowledge-general (stably activated across tasks), domain-specific (activated within the same domain), and task-specific (activated only for specific tasks).
- Ablation validation: masking the top-10 knowledge-general neurons drops accuracy from 37.5% to 4.17%; masking task-specific neurons reduces it by only 2.04%.

### Dynamic Sparse Masking
- **Knowledge-General neuron selection**: The count $r^{ge}_i$ of positive attributions for each neuron across multiple prompts is computed; higher counts indicate more general neurons.
- **Knowledge-Specific neuron selection**: The maximum attribution score $r^{sp}_i$ across prompts is used for each neuron.
- **Entropy-guided dynamic ratio**:
    - The general-neuron ratio $\rho_{ge}$ is determined by the mean normalized entropy $H_{ge}$ of the attribution distribution (higher entropy → more uniform activation → more general neurons).
    - The specific-neuron ratio $\rho_{sp}$ is determined by the entropy $H_{sp}$ of the maximum attribution distribution.
    - In practice, constant scaling factors $a_{ge}/a_{sp}$ and biases $b_{ge}/b_{sp}$ further adjust the ratios.
- **Mask generation**: Neurons satisfying $r^{ge} \geq \tau_{ge}$ or $r^{sp} \geq \tau_{sp}$ are unioned to form a binary mask $m^{(l)}$.
- **Threshold determination**: $\tau_{ge}$ is the $(1-\rho_{ge})$ quantile of the $r^{ge}$ distribution; $\tau_{sp}$ is defined analogously.
- During editing, only the subset of neurons selected by the mask is updated, following AlphaEdit's null-space projection optimization procedure.

### Three Attribution Computation Strategies
- **MPC (Mean Prompt Contribution)**: Uses the mean contribution across editing prompts.
- **PSA (Prompt-Specific Attribution)**: Uses the attribution of a specific prompt.
- **LPS (Layer-wise Prompt Selection)**: Selects the most relevant prompt's attribution per layer.
- All three strategies yield comparable performance; MPC is the fastest (~22 s/edit), while PSA/LPS are slightly slower (~30 s/edit) with marginally better results.

## Key Experimental Results

### Lifelong Knowledge Editing Performance (LLaMA3-8B-Instruct)

| Method | ZsRE T=1000 (Rel./Gen./Loc.) | ZsRE T=2000 (Rel./Gen./Loc.) |
|---|---|---|
| FT | 0.13/0.10/0.02 | 0.07/0.06/0.01 |
| ROME | 0.02/0.01/0.02 | 0.01/0.01/0.02 |
| MEMIT | 0.04/0.04/0.03 | 0.03/0.04/0.03 |
| WISE | 0.41/0.39/- | 0.37/0.36/- |
| AlphaEdit | 0.93/0.84/0.58 | 0.32/0.28/0.06 |
| **NMKE** | **0.95/0.85/0.77** | **0.94/0.85/0.71** |

| Method | CounterFact T=1000 (Rel./Gen./Loc.) | CounterFact T=2000 (Rel./Gen./Loc.) |
|---|---|---|
| AlphaEdit | 0.99/0.76/0.32 | 0.22/0.13/0.04 |
| **NMKE** | **0.99/0.65/0.50** | **0.98/0.67/0.38** |

### General Capability Retention

| General Capability (LLaMA3-8B, ZsRE T=2000) | AlphaEdit | NMKE |
|---|---|---|
| MMLU | ~0.25 (collapsed) | **0.59** (maintained through 5,000 steps) |
| GSM8K | 0.00 (T≥1500) | remains functional |
| HumanEval | 0.00 (T≥1500) | remains functional |
| CommonsenseQA | significant degradation | largely maintained |
| BBH-Zeroshot | significant degradation | largely maintained |

### Editing Efficiency

| Method | Time per Step (s) | T=2000 Rel. | T=2000 Loc. |
|---|---|---|---|
| MEMIT | 16.83 | 0.04 | 0.03 |
| AlphaEdit | 22.16 | 0.62 | 0.14 |
| NMKE (MPC) | 22.25 | 0.93 | 0.77 |
| NMKE (LPS) | 30.42 | 0.94 | 0.74 |

## Highlights & Insights
- This work is the first to identify the root cause of lifelong editing degradation from a neuron functional attribution perspective: coarse-grained updates cumulatively damage irrelevant neurons.
- The distinction between knowledge-general and knowledge-specific neurons, supported by ablation studies, is intuitive and well-motivated.
- The entropy-guided dynamic ratio mechanism outperforms fixed-ratio alternatives by adapting to varying activation patterns across prompt batches.
- MMLU remains at 0.59 after 5,000 consecutive edits, far surpassing all baselines, which collapse to near-random performance.
- t-SNE visualizations clearly demonstrate that NMKE induces far smaller perturbations to parameter distributions than AlphaEdit.
- The MPC variant incurs nearly identical per-step latency as AlphaEdit (22.25 s vs. 22.16 s) while substantially outperforming it.
- Ablation studies across four neuron selection strategies confirm that all variants effectively preserve general capabilities, validating the robustness of the framework.
- Experiments on GPT2-XL and Qwen2.5-7B demonstrate cross-model transferability.

## Limitations & Future Work
- Neuron attribution must be computed at each edit step, resulting in ~30 s per edit (vs. ~22 s for AlphaEdit), a modest efficiency overhead.
- The constant scaling factors $a_{ge}/a_{sp}$ and biases $b_{ge}/b_{sp}$ require manual tuning.
- Experiments are primarily conducted on LLaMA3-8B and GPT2-XL; scalability to 70B+ models remains unexplored.
- The Locality (Loc.) score on CounterFact, while superior to AlphaEdit, reaches only 0.38 in absolute terms, leaving room for improvement.
- The framework focuses exclusively on FFN-layer neurons and does not explore fine-grained editing of attention-layer parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of neuron functional categorization and entropy-guided dynamic masking is novel, though individual components are not entirely new.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Dual datasets ZsRE/CounterFact, 5,000-step continuous editing, five general-capability benchmarks, and extensive ablations.)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, intuitive figures, and complete mathematical derivations.)
- Value: ⭐⭐⭐⭐ (High practical utility for lifelong editing scenarios, though narrower in applicability than architectural innovations.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ICML 2025\] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs](../../ICML2025/knowledge_editing/wikibigedit_understanding_the_limits_of_lifelong_knowledge_editing_in_llms.md)
- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](../../ACL2026/knowledge_editing/representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[NeurIPS 2025\] Rethinking Residual Distribution in Locate-then-Edit Model Editing](rethinking_residual_distribution_in_locate-then-edit_model_editing.md)
- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](../../ACL2025/knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)

</div>

<!-- RELATED:END -->
