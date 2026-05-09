---
title: >-
  [Paper Note] Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification
description: >-
  [ACL 2026][LLM Agent][Tool-augmented agent] This paper proposes Spec-o3, a tool-augmented vision-language agent that simulates the spectral inspection workflow of professional astronomers via Interleaved Multimodal Chain-of-Thought (iMCoT). Through a two-stage training pipeline of cold-start SFT followed by outcome-based RL, Spec-o3 improves macro-F1 from 28.3% to 76.5% on rare celestial object identification, achieving ~50× speedup over manual inspection.
tags:
  - ACL 2026
  - LLM Agent
  - Tool-augmented agent
  - interleaved multimodal chain-of-thought
  - spectral inspection
  - reinforcement learning
  - domain VLM
date: 2026-05-08
content_hash: f54e26b4b8cfd09e
---

# Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification

**Conference**: ACL 2026
**arXiv**: [2601.06498](https://arxiv.org/abs/2601.06498)
**Code**: [Project HomePage](https://github.com/jiaminghui/Spec-o3)
**Area**: LLM Agent
**Keywords**: Tool-augmented agent, interleaved multimodal chain-of-thought, spectral inspection, reinforcement learning, domain VLM

## TL;DR

This paper proposes Spec-o3, a tool-augmented vision-language agent that simulates the spectral inspection workflow of professional astronomers via Interleaved Multimodal Chain-of-Thought (iMCoT). Through a two-stage training pipeline of cold-start SFT followed by outcome-based RL, Spec-o3 improves macro-F1 from 28.3% to 76.5% on rare celestial object identification, achieving ~50× speedup over manual inspection.

## Background & Motivation

**Background**: Modern spectroscopic survey projects (LAMOST, SDSS, DESI) generate massive datasets. Constructing catalogs of rare celestial objects requires a two-stage pipeline — deep learning algorithms for candidate screening, followed by expert visual vetting. The vetting stage remains heavily dependent on human labor.

**Limitations of Prior Work**: (1) Deep learning classifiers produce opaque probability scores with poor out-of-distribution generalization, making them difficult for experts to trust; (2) post-hoc explanation methods (Grad-CAM, SHAP, etc.) yield coarse feature attributions that cannot be reliably mapped to astrophysical structures; (3) manual vetting does not scale — for example, the LAMOST CV catalog required experts to visually inspect 170,000 candidates, ultimately confirming only 323 targets.

**Key Challenge**: The volume of candidates from next-generation surveys will continue to grow exponentially, while the throughput of manual inspection cannot scale accordingly, creating a critical bottleneck in astronomy.

**Goal**: Design a trustworthy, highly generalizable automated vetting agent capable of inspecting spectra in the manner of a trained astronomer.

**Key Insight**: The expert inspection workflow is inherently a "look-and-think" process — first assessing the global morphology, then iteratively zooming into wavelength regions of interest to examine diagnostic features, and finally reaching a verdict. Combining VLMs with spectral visualization tools allows this iterative process to be faithfully simulated.

**Core Idea**: Interleaved Multimodal Chain-of-Thought (iMCoT) — alternating between textual reasoning and fine-grained spectral plots rendered by tools, coupled with two-stage post-training to achieve expert-level vetting capability.

## Method

### Overall Architecture

Built on Qwen2.5-VL, the agent receives a text prompt $T_0$ (containing the discrimination query and expert diagnostic guidelines) along with an initial global spectral plot $I_0$. The agent performs textual reasoning within `<think>...</think>`, issues tool calls to generate zoomed-in views $I_{t+1}$ of local wavelength regions, and iterates until issuing a final verdict within `<answer>...</answer>`. The trajectory is formalized as $\tau = (T_0, I_0, T_1, I_1, T_2, I_2, \ldots, T_N)$.

### Key Designs

1. **Interleaved Multimodal Chain-of-Thought (iMCoT)**: The state is defined as $s_t = \{I_{\leq t}, T_{\leq t}\}$. At each step, the agent autonomously decides whether to directly output an answer or invoke the visualization tool $Tool_t$ to acquire finer-grained evidence. The tool accepts a wavelength interval $\Delta\lambda_t = (\lambda_t^{\min}, \lambda_t^{\max})$ and an optional diagnostic label $l_t$, and returns a locally re-rendered plot. The design motivation is to faithfully replicate the astronomer's workflow of global assessment → local verification → final decision, rendering the reasoning process auditable and physically consistent.

2. **Cold-Start SFT**: Approximately 4k spectra (SNR > 10) are sampled from the official LAMOST catalog, covering five rare object types (CV, CS, SS, MG, WD). GPT-5 first generates initial reasoning trajectories following expert guidelines; astronomers then conduct three rounds of review (initial screening → revision → final vote), yielding ~1k high-quality expert trajectories. Token-level loss masking is applied to tool-returned content during training to prevent the model from memorizing rendered outputs. The design motivation is to inject domain priors and tool-use capabilities from a small number of high-quality expert demonstrations.

3. **Outcome-Based Agentic RL**: The GRPO framework is used for outcome-based RL, leveraging only labeled data (without complete trajectories). The reward function is defined based on prediction correctness and format compliance: correct + format valid $\to r=1$; correct + format violation $\to r=1-\alpha$; incorrect + format valid $\to r=0$; incorrect + format violation $\to r=-\alpha$. The design motivation is that cold-start performance is bounded by scarce expert trajectories, and RL exploits more abundant labeled data to further optimize tool-use strategies.

### Loss & Training

Cold-start training uses standard SFT loss with token-level loss masking on tool-returned tokens. RL training uses GRPO (8 rollouts/question, up to 8 tool calls/trajectory). The two stages are applied sequentially. Base models are Qwen2.5-VL-3B/7B, trained on 8×H100 GPUs.

## Key Experimental Results

### Main Results (SpecVI-Bench, 5 Rare Object Classes)

| Model | CV F1 | CS F1 | SS F1 | MG F1 | WD F1 | Avg. F1 |
|-------|-------|-------|-------|-------|-------|---------|
| GaiaNet (DL Expert Model) | 67.2 | 87.1 | 70.3 | 51.8 | 48.2 | 64.9 |
| o3 (OpenAI) | 57.1 | 53.1 | 53.3 | 60.0 | 37.8 | 52.3 |
| Qwen2.5-VL-7B (base) | 25.4 | 31.5 | 27.3 | 29.0 | 28.1 | 28.3 |
| S1-VL-32B-SFT | 60.7 | 42.8 | 43.7 | 36.3 | 27.4 | 42.2 |
| **Spec-o3-7B** | **81.0** | **80.2** | **84.5** | **83.4** | **53.6** | **76.5** |

### Ablation Study

| # | SFT | RL | Tool | 3B F1 | 7B F1 |
|---|-----|----|------|-------|-------|
| 0 (Full) | ✓ | ✓ | ✓ | 73.3 | 76.5 |
| 1 | ✗ | ✓ | ✓ | 35.7 (-37.6) | 40.5 (-36.0) |
| 2 | ✓ | ✗ | ✓ | 33.1 (-40.2) | 41.6 (-34.9) |
| 4 | ✓ | ✓ | ✗ | 43.5 (-29.8) | 55.8 (-20.7) |

### Key Findings

- **Mutual dependence of two-stage training**: RL-only or SFT-only achieves only ~35–41% F1, whereas the combined pipeline jumps to 73–76% — cold-start provides domain priors while RL optimizes the tool-use strategy.
- **Tools are critical**: Removing tools drops the 7B model from 76.5% to 55.8% F1, confirming that static global views are insufficient to detect subtle diagnostic features.
- **Zero-shot cross-survey generalization**: Spec-o3 maintains 77–81% F1 on SDSS/DESI, while expert DL models degrade by 14–20%, indicating that Spec-o3 relies on transferable diagnostic evidence rather than survey-specific artifacts.
- **Zero-shot cross-task generalization**: The model achieves 76.4% F1 on unseen O/B/A-type spectra (o3: 60.9%), confirming that a general tool-assisted inspection paradigm has been learned.
- Inference efficiency: ~0.2s/sample on 8×H100, approximately 50× faster than expert manual inspection.

## Highlights & Insights

- **This is the first work to apply the "think-with-image" paradigm to scientific data analysis**, extending it from natural images to astronomical spectra and demonstrating the immense potential of tool-augmented VLMs in vertical domains.
- The data construction pipeline is exceptionally rigorous: GPT-5 generation → astronomer screening → revision → dual auditing → final vote, ensuring high cold-start data quality.
- **Cold-start data efficiency is high**: Reducing SFT data from ~1k to ~200 trajectories causes only marginal performance degradation (CV F1: 80.7 → 77.8).
- Tool-use behavior is sufficiently reliable after cold-start that no dedicated tool-use reward is required during RL.

## Limitations & Future Work

- Evaluation is limited to a small set of rare object types; broader spectral subclasses remain uncovered.
- The expert inspection workflow is abstracted as a "zoom-and-reason" loop, whereas real catalog construction also requires cross-matching with external databases and other observational modalities.
- Cold-start still requires expert involvement; the barrier to extending the approach to new tasks or surveys is non-trivial (though the synthetic data pipeline demonstrates the potential to reduce this requirement).
- No production-ready risk control mechanisms (e.g., calibration, abstention, triage) have been provided.
- WD classification achieves a relatively low F1 of 53.6%, likely because white dwarf spectral features are more subtle and may require more refined diagnostic strategies.

## Related Work & Insights

- **o3's think-with-image**: Spec-o3 extends this paradigm from general VQA to scientific data analysis, linking abstract diagnostics to visualized evidence.
- **GRPO (DeepSeek-R1)**: Outcome-based RL is extended from mathematical reasoning to scientific tool-use scenarios.
- Key insight: In vertical-domain VLMs, the critical factor is not model scale but rather tool-use and reasoning paradigms aligned with domain-specific workflows.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First iMCoT agent for astronomical spectroscopy; the two-stage training strategy is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Cross-survey, cross-task, and edge-case generalization evaluations, combined with human expert assessment and ablation studies — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Domain background is well introduced and the methodology is clearly described, though the barrier for non-astronomy readers is relatively high.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a real bottleneck in astronomical observation; the ~50× speedup carries substantial engineering value, and the paradigm is transferable to other scientific domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)

</div>

<!-- RELATED:END -->
