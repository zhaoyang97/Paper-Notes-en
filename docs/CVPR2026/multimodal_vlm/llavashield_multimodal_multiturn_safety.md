---
title: >-
  [Paper Note] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal multi-turn dialogue safety] Addressing three core challenges in multimodal multi-turn VLM dialogues—concealed malicious intent, cumulative contextual risk, and cross-modal joint risk—this work constructs the MMDS dataset (4,484 annotated dialogues) and the MCTS-based MMRT red-teaming framework, and proposes the LLaVAShield auditing model, achieving F1 scores of 95.71%/92.24% on the user/assistant sides respectively, substantially outperforming baselines such as GPT-5-mini.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal multi-turn dialogue safety
  - content moderation
  - red-teaming
  - MCTS
  - risk taxonomy
date: 2026-05-08
content_hash: 6daf61bc34ed677e
---

# LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2509.25896](https://arxiv.org/abs/2509.25896)
**Code**: [Project Page](https://leost123456.github.io/LLaVAShield/)
**Area**: Multimodal VLM / AI Safety
**Keywords**: Multimodal multi-turn dialogue safety, content moderation, red-teaming, MCTS, risk taxonomy

## TL;DR
Addressing three core challenges in multimodal multi-turn VLM dialogues—concealed malicious intent, cumulative contextual risk, and cross-modal joint risk—this work constructs the MMDS dataset (4,484 annotated dialogues) and the MCTS-based MMRT red-teaming framework, and proposes the LLaVAShield auditing model, achieving F1 scores of 95.71%/92.24% on the user/assistant sides respectively, substantially outperforming baselines such as GPT-5-mini.

## Background & Motivation

**State of the Field**: VLMs are being deployed at scale in interactive scenarios such as intelligent assistants and education, making safety concerns increasingly prominent. Existing content moderation tools (e.g., BingoGuard, WildGuard, LlamaGuard) have achieved preliminary progress but are primarily designed for single-turn or single-modality settings.

**Limitations of Prior Work**: Multimodal multi-turn dialogues exhibit three distinctive risk characteristics that render existing moderation methods ineffective. (1) **Concealed malicious intent**—attackers open with benign queries and escalate gradually, dispersing objectives across scattered textual and visual cues in multimodal contexts, with harm amplified substantially through cross-turn association; (2) **Cumulative contextual risk**—attackers decompose their ultimate goal across multiple turns, exploiting the model's "local compliance" to progressively expand the attack surface, with risk accumulating as the dialogue advances; (3) **Cross-modal joint risk**—even apparently normal image–text pairings may trigger unsafe generation, and cross-modal associations can be exploited to elicit harmful outputs.

**Root Cause**: Existing moderation methods either address only single turns or handle only a single modality, and are therefore unable to capture risk accumulated over multi-turn context or joint image–text attacks. Moreover, a severe scarcity of multimodal multi-turn dialogue safety datasets constrains research in this direction.

**Paper Goals**: Three components are needed: (1) an annotated dataset covering multi-dimensional risks; (2) a red-teaming framework capable of automatically and efficiently generating adversarial dialogues; and (3) a safety auditing model that understands full conversational context and cross-modal signals.

**Starting Point**: A unified data–attack–defense pipeline is adopted, yielding the MMDS dataset, the MMRT red-teaming framework, and the LLaVAShield auditing model.

**Core Idea**: MCTS is used to efficiently explore attack trajectories and generate safety data, enabling the training of a multimodal multi-turn dialogue safety model that simultaneously audits user inputs and assistant responses.

## Method

### Overall Architecture
The system comprises three stages: (1) **MMDS dataset construction**—building a safety dataset of 4,484 dialogues through malicious intent generation, image mining, MMRT red-teaming attacks, and multi-layer annotation; (2) **MMRT red-teaming framework**—MCTS-based automated attack path exploration generating unsafe dialogues across turns and modalities; (3) **LLaVAShield model**—a VLM fine-tuned on MMDS that takes as input an instruction, safety policy, and dialogue history, and outputs dual-sided safety scores, violated policy dimensions, and evidence chains.

### Key Designs
1. **MMRT (MCTS-Based Multimodal Multi-Turn Red-Teaming Framework)**:

    - **Function**: Automatically generates unsafe multimodal multi-turn dialogues, providing high-quality adversarial samples for MMDS.
    - **Mechanism**: Red-teaming is modeled as an iterative interaction among an attacker $\mathcal{A}$, a target VLM $\mathcal{T}$, and an evaluator $\mathcal{E}$. At each turn $t$, the attacker generates an attack plan $(q_t, \mathcal{I}_t)$ conditioned on the malicious goal $g$, dialogue context $c_{t-1}$, and strategy set $\Sigma$; the target model produces a response $r_t = \mathcal{T}(q_t, \mathcal{I}_t, c_{t-1})$; the evaluator assigns a score $s_t \in \{1,\ldots,5\}$. Crucially, MCTS (Selection–Expansion–Simulation–Backpropagation) is employed to overcome the search-space limitations of linear attack chains: nodes are selected via the PUCT formula, downstream risk is estimated through $k$-step rollouts, and the reward is defined as $z = (s_{t+k}-1)/4$.
    - **Design Motivation**: The linear $\mathcal{A} \to \mathcal{T} \to \mathcal{E}$ loop offers a limited search space; MCTS efficiently explores branching attack trajectories to discover more covert and effective attack sequences. Attack strategies include progressive guidance, goal reversal, query decomposition, and role-playing.

2. **MMDS Dataset Construction and Annotation Pipeline**:

    - **Function**: Constructs the first annotated dataset for multimodal multi-turn dialogue safety, comprising 4,484 dialogues (2,756 original + 1,728 augmented) and covering a risk taxonomy of 8 primary dimensions and 60 sub-dimensions.
    - **Mechanism**: Two data sources are used—756 unsafe dialogues generated by MMRT and 2,000 safe dialogues sampled from MMDU-45k. Each dialogue is independently annotated on both the user and assistant sides with safety ratings and violated policy dimensions. Four data augmentation strategies are then applied to improve generalization: (a) randomly removing non-violated policy dimensions; (b) rewriting unsafe responses into compliant text using GPT-5-mini to reduce false positives; (c) single-side masking during training; (d) adjusting policy configurations to prevent over-moderation.
    - **Design Motivation**: Label supervision alone provides limited information; a **role-decoupled dual-channel rationale mechanism** is therefore introduced, generating independent explanatory reasoning chains for the user and assistant sides and requiring each reasoning trace to provide key evidence, ensuring traceability and verifiability of classifications.

3. **LLaVAShield Auditing Model**:

    - **Function**: Simultaneously audits the safety of user inputs and assistant responses in multimodal multi-turn dialogues under specified policy dimensions.
    - **Mechanism**: Safety auditing is formulated as a unified seq2seq task. The input concatenates three components: instruction $\mathcal{G}$, policy set $\mathcal{P}$, and $T$-turn dialogue history $\mathcal{C} = \{(V_t^u, x_t^u, x_t^a)\}_{t=1}^T$. The model outputs a structured JSON containing six components—dual-sided safety ratings $S_u, S_a$, violated policies $D_u, D_a$, and evidential reasoning $R_u, R_a$—all generated within `<OUTPUT>...</OUTPUT>` tags.
    - **Design Motivation**: The unified format makes outputs machine-parseable and supports downstream automated processing. Policy dimensions are provided as input parameters, allowing the model to flexibly adapt to the safety specifications of different deployment scenarios.

### Loss & Training
The model is initialized from LLaVA-OV-7B and fine-tuned on the MMDS training set. The optimization objective is conditional log-likelihood maximization: $\max_\theta \sum \log p(\mathcal{Y} | \mathcal{G}, \mathcal{P}, \mathcal{C}; \theta)$. Training uses a learning rate of $2 \times 10^{-5}$ with cosine scheduling and 0.03% total-step warmup, batch size of 1 with 4-step gradient accumulation, for 3 epochs. Training is conducted on 8×NVIDIA A6000 (48 GB) GPUs and completes in approximately 3 hours.

## Key Experimental Results

### Main Results

| Model | User-side F1 (%) | Assistant-side F1 (%) | Notes |
|---|---|---|---|
| LLaVAShield-7B | **95.71** | **92.24** | Open-source, 7B params |
| GPT-5-mini | 75.46 | 77.93 | Closed-source, strongest baseline |
| Gemini-2.5-Pro | 64.00 | 65.62 | Closed-source |
| GPT-4o | 61.54 | 57.92 | Closed-source |
| InternVL3.5-38B | 29.15 | 36.71 | Open-source |
| Llama Guard-4-12B | 14.21 | 28.21 | Dedicated moderation tool |
| Qwen2.5-VL-7B | 1.17 | 1.54 | Same-scale open-source |

LLaVAShield achieves 100% precision and 91.76% recall on the user side, surpassing GPT-5-mini by +20.25 and +14.31 F1 points on the respective sides.

### Generalization on External Benchmarks

| Benchmark | Metric | LLaVAShield | GPT-5-mini | Llama Guard-4 |
|---|---|---|---|---|
| MM-SafetyBench | Avg Recall (%) | **97.62** | 48.44 | 44.49 |
| VLGuard-Test | F1 (%) | **90.55** | 86.39 | 64.87 |

### Ablation Study

| Configuration | User-side F1 (%) | Assistant-side F1 (%) | Notes |
|---|---|---|---|
| Vanilla (with rationale) | 95.71 | 92.24 | Full model |
| w/o rationale | 95.12 | 93.93 | Reasoning chain removed |
| Policy-adaptive FPR | 0% / 0% | — | GPT-5-mini: 30% / 34% |

### Key Findings
- All open-source VLMs exhibit near-zero recall in multimodal multi-turn settings, demonstrating that built-in safety alignment is effectively nullified in this scenario.
- The inclusion of images raises the evaluator's average risk score by 0.375 (ASG), as visual cues transform generic guidance into operationally high-risk content.
- Increasing dialogue turns makes VLMs more prone to generating harmful content, though the effect saturates after approximately 6 turns.
- Attack success rates against mainstream VLMs are extremely high: Qwen2.5-VL-72B reaches 100%, GPT-4o reaches 98.21%, and even Claude-3.7-Sonnet reaches 73.77%.

## Highlights & Insights
- **Precise problem formulation**: The three risk characteristics of multimodal multi-turn dialogues (concealment, accumulation, cross-modality) are systematically identified for the first time, yielding a clear problem model.
- **MCTS red-teaming framework**: Drawing on game-tree search, the framework explores attack trajectories far more efficiently than linear trial-and-error, generating diverse attack trajectories.
- **Comprehensive data augmentation**: Four augmentation strategies address distinct deployment challenges—false positives, incomplete single-side information, and over-moderation.
- **7B model outperforms closed-source giants**: LLaVAShield-7B substantially surpasses GPT-5-mini and Gemini-2.5-Pro on this specialized task, demonstrating that targeted fine-tuning far exceeds general-purpose large models for safety auditing.
- **Strong policy-adaptive capability**: With a FPR of 0%, the model makes judgments strictly according to the input policy dimensions and does not generalize to irrelevant risk categories.

## Limitations & Future Work
- MMDS contains only 4,484 dialogues and is primarily generated by two target VLMs (GPT-4o and Qwen2.5-VL-72B), which may limit diversity.
- The risk taxonomy is manually designed and may struggle to cover emerging risk types such as deepfakes or AI-generated social engineering attacks.
- The safety auditing model introduces additional inference latency; inference speed and deployment overhead have not been reported.
- The attack efficiency of MMRT depends on MCTS hyperparameters and rollout steps; no search efficiency analysis is provided.
- Training uses only LLaVA-OV-7B as the backbone; effectiveness on stronger backbones has not been validated.

## Related Work & Insights
- **vs. Llama Guard series**: Llama Guard-4-12B achieves only 14.21%/28.21% F1 on MMDS, confirming that single-turn moderation tools cannot handle multi-turn context.
- **vs. ShieldVLM**: A concurrent work targeting multimodal implicit toxicity, but still limited to single-turn settings and lacking multi-turn reasoning capability.
- **vs. Red Queen**: A multi-turn red-teaming work restricted to text-only LLMs; the present work extends this to multimodal settings and improves search efficiency via MCTS.
- **vs. IDEATOR**: The attack strategy combination used here (progressive guidance + goal reversal + query decomposition + role-playing) is inspired by IDEATOR but extended to cross-modal image–text attacks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic treatment of multimodal multi-turn dialogue safety; the MCTS red-teaming framework and role-decoupled rationale mechanism are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Includes main experiments, fine-grained analysis across 8 dimensions, 2 external benchmarks, policy adaptation evaluation, VLM vulnerability analysis, and component contribution analysis; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear, the three risk characteristics are argued persuasively, and the experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — The unified dataset + red-teaming framework + auditing model pipeline offers substantial practical value for safe VLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

<!-- RELATED:END -->
