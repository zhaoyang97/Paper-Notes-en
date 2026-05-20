---
title: >-
  [Paper Note] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding
description: >-
  [CVPR 2026][LLM Agent][GUI Grounding] This work presents the first systematic study of discrete vision-language diffusion models (DVLMs) for GUI grounding…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "GUI Grounding"
  - "Discrete Diffusion Models"
  - "LLaDA-V"
  - "Hybrid Masking"
  - "Interface Understanding"
date: 2026-05-08
content_hash: e0906251b7d7838c
---

# Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding

**Conference**: CVPR 2026
**arXiv**: [2603.26211](https://arxiv.org/abs/2603.26211)  
**Code**: None  
**Area**: GUI Agent / Vision-Language Models
**Keywords**: GUI Grounding, Discrete Diffusion Models, LLaDA-V, Hybrid Masking, Interface Understanding

## TL;DR

This work presents the first systematic study of discrete vision-language diffusion models (DVLMs) for GUI grounding, adapting LLaDA-V for single-step action prediction and proposing a hybrid masking schedule (linear + deterministic) to capture geometric hierarchical dependencies among bounding box coordinates. The approach demonstrates the feasibility of diffusion models as a foundation for GUI agents across Web, Desktop, and Mobile interfaces.

## Background & Motivation

GUI grounding is a fundamental capability for building multimodal GUI agents: given a natural language instruction and a screenshot, the model must localize the target element and generate the corresponding action. This is essential for automating software operation and digital workflow.

**Limitations of Prior Work**:
- Autoregressive (AR) vision-language models (e.g., Qwen2.5-VL, CogAgent, UI-TARS) dominate GUI grounding research.
- AR models inherit inherent architectural limitations: **sequential decoding** and **unidirectional attention**.
- These limitations prevent the model from leveraging subsequent context when generating coordinate tokens.

**Potential of Discrete Diffusion Models**:
- Discrete diffusion VLMs such as LLaDA-V and MMaDA have demonstrated strong performance in multimodal understanding and reasoning.
- DVLMs offer three distinctive advantages: **bidirectional attention**, **parallel token generation**, and **iterative refinement**.
- However, their potential for GUI grounding remains entirely unexplored.

**Core Problem**:
GUI grounding outputs are structured action strings (e.g., `lclick [42,180,120,250]`), containing an action type and bounding box coordinates $B = (x_1, y_1, x_2, y_2)$, where $(x_1, y_1)$ is the action anchor and $(x_2, y_2)$ defines the spatial extent, exhibiting geometric hierarchical dependencies. LLaDA-V's default linear masking schedule corrupts all tokens randomly, potentially undermining the model's ability to learn these consistent geometric dependencies.

## Method

### Overall Architecture

Built upon LLaDA-V (8B):
- Language tower: LLaDA (discrete diffusion language model)
- Vision tower: SigLIP-2
- Two-layer MLP projector to align visual embeddings into the language token space

Input: GUI screenshot + natural language instruction
Output: Action string (e.g., `lclick [42,180,120,250]` or `type_in [50,90,200,130] hello`)

### Key Designs

1. **Adapting LLaDA-V for GUI Grounding**:

    - GUI grounding is framed as a **text generation task**: given an image and instruction, generate the action type and bounding box coordinates.
    - Training objective: reconstruct masked action tokens.
    - $L(\theta) = -\mathbb{E}[\frac{1}{t} \sum_i \mathbb{1}[r_t^{1,i}=[M]] \times \log p_\theta(r_0^{1,i} | v, p_0^1, r_t^1)]$
    - At inference: starting from a fully masked sequence, iteratively denoise via the reverse diffusion process using a low-confidence remasking strategy.
    - Design Motivation: leverage the transfer capability from LLaDA-V's three-stage pretraining (vision-language alignment, instruction fine-tuning, reasoning enhancement).

2. **Hybrid Masking Schedule**:

    - **Linear Masking Phase**:
        - Retains LLaDA-V's standard schedule with masking probability $p_{mask} = (1-\varepsilon)t + \varepsilon$.
        - Responsible for learning coarse-grained grounding: predicting action type and anchor coordinates $(x_1, y_1)$.
    - **Full Deterministic Masking Phase**:
        - All response tokens are fully masked.
        - Conditioned on image $I$, instruction $N$, and anchor $(x_1, y_1)$, predicts the remaining coordinates $(x_2, y_2)$.
        - Reinforces the model to learn the conditional probability $p_\theta(x_2, y_2 | a_{type}, x_1, y_1, I, N)$.
    - Design Motivation: The randomness of linear masking rarely produces configurations where the anchor is visible while the extent is masked. The deterministic phase enforces this conditional relationship, simulating a coarse-to-fine refinement process.

3. **Data Scaling Strategy**:

    - Initial feasibility validation with 7K Mind2Web samples.
    - Scaled to 120K multi-domain data: Mind2Web (20K) + WebLinX (20K) + OS-Atlas (60K, covering Web/Mobile/Desktop) + Rico Widget Caption (20K).
    - Random cropping applied to large screenshots (ensuring target element visibility).
    - OCR text-associated annotations used in place of purely icon-level annotations.

4. **Inference Parameter Analysis**:

    - Three key parameters: diffusion steps, generation length, block length.
    - Setting all to 64 achieves the best accuracy–latency trade-off.
    - Beyond this threshold, accuracy plateaus while latency continues to increase.

### Loss & Training

- Training objective: masked language modeling objective for discrete diffusion.
- Initial experiments: 7K Mind2Web samples, 10 epochs.
- Large-scale experiments: 120K samples, mixed multi-domain data.
- Hybrid masking: two phases trained separately (linear + full deterministic).
- Evaluation metrics:
    - **Action-Type F1**: F1 score for action type classification.
    - **Step Success Rate (SSR)**: proportion of predictions where the predicted bounding box center falls within the ground-truth box.

## Key Experimental Results

### Main Results

AR vs. NAR (non-autoregressive) GUI grounding comparison (120K training data):

| Dataset | Metric | Phi (3B) | Qwen2.5-VL (3B) | Qwen2.5-VL (7B) | LLaDA-V (Linear) | LLaDA-V (Hybrid, Ours) |
|--------|------|----------|------------------|------------------|----------------|---------------------|
| Mind2Web | SSR (%) | 56.8 | 79.3 | 81.9 | 82.4 | **83.9** |
|  | F1 (%) | 94.4 | 99.6 | 99.9 | 98.5 | **100.0** |
| ScreenSpot-Web-Icon | SSR (%) | 62.6 | 79.1 | 85.4 | 57.8 | 63.1 |
| ScreenSpot-Web-Text | SSR (%) | 77.0 | 83.0 | 83.0 | 73.5 | 74.8 |
| VisualWebArena | SSR (%) | 68.5 | 88.9 | 87.2 | 61.4 | **67.5** |

SSR improvement from hybrid vs. linear masking:
- Mind2Web: +1.6
- ScreenSpot-Web-Icon: +5.3
- ScreenSpot-Web-Text: +1.3
- VisualWebArena: **+6.1**

### Ablation Study

**Effect of inference parameters** (Mind2Web 7K):

| Diffusion Steps | Gen. Length | Block Length | Convergence Steps | SSR (%) | Latency (s) |
|----------|----------|--------|----------|---------|---------|
| 32 | 32 | 32 | 13 | 78.15 | 2.56 |
| 64 | 64 | 64 | 25 | 80.67 | 4.84 |
| 128 | 128 | 128 | 25 | 80.63 | 5.01 |

**Effect of cropping + OCR annotation** (Mind2Web 7K):

| Configuration | SSR (%) | Latency (s) | Notes |
|------|---------|---------|------|
| Original screenshot | 80.67 | 4.84 | Baseline |
| Cropping + OCR annotation | **83.31** | **4.46** | +2.68 SSR, −0.38s |

**Effect of data scaling** (linear masking):

| Dataset | SSR @ 7K | SSR @ 120K | Gain |
|--------|-----------|-------------|------|
| ScreenSpot-Web-Text | 54.4 | 73.5 | +19.1 |
| ScreenSpot-Web-Icon | 19.9 | 57.8 | +37.9 |
| VisualWebArena | 32.4 | 61.4 | +29.0 |

### Key Findings

1. **DVLMs are capable of GUI grounding**: Even LLaDA-V fine-tuned on only 7K samples achieves 80.67% SSR on Mind2Web, demonstrating that diffusion models can perform spatial localization.
2. **Hybrid masking consistently improves accuracy**: SSR improves by 1.3–6.1 points across all 4 benchmarks, validating the effectiveness of explicitly modeling anchor–extent conditional dependencies.
3. **Data scaling yields substantial gains**: 120K multi-domain data improves SSR by 20+ points on average, while reducing latency by 1–1.5s and convergence steps by 8–9.
4. **A gap with AR models remains**: LLaDA-V (8B) trails Qwen2.5-VL (7B) by approximately 15–20 points on ScreenSpot and VWA; however, given the large disparity in pretraining data, this gap is expected.
5. **Latency is the primary bottleneck**: Hybrid masking introduces additional latency (3–6.5s vs. ~1.1s for AR) due to two-stage sequential inference.

## Highlights & Insights

- **First exploration of diffusion models for GUI grounding**: Fills a gap in DVLM applicability to this important task, and finds that bidirectional attention and iterative refinement do benefit coordinate prediction.
- **Elegant coarse-to-fine design in hybrid masking**: The geometric hierarchy of bounding boxes (anchor → extent) is encoded directly into the masking schedule, representing a principled way to inject domain priors into the diffusion process.
- **Unexpected efficiency gains from data scaling**: More data not only improves accuracy but also reduces convergence steps and latency, suggesting that better priors accelerate the denoising process.
- **Honest presentation of the gap with AR models**: The paper does not shy away from DVLM's disadvantages in pretraining scale and latency, positioning this as exploratory research rather than claiming comprehensive superiority.

## Limitations & Future Work

1. **Severe latency issues**: Multi-step denoising results in latency 3–6× higher than AR models, making real-time interaction scenarios impractical.
2. **Single-step actions only**: The current approach handles only single-step grounding; multi-step planning and action-dependent sequences are left for future work.
3. **Unequal pretraining data**: LLaDA-V's pretraining data is far smaller than Qwen2.5-VL's; the performance gap may narrow with more extensive pretraining.
4. **Two-stage dependency of hybrid masking increases complexity**: The linear phase output serves as input to the deterministic phase, introducing additional sequential computation.
5. **Overly simplified action types**: Only lclick/hover/type_in are supported; real-world GUI operations are more diverse (scrolling, dragging, etc.).
6. **Lack of comparison with recent GUI agent systems**: No end-to-end comparison with complete systems such as UI-TARS or OS-Atlas.

## Related Work & Insights

- **LLaDA-V / MMaDA**: Foundational architectures for discrete diffusion VLMs, demonstrating the viability of the diffusion paradigm for multimodal understanding.
- **CogAgent / UI-TARS**: Representative AR approaches achieving strong GUI grounding through large-scale pretraining and instruction fine-tuning.
- **SeeClick / UGround**: Methods for GUI pretraining and synthetic data augmentation, providing data strategy references for DVLM training.
- **D3PM**: Theoretical foundation for discrete diffusion using uniform category transitions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First application of DVLMs to GUI grounding; hybrid masking design is creative, though the work adapts an existing model)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (4 benchmarks, inference parameter ablations, data scaling analysis; lacks comparison with more GUI agents)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear positioning, honest presentation of strengths and limitations; some table formatting is slightly cluttered)
- **Value**: ⭐⭐⭐⭐ (Opens a new direction for GUI agents via diffusion models, though practical utility is constrained by latency and performance gaps)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](../../AAAI2026/llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)

</div>

<!-- RELATED:END -->
