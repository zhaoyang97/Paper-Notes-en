---
title: >-
  [Paper Note] SpiritSight Agent: Advanced GUI Agent with One Look
description: >-
  [CVPR 2025][LLM Agent][GUI Agent] This paper proposes SpiritSight, a vision-based end-to-end GUI agent, which resolves grounding ambiguity under dynamic high-resolution inputs through a multi-tier dataset of 5.73 million samples named GUI-Lasagne and the Universal Block Parsing (UBP) method. On Multimodal-Mind2Web under the non-candidate element setup, SpiritSight-8B achieves a Step Success Rate (SR) of 52.7%, outperforming all vision, language, and hybrid methods.
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "VLM"
  - "element grounding"
  - "multi-platform"
  - "dynamic resolution"
  - "UBP"
date: 2026-05-08
content_hash: 4ab1bb374db90665
---

# SpiritSight Agent: Advanced GUI Agent with One Look

**Conference**: CVPR 2025  
**arXiv**: [2503.03196](https://arxiv.org/abs/2503.03196)  
**Code**: [https://hzhiyuan.github.io/SpiritSight-Agent](https://hzhiyuan.github.io/SpiritSight-Agent)  
**Area**: LLM Agent  
**Keywords**: GUI Agent, VLM, element grounding, multi-platform, dynamic resolution, UBP

## TL;DR
This paper proposes SpiritSight, a vision-based end-to-end GUI agent, which resolves grounding ambiguity under dynamic high-resolution inputs through a multi-tier dataset of 5.73 million samples named GUI-Lasagne and the Universal Block Parsing (UBP) method. On Multimodal-Mind2Web under the non-candidate element setup, SpiritSight-8B achieves a Step Success Rate (SR) of 52.7%, outperforming all vision, language, and hybrid methods.

## Background & Motivation
**Background**: Existing GUI agent methods are categorized into text-based (HTML/XML input), hybrid vision-language (screenshot + HTML), and vision-only (screenshot only). While text-based and hybrid methods lead in accuracy, they suffer from platform limitations (HTML is web-only), security risks (injection attacks), and latency issues.

**Limitations of Prior Work**: Although vision-only methods are platform-compatible and have low latency, their **element grounding capabilities are severely lacking**—existing vision GUI agents cannot precisely locate small buttons, text fields, etc., on screens. Two-stage approaches (e.g., MobileAgent) introduce external OCR/icon recognition tools to assist in localization, which increases system complexity and latency.

**Key Challenge**: Modern VLMs (e.g., InternVL2) employ a dynamic high-resolution strategy, where the input image is sliced into $448 \times 448$ blocks based on the optimal aspect ratio before patch-concatenation input. However, in GUI scenarios, **the block-flattening operation discards 2D spatial information**. This causes the learned mapping $f: \mathbf{p}'' \to \mathbf{p}$ to be a multi-valued function (where block-normalized coordinates map to different global coordinates depending on grid indices), resulting in grounding ambiguity.

**Goal**: (1) Enhance the VLM's GUI perception and localization capability using a large-scale, high-quality GUI dataset. (2) Address the grounding ambiguity of dynamic resolution from an algorithmic level.

**Key Insight**: Data (GUI-Lasagne covering a three-tier training of perception $\to$ grounding $\to$ navigation) + Method (UBP replaces global coordinates with block-relative coordinates to eliminate multi-valued mapping).

**Core Idea**: Train VLM grounding capabilities using 5.73 million progressive multi-tier GUI samples, and apply UBP to transform the multi-valued mapping into an injective mapping to resolve ambiguity.

## Method

### Overall Architecture
SpiritSight is built upon InternVL2 (2B/8B/26B) and trained in two stages: (1) **Continual Pre-training**: trained on the full GUI-Lasagne dataset with visual encoders, decoders, and MLP layers unfrozen; (2) **Fine-tuning**: LoRA fine-tuning on specific downstream GUI navigation datasets. During inference, it takes the GUI screenshot, task description, and action history as input, and outputs the next action code. The policy is decomposed into three sub-policies: step reasoning $\pi_s$ (generating natural language descriptions), position reasoning $\pi_{pos}$ (locating action coordinates), and attribute reasoning $\pi_{attr}$ (determining the action type).

### Key Designs

1. **GUI-Lasagne Dataset (5.73 million samples)**:

    - **Level 1 - Visual-Text Alignment (3 million samples)**: Trains the model to recognize and locate textual/iconic elements in the GUI. Three tasks are included: text2bbox (locating text), bbox2text (recognizing text given a box), and bbox2dom (generating DOM trees for a given region). Web data is compiled of 755K webpage screenshots and DOM data collected from CommonCrawl and website rankings, while mobile data comes from AitW. An in-house InternVL-Icon (fine-tuned on a 30,000 Alibaba icon library based on InternVL1.5-26B) is utilized for icon annotation. Multiple data pairs are packed into the same training sample to fully leverage context length.
    - **Level 2 - Visual-Function Alignment (1.5 million samples)**: Trains the model to understand the functional semantics of GUI elements. For the function2bbox task (locating elements based on functional descriptions), a back-translation method is used: dividing the screenshot into a $3 \times 3$ grid to describe locations + highlight elements with bounding boxes $\to$ InternVL2-26B generates function descriptions $\to$ InternLM2.5-20B enhances diversity. Human verification shows a 90.9% acceptance rate at an extremely low cost.
    - **Level 3 - Visual GUI Navigation (640K samples)**: CoT-style navigation training. Based on the AitW dataset, GPT-4o is used for cleaning (TPR 93.7%), filtering down 1.48 million source samples to 630K. For each step, GPT-4o does the following: describes the current screenshot $\to$ compares and analyzes changes with the next step's screenshot $\to$ reasons about the rationality of the current action.
    - **Design Motivation**: The three levels of data form a progressive capability ladder: first learning "what is visible" $\to$ then "what can be done" $\to$ and finally "how to navigate". Levels 1 & 2 account for 90% of the dataset and can be collected for free or at very low cost, vastly reducing data construction expenses.

2. **Universal Block Parsing (UBP)**:

    - **Function**: Resolves the positioning ambiguity after dynamic high-resolution cropping.
    - **Mechanism**: Traditional methods train models to predict global coordinates $\mathbf{p}=[x,y]$. However, after block-flattening, the mapping $f: \mathbf{p}'' \to \mathbf{p}$ becomes multi-valued (e.g., $\mathbf{p}''=[1, 168, 245]$ can map to $[168, 693]$ or $[616, 245]$ depending on $n_w$). UBP instead trains the model to predict block-relative coordinates $\mathbf{p}'' = [b_i, x', y']$, converting the mapping into an injective one. During inference, global coordinates are restored via post-processing: $x = x' + (b_i \bmod n_w) \cdot w_{block}$, $y = y' + \lfloor b_i / n_w \rfloor \cdot h_{block}$.
    - Integrated with **2D Block-wise Position Embedding (2D-BPE)**: Adds row and column position embeddings for each block to preserve 2D spatial relationships. All coordinate values are normalized to the 0-999 range and rounded to integers.
    - **Design Motivation**: A rigorous solution based on theoretical analysis—transforming multi-valued mapping into injective mapping to fundamentally solve the ambiguity rather than temporarily "mitigating" it with extra thumbnail images. Experiments demonstrate that UBP primarily improves Ele.Accuracy rather than Op.F1, proving its direct contribution to enhancing element grounding capabilities.

### Loss & Training
- Pre-training: Learning rates are set to 1e-4/1e-4/5e-5 (for 2B/8B/26B), batch size is 1024, and full-parameter training is applied (completely unfreezing the vision encoder, decoder, and MLP).
- Fine-tuning: Full-parameter training for 1 epoch (using Level 3 + downstream training sets), followed by LoRA fine-tuning for 1 epoch (alpha is 32 for the vision encoder and 64 for the LLM decoder).

## Key Experimental Results

### Main Results: Multimodal-Mind2Web (Non-Candidate Element Setting)

| Method | Model Size | Input | Cross-Task SR | Cross-Website SR | Cross-Domain SR |
|------|---------|------|:---:|:---:|:---:|
| SeeAct | - | Text+Image | 40.2% | 32.4% | 36.8% |
| OmniParser | - | Image | 39.4% | 36.5% | 42.0% |
| SeeClick | 9.6B | Image | 25.5% | 16.4% | 20.8% |
| **SpiritSight-2B** | 2B | Image | 44.9% | 37.8% | 36.9% |
| **SpiritSight-8B** | 8B | Image | **52.7%** | **44.0%** | **44.4%** |
| **SpiritSight-26B** | 26B | Image | **54.7%** | **48.1%** | **49.2%** |

### Cross-Platform Navigation & ScreenSpot Functional Grounding

| Benchmark | Prev. SOTA | SpiritSight-8B |
|------|----------|:---:|
| GUI-Odyssey (AMS) | 74.3% | **75.8%** |
| AMEX (AMS) | 70.7% | **80.7%** |
| AndroidControl-High | 64.8% | **68.1%** |
| GUIAct-Multi (Step SR) | 45.4% | **49.3%** |
| ScreenSpot Web | 49.5% (CogAgent) | **68.3%** |
| ScreenSpot Mobile | 65.0% (SeeClick) | **68.4%** |
| ScreenSpot Desktop | 51.1% (SeeClick) | **62.9%** |

### Ablation Study

| Configuration | Mind2Web Step SR | Description |
|------|:---:|------|
| Level 1 only | ~36% | Base grounding already outperforms SeeClick |
| Level 1+2 | ~44% | Functional understanding significantly improved |
| Level 1+2+3 | ~52% | Navigation data provides further gains |
| w/o UBP (baseline) | Lower Ele.Acc | Grounding ambiguity leads to incorrect element selection |
| UBP + 2D-BPE | Best | Both are complementary, achieving optimal performance |

### Key Findings
- The vision-only method SpiritSight-8B outperforms all non-candidate element methods (including text-based and hybrid methods), challenging the common consensus that "vision-only is inferior to structured input."
- Level 1 data (collected for free) contributes the largest basic performance jump—Level 1 training alone outperforms SeeClick.
- UBP mainly acts by boosting Ele.Acc (with Op.F1 almost unchanged), directly proving that it resolves the grounding problem.
- SpiritSight-2B outperforms SeeClick using only 1/8 of the pre-training data, demonstrating the superior data quality of GUI-Lasagne.
- Cross-lingual experiments: Trained solely on English data, the model achieves 50% of the performance of English-Chinese joint training on Chinese tests, showing zero-shot cross-lingual capabilities.

## Highlights & Insights
- **Theoretical elegance of UBP**: Formalizing the grounding ambiguity problem as a transformation from a multi-valued function to an injective function yields a clean yet highly effective solution. This approach can be generalized to any spatial grounding task using dynamic resolution VLMs.
- **The power of "free data"**: Levels 1 & 2 require almost no manual annotation (automatic DOM scraping on the web + InternVL-generated functional descriptions) and account for 90% of the data volume. This data construction paradigm serves as a useful reference for other vertical domains (such as document understanding and chart analysis).
- **Three-tier progressive design**: The training hierarchy of perception $\to$ understanding $\to$ action aligns closely with how humans learn to use new software.

## Limitations & Future Work
- As a vision-only approach, screenshots are always required, presenting privacy and security risks (as screenshots may contain personal information).
- Level 3 navigation data is only sourced from mobile platforms (AitW), lacking web and desktop navigation training datasets.
- Automatic generation of functional descriptions (Level 2) has roughly a 9% error rate, potentially introducing noise.
- Multi-step task planning is not explored—currently, only single-step accuracy is optimized.
- Future work can explore combining Level 1-2 free data with reinforcement learning (e.g., DigiRL) to further improve navigation.

## Related Work & Insights
- **vs SeeClick**: As pioneering visual GUI agents, SpiritSight completely outperforms SeeClick through larger scale grounding training data (5.73M vs. ~1M) and the UBP method. SpiritSight-2B defeats SeeClick with only 1/8 of the data.
- **vs OmniParser**: OmniParser relies on extra OCR/icon detection tools to assist positioning, whereas SpiritSight learns grounding end-to-end, showing a simpler and more effective pipeline.
- **vs CogAgent**: CogAgent uses a specialized model with 18B parameters. SpiritSight-8B dramatically outperforms it with less than half the parameters, proving that data quality and UBP are more important than model scale.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical analysis of UBP is elegant, and the three-tier data design is comprehensive, although the overall contribution leans toward engineering optimization rather than a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks, 3 model sizes, thorough ablation and scaling analyses, and cross-lingual experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-verified data quality (human evaluation + TPR/TNR).
- Value: ⭐⭐⭐⭐⭐ Proves that vision-only GUI agents can fully outperform structured-input methods, exerting a significant impact on domain paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](../../ACL2025/llm_agent/os_genesis_gui_agent_trajectory.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](../../ACL2025/llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](../../ICCV2025/llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)

</div>

<!-- RELATED:END -->
