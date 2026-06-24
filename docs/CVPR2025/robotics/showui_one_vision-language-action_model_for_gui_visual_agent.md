---
title: >-
  [Paper Note] ShowUI: One Vision-Language-Action Model for GUI Visual Agent
description: >-
  [CVPR 2025][Robotics][GUI Visual Agent] Based on Qwen2-VL-2B, ShowUI reduces redundant tokens by 33% and achieves a 1.4x speedup through UI-connected-graph-guided visual token selection. Combined with interleaved vision-language-action streaming and a curated 256K training dataset, it achieves state-of-the-art (SOTA) zero-shot accuracy of 75.1% on ScreenSpot with only 2B parameters.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "GUI Visual Agent"
  - "Visual Token Selection"
  - "Interleaved VLA Streaming"
  - "UI Connected Graph"
  - "Lightweight Model"
date: 2026-05-08
content_hash: 2b5a7e0d7b0831b1
---

# ShowUI: One Vision-Language-Action Model for GUI Visual Agent

**Conference**: CVPR 2025  
**arXiv**: [2411.17465](https://arxiv.org/abs/2411.17465)  
**Code**: [https://github.com/showlab/ShowUI](https://github.com/showlab/ShowUI) (Open Source)  
**Area**: Robotics  
**Keywords**: GUI Visual Agent, Visual Token Selection, Interleaved VLA Streaming, UI Connected Graph, Lightweight Model

## TL;DR
Based on Qwen2-VL-2B, ShowUI reduces redundant tokens by 33% and achieves a 1.4x speedup through UI-connected-graph-guided visual token selection. Combined with interleaved vision-language-action streaming and a curated 256K training dataset, it achieves state-of-the-art (SOTA) zero-shot accuracy of 75.1% on ScreenSpot with only 2B parameters.

## Background & Motivation

**Background**: Current GUI automation is primarily divided into two paradigms: language-based agents that rely on closed-source APIs like GPT-4 along with HTML/DOM tree metadata, and vision-based agents that directly process screenshots but require training large models. Vision-based agent models such as CogAgent (18B) and SeeClick (9.6B) are extremely large and demand massive training data.

**Limitations of Prior Work**: (1) UI screenshots have extremely high resolutions (e.g., 2K), yielding a large number of visual tokens (1000+), which leads to computationally expensive self-attention and substantial redundancy (white spaces, solid color backgrounds); (2) Actions in GUI tasks differ from natural language—different devices have varying action spaces (e.g., CLICK on Web, PRESS HOME on Mobile), and how to unify their modeling remains unclear; (3) Navigation tasks require managing historical sequences of screenshots and actions (interleaved multimodal sequences), a format that existing VLMs are not adept at; (4) GUI data types are diverse and imbalanced, posing challenges in selecting high-quality subsets.

**Key Challenge**: UI screenshots exhibit significant visual redundancy (solid color backgrounds, blank areas) while containing extremely critical small elements (buttons, icons). Thus, it is necessary to preserve the grounding accuracy of key details while substantially compressing tokens.

**Goal**: How to build an efficient and precise GUI visual agent with extremely low parameter sizes and data volume?

**Key Insight**: The fundamental difference between UI screenshots and natural images lies in "structured redundancy"—large solid-color regions correspond to backgrounds/white spaces, whereas regions with color transitions represent functional elements. Therefore, patch-level connected graphs can be constructed in the RGB space to identify redundancy and guide token pruning.

**Core Idea**: Construct a connected graph using the RGB color similarity of UI screenshots to identify redundant patches, sparsely sample tokens based on connected components within self-attention, and integrate them with interleaved VLA streaming to unify navigation and grounding tasks.

## Method

### Overall Architecture
ShowUI is built on Qwen2-VL-2B. The input consists of user queries, action space specifications (represented in a JSON-formatted README), and screenshot observations; the output is a JSON-formatted action prediction (action type + coordinate/text parameters). The training data consists of only 256K samples, covering grounding and navigation data across three device categories: Web, Mobile, and Desktop. Inference is executed in a recurrent loop: predict action $\rightarrow$ update screenshot $\rightarrow$ predict next action.

### Key Designs

1. **UI-Guided Visual Token Selection**:

    - **Function**: Adaptively prune tokens based on the visual redundancy of UI screenshots, reducing computation while preserving key elements.
    - **Mechanism**: Partition the screenshot into patches, where each patch serves as a graph node. If the RGB difference between adjacent patches is less than a threshold $\delta$, they are merged into a single connected component using Union-Find. Large solid-color areas form large components (high redundancy), while text/icon areas form smaller components (low redundancy). During training, tokens are randomly sampled within each connected component according to a preset ratio (e.g., 50%), and the sampled tokens retain their original positional encodings. Resulting effect: Google Search page drops from 1296 to 291 tokens, Overleaf document drops from 1296 to 986 tokens, adaptively matching content complexity.
    - **Design Motivation**: Unlike Token Merging (which pools all tokens in a component into one, causing loss of position information and subsequent failure in grounding tasks), Token Selection retains the original position of the selected tokens, thereby preserving position awareness in self-attention. It requires zero additional parameters, is used during training, and can be optionally enabled/disabled during inference.

2. **Interleaved VLA Streaming**:

    - **Function**: Unify the handling of navigation history and single-frame multi-query scenarios, improving training efficiency and navigation performance.
    - **Mechanism**: Design two streaming modes—(a) *Action-Visual Streaming* for navigation: screenshot 1 $\rightarrow$ action 1 $\rightarrow$ screenshot 2 $\rightarrow$ action 2 $\rightarrow$ ..., allowing the model to reason about the next step with complete visual-action history; (b) *Action-Query Streaming* for grounding: a single screenshot paired with multiple query-action pairs for multi-turn dialogue, avoiding redundant encoding of long screenshots for each query. Actions are unified into a JSON format (`{'action': type, 'value': element, 'position': [x, y]}`), and the system prompt provides a README document of the action space.
    - **Design Motivation**: Screenshot tokens typically exceed 1000, whereas query/action tokens are usually fewer than 10. Associating only one action with one image is highly wasteful; multi-turn dialogues allow reuse of screenshot encodings. For device-specific disparities in action spaces (e.g., Web lacks TAP, Mobile lacks CLICK), a README document is provided to guide the model to "read rules" rather than purely "memorizing actions", supporting zero-shot generalization to new action types during testing.

3. **Curated Small-Scale High-Quality Training Data**:

    - **Function**: Achieve performance comparable to models trained on millions of samples using only 256K training instances.
    - **Mechanism**: For Web data, 40% of "static text" labels are filtered out (since the VLM's inherent OCR capability is already powerful), and only visual interactive elements like buttons and checkboxes are kept (22K screenshots); Desktop data is extremely scarce (only 100 screenshots), so GPT-4o is used to reverse-engineer original annotations and generate three query types—appearance, spatial, and intent—to augment the data (100 to 6K elements); AMEX's functional descriptions are introduced for Mobile. Re-sampling strategies are employed during training to balance different data types.
    - **Design Motivation**: Data quality matters more than quantity—filtering low-value data, augmenting scarce classes, and balanced sampling are more effective than brute-force data stacking.

### Loss & Training
Standard autoregressive next-token prediction loss is used to predict the JSON-formatted action outputs. The Token Selection ratio is set to 0.5, inserted in a cross-layer configuration (every other layer), and the learning rate matches the Qwen2-VL fine-tuning configuration.

## Key Experimental Results

### Main Results

| Benchmark | Metric | ShowUI (2B, 256K) | Prev. SOTA | Description |
|--------|------|------|----------|------|
| ScreenSpot Zero-shot | Average Accuracy | **75.1%** | 73.3% (UGround-7B, 1.3M) | 3.5x smaller, 5x less data |
| ScreenSpot Mobile Icon | Accuracy | **75.5%** | 60.3% (UGround) | +15.2% |
| AITW Mobile | Success Rate | **70.0%** | 67.2% (Qwen2-VL-2B baseline) | +2.8% |
| Mind2Web Cross-Domain | Step SR | **35.2%** | 30.7% (Qwen2-VL-2B baseline) | +4.5% |
| MiniWob Online | Score | **71.5** | 67.0 (SeeClick-9.6B) | +4.5 |

### Ablation Study

| Configuration | Visual Tokens | Training Speedup | ScreenSpot |
|------|---------|---------|---------|
| Baseline (No Compression) | 1344.0 | 1× | 70.8 |
| Token Merge (UI-Graph) | 852.8 | 1.6× | 42.3 |
| Token Selection (Random) | 941.5 | 1.5× | 65.3 |
| **Token Selection (UI-Graph)** | **947.4** | **1.5×** | **70.4** |

### Key Findings
- **Token Merging severely degrades grounding performance** (from 70.8% to 42.3%) since pooling discards positional information; in contrast, Token Selection remains almost lossless (70.8% to 70.4%) by retaining positional encodings, proving that position preservation is crucial for GUI grounding.
- **Cross-layer insertion is significantly superior to applying it to all layers, early layers, or late layers**: across the same 14 layers, cross-layer achieves 70.5% vs. early-layer (68.2%) vs. late-layer (67.6%), indicating the necessity of alternating between full information and compressed information throughout the model depth.
- **Visual history is critical for Mobile navigation (+1.7%) but offers limited benefits for Web navigation**: this is because switching between mobile apps results in large visual changes, whereas operating within the same web page yields minor visual variations in screenshots.
- **Text grounding capability generalizes across platforms, whereas icon grounding does not**: across all methods, Text accuracy $\gg$ Icon accuracy, and icon grounding on Desktop/Web is noticeably weaker than on Mobile, suggesting a need for more visually diverse cross-platform training data.

## Highlights & Insights
- **The RGB connected graph is an exceptionally simple and efficient UI understanding prior**: without any learnable parameters, it distinguishes between "informative" and "non-informative" regions solely using pixel colors. This approach can be applied to any visual understanding tasks involving structured documents or UIs.
- **The insight that Token Selection > Token Merging is highly valuable**: sparse sampling that retains positional encodings performs far better than pooling that loses position, which offers inspiration for all visual token compression methods.
- **Describing the action space with a README document** instead of hardcoding the action set empowers the model with function-calling capabilities to handle new action types, which is highly elegant.
- **Achieving superiority with only 2B parameters and 256K samples against 7B parameters and 1.3M samples** demonstrates that data quality and model design are more crucial than brute-force scaling of parameters and data volume.

## Limitations & Future Work
- Zero-shot performance in online environments (MiniWob) suffers from a significant gap compared to fine-tuning (27.1% vs 71.5%), indicating that offline training is insufficient to handle Out-Of-Distribution (OOD) errors, which necessitates online learning strategies.
- Desktop training data is extremely sparse (only 100 screenshots), leading to lower grounding accuracy for Desktop icons (61.1%), which highlights the need for more cross-platform vision-centric training data.
- The language reasoning capacity of the 2B model is limited, which may lead to insufficient capabilities in complex multi-step planning.
- Cross-website/cross-domain generalization on Mind2Web still has room for improvement, where the bottleneck lies in visual perception rather than textual understanding.

## Related Work & Insights
- **vs CogAgent (18B, 400K)**: ShowUI dramatically outperforms CogAgent on ScreenSpot (75.1% vs. 47.4%) with 9x fewer parameters and 1.6x less data, proving that UI-specific designs are more effective than simply scaling up the model.
- **vs UGround (7B, 1.3M)**: ShowUI achieves comparable accuracy (75.1% vs. 73.3%) despite being 3.5x smaller and requiring 5x less data, showcasing the distinct efficiency advantages of token selection.
- **vs OmniParser**: OmniParser relies on GPT-4V for inference, whereas ShowUI is a standalone end-to-end model, making it more practical.
- **vs Magma**: Magma utilizes SoM annotations combined with large-scale pre-training, whereas ShowUI adopts a lightweight design with curated data; their philosophies are highly complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The UI connected-graph token selection is an ingenious UI-specific design, and the interleaved VLA streaming is also novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers all four environments (Web, Mobile, Desktop, and Online) with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with excellent problem-driven experimental analysis.
- Value: ⭐⭐⭐⭐ An open-source, lightweight GUI agent with strong practicality; its efficiency optimization concepts are widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](../../CVPR2026/robotics/mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)
- [\[CVPR 2026\] Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight](../../CVPR2026/robotics/mantis_a_versatile_vision-language-action_model_with_disentangled_visual_foresig.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

</div>

<!-- RELATED:END -->
