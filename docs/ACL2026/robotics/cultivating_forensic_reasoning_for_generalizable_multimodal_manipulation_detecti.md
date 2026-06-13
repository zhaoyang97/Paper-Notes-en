---
title: >-
  [Paper Note] Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection
description: >-
  [ACL2026][Robotics][Multimodal Forgery Detection] This paper proposes REFORM, which shifts multimodal forgery detection from "direct label fitting" to "learning a verifiable forensic reasoning process." Through the ROM r…
tags:
  - "ACL2026"
  - "Robotics"
  - "Multimodal Forgery Detection"
  - "Forensic Reasoning"
  - "GRPO"
  - "Forgery Localization"
  - "ROM Dataset"
date: 2026-05-08
content_hash: ccf636545eefe266
---

# Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection

**Conference**: ACL2026  
**arXiv**: [2603.01993](https://arxiv.org/abs/2603.01993)  
**Code**: https://github.com/YcZhangSing/REFORM  
**Area**: Multimodal Forensics / AIGC Detection  
**Keywords**: Multimodal Forgery Detection, Forensic Reasoning, GRPO, Forgery Localization, ROM Dataset

## TL;DR
This paper proposes REFORM, which shifts multimodal forgery detection from "direct label fitting" to "learning a verifiable forensic reasoning process." Through the ROM reasoning-annotated dataset, a dual-decoder architecture, and GRPO training, it achieves superior cross-domain generalization and interpretable detection results on ROM, DGM4, and MMFakeBench.

## Background & Motivation
**Background**: Multimodal media forgery has expanded from local facial editing to complex compositional forgeries involving entire news images, backgrounds, headlines, and main bodies. Existing methods like the DGM4 series, knowledge-enhanced approaches, and vision-language models typically model the task as detection, classification, or localization—taking image-text news as input and outputting authenticity, forgery types, and regions.

**Limitations of Prior Work**: Mainstream methods mostly rely on result-oriented supervision, requiring the model to map training samples directly to final labels. While effective on closed-set data, this approach leads models to memorize specific statistical artifacts (e.g., textures of certain generative models, linguistic distributions of specific news domains, or editing patterns) rather than learning "why there is an inconsistency." Consequently, detectors often fail when encountering new test domains, generators, or forgery methods.

**Key Challenge**: Multimodal forensics truly requires a transferable logical evidence chain, yet training signals are often limited to the final answer. Label supervision tells the model "this is fake" but rarely constrains it to find credible visual evidence, textual evidence, or contradictions between the two.

**Goal**: The authors aim to solve three sub-problems: constructing a benchmark with broader coverage and reasoning annotations; enabling the model to explicitly generate forensic rationales while maintaining consistency between rationales and answers; and using reinforcement learning after SFT to constrain the format, accuracy, localization, and consistency of the reasoning chain.

**Key Insight**: The core observation is that generalization should not stem only from larger vision-language models or more external knowledge, but from the optimization of the "forensic thinking process." By rewarding correct, coherent, and localizable reasoning chains in training, the model is more likely to capture cross-domain stable forgery logic.

**Core Idea**: Replace pure result fitting with reasoning-driven optimization, teaching the detector to explain forgery evidence first, then using consistency losses and GRPO to bind explanation, classification, and localization together.

## Method
The contribution of REFORM consists of three parts: data, architecture, and training. On the data side, the authors built ROM, providing models with reasoning annotations alongside image-text pairs and labels. Architecturally, the model employs a Cognitive Priming Encoder and a Reason-Answer Dual-Decoder. For training, it uses reasoning warm-up, joint fine-tuning of answers/rationales, and policy refinement via GRPO.

### Overall Architecture
The input is a multimodal news sample, including the image, text prompts, and the content to be judged. The model encodes the image into visual tokens and the instructions into text tokens. Then, through a frozen Cognitive Priming Encoder, a set of learnable reason tokens extracts forgery clues from the visual and textual contexts. After multimodal encoding, the model connects to two parallel decoders: the Answer Decoder outputs the veracity, forgery type, and localization coordinates, while the Reason Decoder outputs explanatory forensic reasoning.

The training process involves three stages. The first stage trains only the reasoning branch, aligning reason tokens and the Reason Decoder with distilled forensic rationales. The second stage unfreezes the entire model to generate rationales and answers simultaneously, incorporating a rationale-answer consistency constraint. The third stage uses GRPO to let the model learn paths from multiple candidate rationales that best satisfy formatting, accuracy, localization, and semantic consistency rewards.

### Key Designs
1. **ROM Reasoning-Augmented Dataset**:
    - **Function**: Provides broader scene-level data and reasoning supervision for multimodal forgery detection.
    - **Mechanism**: ROM extends beyond the face-related categories of MDSM to include BackgroundReplacement, FullGeneration, and scene-level forgeries combined with TextFabrication. It comprises 704,456 image-text pairs across 5 news domains and 9 forgery categories, with textual reasoning distilled from InternVL3.5-30B for each sample.
    - **Design Motivation**: Traditional DGM4 focuses on face editing, which may lead models to learn local artifacts. ROM expands the scope to full-image generation and background replacement, providing rationales (~130 tokens peak length) that offer richer supervision of the forensic process than short answers.

2. **Cognitive Priming and Dual-Decoder**:
    - **Function**: Separates "finding evidence" and "providing answers" into two related but non-interfering generation tasks.
    - **Mechanism**: The Cognitive Priming Encoder processes $S_{inp}=[T_i;T_r;T_t]$, keeping only updated reason tokens $\hat{T}_r$. The multimodal encoder then reads $S_p=[T_i;\hat{T}_r;T_t]$. The Answer Decoder outputs structured predictions, while the Reason Decoder outputs forensic explanations.
    - **Design Motivation**: Sharing a single decoder can cause gradient conflicts between answer and rationale generation. Dual decoders allow separate optimization and switching between Reasoning Mode and Fast Mode (skipping rationale generation while maintaining prediction).

3. **Three-Stage Reasoning-Driven Training**:
    - **Function**: Transitions the model from "stating rationales" to "rationales supporting answers" and finally to "actively exploring reliable reasoning."
    - **Mechanism**: Stage 1 uses a rationale language modeling loss $\mathcal{L}_{LM_r}$. Stage 2 adds an answer loss $\mathcal{L}_{LM_a}$ and a Rationale-Answer Consistency loss $\mathcal{L}_{RAC}=\max\{0,\eta-\cos(\mathbf{v}^R,\mathbf{v}^A)\}$, with the objective $\mathcal{L}_{RJF}=\mathcal{L}_{LM_r}+\mathcal{L}_{LM_a}+\mathcal{L}_{RAC}$. Stage 3 uses GRPO with multi-dimensional rewards for format, classification accuracy, localization quality, and consistency.
    - **Design Motivation**: SFT alone only mimics annotations and is prone to exposure bias. GRPO allows the model to compare candidate rationales, rewarding chains that are supported by the verifier and consistent with the final answer.

### Loss & Training
The focus of the training strategy is not a simple classification head but the integration of the reasoning chain into the optimization objective. During the Reasoning Warm-up phase, the multimodal encoder and Answer Decoder are frozen. In Joint Fine-Tuning, both are optimized with $\mathcal{L}_{RAC}$ to prevent semantic breakage where the "rationale suggests A, but the answer determines B." During Policy Refinement via GRPO, the Consistency Verifier (TinyBERT with classification heads) determines if generated rationales logically lead to the model's predicted forgery type based on pre-trained rationale-label pairs.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | REFORM | Baseline Comparison | Interpretation |
|--------|------|------|----------|------|
| ROM Cross-Domain | AVG ACC | 88.22 | AMD 85.92 / HAMMER 72.41 / MMD-Agent-34B 57.45 | Significantly outperforms feature alignment, traditional detection, and retrieval-agent pipelines in new domains. |
| ROM Guardian Test Domain | ACC / mAP / mIoU | 81.52 / 67.75 / 81.64 | Specific REFORM values provided in cached table. | Maintains high detection and localization quality in out-of-domain tests. |
| MMFakeBench Zero-Shot | F1 | 74.9 | Multiple 7B/13B LVLM baselines | Strong zero-shot generalization on unseen types (e.g., manual PS) via forensic reasoning. |
| DGM4 | ACC / AVG mAP | 76.65 / 65.72 | Fine-tuned LVLMs: mAP < 47 | Outperforms specialized detectors even on face-centric DGM4. |
| Efficiency | Params / Throughput | 376M / Fast Mode 13.17 pairs/s | FKA-Owl 6.7B, MMD-Agent 34B | Dual-decoder separates explanation/screening; much smaller than LLM agents. |

### Ablation Study
| Configuration | NYT ACC | NYT mAP | NYT mIoU | Guardian ACC | Guardian mAP | Guardian mIoU | Notes |
|------|---------|---------|----------|--------------|--------------|---------------|------|
| $\mathcal{L}_{LM_a}$ | 84.88 | 66.16 | 75.98 | 72.18 | 45.86 | 78.72 | Answer only; result-oriented. |
| $\mathcal{L}_{LM_a}+\mathcal{L}_{LM_r}$ | 87.76 | 73.01 | 77.68 | 74.74 | 53.65 | 79.59 | Reasoning supervision boosts detection/localization. |
| + $\mathcal{L}_{RAC}$ | 87.84 | 73.25 | 78.00 | 75.71 | 54.11 | 79.58 | Consistency adds further gains. |
| + GRPO | 88.22 | 76.08 | 78.48 | 81.52 | 67.75 | 81.64 | RL contributes most, especially to Guardian mAP. |

### Key Findings
- The reasoning branch is not just decorative. Adding $\mathcal{L}_{LM_r}$ alone increased NYT ACC from 84.88 to 87.76.
- GRPO is crucial for cross-domain generalization. The full model on Guardian improved from 75.71 ACC (SFT+RAC) to 81.52 ACC.
- There is a "sweet spot" for reason token length. 32 tokens achieved optimal ACC (88.22); too short loses detail, too long adds generative overhead.
- Teacher quality is not the sole source of performance. Replacing InternVL3.5-30B with Qwen2.5-VL-3B only reduced Guardian ACC by 0.84.
- Explanation comes at a cost. Explainable Mode reaches 1.03 pairs/s compared to Fast Mode's 13.17 pairs/s, though Fast Mode loses no prediction accuracy.

## Highlights & Insights
- The most valuable design is turning interpretability from "post-hoc display" into a "training constraint."
- The importance of ROM lies in its category boundaries being closer to the real forgery ecosystem (background/full-gen/text-mod), forcing the model to focus on multimodal logic rather than just facial textures.
- The dual-decoder is a practical engineering compromise, allowing for both "explanation" and "real-time screening" sessions.
- Using the TinyBERT verifier provides a computable consistency signal for GRPO without turning rationale generation into an uncontrollable long-text reward problem.

## Limitations & Future Work
- REFORM relies on distilled rationales. While human audits show rationales recall ~83% of visual/textual evidence, teacher hallucinations or templated explanations might still propagate to the student.
- High latency in Explanation Mode (1.03 pairs/s) is suited for auditing but not all real-time scenarios; non-autoregressive generation or two-stage deployment could be explored.
- ROM has dual-use risks. The generate pipeline and detailed prompts are not public for ethical control, which impacts external replicability.
- Forensic rationales are currently textual; future work could combine them with visual evidence trajectories or counterfactual edits.

## Related Work & Insights
- **vs HAMMER / HAMMER++**: HAMMER emphasizes feature alignment. REFORM treats the forensic reasoning chain as an optimizable object, leading to stronger cross-domain metrics.
- **vs FKA-Owl**: FKA-Owl uses knowledge augmentation. REFORM internalizes stable judgment logic through reasoning training without needing an external retrieval agent.
- **vs AMD**: AMD introduces Manipulation-Oriented Reasoning. REFORM goes further with rationale-answer consistency and GRPO strategy refinement, outperforming AMD's 85.92 ACC on ROM.
- **vs MMD-Agent**: MMD-Agent uses multi-step agents with high overhead. REFORM's 376M model achieves stronger ROM performance, suggesting that "learning reasoning at training time" can replace "constructing agents at test time."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Specifically reformulates multimodal forgery detection as reasoning-driven optimization with a complete RL loop.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers ROM, MMFakeBench, DGM4, ablations, efficiency, and audit of rationale credibility.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear main line and comprehensive tables, though some appendix details and dense formulas/labels are heavy.
- **Value**: ⭐⭐⭐⭐⭐ Directly insightful for AIGC forensics and small-model generalization via verifiable reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](../../ICLR2026/robotics/vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](../../CVPR2026/robotics/maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](../../NeurIPS2025/robotics/safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[ICML 2026\] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](../../ICML2026/robotics/decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)

</div>

<!-- RELATED:END -->
