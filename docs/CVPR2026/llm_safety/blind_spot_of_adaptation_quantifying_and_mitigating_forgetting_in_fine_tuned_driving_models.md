---
title: >-
  [Paper Note] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models
description: >-
  [CVPR 2026][LLM Safety][catastrophic forgetting] This work systematically investigates catastrophic forgetting when fine-tuning VLMs for autonomous driving scenarios, constructs the large-scale 180K-scene benchmark FidelityDrivingBench, and proposes the Drive Expert Adapter (DEA), which enhances driving task performance via prompt-space routing without corrupting base model parameters.
tags:
  - CVPR 2026
  - LLM Safety
  - catastrophic forgetting
  - VLM
  - autonomous driving
  - benchmark
  - expert adapter
date: 2026-05-08
content_hash: a82970ee4d07d18e
---

# The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models

**Conference**: CVPR 2026
**arXiv**: [2604.04857](https://arxiv.org/abs/2604.04857)
**Code**: [FidelityDrivingBench](https://github.com/FidelityDrivingBench)
**Area**: LLM Safety
**Keywords**: catastrophic forgetting, VLM, autonomous driving, benchmark, expert adapter

## TL;DR

This work systematically investigates catastrophic forgetting when fine-tuning VLMs for autonomous driving scenarios, constructs the large-scale 180K-scene benchmark FidelityDrivingBench, and proposes the Drive Expert Adapter (DEA), which enhances driving task performance via prompt-space routing without corrupting base model parameters.

## Background & Motivation

VLMs are increasingly applied to autonomous driving, yet a fundamental paradox exists: the fine-tuning process used to adapt models to driving data erodes the pre-trained world knowledge that constitutes the primary motivation for employing VLMs in the first place. Catastrophic forgetting induced by fine-tuning causes models to overlook obstacles in long-tail scenarios — such as curbs and rocks — resulting in unsafe trajectories.

Existing benchmarks fail to detect such degradation, as training and test sets maintain similar distributions that obscure genuine knowledge loss. This paper presents the first systematic investigation of catastrophic forgetting in VLM-based autonomous driving, proposing a purpose-built benchmark to quantify the degree of forgetting.

## Method

### Overall Architecture

The work constructs FidelityDrivingBench (180K scenes, 900K long-tail QA pairs, 15 data sources) and uses it to analyze forgetting phenomena, while proposing the DEA framework to perform knowledge adaptation in prompt space via scene-specific routing. The data construction pipeline employs GPT-OSS-120B to extract scene elements from language annotations, computes IDF rarity scores, and automatically mines long-tail scenes.

### Key Designs

1. **Long-tail scene mining pipeline**: Key scene elements are extracted from annotations; an IDF (Inverse Document Frequency) rarity score is computed for each element; the total scene rarity is calculated as the sum of element scores; long-tail scenes are mined automatically. A final set of 1,000 images is selected as the forgetting test set.

2. **Knowledge Retention Rate (KRR)**: Quantifies the degree to which the model retains general non-driving knowledge before and after fine-tuning, providing a standardized metric for forgetting evaluation.

3. **Drive Expert Adapter (DEA)**: Transfers adaptation from weight space to prompt space. Based on scene-specific cues (visibility, traffic density) and prompt semantics, the model dynamically routes to different driving experts. Base model parameters remain frozen, preventing knowledge corruption.

### Loss & Training

DEA trains only lightweight routing and prompt parameters. Full fine-tuning, layer freezing, and LoRA are compared; full fine-tuning is found to cause severe forgetting, while LoRA mitigates forgetting but yields insufficient driving performance. Experiments further reveal that LoRA is susceptible to task-induced attention bias.

### Long-tail Scene Mining Pipeline

Key scene elements (e.g., road conditions, traffic participants) are extracted from language annotations. An IDF rarity score is computed for each element, and total scene rarity equals the sum of element scores. From 180K candidate scenes, 1,000 representative long-tail images are ranked and selected as the forgetting test set.

## Key Experimental Results

### Main Results

| Method | Driving Task Performance | KRR | Notes |
|--------|--------------------------|-----|-------|
| Full Fine-tuning | High | Low | Severe forgetting |
| LoRA | Medium | High | Insufficient performance |
| DEA (Ours) | High | High | Achieves both |

FidelityDrivingBench covers 3 core driving tasks (scene understanding, motion analysis, trajectory planning), 15 data sources (nuScenes, WOD-E2E, etc.), totaling 180K frames and 900K long-tail QA pairs. The long-tail test set is produced through automated IDF-based rarity scoring combined with human review to select 1,000 representative images. KRR evaluates knowledge retention on general non-driving knowledge (e.g., recognition of long-tail obstacles such as curbs and rocks) before and after fine-tuning.

### Key Findings

- Multi-source data training exhibits lower forgetting and higher KRR than single-dataset training.
- Existing benchmarks overemphasize QA quantity while neglecting scene diversity.
- LoRA is insufficient to fully bridge the domain gap and is susceptible to task-induced attention bias.
- DEA effectively decouples driving adaptation from knowledge retention by routing different knowledge experts at the prompt level.

## Highlights & Insights

- This is the first work to systematically reveal forgetting in VLM fine-tuning for autonomous driving, carrying significant safety implications.
- The IDF-based long-tail scene mining pipeline enables automated large-scale discovery of rare scenes.
- The prompt-space routing design of DEA elegantly circumvents forgetting caused by weight modification.

## Limitations & Future Work

- DEA's routing strategy requires scene classification capability and may be constrained by classification accuracy.
- The forgetting test set contains only 1,000 images, limiting the coverage of scene types.
- Visualization analysis on RecogDrive + InternVL3-8B shows that forgetting causes models to overlook long-tail obstacles such as curbs and rocks.
- Dynamic balancing mechanisms among multiple expert routes remain unexplored.
- Ablation analysis reveals that even single-source data of equivalent scale suffers more severe forgetting than multi-source training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of forgetting in driving VLMs
- **Technical Depth**: ⭐⭐⭐⭐ — Benchmark, analysis, and method integrated
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Large-scale validation across 180K scenes
- **Practical Value**: ⭐⭐⭐⭐⭐ — Directly relevant to autonomous driving safety

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](../../AAAI2026/llm_safety/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)

</div>

<!-- RELATED:END -->
