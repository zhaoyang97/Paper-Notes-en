---
title: >-
  [Paper Note] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards
description: >-
  [ICML 2026][LLM Reasoning][Indirect Reward] The authors use "whether a ground street view and a satellite image can be localized to the same coordinate" as a verifiable indirect reward…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Indirect Reward"
  - "RLVR"
  - "Cross-View Pairing"
  - "Geospatial Reasoning"
  - "Zero-Shot Generalization"
date: 2026-05-08
content_hash: b0bb0e5090e1802c
---

# Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards

**Conference**: ICML 2026  
**arXiv**: [2510.00072](https://arxiv.org/abs/2510.00072)  
**Code**: https://github.com/miniHuiHui/Geo-R1  
**Area**: Reinforcement Learning / Multimodal VLM / Geospatial Reasoning  
**Keywords**: Indirect Reward, RLVR, Cross-View Pairing, Geospatial Reasoning, Zero-Shot Generalization

## TL;DR
The authors use "whether a ground street view and a satellite image can be localized to the same coordinate" as a verifiable indirect reward, and apply two-stage post-training (CoT scaffolding + RL self-exploring) with GRPO to Qwen2.5-VL-7B. This enables the model to learn general reasoning abilities that can zero-shot transfer to 25+ geospatial tasks using only GPS metadata.

## Background & Motivation
**Background**: Geospatial imagery (satellite/UAV/street view) is nearly unlimited, but samples with dense semantic labels are rare. Mainstream approaches (MAE/contrastive learning/RS-VLM) excel at representation and retrieval, but lack scene decomposition and reasoning abilities. Recent R1-style RL has succeeded in math/code, but the same bottleneck remains: in geospatial domains, there is no large-scale, strong direct reward signal.

**Limitations of Prior Work**: (1) SFT is constrained by task distribution, leading to narrow learning and OOD collapse; (2) Fully supervised detection/segmentation/VQA annotation is expensive; (3) Existing R1-style works (e.g., GLOBE) still use "geolocation" as the only direct reward, lacking a unified principle for why such tasks induce reasoning.

**Key Challenge**: Metadata (coordinates, timestamps) is easy to obtain and seemingly unrelated to "complex visual reasoning," but if designed properly, its verifiability can serve as the reward basis for proxy tasks.

**Goal**: (1) Demonstrate that "indirect verifiable rewards" can induce complex, transferable geospatial reasoning; (2) Provide theoretical explanation for when indirect rewards are effective; (3) Construct a reproducible RLVR framework and conduct large-scale OOD validation on 25+ tasks.

**Key Insight**: Use cross-view pairing between "street view ↔ satellite image" as a proxy task—the challenge is that it requires transferable geometric semantics such as object geometry, shadow direction, and building layout to accomplish the match.

**Core Idea**: Cross-View Pairing provides a binary verifiable reward. Combined with hard negatives, it blocks shortcut solutions based on nuisance features, forcing the model to learn view-invariant geometric semantics $\Phi$, thereby acquiring general zero-shot geospatial reasoning ability.

## Method

### Overall Architecture
Geo-R1 uses Qwen2.5-VL-7B as the base and applies two-stage post-training: Stage 1 is Geospatial Thinking Scaffolding, using 12.6K high-quality CoT data derived from CV-Cities for SFT, injecting a unified geospatial reasoning template (observe visual cues → cross-view evidence → associate climate/geographical knowledge → draw conclusion). Stage 2 is Reasoning Elevation via Indirect Signals, switching the training objective to Cross-View Pairing: given a ground panorama $I_g$ and $k$ candidate satellite images $\mathcal S=\{I_s^1,\dots,I_s^k\}$ (1 positive + $k-1$ hard negatives from the neighborhood), the model uses CoT reasoning to select the positive. Rewards are optimized using GRPO's group-relative form for the composite reward $r=\lambda_{\mathrm{acc}}r_{\mathrm{acc}}+\lambda_{\mathrm{fmt}}r_{\mathrm{fmt}}+\lambda_{\mathrm{len}}r_{\mathrm{len}}+\lambda_{\mathrm{rep}}r_{\mathrm{rep}}$. Both SFT and RL use full-parameter fine-tuning, 8×H100, leveraging LLama-Factory + VLM-R1 + vLLM for accelerated inference.

### Key Designs

1. **Cross-View Pairing + Hard-Negative Bottleneck**:

    - Function: Constructs a proxy task that is verifiable, scalable, and enforces learning of view-invariant features.
    - Mechanism: Decompose image information into view-invariant geometric semantics $\Phi(I)$ and modality-specific nuisance factors $N(I)$. Hard negatives are sampled from the same spatiotemporal neighborhood, ensuring $\mathcal I(Y;N(\mathcal S))=0$, so any strategy relying only on nuisance yields maximum entropy $\mathcal H(Y|\pi_{shortcut})=\log K$ (Theorem 3.1). Geo-R1's binary reward $r_{acc}\in\{-1,+1\}$ suffices to maximize $\mathcal I(C;Y|\mathcal S)\Leftrightarrow\mathcal I(C;\Phi(I_s^*))$, i.e., the reasoning chain $C$ must encode $\Phi$.
    - Design Motivation: Formalizes the question of whether "indirect rewards can induce reasoning" as a falsifiable theorem, and provides the "difficulty gap $\Delta\mathcal H=\log K$" as the reasoning margin RL must overcome.

2. **CoT Scaffolding: Single-Template Paradigm Injection**:

    - Function: Prevents RL cold-start collapse without introducing catastrophic forgetting from SFT task diversity.
    - Mechanism: Synthesizes 12.6K reasoning traces from CV-Cities, all following the template "analyze visual cues → cross-view evidence → associate geographical knowledge → output answer"; introduces a Fact-Check Engine to verify key entities (coordinates/city names) with metadata, ensuring the scaffold does not teach factual errors.
    - Design Motivation: Traditional SFT cold start tends to cover many tasks for diversity, but this harms subsequent RL; this work takes the opposite approach—injecting only one reasoning paradigm and letting RL explore specific abilities.

3. **GRPO + Composite Verifiable Reward**:

    - Function: Produces structurally compliant, appropriately long, non-repetitive, and highly accurate reasoning chains under outcome-only rewards.
    - Mechanism: Uses verifiable $r_{\mathrm{acc}}$ as the main signal, with format regularization $r_{\mathrm{fmt}}$, length regularization $r_{\mathrm{len}}$, and repetition penalty $r_{\mathrm{rep}}$; GRPO is used for group-relative advantage normalization to stabilize long-horizon updates; no process reward is introduced at intermediate steps, maintaining a purely outcome-based, verifiable process.
    - Design Motivation: Pure outcome rewards are cheap (no process annotation needed) and give the model full freedom to explore; regularization prevents the model from generating abnormally long/repetitive/unformatted outputs to game the reward.

### Loss & Training
Stage 1 uses standard SFT cross-entropy with Fact-Check filtering; Stage 2 uses GRPO, with RL from base yielding Geo-R1-Zero, and RL from Geo-SFT yielding Geo-R1; full-parameter fine-tuning, 8×H100; inference accelerated by vLLM; hard negatives are sampled from the same spatiotemporal neighborhood to ensure nuisance is indistinguishable.

## Key Experimental Results

### Main Results

| Benchmark / Task | Metric | Qwen2.5-VL-7B (base) | Geo-SFT | Geo-R1-Zero | Geo-R1 |
|------------------|--------|----------------------|---------|-------------|--------|
| In-distribution Cross-View Pairing | Accuracy | 19.0% | 23.1% | 78.1% | 82.4% |
| Same as above | Completion length | 204.6 | 1127.6 | 587.4 | 378.8 |
| GeoChain (13 tasks OOD) | Avg accuracy | baseline | — | — | Significantly higher than baseline (Fig. 1) |
| IMAGEO-GSS Global 6152 | City / Country Acc | — | — | — | 0.3272 / 0.8146 (vs GeoCLIP 0.1086 / 0.6361) |
| IMAGEO-GSS | Mean / Median Dist (km) | — | — | — | 568.32 / 69.40 (vs GeoCLIP 943.48 / 266.90) |

### Ablation Study

| Configuration | Key Observation | Interpretation |
|---------------|----------------|---------------|
| SFT only (Geo-SFT) | Only 4.1% higher than base, nearly random | Positive SFT cannot capture indirect signal |
| RL from base only (Geo-R1-Zero) | 78.1%, +59% over base | Indirect reward alone can drive learning |
| SFT + RL (Geo-R1) | 82.4%, shorter and more stable outputs | Scaffolding provides compliant reasoning template |
| MP16-Reason (OOD) | Nearly matches fully supervised GLOBE-7B (1km Street 17.98 vs 17.99; 2500km Continent 93.56 vs 92.52) | Indirect reward can rival direct reward |
| RSTeller Satellite Geolocation | Comparable to o4-mini, surpasses base | Cross-view proxy task generalizes to pure satellite view |
| GeoBench-VLM Satellite Understanding | Surpasses expert models like GeoChat / EarthDial | Reasoning ability transfers to fine-grained RS tasks |

### Key Findings
- Indirect reward training spontaneously induces high-level semantic concepts: MP16-Reason reasoning trace wordclouds show frequent concepts like "architecture," "vegetation," "climate," and "analyzing," indicating the model is truly reasoning about the physical world rather than pattern matching.
- A single proxy task unlocks broad capabilities: Training only on Cross-View Pairing enables significant zero-shot improvements on 25+ entirely different downstream tasks (VQA, geolocation, disaster assessment, land use, etc.).
- Geo-R1 also outperforms the base on satellite-view tasks, confirming that the cross-view task implicitly builds a "ground → overhead" back-projection logic, making the model a more general aerial interpreter.

## Highlights & Insights
- The concept of "verifiable indirect reward" can be extended to any rare domain with "abundant raw data + sparse metadata" (e.g., medical imaging + DICOM metadata, chemical structures + reaction temperatures), representing a new RLVR paradigm.
- The hard-negative bottleneck strictly blocks shortcuts, and the entropy gap $\Delta\mathcal H=\log K$ quantifies the reasoning margin—a rare example of theory-engineering coupling.
- Counterintuitively, "SFT should be narrow, not broad": scaffolding SFT with only one reasoning template yields the best RL warm-up and minimal forgetting.
- Even the simplest binary reward $\{-1,+1\}$ can drive complex reasoning, showing that process reward is not essential—of great engineering value for cost-sensitive domains.

## Limitations & Future Work
- The proxy task depends on "paired ground + satellite images"; finding equivalent cross-view structures in other rare domains is not straightforward;
- Only ground/aerial two views are demonstrated; not yet extended to SAR/LiDAR/multi-temporal or more complex modalities;
- Single 7B model scale; stability of outcome-only RL at 30B+ scale remains unverified;
- Still lags behind closed-source o3 on IMAGEO-GSS, attributed by the authors to differences in parameter count and RL investment.

## Related Work & Insights
- **vs GLOBE (Li et al. 2025a)**: GLOBE uses direct geolocation reward for MP16-Reason training; Geo-R1 never sees the MP16-Reason training set and achieves nearly equal performance using only cross-view pairing indirect reward, making it a strong alternative to direct reward approaches.
- **vs GeoCLIP / RFM-YFCC**: Traditional retrieval/representation learning excels at nearest neighbor localization but lacks reasoning and cross-task transfer; Geo-R1 comprehensively outperforms on IMAGEO-GSS.
- **vs GeoReasoner / GeoChat / EarthDial**: These methods rely on large-scale task-specific supervision; Geo-R1 surpasses them zero-shot with a single proxy task, inspiring the idea that "task breadth can be induced by proxy tasks rather than exhaustive supervision."
- **vs DeepSeek-R1 style math/code RLVR**: This work demonstrates that the RLVR paradigm can extend to verifiable but weakly related proxy rewards, providing the first clear blueprint for "how to transfer R1 to rare domains."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Indirect verifiable reward" + hard-negative bottleneck is a novel paradigm for geospatial RLVR with cross-domain methodological significance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 25+ downstream OOD tasks + multiple benchmarks + horizontal comparison with GLOBE / GeoCLIP / o3, extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clean theory-engineering integration, clear formulas and remarks; some details require appendix reference.
- Value: ⭐⭐⭐⭐⭐ Provides a reproducible RLVR blueprint for rare domains with abundant data but few labels, directly applicable to medical/climate/robotics fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[CVPR 2026\] GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](../../CVPR2026/llm_reasoning/graze_grounded_refinement_and_motion-aware_zero-shot_generation.md)
- [\[ICML 2026\] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](many-shot_cot-icl_making_in-context_learning_truly_learn.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards](../../NeurIPS2025/llm_reasoning/smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)

</div>

<!-- RELATED:END -->
