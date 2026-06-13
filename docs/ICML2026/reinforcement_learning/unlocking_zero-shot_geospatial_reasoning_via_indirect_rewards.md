---
title: >-
  [Paper Note] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards
description: >-
  [ICML 2026][Reinforcement Learning][Indirect rewards] The authors utilize the matching of "ground-level street views and satellite imagery to the same coordinates" as a verifiable indirect reward. Using GRPO…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Indirect rewards"
  - "RLVR"
  - "cross-view pairing"
  - "geospatial reasoning"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 56566f0d53807230
---

# Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards

**Conference**: ICML 2026  
**arXiv**: [2510.00072](https://arxiv.org/abs/2510.00072)  
**Code**: https://github.com/miniHuiHui/Geo-R1  
**Area**: Reinforcement Learning / Multimodal VLM / Geospatial Reasoning  
**Keywords**: Indirect rewards, RLVR, cross-view pairing, geospatial reasoning, zero-shot generalization

## TL;DR
The authors utilize the matching of "ground-level street views and satellite imagery to the same coordinates" as a verifiable indirect reward. Using GRPO, they perform two-stage post-training on Qwen2.5-VL-7B (CoT scaffolding + RL self-exploring). This enables the model to learn universal reasoning capabilities that generalize zero-shot to 25+ geospatial tasks using only GPS metadata.

## Background & Motivation
**Background**: Geospatial imagery (satellite, drone, street view) is nearly infinite, yet samples with dense semantic labels are extremely scarce. Prevailing approaches (MAE, contrastive learning, RS-VLM) excel at representation and retrieval but lack scene decomposition and reasoning capabilities. Recently, R1-style RL has succeeded in mathematics and coding, but the same bottleneck exists: the geospatial domain lacks large-scale, strongly supervised direct reward signals.

**Limitations of Prior Work**: (1) SFT is constrained by task distribution, leading to narrow domain learning and OOD collapse; (2) fully supervised detection, segmentation, and VQA annotations are expensive; (3) existing R1-style works (e.g., GLOBE) still treat "geographic location" as the sole direct reward, lacking a unified principle of why this induces reasoning.

**Key Challenge**: Metadata (coordinates, timestamps) are easily obtained and seemingly unrelated to "complex visual reasoning." However, if designed correctly, their verifiability can serve as the foundation for rewards in "proxy tasks."

**Goal**: (1) Prove that "indirect verifiable rewards" are sufficient to induce complex, transferable geospatial reasoning; (2) provide a theoretical explanation for when indirect rewards are effective; (3) construct a reproducible RLVR framework and perform large-scale OOD validation across 25+ tasks.

**Key Insight**: Treat "street view $\leftrightarrow$ satellite image" cross-view pairing as the proxy task. This challenge requires utilizing transferable geometric semantics such as object geometry, shadow direction, and building layouts to complete the match.

**Core Idea**: Cross-view pairing provides a binary verifiable reward. Combined with hard negatives to block "cheating" via nuisance features, this forces the model to learn view-invariant geometric semantics $\Phi$, thereby acquiring general zero-shot geospatial reasoning capabilities.

## Method

### Overall Architecture
Geo-R1 uses Qwen2.5-VL-7B as the base and undergoes two-stage post-training. Phase 1 is Geospatial Thinking Scaffolding, using 12.6K high-quality CoT data derived from CV-Cities for SFT. This injects a unified geospatial reasoning template (analyze visual cues $\rightarrow$ cross-view evidence verification $\rightarrow$ associate geographic knowledge like climate zones $\rightarrow$ provide conclusion). Phase 2 is Reasoning Elevation via Indirect Signals, switching the training objective to Cross-View Pairing: given a ground panorama $I_g$ and $k$ candidate satellite images $\mathcal S=\{I_s^1,\dots,I_s^k\}$ (containing 1 positive and $k-1$ spatial hard negatives), the model selects the positive example after CoT reasoning. The rewards are optimized using the group-relative form of GRPO with a composite reward $r=\lambda_{\mathrm{acc}}r_{\mathrm{acc}}+\lambda_{\mathrm{fmt}}r_{\mathrm{fmt}}+\lambda_{\mathrm{len}}r_{\mathrm{len}}+\lambda_{\mathrm{rep}}r_{\mathrm{rep}}$. Both SFT and RL undergo full-parameter fine-tuning on 8×H100, accelerated by LLama-Factory, VLM-R1, and vLLM.

### Key Designs

1.  **Cross-View Pairing + Hard-Negative Bottleneck**:
    - **Function**: Constructs a proxy task that is verifiable, scalable, and forces the learning of view-invariant features.
    - **Mechanism**: Decomposes image information into view-invariant geometric semantics $\Phi(I)$ and modality-specific nuisance factors $N(I)$. Sampling hard negatives from the same spatio-temporal neighborhood ensures $\mathcal I(Y;N(\mathcal S))=0$, meaning any strategy relying solely on nuisance features will result in maximum entropy $\mathcal H(Y|\pi_{shortcut})=\log K$ (Theorem 3.1). Geo-R1's binary reward $r_{acc}\in\{-1,+1\}$ is sufficient to maximize $\mathcal I(C;Y|\mathcal S)\Leftrightarrow\mathcal I(C;\Phi(I_s^*))$, meaning the reasoning chain $C$ must encode $\Phi$.
    - **Design Motivation**: Formalizes the question of "whether indirect rewards can stimulate reasoning" into a falsifiable theorem and provides the difficulty gap $\Delta\mathcal H=\log K$ as the reasoning margin RL must cross.

2.  **CoT Scaffolding: Single-Template Paradigm Injection**:
    - **Function**: Prevents RL cold-start collapse without introducing catastrophic forgetting associated with SFT task diversity.
    - **Mechanism**: Synthesizes 12.6K reasoning traces using CV-Cities, following a template of "visual cue analysis $\rightarrow$ cross-view evidence validation $\rightarrow$ geographic knowledge association $\rightarrow$ answer output." A Fact-Check Engine uses metadata to verify key entities like coordinates or city names, ensuring the scaffold does not learn false facts.
    - **Design Motivation**: Traditional SFT cold starts aim for diversity across many tasks, which can disrupt subsequent RL. This work does the opposite — injecting only one reasoning paradigm and letting the RL phase explore specific capabilities.

3.  **GRPO + Composite Verifiable Reward**:
    - **Function**: Produces reasoning chains that are structurally compliant, moderately long, non-repetitive, and highly accurate under outcome-only rewards.
    - **Mechanism**: Uses verifiable $r_{\mathrm{acc}}$ as the primary signal, supplemented by format regularization $r_{\mathrm{fmt}}$, length regularization $r_{\mathrm{len}}$, and repetition penalty $r_{\mathrm{rep}}$. GRPO performs group-relative advantage normalization to stabilize long-horizon updates. No process rewards are introduced, maintaining the "free process, verifiable result" nature of pure outcome-based training.
    - **Design Motivation**: Pure outcome rewards are cheap (no process annotation needed) and leave all degrees of freedom for the model to explore. Regularization prevents the model from generating abnormally long, repetitive, or unformatted output to exploit rewards.

### Loss & Training
Phase 1 uses standard SFT cross-entropy with filtering via Fact-Check. Phase 2 uses GRPO: Geo-R1-Zero is obtained by RL directly from the base, while Geo-R1 continues from Geo-SFT. Full-parameter fine-tuning on 8×H100; inference accelerated by vLLM; hard negatives are sampled in the same spatio-temporal neighborhood to ensure nuisance features are indistinguishable.

## Key Experimental Results

### Main Results

| Benchmark / Task | Metric | Qwen2.5-VL-7B (base) | Geo-SFT | Geo-R1-Zero | Geo-R1 |
|-------------|------|---------------------|--------|-------------|--------|
| In-distribution Cross-View Pairing | Accuracy | 19.0% | 23.1% | 78.1% | 82.4% |
| Same as above | Completion length | 204.6 | 1127.6 | 587.4 | 378.8 |
| GeoChain (13 tasks OOD) | Avg accuracy | baseline | — | — | Significantly higher than baseline (Fig. 1) |
| IMAGEO-GSS (6152 images) | City / Country Acc | — | — | — | 0.3272 / 0.8146 (vs GeoCLIP 0.1086 / 0.6361) |
| IMAGEO-GSS | Mean / Median Dist (km) | — | — | — | 568.32 / 69.40 (vs GeoCLIP 943.48 / 266.90) |

### Ablation Study

| Configuration | Key Observation | Interpretation |
|------|----------|------|
| SFT only (Geo-SFT) | Only 4.1% higher than base, nearly random | Positive-only SFT cannot capture indirect signals |
| RL only from base (Geo-R1-Zero) | 78.1%, +59% over base | Indirect reward alone can drive reasoning |
| SFT + RL (Geo-R1) | 82.4%, shorter and more stable | Scaffolding provides a compliant reasoning template |
| MP16-Reason (OOD) | Nearly equal to fully supervised GLOBE-7B (1km Street 17.98 vs 17.99) | Indirect reward performance is comparable to direct reward |
| RSTeller satellite geoloc. | Same level as o4-mini, exceeds base | Cross-view proxy tasks generalize to pure satellite views |
| GeoBench-VLM satellite understanding | Exceeds experts like GeoChat / EarthDial | Reasoning capability transfers to fine-grained RS tasks |

### Key Findings
- Indirect reward training leads to the spontaneous emergence of high-level semantic concepts: Word clouds of MP16-Reason traces show high frequency for concepts like "architecture," "vegetation," "climate," and "analyzing," indicating physical world reasoning rather than pattern matching.
- A single proxy task unlocks a broad spectrum of capabilities: training only on Cross-View Pairing yields significant zero-shot improvements across 25+ distinct downstream tasks (VQA, geolocation, disaster assessment, land use, etc.).
- Geo-R1 outperforms the base on satellite-view tasks, confirming that cross-view tasks implicitly construct "ground $\rightarrow$ aerial" re-projection logic, making the model a more universal aerial interpreter.

## Highlights & Insights
- The concept of "verifiable indirect rewards" can be generalized to any domain with "massive raw data + sparse metadata" (e.g., medical imaging + DICOM metadata, chemical structures + reaction temperatures), representing a new paradigm for RLVR.
- The rigorous use of a hard-negative bottleneck to block shortcuts, combined with describing the reasoning margin via entropy difference $\Delta\mathcal{H}=\log K$, is a rare example of tight theory-engineering coupling.
- Counter-intuitively, the study emphasizes that "SFT should be narrow rather than broad": scaffolding SFT with only one template leads to the best RL warmup and minimal forgetting.
- The fact that complex reasoning can be driven by a simple binary $\{-1, +1\}$ reward suggests that process rewards are not mandatory, which has significant engineering value for cost-sensitive domains.

## Limitations & Future Work
- The proxy task relies on "paired ground + satellite imagery"; finding equivalent cross-view structures in other rare domains may be difficult.
- Only ground/aerial views are demonstrated, without extension to more complex modalities like SAR, LiDAR, or multi-temporal data.
- Tested on a 7B model; whether outcome-only RL remains stable when scaled to 30B+ is unverified.
- There is still a gap compared to closed-source o3 on IMAGEO-GSS, which the authors attribute to differences in parameter counts and RL compute investment.

## Related Work & Insights
- **vs GLOBE (Li et al. 2025a)**: GLOBE uses direct geolocation rewards within the MP16-Reason training set; Geo-R1 achieves nearly equivalent performance without ever seeing the MP16-Reason training set, serving as a powerful alternative to the direct reward route.
- **vs GeoCLIP / RFM-YFCC**: Traditional retrieval and representation learning methods are good at nearest-neighbor localization but lack reasoning and cross-task transfer; Geo-R1 comprehensively outperforms them on IMAGEO-GSS.
- **vs GeoReasoner / GeoChat / EarthDial**: These methods rely on large-scale task-specific supervision; Geo-R1's zero-shot superiority with a single proxy task suggests that task breadth can be induced by proxy tasks rather than exhaustive supervision.
- **vs DeepSeek-R1 style Math/Code RLVR**: This paper proves the RLVR paradigm extends to verifiable but weakly correlated proxy rewards, providing the first clear blueprint for "migrating R1 to rare domains."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Verifiable indirect rewards" + hard-negative bottleneck is a brand-new paradigm for geospatial RLVR with cross-domain methodological significance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely broad coverage with 25+ downstream OOD tasks, multiple benchmarks, and horizontal comparisons with GLOBE, GeoCLIP, and o3.
- Writing Quality: ⭐⭐⭐⭐ Clean connection between theory and engineering; clear formulas and Remarks, though some details require the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a replicable RLVR blueprint for rare domains with high data volume but few labels, offering direct lessons for medical, climate, robotics, etc.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ICML 2026\] MindZero: Learning Online Mental Reasoning with Zero Annotations](mindzero_learning_online_mental_reasoning_with_zero_annotations.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)

</div>

<!-- RELATED:END -->
