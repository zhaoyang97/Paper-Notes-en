---
title: >-
  [Paper Note] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models
description: >-
  [NeurIPS 2025][Audio & Speech][Data processing systems] Data-Juicer 2.0 is a cloud-scale multimodal data processing system for foundation models, featuring 150+ operators spanning text, image, video…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Data processing systems"
  - "multimodal data"
  - "distributed computing"
  - "data quality"
  - "foundation models"
date: 2026-05-08
content_hash: f03c82faa305d39f
---

# Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2501.14755](https://arxiv.org/abs/2501.14755)  
**Code**: [GitHub](https://github.com/modelscope/data-juicer)  
**Area**: Audio & Speech
**Keywords**: Data processing systems, multimodal data, distributed computing, data quality, foundation models

## TL;DR
Data-Juicer 2.0 is a cloud-scale multimodal data processing system for foundation models, featuring 150+ operators spanning text, image, video, and audio. It supports adaptive distributed execution (Ray/MaxCompute), efficiently processes TB-scale data on 10,000+ CPU cores, and has been widely adopted in products such as Alibaba Cloud PAI.

## Background & Motivation

**Background**: Foundation model training demands massive multimodal data processing pipelines. RedPajama and Dolma primarily target text, while Spark targets traditional big data workloads.

**Limitations of Prior Work**: (a) Insufficient multimodal support: lack of cross-modal alignment and semantic transformation; (b) Efficiency–scalability tension: foundation model workloads consist of simple yet massive per-sample operations; (c) Ecosystem fragmentation: API incompatibilities hinder cross-platform optimization.

**Key Challenge**: No unified framework simultaneously addresses functionality (multimodal + semantic), scale (PB-level), and usability (Python-native).

**Goal**: Build a unified, scalable multimodal data processing system covering cleaning, annotation, synthesis, and post-training.

**Key Insight**: Extending Data-Juicer 1.0 (50 text operators) to multimodal, large-scale, and multi-engine settings via a layered architecture.

**Core Idea**: Layered adaptive architecture + 150+ multimodal operators + unified multi-engine abstraction = cloud-scale data processing covering the full foundation model lifecycle.

## Method

### Overall Architecture
A three-layer architecture: (1) **Capability Layer** — 150+ multimodal operators (7 types); (2) **Interface Layer** — Python API / RESTful / Web UI / NL Agent; (3) **Runtime Layer** — unified Dataset abstraction, adaptive execution, and fault tolerance.

### Key Designs

1. **150+ Multimodal Operator Suite**:

    - Function: Covers cleaning, analysis, synthesis, and annotation across text, image, video, and audio.
    - Mechanism: Five original atomic operators extended with five composite operator types (Grouper / Aggregator / FusedOP / ScriptOP / HumanOP). HumanOP integrates Label Studio to support RLHF human-in-the-loop workflows.
    - Design Motivation: Model-driven operators (SDXL / GPT / Qwen) constitute the majority, reflecting the trend toward semantics-aware processing.

2. **Unified Data-Juicer-Dataset Abstraction**:

    - Function: Shields differences across underlying engines (HF Dataset / Ray / MaxFrame).
    - Mechanism: Facade pattern with a token-aligned intermediate schema (special tokens such as `<__dj__image>` represent multimodal data).
    - Design Motivation: Enables seamless pipeline switching between single-machine, Ray, and MaxCompute environments.

3. **Adaptive Runtime Optimization**:

    - Function: Automatically configures resources, batch sizes, and execution order.
    - Mechanism: The `Adapter` class uses `probe_small_batch()` to profile operator throughput and schedules faster operators first. GPU operators are automatically configured with quantization; I/O operators employ multi-level parallelism. Adaptive partitioning yields 2–3× speedup on Ray.
    - Design Motivation: Uniform parallelism granularity leads to OOM errors or resource waste; adaptive tuning optimizes both efficiency and stability.

4. **Fault Tolerance and Recovery**:

    - Function: Operator-level checkpointing and fine-grained recovery.
    - Mechanism: LLM output pre-validation with automatic retry; operator-level checkpoint-resume replaces Ray's coarse-grained full restart.
    - Design Motivation: Late-stage failures at large scale can waste terabytes of computation.

## Key Experimental Results

### Performance at Different Scales

| Scale | Sample Count | Recommended Engine | Key Findings |
|-------|--------------|--------------------|--------------|
| Small | 560K–2.24M | Single-machine HF | Single-machine is efficient; Ray on 4 nodes yields 138–226% speedup |
| Medium | 5.6M–56M | Ray-DLC | Ray outperforms single-machine; DLC is 24.8% faster than ECS |
| Large | 56M–70B | Multimodal: Ray / Text: MaxCompute | MaxCompute processes text in only 1/4 the time of Ray |

### Large-Scale Deduplication Performance

| Data Volume | CPU Cores | Time |
|-------------|-----------|------|
| 200 GB | 640 | 11.13 min |
| 1 TB | 640 | 50.83 min |
| 5 TB | 1280 | 168.10 min |

### Optimization Gains

| Optimization | Effect |
|--------------|--------|
| Adaptive partitioning | 2–3× speedup; network I/O reduced from 160 MB/s to 60 MB/s |
| Operator reordering + fusion | Up to 70.22% reduction |
| Automatic GPU allocation | Up to 99% savings |
| Batch processing | Up to 84% reduction |

### Key Findings
- Storage–compute–software co-design becomes critical beyond 10M samples: AI-CPFS (3× bandwidth) delivers 2.7× speedup.
- 70B samples processed with 6,400-core Ray in only 7,611 seconds.
- MaxCompute significantly outperforms Ray for pure-text workloads, owing to compute–storage co-design.

## Highlights & Insights
- **Production-grade scale**: Engineering at 12,800 cores, TB-scale data, and 70B samples — not a toy system.
- **Models as operators**: Treating large models as first-class operator citizens instantiates a "using AI to process AI data" paradigm with broad transferability.

## Limitations & Future Work
- GPU backend support is limited; NVIDIA NeMo Curator's GPU acceleration is not fully exploited.
- Multilingual support remains insufficient.
- Systematic evaluation of post-processing data quality is lacking.

## Related Work & Insights
- **vs. NeMo Curator**: Higher GPU acceleration efficiency (1.1 TB on 64 A100s in only 1.8 h), but Data-Juicer 2.0 is more general-purpose and feature-rich.
- **vs. v1.0**: 50 → 150+ operators; 1,000+ → 10,000+ cores; 70M → 70B samples.

## Rating
- Novelty: ⭐⭐⭐ Primarily system integration and engineering; limited innovation at the level of individual techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple scales and engines.
- Writing Quality: ⭐⭐⭐ Content-rich but slightly verbose.
- Value: ⭐⭐⭐⭐ Significant contribution to the data processing ecosystem as open-source infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](../../ICML2026/audio_speech/algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)

</div>

<!-- RELATED:END -->
