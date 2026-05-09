---
title: >-
  [Paper Note] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents
description: >-
  [ICLR 2026][Remote Sensing][Earth Observation] Earth-Agent is the first Earth observation agent framework built upon an MCP-based tool ecosystem. It unifies RGB and spectral remote sensing data, dynamically invoking 104 expert tools to enable cross-modal, multi-step, and quantitative spatiotemporal reasoning. The accompanying Earth-Bench benchmark comprises 248 expert-curated tasks and 13,729 images. Experiments demonstrate that Earth-Agent substantially outperforms both general-purpose agents and remote sensing MLLMs.
tags:
  - ICLR 2026
  - Remote Sensing
  - Earth Observation
  - Agent Framework
  - MCP Tool Ecosystem
  - Multimodal Remote Sensing
  - Benchmark
date: 2026-05-08
content_hash: d7b661eb5649a8e2
---

# Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents

**Conference**: ICLR 2026
**arXiv**: [2509.23141](https://arxiv.org/abs/2509.23141)
**Code**: [opendatalab/Earth-Agent](https://github.com/opendatalab/Earth-Agent)
**Area**: Remote Sensing / LLM Agent
**Keywords**: Earth Observation, Agent Framework, MCP Tool Ecosystem, Multimodal Remote Sensing, Benchmark

## TL;DR
Earth-Agent is the first Earth observation agent framework built upon an MCP-based tool ecosystem. It unifies RGB and spectral remote sensing data, dynamically invoking 104 expert tools to enable cross-modal, multi-step, and quantitative spatiotemporal reasoning. The accompanying Earth-Bench benchmark comprises 248 expert-curated tasks and 13,729 images. Experiments demonstrate that Earth-Agent substantially outperforms both general-purpose agents and remote sensing MLLMs.

## Background & Motivation
Earth Observation (EO) is a critical task for understanding the evolving state of Earth systems. While multimodal large language models (MLLMs) have recently advanced remote sensing research, fundamental capability gaps remain:

Limitations of existing MLLMs in the EO domain:
- **RGB-only perception**: Inability to process spectral data (multispectral, hyperspectral, SAR, etc.), which is central to scientific-grade remote sensing analysis.
- **Shallow reasoning**: Inability to handle complex tasks requiring multi-step reasoning and domain-specific tool invocation.
- **Lack of quantitative capability**: Cannot perform geophysical parameter retrieval or quantitative spatiotemporal analysis requiring precise computation.
- **Absence of systematic evaluation**: No evaluation protocol covering all modalities and assessing both reasoning trajectories and final results.

Limitations of existing agent approaches:
- Restricted to RGB perception; spectral data not supported.
- Insufficient reasoning depth and primitive tool-calling capability.
- No systematic EO-oriented evaluation benchmark.

Starting Point: Earth-Agent models EO analysis as a ReAct-style POMDP process, with an LLM serving as the policy network that dynamically invokes domain expert tools via the MCP protocol, bridging RGB and spectral modalities.

## Method

### Overall Architecture
Earth-Agent adopts a ReAct-style agent architecture centered on a POMDP loop:
- **Input**: Task objective + remote sensing images (RGB / spectral / product data) + interaction history.
- The LLM acts as the policy, iteratively executing: tool invocation → memory update → reasoning → action.
- **Output**: Quantitative analysis results, parameter retrieval values, spatial reasoning conclusions, etc.

### Key Designs

1. **MCP-Based Tool Ecosystem**:
   Earth-Agent integrates 104 specialized tools across five functional suites:
    - **Index Kit**: Spectral index computation (NDVI, NDWI, etc.)
    - **Inversion Kit**: Geophysical parameter retrieval (leaf area index, land surface temperature, etc.)
    - **Perception Kit**: RGB image perception (object detection, scene classification, semantic segmentation, etc.)
    - **Analysis Kit**: Spatiotemporal analysis (change detection, trend analysis, etc.)
    - **Statistics Kit**: Statistical operations (regional statistics, histogram analysis, etc.)

   These tools are managed via the Model Context Protocol (MCP), enabling the LLM to dynamically compose and invoke them. This allows Earth-Agent to transcend the capability ceiling of pretrained MLLMs — for scientific-grade computational tasks (e.g., land surface temperature retrieval from Landsat data), the framework relies on precise physical models rather than the model's implicit knowledge.

2. **Cross-Modal Unified Processing**:
   Unlike existing EO agents restricted to RGB, Earth-Agent natively supports three categories of remote sensing data:
    - **Spectral data**: Multispectral/hyperspectral satellite imagery (e.g., Landsat, Sentinel-2).
    - **Product data**: Pre-processed remote sensing products (e.g., MODIS land surface temperature products).
    - **RGB data**: Conventional visible-light remote sensing imagery.

   The LLM autonomously determines whether to invoke spectral or perception tools based on task requirements.

3. **ReAct-POMDP Decision Process**:
   Complex EO tasks are modeled as Partially Observable Markov Decision Processes. Rather than producing answers in a single pass, the LLM reasons progressively through multi-round "think–act–observe" cycles. For example, analyzing vegetation change trends in a region from 2020 to 2025 requires: extracting multi-temporal NDVI → time-series analysis → trend fitting → generating conclusions.

4. **Earth-Bench Evaluation Benchmark**:
   Earth-Bench contains 248 tasks manually curated by domain experts, covering 13,729 images:
    - **Modality coverage**: Spectral (100 tasks) + Product (88 tasks) + RGB (60 tasks).
    - **Two-tier evaluation protocol**:
      - End-to-end evaluation: Accuracy (final answer correctness) + Efficiency (tool usage efficiency).
      - Trajectory evaluation: Tool-Any-Order (whether all necessary tools were used), Tool-In-Order (whether tool order is correct), Tool-Exact-Match (step-by-step exact match), Parameter Accuracy (accuracy of tool parameters).

### Loss & Training
The core of Earth-Agent is zero-shot inference — no additional training on EO tasks is required. The LLM interprets tasks through prompt engineering and tool descriptions. The paper also explores a Training-Free Evolution approach (analogous to training-free GRPO), which attempts to optimize the agent's tool-calling strategy without fine-tuning model weights.

## Key Experimental Results

### Main Results
Performance of different LLM backends on Earth-Bench:

| Model | Tool-Any-Order | Tool-In-Order | Tool-Exact-Match | Parameter | Accuracy | Efficiency |
|-------|---------------|---------------|-------------------|-----------|----------|------------|
| DeepSeek-V3 (IF) | 0.892 | 0.876 | 0.741 | 0.572 | — | — |
| GPT-5 (AP) | 0.766 | 0.750 | 0.596 | 0.462 | 59.32% | 1.531 |
| Kimi-K2 (IF) | 0.806 | 0.799 | 0.633 | 0.522 | 62.71% | 1.410 |

### Ablation Study

| Comparison | Key Metric | Description |
|------------|-----------|-------------|
| Earth-Agent vs. general-purpose agent frameworks | Accuracy | Earth-Agent significantly outperforms general agents such as LangChain |
| Earth-Agent vs. remote sensing MLLMs | RGB benchmark | Surpasses dedicated remote sensing MLLMs on remote sensing benchmarks |
| Spectral tasks vs. RGB tasks | Tool-Exact-Match | Spectral tasks involve longer and more complex tool chains, making exact matching more difficult |
| Different LLM backbones | Overall performance | Stronger LLMs yield better tool-calling and reasoning capability |

### Key Findings
- DeepSeek-V3 achieves the best tool-use accuracy (Tool-Any-Order: 0.892).
- Kimi-K2 marginally outperforms GPT-5 on final answer accuracy (62.71% vs. 59.32%).
- Efficiency scores are consistently above 1.0, indicating that models tend to invoke more tools than the ground truth requires.
- Parameter Accuracy is the most significant bottleneck (maximum 0.572), revealing limited LLM understanding of domain-specific remote sensing parameters.
- The gap between Tool-In-Order and Tool-Any-Order is small, suggesting models generally grasp the correct tool ordering.

## Highlights & Insights
- **Paradigm shift**: Moving from direct MLLM-based question answering to agent-driven dynamic expert tool invocation — a significant transition in the EO-AI paradigm.
- **Application of MCP protocol**: Using MCP to manage tools is sound engineering practice, enabling an extensible and replaceable toolset.
- **Elegant two-tier evaluation design**: Assessing not only final outcomes but also the reasoning process (tool-calling trajectories), which is essential for understanding agent behavior.
- **Scientific value**: Tasks such as geophysical parameter retrieval and quantitative spatiotemporal analysis go beyond conventional computer vision and carry genuine scientific application value.
- **Construction of 104 tools**: This constitutes a major engineering contribution in its own right, covering the principal components of EO analysis.

## Limitations & Future Work
- Strong dependence on the LLM's capability ceiling — errors in LLM reasoning cause the entire pipeline to fail.
- Parameter Accuracy (maximum 0.572) reveals that LLMs still lack sufficient domain knowledge in remote sensing.
- Efficiency scores above 1.0 indicate a tendency toward redundant tool calls, necessitating optimization of reasoning efficiency.
- Only a limited number of LLM backbones are evaluated; applicability to open-source smaller models remains unknown.
- The scale of Earth-Bench (248 tasks) remains relatively small compared to NLP/CV benchmarks.
- Latency issues are not discussed — the delays incurred by multi-step tool invocation may be problematic in real-world remote sensing applications.
- The effectiveness of Training-Free Evolution has yet to be systematically evaluated.

## Related Work & Insights
- **ReAct (Yao et al., 2023)**: The foundational work on the think–act paradigm; Earth-Agent represents its concrete instantiation in the EO domain.
- **ToolFormer / Gorilla**: Pioneering works on LLM tool use; Earth-Agent extends this to 104 domain expert tools.
- **GeoChat / RS-ChatGPT**: Existing remote sensing MLLMs, limited to RGB processing and lacking tool-calling support.
- **Model Context Protocol (MCP)**: A tool management protocol proposed by Anthropic; Earth-Agent serves as an important application case of MCP in the scientific domain.
- Insight: The agent + domain tools paradigm is equally applicable to other scientific fields such as astronomy, biology, and materials science.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](measuring_the_intrinsic_dimension_of_earth_representations.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[ICLR 2026\] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)

</div>

<!-- RELATED:END -->
