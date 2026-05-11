---
title: >-
  [Paper Note] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments
description: >-
  [AAAI 2026][LLM Agent][Virtual Sensor Data] LLM-driven embodied agents are instantiated to "live" in simulated smart home environments, generating virtual ambient sensor data for pre-training HAR models…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Virtual Sensor Data"
  - "Smart Home"
  - "Human Activity Recognition"
  - "Simulated Environment"
date: 2026-05-08
content_hash: 7aa6eab0361d23ca
---

# AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments

**Conference**: AAAI 2026
**arXiv**: [2506.11773v4](https://arxiv.org/abs/2506.11773v4)
**Code**: [https://github.com/ZikangLeng/AgentSense](https://github.com/ZikangLeng/AgentSense)
**Area**: Human Activity Recognition (HAR) / Embodied AI / Synthetic Data Generation
**Keywords**: LLM Agent, Virtual Sensor Data, Smart Home, Human Activity Recognition, Simulated Environment

## TL;DR
LLM-driven embodied agents are instantiated to "live" in simulated smart home environments, generating virtual ambient sensor data for pre-training HAR models, which yields significant gains in activity recognition under low-resource settings.

## Background & Motivation

### State of the Field

**Background**: Human Activity Recognition (HAR) in smart homes relies on ambient sensors (motion, door, appliance activation, etc.) to monitor daily activities, playing a critical role in healthcare and elder care. However, HAR model development is constrained by **the scarcity of large-scale annotated sensor data** — the diversity of home layouts, sensor configurations, and resident behavioral patterns makes data collection prohibitively costly and raises privacy concerns. Existing synthetic data generation methods focus primarily on wearable sensors (e.g., IMU data generated from video/audio), with insufficient support for ambient sensors. While simulation platforms such as VirtualHome can model household activities, they lack ambient sensor simulation capabilities and cannot directly produce sensor-level data.

### Approach

**Goal**: How can diverse, privacy-preserving ambient sensor data be automatically generated without real-world data collection, so as to alleviate data scarcity in HAR model training? Key challenges include: (1) generating sufficient behavioral diversity to cover different populations and scenarios; (2) translating high-level activity descriptions into fine-grained actions executable by the simulator; and (3) extracting realistic sensor signals from the simulator.

## Method

### Overall Architecture
AgentSense is an end-to-end virtual sensor data generation pipeline: LLM generates diverse personas → LLM generates daily schedules → LLM decomposes schedules into fine-grained actions → action cleaning and validation → execution in X-VirtualHome → virtual sensor data recording → label mapping to target datasets.

### Key Designs
1. **Three-Stage LLM Prompting Pipeline**:

    - **Persona Generation**: LLMs generate diverse virtual personas (age, occupation, health status, lifestyle habits) to capture behavioral diversity.
    - **High-Level Schedule Generation**: Full-day schedules are generated based on persona, day of week, and home environment (room list), distinguishing between "at home" and "out" activities, guided by few-shot examples, and avoiding overly regular time slots.
    - **Low-Level Action Decomposition**: Each high-level activity is decomposed into simulator-executable action sequences (18 predefined actions such as walk, grab, open, etc.); the LLM first selects an appropriate room, then generates actions based on the list of available objects in that room.

2. **LLM Output to Simulator Instruction Conversion** (five-step pipeline):

    - Output cleaning, embedding into the VirtualHome vocabulary (FAISS index), nearest-neighbor retrieval to replace hallucinated LLM tokens, threshold filtering (action threshold 0.8, object threshold 0.6), and final command assembly. LangChain + OpenAI embeddings + FAISS are used for semantic alignment to eliminate LLM hallucinations.

3. **X-VirtualHome Virtual Sensor System**:

    - **Motion Sensors**: Automatically placed according to room area (small ≤30m²: 1 sensor; medium: 2 sensors; large >60m²: 3 sensors); character position is tracked every 0.2 seconds with a detection radius of 5.0m and a motion threshold of ε=0.1m to distinguish genuine movement from jitter.
    - **Door Sensors**: Monitor CLOSED→OPEN state transitions of objects with the CAN_OPEN property in the environment graph (doors, cabinets, etc.).
    - **Appliance Activation Sensors**: Monitor OFF→ON state transitions of objects with the HAS_SWITCH property (microwave, washing machine, etc.).

### Loss & Training
The TDOST framework (text-description-based, layout-agnostic HAR method) is adopted: sensor trigger events are converted to natural language sentences → encoded with all-distilroberta-v1 → classified by a bidirectional LSTM (64 hidden units). Two variants are used: TDOST-Basic (sensor type + location) and TDOST-Temporal (with temporal information). Training uses the Adam optimizer with a learning rate of 1e-4, a ReduceLROnPlateau scheduler, and three-fold stratified cross-validation.

## Key Experimental Results

| Dataset | Metric | Real (TDOST-Basic) | Real+Virtual (TDOST-Basic) | Gain |
|--------|------|------|----------|------|
| Aruba | Accuracy | 91.00 | 93.19 | +2.19 |
| Cairo | Accuracy | 69.01 | 75.61 | +6.60 |
| Orange | Accuracy | 82.40 | 85.21 | +2.81 |
| Aruba | Macro F1 | 63.98 | 72.20 | +8.22 |
| Cairo | Macro F1 | 51.51 | 62.47 | +10.96 |
| Orange | Macro F1 | 21.56 | 41.83 | +20.27 |
| Milan | Macro F1 (Temporal) | 57.20 | 73.41 | +16.21 |
| Aruba | Macro F1 (Temporal) | 68.57 | 77.36 | +8.79 |

Virtual data scale: 18 personas × 22 home layouts = 250 days of data, 3,266 activity windows.

### Ablation Study
- **Real Data Volume**: Using only 5%–10% of real data with virtual pre-training yields substantial improvements (Aruba Macro F1 +~10%, Kyoto7 +45%). On Cairo and Orange, approximately 200 real samples suffice to approach full-data training performance.
- **Component Contributions** (on Aruba): Single persona + single day + single environment: Macro F1 = 68.35% → +multi-environment = 70.69% → +multi-day = 71.01% → +multi-persona = 72.20%. Every diversity dimension contributes positively while total data volume remains constant.
- **Downstream Model Variants**: TDOST-Temporal on Milan improves Macro F1 from 57.20% to 73.41% (+16.21%), indicating that temporal information is particularly important for leveraging virtual data quality.
- **Cross-Layout Generalization**: Despite layout discrepancies between virtual environments and real homes, the pre-trained model improves performance on all 5 real datasets, demonstrating that behavioral diversity matters more than layout matching.

## Highlights & Insights
- **Complete End-to-End Pipeline**: A fully automated workflow from persona generation to sensor data, requiring no real-world data collection.
- **LLM Hallucination Mitigation**: LLM outputs are aligned to the simulator ontology via embedding + FAISS nearest-neighbor retrieval, elegantly resolving the LLM–simulator interface problem.
- **Privacy Preservation**: Entirely simulation-based generation avoids intrusive real-world data collection.
- **"Digital Cousin" Philosophy**: Rather than pursuing one-to-one digital twins, the approach generates diverse data through varied agents and environments.
- **Low-Resource Utility**: Fine-tuning with only a small amount of real data approaches full-data training performance, making the approach highly practical.

## Limitations & Future Work
- **Domain Gap**: Discrepancies exist between virtual environments and real home layouts (e.g., Milan has more rooms); no layout matching is performed.
- **Single-Resident Assumption**: Only single-occupant scenarios are simulated; multi-resident interaction activities cannot be handled.
- **Incomplete Activity Coverage**: Free-form LLM generation may omit certain common activities (e.g., Watch_TV, Enter_Home), requiring more targeted prompting.
- **Single LLM Choice**: Only GPT-4o-mini is tested; the impact of other LLMs on generation quality is unexplored.
- **Limited Sensor Types**: Only three sensor types are implemented (motion, door, appliance activation); temperature, humidity, and light sensors are not covered.
- **Single Evaluation Framework**: Only TDOST is used as the downstream evaluation framework; generalization to other HAR models is not validated.
- **Action Conversion Success Rate**: After the five-step cleaning pipeline, approximately 87% of LLM-generated actions can be successfully converted to simulator commands; the remainder are regenerated by the LLM or discarded.

## Related Work & Insights
- **Generative Agents (Park et al., 2023)**: Also uses LLMs to drive virtual agent behavior, but focuses on narrative and social interaction without producing structured sensor data. AgentSense redirects this paradigm toward the concrete downstream task of HAR data generation.
- **IMUTube / IMUGPT (Kwon et al., 2020; Leng et al., 2024)**: Generate wearable IMU data from video/text, but the methodology does not transfer to ambient sensors due to different spatial and triggering mechanisms. AgentSense fills the gap in synthetic data for ambient sensors.
- **Yonekura et al. (2024)**: Uses LLMs to generate smart home schedules but does not produce sensor data. AgentSense extends this by completing the full pipeline from schedules to sensor signals.

## Related Work & Insights
- **Cross-Modal Synthetic Data Paradigm**: The cross-modal transformation from text (LLM-generated personas and schedules) to time-series sensor data is transferable to other domains with scarce sensor data (e.g., industrial IoT, edge-case sensor simulation in autonomous driving).
- **LLM as Behavioral Prior**: Leveraging LLMs' internalized knowledge of human behavior as a prior for synthetic data generation is a paradigm extensible to other tasks requiring human behavior modeling (e.g., crowd simulation, traffic flow prediction).
- **Simulator + LLM Synergy**: LLMs handle high-level planning and diversity while the simulator ensures physical plausibility and sensor realism; this division of labor offers a reference model for embodied AI data generation.
- **Multimodal Extension Potential**: The paper notes that future work may jointly generate Pose2IMU and Video2IMU data to construct multimodal synchronized datasets, which has potential connections to cross-view learning in video understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ (LLM-driven simulation for ambient sensor data generation is a novel application, though individual modules rely on mature techniques)
- Experimental Thoroughness: ⭐⭐⭐⭐ (5 real datasets + ablation studies, but only one downstream framework is used and direct evaluation of sensor data quality is absent)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete appendix including all prompt templates, detailed method description)
- Value: ⭐⭐⭐⭐ (Addresses a real pain point in the HAR field, strong low-resource performance, open-source code)

## Additional Notes
- The methodology and experimental design of this work offer useful reference for related fields.
- Future work should validate the generalizability and scalability of the approach in more scenarios and at larger scales.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research value.
- Deployment feasibility and computational efficiency should be assessed in light of practical application requirements.
- The choice of datasets and evaluation metrics may affect the generalizability of conclusions; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work offer useful reference for related fields.
- Future work should validate the generalizability and scalability of the approach in more scenarios and at larger scales.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](../../ICLR2026/llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[AAAI 2026\] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)
- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[CVPR 2026\] Gen-n-Val: Agentic Image Data Generation and Validation](../../CVPR2026/llm_agent/gen_n_val_agentic_image_data_generation_and_validation.md)

</div>

<!-- RELATED:END -->
