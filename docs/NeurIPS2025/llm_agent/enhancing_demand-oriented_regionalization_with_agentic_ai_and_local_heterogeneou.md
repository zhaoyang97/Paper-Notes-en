---
title: >-
  [Paper Note] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning
description: >-
  [NeurIPS 2025 (UrbanAI Workshop)][LLM Agent][Agentic AI] This paper proposes a planning support system based on Agentic AI, in which an LLM agent guides non-technical users through data-driven demand-oriented regionaliza…
tags:
  - "NeurIPS 2025 (UrbanAI Workshop)"
  - "LLM Agent"
  - "Agentic AI"
  - "regionalization"
  - "urban planning"
  - "SOM"
  - "human-AI collaboration"
  - "disaster adaptation planning"
date: 2026-05-08
content_hash: 816991b3896f0ad5
---

# Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning

**Conference**: NeurIPS 2025 (UrbanAI Workshop)  
**arXiv**: [2511.10857](https://arxiv.org/abs/2511.10857)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: Agentic AI, regionalization, urban planning, SOM, human-AI collaboration, disaster adaptation planning

## TL;DR

This paper proposes a planning support system based on Agentic AI, in which an LLM agent guides non-technical users through data-driven demand-oriented regionalization. The core algorithm is RepSC-SOM (spatially constrained self-organizing map with representative initialization), supporting iterative human-AI collaborative refinement of regional delineations for disaster risk management and climate adaptation planning.

## Background & Motivation

Contemporary urban governance relies on planning units to design and implement growth, management, and adaptation policies. However, existing planning units suffer from significant shortcomings:

**Mismatch between traditional boundaries and actual needs**: Census tracts, zip codes, and administrative districts are designed for specific purposes (e.g., expediting mail delivery) and often fail to reflect the true spatial distribution of disaster exposure or social vulnerability.

**Rigidity of fixed boundaries**: Traditional units cannot capture the spatial heterogeneity of climate risk, making it difficult to design customized zones for specific hazards or adaptation objectives.

**Underutilization of fine-grained data**: Despite the availability of rich local socioeconomic and environmental data, effective tools for integrating such data into regionalization are lacking.

**High technical barriers**: Data-driven regionalization requires geospatial analysis and programming skills, posing excessive difficulty for urban planners without relevant technical training.

Data-driven regionalization can generate spatially contiguous, homogeneous zones aligned with planning objectives by integrating multidimensional local data, yet implementation faces challenges including constraint complexity, interpretability requirements, and the iterative nature of decision-making. Advances in LLMs and Agentic AI offer new opportunities to support planners in overcoming these challenges.

## Method

### Overall Architecture

The system architecture comprises three core layers:

1. **Conversational interface**: Users specify the study area and hazard type via natural language.
2. **Central Planning Agent**: Orchestrates the entire regionalization workflow, including geocoding, data selection and configuration, feature visualization, user preference elicitation, and algorithm invocation.
3. **RepSC-SOM regionalization algorithm**: The core computational engine.

Workflow: The user specifies a region and hazard → the agent automatically selects relevant geospatial datasets → spatial features are visualized as interactive map layers → data characteristics are summarized and user preferences are elicited → RepSC-SOM is invoked → results are displayed on the map → user feedback drives iterative refinement.

### Key Designs

**RepSC-SOM three-stage pipeline (Embedding–Clustering–Refining):**

**Stage 1: Embedding**
- User-selected input features are projected into a high-dimensional latent space via an autoencoder.
- The autoencoder captures complex interactions and dependencies among input features.
- The SOM is initialized based on a geographic threshold: neuron count and initial states are determined using the semivariogram of the input feature data.

**Stage 2: Clustering**
- Taking the embedding output as input, grid cells in the study area are iteratively assigned to their best matching unit (BMU).
- BMU selection involves a two-stage filtering process:
    - Stage 1: Candidate SOM neurons within a geographic threshold are filtered using Haversine distance (spatial constraint).
    - Stage 2: The neuron with the greatest feature-space similarity among candidates is selected as the BMU.
- Each SOM neuron's weights are updated based on the features of grid cells that select it as their BMU.
- After multiple iterations, each grid cell is assigned the weight of its BMU.

**Stage 3: Refining**
- Post-processing is applied to the spatially constrained SOM output to improve spatial compactness and reduce fragmentation.
- Grid cells are first spatially partitioned into initial regions.
- A region-growing process then iteratively merges regions according to criteria that jointly consider:
    - The user's desired number of regions
    - Feature similarity
    - Spatial constraints

**Role of the Agentic AI:**
- **Feature recommendation**: Dynamically selects and recommends relevant features (socioeconomic indicators, environmental conditions, infrastructure vulnerability, etc.) from databases such as the Florida Geographic Data Library, based on the study area and hazard type.
- **Spatial constraint guidance**: Assists users in understanding and configuring spatial constraint parameters.
- **Interactive exploration support**: Maintains conversational context, responds to user feedback, and updates regionalization results.
- **Lowering technical barriers**: Enables non-technical users to perform advanced spatial analysis.

### Loss & Training

RepSC-SOM is an unsupervised method and does not involve a loss function in the conventional sense. Its optimization objectives are implicitly encoded in:
- The competitive learning mechanism of the SOM (minimizing the feature distance between the BMU and the input);
- The Haversine distance threshold of the spatial constraint;
- The similarity and compactness criteria in region-growing.

## Key Experimental Results

### Main Results

The paper presents a demonstration case study of flood risk in Jacksonville, Florida:

| Step | Description | Output |
|------|-------------|--------|
| User input | Specifies Jacksonville, FL and flooding hazard | Study area and hazard type |
| Feature selection | LLM recommends socioeconomic, environmental, and infrastructure vulnerability indicators | Candidate feature list and interactive map |
| Regionalization | User selects features and number of regions; RepSC-SOM is executed | Regionalization results on an interactive map |
| Human-AI iteration | User provides feedback to adjust features or configuration | Updated regionalization scheme |

As a workshop paper, the work focuses on system demonstration and proof-of-concept; no quantitative comparative experiments or performance metrics are provided.

### Ablation Study

The paper does not include ablation experiments in the traditional sense. However, each system component is amenable to independent evaluation:
- Autoencoder vs. raw features in the Embedding stage
- Impact of geographically adaptive thresholds (based on semivariogram) vs. fixed thresholds on region quality
- Degree of region fragmentation with vs. without the Refining stage
- User task efficiency and output quality with vs. without AI agent assistance

These ablation directions are not pursued in the current paper and constitute important avenues for future work.

### Key Findings

1. **Agentic AI effectively lowers the technical barrier**: Urban planners without programming backgrounds can perform advanced spatial analysis through natural language interaction.
2. **Data-driven regionalization outperforms traditional boundaries**: It better captures the spatial heterogeneity of disaster risk.
3. **Human-AI collaboration improves output quality**: The iterative feedback mechanism produces regions that reflect both computational rigor and users' domain knowledge.
4. **Good scalability of the system**: The framework is applicable to different cities and different types of disaster risk assessment.

## Highlights & Insights

1. **Organic integration of Agentic AI with traditional geospatial methods**: The LLM agent does not replace spatial analysis algorithms but acts as an "intelligent assistant" helping users navigate complex analytical workflows—a pragmatic and effective instantiation of the AI-in-the-loop paradigm.
2. **The three-stage RepSC-SOM design balances flexibility and interpretability**: The output of each stage is transparent and interpretable, allowing user intervention at any point, which aligns well with the iterative decision-making demands of planning practice.
3. **Demand-oriented rather than data-oriented**: The starting point of regionalization is the planning objective (disaster mitigation, resource allocation) rather than pure data clustering, making it more closely aligned with real-world planning needs.
4. **Adaptive design of spatial constraints**: Using semivariograms to determine geographic thresholds captures spatial autocorrelation more rigorously than fixed thresholds.

## Limitations & Future Work

1. **Proof-of-concept only**: As a workshop paper, it lacks systematic quantitative evaluation and does not compare against existing regionalization methods (e.g., SKATER, ClustGeo).
2. **Single case study**: The system is demonstrated only on a flood scenario in Jacksonville; generalizability to different urban structures and hazard types remains unvalidated.
3. **Reliability of the AI agent is unevaluated**: The appropriateness of LLM-recommended features and spatial constraint suggestions has not been systematically validated; the risk of AI misleading users is unaddressed.
4. **Scalability and performance not discussed**: Computational efficiency and memory consumption for large-scale urban areas (e.g., millions of grid cells) are not considered.
5. **No user study**: No planner user testing has been conducted to verify whether the system genuinely improves workflow efficiency or decision quality.
6. **Strong data dependency**: System effectiveness is highly contingent on the availability of high-quality local data, and applicability in data-scarce regions is unclear.

## Related Work & Insights

- **Urban agents**: Works such as CityGPT (Feng et al., 2025) that leverage LLMs for urban analysis; this paper extends that line of research to regionalization planning.
- **Data-driven regionalization**: Methods by Aydin et al. (2021) and Zhang et al. (2024) that generate homogeneous spatial zones from multidimensional data; this paper adds AI assistance and human-computer interaction on top of these foundations.
- **SOM in geospatial applications**: Self-organizing maps have a long history in geographic data clustering; the innovation here lies in the addition of representative initialization and adaptive spatial constraints.
- **Human-AI collaborative planning**: Research by Legacy et al. (2010) and Matern et al. (2020) on stakeholder participation and transparent decision-making; this paper realizes more efficient interaction through an AI agent.
- **Insights**: The application pattern of Agentic AI in professional domains (e.g., urban planning, healthcare)—augmenting rather than replacing domain experts—may represent an important direction for LLM deployment in practice. The spatial constraint approach of RepSC-SOM is generalizable to other spatial clustering problems.

## Rating

- Novelty: ⭐⭐⭐ The idea of combining Agentic AI with regionalization is reasonably novel, though the individual components (SOM, LLM dialogue) are not new in isolation.
- Experimental Thoroughness: ⭐⭐ Only a single demonstration case is provided, with no quantitative comparative experiments or user studies; understandable for a workshop paper but insufficient.
- Writing Quality: ⭐⭐⭐ The structure is clear and the system architecture is well described, though technical details (e.g., autoencoder configuration, SOM parameters) are insufficiently elaborated.
- Value: ⭐⭐⭐ Offers a valuable exploratory direction at the intersection of AI and urban planning, but requires more empirical evidence to substantiate practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)
- [\[NeurIPS 2025\] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects](what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)
- [\[NeurIPS 2025\] Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)

</div>

<!-- RELATED:END -->
