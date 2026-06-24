---
title: >-
  [Paper Note] How to Merge Your Multimodal Models Over Time?
description: >-
  [CVPR 2025][Optimization][Model Merging] This paper proposes the TIME (Temporal Integration of Model Expertise) framework to systematically study the progressive merging of multimodal expert models over time. By defining a search space across three axes—initialization strategy, deployment strategy, and merging technique, the work uncovers key design principles for temporal model merging on the FoMo-in-Flux benchmark.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Model Merging"
  - "Temporal Model Merging"
  - "Multimodal Learning"
  - "Continual Learning"
  - "Foundation Models"
date: 2026-05-08
content_hash: 668abae2055a65e3
---

# How to Merge Your Multimodal Models Over Time?

**Conference**: CVPR 2025  
**arXiv**: [2412.06712](https://arxiv.org/abs/2412.06712)  
**Code**: [https://github.com/explainableml/time](https://github.com/explainableml/time)  
**Area**: Multimodal VLM  
**Keywords**: Model Merging, Temporal Model Merging, Multimodal Learning, Continual Learning, Foundation Models

## TL;DR
This paper proposes the TIME (Temporal Integration of Model Expertise) framework to systematically study the progressive merging of multimodal expert models over time. By defining a search space across three axes—initialization strategy, deployment strategy, and merging technique, the work uncovers key design principles for temporal model merging on the FoMo-in-Flux benchmark.

## Background & Motivation

**Background**: Model merging integrates multiple expert models, fine-tuned from the same foundation model, into a unified model to obtain multi-task capabilities without extra training. Existing merging methods (e.g., Task Arithmetic, TIES, DARE) have demonstrated success in static scenarios.

**Limitations of Prior Work**: Almost all existing model merging research assumes that all expert models are available simultaneously—meaning all experts are collected at once and then merged. In reality, however, new tasks and domains emerge progressively over time: an image classification expert today, an OCR expert tomorrow, and a medical imaging expert the day after. The issues arising from this temporal nature have not been systematically studied.

**Key Challenge**: In sequential settings, key design choices become uncertain: (1) When training a new expert, should the fine-tuning start from the foundation model or the currently merged model? (2) Should all models be merged at each timestep, or only the new ones? (3) Should the final deployment use the currently merged version or re-merge all experts from scratch? The answers to these questions are not obvious, and prior works lack systematic research in this area.

**Goal**: To build a unified framework for temporal model merging and systematically answer these design questions to provide best practices for practitioners.

**Key Insight**: Define three orthogonal design axes—initialization, deployment, and merging techniques—to form a complete search space, identifying the optimal configuration through large-scale experiments.

**Core Idea**: The TIME framework decomposes temporal model merging into three independent decision dimensions. Through systematic experiments on the FoMo-in-Flux multimodal benchmark, it reveals the crucial impacts of the initialization strategy (Base Init vs. Merged Init) and deployment strategy (Continual Merging vs. Full Re-merging) on the final performance.

## Method

### Overall Architecture
Consider a sequence of expert model training tasks arriving chronologically: $\{T_1, T_2, \ldots, T_N\}$. At each timestep $t$, one must: (1) choose how to initialize the fine-tuning of the new expert (I-axis), (2) train the new expert, (3) choose how to merge the new expert with existing knowledge (M-axis), and (4) decide which model to deploy (D-axis). The TIME framework clearly defines the available options for each axis.

### Key Designs

1. **Initialization Phase**:

    - **Function**: Determines the starting weights when training experts for new tasks.
    - **Mechanism**: Two primary strategies are investigated—(a) **Base Init**: each new expert is fine-tuned starting from the original foundation model $\theta_0$, ensuring that each expert's task vector $\tau_i = \theta_i - \dot{\theta}_0$ is computed relative to the same anchor point, which facilitates subsequent merging; (b) **Merged Init**: fine-tunes the new expert starting from the current merged model $\theta_t^{merged}$ at that timestep, leveraging previously learned knowledge to reduce training costs and potentially promote forward transfer. However, this causes the reference base of the task vectors to drift continuously, potentially degrading merging quality.
    - **Design Motivation**: The initialization strategy directly affects the structure of the task vector space. Base Init ensures that the orthogonality assumptions of task vectors are closer to being satisfied, whereas Merged Init sacrifices this for potentially better few-shot performance.

2. **Deployment Phase**:

    - **Function**: Decides what model is actually deployed at each timestep.
    - **Mechanism**: (a) **Continual Merging**: progressively merges each new expert into the current model as they arrive and deploys the currently merged result; (b) **Full Re-merge**: merges all expert models from scratch whenever a new expert arrives (more expensive but potentially more accurate); (c) **Deploy Latest Expert Only**: deploys only the latest expert model each time (acting as a baseline). The deployment strategy also dictates whether to retain all historical expert weights (storage cost).
    - **Design Motivation**: Continual Merging is highly efficient but may accumulate merging errors, whereas Full Re-merge is theoretically superior but incurs high computational and storage overheads.

3. **Merging Technique**:

    - **Function**: Specific algorithms for merging model parameters.
    - **Mechanism**: Several merging methods are experimentally compared in temporal scenarios: (a) **Weight Averaging**: simple parameter average; (b) **Task Arithmetic**: $\theta_{merged} = \theta_0 + \lambda \sum_i \tau_i$; (c) **TIES-Merging**: reduces interference between task vectors through sign consistency and pruning; (d) **DARE**: randomly drops partial dimensions in task vectors to reduce conflict; (e) **Fisher Merging**: weights parameters using the Fisher Information Matrix. Each method may behave differently in temporal progressive scenarios compared to static ones.
    - **Design Motivation**: To systematically test which merging techniques are most robust in temporal progressive scenarios.

### Loss & Training
The framework itself does not introduce new training methodologies; each expert uses standard fine-tuning. The core contribution lies in the systematic framework and large-scale experimental design. Experiments are conducted on the FoMo-in-Flux benchmark, covering different model scales (ViT-B/32, ViT-B/16, ViT-L/14), computational budgets, and learning horizons.

## Key Experimental Results

### Main Results: Comparison of Different TIME Configurations

| Initialization | Deployment | Merging Technique | Average Accuracy | Forgetting Rate | Description |
|---|---|---|---|---|---|
| Base | Re-merge | Task Arithmetic | **Highest** | Lowest | Overall optimal configuration |
| Base | Continual | Task Arithmetic | Suboptimal | Low | Most efficient |
| Merged | Continual | Task Arithmetic | Medium | Medium | Drift leads to performance degradation |
| Merged | Re-merge | Task Arithmetic | Above medium | Medium | Partially mitigates drift |
| Base | Re-merge | Simple Average | Relatively low | Low | Simple average is insufficient |
| Base | Re-merge | TIES | Relatively high | Low | Close to Task Arithmetic |

### Model Scale Impact

| Model | Base+Re-merge vs Merged+Continual Gap | Description |
|---|---|---|
| ViT-B/32 | Large | Smaller models are more sensitive to initialization strategies |
| ViT-B/16 | Medium | Gap narrows at medium scale |
| ViT-L/14 | Small | Large models are more robust to strategy choices |

### Key Findings
- **Base Init significantly outperforms Merged Init**: Initializing each expert from the foundation model is the most important design choice. Merged Init causes anchoring drift of task vectors, which severely compromises merging quality in later stages.
- **Full Re-merge outperforms Continual Merging**: Although costlier, merging all experts from scratch avoids error accumulation inherent in progressive merging. The gap becomes more pronounced over longer horizons.
- **Task Arithmetic remains optimal in temporal contexts**: Despite the advantages of TIES and DARE in static settings, the simplicity of Task Arithmetic becomes an asset in temporal scenarios, as fewer hyperparameters mean less inconsistency across timesteps.
- **Larger scale partially mitigates strategy sensitivity**: Larger models are more robust to the choice of initialization and deployment strategies, though the ranking of optimal strategies remains unchanged.
- **Interaction effects between initialization and deployment**: The performance of the Base Init + Re-merge combination is not merely a linear addition of their individual benefits; a positive interaction effect exists.

## Highlights & Insights
- **Defines an important new problem**: Temporal model merging is a core real-world requirement that has not been systematically studied before. The TIME framework clearly parameterizes the problem space into three orthogonal axes, making the design space searchable and the insights transferable.
- **Counterintuitive yet crucial findings on Base Init**: Intuitively, initializing a new expert from a merged model should utilize prior knowledge to accelerate training. However, experiments demonstrate that this disrupts the structure of task vector space, yielding larger performance penalties during merging. This provides valuable guidance for all scenarios employing task arithmetic.
- **Generality of the framework**: TIME is not limited to CLIP; it applies to any "foundation model + multi-expert fine-tuning + merging" pipeline, including LLM LoRA merging.

## Limitations & Future Work
- Experiments are mainly conducted on CLIP (multimodal vision-language) models and have not been validated on generative models (Stable Diffusion) or pure language models (LLMs).
- While the FoMo-in-Flux benchmark is diverse, the task scale (~dozens of tasks) may still be small compared to real-world scenarios.
- The correlation between expert tasks is not considered—whether highly correlated tasks and completely unrelated tasks should be treated differently during merging remains unexplored.
- While Full Re-merge performs well, it requires storing all historical expert weights, causing storage overhead to scale linearly over time.
- Adaptive strategies—dynamically choosing initialization and merging methods based on the characteristics of the current timestep—were not explored.

## Related Work & Insights
- **vs Task Arithmetic original paper**: Task Arithmetic only considers static, one-time merging. TIME proves that its conclusions still hold in temporal scenarios, but the optimal practice (Base Init + Re-merge) is a novel discovery.
- **vs Continual Learning (CL)**: CL focuses on forgetting but typically assumes sequential training of a single model. Temporal merging preserves all experts, leading to different forgetting patterns and mitigation solutions.
- **vs Model Soups/DARE/TIES**: These methods focus on superior merging techniques. TIME demonstrates that in temporal scenarios, the choice of initialization and deployment strategies is more crucial than the merging technique itself.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically define and study the temporal model merging problem, with a clearly designed framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive combinatorial experiments covering multiple model scales, merging techniques, and initialization/deployment strategies.
- Writing Quality: ⭐⭐⭐⭐ Both the framework and experimental designs are clearly presented with rich charts.
- Value: ⭐⭐⭐⭐ Direct practical guidance for model merging practitioners, particularly the Base Init + Re-merge best practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/optimization/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](../../NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)

</div>

<!-- RELATED:END -->
