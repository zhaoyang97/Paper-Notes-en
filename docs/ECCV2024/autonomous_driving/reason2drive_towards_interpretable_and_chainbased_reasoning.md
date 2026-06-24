---
title: >-
  [Paper Note] Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][chain-based reasoning] This paper constructs the Reason2Drive benchmark dataset (600K+ video-text pairs, covering perception-prediction-reasoning chain tasks), proposes ADRScore as a new metric to evaluate the correctness of chain-based reasoning, and designs a Prior Tokenizer + Instructed Vision Decoder framework to enhance the object-level perception and reasoning capabilities of VLMs, significantly outperforming all baselines on autonomous d…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "chain-based reasoning"
  - "VLM"
  - "benchmark dataset"
  - "interpretable decision-making"
date: 2026-05-08
content_hash: f51f2797958c3de0
---

# Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2312.03661](https://arxiv.org/abs/2312.03661)  
**Code**: [https://github.com/fudan-zvg/Reason2Drive](https://github.com/fudan-zvg/Reason2Drive)  
**Area**: Autonomous Driving  
**Keywords**: autonomous driving, chain-based reasoning, VLM, benchmark dataset, interpretable decision-making

## TL;DR
This paper constructs the Reason2Drive benchmark dataset (600K+ video-text pairs, covering perception-prediction-reasoning chain tasks), proposes ADRScore as a new metric to evaluate the correctness of chain-based reasoning, and designs a Prior Tokenizer + Instructed Vision Decoder framework to enhance the object-level perception and reasoning capabilities of VLMs, significantly outperforming all baselines on autonomous driving reasoning tasks.

## Background & Motivation
1. **Background**: Large vision-language models (VLMs) have attracted widespread attention in autonomous driving due to their complex reasoning capabilities. Compared to end-to-end methods (which treat the system as a black box directly mapping sensor inputs to control signals) and rule-based methods (which rely heavily on hand-crafted rules), VLMs can provide explicit decision-making explanations, offering superior interpretability and generalization potential for autonomous driving.
2. **Limitations of Prior Work**: (a) Lack of datasets: Existing driving language datasets (e.g., Talk2Car, NuScenesQA, DriveLM) oversimplify complex driving processes into basic QAs (boolean answers or limited multiple choices) and lack reasoning chain annotations to explain the decision-making process. (b) Evaluation flaws: Traditional text metrics like BLEU and CIDEr only measure overall text quality without considering the causal relationship between reasoning steps and final conclusions, failing to assess whether the model's reasoning chain truly supports correct decisions.
3. **Key Challenge**: Autonomous driving is not a simple QA process but a multi-step chain decision-making process of perception $\rightarrow$ prediction $\rightarrow$ reasoning. However, neither existing datasets nor evaluation systems can support research on such chain-based reasoning.
4. **Goal**: (a) Construct a large-scale, multi-source autonomous driving benchmark with chain-of-reasoning annotations; (b) Design a metric specifically for evaluating the correctness of chain-of-reasoning; (c) Enhance the model's ability to utilize object-level perception priors to improve reasoning accuracy in VLMs.
5. **Key Insight**: Structure the autonomous driving decision-making process into a three-step chain of "perception $\rightarrow$ prediction $\rightarrow$ reasoning," systematically addressing the problem from dataset construction, evaluation metrics, and model architecture.
6. **Core Idea**: Systematically advance interpretable autonomous driving reasoning research via a large-scale chain-of-reasoning dataset, a causal-aware evaluation metric, and a perception-prior-enhanced VLM architecture.

## Method

### Overall Architecture
**Dataset Level**: Parse annotations from three public datasets (nuScenes, Waymo, ONCE) to construct an object-centric database $\rightarrow$ manually design question templates to generate QA pairs for perception, prediction, and reasoning $\rightarrow$ leverage GPT-4 for validation and diversity enhancement $\rightarrow$ produce a final set of 632,955 video-text pairs.

**Evaluation Level**: Propose the ADRScore assessment framework, consisting of three dimensions: Reasoning Alignment (RA), Redundancy (RD), and Missing Steps (MS), aggregated as $ADRScore = \frac{1}{3}(RA + RD + MS)$, along with its variant ADRScore-S which accounts for the precision of visual elements.

**Model Level**: Built upon the InstructBLIP architecture. Inputs: video frame sequence + perception priors (object locations and motion info) $\rightarrow$ Vision Encoder (EVA-CLIP ViT-G/14) extracts image features $\rightarrow$ Prior Tokenizer (2-layer MLP + RoIAlign + positional encoding) extracts perception prior features $\rightarrow$ Q-Former aligns features to the text space $\rightarrow$ LLM generates text answers (containing special tokens `<LOC>` and `<MOT>`) $\rightarrow$ Instructed Vision Decoder decodes target object locations and trajectory predictions from token embeddings.

### Key Designs

1. **Reason2Drive Dataset**:
    - **Function**: Build the largest language-driven autonomous driving reasoning benchmark.
    - **Mechanism**: Parse driving annotations into an object-centric database (storing object classes, attributes, locations, motions, etc. per frame), then combine them with manual templates to generate QA pairs for three tasks. Tasks are divided into object-level and scene-level granularities:
     - **Perception tasks** (39%, 246K): Identify object classes, attributes (motion state, distance), locations, etc.
     - **Prediction tasks** (34%, 216K): Predict future trajectories of objects, cutting in/out of lanes, turning directions, etc.
     - **Reasoning tasks** (27%, 171K): Analyze current perception and prediction states to derive driving policies and risk assessments.
    - **Design Motivation**: Existing datasets cover at most 13K samples and are restricted to simple QAs, whereas driving decision-making requires understanding the complete causal chain of "what is seen $\rightarrow$ what will happen $\rightarrow$ what should be done."

2. **ADRScore Metric**:
    - **Function**: Evaluate the causal correctness of the generated reasoning chain, rather than just text similarity.
    - **Mechanism**: Align generated reasoning steps $\vec{h}=\{h_1,...,h_N\}$ with ground-truth steps $\vec{r}=\{r_1,...,r_K\}$. The alignment value is calculated based on the cosine similarity of BERT sentence embeddings. Three sub-metrics are defined:
     - **Reasoning Alignment (RA)** = Average alignment (matching degree between generated and ground-truth steps).
     - **Redundancy (RD)** = Minimum alignment (penalizes redundant, irrelevant steps).
     - **Missing Steps (MS)** = Minimum value of reverse alignment (detects omitted key steps).
     - **ADRScore-S**: When steps contain visual elements (e.g., location coordinates, trajectories), geometric MSE replaces semantic similarity to evaluate spatial reasoning more strictly.
    - **Design Motivation**: Traditional metrics like BLEU/CIDEr show low discriminative power for reasoning capability differences—models with different reasoning qualities show minimal gaps, failing to establish an effective benchmark.

3. **Prior Tokenizer**:
    - **Function**: Encode object-level visual priors (locations, motion information) into tokens understandable by the LLM.
    - **Mechanism**: Use RoIAlign to extract region-level features $f_r$ from image features. Map geometric position and motion details to the same dimension using a positional encoding function $E(\cdot)$, then add them and project through a 2-layer MLP: $f_p = F_p(f_r + E(P))$. Finally, align perception prior tokens with visual tokens using Q-Former into the text space.
    - **Design Motivation**: Directly inputting coordinates/trajectories as text into LLMs causes info loss (textual descriptions struggle to capture full spatial details of dynamic scenes), whereas encoding at the visual feature level is more efficient and accurate.

4. **Instructed Vision Decoder**:
    - **Function**: Decode precise object locations and motion trajectory predictions from the LLM outputs.
    - **Mechanism**: Extend the LLM vocabulary with two special tokens, `<LOC>` and `<MOT>`. When the LLM needs to output perception/prediction results, it outputs these tokens. Their last-layer features are extracted and projected via MLP to get the hidden embedding $f_h$, which is then fed into a Transformer decoder along with visual features: $\hat{P} = D(f_v, f_h)$. The decoder includes a feature-alignment layer and task-specific heads (object detection and trajectory prediction).
    - **Design Motivation**: Language-only models cannot output accurate perception results as decoders, yet perception accuracy is a prerequisite for reliable reasoning in driving scenarios. Inspired by LISA, this design directly embeds perception capabilities into the multimodal LLM.

### Loss & Training
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{txt} + \lambda_{per}\mathcal{L}_{per}$, where $\lambda_{per}=1.0$
- **Text Loss** $\mathcal{L}_{txt}$: Autoregressive cross-entropy loss
- **Perception Loss** $\mathcal{L}_{per}$: Binary cross-entropy (classification) + MSE (regression), $\lambda_{reg}=0.25$
- **Two-stage Training**:
    - Pre-training stage: Initialized from InstructBLIP. Freeze the LLM and Vision Encoder, and train the Prior Tokenizer, Q-Former, and Instructed Vision Decoder.
    - Fine-tuning stage: Efficiently fine-tune the LLM with LoRA. Freeze the Vision Encoder and Prior Tokenizer, and perform full fine-tuning on the Instructed Vision Decoder.
- Training configurations: AdamW optimizer, weight decay 0.01, cosine learning rate schedule (max 3e-4), batch size of 8 on 8 $\times$ V100 GPUs.

## Key Experimental Results

### Main Results

| Method | LLM | ADRScore | ADRScore-S | B@4 | METEOR | ROUGE | CIDEr |
|------|-----|----------|------------|------|--------|-------|-------|
| Blip-2 | OPT-2.7B | 0.296 | 0.162 | 0.361 | 0.249 | 0.443 | 0.174 |
| Blip-2 | FlanT5-XL | 0.310 | 0.171 | 0.368 | 0.256 | 0.451 | 0.187 |
| InstructBLIP | FlanT5-XL | 0.329 | 0.187 | 0.376 | 0.269 | 0.462 | 0.196 |
| InstructBLIP | Vicuna-7B | 0.351 | 0.214 | 0.408 | 0.294 | 0.484 | 0.211 |
| MiniGPT-4 | Vicuna-7B | 0.338 | 0.203 | 0.396 | 0.286 | 0.475 | 0.219 |
| Ours | FlanT5-XL | 0.457 | 0.420 | 0.451 | 0.349 | 0.520 | 0.292 |
| **Ours** | **Vicuna-7B** | **0.463** | **0.432** | **0.457** | **0.356** | **0.529** | **0.298** |

### Ablation Study

**Task Combination Ablation**:

| Perception | Prediction | Reasoning | ADRScore | ADRScore-S |
|------------|------------|-----------|----------|------------|
| ✓ | | | 0.282 | 0.253 |
| ✓ | ✓ | | 0.297 | 0.264 |
| | | ✓ | 0.351 | 0.323 |
| ✓ | | ✓ | 0.407 | 0.364 |
| ✓ | ✓ | ✓ | **0.463** | **0.432** |

**Visual Inputs and Perception Priors Ablation**:

| Image-level | Video-level | Region-level | Positional Encoding | ADRScore | ADRScore-S |
|------------|------------|-----------|----------|----------|------------|
| ✓ | | | | 0.414 | 0.379 |
| | ✓ | | | 0.431 | 0.394 |
| | ✓ | ✓ | | 0.447 | 0.418 |
| | ✓ | ✓ | ✓ | **0.463** | **0.432** |

**Perception & Prediction Quality**:

| Prediction Type | Metric | MiniGPT-4 | Kosmos-2 | Ours |
|----------|------|-----------|----------|------|
| Bounding box | Accuracy | 0.723 | 0.745 | **0.806** |
| Trajectory | ADE | 2.334 | 2.563 | **1.875** |

### Key Findings
1. **Reasoning data is the most critical**: Training solely on reasoning tasks (0.351 ADRScore) performs far better than training solely on perception (0.282) or perception+prediction (0.297). This indicates that reasoning data is vital for instruction tuning, while perception and prediction data provide extra gains of +4.1% and +6.8%, respectively.
2. **ADRScore has far higher discriminative power than traditional metrics**: The performance gap among different models evaluated by BLEU/CIDEr is minimal (0.361 $\rightarrow$ 0.457), but is highly significant when using ADRScore (0.296 $\rightarrow$ 0.463) and even larger with ADRScore-S (0.162 $\rightarrow$ 0.432). This effectively resolves the issue of ambiguous evaluations.
3. **Visual prior encoding brings significant gains**: The step-by-step introduction of region-level features (+2.4% ADRScore-S) and positional encoding (+1.4%) consistently boosts performance, validating the importance of perception priors.
4. **Perception accuracy drives reasoning quality**: Object detection accuracy reaches 80.6% and trajectory ADE is 1.875, significantly outperforming baselines. This proves that accurate perception is the cornerstone of reliable reasoning.
5. **Strong generalization ability**: When trained on nuScenes and evaluated on Waymo+ONCE, the ADRScore-S of the proposed model only drops from 0.443 to 0.385, whereas the baseline model (MiniGPT-4) plummets from 0.263 to 0.130.
6. **Downstream planning tasks benefit**: Fine-tuning control signal prediction after pre-training on Reason2Drive reduces the velocity RMSE from 3.743 to 2.842 and the steering angle RMSE from 5.926 to 4.866. This demonstrates that chain-of-reasoning training effectively enhances downstream planning capabilities.

## Highlights & Insights
1. **Systematic contribution**: Advances autonomous driving reasoning research via a three-pronged framework consisting of the dataset (Reason2Drive), evaluation metric (ADRScore), and model architecture (Prior Tokenizer + Instructed Vision Decoder).
2. **Ingenious design of ADRScore**: Deconstructs reasoning chain evaluation into alignment, redundancy, and missing steps. ADRScore-S further replaces semantic similarity with geometric errors to assess spatial reasoning, which is highly suited for driving scenarios.
3. **Perception-reasoning co-enhancement**: The Prior Tokenizer and Instructed Vision Decoder form a closed loop of perception $\rightarrow$ reasoning $\rightarrow$ perception. Reasoning depends on perception, and during the reasoning process, even more precise perception results are output.
4. **Reusable dataset construction methodology**: The workflow of using an object-centric database + templated QA generation + GPT-4 enhancement is highly transferable to other domains requiring structured annotation.

## Limitations & Future Work
1. **Upper bound on automatic annotation quality**: Since the dataset is automatically generated using templates and GPT-4, the diversity and depth of reasoning chains are limited by the template designs. More complex causal reasoning may still require manual annotations.
2. **Difficulty in perceiving distant objects**: The paper acknowledges that the VLM's ability to identify distant risk objects remains insufficient, which is a critical safety issue in high-speed driving scenarios.
3. **Impact of ego-motion**: Relative motion caused by ego-vehicle displacement easily interferes with determining object motion status (e.g., misclassifying a stopped object as moving), requiring improved motion compensation mechanisms.
4. **BERT dependency in evaluation metrics**: The semantic similarity in ADRScore is based on BERT sentence embeddings, which may not be the optimal semantic model for the driving domain. Encoders fine-tuned on driving-specific domains could be explored.
5. **Lack of multi-sensor fusion**: The model utilizes only front-facing camera images. Real-world driving requires multi-source information fusion, including LiDAR, multi-camera views, and GPS.

## Related Work & Insights
- **DriveLM** [Contributors, 2023]: Constructs a driving scene VQA dataset but only covers perception information and lacks reasoning chains. Reason2Drive builds upon this by incorporating complete decision-making reasoning processes.
- **LISA** [Lai et al., 2023]: Embeds segmentation capabilities into multimodal LLMs, inspiring the design of the Instructed Vision Decoder to directly integrate perception capabilities into the VLM.
- **ROSCOE** [Golovneva et al., 2022]: Proposes an evaluation metric framework for reasoning chains, inspiring the design of ADRScore.
- **ADriver-I** [Jia et al., 2023]: A general world model for autonomous driving, inspiring the downstream validation experiments for control signal prediction.
- **Key Insight**: The chain-of-reasoning framework of Reason2Drive can be extended to other scenarios requiring multi-step decision-making (e.g., robotic manipulation, medical diagnosis). Similarly, the design paradigm of ADRScore can be migrated to evaluate any chain-of-reasoning tasks.

## Rating
- ⭐⭐⭐⭐ Novelty: The dataset and evaluation metric design are innovative, and the Prior Tokenizer is clever, though the overall model architecture is relatively standard.
- ⭐⭐⭐⭐⭐ Experimental Thoroughness: Highly comprehensive, featuring main experiments, multi-dimensional ablations (task combinations, visual inputs, decoders, encoders, generalization, downstream tasks), and GPT-4 evaluations.
- ⭐⭐⭐⭐ Writing Quality: The paper has a clear structure, rich diagrams, and detailed expositions on dataset construction and evaluation metrics.
- ⭐⭐⭐⭐⭐ Value: The systematic contributions (dataset + metric + method) play a major role in driving forward autonomous driving VLM research, with the datasets and metrics being widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](../../CVPR2026/autonomous_driving/minddriver_introducing_progressive_multimodal_reasoning_for_autonomous_driving.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving](../../CVPR2026/autonomous_driving/drivecombo_benchmarking_compositional_traffic_rule_reasoning_in_autonomous_drivi.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->
