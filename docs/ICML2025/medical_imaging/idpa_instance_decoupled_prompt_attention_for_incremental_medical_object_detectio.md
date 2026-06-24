---
title: >-
  [Paper Note] iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection
description: >-
  [ICML 2025][Medical Imaging][Incremental Object Detection] This work proposes the iDPA framework, which implements Incremental Medical Object Detection (IMOD) on a frozen vision-language object detection model through two key modules: Instance-level Prompt Generation (IPG) and Decoupled Prompt Attention (DPA). By training only 1.4% of the parameters, it highly outperforms SOTA methods across 13 cross-modal medical datasets.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Incremental Object Detection"
  - "Continual Learning"
  - "Prompt Tuning"
  - "Multimodal Fusion"
  - "Medical Object Detection"
date: 2026-05-08
content_hash: fe2de2836dccfc34
---

# iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection

**Conference**: ICML 2025  
**arXiv**: [2506.00406](https://arxiv.org/abs/2506.00406)  
**Code**: [https://github.com/HarveyYi/iDPA](https://github.com/HarveyYi/iDPA)  
**Area**: Medical Imaging  
**Keywords**: Incremental Object Detection, Continual Learning, Prompt Tuning, Multimodal Fusion, Medical Object Detection

## TL;DR

This work proposes the iDPA framework, which implements Incremental Medical Object Detection (IMOD) on a frozen vision-language object detection model through two key modules: Instance-level Prompt Generation (IPG) and Decoupled Prompt Attention (DPA). By training only 1.4% of the parameters, it highly outperforms SOTA methods across 13 cross-modal medical datasets.

## Background & Motivation

**Core Problem**: Medical object detection requires continuous learning of new disease categories (such as novel lesions or new organs). However, training individual detectors for each task is highly inefficient, while joint training is impractical due to the impossibility of predefining all medical concepts. Continual Learning (CL) is an ideal solution but suffers from the **catastrophic forgetting** problem.

**Limitations of Prior Work**:

**Foreground-background coupling of global prompts**: Existing prompt-based CL methods (such as L2P, DualPrompt, CODA-Prompt, etc.) are designed for classification and encode knowledge using global prompts, which mix foreground and background information. However, object detection requires fine-grained instance-level information. Excess background noise in global prompts can interfere with localization and classification, especially in medical imaging where high similarity across different modalities (CT, MRI, X-ray) easily leads to category confusion.

**Coupled attention between prompts and image-text tokens**: Concatenating prompts directly in front of image/text tokens causes the prompt information to be diluted because the sequence length of image-text tokens far exceeds that of the prompts. This hinders the learning of task-specific knowledge. Furthermore, such concatenation introduces extra interference between visual and textual prompts during the cross-modal interaction of the vision-language model.

**Improper knowledge injection location**: Existing methods inject prompts only into the backbone layer. However, the fine-grained reasoning required for detection occurs in the fusion layers after the backbone, limiting the effectiveness of prompt-based tuning.

## Method

### Overall Architecture

iDPA is built upon a frozen pre-trained Vision-Language Object Detection (VLOD) model (such as GLIP) and consists of two core modules:

- **IPG (Instance-level Prompt Generation)**: Decouples instance-level features from training images to generate prompts focused on dense prediction.
- **DPA (Decoupled Prompt Attention)**: Decouples the original prompt attention to achieve more efficient knowledge injection within the cross-modal fusion encoder.

**Mechanism**: For each incremental task $\mathcal{T}_i$, IPG first extracts instance-level representations from the training data to generate prompts. Then, DPA injects these prompts into the fusion encoder of the frozen model. After training, the generated prompts are stored in a Prompt Pool. During inference, the corresponding prompt is selected based on query-key matching.

### Key Designs

#### 1. Instance-level Prompt Generation (IPG)

**Step 1: Decoupling Instance Features**

For each category $c$ in each task $\mathcal{T}_i$, the frozen model is used to extract image features. RoI Pooling combined with enlarged bounding boxes (scaling factor $\gamma = 1.3^2$) is employed to extract $M$ instance-level feature representations:

$$v_c^{(j)} = \text{RoIPool}(\Phi(\text{Img}, \text{Text}), \gamma b)$$

Where $M=1000$ (full-data setting) or $M=m$ (few-shot setting, with $m$ being the number of available samples per category). Enlarging the bounding boxes captures extra contextual information.

**Step 2: Continual Concept-aware and Knowledge Integration (CCPKI)**

Leverages a cross-attention mechanism to decouple $l$ concepts (where $l$ is the prompt length) from $M$ instance representations, generating task-specific prompts:

$$\dot{p_i} = \text{softmax}\left(\frac{p_i (\mathcal{W}_k \mathcal{I}_i)}{\sqrt{d}}\right)(\mathcal{W}_v \mathcal{I}_i)$$

$$\ddot{p_i} = p_i + \alpha \cdot \sigma(\tau \cdot \dot{p_i})$$

Key design components:
- **Query-Answer Framework**: The initial prompt $p_i \in \mathbb{R}^{l \times d}$ acts as the query, while the instance features $\mathcal{I}_i$ are projected as key-value pairs via $\mathcal{W}_k, \mathcal{W}_v$.
- **Learnable Scaling Factor** $\tau \in \mathbb{R}^{l \times 1}$: Dynamically adjusts the weights of different concepts to adapt to various concepts that may be involved in different tasks.
- **Non-linear Activation** $\sigma(\cdot)$ (tanh): Filters and enhances meaningful concept components.
- **Residual Connection**: Adds the activated concepts, scaled by $\alpha \in \mathbb{R}^{1 \times d}$, to the initial prompt.
- **Cross-task Initialization**: The CCPKI parameters for the $i$-th task are inherited from the $(i-1)$-th task, enabling knowledge transfer.

#### 2. Decoupled Prompt Attention (DPA)

**Design Motivation**: The authors mathematically analyze the behavior of conventional Prompt Attention (PA) in multimodal fusion. After concatenating prompts $p_v, p_t$ with visual/textual features, the attention output can be decomposed into four terms:

- $\text{Attn}_{v \to t}(f_v, f_t)$: Original visual-textual interaction (**Retained**)
- $\text{Attn}_{v \to t}(p_v, f_t)$: Prompt-to-textual knowledge injection (**Retained**)
- $\text{Attn}_{v \to t}(f_v, p_t)$: Attention output to prompt tokens (**Discarded**, as it confuses textual features)
- $\text{Attn}_{v \to t}(p_v, p_t)$: Interaction between prompts (**Discarded**, as it is redundant and increases computation)

**Final Form of DPA**:

$$\tilde{f_t} = f_t + \text{Attn}_{v \to t}(f_v, f_t) + \text{Attn}_{v \to t}(p_v, f_t)$$

$$\tilde{f_v} = f_v + \text{Attn}_{t \to v}(f_t, f_v) + \text{Attn}_{t \to v}(p_t, f_v)$$

This retains three key components:
1. **V↔T**: Original vision-language mutual enhancement.
2. **$P_t \to V$**: Textual prompt-to-visual knowledge injection.
3. **$P_v \to T$**: Visual prompt-to-textual knowledge injection.

**Advantages of DPA**:
- Separates prompt and token representations, accelerating knowledge injection.
- Prevents prompt information from being diluted by long-sequence tokens.
- Preserves the original category distribution, mitigating catastrophic forgetting.
- Reduces computational complexity and GPU memory footprint.

#### 3. Fusion Encoder-level Knowledge Injection

Innovatively extends the location of knowledge injection from the backbone-level to the cross-modal fusion encoder. Because the fine-grained reasoning required for detection primarily occurs during the fusion phase, this placement allows prompt information to influence localization and classification decisions more directly.

### Loss & Training

- **Basic Detection Loss**: Uses the standard detection loss (classification loss + localization loss) of the VLOD model (e.g., GLIP).
- **Freezing Strategy**: Freezes all parameters of the pre-trained VLOD model, only training the IPG and DPA modules (comprising approximately 1.4% of the parameters).
- **Incremental Training**: Trains new prompts for each new task, with CCPKI parameters initialized via inheritance from the previous task.
- **Prompt Pool Management**: Stores the generated prompts into the Pool after training, and selects them during inference via cosine similarity matching.
- **Exemplar-Free**: Eliminates the need to store samples from previous tasks, reducing privacy concerns and storage overhead.

## Key Experimental Results

### Dataset: ODinM-13

The authors collected 13 clinical, cross-modal, multi-organ, and multi-category medical datasets to establish the ODinM-13 benchmark, covering:
- Multiple imaging modalities: CT, MRI, X-ray, PET, dermoscopy, etc.
- Multiple organs and diseases: Diabetic Foot Ulcers (DFUC), gastrointestinal lesions (Kvasir), optic nerve (OpticN), blood cells (BCCD), cell mitosis (CPM-17), breast cancer (BreastC), tuberculosis (TBX11K), kidney tumors (KidneyT), lung nodules (Luna16), Alzheimer's (ADNI), meningioma (Meneng), breast tumors (BreastT), thyroid nodules (TN3k).

### Main Results

| Method | FAP (%) ↑ | CAP (%) ↑ | FFP (%) ↓ | Type |
|------|-----------|-----------|-----------|------|
| Zero-shot | 3.12 | - | - | Baseline |
| Joint (Upper) | 54.67 | - | - | Upper Bound |
| Sequential | 4.40 | 15.87 | 57.81 | Non-Prompt |
| ER | 39.91 | 48.73 | 19.25 | Non-Prompt |
| ZiRa | 3.66 | 16.37 | 49.67 | Non-Prompt |
| L2P | 39.88 | 46.04 | 8.24 | Prompt |
| DualPrompt | 28.89 | 42.24 | 20.57 | Prompt |
| S-Prompt | 41.02 | 46.70 | 8.87 | Prompt |
| CODA | 42.08 | 49.78 | 2.80 | Prompt |
| NoRGa | 44.84 | 49.90 | 4.92 | Prompt |
| **iDPA (Ours)** | **50.28** | **54.10** | **2.48** | Prompt |

**Under the full-data setting, iDPA achieves an FAP of 50.28%, outperforming the best baseline method NoRGa by 5.44 percentage points**, while maintaining the lowest forgetting rate (FFP) of only 2.48%.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| Without IPG, using randomly initialized prompts | FAP drops significantly | Instance-level feature decoupling is crucial for object detection |
| Without DPA, using traditional PA | FAP drops, FFP rises | Coupled attention dilutes prompt information |
| Injecting prompts only in the backbone | Performance worse than fusion layer injection | Fusion encoder is a better injection location |
| Without CCPKI cross-task initialization | Learning efficiency of new tasks drops | Cross-task knowledge transfer is effective |
| Without scaling factor τ | Concept adjustment capability drops | Dynamic concept weights are important for multi-task adaptation |

### Key Findings

1. **Significant breakthrough on the CPM-17 dataset**: iDPA achieves 36.54% AP on cell mitosis detection (CPM-17), vastly outperforming all prompt-based methods (which achieve at most 8.37%). This demonstrates the significant advantage of instance-level feature decoupling in fine-grained detection.
2. **Greater advantage in few-shot scenarios**: In the 10-shot setting, FAP improves by 12.88%, indicating that instance-level knowledge is more valuable in data-scarce scenarios.
3. **Extremely low forgetting rate**: FFP is only 2.48%, thanks to DPA preserving the original category distribution.
4. **Only 1.4% parameters**: Outstanding parameter efficiency makes the method highly scalable to more tasks.
5. **Cross-modal generalization**: Exhibits stable performance across multiple modalities including CT, MRI, X-ray, PET, and dermoscopy.

## Highlights & Insights

1. **Theory-driven architectural design**: By mathematically analyzing the equivalent forms of Prompt Attention, the authors discover that only two of the four attention terms are useful for detection, leading to the design of DPA. This "analysis-first, design-later" paradigm is highly instructive.
2. **Instance-level vs. Global-level Prompts**: First to point out the fundamental flaws of global prompts in object detection tasks, proposing a scheme to generate prompts by decoupling concepts from instance features.
3. **Insights into knowledge knowledge injection locations**: Discovers that the fusion encoder is more suitable for injecting detection knowledge than the backbone, as the fusion stage is where fine-grained reasoning actually happens.
4. **ODinM-13 Benchmark**: Constructs the first large-scale incremental medical object detection benchmark, covering 13 cross-modal datasets, thus filling the gap in evaluation standards within this field.
5. **Exemplar-Free**: Avoids storing previous task samples, bypassing the privacy issues associated with medical data.

## Limitations & Future Work

1. **Linear growth of the Prompt Pool**: Each new task adds a set of prompts, leading to a linear outer expansion of the pool as the number of tasks increases. Future work could explore prompt compression or sharing mechanisms.
2. **Dependence on pre-trained VLOD quality**: The framework is built upon pre-trained models from the natural domain (e.g., GLIP), meaning the quality of their initial representations in the medical domain directly affects the upper performance bound.
3. **RoI Pooling requires bounding box annotations**: The IPG module relies on bounding box annotations of the training set to extract instance features, making it sensitive to annotation quality.
4. **Fixed scaling factor $\gamma$**: $\gamma = 1.3^2$ is fixed, whereas different datasets or organs might require different contextual ranges.
5. **Only evaluates class-incremental learning**: Does not explore domain-incremental (e.g., from CT to MRI) or task-incremental scenarios.
6. **Scalability to 3D detection**: Current work only processes 2D slices and does not address 3D volumetric detection.

## Related Work & Insights

- **GLIP / Grounding DINO**: Foundation VLOD models that provide vision-language fused object detection capabilities.
- **L2P / DualPrompt / CODA-Prompt**: Representative prompt-based continual learning methods, but designed solely for classification.
- **ZiRa**: First work to adapt pre-trained VLOD models for continual object detection, though with limited performance.
- **MQ-Det**: The design of the multi-modal query encoder inspired the instance feature extraction in the IPG module.
- **Eclipse**: An efficient continual panoptic segmentation method that also uses visual prompt tuning to avoid retraining.
- **Inspired Direction**: The decoupled prompt concept of iDPA can be extended to other dense prediction tasks such as continual segmentation and continual pose estimation.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 8 | The combination of IPG + DPA shows theoretical innovation, with an elegant mathematical derivation for DPA. |
| Utility | 8 | Incremental medical detection is a real clinical need, and tuning 1.4% of the parameters is highly efficient for deployment. |
| Experimental Thoroughness | 9 | 13 cross-modal datasets + full-data/few-shot configurations + ablation studies. |
| Writing Quality | 8 | Clear structure, rigorous mathematical derivations, and rich charts/figures. |
| Overall Score | 8.5 | High-quality work addressing an important yet neglected problem. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unleashing the Power of Prompt-driven Nucleus Instance Segmentation](../../ECCV2024/medical_imaging/unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)
- [\[ICML 2025\] The Four Color Theorem for Cell Instance Segmentation](the_four_color_theorem_for_cell_instance_segmentation.md)
- [\[CVPR 2026\] DK-DDIL: Adaptive Knowledge Retention for Dynamic Domain-Incremental Learning in Medical Imaging](../../CVPR2026/medical_imaging/dk-ddil_adaptive_knowledge_retention_for_dynamic_domain-incremental_learning_in_.md)
- [\[ICLR 2026\] ASMIL: Attention-Stabilized Multiple Instance Learning for Whole-Slide Imaging](../../ICLR2026/medical_imaging/asmil_attention-stabilized_multiple_instance_learning_for_whole-slide_imaging.md)
- [\[ICML 2025\] Do Multiple Instance Learning Models Transfer?](do_multiple_instance_learning_models_transfer.md)

</div>

<!-- RELATED:END -->
