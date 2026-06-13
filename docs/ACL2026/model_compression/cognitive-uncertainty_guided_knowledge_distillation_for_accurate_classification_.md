---
title: >-
  [Paper Note] Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions
description: >-
  [ACL 2026][Model Compression][Student Misconception Classification] This paper proposes a two-stage knowledge distillation framework using "Two-Tier Margin-Based Sample Selection" guided by teacher cognitive uncertainty…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Student Misconception Classification"
  - "Two-Stage Distillation"
  - "Uncertainty-Based Sample Selection"
  - "Difficulty-Adaptive Loss"
  - "Edge Deployment"
date: 2026-05-08
content_hash: 5a27fc339db63f34
---

# Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions

**Conference**: ACL 2026  
**arXiv**: [2605.14752](https://arxiv.org/abs/2605.14752)  
**Code**: https://github.com/RoschildRui/acl2026_map (Available)  
**Area**: Model Compression / Knowledge Distillation / AI in Education  
**Keywords**: Student Misconception Classification, Two-Stage Distillation, Uncertainty-Based Sample Selection, Difficulty-Adaptive Loss, Edge Deployment

## TL;DR
This paper proposes a two-stage knowledge distillation framework using "Two-Tier Margin-Based Sample Selection" guided by teacher cognitive uncertainty and a difficulty-adaptive loss. By using only 10.30% of real samples for incremental training, the 4B student model achieves MAP@3 = 0.9585 (+17.8%) on MAP-Charting. On a 220-question middle-school algebra misconception benchmark, it achieves 84.38% accuracy, surpassing GPT-5 (67.73%) and a directly fine-tuned 72B teacher (81.25%), while maintaining an inference speed 23× faster than the teacher.

## Background & Motivation

**Background**: Pedagogical assessment is shifting from simple "correct/incorrect" scoring to understanding students' reasoning processes. In the NLP domain, recent attempts have used LLMs for fine-grained misconception classification (e.g., MisstepMath, MathEDU, Otero 2025). A common approach involves synthesizing large-scale training data with LLMs followed by direct fine-tuning.

**Limitations of Prior Work**: The authors identify three core challenges: (1) **Data Scarcity and Long-tail Distribution**: Real student reasoning is difficult to synthesize; LLM-generated text is often too formal and fluent, creating a distribution gap with real student data, which is colloquial, contains skipped steps, and exhibits logical fractures. (2) **Label Noise and Blurred Boundaries**: Misconception categories are numerous and their boundaries are ambiguous, leading to high annotation noise; traditional hard-label models fail to capture fine-grained differences. (3) **Deployment Paradox**: Large models possess knowledge but may ignore non-standard yet reasonable student logic due to pre-training bias, and face privacy and resource constraints for edge deployment. Small models are deployable but prone to overfitting noisy labels.

**Key Challenge**: The traditional path is "generating more data," but synthetic data deviates significantly from real distributions. Using small models for deployment risks overfitting noise, while using large models as teachers risk distilling their inherent biases to students.

**Goal**: (1) Instead of relying on large-scale synthetic data, mine "high-value samples" from existing real data for incremental training; (2) Enable small student models to inherit inter-class relationships from soft labels while distinguishing ambiguous misconceptions; (3) Use two-stage distillation to resist the "diversity blind spots" of large models, allowing the 4B student to recognize non-standard responses.

**Key Insight**: Samples are categorized based on teacher cognitive uncertainty into groups such as "teacher highly confident + correct," "seemingly correct but low confidence," and "severely incorrect." Secondary distillation is performed only on the most informative "near-miss" and "hard-hard" samples, with weights for CE, KD, and COS losses dynamically assigned based on sample type.

**Core Idea**: Stage 1 performs global distillation to transfer basic capabilities. Stage 2 uses "Two-Tier Margin Selection" to isolate ~10% high-value samples for targeted training with a difficulty-adaptive loss (switching $\alpha/\beta/\gamma$ based on sample type), ensuring the small model learns both answer prediction and boundary distinction.

## Method

### Overall Architecture
The framework consists of two-stage distillation as shown in Figure 2:

1.  **Stage 1 (Global Distillation)**: Employs stratified $n=5$-fold cross-validation to train $n$ teachers $f_t^{(k)}$. Each fold is trained on $\mathcal{D}\setminus\mathcal{D}_k$ and generates soft labels for $\mathcal{D}_k$ to prevent teacher overfitting on seen data. $n$ student models are then trained using $\mathcal{L}=\alpha\mathcal{L}_{\text{CE}}+\beta\mathcal{L}_{\text{KD}}+\gamma\mathcal{L}_{\text{COS}}$ (default weights $(0.33,0.33,0.34)$ via grid search), establishing baseline classification capabilities.
2.  **Stage 2 (High-Value Sample Selection + Adaptive Refinement)**: Based on Stage 1 teacher predictions, samples are categorized into Near-miss ($\mathcal{S}_{\text{NM}}$) and Hard-hard ($\mathcal{S}_{\text{HH}}$). A composite difficulty metric $\mathcal{M}(x_i,y_i)=d(x_i,y_i)\cdot e^{-H(x_i)}$ is used to bisect each category by the median into "close" and "far" subsets. Student models are refined on these four subsets using specific $(\alpha,\beta,\gamma)$ combinations.
3.  The final training data $\mathcal{S}_{\text{NM}} \cup \mathcal{S}_{\text{HH}}$ accounts for approximately 10.30% of the total samples.

### Key Designs

1.  **Two-Tier Margin-Based Sample Selection**:
    *   **Function**: Uses teacher cognitive uncertainty as a signal to select the most informative "near-boundary correct" and "severely deviant" samples from the full dataset for Stage 2 incremental training.
    *   **Mechanism**: (i) The first tier partitions samples based on teacher prediction ranks: $\mathcal{S}_{\text{NM}}=\{(x_i,y_i):[(\hat{y}_i=y_i)\land(p^{(1)}-p^{(2)})\leq\delta)]\lor \text{rank}(y_i)\in\{2,3\}\}$ (correct but low confidence, or ground truth ranked 2nd/3rd), and $\mathcal{S}_{\text{HH}}=\{(x_i,y_i):\text{rank}(y_i)>3\}$ (severely incorrect). (ii) The second tier uses the difficulty metric $\mathcal{M}(x_i,y_i)=d(x_i,y_i)\cdot e^{-H(x_i)}$ (where $d$ is probability margin and $H$ is prediction entropy) to split $\mathcal{S}_{\text{NM}}$ and $\mathcal{S}_{\text{HH}}$ into "close" and "far" subsets. Parameters $\delta=0.05$ and $K=5$ were determined via grid search (Figure 4).
    *   **Design Motivation**: Samples already mastered by the model contribute nothing to the decision boundary; training should focus on "boundary ambiguity" and "complete errors." Near-miss samples define fine-grained boundaries, while Hard-hard samples expose knowledge gaps. The $e^{-H(x_i)}$ term in the difficulty metric is a strategic multiplier—it amplifies difficulty when entropy is low (model is confidently wrong) and attenuates it when entropy is high (model is uncertain, suggesting inherent ambiguity). This "two-tier slicing" ensures that ~10% of the data carries the maximum learning signal.

2.  **Difficulty-Adaptive Loss**:
    *   **Function**: Dynamically adjusts weights for hard label (CE), soft label (KD), and representation alignment (COS) losses based on sample type to avoid a one-size-fits-all approach.
    *   **Mechanism**: Total loss is $\mathcal{L}_{\text{total}}=\alpha\mathcal{L}_{\text{CE}}+\beta\mathcal{L}_{\text{KD}}+\gamma\mathcal{L}_{\text{COS}}$, with $(\alpha,\beta,\gamma)$ adjusted: $\mathcal{S}_{\text{NM}}^{\text{close}} \rightarrow (1,0,0)$ (near-boundary correct, using only hard labels to avoid blurred boundaries caused by soft label smoothing); $\mathcal{S}_{\text{NM}}^{\text{far}} \rightarrow (1,1,1)$ (balancing precision and inter-class relationships); $\mathcal{S}_{\text{HH}}^{\text{close}} \rightarrow (0,1,1)$ (close to truth but deviant, trusting teacher labels to resist noise); $\mathcal{S}_{\text{HH}}^{\text{far}} \rightarrow (1,1,1)$ (extremely difficult, requiring both hard and soft supervision).
    *   **Design Motivation**: Traditional KD uses fixed weights, failing to distinguish between boundary-ambiguous and highly noisy samples. The key insight is that for NM-close samples, the "smoothness" of soft labels is counter-productive as it blurs tight boundaries. Thus, setting $\beta=\gamma=0$ enforces strong hard-label constraints. Letting different signals "speak" for different types of samples is the most refined aspect of the design.

3.  **N-fold Stage-1 Teacher Generation + Path to Surpassing the Teacher**:
    *   **Function**: Enables a 4B student to match or even exceed the 72B teacher, merging the dual goals of compression and performance improvement.
    *   **Mechanism**: (i) Stratified $K$-fold ensures teachers generate soft labels only for unseen samples, preventing confidence inflation. (ii) Stage 2 increases ground-truth supervision weight for NM-far/HH-far, essentially allowing the student to favor the truth over the teacher when the teacher is "confidently wrong." (iii) Incremental training on only 10.30% of real samples prevents the small model from being degraded by noisy samples.
    *   **Design Motivation**: Surpassing the teacher is attributed to three factors: correcting teacher pre-training bias (ignoring non-standard reasoning), task specialization (focusing on uncertainty regions), and adaptive error correction. This provides an empirical case in KD literature where the teacher is not necessarily the upper bound.

### Loss & Training
*   **Stage 1**: $\mathcal{L}_{\text{CE}}=-\log p_s(y_i|x_i)$ + $\mathcal{L}_{\text{KD}}=\tau^2\cdot\text{KL}(p_t\|p_s)$ ($\tau=1.0$) + $\mathcal{L}_{\text{COS}}=1-\cos(p_s,p_t)$ with $(\alpha,\beta,\gamma)=(0.33,0.33,0.34)$. AdamW with lr=$2\times10^{-4}$ (student), $1\times10^{-4}$ (teacher), batch=16, grad acc=4.
*   **Stage 2**: Incremental lr=$1\times10^{-6}$, max_grad_norm=4, with $(\alpha,\beta,\gamma)$ switched per the four sample types (see Appendix A).
*   **Backbones**: Student = Qwen-3-4B / Gemma-2-9B / Llama-3.1-8B; Teacher = Qwen-2.5-72B.

## Key Experimental Results

### Main Results (MAP-Charting + Algebra Misconception)

| Method | MAP-Charting MAP@3 | MAP@10 | Acc | F1@3 | Algebra Misc. MAP@3 | Acc |
|---|---|---|---|---|---|---|
| **Prompting baselines** | | | | | | |
| GPT-5 | 0.8137 | 0.8145 | 0.7225 | 0.4626 | 0.7409 | 0.6773 |
| Claude-4-Sonnet | 0.7833 | 0.7841 | 0.6914 | 0.4579 | 0.6636 | 0.5636 |
| Qwen-2.5-72B (prompt) | 0.7285 | 0.7293 | 0.6222 | 0.4328 | 0.6280 | 0.5320 |
| **Fine-tuned** | | | | | | |
| Qwen-2.5-72B (FT) | 0.9497 | 0.9501 | 0.9014 | 0.4993 | 0.8438 | 0.8125 |
| Qwen-3-4B (FT) | 0.9472 | 0.9475 | 0.8987 | 0.4992 | 0.7552 | 0.7188 |
| **Ours (Two-Stage KD)** | | | | | | |
| **Qwen-3-4B + Ours** | **0.9585** | **0.9587** | **0.9198** | **0.4996** | **0.8750** | **0.8438** |
| Gemma-2-9B + Ours | 0.9560 | 0.9562 | 0.9148 | 0.4995 | 0.8015 | 0.7656 |
| Llama-3.1-8B + Ours | 0.9553 | 0.9555 | 0.9134 | 0.4995 | 0.7865 | 0.7564 |

Qwen-3-4B + Ours outperforms GPT-5 by 17.8% in MAP@3 on MAP-Charting and exceeds the FT 72B teacher by 1.0% MAP@3 and 1.8% Accuracy. The 4B student surpasses the 72B teacher by approximately 0.9-2 percentage points.

### Ablation Study (Qwen-3-4B, MAP-Charting + Algebra)

| Configuration | MAP-Charting MAP@3 / Acc | Algebra MAP@3 / Acc | Note |
|---|---|---|---|
| **Full Method** | **0.9585 / 0.9198** | **0.8750 / 0.8438** | Complete method |
| w/o Adaptive Loss | 0.9540 / 0.9123 | 0.8657 / 0.8321 | Uniform loss weights |
| w/o Sample Selection | 0.9519 / 0.9085 | 0.8603 / 0.8269 | Using full dataset |
| w/o Stage-1 Distillation | 0.9546 / 0.9132 | 0.8679 / 0.8342 | Stage 2 only |
| w/o Stage-2 Distillation | 0.9493 / 0.9024 | 0.7893 / 0.7577 | **Stage 1 only (largest drop)** |

Removing Stage 2 resulted in an 8.6 percentage point drop in accuracy on the Algebra dataset (0.8438 → 0.7577), proving that high-value sample selection and adaptive loss are the core drivers.

### Efficiency (7339 Sample Inference)

| Model | MAP@3 | Time (h) | Hardware |
|---|---|---|---|
| GPT-5 (API) | 0.8137 | 1.50 | Cloud |
| GPT-OSS-120B | 0.7661 | 1.10 | 32× H20 |
| Qwen-2.5-72B teacher (FT) | 0.9497 | 0.186 | 8× H20 |
| **Qwen-3-4B student** | **0.9599** | **0.008** | 8× H20 |

The student model is 187.5× faster than GPT-5, 23.25× faster than the teacher, and 137.5× faster than GPT-OSS-120B, while leading in MAP@3.

### Key Findings
*   **Stage 2 is critical**: Its removal causes an 8.6 drop in Algebra Accuracy, significantly higher than removing Adaptive Loss (1.2), Sample Selection (1.7), or Stage 1 (1.0).
*   **4B Surpassing 72B**: Qwen-3-4B + Ours achieves higher accuracy than the directly fine-tuned 72B teacher with 23× faster efficiency, proving that rational sample selection and loss allocation can break teacher limits.
*   **Hyperparameter Robustness**: $(\alpha,\beta,\gamma)=(0.33,0.33,0.34)$ was consistent as the optimal grid search result across three student backbones.
*   **Optimal $\delta=0.05, K=5$**: Lower $\delta$ provides too few samples; higher $\delta$ introduces noise. $K=5$ balances coverage and stability.
*   **Minimal Data Usage**: Stage 2 uses only 10.30% of samples, avoiding lengthy incremental training and making it friendly for small-data scenarios.

## Highlights & Insights
*   **"High-Value Sample" Philosophy**: The authors demonstrate that in education scenarios with label noise and data scarcity, "selecting the right samples" is more valuable than "generating more data." The two-tier slicing of Near-miss and Hard-hard is a clear implementation of active learning concepts.
*   **Counter-intuitive Handling of NM-close**: Setting KD weights to 0 for "near-boundary correct" samples reveals that soft-label smoothness is an enemy in tight decision boundaries. This provides a warning to KD researchers not to use soft labels indiscriminately.
*   **Dual Capture with $d\cdot e^{-H}$**: This metric captures both distance from truth and model confidence. It effectively identifies "confidently wrong" samples and is a simple, versatile tool for other KD or active learning tasks.
*   **Engineering Solution for Edu-AI Deployment**: The combination of a 4B model, real data, edge-ready performance, 23× speedup, and teacher-surpassing results provides a production-ready recipe for privacy-sensitive school environments.

## Limitations & Future Work
*   The authors acknowledge that: (1) $K$-fold cross-partitioning is computationally expensive; (2) the method has limited effectiveness on inherently "poor quality" data, requiring future combination with data synthesis or repair.
*   Other limitations: The 220-sample scale of the Algebra Misconception dataset makes the statistical significance of surpassing the teacher less robust compared to the 36k-sample MAP-Charting dataset. The method was only tested in middle-school math; cross-disciplinary generalization (e.g., humanities, science) is unverified.
*   Future Work: Automating the learning of the four sample loss weights; combining high-value selection with small-scale data synthesis; using $d\cdot e^{-H}$ as an acquisition function for active learning.

## Related Work & Insights
*   **vs. Self-rewarding / RLAIF (Yuan 2024)**: Self-rewarding expands data via LLM self-evaluation but may amplify bias; this work takes an opposite route by mining "existing real samples" to avoid the synthetic distribution gap.
*   **vs. Curriculum Learning (Bengio 2009) / BatchBALD (Kirsch 2019)**: This work merges curriculum learning (easy to hard) and active learning (high information) by using teacher uncertainty as an information signal and adaptive loss as the curriculum mechanism.
*   **vs. Spot-Adaptive KD (Song 2022)**: While Spot-Adaptive KD selects which layers to distill, this work dynamically selects the loss type ($\text{CE/KD/COS}$) weight per sample, explicitly modeling Near-miss/Hard-hard categories.

## Rating
*   Novelty: ⭐⭐⭐⭐ The combination of two-tier margin selection and difficulty-adaptive loss (especially the $\beta=\gamma=0$ design for NM-close) is insightful.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Multiple student backbones, two datasets, complete ablation, grid search, and 5-fold cross-validation analysis.
*   Writing Quality: ⭐⭐⭐⭐ Clear mapping between challenges, designs, and contributions. Extensively documented appendix for reproducibility.
*   Value: ⭐⭐⭐⭐ Provides a robust recipe for low-budget, privacy-sensitive Edu-AI deployment. The methodology is transferable to any long-tail classification scenario.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/model_compression/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[AAAI 2026\] Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy](../../AAAI2026/model_compression/pairing-free_group-level_knowledge_distillation_for_robust_gastrointestinal_lesi.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)
- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)

</div>

<!-- RELATED:END -->
