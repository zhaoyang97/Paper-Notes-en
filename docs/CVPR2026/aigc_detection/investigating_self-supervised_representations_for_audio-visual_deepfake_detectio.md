---
title: >-
  [Paper Note] Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection
description: >-
  [CVPR 2026][AIGC Detection][Deepfake Detection] This is a systematic "investigation" paper: the authors freeze 12 off-the-shelf self-supervised encoders (audio, visual, and audio-visual) and train only a single-layer linear probe on top of them. They evaluate their capability in audio-visual deepfake detection horizontally across three dimensions: "detection effectiveness, interpretability, and cross-modal complementarity". They find that "audio-driven" representations genera…
tags:
  - "CVPR 2026"
  - "AIGC Detection"
  - "Deepfake Detection"
  - "Self-Supervised Representations"
  - "Linear Probe"
  - "Anomaly Detection"
  - "Interpretability"
date: 2026-05-08
content_hash: da25a5d70a83e25b
---

# Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Boldisor_Investigating_Self-Supervised_Representations_for_Audio-Visual_Deepfake_Detection_CVPR_2026_paper.html)  
**Code**: https://bit-ml.github.io/ssr-dfd (Project Homepage)  
**Area**: AIGC Detection / Audio-Visual Deepfake Detection / Self-Supervised Representations  
**Keywords**: Deepfake Detection, Self-Supervised Representations, Linear Probe, Anomaly Detection, Interpretability

## TL;DR
This is a systematic "investigation" paper: the authors freeze 12 off-the-shelf self-supervised encoders (audio, visual, and audio-visual) and train only a single-layer linear probe on top of them. They evaluate their capability in audio-visual deepfake detection horizontally across three dimensions: "detection effectiveness, interpretability, and cross-modal complementarity". They find that "audio-driven" representations generalize best (with BRAVEn's visual encoder achieving SOTA), whereas the main difficulty in real-world data stems from the intrinsic complexity of the datasets themselves rather than features exploiting shallow shortcuts.

## Background & Motivation

**Background**: Numerous methods exist for audio-visual deepfake detection, ranging from discriminative classifiers to techniques that exploit cross-modal inconsistencies between audio and video. Recent trends lean towards utilizing powerful self-supervised (SSL) backbones—such as CLIP for image detection, Wav2Vec2 for audio detection, and AV-HuBERT for audio-visual detection. These SSL representations encode rich modal structures without requiring task annotations, making them naturally suited for detection.

**Limitations of Prior Work**: Prior works either use a single SSL feature in isolation or bury it within a complex architecture, leaving two questions difficult to answer: is it the feature itself that is effective, or is the complex upper-level network playing the key role? More critically, existing research warns that even extremely subtle distribution differences between real and fake samples (such as leading silence at the beginning) can be exploited by classifiers as spurious correlations (shortcuts). Moreover, these shortcuts can persist across datasets, artificially inflating results. Standard evaluations that simply train a classifier to look at AUC cannot identify such "cheating".

**Key Challenge**: High AUC $\neq$ capturing true forgery forensic clues. Supervised training rewards any signal that can distinguish between the two classes, including dataset artifacts unrelated to the forgery. Therefore, relying solely on detection scores cannot determine whether an SSL representation is "focusing on the correct areas."

**Goal**: To decompose the problem into three research questions—(RQ1) how useful these SSL features actually are for detection, whether they can generalize out-of-domain (OOD), and whether they can transfer to the related task of anomaly detection; (RQ2) where the model is looking, specifically whether it aligns with the manipulated regions and is consistent with human annotations; (RQ3) whether different features are complementary, i.e., whether multiple successful features rely on the same clues or encode distinct information.

**Key Insight**: Utilizing minimal top-level parameters (a linear probe) to "directly measure how much information is already encoded in the representations," coupled with a multi-dimensional evaluation suite to expose shortcuts. By keeping the upper level completely minimalist, the conclusions can be cleanly attributed to the features themselves.

**Core Idea**: Instead of proposing yet another detector, this work establishes a multi-faceted evaluation protocol consisting of "linear probe + anomaly detection proxy tasks + spatio-temporal interpretability + complementarity analysis" to fairly evaluate a wide range of SSL representations, clarifying which features genuinely capture forgery forensic clues.

## Method

### Overall Architecture
Rather than introducing a new model, this paper presents an evaluation methodology. Deepfake detection is modeled as a binary classification task: mapping an input video $x$ to a label $y$ ($1$ for fake, $0$ for real). All evaluated representations are **frozen**, with only a minimal number of parameters trained on top of them, ensuring that different features compete in a comparable setting. The core detection pipeline consists of three steps: (1) extracting local temporal features using a frozen encoder (one embedding $\omega(x)_t$ per frame); (2) applying a learnable linear classifier $w$; (3) aggregating frame-level predictions into a video-level score using a pooling function.

Built upon this unified pipeline, the authors perform evaluations across three dimensions: leveraging a linear probe + anomaly detection proxy tasks to assess "utility and robustness" (RQ1); utilizing temporal/spatial interpretability to inspect "where the model is looking" (RQ2); and measuring predictive correlation + fusion gains to examine "complementarity" (RQ3). These three analysis lines share the exact same set of frozen features, allowing for horizontal alignment of conclusions. The 12 evaluated encoders span three categories: audio-only (Wav2Vec XLS-R, Auto-AVSR ASR, AV-HuBERT(A), BRAVEn(A)), visual-only (CLIP, FSFM, VideoMAE, Auto-AVSR VSR, AV-HuBERT(V), BRAVEn(V)), and audio-visual (Auto-AVSR, AV-HuBERT).

### Key Designs

**1. Linear probe + log-sum-exp pooling: Cleanly attributing detection capability to the features themselves**

Addressing the limitation where "complex architectures obscure the true contribution of features", the authors simplify the top level to the extreme—learning only a single linear layer. The frame-level features $\omega(x)_t$ pass through the linear classifier and are aggregated into a video-level score using log-sum-exp:

$$s(x; w) = \log \sum_t \exp\left(w^\top \omega(x)_t\right)$$

The log-sum-exp function acts as a differentiable approximation of the max function, implying that if even **a single frame or region in the video is classified as fake**, the entire clip tends to be classified as fake. This aligns perfectly with the detection requirements of local manipulations (e.g., AV-Deepfake1M, where only a few seconds corresponding to transcript edits are modified). The linear layer is trained with cross-entropy loss on video-level labels. Although some representations intrinsically encode global temporal information, this design of "local scoring + max pooling" naturally endows the model with weak-supervision localization capability (see Design 3). The authors also verify that replacing the linear layer with a stronger Transformer classifier yields similar results, showing that **features are more critical than the classifier**, further justifying the attribution to the minimalist top level.

**2. Anomaly detection proxy tasks trained only on real data: Breaking true/fake asymmetry and bypassing shortcuts**

Supervised classifiers tend to exploit any subtle distribution shift between real and fake samples to "cheat". To bypass this, the authors design two proxy tasks **trained exclusively on real data**, based on the assumption that "deviation from the real data distribution indicates forgery". The first is **next-token prediction (NTP)**: a 4-layer, 4-head decoder-only Transformer with a feature dimension of 512 and a feedforward dimension of 1024 is used to predict the next-frame representation $x_t$ given the history $x_1,\dots,x_{t-1}$ using mean squared error (MSE) on real videos. During testing, the maximum frame-level MSE is used as the video forgery score. The second is **audio-visual synchronization**: a 4-layer MLP alignment network $\phi$ with LayerNorm and ReLU scores the concatenated $L_2$-normalized audio feature $a$ and visual feature $v$. The training objective is to ensure the probability of matching the audio frame $a_i$ with its corresponding video frame $v_i$ is higher than with its neighboring frames $v_k$ ($k\in N(i)$):

$$p(v_i \mid a_i) = \frac{\exp\big(\phi(a_i, v_i)\big)}{\sum_{k\in N(i)} \exp\big(\phi(a_i, v_k)\big)}$$

During testing, the inverse of the frame-level alignment score serves as the forgery metric, which is then pooled via log-sum-exp. Because these tasks have never seen fake samples, they cannot learn shortcuts based on differences between real and fake. This is confirmed by experiments: when running these two proxy tasks with randomly initialized features, the synchronization task drops to random performance and NTP falls to a moderate level, exposing the shortcut issue in supervised models that remain artificially high even on random features.

**3. Spatio-temporal interpretability: Verifying if the model focuses on the correct regions**

Addressing the issue that "high AUC does not guarantee focusing on the right clues", the authors extract implicit localization directly from the linear classifier. Since the pooling is log-sum-exp (a simple transformation of the input), the video-level prediction can be viewed as an aggregation of frame-level predictions. Thus, the **temporal explanation** directly uses the frame-level scores $s_t = w^\top \omega(x)_t$ to measure which period contributes the most, which is then compared against the annotated manipulated segments in AV1M (treating each frame as an independent sample to compute localization AUC). For the **spatial explanation**, leveraging the linearity of the frame-level classifier: if the frame-level features are simply averages of patch-level features, the linear classifier propagates directly to the patch level; if non-linear aggregation is used, Grad-CAM is applied instead. The spatial explanations are compared with human click annotations of forgery artifact positions on the ExDDV dataset, using the Mean Absolute Error (MAE) between the peak Grad-CAM coordinates and human click coordinates to quantify human-machine alignment. Since the classifier is trained solely using video-level supervision, this comparison also serves as a weakly supervised localization evaluation.

**4. Complementarity analysis: Determining if different features encode distinct information**

RQ3 asks: "Since multiple features perform well, are they relying on the same or different clues?" The authors quantify this from two angles: (1) calculating the Pearson correlation coefficient between the predicted outputs of each model pair, where a weak correlation suggests that different information is encoded; (2) measuring the downstream performance gain from multi-model ensembles (late fusion, averaging predictions). The findings show that correlations are generally weak-to-moderate, and the **gain increases with higher complementarity**—with some exceptions (e.g., VideoMAE gets a higher gain from the more aligned CLIP than from the more complementary AV-HuBERT(V)), indicating that fusion dynamics are more nuanced than simply looking at complementarity alone.

### Loss & Training
The linear probe is trained using cross-entropy loss on video-level labels; the NTP proxy task is trained on real videos using MSE to train the decoder-only Transformer; the synchronization proxy task is trained on a subset of real data (VoxCeleb) using the contrastive alignment objective (Equation 2). All SSL encoders are frozen throughout, with only the minimal top-level parameters trained.

## Key Experimental Results

The evaluation utilizes four datasets: FakeAVCeleb (FAVC: academic, face-swapping/lip-sync/voice cloning), AV-Deepfake1M (AV1M: million-scale, local manipulations), AVLips (AVL: lip-sync forged), and DeepfakeEval-2024 (DFE-2024: real-world, 52 languages, unknown manipulation types). The evaluation metric is AUC (50% random baseline).

### Main Results

The following table shows a representative excerpt of the average out-of-domain (OOD) AUC across 9 "train set $\rightarrow$ test set" combinations (the last column of Tab. 2):

| Representation | Modality | Average OOD AUC | DFE-2024 Best OOD |
|------|------|------|------|
| BRAVEn (V) | Visual (Audio-driven) | **84.6** | **76.0** |
| AV-HuBERT (V) | Visual (Audio-driven) | 80.0 | 67.7 |
| AV-HuBERT (AV) | Audio-Visual | 78.2 | 54.3 |
| Wav2Vec2 | Audio | 70.8 | 58.6 |
| AV-HuBERT (A) | Audio | 67.3 | 48.3 |
| CLIP ViT-L/14 | Visual | 63.2 | 43.5 |
| AV-HuBERT (A) Random | Audio | 63.5 | 46.4 |
| FSFM | Visual | 55.0 | 43.5 |

Comparison with SOTA methods (consistently trained on 23k samples from AV1M, reporting cross-dataset average AUC):

| Method | Supervised? | All4 Average | Last3 Average (w/o AV1M) |
|------|------|------|------|
| BRAVEn (V) + Linear Probe | Yes | **91.1** | **90.5** |
| AV-HuBERT (AV) + Linear Probe | Yes | 84.5 | 79.4 |
| Wav2Vec2 + Linear Probe | Yes | 78.7 | 71.6 |
| SpeechForensics [44] | No | 84.1 | 89.4 |
| AuViRe [41] | No | 81.1 | 74.7 |
| RealForensics [22] | No | 75.2 | 80.1 |
| AVFF [58] | Yes | 74.8 | 67.1 |
| AVAD [19] | No | 71.5 | 77.7 |

A single frozen BRAVEn(V) encoder paired with a one-layer linear probe outperforms these more complex, specialized approaches. Only SpeechForensics is comparable, but the authors point out that this is because it inherently utilizes AV-HuBERT-like features and models detection as an anomaly detection task.

### Ablation Study (Anomaly Detection Proxy Tasks, AUC % on AV1M)

| Feature | Supervised (Sup) | Next-Token Prediction (NTP) | Sync |
|------|------|------|------|
| AV-HuBERT (A) (Single) | 99.0 | 90.6 | N/A |
| Wav2Vec2 (Single) | 96.6 | 56.6 | N/A |
| AV-HuBERT (V) (Single) | 64.1 | 46.1 | N/A |
| CLIP (Single) | 71.1 | 47.3 | N/A |
| AV-H (A+V) Random | 74.0 | 64.4 | 50.0 |
| AV-H (A+V) | 97.2 | 84.5 | **87.3** |
| AV-H (A) + CLIP | 99.0 | 86.9 | 50.0 |
| W2V2 + AV-H (V) | 96.2 | 60.6 | 86.5 |

Random features can still be artificially inflated under supervised setups (e.g., randomized AV-H(A+V) achieves 74.0% with supervision), but fall sharply in anomaly detection scenes: the synchronization task plummets to a random-chance level of 50.0%, and NTP drops to a modest 64.4%—confirming that these proxy tasks are indeed shortcut-proof. Achieving reasonable anomaly detection scores requires specific feature combinations: NTP relies heavily on AV-HuBERT(A), whereas synchronization thrives on AV-HuBERT(V); the only model approaching supervised levels is the AV-HuBERT(A)+AV-HuBERT(V) synchronization model (87.3%).

### Key Findings
- **Audio-driven representations generalize best**: On AVLips/DFE-2024 where audio manipulations are absent, "audio-trained but visual-extracting" AV-HuBERT(V) and BRAVEn(V) showcase the strongest performance; BRAVEn(V) achieves SOTA. Conversely, audio-only features perform well only when the dataset contains speech-level manipulations.
- **Random features are not entirely random**: Randomly initialized models score an AUC significantly above 50%, showing that structural inductive biases can implicitly encode discriminative information—yet RQ2 analyses demonstrate that these are mostly shortcuts (such as leading silence), which also explains why random audio models perform better than random visual ones.
- **Temporal explanations consistently align with manipulated regions**: For most features, the localization AUC is close to their classification AUC; only the random model and FSFM drop significantly. While audio models do gaze at leading silence, they simultaneously focus on the manipulated regions. Wav2Vec2 tends to detect transition boundaries, AV-HuBERT(V) yields the cleanest temporal predictions, and CLIP is the noisiest.
- **Spatial explanations partially align with human attention**: For CLIP-based models (71.3% AUC on ExDDV), Grad-CAM highlights tend to fall on the forehead, whereas human clicks land near eyes and mouths. Model MAE outperforms the "frame center" baseline but falls short of dedicated click predictors—though even those click predictors are only marginally better than the "face center" baseline, implying that human annotations themselves might not convey much localization info beyond indicating that "artifacts exist somewhere on the face." Crucially, models do not rely on background spurious features.
- **Visual models exhibit higher complementarity than audio models**: Cross-model prediction correlations are generally weak-to-moderate; audio models (AV-HuBERT(A) and Wav2Vec2) demonstrate the highest mutual correlation. AV-HuBERT(V), focusing solely on lips and co-trained with audio, correlates more closely with audio models.
- **Real-world data is truly more challenging**: On DFE-2024, even the best in-domain setup reaches only 75.5% AUC, and the best OOD tops out at 76.0%. This gap does not occur due to features latching onto shallow patterns (as spatio-temporal explanations genuinely align with semantic artifacts) but rather originates from the intrinsic difficulty and diversity of the dataset (e.g., missing modalities and domain shifts).

## Highlights & Insights
- **Clean evidence of "Feature > Classifier"**: Replacing the linear layer with a Transformer yields comparable results, demonstrating that backbones dictate performance in deepfake tasks, providing a direct answer to the engineering trade-off: "should we invest in pre-training or detection heads?"
- **Anomaly detection as a "shortcut detector"**: Trained strictly on real data, these tasks are inherently incapable of learning artificial shortcuts between real and fake, thus acting as a "magic mirror" that exposes inflated performance from random features in supervised settings. This approach can be generalized to any classification evaluation suspected of containing distribution shortcuts.
- **Directly deriving localization from the linear probe**: Because the classifier is linear and pooling leverages log-sum-exp, frame-level and patch-level contributions can be resolved with zero extra training. This essentially yields a free weakly-supervised localization evaluation, offering a highly practical interpretability trick.
- **"Audio-driven visual features" counter-intuitively perform best**: AV-HuBERT(V)/BRAVEn(V) extract visual (lip-reading) features, yet their training is shaped by audio signals. Strikingly, they generalize best on datasets without audio forgery—suggesting that joint self-supervised learning of lip and speech captures a universal forgery clue: cross-modal consistency.

## Limitations & Future Work
- **Real-world generalization remains a hard nut to crack**: Neither the evaluated representations nor existing methods generalize well to DFE-2024. The authors attribute this to the difficulty and diversity of real-world data (missing modalities, video domain shifts) and highlight the need for specialized schemes that explicitly model this diversity.
- **Evaluation focuses primarily on linear probes**: Although the authors verified similar results with a Transformer head, a minimalist top level might understate the potential of certain representations that require non-linear combinations to unleash their benefits.
- **Limited "ground truth" in spatial explanations**: Human click annotations were found to convey little additional localization information beyond the simple "face center" baseline, suggesting that conclusions drawn from spatial alignment require cautious interpretation (⚠️ this is an analytic finding, please refer to the original paper's Fig. 4 for exact numerical details).
- **Lack of coverage on more aggressive manipulation/generation methods**: The generation methods of the evaluated datasets are relatively concentrated; the timeliness of these conclusions needs to be monitored against evolving next-generation generative models.

## Related Work & Insights
- **vs detectors using a single SSL feature (e.g., CLIP detection [57], Wav2Vec audio detection, AV-HuBERT audio-visual detection)**: These methods individually wager on a single feature type, often buried beneath complex architectures; in contrast, this work evaluates a broad spectrum of features under a unified minimalist setup, isolating the "contribution of the features themselves" from "architectural contributions."
- **vs SpeechForensics [44]**: It utilizes AV-HuBERT features and frames detection as anomaly detection, representing the only SOTA method comparable to our linear probe. This paper notes that their strength derives from a shared lineage—both benefit from the cross-modal consistency in AV-HuBERT-like features.
- **vs works focusing on "shortcuts/spurious correlations" [10,36,52,73]**: While prior studies warn that distribution shifts can be exploited, this work goes a step further by actively exposing shortcuts using real-data-only anomaly detection proxy tasks and demonstrating that random features do not align with ground-truth manipulations.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new model, but the evaluation methodology comprising a "unified minimalist setup + anomaly detection as a shortcut detector + spatio-temporal interpretability alignment" is a solid and inspiring contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, spanning 12 encoders $\times$ 4 datasets $\times$ 3 dimensions (detection/interpretability/complementarity), while also comparing against 5 SOTA methods.
- Writing Quality: ⭐⭐⭐⭐ The three research questions are clearly organized and findings are explicitly presented; table column encoding (A–J) is slightly difficult to read.
- Value: ⭐⭐⭐⭐ Provides actionable conclusions regarding which SSL features to use and why real-world generalizability remains challenging; the strong BRAVEn(V) + linear probe baseline is highly practical for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection](enabling_supervised_learning_of_generative_signatures_for_generalized_ai-generat.md)
- [\[ICLR 2026\] A Rich Knowledge Space for Scalable Deepfake Detection](../../ICLR2026/aigc_detection/a_rich_knowledge_space_for_scalable_deepfake_detection.md)
- [\[CVPR 2026\] Learning Forgery-Aware Lip Representations Without Forgery Priors](learning_forgery-aware_lip_representations_without_forgery_priors.md)
- [\[CVPR 2026\] Inconsistency-aware Multimodal Schrodinger Bridge for Deepfake Localization](inconsistency-aware_multimodal_schrodinger_bridge_for_deepfake_localization.md)
- [\[ICLR 2026\] Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images](../../ICLR2026/aigc_detection/semantic_visual_anomaly_detection_and_reasoning_in_ai-generated_images.md)

</div>

<!-- RELATED:END -->
