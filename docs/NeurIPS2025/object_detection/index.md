---
title: >-
  NeurIPS2025 Object Detection Papers · 27 Notes
description: >-
  27 NeurIPS2025 papers in the Object Detection area, covering Anomaly Detection, Object Detection, Adversarial Robustness, Time-Series Forecasting and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "NeurIPS2025"
  - "Object Detection"
  - "AI paper notes"
  - "paper summaries"
  - "Anomaly Detection"
  - "Adversarial Robustness"
  - "Time-Series Forecasting"
item_list:
  - u: "adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre/"
    t: "ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining"
  - u: "an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data/"
    t: "EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination"
  - u: "ascent_fails_to_forget/"
    t: "Ascent Fails to Forget"
  - u: "automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent/"
    t: "Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent"
  - u: "autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp/"
    t: "AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing"
  - u: "burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes/"
    t: "BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes"
  - u: "cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob/"
    t: "CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection"
  - u: "dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano/"
    t: "DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection"
  - u: "detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f/"
    t: "DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding"
  - u: "detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r/"
    t: "DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning"
  - u: "dithub_a_modular_framework_for_incremental_openvocabulary_ob/"
    t: "DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection"
  - u: "flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f/"
    t: "FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies"
  - u: "lr_yolo_lipschitz_continuity_image_restoration_object_detection/"
    t: "Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy"
  - u: "mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling/"
    t: "MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling"
  - u: "multimodal_generative_flows_for_lhc_jets/"
    t: "Multimodal Generative Flows for LHC Jets"
  - u: "normal-abnormal_guided_generalist_anomaly_detection/"
    t: "Normal-Abnormal Guided Generalist Anomaly Detection"
  - u: "recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct/"
    t: "ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction"
  - u: "recon_region-controllable_data_augmentation_with_rectification_and_alignment_for/"
    t: "ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection"
  - u: "rethinking_evaluation_of_infrared_small_target_detection/"
    t: "Rethinking Evaluation of Infrared Small Target Detection"
  - u: "scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma/"
    t: "Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching"
  - u: "scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete/"
    t: "ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection"
  - u: "semi-supervised_graph_anomaly_detection_via_robust_homophily_learning/"
    t: "Semi-supervised Graph Anomaly Detection via Robust Homophily Learning"
  - u: "spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection/"
    t: "Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection"
  - u: "stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif/"
    t: "Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification"
  - u: "structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly/"
    t: "Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection"
  - u: "test-time_adaptive_object_detection_with_foundation_model/"
    t: "Test-Time Adaptive Object Detection with Foundation Model"
  - u: "video-rag_visually-aligned_retrieval-augmented_long_video_comprehension/"
    t: "Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension"
item_total: 27
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🎯 Object Detection

**🧠 NeurIPS2025** · **27** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (97)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

🔥 **Top topics:** Anomaly Detection ×9 · Object Detection ×6 · Adversarial Robustness ×3 · Time-Series Forecasting ×2

**[ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)**

:   This work presents ADPretrain, the first dedicated representation pretraining framework for industrial anomaly detection. By learning residual feature representations via angle-oriented and norm-oriented contrastive losses on the large-scale RealIAD dataset, the pretrained features consistently improve five mainstream embedding-based AD methods across five datasets and five backbone networks when substituted for the original features.

**[EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)**

:   EPHAD proposes a test-time post-processing framework that corrects the output of anomaly detection models trained on contaminated data via Bayesian-style fusion with external evidence (e.g., CLIP, LOF) through exponential tilting. The framework requires no access to the training pipeline and consistently improves detection performance of contaminated models across 8 visual and 26 tabular AD datasets.

**[Ascent Fails to Forget](ascent_fails_to_forget.md)**

:   Starting from the statistical dependence between the forget set and the retain set, this paper theoretically and empirically demonstrates that the widely adopted gradient ascent / Descent-Ascent (DA) family of machine unlearning methods fails systematically in the presence of data correlations. In logistic regression, the DA solution is provably farther from the oracle than the original model, and in non-convex settings DA traps the model in inferior local minima.

**[Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)**

:   This paper proposes a self-reflective agent framework that automatically detects attribute reliance in visual models through an iterative hypothesis generation–testing–verification–reflection loop (e.g., CLIP recognizing "teacher" via classroom backgrounds, YOLOv8 detecting pedestrians via crosswalks). Evaluated on a benchmark of 130 models with injected known attribute dependencies, self-reflection is shown to significantly improve detection accuracy.

**[AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)**

:   This work proposes the AutoSciDACT pipeline, which first employs supervised contrastive learning to compress high-dimensional scientific data into a 4-dimensional embedding space, then applies NPLM (New Physics Learning Machine) likelihood-ratio testing to statistically quantify distributional deviations in the embedding space. The pipeline achieves $\geq 3\sigma$ discovery at signal injection ratios of $\leq 1\%$ across astronomical, particle physics, pathology, image, and synthetic datasets.

**[BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)**

:   This paper introduces BurstDeflicker, the first large-scale benchmark dataset for multi-frame flicker removal (MFFR), comprising three complementary subsets — Retinex-based synthetic data, real-world static data, and green-screen dynamic data — systematically addressing the core bottleneck of obtaining aligned flickering–clean image pairs in dynamic scenes.

**[CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)**

:   To address positive gradient dilution and hard-negative gradient dilution in large-vocabulary (>10K category) object detection, this paper proposes CQ-DINO: replacing the classification head with learnable category queries and using image-guided Top-K category selection to reduce the negative space by 100×. CQ-DINO surpasses the previous SOTA by 2.1% AP on V3Det (13,204 categories) while remaining competitive on COCO.

**[DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)**

:   This work constructs DCAD-2000, a multilingual dataset covering 2,282 languages and 46.72 TB of text, and proposes a language-agnostic data cleaning framework that reformulates cleaning as anomaly detection. The framework extracts 8-dimensional statistical features per document and applies Isolation Forest for dynamic noise filtering. Effectiveness is validated on multiple multilingual benchmarks, with particularly notable gains on low-resource languages.

**[DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)**

:   DetectiumFire constructs the largest multi-modal fire understanding dataset — 14.5K real images + 2.5K videos + 8K synthetic images + 12K RLHF preference pairs — with a low duplication rate (0.03 PHash vs. D-Fire 0.15), a 4-level severity classification scheme, and detailed scene descriptions. Fine-tuning YOLOv11m achieves mAP 43.74, and fine-tuning LLaMA-3.2-11B yields 83.84% accuracy on fire severity classification.

**[DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)**

:   This paper proposes DETree, a framework that constructs a Hierarchical Affinity Tree (HAT) to model the hierarchical relationships among diverse human-AI collaborative text generation processes, and designs a Tree-Structured Contrastive Loss (TSCL) to align the representation space. DETree achieves significant advantages in mixed-text detection and OOD generalization scenarios.

**[DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)**

:   DitHub reformulates the incremental adaptation problem in open-vocabulary object detection as a "version control" problem — training independent LoRA expert modules per category and managing an ever-growing module library via three primitives: branch, fetch, and merge. On ODinW-13 with full data, the method achieves 62.19 mAP, surpassing ZiRa by 4.21 points, while maintaining 47.01 zero-shot COCO performance.

**[FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)**

:   This paper proposes FlexEvent, a framework that achieves flexible object detection with event cameras across varying operational frequencies through an adaptive event-frame fusion module (FlexFuse) and a frequency-adaptive fine-tuning mechanism (FlexTune). The framework maintains robust performance in the range of 20Hz to 180Hz, significantly outperforming existing methods.

**[Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy](lr_yolo_lipschitz_continuity_image_restoration_object_detection.md)**

:   This paper analyzes the root cause of instability in cascaded image restoration and object detection frameworks from a Lipschitz continuity perspective. It identifies an order-of-magnitude smoothness gap between the two networks and proposes LR-YOLO, which integrates the restoration task into the detection backbone's feature learning to regularize the detector's Lipschitz constant, consistently improving detection stability on dehazing and low-light enhancement benchmarks.

**[MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)**

:   This paper presents MSTAR, the first multi-query scene text retrieval method that requires no bounding box annotations. Through Progressive Vision Embedding (PVE), MSTAR progressively shifts attention from salient to non-salient regions. Combined with style-aware instructions and a Multi-Instance Matching (MIM) module, it achieves unified retrieval across four query types—word, phrase, combined, and semantic—and introduces MQTR, the first multi-query text retrieval benchmark.

**[Multimodal Generative Flows for LHC Jets](multimodal_generative_flows_for_lhc_jets.md)**

:   This paper proposes a Transformer-based multimodal flow matching framework (MMF) that jointly models continuous flow matching and continuous-time Markov jump bridges, enabling unified generation of particle kinematics (continuous) and flavor quantum numbers (discrete) in LHC jets.

**[Normal-Abnormal Guided Generalist Anomaly Detection](normal-abnormal_guided_generalist_anomaly_detection.md)**

:   NAGL is the first framework to incorporate mixed normal-and-abnormal reference samples into Generalist Anomaly Detection (GAD). Through two attention modules—Residual Mining (RM) and Anomaly Feature Learning (AFL)—it learns transferable anomaly patterns in residual space, substantially outperforming normal-reference-only methods in cross-domain scenarios with as few as 1 anomaly reference sample.

**[ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction](recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)**

:   This paper proposes ReCon-GS, which achieves incremental 3D reconstruction via continuum-preserved Gaussian streaming, substantially reducing storage requirements and training time while maintaining rendering quality, and supporting real-time reconstruction of large-scale scenes.

**[ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)**

:   ReCon proposes a training-free, region-controllable data augmentation framework that enhances the detection data quality of existing structure-controllable generative models through Region-Guided Rectification (RGR) and Region-Aligned Cross-Attention (RACA), achieving 35.5 mAP on COCO—surpassing GeoDiffusion, which requires fine-tuning.

**[Rethinking Evaluation of Infrared Small Target Detection](rethinking_evaluation_of_infrared_small_target_detection.md)**

:   This paper systematically identifies three critical limitations in existing evaluation protocols for infrared small target detection (IRSTD), and proposes a hierarchical analysis framework comprising the hybrid-level metric hIoU, a systematic error analysis methodology, and a cross-dataset evaluation setting.

**[Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)**

:   This paper proposes TCCM (Time-Conditioned Contraction Matching), a flow matching-inspired semi-supervised anomaly detection method for tabular data. By learning a time-conditioned velocity field that contracts normal data toward the origin, TCCM computes anomaly scores in a single forward pass, achieving top AUROC and AUPRC rankings across 47 ADBench datasets while running 1573× faster than DTE.

**[ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   This paper proposes *scattering* as a novel inductive bias for anomaly detection — anomalous samples are more dispersed than normal samples in the high-dimensional representation space. A dual-encoder architecture (temporal + topological) combined with hyperspherical scattering center constraints and contrastive fusion is used to learn joint temporal-topological representations, achieving best performance in 15/24 settings across 6 industrial IoT datasets.

**[Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)**

:   This paper proposes RHO (Robust Homophily Learning), which addresses the homophily diversity of normal nodes in semi-supervised graph anomaly detection via an adaptive frequency response filter (AdaFreq) and a Graph Normality Alignment (GNA) module, outperforming existing methods on 8 real-world datasets.

**[Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)**

:   This paper proposes the first graph anomaly detection benchmark for non-grid spatio-temporal systems in the maritime domain. It extends the OMTAD dataset to support node/edge/graph-level anomaly detection, and plans to employ LLM agents for trajectory synthesis and anomaly injection.

**[Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)**

:   This paper proposes DPSBA, a clean-label backdoor attack framework for graph classification that generates in-distribution trigger subgraphs via adversarial training while suppressing both structural and semantic anomalies, achieving high attack success rates with significantly improved stealthiness.

**[Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)**

:   This paper proposes OracleAD, a framework that learns causal embeddings for each variable (via LSTM encoding and attention pooling) and constructs a Stable Latent Structure (SLS) to model inter-variable relationships under normal conditions. A dual scoring mechanism combining prediction error and SLS deviation enables interpretable multivariate time series anomaly detection and root cause localization.

**[Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)**

:   This paper proposes TTAOD, a source-free open-vocabulary test-time adaptive object detection framework that combines multimodal Prompt Tuning, Mean-Teacher, an Instance Dynamic Memory (IDM) module, and memory augmentation/hallucination strategies. It achieves 56.2% AP50 on Pascal-C (+11.0 vs. SOTA) and demonstrates consistent gains across 13 cross-domain datasets.

**[Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)**

:   This paper proposes Video-RAG, a training-free, plug-and-play RAG pipeline that extracts visually-aligned auxiliary texts (OCR, ASR, object detection) from video, retrieves relevant content, and feeds it into LVLMs. With an overhead of only ~2K tokens, it improves average Video-MME performance by 2.8% across seven open-source LVLMs, and the 72B model surpasses GPT-4o.
