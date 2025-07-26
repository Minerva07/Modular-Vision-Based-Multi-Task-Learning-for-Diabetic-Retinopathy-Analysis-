# Modular-Vision-Based-Multi-Task-Learning-for-Diabetic-Retinopathy-Analysis-
A modular multi-task deep learning framework for Diabetic Retinopathy analysis using the IDRiD dataset, combining ResNet-50 + U-Net with attention for lesion segmentation and 5-class DR grading, enhanced with a Mixture-of-Experts routing mechanism.

**Model Design and Justification**
The model employs a shared encoder-decoder architecture with the following key components: 
• Shared Encoder: ResNet-50 pretrained backbone extracts hierarchical features across 5 levels (64→256→512→1024→2048 channels). 
• Mixture of Experts (MoE): Two expert networks process the deepest features (2048dimensional) with dynamic routing based on image content, fostering specialized feature learning. 
• Classification Head: Adaptive average pooling followed by fully connected layer for 5class DR grading. 
• U-Net Decoder with Attention: Skip connections from , modulated by attention gates, enable precise pixel-level lesion segmentation for four lesion types (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates). Attention gates help the decoder focus on relevant regions, leading to cleaner segmentation masks. 
Design Justification: 
The multi-task architecture is justified by the inherent relationship between DR grading and lesion segmentation in DR diagnosis. By training both tasks simultaneously, the model can 
• Leverage shared representations: Features learned for segmentation (e.g., detecting exudates) can benefit classification (e.g., grading severity based on exudate presence). 
• Improve generalization: Learning multiple related tasks can lead to more robust and generalizable feature extractors. 
• Efficiency: A single model performs two tasks, reducing computational overhead compared to separate models 
**Dataset Handling and Preprocessing** 
The project utilizes the IDRiD (Indian Diabetic Retinopathy Image Dataset), which provides both retinal images, corresponding segmentation masks for various lesions, and disease grading labels. 
Dataset Structure: 
• Training Set: 54 images with corresponding segmentation masks and disease grades 
• Testing Set: 27 images for validation 
• Lesion Types: 4 categories (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates) 
• Disease Grades: 5-level DR severity classification (0-4) Preprocessing Pipeline: 
• Image Normalization: Used ImageNet statistics (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) 
• Resizing: Standardized to 512×512 pixels using area interpolation 
• Mask Processing: Raw masks normalized to [0,1] range, with nearest neighbor interpolation preserving boundaries. 
• Data Validation: Automatic filtering ensures image-mask correspondence 
Data Augmentation Strategy: 
To enhance robustness, training augmentations include: 
• Geometric Augmentations: HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate, ElasticTransform, GridDistortion.  
• Color Augmentations: GaussNoise, ISONoise, MultiplicativeNoise, ColorJitter.  
**Training and Evaluation Strategy** 
Training Configuration: 
• Optimizer: Adam with learning rate 1e-4, weight decay 1e-5 
• Batch Size: 4 (memory-constrained optimization) 
• Epochs: 20 (demonstration setup) 
• Loss Functions:  
o Classification: CrossEntropyLoss 
o Segmentation: A weighted sum of BCEWithLogitsLoss and DiceLoss 
o CombinedMulti-task Loss Formulation: L_total = α·L_classification + α·L_segmentation + β·L_routing 
Evaluation Metrics: 
o Classification: Accuracy (correct predictions / total samples) 
o Segmentation: Intersection over Union (IoU) score 
o Combined Metric: Average of accuracy and IoU for model selection 
Performance Metrics and Analysis 
The model architecture is designed to achieve: 
**• Classification: **
Accuracy: The proportion of correctly classified disease grades in the validation set. This provides a straightforward measure of the model's ability to categorize DR severity. Equal weighting ensures neither task dominates training 
**• Segmentation:  **
Intersection over Union (IoU): The iou_score function calculates this metric, applying a sigmoid activation and a threshold (0.4) to the raw predictions. 
Dice Score: Another common metric for segmentation, especially useful for imbalanced classes. It is closely related to IoU. The dice_score function calculates this with a threshold of 0.3. 
The Trainer class tracks these metrics over epochs and uses a combined metric to determine the "best" model for saving and for learning rate scheduling. 
Training curves visualization for convergence analysis 
**Visualizations of Results**
• Ground Truth Analysis (visualize_sample): Displays original images, binarized, and raw ground truth masks to verify data loading and preprocessing. 
• Prediction Comparison(visualize_predictions): Shows original image, ground truth masks, and binarized predicted masks, enabling qualitative assessment of segmentation performance. 
• Training Dynamics (plot_training_curves): Plots training losses (total, classification, segmentation) and validation metrics (accuracy, IoU) over epochs. These curves help monitor convergence, identify overfitting, and assess generalization ability. 
**Conclusion** 
This multi-task vision model represents a comprehensive approach to diabetic retinopathy analysis, combining disease grading and lesion segmentation in a unified framework. The architecture leverages proven components (ResNet-50, U-Net) while incorporating modern techniques (mixture of experts) for enhanced performance. The robust preprocessing pipeline and extensive visualization capabilities ensure reliable training and interpretable results, making it suitable for both research and clinical applications. 
The modular design allows for easy adaptation to different datasets and task combinations, while the comprehensive evaluation framework provides thorough performance assessment across multiple metrics and visualization modalities. 
Observations from the evaluation indicate that the model is able to classify the disease grading well, demonstrating strong performance in categorizing the severity of diabetic retinopathy. However, the overlap of predicted masks over ground truth for segmentation is not as robust, leading to a lower Intersection over Union (IoU) score for the segmentation task. This suggests that while the model effectively identifies the presence and overall grade of the disease, pixel-level precision in delineating specific lesions could be further improved.
