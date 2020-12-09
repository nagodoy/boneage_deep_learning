# Predicting Bone Age Using Deep Learning

Sunna Jo



## Objective

Determine factors that may be important to consider in using deep learning to predict bone age by training and comparing different models



## Data

RSNA Pediatric Bone Age Challenge 2017 dataset

- 14,236 X-rays of the left hand and wrist of pediatric patients
- 1-228 months
- 54% M, 46% F

Images provided by: Stanford University, University of Colorado, UCLA

Accessed via:

- https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017
- https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739
- https://www.kaggle.com/kmader/rsna-bone-age



## Process

1) Data cleaning, exploratory data analysis

2) Image pre-processing, including contrast enhancement and thresholding methods

3) Data augmentation

4) Modeling

- Trained several different convolutional neural network-based models to attempt to determine which factors are salient in bone age prediction using this infrastructure
- Transfer learning
  - Used pre-trained CNN (Xception, which is relatively lightweight and accurate) as convolutional base, removing final densely connected layer
  - Fine-tuned convolutional base by unfreezing the last 1-3 blocks of convolutional layers and re-training these layers along with the additional networks
- Attention mechanism
- Models:
  - Baseline: Xception --> global average pooling (GAP) --> fully connected layers
  - Baseline with sex as additional input
  - Attention mechanism: Xception --> sequential convolutional layers --> GAP --> fully connected layers
  - Attention mechanism + sex as additional input

The architecture for the sex input model was inspired by previously developed models, including the winner of the RSNA Bone Age Challenge (https://www.16bit.ai/blog/ml-and-future-of-radiology).

In order to implement the attention mechanism and augment data appropriately with multiple inputs, code was adapted from the following sources:

​    \- 'Attention on Pretrained-VGG16 for Bone Age' notebook by K Scott Mader (https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)

​    \- 'KU BDA 2019 boneage project' notebook by Mads Ehrhorn (https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project)

- Loss/error metric: mean absolute error in months (easily interpretable, less affected by outliers)
- Comparison of models
  - MAE
  - Diagnostics: testing regression assumptoins, residuals analysis
- Results analysis



## Results

- The model implementing the attention mechanism and incorporating sex as a feature/additional input yielded the best performance on the test set, with a MAE of 8.9 months.
- Incorporating sex as a feature/additional input in the model improved performance for both the baseline and attention mechanism models.
- Across all models, the MAE was higher for female images compared to male images and this discrepancy decreased in those models where sex was incorporated as a feature.
- The attention mechanism as it was implemented in this project did not significantly improve performance of the baseline model but, when sex was also incorporated into the model as a feature, performance of the model improved significantly and more than when sex was incorporated as a feature alone.
- MAE was higher for younger ages



## Takeaways

- Consider sex-related differences
- Sex-related differences may be associated with regions of interest
- Identifying regions of interest prior to or early on in the modeling process may improve performance
- More refined attention mechanisms or image segmentation may be necessary to identify salient features
- Consider age-related differences



## Potential Impact

Bone age is an important clinical tool that has been in existence for >75 years and has both clinical and non-clinical applications/use cases. Determining bone age can be a difficult task as it is complex and time-consuming and there are limitations to current methods, including inter-observer variability and lack of generalizability to certain populations. As such, this tool would benefit greatly from automation and there are several studies that have applied deep learning to this interpretation. Determining which factors are important to consider in using deep learning/building models to achieve this task is important in streamlining pipelines, as well as improving existing models.



## Considerations/Future Work

- Obtain additional data
- Incorporate other features, additional feature engineering
- Refine attention mechanism, image segmentation
- Ensembling to improve model performance
- Build out application to include educational information and resources



## Techniques/Skills

- Deep learning
  - Convolutional neural networks
  - Transfer learning
  - Attention mechanism
  - Image pre-processing
  - Data augmentation
- Regression analysis
- Data visualization
- Web application development

## Tools/Libraries

- TensorFlow
- Keras
- pandas
- numpy
- matplotlib
- seaborn
- Streamlit



## References

Data sources:https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739https://www.kaggle.com/kmader/rsna-bone-age
Chollet, F. https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb. [Accessed Dec 2020].
Creo AL, Schwenk WF. Bone age: a handy tool for pediatric providers. *Pediatrics*. 2017;140(6):e20171486.
F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, 2017, pp. 1800-1807, doi: 10.1109/CVPR.2017.195.
H. Fukui, T. Hirakawa, T. Yamashita and H. Fujiyoshi, "Attention Branch Network: Learning of Attention Mechanism for Visual Explanation," *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 2019, pp. 10697-10706, doi: 10.1109/CVPR.2019.01096.
Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. The RSNA Pediatric Bone Age Machine Learning Challenge. Radiology 2018; 290(2):498-503. 
Model architecture and code adapted from:M. Cicero and A. Bilbily, “Machine Learning and the Future of Radiology: How we won the 2017 RSNA ML Challenge,” 16bit.ai, Nov. 23, 2017. [Online]. Available: https://www.16bit.ai/blog/ml-and-future-of-radiology. [Accessed Dec 2020].Mader, KS. “Attention on Pretrained-VGG16 for Bone Age”.  https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age. [Accessed Dec 2020].Ehrhorn, M. “KU BDA 2019 boneage project”. https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project. [Accessed Dec 2020].