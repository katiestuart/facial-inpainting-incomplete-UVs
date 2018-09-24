# Deep Learning for Facial Inpainting of Incomplete UV Maps on multi-pose data in-the-wild

Image Inpainting has been an important topic of research for decades, since the beginning of Machine Learning, from an academia and industry perspective.
Much progress has been made in this area, firstly because of CNNs which were built for machine vision and more recently GANs. In this project three approaches have been used for face inpainting. PCA, a traditional machine learning technique. Partial Convolution, a novel extension of CNNs and finally an adaptive GAN to tackle inpainting problems. The novelty of this project comes from applying these techniques to images in-the-wild, so this problem is completely unsupervised making it much more challenging. In both of the original papers for Partial Convolution and GAN, ground truths were available and so the models could learn to inpaint from real data. In this project, using the Multi-Attribute Facial Landmark dataset, PCA successfully inpainted missing data and completed 3D representations of the face across all pose variations. Furthermore, developed GAN Cascade and blended PCA-PConv architectures improved face reconstruction detail and obtained more refined inpainting results. The learnings from within this project lay a great foundation for future developments in the area. 
<br/><br/><br/>

### 0. Data Prep
Includes data prep functions and scattered interpolation code
### 1. PCA
PCA master code & per epoch results for 32, 64, 128, 256 hidden units
### 2. ANNs
Deep Learning master code & per epoch results for 4 tested architectures
### 3. PCONV
Includes Grid Search code, PConv master code, Dilation code and results for 19k, 600 sample and dilation tests.
### 4. PCA PCONV Cascade
Includes cascade code and output per thousand iterations.
### 5. GANs
GAN master code included, along with results at 43, 53 and 97 epochs
### 6. PCA PCONV GAN Cascade
Contains code and results for the cascade model
### 7. Blended PCA PCONV
Contains code and results for the blended model

<br/><br/>
### Acknowledgments

Partial Convolutions code:  
https://github.com/naoto0804/pytorch-inpainting-with-partial-conv

Atrous Convolutions:
https://github.com/DeepMotionAIResearch/DenseASPP/blob/master/models/MobileNetDenseASPP.py

GAN Context Encoder:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/context_encoder.py

<br/><br/>
### Author
kastuart9@gmail.com
