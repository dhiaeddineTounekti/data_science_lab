## 1. research questions

There is a lack of images with cencer cells with in different organs, due to lack of sample images under microscope and mutual agreements from patients.  
Thus, CNN and other classification network do not have enough data to train the model and achieve a high accuracy.

Based on images segmentation results of brain MRI iamges(or other datasets)
First we try to do segmentation of cancer cells of data sets of infected+ normal cells. 
We implement a GAN network, to generate new MRI brain images with brain tumor.

Last, We will train a model with these new images together with real images, and compare with the existing model with only real data.

Research Questions. We mainly address two questions:
- GAN Selection: Which GAN architecture works the best for realistic medical image generation, especially for Brain MRI images?
- CNN Image classifier: Can generated images of Brain MRI scan can imrpove accuracy of Medical Image classifier?

## 2. Research Objectives of this paper:
We expect that GAN networks can generate images which cannot be distinguished by general public and brain tumor classifier.
We also expect that data augmentation method proposed in this paper can help increase medical images.


(Generative Adversarial Networks, or GANs, are an architecture for training generative models, such as deep convolutional neural networks for generating images.)


## 3. choices between 2+ datasets
- Brain images datasets from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation/notebooks


## 4. State of the art(Image segmentation & cGAN/GAN/image generation)
Feature pyramid network(2017) vs UNet(2015) vs sliding window (2012)vs hypercolumn vs fully connected network(2014)
keyword: backward convolution

#### BERT model (transfer learning to other medical images)

#### GAN-based synthetic brain MR image generation) (83 cite)
Sana. https://ieeexplore.ieee.org/document/8363678 

#### Synthetic data augmentation using GAN for improved liver lesion classification) (237 cite)
Dhia, https://ieeexplore.ieee.org/document/8363576 

#### MRI images Enhancement and Tumor Segmentation for Brain) (10 cite)
- https://sci-hub.se/10.1109/PDCAT.2017.00051

#### A Comparative study and analysis of Contrast Enhancement algorithms for MRI Brain Image sequences)
- https://sci-hub.se/10.1109/ICCCNT.2018.8494068 

#### Classification of Brain Cancer Using Artificial Neural Network ) (147 cite)
- https://sci-hub.se/https://doi.org/10.1109/ICECTECH.2010.5479975 

#### Enhanced classifier training to improve precision of a convolutional neural network to identify images of skin lesions) (3 cite)
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0218713

#### A deep learning-based algorithm for 2-D cell segmentation in microscopy images) (17cite)
- https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2375-z 

#### (Using Deep Learning for Segmentation and Counting within Microscopy Data) (cite 10)
- https://arxiv.org/pdf/1802.10548.pdf 

#### Detection and segmentation of morphologically complex eukaryotic cells in fluorescence microscopy images via feature pyramid fusion (new)
- https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008179

#### Multi-grade brain tumor classification using deep CNN with extensive data augmentation) (74 cite)
- https://www.sciencedirect.com/science/article/abs/pii/S1877750318307385 

####  Cell Image Segmentation Using Generative Adversarial Networks, Transfer Learning, and Augmentations)
- https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Majurski_Cell_Image_Segmentation_Using_Generative_Adversarial_Networks_Transfer_Learning_and_CVPRW_2019_paper.pdf 

#### Khouloud: (Accurate cell segmentation in microscopy images using membrane patterns) (101 cite)
- https://academic.oup.com/bioinformatics/article/30/18/2644/2475348 

#### Conditional Adversarial Network for Semantic Segmentation of Brain Tumor
- https://arxiv.org/pdf/1708.05227.pdf

#### U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/pdf/1505.04597.pdf

#### Understanding Deep Learning Techniques for Image Segmentation
 - https://arxiv.org/pdf/1907.06119v1.pdf

#### Segmentation of brain tumor tissues with convolutional neural networks.
https://www.researchgate.net/publication/303703706_Segmentation_of_Brain_Tumor_Tissues_with_Convolutional_Neural_Networks

####  The multimodal brain tumor image segmentation benchmark
https://ieeexplore.ieee.org/document/6975210

#### Exploring Deep Learning Networks for Tumour Segmentation in Infrared Images
https://www.qirt2018.de/portals/qirt18/doc/We.4.B.4.pdf

## 5. Example analysis that we gonna perform in phase 2
#### Examples: 
https://towardsdatascience.com/how-we-built-an-easy-to-use-image-segmentation-tool-with-transfer-learning-546efb6ae98

https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0

Paper with codes implementing data segmentation!
https://paperswithcode.com/task/brain-tumor-segmentation#code

Checker board issues when using U-NET
- https://distill.pub/2016/deconv-checkerboard/

## 6.approaches we gonna try in this paper: UNET/pix2pix/ etc
TumorGAN/cGAN described in this paper. 
- https://www.mdpi.com/1424-8220/20/15/4203?fbclid=IwAR1Ny-92mX-2DPMcIHIWuAegxRRNh1FBwdr_Oo5j_ErZEbuA7p2XkFpzQmg

## 7. Phase 2 TO-DO list
#### Assume that we will use PyTorch as our coding framework.
#### - do same data exploration (define the data, and analyse it, pick point of interest)
#### - what we did for the classic data augementation (rotation, zooming, adding different kind of noise, .....
- look for augmentation for cGAN
- trying to use "transform compose" method in pyTorch.
#### - write about the methodoligies and describe pix2pix/TimorGAN/cGAN  ....

### Example code
- GAN: https://github.com/eriklindernoren/PyTorch-GAN
- TensorFlow: https://github.com/kozistr/Awesome-GANs?fbclid=IwAR3IHfEsZbv6KZdctyceA9jMAX_AU3fjGPdQ3bGG16bNv_RQp8iKaij5Gl8
## 8. Link to overleaf
https://www.overleaf.com/project/5fbd1c0388690469a98d07f7