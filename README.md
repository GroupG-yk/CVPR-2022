# CVPR-2022


##  Gan

### Unsupervised Domain Adaptation for Nighttime Aerial Tracking

Junjie Ye†, Changhong Fu†,*, Guangze Zheng†, Danda Pani Paudel‡, and Guang Chen†, †Tongji University, China ‡ETH Zurich, Switzerland


Previous advances in object tracking mostly reported on favorable illumination circumstances while neglecting performance at nighttime, which significantly impeded the development of related aerial robot applications. This work instead develops a novel unsupervised domain adaptation framework for nighttime aerial tracking (named UDAT). Specifically, a unique object discovery approach is provided to generate training patches from raw nighttime tracking videos. To tackle the domain discrepancy, we employ a Transformer-based bridging layer post to the feature extractor to align image features from both domains. With a Transformer day/night feature discriminator, the daytime tracking model is adversarially trained to track at night. Moreover, we construct a pioneering benchmark namely NAT2021 for unsupervised domain adaptive nighttime tracking, which comprises a test set of 180 manually annotated tracking sequences and a train set of over 276k unlabelled nighttime tracking frames. Exhaustive experiments demonstrate the robustness and domain adaptability of the proposed framework in nighttime aerial tracking. The code and benchmark are available at https: //github.com/vision4robotics/UDAT. 

![image](https://user-images.githubusercontent.com/61666816/166941455-3ccfa1cd-28e1-4cc0-9ec6-3b7f3a2348b2.png)


##  图像去噪/Image Denoising

### Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots

Zejin Wang1,2 Jiazheng Liu1,3 Guoqing Li1 Hua Han1,3,*
1National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences
2School of Artificial Intelligence, University of Chinese Academy of Sciences
3School of Future Technology, University of Chinese Academy of Sciences

Real noisy-clean pairs on a large scale are costly  and difficult to obtain. Meanwhile, supervised denoisers  trained on synthetic data perform poorly in practice. Selfsupervised denoisers, which learn only from single noisy  images, solve the data collection problem. However, selfsupervised denoising methods, especially blindspot-driven  ones, suffer sizable information loss during input or network design. The absence of valuable information dramatically reduces the upper bound of denoising performance.
In this paper, we propose a simple yet efficient approach  called Blind2Unblind to overcome the information loss in  blindspot-driven denoising methods. First, we introduce a  global-aware mask mapper that enables global perception  and accelerates training. The mask mapper samples all pixels at blind spots on denoised volumes and maps them to  the same channel, allowing the loss function to optimize  all blind spots at once. Second, we propose a re-visible  loss to train the denoising network and make blind spots  visible. The denoiser can learn directly from raw noise images without losing information or being trapped in identity  mapping. We also theoretically analyze the convergence of  the re-visible loss. Extensive experiments on synthetic and  real-world datasets demonstrate the superior performance  of our approach compared to previous work. Code is available at https://github.com/demonsjin/Blind2Unblind.

![image](https://user-images.githubusercontent.com/61666816/166947009-0129d927-5aa5-4171-a506-30cc968aee15.png)


### Learning to Deblur using Light Field Generated and Real Defocus Images

Lingyan Ruan1∗ Bin Chen2∗ Jizhou Li3  Miuling Lam1†
1City University of Hong Kong 2Max-Planck-Institut fur Informatik  3Stanford University
http://lyruan.com/Projects/DRBNet

Defocus deblurring is a challenging task due to the spatially varying nature of defocus blur. While deep learning  approach shows great promise in solving image restoration  problems, defocus deblurring demands accurate training  data that consists of all-in-focus and defocus image pairs,  which is difficult to collect. Naive two-shot capturing cannot achieve pixel-wise correspondence between the defocused and all-in-focus image pairs. Synthetic aperture of  light fields is suggested to be a more reliable way to generate accurate image pairs. However, the defocus blur generated from light field data is different from that of the images captured with a traditional digital camera. In this paper, we propose a novel deep defocus deblurring network  that leverages the strength and overcomes the shortcoming  of light fields. We first train the network on a light fieldgenerated dataset for its highly accurate image correspondence. Then, we fine-tune the network using feature loss on  another dataset collected by the two-shot method to alleviate the differences between the defocus blur exists in the two  domains. This strategy is proved to be highly effective and  able to achieve the state-of-the-art performance both quantitatively and qualitatively on multiple test sets. Extensive  ablation studies have been conducted to analyze the effect  of each network module to the final performance.

![image](https://user-images.githubusercontent.com/61666816/166947420-e9dbdb21-9697-4110-8ced-6a9906ef4737.png)


###  Dancing under the stars: video denoising in starlight
Kristina Monakhova   Stephan R. Richter  Laura Waller  Vladlen Koltun
UC Berkeley            Intel Labs        UC Berkeley     Intel Labs

Imaging in low light is extremely challenging due to  low photon counts. Using sensitive CMOS cameras, it is  currently possible to take videos at night under moonlight
(0.05-0.3 lux illumination). In this paper, we demonstrate  photorealistic video under starlight (no moon present,
<0.001 lux) for the first time. To enable this, we develop a
GAN-tuned physics-based noise model to more accurately  represent camera noise at the lowest light levels. Using  this noise model, we train a video denoiser using a  combination of simulated noisy video clips and real noisy  still images. We capture a 5-10 fps video dataset with  significant motion at approximately 0.6-0.7 millilux with no  active illumination. Comparing against alternative methods,  we achieve improved video quality at the lowest light levels,  demonstrating photorealistic video denoising in starlight for  the first time.

![image](https://user-images.githubusercontent.com/61666816/166947737-3592498a-1ef0-42a4-bfd1-0fe4161625d6.png)


## 深度估计/Depth Estimation

### Occlusion-Aware Cost Constructor for Light Field Depth Estimation
Yingqian Wang, Longguang Wang, Zhengyu Liang, Jungang Yang, Wei An, Yulan Guo
National University of Defense Technology

Matching cost construction is a key step in light field
(LF) depth estimation, but was rarely studied in the deep  learning era. Recent deep learning-based LF depth estimation methods construct matching cost by sequentially  shifting each sub-aperture image (SAI) with a series of predefined offsets, which is complex and time-consuming. In  this paper, we propose a simple and fast cost constructor to  construct matching cost for LF depth estimation. Our cost  constructor is composed by a series of convolutions with  specifically designed dilation rates. By applying our cost  constructor to SAI arrays, pixels under predefined disparities can be integrated and matching cost can be constructed  without using any shifting operation. More importantly, the  proposed cost constructor is occlusion-aware and can handle occlusions by dynamically modulating pixels from different views. Based on the proposed cost constructor, we  develop a deep network for LF depth estimation. Our network ranks first on the commonly used 4D LF benchmark  in terms of the mean square error (MSE), and achieves a  faster running time than other state-of-the-art methods.

![image](https://user-images.githubusercontent.com/61666816/166949284-069da69e-755c-458c-b080-2e34f20fed73.png)


## 图像复原/Image Restoration

### Deep Generalized Unfolding Networks for Image Restoration
Chong Mou†, Qian Wang†, Jian Zhang†,‡
†Peking University Shenzhen Graduate School, Shenzhen, China
‡Peng Cheng Laboratory, Shenzhen, China

Deep neural networks (DNN) have achieved great success in image restoration. However, most DNN methods are  designed as a black box, lacking transparency and interpretability. Although some methods are proposed to combine traditional optimization algorithms with DNN, they  usually demand pre-defined degradation processes or handcrafted assumptions, making it difficult to deal with complex and real-world applications. In this paper, we propose  a Deep Generalized Unfolding Network (DGUNet) for image restoration. Concretely, without loss of interpretability,  we integrate a gradient estimation strategy into the gradient descent step of the Proximal Gradient Descent (PGD)  algorithm, driving it to deal with complex and real-world  image degradation. In addition, we design inter-stage information pathways across proximal mapping in different
PGD iterations to rectify the intrinsic information loss in  most deep unfolding networks (DUN) through a multi-scale  and spatial-adaptive way. By integrating the flexible gradient descent and informative proximal mapping, we unfold  the iterative PGD algorithm into a trainable DNN. Extensive experiments on various image restoration tasks demonstrate the superiority of our method in terms of state-of-theart performance, interpretability, and generalizability. The  source code is available at github.com/MC-E/DGUNet.

![image](https://user-images.githubusercontent.com/61666816/166950145-3d51efce-a805-404a-9a2b-74dcf2018caf.png)


## 其他/Others
### Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection
Mingtao Feng1* Kendong Liu1∗ Liang Zhang1† Hongshan Yu2 Yaonan Wang2 Ajmal Mian3
1Xidian University, 2Hunan University, 3The University of Western Australia


Saliency detection with light field images is becoming attractive given the abundant cues available, however, this  comes at the expense of large-scale pixel level annotated  data which is expensive to generate. In this paper, we  propose to learn light field saliency from pixel-level noisy  labels obtained from unsupervised hand crafted featuredbased saliency methods. Given this goal, a natural question  is: can we efficiently incorporate the relationships among  light field cues while identifying clean labels in a unified  framework? We address this question by formulating the  learning as a joint optimization of intra light field features  fusion stream and inter scenes correlation stream to generate the predictions. Specially, we first introduce a pixel  forgetting guided fusion module to mutually enhance the  light field features and exploit pixel consistency across iterations to identify noisy pixels. Next, we introduce a cross  scene noise penalty loss for better reflecting latent structures of training data and enabling the learning to be invariant to noise. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our framework  showing that it learns saliency prediction comparable to  state-of-the-art fully supervised light field saliency methods. Our code is available at https://github.com/
OLobbCode/NoiseLF.

![image](https://user-images.githubusercontent.com/61666816/166952168-fa2458a3-6069-4412-be70-25a47df0a3f8.png)

### Toward Fast, Flexible, and Robust Low-Light Image Enhancement

Long Ma†, Tengyu Ma†, Risheng Liu‡*, Xin Fan‡, Zhongxuan Luo†
†School of Software Technology, Dalian University of Technology ‡International School of Information Science & Engineering, Dalian University of Technology

Existing low-light image enhancement techniques are mostly not only difficult to deal with both visual quality and computational efficiency but also commonly invalid in unknown complex scenarios. In this paper, we develop a new Self-Calibrated Illumination (SCI) learning framework for fast, flexible, and robust brightening images in real-world low-light scenarios. To be specific, we establish a cascaded illumination learning process with weight sharing to handle this task. Considering the computational burden of the cascaded pattern, we construct the self-calibrated module which realizes the convergence between results of each stage, producing the gains that only use the single basic block for inference (yet has not been exploited in previous works), which drastically diminishes computation cost. We then define the unsupervised training loss to elevate the model capability that can adapt general scenes. Further, we make comprehensive explorations to excavate SCI s inherent properties (lacking in existing works) including operation-insensitive adaptability (acquiring stable performance under the settings of different simple operations) and model-irrelevant generality(can be applied to illumination-based existing works to improve performance). Finally, plenty of experiments and ablation studies fully indicate our superiority in both quality and efficiency. Applications on low-light face detection  and nighttime semantic segmentation fully reveal the latent  practical values for SCI. The source code is available at  https://github.com/vis-opt-group/SCI.

![image](https://user-images.githubusercontent.com/61666816/166952566-bd53a7a7-fd9c-4b2a-a0d7-6c1398f5cacb.png)


SIMBAR: Single Image-Based Scene Relighting For Effective Data Augmentation For Automated Driving Vision Tasks
Xianling Zhang1*, Nathan Tseng2*, Ameerah Syed1, Rohan Bhasin1, Nikita Jaipuria1
1Ford Greenfield Labs, Palo Alto 2University of Michigan

Real-world autonomous driving datasets comprise of images aggregated from different drives on the road. The  ability to relight captured scenes to unseen lighting conditions, in a controllable manner, presents an opportunity  to augment datasets with a richer variety of lighting conditions, similar to what would be encountered in the realworld. This paper presents a novel image-based relighting pipeline, SIMBAR, that can work with a single image  as input. To the best of our knowledge, there is no prior  work on scene relighting leveraging explicit geometric representations from a single image. We present qualitative  comparisons with prior multi-view scene relighting baselines. To further validate and effectively quantify the benefit of leveraging SIMBAR for data augmentation for automated driving vision tasks, object detection and tracking  experiments are conducted with a state-of-the-art method,  a Multiple Object Tracking Accuracy (MOTA) of 93.3% is  achieved with CenterTrack on SIMBAR-augmented KITTI -  an impressive 9.0% relative improvement over the baseline
MOTA of 85.6% with CenterTrack on original KITTI, both  models trained from scratch and tested on Virtual KITTI.
For more details and relit datasets, please visit our project  website (https://simbarv1.github.io).


![image](https://user-images.githubusercontent.com/61666816/166953046-7043c32a-4164-44c0-a2cb-df482816cdae.png)

