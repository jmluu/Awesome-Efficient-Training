# **Awesome Efficient Training**
A collection of research papers on efficient training of DNNs. If you find some ignored papers, please open issues or pull requests.

## **Contents**
  - [**Algorithm**](#algorithm)
    - [**Quantization**](#quantization)
    - [**Pruning**](#pruning)
    - [**Others**](#others)
  - [**Hardware**](#hardware)
    - [**ASIC**](#asic)
    - [**FPGA**](#fpga)
    - [**PIM**](#pim)
    - [**SNN**](#snn)

## **Algorithm**

### **Quantization**
-  [**2021 | AAAI**] Distribution Adaptive INT8 Quantization for Training CNNs [[paper](https://arxiv.org/abs/2102.04782)]
-  [**2021 | ICLR**] CPT: Efficient Deep Neural Network Training via Cyclic Precision [[paper](http://arxiv.org/abs/2101.09868)] [[code](https://github.com/RICE-EIC/CPT)]
-  [**2021 | tinyML**] TENT: Efficient Quantization of Neural Networks on the tiny Edge with Tapered FixEd PoiNT [[paper](http://arxiv.org/abs/2104.02233)] 
-  [**2021 | arXiv**] RCT: Resource Constrained Training for Edge AI [[paper](http://arxiv.org/abs/2103.14493)]
-  [**2021 | arXiv**] A Simple and Efficient Stochastic Rounding Method for Training Neural Networks in Low Precision [[paper](http://arxiv.org/abs/2103.13445)]
-  [**2021 | arXiv**] Enabling Binary Neural Network Training on the Edge [[paper](http://arxiv.org/abs/2102.04270)]
-  [**2021 | arXiv**] In-Hindsight Quantization Range Estimation for Quantized Training [[paper](http://arxiv.org/abs/2105.04246)]
-  [**2021 | arXiv**] Towards Efficient Full 8-bit Integer DNN Online Training on Resource-limited Devices without Batch Normalization [[paper](http://arxiv.org/abs/2105.13890)]
-  [**2021 | arXiv**]Low-Precision Training in Logarithmic Number System using Multiplicative Weight Update [[paper](http://arxiv.org/abs/2106.13914)]
-  [**2020 | Neural Networks**] Training High-Performance and Large-Scale Deep Neural Networks with Full 8-bit Integers [[paper](https://arxiv.org/abs/1909.02384))] [[code](https://github.com/yang-yk/wageubn)]
-  [**2020 | TC**] Evaluations on Deep Neural Networks Training Using Posit Number System [[paper](https://ieeexplore.ieee.org/document/9066876)]
-  [**2020 | CVPR**] Towards Unified INT8 Training for Convolutional Neural Network [[paper](https://arxiv.org/abs/1912.12607)]
-  [**2020 | CVPR**] Fixed-Point Back-Propagation Training [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Fixed-Point_Back-Propagation_Training_CVPR_2020_paper.pdf)]
-  [**2020 | ICLR**] Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks [[paper](https://arxiv.org/abs/2001.05674)]
-  [**2020 | ICML**] Multi-Precision Policy Enforced Training (MuPPET): A precision-switching strategy for quantised fixed-point training of CNNs [[paper](https://arxiv.org/abs/2006.09049)]
-  [**2020 | IJCAI**] Reducing Underflow in Mixed Precision Training by Gradient Scaling [[paper](https://www.ijcai.org/proceedings/2020/404)]
-  [**2020 | NIPS**] FracTrain: Fractionally Squeezing Bit Savings Both Temporally and Spatially for Efficient DNN Training [[paper](https://arxiv.org/abs/2012.13113)] [[code](https://github.com/RICE-EIC/FracTrain)]
-  [**2020 | NIPS**] Ultra-Low Precision 4-bit Training of Deep Neural Networks [[paper](https://papers.nips.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf)]
-  [**2020 | NIPS**] A Statistical Framework for Low-bitwidth Training of Deep Neural Networks [[paper](https://arxiv.org/abs/2010.14298)] [[code](https://github.com/cjf00000/StatQuant)]
-  [**2020 | arXiv**] Adaptive Precision Training for Resource Constrained Devices [[paper](https://arxiv.org/abs/2012.12775)]
-  [**2020 | arXiv**] Training and Inference for Integer-Based Semantic Segmentation Network [[paper](https://arxiv.org/abs/2011.14504)] [[code](https://github.com/MarkYangjiayi/Semantic-Quantization)]
-  [**2020 | arXiv**] NITI: Training Integer Neural Networks Using Integer-only Arithmetic [[paper](https://arxiv.org/abs/2009.13108)] [[code](https://github.com/wangmaolin/niti)]
-  [**2020 | arXiv**] Neural gradients are lognormally distributed: understanding sparse and quantized training [[paper](http://arxiv.org/abs/2006.08173)] [[code](https://github.com/brianchmiel/Neural-gradients-are-lognormally-distributed-understanding-sparse-and-quantized-training)]
-  [**2020 | arXiv**] Exploring the Potential of Low-bit Training of Convolutional Neural Networks [[paper](https://arxiv.org/abs/2006.02804)]
-  [**2019 | JETCAS**] FloatSD: A New Weight Representation and Associated Update Method for Efficient Convolutional Neural Network Training [[paper](https://ieeexplore.ieee.org/document/8693838)]
-  [**2019 | ICLR**] Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm [[paper](https://arxiv.org/abs/1812.11732)]
-  [**2019 | ICLR**] Accumulation Bit-Width Scaling For Ultra-Low Precision Training Of Deep Networks [[paper](https://arxiv.org/abs/1901.06588)]
-  [**2019 | ICML**] SWALP: Stochastic Weight Averaging in Low-Precision Training [[paper](https://arxiv.org/abs/1904.11943)] [[code](https://github.com/stevenygd/SWALP)]
-  [**2019 | NIPS**] Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks
 [[paper](https://proceedings.neurips.cc/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf)]
-  [**2019 | NIPS**] Backprop with Approximate Activations for Memory-efficient Network Training [[paper](https://arxiv.org/abs/1901.07988v1)] [[code](https://github.com/ayanc/blpa)]
-  [**2019 | NIPS**] Dimension-Free Bounds for Low-Precision Training [[paper](https://papers.nips.cc/paper/2019/file/d4cd91e80f36f8f3103617ded9128560-Paper.pdf)]
-  [**2019 | arXiv**] Cheetah: Mixed Low-Precision Hardware & Software Co-Design Framework for DNNs on the Edge [[paper](https://arxiv.org/abs/1908.02386v1)]
-  [**2019 | arXiv**] Distributed Low Precision Training Without Mixed Precision [[paper](https://arxiv.org/abs/1911.07384)]
-  [**2019 | arXiv**] Mixed Precision Training With 8-bit Floating Point [[paper](https://arxiv.org/abs/1905.12334)]
-  [**2019 | arXiv**] A Study of BFLOAT16 for Deep Learning Training [[paper](https://arxiv.org/abs/1905.12322)]
-  [**2018 | ACL**] Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq [[paper](https://arxiv.org/abs/1805.10387)]
-  [**2018 | ECCV**] Value-aware Quantization for Training and Inference of Neural Networks [[paper](https://arxiv.org/abs/1804.07802)]
-  [**2018 | ICCD**] Training Neural Networks with Low Precision Dynamic Fixed-Point [[paper](https://ieeexplore.ieee.org/document/8615717)]
-  [**2018 | ICLR**] Mixed Precision Training [[paper](https://arxiv.org/abs/1710.03740)]
-  [**2018 | ICLR**] Training and Inference with Integers in Deep Neural Networks [[paper](https://arxiv.org/abs/1802.04680)] [[code](https://github.com/boluoweifenda/WAGE)]
-  [**2018 | ICLR**] Mixed Precision Training of Convolutional Neural Networks using Integer Operations [[paper](https://arxiv.org/abs/1802.00930v2)]
-  [**2018 | NIPS**] Scalable Methods for 8-bit Training of Neural Networks [[paper](https://papers.nips.cc/paper/2018/file/e82c4b19b8151ddc25d4d93baf7b908f-Paper.pdf)] [[code](https://github.com/eladhoffer/quantized.pytorch)]
-  [**2018 | NIPS**] Training Deep Neural Networks with 8-bit Floating Point Numbers [[paper](https://arxiv.org/abs/1812.08011)]
-  [**2018 | NIPS**] Training DNNs with Hybrid Block Floating Point [[paper](https://arxiv.org/abs/1804.01526)]
-  [**2018 | arXiv**] High-Accuracy Low-Precision Training [[paper](https://arxiv.org/abs/1803.03383)]
-  [**2018 | arXiv**] Low-Precision Floating-Point Schemes for Neural Network Training [[paper](https://arxiv.org/abs/1804.05267)]
-  [**2018 | arXiv**] Training Deep Neural Network in Limited Precision [[paper](https://arxiv.org/abs/1810.05486)]
-  [**2017 | ICML**] The ZipML Framework for Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning [[paper](https://arxiv.org/abs/1611.05402)] [[code](https://github.com/IST-DASLab/smart-quantizer)]
-  [**2017 | IJCNN**] FxpNet: Training a deep convolutional neural network in fixed-point representation [[paper](https://ieeexplore.ieee.org/document/7966159)]
-  [**2017 | NIPS**] Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks [[paper](https://arxiv.org/abs/1711.02213)]
-  [**2016 | arXiv**] DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [[paper](https://arxiv.org/abs/1606.06160)] [[code](https://github.com/hpi-xnor/BMXNet-v2)]
-  [**2016 | arXiv**] Convolutional Neural Networks using Logarithmic Data Representation [[paper](https://arxiv.org/abs/1603.01025)]
-  [**2015 | ICLR**] Training deep neural networks with low precision multiplications [[paper](https://arxiv.org/abs/1412.7024)]
-  [**2015 | ICML**] Deep Learning with Limited Numerical Precision [[paper](https://arxiv.org/abs/1502.02551)]
-  [**2015 | arXiv**] 8-Bit Approximations for Parallelism in Deep Learning [[paper](https://arxiv.org/abs/1511.04561)]
-  [**2014 | INTERSPEECH**] 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs [[paper](https://isca-speech.org/archive/archive_papers/interspeech_2014/i14_1058.pdf)]
 
### **Pruning**
-  [**2021 | IEEE Access**] Roulette: A Pruning Framework to Train a Sparse Neural Network From Scratch [[paper](https://ieeexplore.ieee.org/document/9374983)]
-  [**2021 | CVPR**] The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models [[paper](https://arxiv.org/abs/2012.06908)] [[code](-)]
-  [**2021 | ICLR**] Progressive Skeletonization: Trimming more fat from a network at initialization
-  [**2021 | ICLR**] Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch
-  [**2021 | IVLR**] PRUNING NEURAL NETWORKS AT INITIALIZATION: WHY ARE WE MISSING THE MARK?
-  [**2021 | ICS**] ClickTrain: Efficient and Accurate End-to-End Deep Learning Training via Fine-Grained Architecture-Preserving Pruning [[paper](https://arxiv.org/abs/2011.10170)] [[code](-)]
-  [**2021 | arXiv**] Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks [[paper](https://arxiv.org/abs/2102.08124)] [[code](https://github.com/papers-submission/structured_transposable_masks)]
-  [**2021 | arXiv**] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration [[paper](http://arxiv.org/abs/2106.10404)]
-  [**2021 | arXiv**] FreeTickets: Accurate, Robust and Efficient Deep Ensemble by Training with Dynamic Sparsity [[paper](http://arxiv.org/abs/2106.14568)]
-  [**2020 | TCAD**] Enabling On-Device CNN Training by Self-Supervised Instance Filtering and Error Map Pruning [[paper](https://arxiv.org/abs/2007.03213)] [[code](-)]
-  [**2020 | ECCV**] Accelerating CNN Training by Pruning Activation Gradients [[paper](https://arxiv.org/abs/1908.00173)]
-  [**2020 | ICLR**] Picking Winning Tickets Before Training by Preserving Gradient Flow [[paper](https://arxiv.org/abs/2002.07376)] [[code](https://github.com/alecwangcq/GraSP)]
-  [**2020 | ICLR**] Dynamic Sparse Training: Find Efficient Sparse Network From Scratch With Trainable Masked Layers [[paper](https://arxiv.org/abs/2005.06870)] [[code](https://github.com/junjieliu2910/DynamicSparseTraining)]
-  [**2020 | ICLR**] Drawing early-bird tickets: Towards more efficient training of deep networks [[paper](https://arxiv.org/abs/1909.11957)] [[code](https://github.com/RICE-EIC/Early-Bird-Tickets)]
-  [**2020 | MICRO**] Procrustes: a Dataflow and Accelerator for Sparse Deep Neural Network Training [[paper](https://arxiv.org/abs/2009.10976)]
-  [**2020 | NIPS**] Sparse Weight Activation Training [[paper](https://arxiv.org/abs/2001.01969)]
-  [**2020 | arXiv**] Progressive Gradient Pruning for Classification, Detection and DomainAdaptation [[paper](https://arxiv.org/abs/1906.08746)] [[code](https://github.com/Anon6627/Pruning-PGP)]
-  [**2020 | arXiv**] Gradual Channel Pruning while Training using Feature Relevance Scores for Convolutional Neural Networks [[paper](https://arxiv.org/abs/2002.09958)] [[code](https://github.com/purdue-nrl/Gradual-Channel-Pruning-using-FRS)]
-  [**2020 | arXiv**] Campfire: Compressible, Regularization-Free, Structured Sparse Training for Hardware Accelerators [[paper](https://arxiv.org/abs/2001.03253)] [[code](-)]
-  [**2019 | SysML**] Full deep neural network training on a pruned weight budget [[paper](https://arxiv.org/abs/1806.06949)]
-  [**2019 | SC**] PruneTrain: Fast Neural Network Training by Dynamic Sparse Model Reconfiguration [[paper](https://arxiv.org/abs/1901.09290)]
-  [**2018 | ICLR**] Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training [[paper](https://arxiv.org/abs/1712.01887)] 
-  [**2017 | ICML**] meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting [[paper](https://arxiv.org/abs/1706.06197)] 

### **Others**
-  [**2021 | ICLR**] Revisiting Locally Supervised Learning: an Alternative to End-to-end Training [[paper](https://arxiv.org/abs/2101.10832)] [[code](https://github.com/blackfeather-wang/InfoPro-Pytorch)]
-  [**2021 | ICLR**] Optimizer Fusion: Efficient Training with Better Locality and Parallelism [[paper]()] [[code](-)]
-  [**2021 | MLSys**] Wavelet: Efficient DNN Training with Tick-Tock Scheduling [[paper](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf)] 
-  [**2021 | arXiv**] AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning [[paper](https://arxiv.org/abs/2102.01386)]
-  [**2020 | NIPS**] Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures [[paper](https://arxiv.org/abs/2006.12878)] [[code](https://github.com/lightonai/dfa-scales-to-modern-deep-learning)]
-  [**2020 | NIPS**] TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning [[paper](https://arxiv.org/abs/2007.11622)]
-  [**2019 | ICML**] Training Neural Networks with Local Error Signals [[paper](https://arxiv.org/abs/1901.06656)] [[code](https://github.com/anokland/local-loss)]
-  [**2019 | ICML**] Error Feedback Fixes SignSGD and other Gradient Compression Schemes [[paper](https://arxiv.org/abs/1901.09847)] [[code](https://github.com/epfml/error-feedback-SGD)]
-  [**2019 | NIPS**] E2-Train: Training State-of-the-art CNNs with Over 80% Energy Savings [[paper](https://arxiv.org/abs/1910.13349)]
-  [**2019 | NIPS**] AutoAssist: A Framework to Accelerate Training of Deep Neural Networks [[paper](https://arxiv.org/abs/1905.03381v1)] [[code](https://github.com/zhangjiong724/autoassist-exp)]
-  [**2018 | ICML**] signSGD: Compressed Optimisation for Non-Convex Problems [[paper](https://arxiv.org/abs/1802.04434)] [[code](https://github.com/jxbz/signSGD)]
-  [**2017 | ICML**] Understanding Synthetic Gradients and Decoupled Neural Interfaces [[paper](https://arxiv.org/abs/1703.00522)] [[code](https://github.com/quangvu0702/Synthetic-Gradients)]
-  [**2017 | NIPS**] The Reversible Residual Network: Backpropagation Without Storing Activations [[paper](https://arxiv.org/abs/1707.04585)] [[code](https://github.com/renmengye/revnet-public)]
-  [**2016 | ICML**] Decoupled Neural Interfaces using Synthetic Gradients [[paper](https://arxiv.org/abs/1608.05343)] [[code](https://github.com/TheoryDev/Deep-neural-network-training-optimisation)]
-  [**2016 | arXiv**] Training Deep Nets with Sublinear Memory Cost [[paper](https://arxiv.org/abs/1604.06174)] [[code](https://github.com/dmlc/mxnet-memonger)]


## **Hardware**
### **Survey**
- [**2021 | OJSSC**] An Overview of Energy-Efficient Hardware Accelerators for On-Device Deep-Neural-Network Training
### **ASIC**
- [**2022 | ISCA**] Anticipating and Eliminating Redundant Computations in Accelerated Sparse Training
- [**2022 | TCAS-I**] TSUNAMI: Triple Sparsity-Aware Ultra Energy-Efficient Neural Network Training Accelerator With Multi-Modal Iterative Pruning
- [**2022 | HPCA**] FAST: DNN Training Under Variable Precision Block Floating Point with Stochastic Rounding
- [**2022 | JSSC**] A 7-nm Four-Core Mixed-Precision AI Chip With 26.2-TFLOPS Hybrid-FP8 Training, 104.9-TOPS INT4 Inference, and Workload-Aware Throttling
- [**2022 | ArXiv**] EcoFlow: Efficient Convolutional Dataflows for Low-Power Neural Network Accelerators
-  [**2021 | JSSC**] HNPU: An Adaptive DNN Training Processor Utilizing Stochastic Dynamic Fixed-Point and Active Bit-Precision Searching [[paper](https://ieeexplore.ieee.org/document/9383824?arnumber=9383824)]
-  [**2021 | JSSC**] GANPU: An Energy-Efficient Multi-DNN Training Processor for GANs With Speculative Dual-Sparsity Exploitation [[paper](https://ieeexplore.ieee.org/document/9410650/)]
-  [**2021 | JSSC**] A Neural Network Training Processor With 8-Bit Shared Exponent Bias Floating Point and Multiple-Way Fused Multiply-Add Trees
-  [**2021 | ISSCC**] A 7nm 4-Core AI Chip with 25.6TFLOPS Hybrid FP8 Training, 102.4TOPS INT4 Inference and Workload-Aware Throttling [[paper](https://ieeexplore.ieee.org/document/9365791)]
-  [**2021 | ISSCC**] A 40nm 4.81TFLOPS/W 8b Floating-Point Training Processor for Non-Sparse Neural Networks Using Shared Exponent Bias and 24-Way Fused Multiply-Add Tree [[paper](https://ieeexplore.ieee.org/document/9366031)]
-  [**2021 | ISCA**] RaPiD: AI Accelerator for Ultra-low Precision Training and Inference [[paper](https://ieeexplore.ieee.org/document/9366031)]
-  [**2021 | ISCA**] Cambricon-Q: A Hybrid Architecture for Efficient Training 
-  [**2021 | ISCA**] NASA: Accelerating Neural Network Design with a NAS Processor
-  [**2021 | ISCA**] Ten Lessons From Three Generations Shaped Google’s TPUv4i : Industrial Product
-  [**2021 | ISCAS**] A 3.6 TOPS/W Hybrid FP-FXP Deep Learning Processor with Outlier Compensation for Image-to-image Application
-  [**2021 | VLSI**] A 28nm 276.55TFLOPS/W Sparse Deep-Neural-Network Training Processor with Implicit Redundancy Speculation and Batch Normalization Reformulation
-  [**2021 | COOL**] An Energy-Efficient Deep Neural Network Training Processor with Bit-Slice-Level Reconfigurability and Sparsity Exploitation 
-  [**2021 | MICRO**] FPRaker: A Processing Element For Accelerating Neural Network Training
-  [**2021 | MICRO**] Equinox: Training (for Free) on a Custom Inference Accelerator
-  [**2021 | TC**] A Deep Neural Network Training Architecture with Inference-aware Heterogeneous Data-type
-  [**2021 |TCAS-I**] Memory Access Optimization for On-Chip Transfer Learning
-  [**2021 | TCAS-II**] A 64.1mW Accurate Real-time Visual Object Tracking Processor with Spatial Early Stopping on Siamese Network
-  [**2020 | IEEE Access**] Training Hardware for Binarized Convolutional Neural Network Based on CMOS Invertible Logic [[paper](https://ieeexplore.ieee.org/document/9217452)]
-  [**2020 | JSSC**] Evolver: A Deep Learning Processor With On-Device Quantization–Voltage–Frequency Tuning [[paper](https://ieeexplore.ieee.org/document/9209075)]
-  [**2020 | JSSC**] DF-LNPU: A Pipelined Direct Feedback Alignment-Based Deep Neural Network Learning Processor for Fast Online Learning [[paper](https://ieeexplore.ieee.org/document/9307218?arnumber=9307218)]
-  [**2020 | JSSC**] An Energy-Efficient Deep Convolutional Neural Network Training Accelerator for In Situ Personalization on Smart Devices [[paper](https://ieeexplore.ieee.org/document/9137200/)]
-  [**2020 | LSSC**] PNPU: An Energy-Efficient Deep-Neural-Network Learning Processor With Stochastic Coarse–Fine Level Weight Pruning and Adaptive Input/Output/Weight Zero Skipping [[paper]()]
-  [**2020 | TETC**] SPRING: A Sparsity-Aware Reduced-Precision Monolithic 3D CNN Accelerator Architecture for Training and Inference [[paper](https://ieeexplore.ieee.org/abstract/document/9120209)]
-  [**2020 | DAC**] SCA: A Secure CNN Accelerator for Both Training and Inference [[paper](https://dl.acm.org/doi/10.5555/3437539.3437670)]
-  [**2020 | DAC**] Prediction Confidence based Low Complexity Gradient Computation for Accelerating DNN Training [[paper](https://ieeexplore.ieee.org/document/9218650)]
-  [**2020 | DAC**] SparseTrain: Exploiting Dataflow Sparsity for Efficient Convolutional Neural Networks Training [[paper](https://dl.acm.org/doi/abs/10.5555/3437539.3437644)]
-  [**2020 | DAC**] A Pragmatic Approach to On-device Incremental Learning System with Selective Weight Updates [[paper](https://ieeexplore.ieee.org/document/9218507/)]
-  [**2020 | ISLPED**] SparTANN: sparse training accelerator for neural networks with threshold-based sparsification [[paper](https://dl.acm.org/doi/10.1145/3370748.3406554)]
-  [**2020 | ISSCC**] GANPU: A 135TFLOPS/W Multi-DNN Training Processor for GANs with Speculative Dual-Sparsity Exploitation [[paper](https://ieeexplore.ieee.org/document/9062989/)]
-  [**2020 | MICRO**] Procrustes: a Dataflow and Accelerator for Sparse Deep Neural Network Training [[paper](https://arxiv.org/abs/2009.10976)]
-  [**2020 | MICRO**] TensorDash: Exploiting Sparsity to Accelerate Deep Neural Network Training [[paper](https://ieeexplore.ieee.org/document/9251995)]
-  [**2020 | HPCA**] SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects for DNN Training [[paper](https://ieeexplore.ieee.org/document/9065523)]
-  [**2020 | VLSI**] A 3.0 TFLOPS 0.62V Scalable Processor Core for High Compute Utilization AI Training and Inference [[paper](https://ieeexplore.ieee.org/document/9162917/)]
-  [**2020 | VLSI**] A 146.52 TOPS/W Deep-Neural-Network Learning Processor with Stochastic Coarse-Fine Pruning and Adaptive Input/Output/Weight Skipping [[paper](https://ieeexplore.ieee.org/document/9162795/)]
-  [**2020 | arXiv**] FPRaker: A Processing Element For Accelerating Neural Network Training [[paper](https://arxiv.org/abs/2010.08065)]
-  [**2020 | ISCAS**] TaxoNN: A Light-Weight Accelerator for Deep Neural Network Training [[paper](https://ieeexplore.ieee.org/document/9181001)]
-  [**2019 | LSSC**] A 2.6 TOPS/W 16-bit Fixed-Point Convolutional Neural Network Learning Processor in 65nm CMOS [[paper]()]
-  [**2019 | LSSC**] An Energy-Efficient Deep Reinforcement Learning Accelerator With Transposable PE Array and Experience Compression [[paper](https://ieeexplore.ieee.org/document/8836619/)]
-  [**2019 | LSSC**] An Energy-Efficient Sparse Deep-Neural-Network Learning Accelerator with Fine-grained Mixed Precision of FP8-FP16 [[paper](https://ieeexplore.ieee.org/document/8813090/)]
-  [**2019 | TCAS-I**] A Low-Power Deep Neural Network Online Learning Processor for Real-Time Object Tracking Application [[paper](https://ieeexplore.ieee.org/document/8554277/)]
-  [**2019 | ASPDAC**]  TNPU: an efficient accelerator architecture for training convolutional neural networks[[paper](http://dl.acm.org/citation.cfm?doid=3287624.3287641)]
-  [**2019 | ASSCC**] A 2.25 TOPS/W Fully-Integrated Deep CNN Learning Processor with On-Chip Training [[paper](https://ieeexplore.ieee.org/document/9056967/)]
-  [**2019 | DAC**] Acceleration of DNN Backward Propagation by Selective Computation of Gradients [[paper](https://dl.acm.org/doi/10.1145/3316781.3317755)]
-  [**2019 | DAC**] An Optimized Design Technique of Low-bit Neural Network Training for Personalization on IoT Devices [[paper](https://dl.acm.org/doi/10.1145/3316781.3317769)]
-  [**2019 | ISSCC**] LNPU: A 25.3TFLOPS/W Sparse Deep-Neural-Network Learning Processor with Fine-Grained Mixed Precision of FP8-FP16 [[paper](https://ieeexplore.ieee.org/document/8662302/)]
-  [**2019 | SysML**] Mini-batch Serialization: CNN Training with Inter-layer Data Reuse [[paper](http://arxiv.org/abs/1810.00307)]
-  [**2019 | VLSI**] A 1.32 TOPS/W Energy Efficient Deep Neural Network Learning Processor with Direct Feedback Alignment based Heterogeneous Core Architecture [[paper]()]
-  [**2018 | LSSC**] A Scalable Multi-TeraOPS Core for AI Training and Inference [[paper](https://ieeexplore.ieee.org/document/8657745/)]
-  [**2018 | VLSI**] A Scalable Multi- TeraOPS Deep Learning Processor Core for AI Trainina and Inference [[paper](https://ieeexplore.ieee.org/document/8502276/)]
-  [**2017 | DAC**] Design of an Energy-Efficient Accelerator for Training of Convolutional Neural Networks using Frequency-Domain Computation [[paper](http://dl.acm.org/citation.cfm?doid=3061639.3062228)]
-  [**2017 | ISCA**] SCALEDEEP: A scalable compute architecture for learning and evaluating deep networks [[paper](https://dl.acm.org/doi/10.1145/3079856.3080244)]
-  [**2014 | MICRO**] DaDianNao: A Machine-Learning Supercomputer [[paper](http://ieeexplore.ieee.org/document/7011421/)]

### **FPGA**
- [**2022 | TNNLS**] ETA: An Efficient Training Accelerator for DNNs Based on Hardware-Algorithm Co-Optimization
-  [**2021 | ICS**] Enabling Energy-Efficient DNN Training on Hybrid GPU-FPGA Accelerators [[paper](https://dl.acm.org/doi/10.1145/3447818.3460371)]
-  [**2020 | TC**] A Neural Network-Based On-Device Learning Anomaly Detector for Edge Devices [[paper](https://ieeexplore.ieee.org/document/9000710?arnumber=9000710)]
-  [**2020 | ICCAD**] FPGA-based low-batch training accelerator for modern CNNs featuring high bandwidth memory [[paper]()]
-  [**2020 | IJCAI**] Efficient and Modularized Training on FPGA for Real-time Applications [[paper](https://www.ijcai.org/proceedings/2020/755)]
-  [**2020 | ISCAS**] Training Progressively Binarizing Deep Networks Using FPGAs [[paper](https://arxiv.org/abs/2001.02390)]
-  [**2020 | FPL**] Dynamically Growing Neural Network Architecture for Lifelong Deep Learning on the Edge [[paper](https://ieeexplore.ieee.org/document/9221575)]
-  [**2019 | FPT**] Training Deep Neural Networks in Low-Precision with High Accuracy Using FPGAs [[paper](https://ieeexplore.ieee.org/document/8977908/)]
-  [**2019 | NEWCAS**] Efficient Hardware Implementation of Incremental Learning and Inference on Chip [[paper](http://arxiv.org/abs/1911.07847)]
-  [**2019 | FPL**] FPGA-Based Training Accelerator Utilizing Sparseness of Convolutional Neural Network [[paper](https://ieeexplore.ieee.org/document/8892059/)]
-  [**2019 | FPL**] Automatic Compiler Based FPGA Accelerator for CNN Training [[paper](http://arxiv.org/abs/1908.06724)]
-  [**2019 | FCCM**] Towards Efficient Deep Neural Network Training by FPGA-Based Batch-Level Parallelism [[paper](https://ieeexplore.ieee.org/document/8735548)]
-  [**2019 | FPGA**] Compressed CNN Training with FPGA-based Accelerator [[paper](https://dl.acm.org/doi/10.1145/3289602.3293977)]
-  [**2018 | FCCM**] FPDeep: Acceleration and Load Balancing of CNN Training on FPGA Clusters [[paper](https://ieeexplore.ieee.org/document/8457636)]
-  [**2018 | FPL**] A Framework for Acceleration of CNN Training on Deeply-Pipelined FPGA Clusters with Work and Weight Load Balancing [[paper]()]
-  [*2018* | FPL**] ClosNets: Batchless DNN Training with On-Chip a Priori Sparse Neural Topologies [[paper](https://ieeexplore.ieee.org/document/8532585)]
-  [**2018 | ReConFig**] A Highly Parallel FPGA Implementation of Sparse Neural Network Training [[paper](https://arxiv.org/abs/1806.01087)]
-  [**2018 | ISLPED**] TrainWare: A Memory Optimized Weight Update Architecture for On-Device Convolutional Neural Network Training [[paper](https://dl.acm.org/doi/10.1145/3218603.3218625)]
-  [**2017 | FPT**] An FPGA-based processor for training convolutional neural networks [[paper](http://ieeexplore.ieee.org/document/8280142/)]
-  [**2017 | FPT**] FPGA-based training of convolutional neural networks with a reduced precision floating-point library [[paper](http://ieeexplore.ieee.org/document/8280150/)]
-  [**2016 | ASAP**] F-CNN: An FPGA-based framework for training Convolutional Neural Networks [[paper](http://ieeexplore.ieee.org/document/7760779/)]



### **PIM**
-  [**2021 | VLSI**] CHIMERA: A 0.92 TOPS, 2.2 TOPS/W Edge AI Accelerator with 2 MByte On-Chip Foundry Resistive RAM for Efficient Training and Inference
-  [**2021 | TC**] AILC: Accelerate On-chip Incremental Learning with Compute-in-Memory Technology [[paper](https://ieeexplore.ieee.org/document/9329153?arnumber=9329153)]
-  [**2021 | TC**] PANTHER: A Programmable Architecture for Neural Network Training Harnessing Energy-efficient ReRAM [[paper](http://arxiv.org/abs/1912.11516)]
-  [**2019 | TC**] A Scalable Near-Memory Architecture for Training Deep Neural Networks on Large In-Memory Datasets [[paper](https://ieeexplore.ieee.org/document/8502059/)]
-  [**2018 | TCAD**] DeepTrain: A Programmable Embedded Platform for Training Deep Neural Networks [[paper](https://ieeexplore.ieee.org/document/8418347/)]
-  [**2017 | HPCA**] PipeLayer: A Pipelined ReRAM-Based Accelerator for Deep Learning [[paper](http://ieeexplore.ieee.org/document/7920854/)]
<!-- 
### **SNN**
-  [**2019 | JSSC**] A 65-nm Neuromorphic Image Classification Processor With Energy-Efficient Training Through Direct Spike-Only Feedback [[paper](https://ieeexplore.ieee.org/document/8867974/)]


-  [** | **] [[paper]()][[code]()] -->

