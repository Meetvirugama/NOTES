# 📚 Chapter 14: Deep Computer Vision Using Convolutional Neural Networks
### Complete Study Notes — Visual & Exam-Oriented

> **All 52 pages analyzed. All concepts covered. 36 diagrams mapped. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Diagrams)

> All visuals are in the [`Visuals/`](Visuals/) folder and are embedded inside their respective modules.
> Re-generate anytime: `python3 generate_visuals.py`

| Fig # | Diagram / Graph Title | Module | File |
|---|---|---|---|
| 01 | Human Eye to Visual Cortex Pipeline | 1 | [01_visual_cortex_pipeline.png](Visuals/01_visual_cortex_pipeline.png) |
| 02 | Edge and Shape Detection Hierarchy | 1 | [02_shape_hierarchy.png](Visuals/02_shape_hierarchy.png) |
| 03 | Biological Inspiration Behind CNNs | 1 | [03_biological_inspiration.png](Visuals/03_biological_inspiration.png) |
| 04 | Sliding Kernel Operation | 1 | [04_sliding_kernel.png](Visuals/04_sliding_kernel.png) |
| 05 | Input Image ──→ Convolution ──→ Feature Map | 1 | [05_conv_flow.png](Visuals/05_conv_flow.png) |
| 06 | Mathematical Convolution Visualization | 1 | [06_math_convolution.png](Visuals/06_math_convolution.png) |
| 07 | Edge Detection Filters Output | 1 | [07_edge_filters.png](Visuals/07_edge_filters.png) |
| 08 | Blur, Sharpen, and Sobel Filter Grids | 1 | [08_filter_effects.png](Visuals/08_filter_effects.png) |
| 09 | Feature Resolution Across Layer Depths | 1 | [09_feature_maps_depth.png](Visuals/09_feature_maps_depth.png) |
| 10 | Max Pooling vs. Average Pooling | 2 | [10_max_avg_pooling.png](Visuals/10_max_avg_pooling.png) |
| 11 | Dimensionality Reduction Comparison | 2 | [11_pooling_reduction.png](Visuals/11_pooling_reduction.png) |
| 12 | LeNet-5 Complete Architecture Flow | 2 | [12_lenet5_architecture.png](Visuals/12_lenet5_architecture.png) |
| 13 | AlexNet Dual-GPU Split Architecture | 2 | [13_alexnet_architecture.png](Visuals/13_alexnet_architecture.png) |
| 14 | ReLU vs. Tanh Activation Function | 2 | [14_relu_activation.png](Visuals/14_relu_activation.png) |
| 15 | GoogLeNet Inception Module | 3 | [15_inception_block.png](Visuals/15_inception_block.png) |
| 16 | Multi-Scale Feature Extraction Parallelism | 3 | [16_multiscale_extraction.png](Visuals/16_multiscale_extraction.png) |
| 17 | VGG-16 Deep Network Channel Progression | 2 | [17_vgg16_architecture.png](Visuals/17_vgg16_architecture.png) |
| 18 | Stacked 3x3 Convs Receptive Field Equivalent to 5x5 Conv | 2 | [18_stacked_convolutions.png](Visuals/18_stacked_convolutions.png) |
| 19 | ResNet Basic Skip Connection Unit | 3 | [19_resnet_block.png](Visuals/19_resnet_block.png) |
| 20 | Gradient Bypass Path During Backpropagation | 3 | [20_resnet_gradient_flow.png](Visuals/20_resnet_gradient_flow.png) |
| 21 | Depthwise Separable Convolution Components | 3 | [21_depthwise_separable_conv.png](Visuals/21_depthwise_separable_conv.png) |
| 22 | Xception Flow Structure Blocks | 3 | [22_xception_architecture.png](Visuals/22_xception_architecture.png) |
| 23 | Squeeze-and-Excitation Recalibration Block | 3 | [23_squeeze_excitation.png](Visuals/23_squeeze_excitation.png) |
| 24 | Squeeze-and-Excitation Channel Attention Reweighting | 3 | [24_channel_attention.png](Visuals/24_channel_attention.png) |
| 25 | Pretrained Model Transfer Learning Workflow | 4 | [25_transfer_learning_workflow.png](Visuals/25_transfer_learning_workflow.png) |
| 26 | Layer Freezing Timeline vs Fine-Tuning Steps | 4 | [26_frozen_vs_trainable.png](Visuals/26_frozen_vs_trainable.png) |
| 27 | Classification vs. Localization Comparison | 5 | [27_classification_vs_localization.png](Visuals/27_classification_vs_localization.png) |
| 28 | Dual-Head Bounding Box Prediction Network | 5 | [28_bounding_box_prediction.png](Visuals/28_bounding_box_prediction.png) |
| 29 | Faster R-CNN Detection Pipeline Stages | 5 | [29_object_detection_pipeline.png](Visuals/29_object_detection_pipeline.png) |
| 30 | Multi-Object Detection with Class Labels | 5 | [30_multiobject_detection.png](Visuals/30_multiobject_detection.png) |
| 31 | FCN Dense Layer to 1x1 Convolution Conversion | 5 | [31_fcn_conversion.png](Visuals/31_fcn_conversion.png) |
| 32 | Symmetric U-Net Encoder-Decoder Flow | 5 | [32_encoder_decoder_flow.png](Visuals/32_encoder_decoder_flow.png) |
| 33 | YOLO Grid Cell Mapping | 5 | [33_yolo_grid.png](Visuals/33_yolo_grid.png) |
| 34 | YOLO Real-Time Single Forward Pass Workflow | 5 | [34_yolo_workflow.png](Visuals/34_yolo_workflow.png) |
| 35 | Input Image vs. Semantic Segmentation Mask | 5 | [35_original_vs_segmentation.png](Visuals/35_original_vs_segmentation.png) |
| 36 | Pixel-Wise Category Classification Map | 5 | [36_pixel_segmentation.png](Visuals/36_pixel_segmentation.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | **Visual Cortex & Conv Foundations**: Visual cortex discovery, Convolution layers mechanics,VALID vs SAME padding, stride reduction, Sobel/Blur/Sharpen filters, and feature maps depth. | [01_The_Architecture_of_the_Visual_Cortex_and_Convolutional_Layers.md](Detailed_Notes/01_The_Architecture_of_the_Visual_Cortex_and_Convolutional_Layers.md) | pp. 445–458 |
| 02 | **Pooling & Classic Architectures**: Max pooling, Average pooling, Global Average Pooling, LeNet-5, AlexNet, and VGG-16 stacked convolutions. | [02_Pooling_Layers_and_Classic_CNN_Architectures.md](Detailed_Notes/02_Pooling_Layers_and_Classic_CNN_Architectures.md) | pp. 458–465 |
| 03 | **Advanced CNN Architectures**: Inception modules (GoogLeNet), residual bypass blocks (ResNet), depthwise separable convolutions (Xception), and channel attention mechanisms (SENet). | [03_Advanced_CNN_Architectures.md](Detailed_Notes/03_Advanced_CNN_Architectures.md) | pp. 465–479 |
| 04 | **Pretrained Models & Transfer Learning**: Pretrained Keras models workflow, layer freezing warmup strategy, and fine-tuning rules. | [04_Pretrained_Models_and_Transfer_Learning.md](Detailed_Notes/04_Pretrained_Models_and_Transfer_Learning.md) | pp. 479–483 |
| 05 | **Deep Computer Vision Tasks**: Classification/Localization, Object Detection (IoU, NMS, Faster R-CNN), FCN conversion, YOLO grid cells, and Semantic Segmentation (U-Net, skip connections). | [05_Deep_Computer_Vision_Tasks.md](Detailed_Notes/05_Deep_Computer_Vision_Tasks.md) | pp. 483–496 |

---

## ⚡ One-Page Chapter Summary

### The History of Computer Vision Architectures

```
1959: Hubel & Wiesel Visual Cortex Receptive Fields ────→ Local receptive zones, orientation columns.
1998: LeNet-5 (Lecun et al.) ─────────────────────────→ First digit recognition CNN with backprop.
2012: AlexNet (Krizhevsky et al.) ────────────────────→ ReLU, Dropout, dual GPU split, ImageNet winner.
2014: VGG-16 (Simonyan & Zisserman) ──────────────────→ Stacked uniform 3x3 convolutions.
2014: GoogLeNet (Szegedy et al.) ─────────────────────→ Inception block parallel paths & 1x1 convs.
2015: ResNet (He et al.) ─────────────────────────────→ Skip connections bypass layers (gradient highways).
2016: Xception (Chollet) ─────────────────────────────→ Depthwise separable convolutions (9x savings).
2017: SENet (Hu et al.) ──────────────────────────────→ Squeeze-and-excitation channel attention.
```

### Core Code Snippet (ResNet block & localization Keras pipeline)

```python
import tensorflow as tf
from tensorflow import keras

# 1. Residual Unit (projection shortcut if shape changes)
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation(z + skip_z)

# 2. Multi-Head Pipeline
inputs = keras.layers.Input(shape=[224, 224, 3])
base = keras.applications.Xception(input_tensor=inputs, include_top=False, weights="imagenet")
avg = keras.layers.GlobalAveragePooling2D()(base.output)

# Heads
cls_head = keras.layers.Dense(10, activation="softmax", name="class_out")(avg)
box_head = keras.layers.Dense(4, name="box_out")(avg) # outputs [x, y, w, h]

model = keras.models.Model(inputs=inputs, outputs=[cls_head, box_head])
```

---

## 🏆 Top 5 Things to Remember

1.  **SAME vs. VALID Padding**: VALID padding ignores boundary pixels if they do not fit the kernel size, causing shape shrinkage. SAME padding pads borders with zeros so that the output size equals the input size divided by stride (rounded up).
2.  **Stacked Convolutions**: Stacking two $3 \times 3$ convolutions has the same spatial receptive field as a single $5 \times 5$ convolution, but uses **28% fewer parameters** and introduces more non-linear activation boundaries.
3.  **ResNet Skip Connections**: Residual skip connections add the input directly to the output of convolutional blocks ($F(x) + x$). During backpropagation, this creates a **gradient highway** that prevents vanishing gradients in deep networks.
4.  **Depthwise Separable Convolutions**: Splitting convolutions into spatial filtering (depthwise) and channel mixing (1x1 pointwise) reduces computational requirements by a factor of 9x for $3 \times 3$ filters.
5.  **Pretrained Layer Freezing**: When utilizing transfer learning, lock/freeze the weights of the pretrained base before warming up a newly initialized classifier head to prevent **catastrophic forgetting**.

---

*Notes created from 52 textbook pages covering pp. 445–496 of Hands-On ML with Scikit-Learn, Keras, and TensorFlow (2nd edition) by Aurélien Géron.*
