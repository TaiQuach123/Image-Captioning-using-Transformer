# Image Captioning using Transformer (Pytorch)
This repository is a simple baseline for Image Captioning using Transformer with Pytorch. 
## Outline
1. Prepare and preprocess data
2. Building Modules for Image Captioning model
3. Usage

## Prepare and preprocess data
In this project, I use the Flickr8k Dataset, which contains about 8000 images, each image has 5 captions describing it. Here, I have already written a file that automatically downloads the dataset.
Use the **prepare_data.py** to create a data folder, all the data required will be downloaded into this folder.
```
python prepare_data.py -p *your_directory_to_store_data*
```
The **dataset.py** contains a MyDataset class and functions to preprocess the images and captions belonging to images. We will use the functions and class in this file to create a torch.utils.data.Dataset class and preprocess the data later.

## Building Modules for Image Captioning model
The baseline for Image Captioning in this repository is very simple. First, use a CNN backbone to extract features, then feed these features into a Transformer (Encoder and Decoder). 
I use the EfficientNetB1 backbone to extract the features of images, then feed them into a TransformerEncoder. The output of the encoder is then fed into the TransformerDecoder using cross-attention. I also use a custom causal mask to prevent the contributions of padding tokens (which have no meaningful information). My model has only 1 layer of TransformerEncoderLayer and 1 layer of TransformerDecoderLayer.
See the **model.py** for more details.

## Usage
First, clone this repository
```
git clone https://github.com/TaiQuach123/Image-Captioning-using-Transformer.git
```
For training, run the **train.py** as following
```
python train.py --path *your_images_directory_path* --captions *your_captions_path*
```
*your_images_directory_path* is the path to the directory that stores your images. Similarly, *your_captions_path* is the text file containing images and the captions for each of them.


For inference, run the **inference.py** as following
```
python inference.py --pretrained_weights *path_to_pretrained_weights* -g boolean -i *path_to_image*
```
where g is a boolean value, if True, it will generate captions for some random images in your test dataset. Flag -i indicates the image path that you want to generate the caption.
