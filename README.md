# Advanced DCGAN for Rectangles
An in-depth look at DCGANs, the motivation behind them, and a highly detailed overview of the optimization techniques and tricks necessary to stabilize training between a Generator/Discriminator pair.

## [See Notebook with Exaplantions](https://nbviewer.jupyter.org/github/IliaZenkov/Advanced-DCGAN/blob/main/DCGAN-Advanced.ipynb)

I build a GAN trained on and capable of generating CelebA in original aspect ratio (CelebA aligned: 218x178; 5:4).
It was notably harder to get good-looking rectangular images compared to square images; but tuning the model to accomodate rectangular images is simply a matter of playing with kernel, stride, and padding sizes. 

## 59 Epochs at 157x128
<img src="generated_images/007459.jpg">
