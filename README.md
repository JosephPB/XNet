# XNet

XNet is a Convolutional Neural Network designed for the segmentation
of X-Ray images into bone, soft tissue and open beam
regions. Specifically, it performs well on small datasets with the aim
to minimise the number of false positives in the soft tissue class.

## Architecture

![](architecture.jpg)

* Built on a typical encoder-decoder architecture as
inspired by [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/).

* Additional feature extraction stage, with weight sharing across some
  layers.

* Fine and coarse grained feature preservation through concatenation
  of layers.

* L2 regularisation at each of the convolutional layers, to decrease overfitting. 

The architecture is described in the ```XNet.py``` file.

## Output

XNet outputs a mask of equal size to the input images.

![](predictions.png)

## Training

XNet is trained on a small dataset which has undergone
augmention. Examples of this augmentation step can be found in the
```Augmentation``` notebook.

An example of network training is seen in the ```Training``` notebook.

## Benchmarking

XNet was benchmarked against two of the leading segmentation networks:

* Simplified [SegNet](https://arxiv.org/abs/1511.00561) (found in the
  ```SimpleSegNet.py``` file)

* [UNet](https://arxiv.org/abs/1505.04597) (found in the ```UNet.py```
  file)

## More information

For more information and context see the conference poster
```Poster.pdf```.

