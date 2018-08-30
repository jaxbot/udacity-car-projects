# Semantic Segmentation
In this project, a modified VGG16 model is used for segmenting pixels in an image that correspond to a road. This is done by adding skip layers and training on the Kitti Road dataset.

## Run
Run the following command to run the project:
```
python main.py
```

## Results

<img src="examples/um_000013_normalization.png" alt="A dashcam image from a car where the road has been segmented in green pixels.">

## Other notes

Batch normalization is applied between each skip connection. This causes a reduction in loss from 0.332 to 0.012. Without normalization, the above scene looks like this:

<img src="examples/um_000013_no_normalization.png" alt="A dashcam image from a car where the road has been segmented in green pixels, but with high loss and several non-road pixels also segmented green.">
