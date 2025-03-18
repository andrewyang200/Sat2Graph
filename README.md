# Sat2Graph

## Effect of Dataloader Bug:

When testing to see the effect of applying the rotate function on self.samplepoints versus not applying it (default), we found that there was no comparable advantage in the loss curve when it comes to the validation dataset. However, on the training set, the default data loader performs better with a lower loss at respective training steps. Therefore, it is recommended that we leave the data loader in its default configuration. One potential reason for the improvement on the training set could be that the data loader extracts batches based on the nodes contained in the sample points dictionary. If we rotated the sample points along with the input image and the ground truth, then the images would always be inline with each other. This won’t have much of an effect on closely packed metropolitan areas with a lot of roads. However, in more rural areas or areas with less roads, this prevents the model from ever seeing images that contain mainly empty space. In essence, by not applying the rotate function on the self.samplepoints, it augments the data by introducing a greater variety of images for the model. Instead of the presumed bug, this could rather be seen as a feature. The result can be seen in both Tensorflow and Pytorch models. For Pytorch, reference runs 3 (default) vs run 8 (applies rotate function). 

## Effect of Batchnorm in training mode:

In the original tensorflow model, we noticed that few specific layers containing the batchnorm would still be configured to training mode when applied in the validation set. In standard practice, all layers of the model should be set to eval mode. We tested to see if this produced a significant difference and discovered that there the difference was not significant enough to leave some batch norms in training mode while the rest of the model is set to eval mode. For reference, see run 7 and run 5. 

## Effect of Initializer:

Across the board, we deemed that it was beneficial to apply a weight initializer. Doing so resulted in better loss curves on the training and validation sets. While the difference isn’t drastic, it was enough of an improvement to justify the use of it. For reference, see run 3 (with initializer) vs run 2 (without initializer) or run 9 (with initializer) vs run 7 (without initializer). Note for run 7 had a lower loss on the training set but run 9 had a lower loss on the validation set. 

## Effect of Regularizer:

Across the board, we determined that the use of a regularizer actually yielded worse results. For reference, see run 4 (with regularizer) vs run 3 (without regularizer) or run 6 (with) vs run 7 (without).

## Pytorch default transpose convolution layer vs Tensorflow convolution layer:

We determined that there was not a substantial difference between using the default TransConv that pytorch provides and the one that we manually adjusted to model after tensorflow’s behavior. For reference, see run 5 (default) vs run 2 (models after tensorflow)

## Use of a scheduler:

We also tested for the need of applying a 0.5 learning rate decay after every 50,000 steps. It was determined that the decay yielded significantly better results on the validation set. For reference, see run 10 (no lr decay) vs run 7 (with lr decay).

## Final verdict:

When comparing all the different runs, we determined run 9 to have the best performance and the most optimized in terms of Pytorch’s default implementation. The run closely models after run 3, which was one of the base cases used for testing different components of the model.


