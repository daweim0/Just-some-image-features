# Some overkill image features

This is some code for generating dense, scale and orientation invariant features for object and pose recognition. A more detailed readme is coming.

The following images show optical flow calculated by a nearest neighbor search on image feature maps, individually taking every pixel in the left image and looking for the most similar pixel in the right image. The visualization is direct output from the nearest neighbor search, **no post-processing or smoothing** has been done (since the point is to demonstrate how discriminative the features are, not fancy optical flow algorithms).

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_21.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_28.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_57.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_59.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_62.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_64.png)

### License

DA-RNN is released under the MIT License (refer to the LICENSE file for details).
