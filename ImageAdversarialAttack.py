import subprocess
import settings as st
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


class ImageEditor:
    def __init__(self, image_path):
        self.__image_path_ = image_path
        self.__output_path_ = self.getOutputPath(self.__image_path_)
        self.__original_pic_ = imageio.imread(self.__image_path_)
        self.__augmented_pic_ = None
        self.__img_ = self.__original_pic_
        self.__seq_ = iaa.Sequential(
            [
                iaa.Fliplr(st.P_HORIZONTAL_FLIP),
                iaa.Flipud(st.P_VERTICAL_FLIP),  # vertically flip
                # crop images
                sometimes(iaa.CropAndPad(
                    percent=(st.MIN_CROP, st.MAX_CROP),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    # scale images
                    scale={"x": (st.MIN_SCALE, st.MAX_SCALE), "y": (st.MIN_SCALE, st.MAX_SCALE)},
                    # translate
                    translate_percent={"x": (st.MIN_TRANSLATE, st.MAX_TRANSLATE), "y": (st.MIN_TRANSLATE, st.MAX_TRANSLATE)},
                    rotate=(st.MIN_ROTATION, st.MAX_ROTATION),  # rotate
                    shear=(st.MIN_SHEAR, st.MAX_SHEAR),  # shear
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((st.MIN_BLUR_SIGMA, st.MAX_BLUR_SIGMA)),  # blur images
                                   iaa.AverageBlur(k=(st.MIN_LOCAL_BLUR_SIZE, st.MAX_LOCAL_BLUR_SIZE)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(st.MIN_SHARPEN_ALPHA, st.MAX_SHARPEN_ALPHA), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(st.MIN_EMBOSS_ALPHA, st.MAX_EMBOSS_ALPHA), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.LinearContrast((0.5, 2.0))
                                   )
                               ]),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    @staticmethod
    def exif_remover(image_path: str):
        if image_path is None:
            raise TypeError
        output = subprocess.run(['exiftool', '-all=', image_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if output.stderr is not None:
            raise OSError
        else:
            return output.stdout.decode('utf-8')

    @staticmethod
    def getOutputPath(image_path: str):
        if image_path is None:
            raise TypeError
        index_last_point = image_path.rindex('.')
        return image_path[:index_last_point] + "_edited.png"

    def buildCoverImage(self):
        self.__augmented_pic_ = self.__seq_.augment_image(self.__original_pic_)
        imageio.imwrite(self.__output_path_, self.__augmented_pic_)
        self.exif_remover(self.__output_path_)


if __name__ == "__main__":
    ie = ImageEditor("data/a.jpg")
    ie.buildCoverImage()
