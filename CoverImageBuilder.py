import subprocess

import imageio

from ImageAugmentor import ImageAugmentor


class CoverImageBuilder:
    def __init__(self, image_path):
        self.__image_path_ = image_path
        self.__output_path_ = self.getOutputPath(self.__image_path_)
        self.__original_pic_ = imageio.imread(self.__image_path_)
        self.__augmentor_ = ImageAugmentor(self.__original_pic_)
        self.__img_ = self.__original_pic_

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
        augmented_img = self.__augmentor_.getAugmentedImage()
        imageio.imwrite(self.__output_path_, augmented_img)
        self.exif_remover(self.__output_path_)


if __name__ == "__main__":
    ie = CoverImageBuilder("data/test_image.jpg")
    ie.buildCoverImage()
