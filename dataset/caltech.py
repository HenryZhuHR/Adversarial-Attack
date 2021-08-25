import os


class Caltech101():
    """
        url: http://www.vision.caltech.edu/Image_Datasets/Caltech101
        
        If you are using the Caltech 101 dataset for testing your recognition algorithm you should try and make your results comparable to the results of others. We suggest training and testing on fixed number of pictures and repeating the experiment with different random selections of pictures in order to obtain error bars. Popular number of training images: 1, 3, 5, 10, 15, 20, 30. Popular numbers of testing images: 20, 30. See also the discussion below.
        When you report your results please keep track of which images you used and which were misclassified. We will soon publish a more detailed experimental protocol that allows you to report those details. See the Discussion section for more details.
    """

    def __init__(
        self,
        dataset_dir:str,  # .../101_ObjectCategories
    ) -> None:
        if os.path.exists(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            raise FileNotFoundError(
                'No such file\033[0;31m%s\033[0m' % dataset_dir)
    
    def convert_for_classify(self, save_dir: str):
        pass 
        