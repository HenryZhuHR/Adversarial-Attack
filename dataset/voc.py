import os
import warnings


class VOC():
    def __init__(self,
                 vocdevkit_dir,  # Path to folder 'VOCdevkit'
                 ) -> None:
        """
        Init
        ---
        Parameter
        --
        - `voc_devkit_dir` Path to folder 'VOCdevkit'
        """
        super().__init__()
        self.__check_dir(vocdevkit_dir)

        self.dir_ImageSets = os.path.join(
            vocdevkit_dir, 'VOC2012', 'ImageSets')
        self.dir_JPEGImages = os.path.join(
            vocdevkit_dir, 'VOC2012', 'JPEGImages')
        self.dir_Annotations = os.path.join(
            vocdevkit_dir, 'VOC2012', 'Annotations')

    def convert_for_classify(self, save_dir: str):
        dir_ImageSets_Main = os.path.join(self.dir_ImageSets, 'Main')
        txt_files = {
            'train': 'train.txt',
            'valid': 'val.txt',
            'test': 'test.txt',
        }
        for type, txt_file in txt_files.items():
            sub_save_dir = os.path.join(save_dir, type)
            file_path = os.path.join(dir_ImageSets_Main, txt_file)

            if os.path.exists(file_path):
                os.makedirs(sub_save_dir,exist_ok=True)                
            else:
                warnings.warn('File \033[0;33m%s\033[0m not found, \033[0;33m%s\033[0m will not be create in \033[0;33m%s\033[0m' % (
                    file_path, type,save_dir))
                continue

            with open(file_path,'r',encoding='utf-8') as f:
                contents=[]
                for line_content in f.readlines():
                    file_id=line_content.rstrip('\n')
                    jpg_file=os.path.join(self.dir_JPEGImages,'%s.jpg'%file_id)
                    xml_file=os.path.join(self.dir_Annotations,'%s.xml'%file_id)
                    # TODO: find class from xml
                    contents.append([jpg_file,xml_file])
                print(contents)

    def __copy_file_by_txt(self, txtfile: str):
        pass

    def __check_dir(self, vocdevkit_dir: str):
        """
        check folder integrity
        """
        vocdevkit_dir_list = os.listdir(vocdevkit_dir)
        if 'VOC2012' not in vocdevkit_dir_list:
            raise FileNotFoundError('no such file \033[0;31m%s\033[0m in folder: \033[0;31m%s\033[0m' % (
                'VOC2012', vocdevkit_dir))

        voc_dir_list = os.listdir(os.path.join(vocdevkit_dir, 'VOC2012'))
        if 'Annotations' not in voc_dir_list:
            raise FileNotFoundError('make sure folder \033[0;31m%s\033[0m in \033[0;31m%s\033[0m' % (
                'Annotations', os.path.join(vocdevkit_dir, 'VOC2012')))
        if 'ImageSets' not in voc_dir_list:
            raise FileNotFoundError('make sure folder \033[0;31m%s\033[0m in \033[0;31m%s\033[0m' % (
                'ImageSets', os.path.join(vocdevkit_dir, 'VOC2012')))
        if 'JPEGImages' not in voc_dir_list:
            raise FileNotFoundError('make sure folder \033[0;31m%s\033[0m in \033[0;31m%s\033[0m' % (
                'JPEGImages', os.path.join(vocdevkit_dir, 'VOC2012')))
