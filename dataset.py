from dataset import Caltech101

# voc_Datset=VOC('~/Dataset/VOCdevkit')
voc_Datset=Caltech101('C:/Users/Henryzhu/Dataset/Caltech101')
voc_Datset.convert_for_classify('data')