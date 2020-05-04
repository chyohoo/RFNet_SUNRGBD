import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each classes-----------')
        print("wall          : %.6f" % (Acc[0] * 100.0), "%\t")
        print("floor         : %.6f" % (Acc[1] * 100.0), "%\t")
        print("cabinet       : %.6f" % (Acc[2] * 100.0), "%\t")
        print("bed           : %.6f" % (Acc[3] * 100.0), "%\t")
        print("chair         : %.6f" % (Acc[4] * 100.0), "%\t")
        print("sofa          : %.6f" % (Acc[5] * 100.0), "%\t")
        print("table         : %.6f" % (Acc[6] * 100.0), "%\t")
        print("door          : %.6f" % (Acc[7] * 100.0), "%\t")
        print("window        : %.6f" % (Acc[8] * 100.0), "%\t")
        print("bookshelf     : %.6f" % (Acc[9] * 100.0), "%\t")
        print("picture       : %.6f" % (Acc[10] * 100.0), "%\t")
        print("counter       : %.6f" % (Acc[11] * 100.0), "%\t")
        print("blinds        : %.6f" % (Acc[12] * 100.0), "%\t")
        print("desk          : %.6f" % (Acc[13] * 100.0), "%\t")
        print("shelves       : %.6f" % (Acc[14] * 100.0), "%\t")
        print("curtain       : %.6f" % (Acc[15] * 100.0), "%\t")
        print("dresser       : %.6f" % (Acc[16] * 100.0), "%\t")
        print("pillow        : %.6f" % (Acc[17] * 100.0), "%\t")
        print("mirror        : %.6f" % (Acc[18] * 100.0), "%\t")
        print("floor_mat     : %.6f" % (Acc[19] * 100.0), "%\t")
        print("clothes       : %.6f" % (Acc[20] * 100.0), "%\t")
        print("ceiling       : %.6f" % (Acc[21] * 100.0), "%\t")
        print("books         : %.6f" % (Acc[22] * 100.0), "%\t")
        print("fridge        : %.6f" % (Acc[23] * 100.0), "%\t")
        print("tv            : %.6f" % (Acc[24] * 100.0), "%\t")
        print("paper         : %.6f" % (Acc[25] * 100.0), "%\t")
        print("towel         : %.6f" % (Acc[26] * 100.0), "%\t")
        print("shower_curtain: %.6f" % (Acc[27] * 100.0), "%\t")
        print("box           : %.6f" % (Acc[28] * 100.0), "%\t")
        print("whiteboard    : %.6f" % (Acc[29] * 100.0), "%\t")
        print("person        : %.6f" % (Acc[30] * 100.0), "%\t")
        print("night_stand   : %.6f" % (Acc[31] * 100.0), "%\t")
        print("toilet        : %.6f" % (Acc[32] * 100.0), "%\t")
        print("sink          : %.6f" % (Acc[33] * 100.0), "%\t")
        print("lamp          : %.6f" % (Acc[34] * 100.0), "%\t")
        print("bathhub       : %.6f" % (Acc[35] * 100.0), "%\t")
        print("bag           : %.6f" % (Acc[36] * 100.0), "%\t")
        # if self.num_class == 20:
        #     print("small obstacles: %.6f" % (Acc[19] * 100.0), "%\t")
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        print("wall          : %.6f" % (MIoU[0] * 100.0), "%\t")
        print("floor         : %.6f" % (MIoU[1] * 100.0), "%\t")
        print("cabinet       : %.6f" % (MIoU[2] * 100.0), "%\t")
        print("bed           : %.6f" % (MIoU[3] * 100.0), "%\t")
        print("chair         : %.6f" % (MIoU[4] * 100.0), "%\t")
        print("sofa          : %.6f" % (MIoU[5] * 100.0), "%\t")
        print("table         : %.6f" % (MIoU[6] * 100.0), "%\t")
        print("door          : %.6f" % (MIoU[7] * 100.0), "%\t")
        print("window        : %.6f" % (MIoU[8] * 100.0), "%\t")
        print("bookshelf     : %.6f" % (MIoU[9] * 100.0), "%\t")
        print("picture       : %.6f" % (MIoU[10] * 100.0), "%\t")
        print("counter       : %.6f" % (MIoU[11] * 100.0), "%\t")
        print("blinds        : %.6f" % (MIoU[12] * 100.0), "%\t")
        print("desk          : %.6f" % (MIoU[13] * 100.0), "%\t")
        print("shelves       : %.6f" % (MIoU[14] * 100.0), "%\t")
        print("curtain       : %.6f" % (MIoU[15] * 100.0), "%\t")
        print("dresser       : %.6f" % (MIoU[16] * 100.0), "%\t")
        print("pillow        : %.6f" % (MIoU[17] * 100.0), "%\t")
        print("mirror        : %.6f" % (MIoU[18] * 100.0), "%\t")
        print("floor_mat     : %.6f" % (MIoU[19] * 100.0), "%\t")
        print("clothes       : %.6f" % (MIoU[20] * 100.0), "%\t")
        print("ceiling       : %.6f" % (MIoU[21] * 100.0), "%\t")
        print("books         : %.6f" % (MIoU[22] * 100.0), "%\t")
        print("fridge        : %.6f" % (MIoU[23] * 100.0), "%\t")
        print("tv            : %.6f" % (MIoU[24] * 100.0), "%\t")
        print("paper         : %.6f" % (MIoU[25] * 100.0), "%\t")
        print("towel         : %.6f" % (MIoU[26] * 100.0), "%\t")
        print("shower_curtain: %.6f" % (MIoU[27] * 100.0), "%\t")
        print("box           : %.6f" % (MIoU[28] * 100.0), "%\t")
        print("whiteboard    : %.6f" % (MIoU[29] * 100.0), "%\t")
        print("person        : %.6f" % (MIoU[30] * 100.0), "%\t")
        print("night_stand   : %.6f" % (MIoU[31] * 100.0), "%\t")
        print("toilet        : %.6f" % (MIoU[32] * 100.0), "%\t")
        print("sink          : %.6f" % (MIoU[33] * 100.0), "%\t")
        print("lamp          : %.6f" % (MIoU[34] * 100.0), "%\t")
        print("bathhub       : %.6f" % (MIoU[35] * 100.0), "%\t")
        print("bag           : %.6f" % (MIoU[36] * 100.0), "%\t")
        # if self.num_class == 20:
        #     print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




