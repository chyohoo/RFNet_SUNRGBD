import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='SUNRGBD'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))  # change for val
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'SUNRGBD':
        n_classes = 37
        label_colours = get_SUNRGBD_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # change for val
    # rgb = torch.ByteTensor(3, label_mask.shape[0], label_mask.shape[1]).fill_(0)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # r = torch.from_numpy(r)
    # g = torch.from_numpy(g)
    # b = torch.from_numpy(b)

    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_SUNRGBD_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask



def get_SUNRGBD_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0],
                 # 0=background
                 [148, 65, 137],
                 [202, 179, 158], [155, 99, 235], [76, 152, 126], [84, 62, 35],
                 [44, 80, 130], [31, 184, 157], [101, 144, 77],
                 [23, 197, 62], [141, 168, 145], [142, 151, 136],
                 [115, 201, 77], [100, 216, 255], [57, 156, 36],
                 [88, 108, 129], [105, 129, 112], [42, 137, 126],
                 [155, 108, 249],  [81, 91, 87],
                 [100, 124, 51], [73, 131, 121], [157, 210, 220],
                 [134, 181, 60], [221, 223, 147], [123, 108, 131],
                 [161, 66, 179], [163, 221, 160], [31, 146, 98],
                 [99, 121, 30], [49, 89, 240], [116, 108, 9],
                 [161, 176, 169], [80, 29, 135], [177, 105, 197],
                 [139, 110, 246]])


def colormap_bdd(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([148, 65, 137])
    cmap[1,:] = np.array([202, 179, 158])
    cmap[2,:] = np.array([ 155, 99, 235])
    cmap[3,:] = np.array([155, 99, 235])
    cmap[4,:] = np.array([76, 152, 126])
    cmap[5,:] = np.array([84, 62, 35])
    cmap[6,:] = np.array([44, 80, 130])
    cmap[7,:] = np.array([31, 184, 157])
    cmap[8,:] = np.array([101, 144, 77])
    cmap[9,:] = np.array([23, 197, 62])
    cmap[10,:]= np.array([141, 168, 145])
    cmap[11,:]= np.array([142, 151, 136])
    cmap[12,:]= np.array([115, 201, 77])
    cmap[13,:]= np.array([100, 216, 255])
    cmap[14,:]= np.array([57, 156, 36])
    cmap[15,:]= np.array([88, 108, 129])
    cmap[16,:]= np.array([105, 129, 112])
    cmap[17,:]= np.array([42, 137, 126])

    cmap[18,:]= np.array([155, 108, 249])
    cmap[19,:]= np.array([81, 91, 87]) #多加了一类small obstacle
    cmap[20,:] = np.array([100, 124, 51])

    cmap[21,:] = np.array([73, 131, 121])
    cmap[22,:] = np.array([157, 210, 220])
    cmap[23,:] = np.array([134, 181, 60])

    cmap[24,:] = np.array([221, 223, 147])
    cmap[25,:] = np.array([123, 108, 131])
    cmap[26,:] = np.array([161, 66, 179])

    cmap[27,:] = np.array([163, 221, 160])
    cmap[28,:] = np.array([31, 146, 98])
    cmap[29,:] = np.array([99, 121, 30])

    cmap[30,:] = np.array([49, 89, 240])
    cmap[31,:] = np.array([116, 108, 9])
    cmap[32,:] = np.array([161, 176, 169])

    cmap[33,:] = np.array([80, 29, 135])
    cmap[34,:] = np.array([177, 105, 197])
    cmap[35,:] = np.array([139, 110, 246])

    cmap[36,:] = np.array([0, 0, 0])
    # cmap[37,:] = np.array([153, 153, 153])
 


    return cmap

class Colorize:

    def __init__(self, n=37): # n = nClasses
        # self.cmap = colormap(256)
        self.cmap = colormap_bdd(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        # print(size)
        color_images = torch.ByteTensor(size[0], 3, size[1], size[2]).fill_(0)
        # color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        # for label in range(1, len(self.cmap)):
        for i in range(color_images.shape[0]):
            for label in range(0, len(self.cmap)):
                mask = gray_image[0] == label
                # mask = gray_image == label

                color_images[i][0][mask] = self.cmap[label][0]
                color_images[i][1][mask] = self.cmap[label][1]
                color_images[i][2][mask] = self.cmap[label][2]

        return color_images
