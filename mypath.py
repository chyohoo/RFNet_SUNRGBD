class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'SUNRGBD':
            return 'E:\\datasets\\cityscapes\\'      # folder that contains leftImg8bit/
         # folder that mixes Cityscapes and Lost and Found
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
