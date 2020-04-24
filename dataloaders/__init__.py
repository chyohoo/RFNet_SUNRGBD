from dataloaders.datasets import Data_SUNRGBD
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'SUNRGBD':
        train_set = Data_SUNRGBD.SUNRGBD(args,phase='train')
        val_set = Data_SUNRGBD.SUNRGBD(args,phase='val')
        test_set = Data_SUNRGBD.SUNRGBD(args,phase='test')
        num_class = 37
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

   
   
    else:
        raise NotImplementedError

