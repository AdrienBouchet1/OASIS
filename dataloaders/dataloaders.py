import torch
import sys 
import os 
def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode=="EM":
        print("EM bien reconnu")
        return "EM_Dataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)
    print(os.listdir() )
    sys.path.append("/home/adrienb/Documents/Adrien/Code/OASIS")
    file = __import__("dataloaders."+dataset_name)
    #file = __import__(dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True, num_workers = opt.num_workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers = opt.num_workers)

    return dataloader_train, dataloader_val