

import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from torchvision import transforms as TR
from PIL import Image
import torch 
import numpy as np 

import os 
class OasisInferenceEM : 


    def __init__(self,opt,output_folder): 

        opt.load_size = 256
        opt.crop_size=256
        opt.label_nc = 6
        opt.contain_dontcare_label = True
        opt.semantic_nc = 7
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        self.opt=opt
        
        self.model = models.OASIS_model(self.opt)
        self.model = models.put_on_multi_gpus(self.model, self.opt)
        self.model.eval()

        self.output_folder=output_folder
        pass

    def __preprocess_label(self,label) : 
        """voir le preprocessing du label dans model.preprocess_input"""
        label=label.to("cuda")
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        label.unsqueeze_(0)
        label.unsqueeze_(0)

        bs, _, h, w = label.shape
        nc = opt.semantic_nc
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label, 1.0)
        return input_semantics
    

    def __tens_to_im(self,tens):
        """
        Directement copié depuis utils
        """
        out = (tens + 1) / 2
        out.clamp(0, 1)
        return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))
    

    def __inference(self, x: torch.tensor) : 

        generated = self.model(None, x, "generate", None)

        return generated
        

    def inference_expe(self) : 



        for cat in range(1,6) : 

            os.makedirs(os.path.join(self.output_folder,str(cat)))


            for iter in range(50) : 

                label=torch.from_numpy(np.full((256,256),fill_value=cat))
                label=self.__preprocess_label(label)
                img=self.__inference(label).squeeze(0)

                # img=img*255
                # img=img.squeeze_(0).squeeze_(0).to(torch.uint8)
                # img=img.cpu().numpy()
               
                img=self.__tens_to_im(img)*255
                print("!!!",img)
                im=Image.fromarray(np.squeeze(img.astype(np.uint8)))
                im.save(os.path.join(self.output_folder,str(cat),"{}_{}.png".format(int(cat),int(iter))))





if __name__=="__main__" : 


    opt = config.read_arguments(train=False)
    Oa=OasisInferenceEM(opt,output_folder="/home/adrienb/Documents/Adrien/Code/OASIS/inference_output_2")
    Oa.inference_expe()