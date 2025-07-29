import argparse
import os 
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import (
    ggplot, aes, geom_line, geom_text, theme, element_blank,scale_color_brewer
)
import pickle as pkl 
import glob




class GS_resultProcessor__Lr_g_Lr_D : 

    """
    attend un dossier de structure 
    input_folder / 

        expe_1 / 
        expe_2/ 
        /// 


    Exp_parser doit être une classe : 

        
    
    
    
    
    """

    def __init__(self,input_folder) :

        assert os.path.exists(input_folder),"le dossier sélectionné n'existe pas"
        self.list_exp=[os.path.join(input_folder,i) for i in os.listdir(input_folder)] 
        self.input_folder=input_folder
    

    def __get_lr_g_d(self,path) : 
        print(path)
        lr_d=path.split("_d_")[1].split("___lr_g")[0]
        lr_g=path.split("___lr_g_")[1].split("/")[0]
        return lr_d,lr_g

    def __extract_FID(self) : 
        df=pd.DataFrame({"lr_d_lr_g" : [], "iter" : [], "fid" : []})
        for exp in self.list_exp : 
            path=os.path.join(exp,"oasis_EM_d*_g*/FID/fid_log.npy")
            if len(glob.glob(path)) != 0 :

                tab=np.load(glob.glob(path)[0])
                name_comb="d:{},g:{}".format(*self.__get_lr_g_d(path))
                df_trans=pd.DataFrame({"iter" : tab[0], "fid" : tab[1]})
                df_trans["lr_d_lr_g"]=name_comb
             
                df=pd.concat([df,df_trans])

        self.df_fid=df

    def __plot_FID_matrix(self) : 
        


        min_=self.df_fid.groupby("lr_d_lr_g")["fid"].min()
        combs=[(i.split(",")[0].split(":")[1],i.split(",")[1].split(":")[1]) for i in pd.unique(self.df_fid["lr_d_lr_g"])]
        list_d=list(set([i[0] for i in combs]))
        list_g=list(set([i[1] for i in combs]))
        list_d.sort()
        list_g.sort()
        #print(combs)
        matrix=np.full((len(list_g),len(list_d)),fill_value=0)
        
        #print(min_)
        for di,d in enumerate(list_d) : 
            for gi,g in enumerate(list_g): 
                if "d:{},g:{}".format(d,g) in min_.index:
                  
                  matrix[gi,di]=float(min_.loc["d:{},g:{}".format(str(d),str(g))])
                else : 
                   pass
        

        
        print(list_d)
        print(list_g)
        print(matrix)


        plot=plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, xticklabels=list_g, yticklabels=list_d, annot=True,  fmt=".1f",cmap="Blues")
        plt.xlabel("Generator Learning rate " )
        plt.ylabel("Discriminator Learning rate " )
        plt.title("Heatmap")
        plt.tight_layout()
        plot.savefig(os.path.join(self.output_folder,"fid_matrix.pdf"),dpi=1000)
        plt.close("all")
    
    def __plot_FID_lines(self) : 

                
            plot = (
                ggplot(self.df_fid, aes(x="iter", y="fid", color="lr_d_lr_g"))
                + geom_line()
                + scale_color_brewer(type='qual', palette='Paired')
            )


            plot.save(os.path.join(self.output_folder,"fid_lines.pdf"),dpi=1000)
            
            plt.close("all")

    
    def __prepare_output(self) : 

        assert not os.path.exists(os.path.join(self.input_folder,"analysis_output")),"Un dossier d'analyse est déja présent"
        self.output_folder=os.path.join(self.input_folder,"analysis_output")
        os.makedirs(self.output_folder)       

    

    def __extract_loss(self): 
            df=pd.DataFrame()
            self.available_loss=list()
            for exp in self.list_exp : 
                path=os.path.join(exp,"oasis_EM_d*_g*/losses/losses.npy")
                if len(glob.glob(path)) != 0 :
                     path_opt=os.path.join(exp,"oasis_EM_d*_g*/opt.pkl")
                     
                     with open(glob.glob(path_opt)[0],"rb") as file : 
                         
                         u=pkl.load(file)
                     
                     freq_loss_save=u.freq_save_loss
                    
                     tab=np.load(glob.glob(path)[0],allow_pickle=True).item()
                     for k in tab.keys() : 
                         if k not in self.available_loss: 
                             self.available_loss.append(k)
                     iters=np.arange (0,freq_loss_save*len(tab[list(tab.keys())[0]]),freq_loss_save)
                    
           
                     subset_df=pd.DataFrame(tab)
                     subset_df["iter"]=iters
                     subset_df["lr_d_lr_g"]="d:{},g:{}".format(*self.__get_lr_g_d(path))
              
                     df=pd.concat([df,subset_df])
            self.loss_df=df
            


    def __plot_loss_curves(self) : 

        self.loss_df=self.loss_df[self.loss_df["lr_d_lr_g"]!="d:0.0001,g:0.0001"]
        for k in self.available_loss : 

              plot = (
                ggplot(self.loss_df, aes(x="iter", y=k, color="lr_d_lr_g"))
                + geom_line()
                + scale_color_brewer(type='qual', palette='Paired')
            )
              
              plot.save(os.path.join(self.output_folder,"{}_curve.pdf".format(k)),dpi=1000)


        

    def __call__(self) : 
        self.__prepare_output()
        self.__extract_FID()
        self.__plot_FID_matrix()
        self.__plot_FID_lines()
        self.__extract_loss()
        self.__plot_loss_curves()




parser=argparse.ArgumentParser()
parser.add_argument("--input_folder",type=str,help="input folder")




if __name__=="__main__" : 

    args=parser.parse_args()
    processor=GS_resultProcessor__Lr_g_Lr_D(input_folder=args.input_folder)
    processor()





