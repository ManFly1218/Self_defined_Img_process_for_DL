import PIL.Image as Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
class Image_preprocess():
    def __init__(self,img,re_imsize,trim_operate=True):
        self.init_img=img
        self.trim_operate=trim_operate
        self.resize=re_imsize
        self.Img_objects={}
        self.Img_objects['1Initial image']=None
        self.Img_objects['2Trim image']=None
        self.Img_objects['3Resized image']=None
        self.Img_objects['4Resized grayscale image']=None
        
        self.ImgArrays={}
        self.ImgArrays['1Initial RGB array']=None
        self.ImgArrays['2Trim RGB array']=None
        self.ImgArrays['3Resized RGB tensor']=None
        self.ImgArrays['4Resized grayscale tensor']=None
        
    def Func_1_Trim_operator(self,show=False):
        """""

        """""
        #load image by given path
        image=Image.open(self.init_img)
        self.Img_objects['1Initial image']=image
        #Convert to RGB array
        img_array=np.asarray(image.convert('RGB'))
        self.ImgArrays['1Initial RGB array']=img_array
        #If need trim
        if self.trim_operate:
            idx_array=np.array(np.where(img_array==255)[:-1])#Irrevelent to the channel part
            Trim_array=img_array[np.min(idx_array[0,:]):np.max(idx_array[0,:]),\
                           np.min(idx_array[-1,:]):np.max(idx_array[-1,:]),:]
            self.ImgArrays['2Trim RGB array']=Trim_array
            Trim_img=Image.fromarray(np.uint8(Trim_array))
            self.Img_objects['2Trim image']=Trim_img
        else:
        #No need trim
            self.ImgArrays['2Trim RGB array']=img_array
            self.Img_objects['2Trim image']=image
        #If need to show trim-image
        if show:
            plt.imshow(self.Img_objects['2Trim image'])
            plt.axis('off')#Turn off the axis scale
            plt.show()
#        return self.Img_objects,self.ImgArrays

    def Func_2_RGB_Resized_operator(self,show=False):
        #self.Img_objects,self.ImgArrays=self.Func_1_Trim_operator()
        self.Func_1_Trim_operator()
        #Define loader and unloader
        #loader: take image as input, transfer to tensor
        loader=transforms.Compose([\
                  transforms.Resize((self.resize,self.resize)),\
                  transforms.ToTensor(),\
                  transforms.Normalize((0.485,0.456,0.406),\
                                       (0.229,0.224,0.225))])
        #unloader: take tensor as input, transfer back to image
        unloader=transforms.Compose([\
                            transforms.Normalize((-0.485/0.229,\
                                                -0.456/0.224,\
                                                -0.406/0.225),\
                                                (1/0.229,1/0.224,1/0.225)),\
                            transforms.ToPILImage()])
        #loading
        Resized_img_tensor=loader(self.Img_objects['2Trim image'])
        self.ImgArrays['3Resized RGB tensor']=Resized_img_tensor
        #unloading
        Resized_img=unloader(self.ImgArrays['3Resized RGB tensor'])
        self.Img_objects['3Resized image']=Resized_img
        if show:
            plt.imshow(self.Img_objects['3Resized image'])
            plt.axis('off')#Turn off the axis scale
            plt.show()
#        return self.Img_objects,self.ImgArrays
    
    def Func_3_Gray_scale(self,show=False):
        #self.Img_objects,self.ImgArrays=self.Func_2_RGB_Resized_operator()
        self.Func_2_RGB_Resized_operator()
        grayscale_loader = transforms.Compose(\
                        [transforms.Grayscale(num_output_channels=1),\
                         transforms.ToTensor(),\
                         transforms.Normalize(mean=[0.5], std=[0.5])])

        grayscale_unloader = transforms.Compose(\
                          [transforms.Normalize(mean=[-1], std=[1/0.5]),\
                            transforms.ToPILImage()])
        grayscale_tensor=grayscale_loader(self.Img_objects['3Resized image'])
        self.ImgArrays['4Resized grayscale tensor']=grayscale_tensor
        gray_scale_img=grayscale_unloader(self.ImgArrays['4Resized grayscale tensor'])
        self.Img_objects['4Resized grayscale image']=gray_scale_img
        if show:
            plt.imshow(self.Img_objects['4Resized grayscale image'])
            plt.axis('off')#Turn off the axis scale
            plt.show()
    def Func_4_implementation(self,show=False):
#        self.Func_1_Trim_operator()
#        self.Func_2_RGB_Resized_operator()
        self.Func_3_Gray_scale()
        if show:
            fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2, figsize=(7,7))#share the y_axis
            #plot 1Initial image
            ax0.imshow(self.Img_objects['1Initial image'])
            ax0.set_title('The initial image,\n the array shape: {}'\
                          .format(self.ImgArrays['1Initial RGB array'].shape), fontsize=7.5)#title of subplot
            ax0.axis('off')
            #plot 2Trim image
            ax1.imshow(self.Img_objects['2Trim image'])
            ax1.set_title('The pruned image,\n the array shape: {}'\
                          .format(self.ImgArrays['2Trim RGB array'].shape), fontsize=7.5)
            ax1.axis('off')
            #plot 3Resized image
            ax2.imshow(self.Img_objects['3Resized image'])
            ax2.set_title('Resized image,\n the tensor shape: {}'\
                          .format(self.ImgArrays['3Resized RGB tensor'].shape), fontsize=7.5)
            ax2.axis('off')
            #plot 4Resized grayscale image
            ax3.imshow(self.Img_objects['4Resized grayscale image'])
            ax3.set_title('Resized grayscale image,\n the tensor shape: {}'\
                          .format(self.ImgArrays['4Resized grayscale tensor'].shape), fontsize=7.5)
            ax3.axis('off')
            #adjust grid of three subplots
            plt.show()            
#print(Trim.shape)
