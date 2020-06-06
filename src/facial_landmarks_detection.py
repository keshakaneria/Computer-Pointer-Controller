'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class FacialLandmarksDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.network = None
        self.plugin = None
        self.exec_net = None
        self.inp_name = None
        self.outp_names = None
        self.inp_shape = None
        self.outp_shape = None


        try:
            self.ie = IECore()
            self.model=self.ie.read_network(model=self.model_structure,weights=self.model_weights)
        except Exception as e:
            raise ValueError("Network was not able to initialized! Please Enter the correct model path")

        self.inp_name=next(iter(self.model.inputs))
        self.inp_shape=self.model.inputs[self.inp_name].shape
        self.outp_names=next(iter(self.model.outputs))
        self.outp_shape=self.model.outputs[self.outp_names].shape

    def load_model(self):
        
        self.plugin=IECore()       
        
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device)
        layers_unsupported = [l for l in self.model.layers.keys() if l not in layers_supported]

        if len(layers_unsupported)!=0:
            print("Unsupported layers found!!!")
            exit(1)
        self.exec_net=self.plugin.load_network(network=self.model,device_name=self.device,num_requests=1)
        
    def predict(self, image):
        
        self.processed_image=self.preprocess_input(image)
        outputs = self.exec_net.infer({self.inp_name:self.processed_image})
        coords = self.preprocess_output(outputs)
        
        h=image.shape[0]
        w=image.shape[1]
        
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) 

        right_xmin=coords[2]-10
        right_xmax=coords[2]+10
        right_ymin=coords[3]-10              
        right_ymax=coords[3]+10

        left_xmin=coords[0]-10
        left_xmax=coords[0]+10
        left_ymin=coords[1]-10            
        left_ymax=coords[1]+10
        
        left_eye =  image[left_ymin:left_ymax, left_xmin:left_xmax]
        right_eye = image[right_ymin:right_ymax, right_xmin:right_xmax]

        eye_coords = [[left_xmin,left_ymin,left_xmax,left_ymax], [right_xmin,right_ymin,right_xmax,right_ymax]]
        
        return left_eye, right_eye, eye_coords


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        image_ct = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image=cv2.resize(image_ct,(self.inp_shape[3],self.inp_shape[2]))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
        return self.image

    def preprocess_output(self, outputs):
    
        res=outputs[self.output_name][0]
        lx = res[0].tolist()[0][0]
        ly = res[1].tolist()[0][0]
        rx = res[2].tolist()[0][0]
        ry = res[3].tolist()[0][0]
        return(lx,ly,rx,ry)
