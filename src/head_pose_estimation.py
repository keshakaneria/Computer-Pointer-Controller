'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class HeadPoseEstimationModel:
    
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


        self.inp_name = next(iter(self.model.inputs))
        self.inp_shape = self.model.inputs[self.inp_name].shape
        self.outp_names = [a for a in self.model.outputs.keys()]

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
        Results = self.preprocess_output(outputs)
        return Results

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        self.image=cv2.resize(image,(self.inp_shape[3],self.inp_shape[2]))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
        return self.image

    def preprocess_output(self, outputs):
    
        result = []
        result.append(outputs['angle_y_fc'].tolist()[0][0])
        result.append(outputs['angle_p_fc'].tolist()[0][0])
        result.append(outputs['angle_r_fc'].tolist()[0][0])
        return result

