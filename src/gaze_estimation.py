'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import math
from openvino.inference_engine import IECore,IENetwork


class GazeEstimationModel:
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

        self.inp_name = [i for i in self.model.inputs.keys()]
        self.inp_shape = self.model.inputs[self.inp_name[1]].shape
        self.outp_names = [a for a in self.model.outputs.keys()]

    def load_model(self):
        
        self.plugin=IECore()       
        
        layers_supported = self.plugin.query_network(network=self.model, device_name=self.device)
        layers_unsupported = [l for l in self.model.layers.keys() if l not in layers_supported]

        if len(layers_unsupported)!=0:
            print("Unsupported layers found!!!")
            exit(1)
        self.exec_net=self.plugin.load_network(network=self.model,device_name=self.device,num_requests=1)

    def predict(self, l_eye,r_eye,angle):
        
        le_img_processed, re_img_processed = self.preprocess_input(l_eye, r_eye)
        outputs = self.exec_net.infer({'head_pose_angles':angle, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,angle)
        return new_mouse_coord, gaze_vector
    
    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, lefteye,righteye):
    
        self.lefteye=cv2.resize(lefteye,(self.inp_shape[3],self.inp_shape[2]))
        self.righteye=cv2.resize(righteye,(self.inp_shape[3],self.inp_shape[2]))
        self.lefteye=self.lefteye.transpose((2, 0, 1))
        self.righteye=self.righteye.transpose((2, 0, 1))  
        self.lefteye=self.lefteye.reshape(1, *self.lefteye.shape)
        self.righteye=self.righteye.reshape(1, *self.righteye.shape)
        return self.lefteye,self.righteye

    def preprocess_output(self, outputs,angle):
        
        gaze_vector = outputs[self.outp_names[0]].tolist()[0]
        x = angle[2] 
        cosValue = math.cos(x * math.pi / 180.0)
        sinValue = math.sin(x * math.pi / 180.0)
        
        xc = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        yc = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (xc,yc), gaze_vector
