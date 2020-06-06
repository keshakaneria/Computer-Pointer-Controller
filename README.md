# Computer Pointer Controller

This project focuses on controlling the computer mouse pointer with the help of human eyes direction. It is able to take Video file as well as WebCam inputs and finally gives the output.

The part of the video as a screenshot is shown below:

![](https://github.com/keshakaneria/Computer-Pointer-Controller/blob/master/media/Working_Video.png?raw=true)

## Installation to Project Set Up 

1. Clone this repository from: 
	https://github.com/keshakaneria/Computer-Pointer-Controller.git

2. Install OpenVino Toolkit on your local computer from [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html).

3. Initialze the OpenVino Environment on your local setup. Given below are the commands to initialize:

	`cd C:\Program Files (x86)\IntelSWTools\openvino\bin\`
	
	`setupvars.bat`

4. As you have the repository, models are already downloaded and added in the models folder.
										--OR--
4. Download the models required for the project from [here](https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1/).

Given below are the commands to download manually:

 **1. Face Detection Model**

	`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"`

 **2. Gaze Estimation Model**

	`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"`
 
 **3. Facial Landmarks Detection Model**
 
	`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"`

 **4. Head Pose Estimation Model**

	`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"`


## How to run a Demo

- Open a new terminal everytime to run with the following commands:
	
	`cd C:\Program Files (x86)\IntelSWTools\openvino\bin\`
	
	`setupvars.bat`

- To switch to directory where files are writtten:

	`cd <project-repo-path>/src`

- Run the main.py file

*For CPU*
	
```
python main.py -fd  models\Face_detection\face-detection-adas-binary-0001.xml  -fl models\Landmarks_detection\FP32\landmarks-regression-retail-0009.xml  -hp models\Head_Pose\FP32\head-pose-estimation-adas-0001.xml  -ge models\Gaze_Estimation\FP32\gaze-estimation-adas-0002.xml  -i bin\demo.mpeg -l opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libinference_engined.dylib -d CPU -pt 0.6
```

*For GPU*

```
python main.py -fd 'Path of xml file of face detection model' -fl 'Path of xml file of facial landmarks detection model' -hp 'Path of   xml file of head pose estimation model' -ge 'Path of xml file of gaze estimation model' -i 'Path of input video file or enter cam for   taking input video from webcam' -d 'GPU'
```

*For FPGA*

```
python main.py -fd 'Path of xml file of face detection model' -fl 'Path of xml file of facial landmarks detection model' -hp 'Path of   xml file of head pose estimation model' -ge 'Path of xml file of gaze estimation model' -i 'Path of input video file or enter cam for   taking input video from webcam' -d 'HETERO:FPGA,CPU'
```
 

### Command Line Arguments

Following commands line arguments are used for running main.py file:

1. `-i` (required) : Specify the path of input video file or enter cam for taking input video from webcam

2. `-fl` (required) : Path to .xml file of Facial Landmark Detection model
3. `-hp` (required) : Path to .xml file of Head Pose Estimation model
4. `-ge` (required) : Path to .xml file of Gaze Estimation model.
5. `-fd` (required) : Specify the path of Face Detection model's xml file

6. `-d` (optional) : Specify the target device to infer on,"CPU, GPU, FPGA or MYRIAD is acceptable. Looks for a suitable plugin for device specified "(CPU by default)".

7. `-l` (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.
8. `-pt` (optional): Probability threshold for model to detect the face accurately from the video frame.

### Overview of the files

![](https://github.com/keshakaneria/Computer-Pointer-Controller/blob/master/media/structure.png?raw=true)

- The repository contains a 'media' folder which has a 'demo.mp4' video file, can be used as the input file for the project. It also has a screenshot which shows the working of video.

- It has requirements.txt file which contains all the necessary dependencies to be installed before running the project. The media folder includes a screenshot which shows the installation of dependencies required. GIven image shows the requirements getting installed:
![](https://github.com/keshakaneria/Computer-Pointer-Controller/blob/master/media/Installed_Requirements.png?raw=true)

#### Models used in the project are saved in the 'models' folder:

- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) was used.

- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html) was one.

- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) was used.

- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) was also the one.

* The src folder in project directory contains the following python files:

1. The face_detection.py, head_pose_estimation.py, facial_landmarks_detection.py, gaze_estimation.py:

	Contains class functions to preprocess the inputs and run inference models on them. It sends it to mouse_controller to move the mouse position.

2. The mouse_controller.py:

	It takes x and y co-ordinates from the gaze.py to move the mouse.

3. The input_feeder.py:

	Input is feeded to the files such as a Video file or a WebCam. It results the frames for running inference.

4. main.py:

	Main file to run the app.

## Benchmarking Results

* CPU with multiple model precisions:

Face Detection Model | Landmarks Detection Model | Head Pose Estimation Model | Gaze Estimation Model | Model Loading Time | Inference Time | FPS |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
INT 1 | FP32 | FP32 | FP32 | 0.605s | 20.8s | 2.41 |
INT 1 | FP32 | FP32 | INT8 | 2.221s | 31.2s | 1.61 |
INT 1 | FP16 | FP16 | INT8 | 1.209s | 31.8s | 1.91 |
INT 1 | FP16 | FP16 | FP16 | 1.48s | 31.1s | 1.82 |

* GPU with multiple model precisions:

Face Detection Model | Landmarks Detection Model | Head Pose Estimation Model | Gaze Estimation Model | Model Loading Time | Inference Time | FPS |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
INT 1 | FP32 | FP32 | FP32 | 22.15s | 22.1s | 2.57 |
INT 1 | FP32 | FP32 | INT8 | 27.68s | 29.6s | 2.18 |
INT 1 | FP16 | FP16 | INT8 | 30.12s | 29.3s | 2.07 |
INT 1 | FP16 | FP16 | FP16 | 29.32s | 28.3s | 2.02 |

* I tried 4 combinations with different precisions like INT8,FP16,FP32 on CPU and GPU as well. I tried to reduce precision value where precision reduced accuracy also.

*Note: When we use lower precision model, we get lower accuracy than higher precision model.*

#### Importants:

1. GPU with precision FP16 posses more Frames especailly because GPU has several execution units and their instruction sets are optimized for 16bit floating point data types.

2. CPU with precision FP32 was a best fit giving us the best results whether we compare FPS, Model Load Time or even Inference Time.

3. Average FPS for GPU as well for most of the combinations FPS was 2.21. Inference time was giving an average of 27.325s for all combinations for GPU.

4. For CPU, FPS was averaging around 1.94.


## Suggestions to Stand Out

* Both video file and webcam feed as input are tried and worked successfully.

* Allowing the user to select their input option in the command line arguments:

	`-i` argument takes the input video file or a webcam, for accessing video file the command is `-i "path of video file"` whereas for accessing webcam `-i "cam"`.
 
* Depending on chosen option it will work.

### Edge Cases

* It prints 'unable to detect the face", if model can't detect the face. It reads another frame until it detects the face or user closes the window.

* Model takes the first detected face, if there are more than one face detected in the frame and controls the mouse pointer.
