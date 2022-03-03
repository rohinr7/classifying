# COSMONiO Classifier

### Requirements
It is recommended to install the following requirements in a separated environment. This can be done by using tools such as Anaconda. Refer to this link for more information: https://docs.anaconda.com/anaconda/user-guide/getting-started/.

To run the COSMONiO Classifier you will first need to install Python 3.6 (64-bit). If you are on Windows you will need Visual studio build tools as well which can be downloaded from the following link: https://visualstudio.microsoft.com/visual-cpp-build-tools/.
After that you need to install some requirements. These can be found in the project folder and can be installed by executing the following command: `pip install -r requirements.txt`.

The system PATH needs to be modified for OpenVINO. Please follow the instructions given in this [link](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_pip.html).

| DISCLAIMER: |
| ----------- |
This repo is tested based on OpenVINO version 2021.2. The installation may not work, or the stability of the code might be affected with other versions.

### OpenVINO Model
The labels and OpenVINO model can be found in the model directory. The model can be updated by exporting an OpenVINO model
 from NOUS and replacing the inference_model.bin and inference_model.graph files in the model directory.

### Starting the classifier
Now the Classifier can be started. You can choose to either supply a video file or start the script without specifying anything.
If nothing is specified by default the first camera available on the system is selected.
You can start the Classifier with a video like this: `python classifier.py --file path/to/video.mp4`. The classifier can be quit by pressing the Q key.