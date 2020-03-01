# DeepKneeExplainer
Explainable Knee Osteoarthritis DiagnosisBased on Radiographs and Magnetic Resonance Images

There are only codes with simple file structures. Other codes, trained models and processed images are in the path /data/jiao of our GPU server. There would be another ReadMe file.
So codes uploaded here are separated by steps in the pipeline. In each Python file, there are several comments so that you can know where to change.

In general, first run codes for different modalities in Preprocessing folder, then you get processed images, which are the inputs of ROI step. ROI steps also need labelled bounding boxes, which you can get from GPU server. After ROI steps, you get the segmentation results, which can be dealt with function cv2.findContours() and boundingRect() in OpenCV. As I explained in thesis, here I used the method FRCNN so I dont write codes directly in the project.

From extracted ROIs, you can run classifier codes, then you will get trained models, which can be run for Grad-CAM visualization.
Before run codes, change related paths in the codes, where I have commented in the codes. Then direct run python xxx.py.
For classifiers, run VGG_new.py for VGG, ResNet_newJSN.py for ResNet and DenseNet_newJSN.py for DenseNet.
For visualization, run gradcam_plus_plus.py.
