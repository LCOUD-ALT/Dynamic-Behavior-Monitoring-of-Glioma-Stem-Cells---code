# Dynamic-Behavior-Monitoring-of-Glioma-Stem-Cells---code

## 🚀 Project Introduction
Cellular dynamic behavior analysis constitutes the fundamental approach to deciphering life activity mechanisms, with its research depth directly determining advancements in critical biomedical domains including tissue regeneration, tumor metastasis intervention, and precision drug evaluation. To address the persistent limitations of conventional microscopic tracking technologies in complex physiological environments, specifically temporal continuity disruption, insufficient multi-scale dynamic feature capture, and inefficient cross-modal data fusion, this study introduces a novel cellular behavior monitoring paradigm integrating spatiotemporal priors and morphological constraints. First, a cross-frame semantic propagation network is developed by fusing temporal microscopic imaging sequences with optical flow motion priors to establish continuous spatiotemporal representations of cellular morphological evolution, resolving topological abruptness from single-frame segmentation. Second, a deformable spatiotemporal interaction module combines dynamic convolutional kernels with gradient-sensitive attention mechanisms to address feature extraction challenges in densely adherent and rapidly deforming cellular populations. Finally, a bidirectional co-optimization architecture embeds trajectory consistency constraints into feature learning through segmentation-tracking mutual reinforcement. Experimental results on glioblastoma stem cell dynamic imaging datasets demonstrate 96.54% trajectory integrity and 93% division detection accuracy, enabling precise quantification of collective migration velocities and directional motility patterns. This framework provides a robust analytical tool for unraveling cellular dynamic behavior mechanisms, driving a paradigm shift in life sciences from static phenotypic observation to dynamic process deconstruction.
## 📥 Environment
```python
python==3.8.20 
pytorch>=2.4.1
```

## 🧩 Installation
### Clone repo  
```python
git clone "https:https://github.com/LCOUD-ALT/Dynamic-Behavior-Monitoring-of-Glioma-Stem-Cells---code.git”  HG+RFA-YOLO
cd HG+RFA-YOLO
```
 
### Install dependencies   
```python
pip install -r requirements.txt
```

## 🧠 Training & Inference
1. ### Train 
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('HG+RFA.yaml')

    model.train(data='path/to/your/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                amp=True,  
                project='runs/train',
                name='exp',
                )
```
2. ### Inference
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('path/to/best.pt')
    results = model.predict("image.jpg",  conf=0.5)  
    results[0].show()  # Display results
```
## 📜 License
Licensed under the MIT License; see LICENSE for details.
