<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_uniface_detection</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_uniface_detection">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_uniface_detection">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_uniface_detection/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_uniface_detection.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run inference for multi-face detection using UniFace library. UniFace is a lightweight production-ready face analysis library built on ONNX Runtime, supporting multiple face detection models including RetinaFace, YOLOv5Face, SCRFD, and YOLOv8Face.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_detection/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_uniface_detection", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_detection/main/images/example.jpg")

# Display result
display(detector.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_uniface_detection", auto_connect=True)

detector.set_parameters({
    "model_name": "retinaface",
    "conf_thres": "0.6",
    "nms_thres": "0.4",
})

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_detection/main/images/example.jpg")

# Display result
display(detector.get_image_with_graphics())
```

- **model_name** (str, default="retinaface"): Face detection model to use. Options: "retinaface", "yolov5face", "scrfd", "yolov8face".
- **conf_thres** (float, default="0.6"): Object detection confidence threshold.
- **nms_thres** (float, default="0.4"): Non-maximum suppression threshold.

***Note***: parameter key and value should be in **string format** when added to the dictionary.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_uniface_detection", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_detection/main/images/example.jpg")

# Iterate over outputs
for output in detector.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

