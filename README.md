# Retico Roboflow Detection Transformer Module

A ReTico module for RfDeTr that works with Roboflow's detection models. The webcam captures the user information, which passes through the model and outputs the name of the detectable object. The model contains two different types of constructors, one using the original RfDeTr and another using the new Hugging Face model. The native RfDeTr is recommended due to its reliability compared to the current Hugging Face model.

## Installation and requirements

* Install retico_core:
```pip install git+https://github.com/retico-team/retico-core.git```
* Install the retico-rfdetr:
```pip install retico-rfdetr```

## Modules

### `RFDETRModule` (Native RF-DETR)
Uses Roboflow's `rfdetr` package. Recommended for real-time webcam use.

**Model options:** `nano`, `small`, `medium`, `large`

**Segmentation model options:** `nano`, `small`, `medium`, `large`

#### Arguments:
* `model` : Model size to use, defaults to small
* `pretain` : Path to custom checkpoint, defaults to COCO weights
* `use_seg` : Segmentation model variation, defaults to False
* `show` : Displayable output window, defaults to False
* `threshold` : Confidence threshold, default to 0.25

### `HFRFDETRModule` (HuggingFace RF-DETR)
Uses HuggingFace transformers to load RF-DETR models from the Roboflow HuggingFace.

**Detection model options:** `medium`, `large`

**Segmentation model options:** `medium`, `large`

#### Arguments:
* `model` : Model size to use, defaults to small
* `pretain` : Path to custom checkpoint, defaults to COCO weights
* `use_seg` : Segmentation model variation, defaults to False
* `show` : Displayable output window, defaults to False
* `threshold` : Confidence threshold, default to 0.25

## Example

### Utilizing the original RfDeTr model

```
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
prefix = "/path/to/prefix/"

sys.path.append(prefix + "retico-core")
sys.path.append(prefix + "retico-vision")
sys.path.append(prefix + "retico-rfdetr")

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule
from retico_rfdetr.rfdetr import RFDETRModule

rfdetr = RFDETRModule(model="large", pretrain=None, use_seg=True, show=True)
webcam = WebcamModule()
debug = DebugModule(print_payload_only=True)

webcam.subscribe(rfdetr)
rfdetr.subscribe(debug)

print(f"RFDETR Model running")

webcam.run()
rfdetr.run()
debug.run()

input()

webcam.stop()
rfdetr.stop()
debug.stop()
```

### Utilizing the Hugging Face RfDeTr model

```
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
prefix = "/path/to/prefix/"

sys.path.append(prefix + "retico-core")
sys.path.append(prefix + "retico-vision")
sys.path.append(prefix + "retico-rfdetr")

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule
from retico_rfdetr.hfrfdetr import HFRFDETRModule

rfdetr = HFRFDETRModule(model="large", use_seg=True, show=True)
webcam = WebcamModule()
debug = DebugModule(print_payload_only=True)

webcam.subscribe(rfdetr)
rfdetr.subscribe(debug)

print(f"RFDETR Model running")

webcam.run()
rfdetr.run()
debug.run()

input()

webcam.stop()
rfdetr.stop()
debug.stop()
```