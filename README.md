# Retico Roboflow Detection Transformer Module

A ReTico module for RfDeTr that works with Roboflow's detection models. The webcam captures the user information, which passes through the model and outputs the name of the detectable object. The model contains two different types of constructors, one using the original RfDeTr and another using the new Hugging Face model.

## Installation and requirements

* Install retico_core: 
```https://github.com/retico-team/retico-core.git```
* Install the retico-vision package:
```pip install git+https://github.com/retico-team/retico-vision.git```
* pip install pywebcam

For HuggingFace login use this command and provide your token.

* huggingface-cli login

Some Hugging Face models will require authorization. Go to the model's page and request access to the model.

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