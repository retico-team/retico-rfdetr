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
