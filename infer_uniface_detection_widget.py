"""
Module that implements the UI widget of the algorithm.
"""
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_uniface_detection.infer_uniface_detection_process import InferUnifaceDetectionParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


class InferUnifaceDetectionWidget(core.CWorkflowTaskWidget):
    """
    Class that implements UI widget to adjust algorithm parameters.
    Inherits PyCore.CWorkflowTaskWidget from Ikomia API.
    """

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferUnifaceDetectionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Model selection
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_model.addItem("retinaface")
        self.combo_model.addItem("yolov5face")
        self.combo_model.addItem("scrfd")
        self.combo_model.addItem("yolov8face")
        self.combo_model.setCurrentText(self.parameters.model_name)

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.0,
            max=1.0,
            step=0.05
        )

        # NMS threshold
        self.spin_nms_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "NMS threshold",
            self.parameters.nms_thres,
            min=0.0,
            max=1.0,
            step=0.05
        )

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.nms_thres = self.spin_nms_thres.value()
        self.parameters.update = True

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


class InferUnifaceDetectionWidgetFactory(dataprocess.CWidgetFactory):
    """
    Factory class to create algorithm widget object.
    Inherits PyDataProcess.CWidgetFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_uniface_detection"

    def create(self, param):
        """Instantiate widget object."""
        return InferUnifaceDetectionWidget(param, None)
