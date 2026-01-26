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

        # Output anonymized checkbox
        self.check_anonymized = pyqtutils.append_check(
            self.grid_layout,
            "Output anonymized image",
            self.parameters.output_anonymized
        )

        # Blur method selection
        self.combo_blur_method = pyqtutils.append_combo(self.grid_layout, "Blur method")
        self.combo_blur_method.addItem("gaussian")
        self.combo_blur_method.addItem("pixelate")
        self.combo_blur_method.addItem("blackout")
        self.combo_blur_method.addItem("elliptical")
        self.combo_blur_method.addItem("median")
        self.combo_blur_method.setCurrentText(self.parameters.blur_method)

        # Blur strength (for gaussian, elliptical, median)
        self.spin_blur_strength = pyqtutils.append_double_spin(
            self.grid_layout,
            "Blur strength",
            self.parameters.blur_strength,
            min=1.0,
            max=10.0,
            step=0.5
        )

        # Pixel blocks (for pixelate)
        self.spin_pixel_blocks = pyqtutils.append_spin(
            self.grid_layout,
            "Pixel blocks",
            self.parameters.pixel_blocks,
            min=1,
            max=50,
            step=1
        )

        # Margin
        self.spin_margin = pyqtutils.append_spin(
            self.grid_layout,
            "Margin",
            self.parameters.margin,
            min=0,
            max=100,
            step=5
        )

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.nms_thres = self.spin_nms_thres.value()
        
        # Anonymization parameters
        self.parameters.output_anonymized = self.check_anonymized.isChecked()
        self.parameters.blur_method = self.combo_blur_method.currentText()
        self.parameters.blur_strength = self.spin_blur_strength.value()
        self.parameters.pixel_blocks = self.spin_pixel_blocks.value()
        self.parameters.margin = self.spin_margin.value()
        
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
