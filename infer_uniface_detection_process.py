"""
Module that implements the core logic of algorithm execution.
"""
import os
import copy
import numpy as np

from ikomia import core, dataprocess

from uniface.privacy import BlurFace

from .models.model_loader import create_detector


class InferUnifaceDetectionParam(core.CWorkflowTaskParam):
    """
    Class to handle the algorithm parameters.
    Inherits PyCore.CWorkflowTaskParam from Ikomia API.
    """
    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Options: "retinaface", "yolov5face", "scrfd", "yolov8face"
        self.model_name = "retinaface"
        self.conf_thres = 0.6
        self.nms_thres = 0.4

        # Anonymization parameters
        self.output_anonymized = False
        # Available methods: 'gaussian', 'pixelate', 'blackout', 'elliptical', 'median'
        self.blur_method = "pixelate"
        self.blur_strength = 3.0
        self.pixel_blocks = 15
        self.margin = 20

        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        self.model_name = params.get("model_name", "retinaface")
        self.conf_thres = float(params["conf_thres"])
        self.nms_thres = float(params["nms_thres"])

        # Anonymization parameters
        self.output_anonymized = params.get(
            "output_anonymized", "False") == "True"
        self.blur_method = params.get("blur_method", "pixelate")
        self.blur_strength = float(params.get("blur_strength", 3.0))
        self.pixel_blocks = int(params.get("pixel_blocks", 15))
        self.margin = int(params.get("margin", 20))

        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "model_name": str(self.model_name),
            "conf_thres": str(self.conf_thres),
            "nms_thres": str(self.nms_thres),
            # Anonymization parameters
            "output_anonymized": str(self.output_anonymized),
            "blur_method": str(self.blur_method),
            "blur_strength": str(self.blur_strength),
            "pixel_blocks": str(self.pixel_blocks),
            "margin": str(self.margin)
        }
        return params


class InferUnifaceDetectionParamFactory(dataprocess.CTaskParamFactory):
    """Factory class to create parameters object."""

    def __init__(self):
        dataprocess.CTaskParamFactory.__init__(self)
        self.name = "infer_uniface_detection"

    def create(self):
        """Instantiate parameters object."""
        return InferUnifaceDetectionParam()


class InferUnifaceDetection(dataprocess.CObjectDetectionTask):
    """
    Class that implements the algorithm.
    Inherits PyCore.CWorkflowTask or derived from Ikomia API.
    """

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)

        # Add image output for visualization
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferUnifaceDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.names = ["face"]
        self.model_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights")
        self.detector = None
        self.blurrer = None

    def _load_model(self):
        """Load the detector model using the refactored model loader."""
        param = self.get_param_object()

        # Create detector with model weights saved in self.model_folder
        self.detector = create_detector(
            model_type=param.model_name,
            model_folder=self.model_folder,
            confidence_threshold=param.conf_thres,
            nms_threshold=param.nms_thres,
        )

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def get_progress_steps(self):
        """
        Ikomia Studio only.
        Function returning the number of progress steps for this algorithm.
        This is handled by the main progress bar of Ikomia Studio.
        """
        return 1

    def run(self):
        """Main function and entry point for algorithm execution."""
        # Call begin_task_run() for initialization
        self.begin_task_run()
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Convert RGBA to RGB if necessary
        if src_image.shape[-1] == 4:  # RGBA format
            src_image = src_image[:, :, :3]  # Keep only RGB channels

        # Load model if needed
        if self.detector is None or param.update:
            self._load_model()

        faces = self.detector.detect(src_image)

        if faces is None:
            self.end_task_run()
            return

        self.set_names(self.names)
        # Process each detected face
        for i, face in enumerate(faces):
            # Extract bounding box coordinates
            # bbox format is typically [x1, y1, x2, y2] or [x, y, w, h]
            bbox = face.bbox
            confidence = face.confidence

            x1, y1, x2, y2 = bbox
            w = float(x2 - x1)
            h = float(y2 - y1)

            # Add object: (object_id, class_id, confidence, x, y, width, height)
            self.add_object(i+1, 0, float(confidence),
                            float(x1), float(y1), w, h)

        # Create anonymized output with blurred faces (if enabled)
        if param.output_anonymized:
            # Initialize or update blurrer if needed
            if self.blurrer is None or param.update:
                self.blurrer = BlurFace(
                    method=param.blur_method,
                    blur_strength=param.blur_strength,
                    pixel_blocks=param.pixel_blocks,
                    margin=param.margin
                )

            # Apply anonymization
            anonymized_image = self.blurrer.anonymize(src_image.copy(), faces)
            img = np.array(anonymized_image)
            output = self.get_output(2)
            output.set_image(img)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


class InferUnifaceDetectionFactory(dataprocess.CTaskFactory):
    """
    Factory class to create process object.
    Inherits PyDataProcess.CTaskFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_uniface_detection"
        self.info.short_description = "Lightweight production-ready face analysis library built on ONNX Runtime."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Yakhyokhuja Valikhujaev"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "MIT License"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.16.0"

        # URL of documentation
        self.info.documentation_link = "https://yakhyo.github.io/uniface/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_uniface_detection"
        self.info.original_repository = "https://github.com/yakhyo/uniface"

        # Keywords used for search
        self.info.keywords = "uniface, detection, face detection"

        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        """Instantiate algorithm object."""
        return InferUnifaceDetection(self.info.name, param)
