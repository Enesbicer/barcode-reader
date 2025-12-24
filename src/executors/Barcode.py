import os
import sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_barcode

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor
from capsules.BarcodeDetection.src.utils.response import build_response
from capsules.BarcodeDetection.src.models.PackageModel import PackageModel, Detection


class BarcodeReader(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**self.request.data)
        self.image_param = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def read_barcode(self, image_array: np.ndarray):
        """Detect and decode Barcode from an image array using pyzbar."""
        if len(image_array.shape) == 3:
            gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image_array

        detected_barcodes = decode_barcode(gray_img)

        if not detected_barcodes:
            return [], image_array

        decoded_outputs = []
        for barcode in detected_barcodes:
            try:
                decoded_text = barcode.data.decode("utf-8")
                barcode_type = barcode.type

                if not decoded_text:
                    decoded_text = "Could not read data from Barcode"
            except Exception as e:
                print(f"Decode failed: {e}")
                decoded_text = "Decode failed"
                barcode_type = "Unknown"

            (x, y, w, h) = barcode.rect
            bbox_xyxy = [x, y, x + w, y + h]

            decoded_outputs.append(
                {
                    "data": decoded_text,
                    "type": barcode_type,
                    "bbox": bbox_xyxy,
                    "confidence": 1.0,
                }
            )

        return decoded_outputs, image_array

    def build_detection_list(self, detection_outputs, img_uid):
        """Build detection results in the expected format."""
        detection_list = []

        if not detection_outputs:
            detection = Detection(
                boundingBox=None,
                confidence= -1,
                classLabel="Barcode",
                data="No barcode detected",
                classId=0,
                imgUID=img_uid,
            )
            detection_list.append(detection)
        else:
            for output in detection_outputs:
                bbox_coords = output["bbox"]

                bbox = BoundingBox(
                    left=bbox_coords[0],
                    top=bbox_coords[1],
                    width=bbox_coords[2] - bbox_coords[0],
                    height=bbox_coords[3] - bbox_coords[1],
                )

                full_data_string = f"{output['type']}: {output['data']}"

                detection = Detection(
                    boundingBox=bbox,
                    confidence=output["confidence"],
                    classLabel="Barcode",
                    data=full_data_string,
                    classId=1,
                    imgUID=img_uid,
                )
                detection_list.append(detection)

        return detection_list

    def detection_inference(self, image_model: Image):
        """Perform the Barcode detection and formatting."""
        detection_output, updated_image = self.read_barcode(np.array(image_model.value))
        image_model.value = updated_image
        return self.build_detection_list(detection_output, image_model.uID)

    def run(self):
        """Main function to run capsule."""
        image = Image.get_frame(img=self.image_param, redis_db=self.redis_db)
        self.detection = self.detection_inference(image)
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()