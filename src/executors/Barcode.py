import sys
import os
import cv2
import numpy as np

from pyzbar.pyzbar import decode as zbar_decode

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor
from capsules.Barcode.src.utils.response import build_response_qreader
from capsules.Barcode.src.models.PackageModel import PackageModel, Detection
from capsules.Barcode.src.configs.config import ALLOWED_BARCODES

class BarcodeReader(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**self.request.data)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def process_detections(self, image_array: np.ndarray, img_uid: str):

        if image_array.dtype != np.uint8:image_array = image_array.astype(np.uint8)
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        decoded_objects = zbar_decode(gray)
        detection_list = []

        for obj in decoded_objects:
            if obj.type not in ALLOWED_BARCODES:
                continue
            x, y, w, h = obj.rect

            try:
                decoded_text = obj.data.decode("utf-8")
            except Exception:
                decoded_text = "Decode failed"

            detection_list.append(
                Detection(
                    boundingBox=BoundingBox(
                        left=x,
                        top=y,
                        width=w,
                        height=h
                    ),
                    confidence=1.0,
                    classLabel=obj.type,
                    data=decoded_text,
                    classId=0,
                    imgUID=img_uid,
                )
            )

        return detection_list

    def run(self):
        image_obj = Image.get_frame(img=self.request.get_param("inputImage"),redis_db=self.redis_db)
        self.detection = self.process_detections(np.array(image_obj.value),image_obj.uID)
        return build_response_qreader(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
