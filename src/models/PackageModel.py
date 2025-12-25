
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Detection


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class Detection(Detection):
    data: str
    imgUID: str


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detection"


class BarcodeInputs(Inputs):
    inputImage: InputImage


class BarcodeOutputs(Outputs):
    outputData: OutputData


class BarcodeRequest(Request):
    inputs: Optional[BarcodeInputs]
    configs: BarcodeConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class BarcodeResponse(Response):
    outputs: BarcodeOutputs


class BarcodeExecuter(Config):
    name: Literal["Barcode"] = "Barcode"
    value: Union[BarcodeRequest, BarcodeResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Barcode"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[BarcodeExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Barcode"] = "Barcode"
