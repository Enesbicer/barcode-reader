
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


class PyzbarInputs(Inputs):
    inputImage: InputImage


class PyzbarOutputs(Outputs):
    outputData: OutputData


class PyzbarRequest(Request):
    inputs: Optional[PyzbarInputs]


    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PyzbarResponse(Response):
    outputs: PyzbarOutputs


class PyzbarExecutor(Config):
    name: Literal["Pyzbar"] = "Pyzbar"
    value: Union[PyzbarRequest, PyzbarResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Pyzbar"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PyzbarExecutor]
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
    name: Literal["BarcodeReader"] = "BarcodeReader"
