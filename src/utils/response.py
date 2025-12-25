
from sdks.novavision.src.helper.package import PackageHelper
from capsules.BarcodeReader.src.models.PackageModel import PackageModel, OutputData, ConfigExecutor, PackageConfigs, PyzbarExecutor, PyzbarOutputs, PyzbarResponse

def build_response(context):
    outputData = OutputData(value=context.detection)
    Outputs = PyzbarOutputs(outputData=outputData)
    packageResponse = PyzbarResponse(outputs=Outputs)
    packageExecutor = PyzbarExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel