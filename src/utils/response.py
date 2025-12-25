
from sdks.novavision.src.helper.package import PackageHelper
from components.Package.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PackageOutputs, PackageResponse, PackageExecutor, OutputImage


def build_response(context):
    outputData = OutputData(value=context.detection)
    Outputs = PyzbarOutputs(outputImage=outputData)
    packageResponse = PyzbarResponse(outputs=Outputs)
    packageExecutor = PyzbarExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel