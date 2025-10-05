from sagemaker.pytorch import PyTorchModel
import sagemaker


def deploy_endpoint():
    sagemaker.Session()
    role = "arn:aws:iam::142464893686:role/sentiment-model-deploy-endpoint-role"
    model_uri = "s3://sentiment-model-saas/inference/model.tar.gz"

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )


if __name__ == "__main__":
    deploy_endpoint()
