# Import ray serve and FastAPI libraries
from ray import serve
from typing import Literal
from fastapi import FastAPI

# library for pre-trained models
from transformers import pipeline
# INFO: Ray Serve is a framework for serving ML applications
# A Ray Serve Cluster is made up of one or more Applications.
# An Application is made up of one more DEPLOYMENTS that work together.
# Key characteristics are:
#
# INFO: Applications
# APPLICATIONS are coarse-grained units of functionality:
#   - They can be independently upgraded without affecting other Applications.
#   - They provide isolation and separate deployment lifecycles.
#
# INFO: Deployments
# A DEPLOYMENT is the fundamental building block in Ray Serve's architecture.
# Deployments enable:
#   - Separation of concerns (e.g different models, business logic, data transformations, etc.)
#   - Independent scaling, including AUTOSCALING capabilities
#
# INFO: Replicas
# A REPLICA is a worker process (Ray Actor) with its own request processing queue.
# Replicas offer flexible configuration options:
#   - Specifiy its own hardware resources (CPU, GPU, memory, etc.)
#   - Specify its own runtime environment (e.g. libraries)
#   - Maintain state (e.g models)

# INFO: Architecture
# We would like to setup a FastAPI app to serve HTTP requests. This app is embedded
# into a Ray Serve deployment.
# Ray Serve takes care of routing incoming requests to several replicas of the same model

# Define a FastAPI app
app = FastAPI()


# NOTE: Define a Ray Serve deployment with the `deployment` decorator
@serve.deployment(num_replicas=2)  # num_replicas specifies the number of replicas
@serve.ingress(app)  # This allows the FastAPI app to be served by Ray Serve
class MySentimentModel:
    def __init__(
        self,
        task: Literal["summarization"] | Literal["text-classification"],
        model: str,
    ):
        # NOTE: We load a pre-trained model from HuggingFace
        self.model = pipeline(task=task, model=model)

    @app.post(
        "/predict"
    )  # we define this method as an endpoint, exposed by the FastAPI app
    def predict(
        self, text: str
    ):  # WARN: `text` is a required query parameter since it is not Pydantic model
        """
        Summarize the given text using the model.

        Args:
            text (str): The text to summarize.

        Returns:
            The summarized text.

        """
        result = self.model(text)
        return result


# NOTE: We bind the deployment to the Ray Serve runtime
serve.run(
    MySentimentModel.bind(
        task="text-classification",
    )
)
