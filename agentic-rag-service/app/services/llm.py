"""LLM service using AWS Bedrock for DSPy."""

import boto3
import dspy
from typing import Optional
from loguru import logger
from app.config import settings


class BedrockLM(dspy.LM):
    """DSPy LM wrapper for AWS Bedrock."""

    def __init__(
        self,
        model: str,
        region_name: str = "us-east-1",
        **kwargs,
    ):
        """Initialize Bedrock LM."""
        super().__init__(model=model, **kwargs)

        self.model = model
        self.region_name = region_name
        self.kwargs = kwargs

        # Initialize Bedrock client
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.AWS_ACCESS_KEY_ID else None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.AWS_SECRET_ACCESS_KEY else None,
        )

        logger.info(f"Initialized Bedrock LM with model: {model}")

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """
        Call Bedrock model with prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        try:
            # Prepare request based on model type
            if "anthropic.claude" in self.model:
                response = self._call_claude(prompt, **kwargs)
            elif "meta.llama" in self.model:
                response = self._call_llama(prompt, **kwargs)
            elif "amazon.titan" in self.model:
                response = self._call_titan(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported model: {self.model}")

            return [response]

        except Exception as e:
            logger.error(f"Error calling Bedrock: {e}")
            raise

    def _call_claude(self, prompt: str, **kwargs) -> str:
        """Call Claude models."""
        import json

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Prepare request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
        }

        # Invoke model
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
        )

        # Parse response
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _call_llama(self, prompt: str, **kwargs) -> str:
        """Call Llama models."""
        import json

        request_body = {
            "prompt": prompt,
            "max_gen_len": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
        }

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
        )

        response_body = json.loads(response["body"].read())
        return response_body["generation"]

    def _call_titan(self, prompt: str, **kwargs) -> str:
        """Call Titan models."""
        import json

        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
                "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
            },
        }

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
        )

        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]


class LLMService:
    """Service for LLM operations."""

    def __init__(self):
        self.lm: Optional[BedrockLM] = None
        self._initialize()

    def _initialize(self):
        """Initialize LLM with Bedrock."""
        logger.info(f"Initializing Bedrock LLM with model: {settings.LLM_MODEL}")

        self.lm = BedrockLM(
            model=settings.LLM_MODEL,
            region_name=settings.AWS_REGION,
        )

        # Configure DSPy to use this LM
        dspy.settings.configure(lm=self.lm)

        logger.info("Bedrock LLM initialized and configured for DSPy")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        responses = self.lm(prompt, **kwargs)
        return responses[0] if responses else ""


# Global LLM service instance
llm_service = LLMService()
