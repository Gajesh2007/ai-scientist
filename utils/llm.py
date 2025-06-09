"""utils.llm

Unified, production-ready abstraction layer for working with multiple Large-Language-Model (LLM)
providers through a single, cohesive interface.

Key features
------------
1. Uniform API for the official OpenAI, Anthropic Claude and any OpenAI-compatible router such as
   OpenRouter.
2. Support for standard chat completions, *function-calling* workflows and strongly-typed structured
   outputs (powered by the `instructor` library and Pydantic models).
3. Sensible defaults: automatic API-key discovery via environment variables, configurable
   time-outs/retries, and provider-specific rate-limit handling.
4. `UnifiedLLM` facade that lets you add several providers at runtime and switch between them with a
   single parameter.

Quick-start
~~~~~~~~~~~
>>> from utils.llm import UnifiedLLM, Message
>>> llm = UnifiedLLM()
>>> llm.add_provider("openai", default=True)   # uses $OPENAI_API_KEY
>>> resp = llm.complete([Message(role="user", content="Write a haiku about the ocean")])
>>> print(resp.content)

Module structure
~~~~~~~~~~~~~~~~
LLMProvider   – Enum of supported providers
ModelType     – Enum of canonical model identifiers
LLMConfig     – Configuration dataclass (shared across providers)
BaseLLM       – Abstract base with common helpers
OpenAILLM / AnthropicLLM / OpenRouterLLM – concrete provider implementations
LLMFactory    – Simple factory for the above classes
UnifiedLLM    – High-level convenience wrapper used by most client code
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Type, Callable
from abc import ABC, abstractmethod
from enum import Enum
import logging
from dataclasses import dataclass, field
from datetime import datetime

import openai
import anthropic
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class ModelType(Enum):
    """Model types with their capabilities."""
    # OpenAI Models - Latest
    GPT_4_1 = "gpt-4.1"  # Latest GPT-4.1 series (April 2025)
    GPT_4_1_MINI = "gpt-4.1-mini"  # Mini version of 4.1
    O4_MINI = "o4-mini"  # Latest mini reasoning model (April 2025)
    O3 = "o3"  # Latest reasoning model 
    GPT_4O = "gpt-4o"  # Latest version: 2024-11-20

    # Anthropic Models - Latest Claude 4 and 3.x series
    CLAUDE_4_OPUS = "claude-opus-4-20250514"  # Latest and most capable (May 2025)
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"  # High performance Claude 4 (May 2025)
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"  # Claude 3.7 with extended thinking (Feb 2025)

    # OpenRouter Models - Latest models available
    OR_CLAUDE_4_OPUS = "anthropic/claude-opus-4-20250514"
    OR_CLAUDE_4_SONNET = "anthropic/claude-sonnet-4-20250514"
    OR_CLAUDE_37_SONNET = "anthropic/claude-3-7-sonnet-20250219"
    OR_GPT_4O = "openai/gpt-4o"
    OR_O3_MINI = "openai/o3-mini"
    OR_QWEN_3_235B = "qwen/qwen-3-235b"  # Latest Qwen 3 (2025)
    OR_GEMINI_25_FLASH = "google/gemini-2.5-flash"  # Latest Gemini 2.5
    OR_GEMINI_25_PRO = "google/gemini-2.5-pro-preview" # Gemini 2.5 Pro (June 2025)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    default_model: Optional[ModelType] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.api_key:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.OPENROUTER:
                self.api_key = os.getenv("OPENROUTER_API_KEY")
                self.base_url = self.base_url or "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError(f"API key not found for {self.provider.value}. "
                           f"Please set the environment variable or provide it in config.")


@dataclass
class Message:
    """Unified message format."""
    role: str  # "system", "user", "assistant"
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMResponse:
    """Unified response format."""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, int]] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    structured_output: Optional[Any] = None
    raw_response: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Send a completion request to the LLM."""
        pass
    
    @abstractmethod
    def complete_structured(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Get a structured response using Pydantic models."""
        pass
    
    @abstractmethod
    def complete_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete with function calling support."""
        pass
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert unified messages to provider format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"function_call": msg.function_call} if msg.function_call else {}),
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {})
            }
            for msg in messages
        ]


class OpenAILLM(BaseLLM):
    """OpenAI implementation."""
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        return OpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
    
    def complete(
        self,
        messages: List[Message],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """OpenAI completion."""
        model = model or self.config.default_model or ModelType.GPT_4O
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Check if this is a reasoning model
        is_reasoning_model = model.value in ["o3", "o3-mini", "o4-mini"]
        
        # Prepare kwargs
        completion_kwargs = {
            "model": model.value,
            "messages": self._convert_messages(messages),
            **kwargs
        }
        
        # Reasoning models have limitations
        if not is_reasoning_model:
            completion_kwargs["temperature"] = temperature
            if max_tokens:
                completion_kwargs["max_tokens"] = max_tokens
        
        try:
            if stream:
                response = self.client.chat.completions.create(
                    **completion_kwargs,
                    stream=True
                )
                # For streaming, return a generator wrapped in LLMResponse
                return response
            else:
                response = self.client.chat.completions.create(**completion_kwargs)
                
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider=LLMProvider.OPENAI,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None,
                    function_call=response.choices[0].message.function_call.__dict__ if response.choices[0].message.function_call else None,
                    tool_calls=[tc.__dict__ for tc in response.choices[0].message.tool_calls] if response.choices[0].message.tool_calls else None,
                    raw_response=response
                )
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise
    
    def complete_structured(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Get structured output using instructor."""
        model = model or self.config.default_model or ModelType.GPT_4O
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Use instructor for structured outputs
        instructor_client = instructor.from_openai(self.client)
        
        try:
            structured_response = instructor_client.chat.completions.create(
                model=model.value,
                messages=self._convert_messages(messages),
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=structured_response.model_dump_json(indent=2),
                model=model.value,
                provider=LLMProvider.OPENAI,
                structured_output=structured_response,
                raw_response=structured_response
            )
        except Exception as e:
            logger.error(f"OpenAI structured completion error: {e}")
            raise
    
    def complete_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete with function calling."""
        model = model or self.config.default_model or ModelType.GPT_4O
        
        # Note: O1/O3/O4 reasoning models don't support function calling
        if model.value in ["o3", "o3-mini", "o4-mini"]:
            raise ValueError(f"Model {model.value} does not support function calling")
        
        completion_kwargs = {
            "model": model.value,
            "messages": self._convert_messages(messages),
            "functions": functions,
            "temperature": temperature if temperature is not None else self.config.temperature,
            **kwargs
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        if function_call is not None:
            completion_kwargs["function_call"] = function_call
        
        try:
            response = self.client.chat.completions.create(**completion_kwargs)
            
            message = response.choices[0].message
            return LLMResponse(
                content=message.content or "",
                model=response.model,
                provider=LLMProvider.OPENAI,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                function_call=message.function_call.__dict__ if message.function_call else None,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenAI function calling error: {e}")
            raise


class AnthropicLLM(BaseLLM):
    """Anthropic implementation."""
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        return Anthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
    
    def _convert_messages_anthropic(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert messages to Anthropic format (system message separate)."""
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, anthropic_messages
    
    def complete(
        self,
        messages: List[Message],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Anthropic completion."""
        model = model or self.config.default_model or ModelType.CLAUDE_4_SONNET
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens or 4096
        
        system_message, anthropic_messages = self._convert_messages_anthropic(messages)
        
        try:
            req_kwargs = dict(
                model=model.value,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if system_message is not None:
                req_kwargs["system"] = system_message

            if stream:
                response = self.client.messages.create(stream=True, **req_kwargs)
                return response
            else:
                response = self.client.messages.create(**req_kwargs)
                
                return LLMResponse(
                    content=response.content[0].text if response.content else "",
                    model=response.model,
                    provider=LLMProvider.ANTHROPIC,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    } if hasattr(response, 'usage') else None,
                    raw_response=response
                )
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise
    
    def complete_structured(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Get structured output using instructor with Anthropic."""
        model = model or self.config.default_model or ModelType.CLAUDE_4_SONNET
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens or 4096
        
        # Use instructor for structured outputs
        instructor_client = instructor.from_anthropic(self.client)
        
        system_message, anthropic_messages = self._convert_messages_anthropic(messages)
        
        try:
            req_kwargs = dict(
                model=model.value,
                messages=anthropic_messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if system_message is not None:
                req_kwargs["system"] = system_message

            structured_response = instructor_client.messages.create(**req_kwargs)
            
            return LLMResponse(
                content=structured_response.model_dump_json(indent=2),
                model=model.value,
                provider=LLMProvider.ANTHROPIC,
                structured_output=structured_response,
                raw_response=structured_response
            )
        except Exception as e:
            logger.error(f"Anthropic structured completion error: {e}")
            raise
    
    def complete_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Anthropic function calling using tool use."""
        model = model or self.config.default_model or ModelType.CLAUDE_4_SONNET
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens or 4096
        
        # Convert functions to Anthropic tool format
        tools = []
        for func in functions:
            tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {})
            })
        
        system_message, anthropic_messages = self._convert_messages_anthropic(messages)
        
        try:
            req_kwargs = dict(
                model=model.value,
                messages=anthropic_messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if system_message is not None:
                req_kwargs["system"] = system_message

            response = self.client.messages.create(**req_kwargs)
            
            # Extract tool calls from response
            tool_calls = []
            content_text = ""
            
            for content in response.content:
                if content.type == "text":
                    content_text += content.text
                elif content.type == "tool_use":
                    tool_calls.append({
                        "id": content.id,
                        "type": "function",
                        "function": {
                            "name": content.name,
                            "arguments": json.dumps(content.input)
                        }
                    })
            
            return LLMResponse(
                content=content_text,
                model=response.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                } if hasattr(response, 'usage') else None,
                tool_calls=tool_calls if tool_calls else None,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Anthropic function calling error: {e}")
            raise


class OpenRouterLLM(BaseLLM):
    """OpenRouter implementation using OpenAI-compatible API."""
    
    def _initialize_client(self):
        """Initialize OpenRouter client using OpenAI SDK."""
        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://openrouter.ai/api/v1",
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://github.com/gaj-ai-scientist"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "GAJ AI Scientist")
            }
        )
    
    def complete(
        self,
        messages: List[Message],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """OpenRouter completion."""
        model = model or self.config.default_model or ModelType.OR_CLAUDE_4_SONNET
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        completion_kwargs = {
            "model": model.value,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        
        try:
            if stream:
                response = self.client.chat.completions.create(
                    **completion_kwargs,
                    stream=True
                )
                return response
            else:
                response = self.client.chat.completions.create(**completion_kwargs)
                
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider=LLMProvider.OPENROUTER,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None,
                    raw_response=response
                )
        except Exception as e:
            logger.error(f"OpenRouter completion error: {e}")
            raise
    
    def complete_structured(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Get structured output using instructor with OpenRouter."""
        model = model or self.config.default_model or ModelType.OR_CLAUDE_4_SONNET
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Use instructor for structured outputs
        instructor_client = instructor.from_openai(self.client)
        
        try:
            structured_response = instructor_client.chat.completions.create(
                model=model.value,
                messages=self._convert_messages(messages),
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=structured_response.model_dump_json(indent=2),
                model=model.value,
                provider=LLMProvider.OPENROUTER,
                structured_output=structured_response,
                raw_response=structured_response
            )
        except Exception as e:
            logger.error(f"OpenRouter structured completion error: {e}")
            raise
    
    def complete_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        model: Optional[ModelType] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete with function calling through OpenRouter."""
        model = model or self.config.default_model or ModelType.OR_GPT_4O
        
        # For OpenRouter, we should use models that support function calling
        # Check if the model supports it (GPT models typically do)
        if not any(provider in model.value for provider in ["openai/", "gpt"]):
            logger.warning(f"Model {model.value} may not support function calling on OpenRouter")
        
        completion_kwargs = {
            "model": model.value,
            "messages": self._convert_messages(messages),
            "functions": functions,
            "temperature": temperature if temperature is not None else self.config.temperature,
            **kwargs
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        if function_call is not None:
            completion_kwargs["function_call"] = function_call
        
        try:
            response = self.client.chat.completions.create(**completion_kwargs)
            
            message = response.choices[0].message
            return LLMResponse(
                content=message.content or "",
                model=response.model,
                provider=LLMProvider.OPENROUTER,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                function_call=message.function_call.__dict__ if message.function_call else None,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenRouter function calling error: {e}")
            raise


class LLMFactory:
    """Factory class to create LLM instances."""
    
    @staticmethod
    def create(provider: Union[LLMProvider, str], **config_kwargs) -> BaseLLM:
        """Create an LLM instance based on provider."""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        config = LLMConfig(provider=provider, **config_kwargs)
        
        if provider == LLMProvider.OPENAI:
            return OpenAILLM(config)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicLLM(config)
        elif provider == LLMProvider.OPENROUTER:
            return OpenRouterLLM(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Convenience class for easy usage
class UnifiedLLM:
    """Unified interface to work with multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLM] = {}
        self._default_provider: Optional[LLMProvider] = None
    
    def add_provider(
        self,
        provider: Union[LLMProvider, str],
        api_key: Optional[str] = None,
        default: bool = False,
        **config_kwargs
    ) -> None:
        """Add a provider configuration."""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.providers[provider] = LLMFactory.create(
            provider=provider,
            api_key=api_key,
            **config_kwargs
        )
        
        if default or self._default_provider is None:
            self._default_provider = provider
    
    def get_provider(self, provider: Optional[Union[LLMProvider, str]] = None) -> BaseLLM:
        """Get a specific provider or the default one."""
        if provider is None:
            if self._default_provider is None:
                raise ValueError("No default provider set")
            return self.providers[self._default_provider]
        
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured")
        
        return self.providers[provider]
    
    def complete(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        provider: Optional[Union[LLMProvider, str]] = None,
        model: Optional[Union[ModelType, str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete using specified or default provider."""
        # Convert dict messages to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) for msg in messages]
        
        # Convert string model to ModelType
        if isinstance(model, str):
            model = ModelType(model)
        
        llm = self.get_provider(provider)
        return llm.complete(messages, model=model, **kwargs)
    
    def complete_structured(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        response_model: Type[BaseModel],
        provider: Optional[Union[LLMProvider, str]] = None,
        model: Optional[Union[ModelType, str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Get structured output."""
        # Convert dict messages to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) for msg in messages]
        
        # Convert string model to ModelType
        if isinstance(model, str):
            model = ModelType(model)
        
        llm = self.get_provider(provider)
        return llm.complete_structured(messages, response_model, model=model, **kwargs)
    
    def complete_with_functions(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        functions: List[Dict[str, Any]],
        provider: Optional[Union[LLMProvider, str]] = None,
        model: Optional[Union[ModelType, str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete with function calling."""
        # Convert dict messages to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) for msg in messages]
        
        # Convert string model to ModelType
        if isinstance(model, str):
            model = ModelType(model)
        
        llm = self.get_provider(provider)
        return llm.complete_with_functions(messages, functions, model=model, **kwargs)
    
    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        provider: Optional[Union[LLMProvider, str]] = None,
        model: Optional[Union[ModelType, str]] = None,
        **kwargs
    ):
        """Stream responses."""
        # Convert dict messages to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) for msg in messages]
        
        # Convert string model to ModelType
        if isinstance(model, str):
            model = ModelType(model)
        
        llm = self.get_provider(provider)
        return llm.complete(messages, model=model, stream=True, **kwargs)