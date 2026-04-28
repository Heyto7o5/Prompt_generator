"""
LLM Prompt生成模块
"""
import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class GeneratedPrompt:
    """生成的Prompt"""
    prompt_id: str
    combination_id: str
    llm_provider: str
    llm_model: str
    difficulty: Dict[str, Any]
    sampling: Dict[str, Any]
    text: str
    text_length: int
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'prompt_id': self.prompt_id,
            'combination_id': self.combination_id,
            'difficulty': self.difficulty,
            'sampling': self.sampling,
            'text': self.text,
            'text_length': self.text_length,
            'llm': {
                'provider': self.llm_provider,
                'model': self.llm_model
            },
            'metadata': {
                'created_at': self.created_at
            }
        }


class LLMProvider(ABC):
    """LLM提供者基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', '')
        self.api_key = os.environ.get(config.get('api_key_env', ''), '')
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """生成prompt文本"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class GeminiProvider(LLMProvider):
    """Gemini提供者 - 支持API Key和服务账号认证"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self.service_account_path = config.get('service_account_path', '')
    
    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                
                # 优先使用服务账号认证 (Vertex AI)
                if self.service_account_path and os.path.exists(self.service_account_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.service_account_path
                    self._client = genai.Client(
                        vertexai=True,
                        project=self._get_project_id(),
                        location="global"
                    )
                elif self.api_key:
                    # 使用标准API Key方式
                    self._client = genai.Client(api_key=self.api_key)
                elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                    # 使用环境变量中的服务账号
                    self._client = genai.Client(
                        vertexai=True,
                        project=self._get_project_id_from_env(),
                        location="global"
                    )
                else:
                    raise ValueError("需要设置GEMINI_API_KEY或GOOGLE_APPLICATION_CREDENTIALS环境变量")
                    
            except ImportError:
                raise ImportError("请安装 google-genai: pip install google-genai")
        return self._client
    
    def _get_project_id(self) -> str:
        """从服务账号文件中获取project_id"""
        if self.service_account_path and os.path.exists(self.service_account_path):
            with open(self.service_account_path, 'r') as f:
                data = json.load(f)
                return data.get('project_id', '')
        return ''
    
    def _get_project_id_from_env(self) -> str:
        """从环境变量中的服务账号文件获取project_id"""
        env_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if env_creds and os.path.exists(env_creds):
            with open(env_creds, 'r') as f:
                data = json.load(f)
                return data.get('project_id', '')
        return ''
    
    def is_available(self) -> bool:
        # 检查API Key
        if self.api_key:
            return True
        # 检查服务账号文件
        if self.service_account_path and os.path.exists(self.service_account_path):
            return True
        # 检查环境变量中的服务账号
        env_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if env_creds and os.path.exists(env_creds):
            return True
        return False
    
    def generate(self, system_prompt: str) -> str:
        from google.genai import types
        client = self._get_client()
        
        response = client.models.generate_content(
            model=self.model,
            contents=system_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.7
            )
        )
        
        # Check for truncation
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason.name == 'MAX_TOKENS':
                    print(f"Warning: Response truncated due to max tokens limit")
        
        return response.text.strip()


class QwenProvider(LLMProvider):
    """通义千问提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, system_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a world-class, all-round visual director and prompt architect specializing in video generation prompts."},
                {"role": "user", "content": system_prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        
        # Check for truncation
        if response.choices[0].finish_reason == 'length':
            print(f"Warning: Response truncated due to max tokens limit")
        
        return response.choices[0].message.content.strip()


class GPTProvider(LLMProvider):
    """GPT提供者，支持 OpenAI 原生接口和 Azure OpenAI 部署"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self.api_type = config.get('api_type', 'openai').lower()
        self.base_url = config.get('base_url', '')
        self.azure_endpoint = self._normalize_azure_endpoint(config.get('azure_endpoint', ''))
        self.api_version = config.get('api_version', '2024-02-15-preview')
        self.deployment_name = config.get('deployment_name', self.model)
        self.max_tokens = config.get('max_tokens', 8192)
        self.temperature = config.get('temperature', 0.7)

    def _normalize_azure_endpoint(self, endpoint: str) -> str:
        """AzureOpenAI 需要资源根地址，兼容旧脚本里写成完整 completions URL 的情况。"""
        if not endpoint:
            return ''
        if '/openai/' in endpoint:
            endpoint = endpoint.split('/openai/', 1)[0]
        return endpoint.rstrip('/')
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI, OpenAI
                if self.api_type == 'azure':
                    self._client = AzureOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint,
                    )
                else:
                    kwargs = {'api_key': self.api_key}
                    if self.base_url:
                        kwargs['base_url'] = self.base_url
                    self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def is_available(self) -> bool:
        if self.api_type == 'azure':
            return bool(self.api_key and self.azure_endpoint and self.deployment_name)
        return bool(self.api_key)
    
    def generate(self, system_prompt: str) -> str:
        client = self._get_client()
        model_name = self.deployment_name if self.api_type == 'azure' else self.model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a world-class, all-round visual director and prompt architect specializing in video generation prompts."},
                {"role": "user", "content": system_prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Check for truncation
        if response.choices[0].finish_reason == 'length':
            print(f"Warning: Response truncated due to max tokens limit")
        
        return response.choices[0].message.content.strip()


class DpskProvider(LLMProvider):
    """DeepSeek V3 provider through Baidu Qianfan OpenAI-compatible API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://qianfan.baidubce.com/v2')
        self.default_headers = config.get('default_headers', {})
        self.max_tokens = config.get('max_tokens', 8192)
        self.temperature = config.get('temperature', 0.7)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers=self.default_headers,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def generate(self, system_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a world-class, all-round visual director and prompt architect specializing in video generation prompts."},
                {"role": "user", "content": system_prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if response.choices[0].finish_reason == 'length':
            print(f"Warning: Response truncated due to max tokens limit")

        return response.choices[0].message.content.strip()


class PromptGenerator:
    """Prompt生成器"""
    
    PROVIDERS = {
        'gemini': GeminiProvider,
        'qwen': QwenProvider,
        'gpt': GPTProvider,
        'dpsk': DpskProvider,
    }
    
    def __init__(self, llm_config: Dict[str, Any], active_llms: List[str]):
        self.providers: Dict[str, LLMProvider] = {}
        self.active_llms = active_llms
        
        for llm_name in active_llms:
            if llm_name in llm_config:
                provider_class = self.PROVIDERS.get(llm_name)
                if provider_class:
                    provider = provider_class(llm_config[llm_name])
                    if provider.is_available():
                        self.providers[llm_name] = provider
    
    def _build_system_prompt(self, combination: Any) -> str:
        """构建发送给LLM的系统提示"""

        # System prompt - Professional audiovisual director
        system_prompt = """You are a world-class multimodal video director, sound designer, and prompt architect for video generation. You are equally strong in cinematography, scene composition, motion design, sound staging, voice and speech direction, ambient audio, music cues, audio-visual synchronization, visual effects, physical simulation, and animation principles. Your task is to transform fragmented concept elements into a coherent video prompt. Every concept in the mandatory set has equal priority and must be explicitly expressed in its natural modality: subjects as visible entities, motions as concrete actions or changes, scenes as environments, audio concepts as audible sounds or speech, and any future category according to its semantic role."""

        # Category mapping to English
        category_names = {
            'subject': 'Subject',
            'motion': 'Motion',
            'scene': 'Scene',
            'audio': 'Audio'
        }
        
        # Extract concept information
        concepts_info = []
        for cat_key, concept in combination.concepts.items():
            cat_name = category_names.get(cat_key, cat_key)
            if concept.leaf:
                concepts_info.append(
                    f"- {cat_name} (mandatory): {concept.leaf} "
                    f"(Category: {concept.level3_category})"
                )
            else:
                concepts_info.append(
                    f"- {cat_name} (mandatory): {concept.level3_category}"
                )
        
        # Challenge elements
        challenge_info = []
        if combination.challenge_elements:
            challenge_info.append("Challenge Requirements (must be reflected in the output):")
            for elem in combination.challenge_elements:
                challenge_info.append(f"- {elem['name']}: {elem.get('description', '')}")
        
        # Difficulty constraints
        params = combination.difficulty_params
        
        length_constraint = ""
        if params.text_length_max:
            length_constraint = f"Text length: maximum {params.text_length_max} words"
        elif params.text_length_min:
            length_constraint = f"Text length: minimum {params.text_length_min} words"
        
        modifier_constraint = f"Use {params.modifier_count_min}-{params.modifier_count_max} descriptive adjectives/modifiers"

        # Build the user prompt
        user_prompt = f"""Generate one professional video generation prompt from the mandatory concept set below.

Mandatory Concept Set:
{chr(10).join(concepts_info)}

{chr(10).join(challenge_info) if challenge_info else ''}

Constraints:
- Difficulty Level: {combination.difficulty_level.upper()}
- Every selected concept above is mandatory and must be explicitly expressed in the final prompt
- DO NOT omit, weaken, replace, or generalize away any selected concept
- If the concepts feel difficult to combine, rewrite the scene so that all of them are still present
- Treat all selected categories as equal-priority requirements
- Express each selected concept in its appropriate modality; for example, audio concepts must be clearly audible when selected
- Do not require unselected categories, but natural incidental details are allowed if they do not obscure the selected concepts
- {modifier_constraint}
{f'- {length_constraint}' if length_constraint else ''}

Guidelines:
1. Create a coherent, specific, and visually descriptive prompt
2. Avoid vague adjectives (e.g., "beautiful", "amazing", "stunning")
3. Describe the action, motion, or event concretely
4. Include camera movement or shot type if appropriate
5. Ensure physical plausibility and visual clarity
6. Translate any Chinese concept names to appropriate English equivalents
7. Every selected category must be explicitly represented in the final text
8. Subject-like concepts must be visible entities or unmistakable visual objects
9. Motion-like concepts must be concrete actions, events, or visual changes
10. Scene-like concepts must be stated as the environment or location
11. Audio-like concepts must be explicitly audible, such as sound, voice, speech, broadcast, music, ambient noise, narration, or spoken content
12. For any future category, infer its semantic role and express it explicitly rather than omitting it
13. A response is invalid if any selected concept is missing, only implied, or replaced by a looser concept

Self-check before answering:
- Does the prompt explicitly include every selected mandatory concept?
- Is each selected concept expressed in the correct modality (visual, action, spatial, audible, or another relevant modality)?
- Are the challenge requirements reflected when provided?
- Is the result still a single natural-sounding video prompt?

Output rules:
- Output exactly one plain English paragraph
- Start directly with the scene description
- Do NOT use Markdown formatting, bold text, headings, labels, bullets, numbering, JSON, code blocks, or line breaks
- Do NOT include field labels such as "Prompt:", "Subject:", "Motion:", "Scene:", "Audio:", "Concepts:", or "Description:"
- Do NOT output explanations, notes, alternatives, translations, or analysis
- Output ONLY the final prompt text

Prompt:"""

        return system_prompt + "\n\n" + user_prompt
    
    def generate(self, combination: Any) -> List[GeneratedPrompt]:
        """为一个组合生成prompt"""
        results = []
        system_prompt = self._build_system_prompt(combination)
        
        for llm_name in self.active_llms:
            if llm_name not in self.providers:
                print(f"警告: LLM {llm_name} 不可用，跳过")
                continue
            
            provider = self.providers[llm_name]
            
            try:
                text = provider.generate(system_prompt)
                
                prompt_id = f"P-{len(results) + 1:05d}"
                
                generated = GeneratedPrompt(
                    prompt_id=prompt_id,
                    combination_id=combination.combination_id,
                    llm_provider=llm_name,
                    llm_model=provider.model,
                    difficulty={
                        'level': combination.difficulty_level.upper()
                    },
                    sampling={
                        'combination_id': combination.combination_id,
                        'categories_selected': list(combination.concepts.keys()),
                        'concepts': {k: v.to_dict() for k, v in combination.concepts.items()},
                        'challenge_elements': combination.challenge_elements
                    },
                    text=text,
                    text_length=len(text.split()),
                    created_at=datetime.now().isoformat()
                )
                results.append(generated)
                
            except Exception as e:
                print(f"生成失败 ({llm_name}): {e}")
        
        return results
    
    def get_available_providers(self) -> List[str]:
        """获取可用的LLM列表"""
        return list(self.providers.keys())
