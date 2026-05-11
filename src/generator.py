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
from .text_metrics import count_chinese_chars

VIDEO_DIRECTOR_SYSTEM_MESSAGE = (
    "你是一位世界级多模态视频导演和视频生成提示词架构师，"
    "擅长按给定概念自然融合不同模态与语义角色；"
    "只显式表达被选中的概念，不主动补写未选中的大类。"
)


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
        concepts = self.sampling.get('concepts', {})
        challenge_elements = self.sampling.get('challenge_elements', [])
        selection_trace = self.sampling.get('selection_trace', {})
        return {
            'prompt_id': self.prompt_id,
            'combination_id': self.combination_id,
            'difficulty': self.difficulty,
            'concepts': concepts,
            'challenge_elements': challenge_elements,
            'selection_trace': selection_trace,
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
                max_output_tokens=65536,
                temperature=0.7
            )
        )
        
        # Check for truncation
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                finish_reason_name = getattr(finish_reason, 'name', str(finish_reason))
                if finish_reason_name == 'MAX_TOKENS':
                    print(f"Warning: Response truncated due to max tokens limit")
        
        return self._extract_text_from_response(response)

    def _extract_text_from_response(self, response) -> str:
        """Extract only text parts and ignore non-text parts like thought_signature.

        google-genai's response.text convenience property warns when the model
        returns non-text parts. Direct part traversal avoids that noisy warning
        while preserving the actual generated text.
        """
        texts = []
        for candidate in getattr(response, 'candidates', []) or []:
            content = getattr(candidate, 'content', None)
            for part in getattr(content, 'parts', []) or []:
                text = getattr(part, 'text', None)
                if text:
                    texts.append(text)

        result = ''.join(texts).strip()
        if not result:
            raise ValueError("Gemini response contains no text parts")
        return result


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
                {"role": "system", "content": VIDEO_DIRECTOR_SYSTEM_MESSAGE},
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
                {"role": "system", "content": VIDEO_DIRECTOR_SYSTEM_MESSAGE},
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
    """DeepSeek provider through Baidu Qianfan OpenAI-compatible API."""

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
                {"role": "system", "content": VIDEO_DIRECTOR_SYSTEM_MESSAGE},
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
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        active_llms: List[str],
        dimensions_config: Optional[List[Dict[str, Any]]] = None,
    ):
        self.providers: Dict[str, LLMProvider] = {}
        self.active_llms = active_llms
        self.dimensions_config = dimensions_config or []
        self.category_names = {
            item.get('key'): item.get('sheet', item.get('key'))
            for item in self.dimensions_config
            if item.get('key')
        }
        self.category_names.update({
            'subject': self.category_names.get('subject', '主体'),
            'motion': self.category_names.get('motion', '运动'),
            'scene': self.category_names.get('scene', '场景'),
            'audio': self.category_names.get('audio', '音频类型'),
        })
        
        for llm_name in active_llms:
            if llm_name in llm_config:
                provider_class = self.PROVIDERS.get(llm_name)
                if provider_class:
                    provider = provider_class(llm_config[llm_name])
                    if provider.is_available():
                        self.providers[llm_name] = provider
    
    def _build_system_prompt(self, combination: Any) -> str:
        """构建发送给LLM的系统提示"""

        # System prompt - professional multimodal video director.
        system_prompt = (
            "你是一位世界级多模态视频导演和视频生成提示词架构师。"
            "你同时擅长摄影调度、场景构图、动作设计、空间组织、节奏设计、"
            "多模态表达、视觉特效、物理模拟和动画原理。"
            "你的任务是把离散的概念元素转化为一个连贯、专业、可直接用于视频生成的中文提示词。"
            "每个给定概念都应被视为同等重要的创作素材，并按照它最自然的语义角色融入画面："
            "实体、动作、环境、声音、风格、镜头、文字、交互或未来新增类目，都只在被选中时按自身语义表达。"
            "未被选中的大类不应被主动补写成显式内容。"
            "你需要让概念彼此服务，形成一个自然完整的视频场面，而不是机械罗列或强行拼接。"
        )

        selected_category_keys = set(combination.concepts.keys())
        selected_category_lines = [
            f"- {self.category_names.get(key, key)} ({key})"
            for key in combination.concepts.keys()
        ]
        unselected_category_lines = [
            f"- {self.category_names.get(item.get('key'), item.get('key'))} ({item.get('key')})"
            for item in self.dimensions_config
            if item.get('key') and item.get('key') not in selected_category_keys
        ]
        if not unselected_category_lines:
            unselected_category_lines = ["- 无"]
        
        # Extract concept information
        concepts_info = []
        for cat_key, concept in combination.concepts.items():
            cat_name = self.category_names.get(cat_key, cat_key)
            path_text = ' > '.join(concept.level3_path) if concept.level3_path else concept.level3_category
            if concept.leaf:
                concepts_info.append(
                    f"- {cat_name}：{path_text} > 叶子候选：{concept.leaf}"
                )
            else:
                concepts_info.append(
                    f"- {cat_name}：{path_text}"
                )
        
        # Challenge elements
        challenge_info = []
        if combination.challenge_elements:
            challenge_info.append("已选挑战性要素（必须与概念自然兼容，并在最终文本中可观察表达）:")
            for elem in combination.challenge_elements:
                reason = elem.get('reason', '')
                reason_text = f"；选择理由：{reason}" if reason else ""
                challenge_info.append(
                    f"- {elem.get('id', '')} | {elem.get('name', '')}: "
                    f"{elem.get('description', '')}{reason_text}"
                )
        
        # Difficulty constraints
        params = combination.difficulty_params
        
        length_constraint = self._build_length_constraint(params)
        
        modifier_constraint = f"使用 {params.modifier_count_min}-{params.modifier_count_max} 个必要的具象修饰，不要堆砌空泛形容词"

        # Build the user prompt
        user_prompt = f"""请根据下面的概念素材，生成一段专业的中文视频生成提示词。

需要融合的概念素材:
{chr(10).join(concepts_info)}

已选大类白名单:
{chr(10).join(selected_category_lines)}

未选大类禁止清单:
{chr(10).join(unselected_category_lines)}

{chr(10).join(challenge_info) if challenge_info else ''}

创作约束:
- 难度等级：{combination.difficulty_level.upper()}
- 必须严格按照概念素材的完整路径理解语义，不能只抓叶子词；若叶子词存在多义性或跨类联想，必须服从父级类别。例如“食品 > 鱼类及海鲜 > 鳕鱼”应表现为食材或食品形态，不得写成海中游动、跃出水面或具有自主行为的活体鳕鱼。
- 完整路径只用于理解语义边界，不要求逐字写出 level1/level2；最终文本应优先自然表达 level3/leaf，不要把“特定功能地点、公共室内、物体运动、环境音、互动音”等父级类目标签直接写成画面内容。
- 若概念包含“叶子候选”，这些候选是同一三级概念下的可选具体表达。默认只选择其中一个最兼容当前场景的 leaf 写入最终文本；不要罗列多个 leaf 候选，也不要把多个候选合并成复合场景，否则会导致视频核心要素不集中。
- 所有概念都需要以可感知、可验证的方式自然出现，并具有主次层级；核心概念作为画面焦点，辅助概念通过动作、材质、位置、声音、光线、镜头或空间关系轻量融入，不要平均堆砌。
- 不得删除、替换、歪曲或过度扩写概念；每个概念必须保持原始中文语义，不能为了连贯性引入偏离原义的新概念。
- 每个概念应按其自然语义角色出现：食品类表现为食材、食品、加工对象、摆放对象、烹饪对象或食用场景内容；音乐类表现为声音来源或听觉氛围；自然现象类主要作为时间、光线、环境或氛围线索。
- 最终提示词只能显式表达“已选大类白名单”中的大类；“未选大类禁止清单”中的大类不得以主体、动作、环境、声音、风格、镜头、文字、交互、氛围、背景补充等形式出现。
- 为了连贯性可以加入少量连接细节，但连接细节必须服务于已选概念，或仅承担空间、时间、动作衔接作用，不能引入未选大类的可感知内容，也不能新增未选中的具体场所、道具或子场景来替代父级类目。
- 画面必须有一个明确主焦点，并包含明确的时间推进；至少出现一个可见变化过程，如主体移动、形态变化、光影变化、材质状态变化、环境扰动、镜头运动或物体交互。
- 每段提示词最多设置一个核心动作链，体现“起始状态 → 变化过程 → 结果反馈”，不要只写静态构图、静态质感或名词并列。
- 优先通过已选概念之间的动作、因果和状态变化完成融合，不要仅靠并列摆放满足概念覆盖。
- 镜头描述必须连续，动作主体进入画面时要交代位置关系，如“从画外伸入”“从前景掠过”“在背景中靠近”；不要突然切换无法衔接的视角、距离或空间位置。
- 所有动作、声音和材质反应必须符合物体物理属性；软质、湿润、轻薄物体不得产生金属、玻璃、硬币等硬质碰撞声，除非画面中存在明确硬质声源。
- 所有声音必须有明确来源：来自画面内可见物体、合理画外物体，或明确写成主观听觉；若来自画外，必须交代“画外”“远处”“身旁”“背景中”等来源，不得出现无来源音效。
- 若多个概念难以由同一主体合理承载，应采用“分属不同对象”“前后动作衔接”“画内/画外分离”“材质来源明确化”等方式融合，不要强行制造概念冲突。
- 避免使用无法被画面稳定呈现的抽象评价词，如“高级感”“震撼感”“神秘感”“不可思议感”；如需表达氛围，必须通过具体光线、动作、声音、材质或空间关系呈现。
- 不要添加与已选概念无直接服务关系的剧情反转、人物心理、装饰性细节或额外道具。
- 输出前执行白名单检查和路径语义复核：若某个短语不能服务于已选概念，或会让读者感知到未选大类，或使叶子词落入其他大类语义，应删除或重写。
- 已选挑战性要素不是装饰建议，而是必须在最终文本中可观察表达的控制项；每个 challenge 至少对应一个具体画面或声音证据，如动作链、交互动作、物理变化、空间遮挡、镜头变化或因果反馈。
- 不要只写挑战名称或抽象词，例如“复杂轨迹、互动感、物理模拟、时间推进”；必须把它们写成具体事件。多个 challenge 可以共用同一个核心动作链承载，但不能完全缺失。
- {modifier_constraint}
{f'- {length_constraint}' if length_constraint else ''}

质量要求:
1. 概念保真与聚焦：概念素材准确、完整，核心概念有合理视觉或语义焦点。
2. 内部一致性：避免同一时空下互斥的物理、时间、空间、镜头或语义设定。
3. 清晰度与具象化：已选概念和必要画面信息要具体，避免抽象堆词和关键词沙拉。
4. 语言质量：中文流畅，无严重错别字，不要无序混杂其他语言。
5. 视频生成可用性：适合直接喂给视频模型，不要写成纯静态图片提示词，也不要塞入整部电影剧情。

输出规则:
- 只输出一段中文提示词。
- 直接从画面描述开始。
- 不要使用 Markdown、加粗、标题、标签、编号、JSON、代码块或换行。
- 不要输出解释、备注、翻译、备选方案、自查过程或字数统计。
- 只输出最终提示词文本。

提示词:"""

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
                        'challenge_elements': combination.challenge_elements,
                        'selection_trace': combination.selection_trace or {},
                        'phase': combination.phase,
                    },
                    text=text,
                    text_length=count_chinese_chars(text),
                    created_at=datetime.now().isoformat()
                )
                results.append(generated)
                
            except Exception as e:
                print(f"生成失败 ({llm_name}): {e}")
        
        return results
    
    def get_available_providers(self) -> List[str]:
        """获取可用的LLM列表"""
        return list(self.providers.keys())

    def _build_length_constraint(self, params: Any) -> str:
        min_len = params.text_length_min
        max_len = params.text_length_max

        if min_len is not None and max_len is not None:
            return f"长度要求：正文控制在 {min_len}-{max_len} 个中文汉字之间，含上下限，按汉字数计算"
        if max_len is not None:
            return f"长度要求：正文不超过 {max_len} 个中文汉字，按汉字数计算"
        if min_len is not None:
            return f"长度要求：正文不少于 {min_len} 个中文汉字，按汉字数计算"
        return ""
