
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from modules.prompt import SYSTEM_PROMPT, CLINCIAL_PROMPT
import requests

@dataclass
class RefinementPayload:
    candidate_reports: List[str]
    trusted_descriptions: List[str]
    disease_votes: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMRefiner:
    def refine_batch(self, payloads: List[RefinementPayload]) -> List[str]:
        return [self.refine_one(p) for p in payloads]
    def refine_one(self, payload: RefinementPayload) -> str:
        raise NotImplementedError

class NullRefiner(BaseLLMRefiner):
    def refine_one(self, payload: RefinementPayload) -> str:
        return payload.candidate_reports[0].strip() if payload.candidate_reports else ''

class OpenAICompatibleRefiner(BaseLLMRefiner):
    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None,
                 timeout: int = 120, temperature: float = 0.0, max_tokens: int = 256,
                 cache_path: Optional[str] = None):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.cache[obj['key']] = obj['value']
                    except Exception:
                        pass

    def _prompt_messages(self, payload: RefinementPayload):
        trusted = ''.join(f'- {x}' for x in payload.trusted_descriptions) if payload.trusted_descriptions else '- None'
        candidates = ''.join([f'Observation {i+1}: {r}' for i, r in enumerate(payload.candidate_reports)])
        votes = ''
        if payload.disease_votes:
            votes = ''.join(f'- {k}: {v}' for k, v in payload.disease_votes.items())
        user = CLINCIAL_PROMPT.format(Candidate_Clinical_Observations=candidates, 
                                      Trusted_Disease_Classes=trusted, 
                                      Consensus_Disease_Status_Hints=votes if votes else '- None')
    
        return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]

    def _cache_key(self, payload: RefinementPayload):
        raw = json.dumps({
            'candidate_reports': payload.candidate_reports,
            'trusted_descriptions': payload.trusted_descriptions,
            'disease_votes': payload.disease_votes,
            'model': self.model,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _write_cache(self, key: str, value: str):
        if not self.cache_path:
            return
        folder = os.path.dirname(self.cache_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '')

    def refine_one(self, payload: RefinementPayload) -> str:
        key = self._cache_key(payload)
        if key in self.cache:
            return self.cache[key]
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        body = {
            'model': self.model,
            'messages': self._prompt_messages(payload),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        resp = requests.post(self.api_base + '/chat/completions', headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data['choices'][0]['message']['content'].strip()
        self.cache[key] = text
        self._write_cache(key, text)
        return text

class LocalHFRefiner(BaseLLMRefiner):
    def __init__(self, model_name: str, device_map: str = 'auto', max_new_tokens: int = 256, temperature: float = 0.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def refine_one(self, payload: RefinementPayload) -> str:
        helper = OpenAICompatibleRefiner(api_base='http://local.invalid', model='local')
        prompt = ''
        for m in helper._prompt_messages(payload):
            prompt += f"[{m['role'].upper()}]{m['content']}"
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                        temperature=max(self.temperature, 1e-5), return_full_text=False)
        return out[0]['generated_text'].strip()


def build_llm_refiner(args):
    provider = getattr(args, 'llm_refiner_provider', 'none')
    if provider == 'none':
        return NullRefiner()
    if provider == 'openai_compatible':
        return OpenAICompatibleRefiner(
            api_base=args.llm_api_base,
            model=args.llm_model_name,
            api_key=getattr(args, 'llm_api_key', ''),
            timeout=getattr(args, 'llm_timeout', 120),
            temperature=getattr(args, 'llm_temperature', 0.0),
            max_tokens=getattr(args, 'llm_max_tokens', 256),
            cache_path=getattr(args, 'llm_cache_path', None),
        )
    if provider == 'local_hf':
        return LocalHFRefiner(
            model_name=args.llm_model_name,
            device_map=getattr(args, 'llm_device_map', 'auto'),
            max_new_tokens=getattr(args, 'llm_max_tokens', 256),
            temperature=getattr(args, 'llm_temperature', 0.0),
        )
    raise ValueError(f'Unsupported llm_refiner_provider: {provider}')
