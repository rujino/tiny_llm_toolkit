import json
from tiny_llm_toolkit.prompt_chaining.ai_agent import PromptChainingAgent
from transformers import Pipeline
from typing import Optional

class JSONPromptChainingAgent(PromptChainingAgent):
    def _to_json(self, data):
        try:
            return json.dumps({"response": data})
        except (TypeError, ValueError) as e:
            return None

    def _generate_with_retry(self, generate_func, *args, max_attempts=3, **kwargs):
        for attempt in range(max_attempts):
            generate_func(*args, **kwargs)
            json_response = self._to_json(self.response)
            if json_response is not None:
                print(f"JSON 변환 성공, 재시도 {attempt + 1}/{max_attempts}")
                return self
            print(f"JSON 변환 실패, 재시도 {attempt + 1}/{max_attempts}")
        raise ValueError("JSON 변환 실패: 최대 시도 횟수 초과")

    def init_generate_chain(self, pipeline: Pipeline, system_prompt: str, user_prompt: str, rag_search: bool = False, search_k: Optional[int] = None, max_attempts: int = 3):
        return self._generate_with_retry(super().init_generate_chain, pipeline, system_prompt, user_prompt, rag_search, search_k, max_attempts=max_attempts)

    def semi_generate_chain(self, pipeline: Pipeline, system_prompt: str, rag_search: bool = False, search_k: Optional[int] = None, max_attempts: int = 3):
        return self._generate_with_retry(super().semi_generate_chain, pipeline, system_prompt, rag_search, search_k, max_attempts=max_attempts)
    
    def init_generate_chain_without_json(self, pipeline: Pipeline, system_prompt: str, user_prompt: str, rag_search: bool = False, search_k: Optional[int] = None):
        # 부모 클래스의 init_generate_chain 메소드 호출
        return super().init_generate_chain(pipeline, system_prompt, user_prompt, rag_search, search_k)
    
    def semi_generate_chain_without_json(self, pipeline: Pipeline, system_prompt: str, rag_search: bool = False, search_k: Optional[int] = None):
        # 부모 클래스의 semi_generate_chain 메소드 호출
        return super().semi_generate_chain(pipeline, system_prompt, rag_search, search_k)