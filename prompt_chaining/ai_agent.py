from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import transformers
from typing import Optional
class PromptChainingAgent:
    def __init__(self, pipeline: transformers.pipeline, chroma_db: Chroma, embedding_model: HuggingFaceEmbeddings, response: Optional[str] = None): # type: ignore
        self.pipeline = pipeline
        self.chroma_db = chroma_db
        self.embedding_model = embedding_model
        self.response = response if response is not None else ""
        self.init_chain = False
        
    def _message_template(self, rag_search: bool, system_prompt: str, user_prompt_or_response: str, search_k: int = 1):
        retriever = self.chroma_db.as_retriever(search_kwargs={"k": search_k})
        
        if rag_search and system_prompt and user_prompt_or_response:
            docs = retriever.get_relevant_documents(user_prompt_or_response)
            context = "\n\n".join([doc.page_content for doc in docs])
                                    
            print("context", context)
                                    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "text", "content": user_prompt_or_response},
                {"role": "document", "content": context},
            ]
        
        elif system_prompt and user_prompt_or_response:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "text", "content": user_prompt_or_response}
            ]
            
        else:
            raise ValueError("system_prompt과 user_prompt_or_response 중 하나 이상이 필요합니다.")
        
        return messages
        
    def init_generate_chain(self, system_prompt: str, user_prompt: str, rag_search: bool = False, search_k: Optional[int] = None):
        self.init_chain = True

        # 메세지 생성
        messages = self._message_template(rag_search, system_prompt, user_prompt, search_k) # type: ignore
        
        prompt = self.pipeline.tokenizer.apply_chat_template( # type: ignore
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 종료 토큰 설정
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # 결과 출력 (프롬프트 부분 제거)
        response = outputs[0]["generated_text"][len(prompt):].strip()
        self.response = response

        print("init_generate_chain LLM 응답:", response)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        return self
    
    def semi_generate_chain(self, system_prompt: str, rag_search: bool = False, search_k: Optional[int] = None):
        if not self.init_chain:
            raise RuntimeError("init_generate_chain()을 먼저 호출해야 합니다.")

        # 채팅 메시지 구성
        messages = self._message_template(rag_search, system_prompt, self.response, search_k) # type: ignore

        # 토크나이저 적용
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 종료 토큰 설정
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5,
            top_p=0.5
        )
        
        # 결과 출력 (프롬프트 부분 제거)
        response = outputs[0]["generated_text"][len(prompt):].strip()
        self.response = response
        
        print("semi_generate_chain LLM 응답:", response)
        print("==========================================================")
        
        return self
        
    def generate_response(self):
        if not self.init_chain:
            raise RuntimeError("init_generate_chain()을 먼저 호출해야 합니다.")
        if self.response is None:
            raise RuntimeError("실행되었지만 응답이 없습니다.")
        
        print("generate_response LLM 응답:", self.response)
        return self.response