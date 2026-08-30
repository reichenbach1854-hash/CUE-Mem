from abc import ABC, abstractmethod

from scripts.common.llm import message_content_to_text, openai_client, usage_value

from benchmark.security import safe_runtime_error

class BaseLLM(ABC):
    """
    Provides a convenient interface to utilize the powerful capability of different large language models.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def fast_run(self, query):
        pass

class APILLM(BaseLLM):
    """
    Utilize LLM from APIs.
    """
    def __init__(self, config):
        super().__init__(config)

        self.client = openai_client(
            api_key=getattr(self.config, 'api_key', None) or None,
            base_url=getattr(self.config, 'base_url', None) or None,
        )

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': message_content_to_text(response.choices[0].message.content),
            'input_token': usage_value(getattr(response, 'usage', None), 'prompt_tokens'),
            'output_token': usage_value(getattr(response, 'usage', None), 'completion_tokens'),
        }

    def run(self, message_list):
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=message_list,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            raise safe_runtime_error('LLM request failed', exc) from None
        response = self.parse_response(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']

class LocalVLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM, SamplingParams
        self.model = LLM(config.name)

        self.sampling_params = SamplingParams(temperature=config.temperature)
    
    def run(self, message_list):
        return self.model.chat(message_list,self.sampling_params)[0]
    
    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response.outputs[0].text
