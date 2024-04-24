from groq import Groq
import os
import dotenv
from dotenv import load_dotenv
load_dotenv()
class GroqModels():
    def __init__(self,model):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model=model
    def stream(self,prompt,sys_prompt="you are a helpful AI Bot"):
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
        {
            "role": "user",
            "content":prompt 
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)
        for chunk in completion:
            yield chunk.choices[0].delta.content or ""


if __name__=="__main__":
    model=GroqModels("llama3-8b-8192")
    for i in model.stream("who are ya"):
        print(i)

