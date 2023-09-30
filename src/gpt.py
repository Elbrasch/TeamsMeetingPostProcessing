
from structlog import get_logger
import openai
import json

class Gpt:
    def __init__(self, config) -> None:
        self.config = config
        self.logger = get_logger(__file__)
        self.deployment_name = self.config["openai-deployment-name"]
        openai.api_type = "azure"
        openai.api_version = self.config["openai-api-version"]
        openai.api_base = self.config["openai-api-base"]
        openai.api_key = self.config["openai-api-key"]
        self.history = []
        self.context = """Assistant is an AI chatbot that helps users answer questions about imperfect transcripts of video conference meetings. If you can't answer a question based on the context, you say so. Unknown Speaker is a placeholder when no speaker could be determined and can represent multiple speakers.\nTranscript:\n"""
        

    def chat(self, context, question):
        self.history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": self.context + context}] + self.history
        response = openai.ChatCompletion.create(
        engine=self.deployment_name, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ], temperature=0
        )
        self.history.append({"role": "assistant", "content": response['choices'][0]['message']['content'].strip("\n").strip().strip("\n")})
        return self.history[-1]['content']

    def completion(self, context, question):
        response = openai.ChatCompletion.create(
        engine=self.deployment_name, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ], temperature=0
        )
        return response['choices'][0]['message']['content'].strip("\n").strip().strip("\n")
    
    def speaker_detection(self, transcript):
        context = self.context + transcript
        question = "Identify the names of the different Speakers based on context except Unknown Speaker. Return as a json dictionary. Use 'Unidentified' for speakers you can't identify"
        speakers_raw = self.completion(context, question)
        self.logger.info(speakers_raw)
        try:
            speakers_raw = json.loads(speakers_raw)
            print(speakers_raw)
            speakers = dict()
            for speaker, replacement in speakers_raw.items():
                if replacement == "Unknown Speaker" or replacement == "Not mentioned" or replacement== "Unidentified":
                    continue
                speakers[speaker] = replacement           

        except Exception as e:
            self.logger.error(e)
            speakers = {}
        return speakers




if __name__ == "__main__":
    import json
    with open("config.json") as f:
        config = json.load(f)
    model = Gpt(config)
    context = "You are a helpfull AI Assistant"
    question = "say hello"
    print(model.completion(context, question))