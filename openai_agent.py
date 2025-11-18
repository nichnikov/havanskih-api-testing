
import os
import re
import time
import pandas as pd
from openai import OpenAI
from itertools import groupby
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()

class GPT_Validator:
    client = OpenAI(
    api_key=os.getenv("API_KEY"), # ваш ключ в VseGPT после регистрации
    base_url=os.getenv("BASE_URL"),)


    def gpt_validation(self, init_prompt: str, dialogue: str):
        prompt = init_prompt.format(str(dialogue))

        messages = []
        messages.append({"role": "user", "content": prompt})

        response_big = self.client.chat.completions.create(
            model="openai/gpt-4o-mini", # id модели из списка моделей - можно использовать OpenAI, Anthropic и пр. меняя только этот параметр openai/gpt-4o-mini
            messages=messages,
            temperature=0.7,
            n=1,
            max_tokens=3000, # максимальное число ВЫХОДНЫХ токенов. Для большинства моделей не должно превышать 4096
            extra_headers={ "X-Title": "My App" }, # опционально - передача информация об источнике API-вызова
        )

        #print("Response BIG:",response_big)
        return response_big.choices[0].message.content

    def __call__(self, p:str, d: str):
        return self.gpt_validation(p, d)


if __name__ == "__main__":

    prompt1 = """Ты - опытный специалист службы контроля качества работы колл-центра экспертной поддержки.
            Тебе для анализа передали диалог Оператора с Пользователем: 
            Текст диалога: {}
            
            Оцени качество работы оператора. Остался ли Пользователь доволен ответами, не выражал ли Пользователь неудовольствие.
            Выдай аргументированный ответ. В конце прими решение, достоин ли Оператор новогодней премии. 
            Лишать Оператора премии можно только в случае явного недовольства пользователя.
            Напиши резолюцию
            """

    prompt2 = """Ты - мудрый и опытный руководитель колл-центра экспертной поддержки. 
            Ты получил сообщение от Специалиста службы контроля качества, который оценил работу твоих сотрудников.
            Твоя задача быть справедливым к своим сотрудникам и штрафовать Операторов только в тех случаях, 
            когда в докладе явно указано на недовольство Пользователя ответами Оператора.
            
            Доклад Специалиста отдела по оценке качества сотрудников колл-центра:
            {}
            
            Внимательно прочитай доклад и прими решение о том, необходимо ли штрафовать Оператора.

            Напиши краткую резолюцию, без объяснений: 
            ### Не штрафовать Оператора / Оштрафовать Оператора
            """
    
    fale_names = ["chats_with_autophrases.csv", "chats_without_autophrases.csv"]
    validator = GPT_Validator()

    for fale_name in fale_names:

        data_df = pd.read_csv(os.path.join("data", fale_name), sep="\t")

        patterns = re.compile(r"\n|\¶|(?P<url>https?://[^\s]+)|<a href=|</a>|/#/document/\d\d/\d+/|\"\s*\">|\s+")

        for col in ["chat_id", "text"]:
            data_df[col] = data_df[col].apply(lambda x: patterns.sub(" ", str(x)))
        
        data_df["discriminator"] = data_df["discriminator"].apply(lambda x: re.sub(r"\s+", "", str(x))) 
        data_df["Autor"] = "Нет"
        
        user_messages = ["UserMessage", "UserNewsPositiveReactionMessage"]
        operator_messages = ["AutoGoodbyeMessage", "AutoHello2Message",  "AutoHelloMessage", "AutoHelloNewsMessage", "AutoHelloOfflineMessage",
                             "AutoRateMessage", "HotlineNotificationMessage", "MLRoboChatMessage", "NewsAutoMessage", "OperatorMessage"]

        print(data_df["discriminator"].unique())
        data_df["Autor"][data_df["discriminator"].isin(user_messages)] = "Пользователь"
        data_df["Autor"][data_df["discriminator"].isin(operator_messages)] = "Оператор"
        
        data_dics = data_df[["chat_id", "Autor", "text"]].to_dict(orient="records")

        # группировка текстов по чатам:
        data_dics.sort(key=itemgetter('chat_id'))
        dict_of_chats = {int(k): [{"Autor": d["Autor"], "Phrase": d["text"]} for d in list(g)] for k, g in groupby(data_dics, itemgetter("chat_id"))}

        dict_results = []
        k = 1

        
        for i in dict_of_chats:
            try:
                dialogue = "\n\t".join([str(d["Autor"]) + ": " + str(d["Phrase"]) for d in dict_of_chats[i]])

                print(dialogue)

                if k > 5000:
                    break
                
                k += 1

                cheking_report = validator(prompt1, dialogue)
                val = validator(prompt2, cheking_report)
                print(k, "/", len(dict_of_chats), "val:", val, "\n\n")
                dict_results.append({"chat_id": i, "dialogue": dialogue, "val": val})

                out_fn = "ai_agent_" + fale_name
                results_df = pd.DataFrame(dict_results)
                results_df.to_csv(os.path.join("results", out_fn), sep="\t", index=False)
            
            except Exception as e:
                pass
            
            time.sleep(1)