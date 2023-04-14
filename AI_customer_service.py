import os
from dotenv import load_dotenv
load_dotenv()
import re, json, time
import pandas as pd
from datetime import datetime
from func_timeout import func_timeout
import jieba
import openai
import tiktoken
from db import DBhelper
from log import logger
from AI_customer_service_utils import translation_stw
from likr_Search_engine import Search_engine

class ChatGPT_AVD:
    def __init__(self):
        self.OPEN_AI_KEY_DICT = eval(os.getenv('OPENAI_API_KEY'))

    def get_keys(func):
        def inner(self,message, model="gpt-3.5-turbo", timeout=60):
            # get token_id
            query = 'SELECT id, counts FROM web_push.AI_service_token_counter x ORDER BY counts limit 1;'
            if 'gpt-4' in model:
                query = 'x WHERE id < 6'.join(query.split('x'))
            token_id = DBhelper('jupiter_new').ExecuteSelect(query)[0][0]

            # update token counter
            DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts + 1 WHERE id = {token_id}')
            try:
                res = func_timeout(timeout, func, (self, message, token_id, model))
            except Exception as e:
                res = 'timeout'
            DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts - 1 WHERE id = {token_id}')
            return res
        return inner

    @get_keys
    def ask_gpt(self, message, token_id=None, model="gpt-3.5-turbo"):
        openai.api_key = self.OPEN_AI_KEY_DICT[token_id]
        if type(message) == str:
            message = [{'role': 'user', 'content': message}]
        completion = openai.ChatCompletion.create(model=model, messages=message)
        return completion['choices'][0]['message']['content']

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo-0301":
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo")
        elif model == "gpt-4":
            return self.num_tokens_from_messages(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def get_gpt_query(self, result, message, history, web_id_conf):
        '''
        :param query: result from likr_search
        :param query: question for chatgpt
        -------
        chatgpt_query
            Results:

            [1] "result[0]['htmlTitle']}",snippet= "{result[0]['snippet']}",description = "{result[0]['snippet']}"

            [2] "{g[1]['snippet']}"
            URL: "{g[1]['link']}"

            [3] "{g[2]['snippet']}"
            URL: "{g[2]['link']}"

            Current date: {date}

            Instructions: If you are "{web_id_conf['web_name']}" customer service. Using the information of results or following the flow of conversation, write a comprehensive reply to the given query in 繁體中文 and following the rules below:
            Always cite the information from the provided results using the [number] notation in the end of that sentence.
            Write Bullet list for each subject if you recommend products.
            "親愛的顧客您好，" in the beginning.
            "祝您愉快！" in the end.

            Query: {message}
        '''
        date = datetime.today().strftime('%Y/%m/%d')
        gpt_query = history if history else [{"role": "system", "content": f"我們是{web_id_conf['web_name']}(代號：{web_id_conf['web_id']},官方網站：{web_id_conf['web_url']}),{web_id_conf['description']}"}]
        if type(result) != str:
            linkList = []
            chatgpt_query = """\nResults:"""
            for v in result:
                if not v.get('link') or len(linkList) == 3:
                    continue
                url = v.get('link')
                url = re.search(r'.+detail/[\w\-]+/', url).group(0) if re.search(r'.+detail/[\w\-]+/', url) else url
                print('搜尋結果:\t', url)
                if url in linkList:
                    continue
                linkList.append(url)
                if v.get('htmlTitle'):
                    chatgpt_query += f"""\n\n[{len(linkList)}] "{v.get('htmlTitle')}"""
                if v.get('snippet'):
                    chatgpt_query += f""",snippet = "{v.get('snippet')}"""
                if v.get('pagemap') and v.get('pagemap').get('metatags') and v.get('pagemap').get('metatags')[0].get('og:description'):
                    chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
                chatgpt_query += f"""\nURL: "{url}" """
            chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: If you are "{web_id_conf['web_name']}" customer service. Using the information of results or following the flow of conversation, write a comprehensive reply to the given query in 繁體中文 and following the rules below:\nAlways cite the information from the provided results using the [number] notation in the end of that sentence.\nWrite Bullet list for each subject if you recommend products.\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {message}"""
        else:
            chatgpt_query = f"""\n\n\nCurrent date: {date}\n\nInstructions: If you are the brand, "{web_id_conf['web_name']}"({web_id_conf['web_id']}) customer service and there is no search result in product list, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {message}"""
        gpt_query += [{'role': 'user', 'content': chatgpt_query}]
        while self.num_tokens_from_messages(gpt_query) > 3500 and len(gpt_query) > 3:
            gpt_query = [gpt_query[0]] + gpt_query[3:]
        return gpt_query, linkList

class QA_api:
    def __init__(self, frontend, logger):
        self.CONFIG = self.get_config()
        self.ChatGPT = ChatGPT_AVD()
        self.Search = Search_engine()
        self.logger = logger
        self.frontend = frontend
        if frontend == 'line':
            self.table_suffix = '_api'
            self.url_format = lambda x: ' ' + x + ' '
        elif frontend == 'slack':
            self.table_suffix = ''
            self.url_format = lambda x: '<' + x + '|查看更多>'

    def get_config(self):
        '''
        Returns {web_id: config from jupiter_new -> web_push.AI_service_config}
        '''
        config_dict = {}
        config = DBhelper('jupiter_new').ExecuteSelect("SELECT * FROM web_push.AI_service_config where mode != 0;")
        config_col = [i[0] for i in
                      DBhelper('jupiter_new').ExecuteSelect("SHOW COLUMNS FROM web_push.AI_service_config;")]
        for conf in config:
            config_dict[conf[1]] = {}
            for k, v in zip(config_col, conf):
                config_dict[conf[1]][k] = v
        return config_dict

    def check_message_length(self, message: str, length: int=50):
        for url in re.findall(r'https?:\/\/[\w\.\-\/\?\=\+&#$%^;%_]+', message):
            if self.fetch_url_response(url):
                message = message.replace(url, '')
        return len(message) <= length

    def get_question_keyword(self, message: str, web_id: str) -> list:
        forbidden_words = {'client_msg_id', '我', '你', '妳', '們', '沒', '怎', '麼', '要', '沒有',
                           '^如何$', '^賣$', '^有$', '^可以$', '暢銷', '^商品$', '熱賣', '特別', '最近', '幾天', '常常'}
        # remove web_id from message
        message = translation_stw(message).lower()
        for i in [web_id, self.CONFIG[web_id]['web_name']] + eval(self.CONFIG[web_id]['other_name']):
            message = re.sub(i, '', message)
            for j in list(jieba.cut(i)):
                message = re.sub(j, '', message)
        # segmentation
        reply = self.ChatGPT.ask_gpt([{'role': 'system', 'content': '你會將user的content進行切詞,再依重要性評分數,若存在商品名詞,商品名詞的分數為原本的兩倍。並且只回答此格式為 {"詞":分數} ,不需要其他解釋。'},
                                      {'role': 'user', 'content': f'{message}'}],
                                     model='gpt-4')
        if reply == 'timeout': return 'timeout'
        keyword_list = [k for k, _ in sorted(eval(reply).items(), key=lambda x: x[1], reverse=True) if k in message and not any(re.search(w, k) for w in forbidden_words)]
        if not keyword_list:
            keyword_list = [self.ChatGPT.ask_gpt([{'role': 'system', 'content': '你會將user的content選擇一個最重要的關鍵字。並且只回答此格式 "關鍵字" ,不需要其他解釋'},
                                                  {'role': 'user', 'content': f'{message}'}],
                                                 model='gpt-4')]
        return keyword_list

    def reset_result_order(self, result_search, result_recommend, flags, web_id):
        qa_url = [i.strip('*') for i in self.CONFIG[web_id]['qa_url'].split(',')]
        result = []
        if type(result_search) == str:
            for v in result_search[:5]:
                if v.get('link') and qa_url in v.get('link'):
                    result += v
        if flags.get('is_hot') or type(result_search) == str:
            result += (result_recommend[:2] + result_search)
        else:
            result += (result_recommend[:1] + result_search)
        return result

    def adjust_ans_url_format(self, answer: str, linkList: list) -> str:
        url_set = sorted(list(set(re.findall(r'https?:\/\/[\w\.\-\?/=+&#$%^;%_]+', answer))), key=len, reverse=True)
        for url in url_set:
            reurl = url
            for char in '?':
                reurl = reurl.replace(char, '\\' + char)
            answer = re.sub(reurl + '(?![\w\.\-\?/=+&#$%^;%_\|])', self.url_format(url), answer)
        for i, url in enumerate(linkList):
            answer = re.sub(f'\[{i+1}\]', f'[{self.url_format(url)}]', answer)
        return answer

    def adjust_ans_format(self, answer: str) -> str:
        if self.frontend == 'line':
            answer.replace('"', "'")
        replace_words = {'此致', '敬禮', '<b>', '</b>', r'\[?\[\d\]?\]?|\[?\[?\d\]\]?', '\w*(抱歉|對不起)\w{0,3}(，|。)'}
        for w in replace_words:
            answer = re.sub(w, '', answer).strip('\n')
        if '親愛的' in answer:
            answer = '親愛的' + '親愛的'.join(answer.split("親愛的")[1:])
        if '祝您愉快！' in answer:
            answer = '祝您愉快！'.join(answer.split("祝您愉快！")[:-1]) + '祝您愉快！'
        return answer

    def get_history_df(self, web_id: str, info: str|list) -> pd.DataFrame:
        if self.frontend == 'line':
            query = f"""SELECT id, web_id, group_id, counts, question, answer, q_a_history, add_time FROM web_push.AI_service_api WHERE group_id = '{info}' and web_id = '{web_id}';"""
            df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                              columns=['id', 'web_id', 'group_id', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
        if self.frontend == 'slack':
            query = f"""SELECT id, web_id, user_id, ts, counts, question, answer, q_a_history, add_time FROM web_push.AI_service WHERE ts='{info[1]}';"""
            df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                              columns=['id', 'web_id', 'user_id', 'ts', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
        return df

    def update_history_df(self, web_id: str, info: str | list, history_df: pd.DataFrame,
                          message: str, answer: str, keyword: str, response_time: float,
                          gpt_query: list, gpt_answer: str) -> pd.DataFrame:
        if len(history_df) == 0:
            if self.frontend == 'line':
                history_df = pd.DataFrame([[web_id, info, 0, datetime.now()]],
                                          columns = ['web_id', 'group_id', 'counts', 'add_time'])
            elif self.frontend == 'slack':
                history_df = pd.DataFrame([[web_id, info[0], info[1], 0, datetime.now()]],
                                          columns=['web_id', 'user_id', 'ts', 'counts', 'add_time'])
        history_df['counts'] += 1
        history_df[['question', 'answer', 'keyword', 'response_time', 'q_a_history']] = [message, answer, keyword, response_time, json.dumps(gpt_query + [{"role": "assistant", "content": f"{gpt_answer}"}])]
        _df = history_df.drop(columns=['keyword', 'response_time'])
        DBhelper.ExecuteUpdatebyChunk(_df, db='jupiter_new', table=f'AI_service{self.table_suffix}', is_ssh=False)
        _df = history_df.drop(columns=['q_a_history'])
        if 'id' in history_df.columns:
            _df = _df.drop(columns=['id'])
        DBhelper.ExecuteUpdatebyChunk(_df, db='jupiter_new', table=f'AI_service_cache{self.table_suffix}', is_ssh=False)


    def error(self, *arg):
        self.logger.print(*arg, level="WARNING")
        return '客服忙碌中，請稍後再試。'

    def QA(self, web_id: str, message: str, info: str | list):
        start_time = time.time()
        self.logger.print(f'Get Message:\t{message}')
        if not self.check_message_length(message, 50):
            self.logger.print('USER ERROR: Input too long!')
            return "親愛的顧客您好，您的提問長度超過限制，請縮短問題後重新發問。"
        history_df = self.get_history_df(web_id, info)

        # Step 1: get keyword from chatGPT
        keyword_list = self.get_question_keyword(message, web_id)
        if keyword_list == 'timeout':
            return self.error('keyword_timeout')
        self.logger.print('關鍵字:\t', keyword_list)

        # Step 2: get gpt_query with search results from google search engine and likr recommend engine
        try:
            result, keyword = func_timeout(10, self.Search.likr_search, (keyword_list, self.CONFIG[web_id]))
            self.logger.print(f'Search_result:\t {[i.get("link") for i in result if i.get("link")], keyword}')
            ##todo 推薦引擎 result_recommend, flags = func_timeout(20, engine.likr_recommend_engine, (message, web_id))
        except Exception as e:
            self.logger.print(f'{e.__traceback__}\n ERROR: {e}', level='ERROR')
            return self.error('search_timeout')
        if result[0].get('URL ERROR'):
            return self.error(str(result))
        if result[0].get('NO RESULTS') and self.CONFIG[web_id]['mode'] == 2:
            gpt_answer = gpt3_answer_slack = f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"

        # Step 3: response from ChatGPT
        else:
            ##todo 推薦引擎 result = reset_result_order(result, result_recommend, flags, web_id)
            history = json.loads(history_df['q_a_history'].iloc[0]) if len(history_df) > 0 else None
            self.logger.print('QA歷史紀錄:\n', history)

            gpt_query, linkList = self.ChatGPT.get_gpt_query(result, message, history, self.CONFIG[web_id])
            self.logger.print('輸入連結:\n', '\n'.join(linkList))
            self.logger.print('ChatGPT輸入:\t', gpt_query)

            gpt_answer = translation_stw(self.ChatGPT.ask_gpt(gpt_query, timeout=60)).replace('，\n', '，')
            if gpt_answer == 'timeout':
                return self.error('gpt3_answer_timeout')
            answer = self.adjust_ans_format(self.adjust_ans_url_format(gpt_answer, linkList))
            self.logger.print('ChatGPT輸出:\t', answer)

        # Step 4: update database
        self.update_history_df(web_id, info, history_df, message, answer, keyword, time.time()-start_time, gpt_query, gpt_answer)
        return answer

if __name__ == "__main__":
    # slack
    AI_customer_service = QA_api('slack', logger())
    print(AI_customer_service.QA('pure17', '有沒有健步機', ['U03PN370PRU', '1679046590.110499']))
    # line
    AI_customer_service = QA_api('line', logger())
    print(AI_customer_service.QA('pure17', '有沒有健步機', 'test4466'))







