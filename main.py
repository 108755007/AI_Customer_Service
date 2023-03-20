import sys
sys.path.append("..")
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import logging
import requests
import json
from datetime import datetime
import pandas as pd
from db import DBhelper
import re
import openai
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    filename='./log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

openai.api_key = os.getenv('OPENAI_API_KEY')
google_serch_key = os.getenv('GOOGLE_SERCH_KEY')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
CX = eval(os.getenv('CX'))
vip = eval(os.getenv('VIP'))
_channel = eval(os.getenv('CHANNEL'))



app = App(token=SLACK_BOT_TOKEN, name="Bot")
date = datetime.today().strftime('%Y/%m/%d')

def webchatgpt_google(keyword, q, web_id='nineyi000360'):
	html = f'https://www.googleapis.com/customsearch/v1?cx={CX[web_id]}&key={google_serch_key}&q={keyword}'
	r = requests.get(html)
	count = 0
	while r.status_code != 200 and count < 10:
		try:
			r = requests.get(html)
		except:
			count +=1
	if count == 10:
		return '網頁錯誤', '_'
	res = r.json().get('items')
	if not res:
		return '無搜尋結果', '_'
	chatgpt_query = """\nWeb search results:"""
	for i, v in enumerate(res[:3]):
		if v.get('htmlTitle'):
			chatgpt_query += f"""\n\n[{i + 1}] "{v.get('htmlTitle')}"""
		if v.get('snippet'):
			chatgpt_query += f""",snippet = "{v.get('snippet')}"""
		if v.get('pagemap').get('metatags'):
			chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
		if v.get('link'):
			url = v.get('link')
			if web_id == 'nineyi000360':
				url = re.findall(r'.+detail/[\w]+/', url)[0] if re.findall(r'.+detail/[\w]+/', url) else url
			chatgpt_query += f"""\nURL: "{url}" """
	chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.\nQuery: {q}\nReply in 繁體中文\n"""
	return chatgpt_query, res[:3]

# 	chatgpt_query = f"""
#     Web search results:
#
#     [1] "{g[0]['snippet']}"
#     URL: "{g[0]['link']}"
#
#     [2] "{g[1]['snippet']}"
#     URL: "{g[1]['link']}"
#
#     [3] "{g[2]['snippet']}"
#     URL: "{g[2]['link']}"
#
#
#     Current date: {date}
#
#     Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
#     Query: {question}
#     Reply in 繁體中文
#     """

ts_set = set()
actions_ts = set()
now_ts = datetime.timestamp(datetime.now())


def message_gpt(history, text):
	message = json.loads(history)
	message.append({"role": "user", "content":f"{text}"})
	return message

def filter_ans(gpt3_ans, googleSearchResult):
	if googleSearchResult == '_':
		return gpt3_ans
	for v in googleSearchResult:
		url = v.get('link')
		url_ = re.findall(r'.+detail/[\w]+/', url)[0] if 'detail' in url else url
		if f'<{url}|查看更多>' not in gpt3_ans:
			gpt3_ans = gpt3_ans.replace(f'{url}',f'<{url_}|查看更多>')
			if f'<{url_}|查看更多>' not in gpt3_ans:
				gpt3_ans = gpt3_ans.replace(f'{url_}', f'<{url_}|查看更多>')
	ban_word = ['抱歉', '錯誤', '對不起']
	for bw in ban_word:
		if bw in gpt3_ans:
			gpt3_ans = gpt3_ans.split('，')
			gpt3_ans = ('，').join(i for i in gpt3_ans if bw not in i)
	return gpt3_ans

def get_keyword(q):
	message = [{'role': 'user', 'content': f'幫我從"{q}"選出一個重要詞彙,只要回答詞彙就好'}]
	completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
	return completion['choices'][0]['message']['content'].replace('\n','').replace('"','')

def is_goods(keyword):
	message = [{'role': 'user', 'content': f'幫我判斷"{keyword}"是否是一個商品,只要回答"True"或"False"'}]
	completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
	return completion['choices'][0]['message']['content'].replace('\n','').replace('"','').replace("。",'')

@app.message(re.compile(".*"))  # type: ignore
def show_bert_qa(message, body, say):
	dm_channel = message["channel"]
	user_id = message["user"]
	text = message['text']
	ts = message['ts']
	thread_ts = body.get('event').get('thread_ts')
	if ts in ts_set or float(now_ts) > float(ts):
		return
	ts_set.add(ts)
	if dm_channel not in _channel:
		return
	if user_id not in vip:
		return
	if not thread_ts:
		if body.get('event').get('blocks')[0].get('text'):
			text = body.get('event').get('blocks')[0].get('text').get('text')

		#similer_QA = bert_similer(text, Q, A,model_s)

		#act = similer_QA[0][1]

		#QA_report_df = pd.DataFrame([[user_id,text,act,ts]],columns=['user_id', 'question', 'answer', 'timetamp'])
		#MySqlHelper.ExecuteUpdatebyChunk(QA_report_df, db='api02', table='slack_BertQA_history', chunk_size=100000,is_ssh=False)
		if False:
			say(text=act,blocks=[
				{
					"type": "section",
					"text": {
						"type": "plain_text",
						"text": f"{act}",
						"emoji": True
					}
				},
				{
					"type": "actions",
					"elements": [
						{
							"type": "button",
							"text": {
								"type": "plain_text",
								"text": "滿意",
								"emoji": True
							},
							"value": f"{text}",
							"action_id": "bo1"
						},
						{
							"type": "button",
							"text": {
								"type": "plain_text",
								"text": "不滿意",
								"emoji": True
							},
							"value": f"{text}",
							"action_id": "bo2"
						}
					]
				}
			],channel=dm_channel,thread_ts=ts)
		else:
			say(text=f"請稍等為您提供回覆...", channel=dm_channel,thread_ts=ts)
			keyword = get_keyword(text)
			print(keyword)
			gpt_query,gs = webchatgpt_google(keyword,text)
			if gpt_query == '網頁錯誤':
				say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
				ts_set.add(ts)
				return
			elif gpt_query == '無搜尋結果':
				gpt_query = text.replace('你們', '全家電商').replace('妳們', '全家電商').replace('你', '全家電商').replace('妳', '全家電商')
			print(gpt_query)
			m = 0
			while m == 0:
				try:
					completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": f"{gpt_query}"}])
					m = 1
				except:
					pass
			gpt3_answer	= completion['choices'][0]['message']['content']
			print(gpt3_answer)
			gpt3_answer_f = filter_ans(gpt3_answer, gs)
			gpt3_history = json.dumps([{"role": "user", "content": f"{gpt_query}"}, {"role": "assistant", "content":f"{gpt3_answer}"}])
			print(gpt3_history)
			QA_report_df = pd.DataFrame([[user_id, ts, 1, gpt_query, gpt3_answer_f, gpt3_history, datetime.now()]],columns=['user_id', 'ts', 'counts', 'last_question', 'last_answer', 'q_a_history', 'add_time'])
			DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_chatgpt', chunk_size=100000,is_ssh=False)
			say(text=f"{gpt3_answer_f}", channel=dm_channel,thread_ts=ts)

	else:
		say(text=f"請稍等為您提供回覆...", channel=dm_channel, thread_ts=ts)
		query = f"""SELECT id, counts, last_question, last_answer, q_a_history FROM web_push.slack_chatgpt WHERE ts='{thread_ts}';"""
		data = DBhelper('jupiter_new').ExecuteSelect(query)
		df_history = pd.DataFrame(data, columns=['id', 'counts', 'last_question', 'last_answer', 'q_a_history'])
		print(json.loads(df_history['q_a_history'].iloc[0]))
		keyword = get_keyword(text)
		print(keyword)
		gpt_query, gs = webchatgpt_google(keyword, text)
		print(gpt_query)
		if gpt_query == '網頁錯誤':
			say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
			ts_set.add(ts)
			return
		elif gpt_query == '無搜尋結果':
			gpt_query = text.replace('你們', '全家電商').replace('妳們', '全家電商').replace('你', '全家電商').replace('妳','全家電商')
		history = json.loads(df_history['q_a_history'].iloc[0])
		history.append({"role": "user", "content": f"{gpt_query}"})
		print(history)
		while len(str(history)) > 5000 and len(history) > 3:
			history = history[2:]
		try:
			completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=history)
			gpt3_answer	= completion['choices'][0]['message']['content']
			gpt3_answer_f = filter_ans(gpt3_answer, gs)
		except:
			say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
			return
		history.append({"role": "assistant", "content": f"{gpt3_answer}"})
		df_history['counts'] += 1
		df_history['last_question'] = gpt_query
		df_history['last_answer'] = gpt3_answer_f
		df_history['q_a_history'] = json.dumps(history)
		DBhelper.ExecuteUpdatebyChunk(df_history, db='jupiter_new', table='slack_chatgpt', chunk_size=100000, is_ssh=False)
		say(text=f"{gpt3_answer_f}", channel=dm_channel, thread_ts=ts)
	return

@app.action("bo1")
def handle_some_action(ack,body, say):
	ack()
	bts = body['message']['thread_ts']
	if bts in actions_ts or float(now_ts) > float(bts):
		return
	actions_ts.add(bts)
	say(text=f"不客氣",channel=body['container']['channel_id'], thread_ts=body['container']['thread_ts'])
	return

@app.action("bo2")
def handle_some_action(ack, body, say):
	ack()
	bts = body['message']['thread_ts']
	if bts in actions_ts or float(now_ts) > float(bts):
		return
	actions_ts.add(bts)
	ts = body['container']['thread_ts']
	say(text=f"請稍等為您提供其他答案...", channel=body['container']['channel_id'],thread_ts=body['container']['thread_ts'])
	text = body['actions'][0]['value']
	gpt_query = webchatgpt_google(text)
	m = 0
	while m == 0:
		try:
			completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": f"{gpt_query}"}])
			m = 1
		except:
			pass
	QA_report_df = pd.DataFrame([[body['user']['id'],ts,text,gpt_query,completion['choices'][0]['message']['content'],1]],columns=['user_id', 'ts','question','question1', 'answer1','counts'])
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_webchatgpt', chunk_size=100000,is_ssh=False)
	say(text=f"{completion['choices'][0]['message']['content']}", channel=body['container']['channel_id'],thread_ts=body['container']['thread_ts'])
	return

@app.event("message")
def handle_message_events(body, logger):
     logger.info(body)

def main():
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
	print('START!!')
	main()
