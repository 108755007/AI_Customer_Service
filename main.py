import os, sys, logging
import re, json, requests
from dotenv import load_dotenv
sys.path.append("..")
load_dotenv()
from datetime import datetime
import pandas as pd
from opencc import OpenCC
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai
from db import DBhelper

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
cc = OpenCC('s2twp')
date = datetime.today().strftime('%Y/%m/%d')
ts_set = set()
actions_ts = set()
now_ts = datetime.timestamp(datetime.now())

def ask_gpt(message, model="gpt-3.5-turbo"):
	if type(message) == str:
		message = [{'role': 'user', 'content': message}]
	completion = openai.ChatCompletion.create(model=model, messages=message)
	return completion['choices'][0]['message']['content']

def question_pos_parser(question):
	stopSwitch, retry = False, 3
	mappingDict = {'noun': '名詞', 'verb': '動詞'}
	while not stopSwitch and retry:
		question_keywords = ask_gpt(f'Choose {min(len(question) + 2 // 3, 3)} important words from "{question}" and give me what part of speech it is. Using [word/pos] with sep by ", "').replace('\n','').replace('"','').replace("。",'')
		keyword_pos = {}
		for k in question_keywords.split(','):
			k = k.split('/')
			if len(k) != 2:
				continue
			keyword_pos[k[0]] = mappingDict.get(k[1], k[1])
		stopSwitch = len(keyword_pos) > 0
		retry -= 1
	return keyword_pos

def get_gpt_query(keyword, query, web_id='nineyi000360'):
	'''
	:param keyword: for google search
	:param query: question for chatgpt
	######## chatgpt_query ########
		Web search results:

		[1] "{g[0]['snippet']}"
		URL: "{g[0]['link']}"

		[2] "{g[1]['snippet']}"
		URL: "{g[1]['link']}"

		[3] "{g[2]['snippet']}"
		URL: "{g[2]['link']}"


		Current date: {date}

		Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
		Query: {question}
		Reply in 繁體中文
	'''
	html = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx={CX[web_id]}&key={google_serch_key}&q={keyword}'
	stopSwitch, count, result = False, 10, None
	while not stopSwitch and count:
		print(count)
		response = requests.get(html)
		if response:
			stopSwitch = response.status_code == 200
			result = response.json().get('items')
		count -=1
	if not count: return '網頁錯誤'
	if not result: return '無搜尋結果'
	linkSet = set()
	chatgpt_query = """\nWeb search results:"""
	for v in result:
		if not v.get('link') or len(linkSet) == 3:
			continue
		url = v.get('link')
		url = re.search(r'.+detail/[\w]+/', url).group(0) if re.search(r'.+detail/[\w]+/', url) else url
		print(url)
		if url in linkSet:
			continue
		linkSet.add(url)

		if v.get('htmlTitle'):
			chatgpt_query += f"""\n\n[{len(linkSet)}] "{v.get('htmlTitle')}"""
		if v.get('snippet'):
			chatgpt_query += f""",snippet = "{v.get('snippet')}"""
		if v.get('pagemap').get('metatags'):
			chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
		chatgpt_query += f"""\nURL: "{url}" """
	chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.\nQuery: {query}\nReply in 繁體中文\n"""
	return chatgpt_query

def filter_ans(gpt3_ans):
	for url_wrong_fmt, url in re.findall(r'(<(https?:\/\/[\da-z\.-\/]+)\|.*>)', gpt3_ans):
		gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
	gpt3_ans = cc.convert(gpt3_ans)
	for url in set(re.findall(r'https?:\/\/[\w\.-\/]+', gpt3_ans)):
		print(url)
		gpt3_ans = gpt3_ans.replace(url, '<' + url + '|查看更多>')
	ban_word = ['抱歉', '錯誤', '對不起']
	for bw in ban_word:
		if bw in gpt3_ans:
			gpt3_ans = gpt3_ans.split('，')
			gpt3_ans = ('，').join(i for i in gpt3_ans if bw not in i)
	gpt3_ans = '親愛的顧客您好，'+'，'.join(gpt3_ans.split('，')[1:])
	return gpt3_ans

def is_goods(keyword):
	message = [{'role': 'user', 'content': f'幫我判斷"{keyword}"是否是一個商品,只要回答"True"或"False"'}]
	completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
	return completion['choices'][0]['message']['content']

def update_history(user_id, ts, counts, question, keyword, answer):
	QA_report_df = pd.DataFrame([[user_id, ts, counts, question, keyword, answer, datetime.now()]],columns=['user_id', 'ts', 'counts', 'question', 'keyword', 'answer','add_time'])
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_chatgpt_history', chunk_size=100000,is_ssh=False)
	return

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
			keyword_pos = question_pos_parser(text)
			keyword = '+'.join(k.strip() for k, v in keyword_pos.items() if v == '名詞')
			if not keyword:
				keyword = ask_gpt(f'幫我從"{text}"選出一個重要詞彙,只要回答詞彙就好').replace('\n', '').replace('"','').replace("。", '')
			print(keyword)
			gpt_query = get_gpt_query(keyword, text)
			if gpt_query == '網頁錯誤':
				say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
				ts_set.add(ts)
				return
			elif gpt_query == '無搜尋結果':
				say(text=f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。", channel=dm_channel, thread_ts=ts)
				return
			print(gpt_query)
			gpt3_answer	= ask_gpt(gpt_query)
			print(gpt3_answer)
			gpt3_answer_f = filter_ans(gpt3_answer)
			gpt3_history = json.dumps([{"role": "user", "content": f"{gpt_query}"}, {"role": "assistant", "content":f"{gpt3_answer}"}])
			print(gpt3_history)
			QA_report_df = pd.DataFrame([[user_id, ts, 1, gpt_query, gpt3_answer_f, gpt3_history, datetime.now()]],columns=['user_id', 'ts', 'counts', 'last_question', 'last_answer', 'q_a_history', 'add_time'])
			DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_chatgpt', chunk_size=100000,is_ssh=False)
			update_history(user_id, ts, 1, text, keyword, gpt3_answer_f)
			say(text=f"{gpt3_answer_f}", channel=dm_channel,thread_ts=ts)

	else:
		say(text=f"請稍等為您提供回覆...", channel=dm_channel, thread_ts=ts)
		query = f"""SELECT id, counts, last_question, last_answer, q_a_history FROM web_push.slack_chatgpt WHERE ts='{thread_ts}';"""
		data = DBhelper('jupiter_new').ExecuteSelect(query)
		df_history = pd.DataFrame(data, columns=['id', 'counts', 'last_question', 'last_answer', 'q_a_history'])
		print(json.loads(df_history['q_a_history'].iloc[0]))
		keyword_pos = question_pos_parser(text)
		keyword = '+'.join(k.strip() for k, v in keyword_pos.items() if v == '名詞')
		if not keyword:
			keyword = ask_gpt(f'幫我從"{text}"選出一個重要詞彙,只要回答詞彙就好').replace('\n', '').replace('"', '').replace("。", '')
		print(keyword)
		gpt_query = get_gpt_query(keyword, text)
		print(gpt_query)
		if gpt_query == '網頁錯誤':
			say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
			ts_set.add(ts)
			return
		elif gpt_query == '無搜尋結果':
			say(text=f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。", channel=dm_channel, thread_ts=ts)
			return
		history = json.loads(df_history['q_a_history'].iloc[0])
		history.append({"role": "user", "content": f"{gpt_query}"})
		print(history)
		while len(str(history)) > 3000 and len(history) > 3:
			history = history[2:]
		gpt3_answer	= ask_gpt(history)
		gpt3_answer_f = filter_ans(gpt3_answer)
		history.append({"role": "assistant", "content": f"{gpt3_answer}"})
		df_history['counts'] += 1
		df_history[['last_question', 'last_answer', 'q_a_history']] = [gpt_query, gpt3_answer_f ,json.dumps(history)]
		DBhelper.ExecuteUpdatebyChunk(df_history, db='jupiter_new', table='slack_chatgpt', chunk_size=100000, is_ssh=False)
		update_history(user_id, ts, 1, text, keyword, gpt3_answer_f)
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
	gpt_query = get_gpt_query(text)
	completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{gpt_query}"}])
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