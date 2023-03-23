import os, sys, logging
import re, json, requests
import itertools
from dotenv import load_dotenv
sys.path.append("..")
load_dotenv()
from datetime import datetime
import pandas as pd
from collections import Counter
import jieba
from opencc import OpenCC
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai
from db import DBhelper

DEBUG = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    filename='./log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

openai.api_key = eval(os.getenv('OPENAI_API_KEY'))[1]
google_serch_key = os.getenv('GOOGLE_SERCH_KEY')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
CX = eval(os.getenv('CX'))
VIP = eval(os.getenv('VIP'))
VVIP = eval(os.getenv('VVIP'))
CHANNEL = eval(os.getenv('CHANNEL'))

app = App(token=SLACK_BOT_TOKEN, name="Bot")

date = datetime.today().strftime('%Y/%m/%d')
ts_set = set()
actions_ts = set()
now_ts = datetime.timestamp(datetime.now())

def ask_gpt(message, model="gpt-3.5-turbo"):
	if type(message) == str:
		message = [{'role': 'user', 'content': message}]
	completion = openai.ChatCompletion.create(model=model, messages=message)
	return completion['choices'][0]['message']['content']

def question_pos_parser(question, retry = 3, mode='N'):
	'''
	:param mode: N => just filter noun
	--------
	It will early return when there's only one word after segmentation.
	It will return one word chosen by chatGPT when there are no words after filtering by chatGPT.
	'''
	question = translation_stw(question)
	seg_list = list(jieba.cut(question))
	if len(seg_list) == 1:
		return seg_list
	stopSwitch, retry, keyword = False, retry, ''
	forbidden_words = {'client_msg_id', '什麼'}
	while not stopSwitch and retry:
		if mode == 'N':
			print('ask')
			keyword = ask_gpt(f'To "{question}", choose the {min((len(seg_list) + 2) // 3, 3)} most important NOUN from "{seg_list}" with sep by ", "').replace('\n','').replace('"','').replace("。",'')
			print('ask_finished')
			keyword = [i.strip() for i in keyword.split(',') if not any(re.search(w,i) for w in forbidden_words)]
		stopSwitch = len(keyword) > 0
		retry -= 1
	if not keyword:
		keyword = ask_gpt(f'幫我從"{question}"選出一個重要詞彙,只要回答詞彙就好').replace('\n', '').replace('"','').replace("。", '')
	return keyword

def translation_stw(text):
	cc = OpenCC('s2twp')
	return cc.convert(text)

def likr_search(keyword_list, web_id='nineyi000360'):
	keyword_combination = []
	for i in range(len(keyword_list), 0, -1):
		keyword_combination += list(itertools.combinations(keyword_list, i))
	htmls = [f'https://www.googleapis.com/customsearch/v1/siterestrict?cx={CX[web_id]}&key={google_serch_key}&q=',
			 f'https://www.googleapis.com/customsearch/v1?cx={CX[web_id]}&key={google_serch_key}&q=']

	for kw in keyword_combination:
		kw = '+'.join(kw)
		print('搜尋關鍵字:\t', kw)
		for html in htmls:
			html += kw
			stopSwitch, count, result = False, 10, None
			while not stopSwitch and count:
				print(f'第{11 - count}次搜尋')
				response = requests.get(html)
				if response:
					stopSwitch = response.status_code == 200
					result = response.json().get('items')
					result_kw = kw
				count -= 1
			if stopSwitch: break
		if not count: return '網頁錯誤', '+'.join(keyword_combination)
		if result: break
	if not result: return '無搜尋結果', '+'.join(keyword_combination)
	return result, result_kw

def get_gpt_query(result, query):
	'''
	:param query: result from likr_search
	:param query: question for chatgpt
	-------
	chatgpt_query
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
	linkSet = set()
	chatgpt_query = """\nWeb search results:"""
	for v in result:
		if not v.get('link') or len(linkSet) == 3:
			continue
		url = v.get('link')
		url = re.search(r'.+detail/[\w]+/', url).group(0) if re.search(r'.+detail/[\w]+/', url) else url
		print('搜尋結果:\t', url)
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

def replace_answer(gpt3_ans):
	for url_wrong_fmt, url in re.findall(r'(<(https?:\/\/[\da-z\.-\/]+)\|.*>)', gpt3_ans):
		gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
	gpt3_ans = translation_stw(gpt3_ans)
	for url in set(re.findall(r'https?:\/\/[\w\.-\/]+', gpt3_ans)):
		gpt3_ans = re.sub(url+'(?!\w)', '<' + url + '|查看更多>',gpt3_ans)
	forbidden_words = ['抱歉', '錯誤', '對不起']
	for w in forbidden_words:
		if w in gpt3_ans:
			gpt3_ans = gpt3_ans.split('，')
			gpt3_ans = ('，').join(i for i in gpt3_ans if w not in i)
	gpt3_ans = '親愛的顧客您好，'+'，'.join(gpt3_ans.split('，')[1:])
	return gpt3_ans

def gpt_QA(message, dm_channel, user_id, ts, thread_ts, say):
	say(text=f"請稍等為您提供回覆...", channel=dm_channel, thread_ts=ts)
	query = f"""SELECT id, counts, question, answer, q_a_history FROM web_push.slack_chatgpt WHERE ts='{thread_ts}';"""
	data = DBhelper('jupiter_new').ExecuteSelect(query)
	QA_report_df = pd.DataFrame(data, columns=['id', 'counts', 'question', 'answer', 'q_a_history'])

	# Step 1: get keyword from chatGPT
	keyword_list = question_pos_parser(message, 3)
	print('關鍵字:\t', keyword_list)

	# Step 2: get gpt_query with search results from google search engine
	result, keyword = likr_search(keyword_list)
	gpt_query = get_gpt_query(result, message)
	history = None
	if len(QA_report_df) > 0:
		history = json.loads(QA_report_df['q_a_history'].iloc[0])
		history.append({"role": "user", "content": f"{gpt_query}"})
		while len(str(history)) > 3000 and len(history) > 3:
			history = history[2:]
		print('歷史紀錄:\t', history)
	print('chatGPT輸入:\t', gpt_query)

	if gpt_query == '網頁錯誤':
		say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
		ts_set.add(ts)
		return
	elif gpt_query == '無搜尋結果':
		gpt3_answer = gpt3_answer_slack =f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"
	else:
		# Step 3: response from chatGPT
		gpt3_answer = ask_gpt(history if history else gpt_query)
		gpt3_answer_slack = replace_answer(gpt3_answer)
		print('cahtGPT輸出:\t', gpt3_answer_slack)

	if history:
		history.append({"role": "assistant", "content": f"{gpt3_answer}"})
		QA_report_df['counts'] += 1
		QA_report_df[['question', 'answer', 'q_a_history']] = [gpt_query, gpt3_answer_slack, json.dumps(history)]
	else:
		gpt3_history = json.dumps([{"role": "user", "content": f"{gpt_query}"}, {"role": "assistant", "content": f"{gpt3_answer}"}])
		QA_report_df = pd.DataFrame([[user_id, ts, 1, gpt_query, gpt3_answer_slack, gpt3_history, datetime.now()]],
									columns=['user_id', 'ts', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_chatgpt', chunk_size=100000, is_ssh=False)
	QA_report_df = QA_report_df.drop(['q_a_history'], axis=1)
	QA_report_df['keyword'] = keyword
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='slack_chatgpt_cache', chunk_size=100000, is_ssh=False)
	say(text=f"{gpt3_answer_slack}", channel=dm_channel, thread_ts=ts)

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
	# no reply
	if ts in ts_set or float(now_ts) > float(ts) or \
		dm_channel not in CHANNEL or \
		'bot_profile' in body['event']:
		return
		# user_id not in VIP or \
	ts_set.add(ts)
	if DEBUG and user_id not in VVIP:
		say(text=f"對不起！目前正在維修中,請稍後再嘗試。", channel=dm_channel, thread_ts=ts)
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
	gpt_QA(text, dm_channel, user_id, ts, thread_ts, say)
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