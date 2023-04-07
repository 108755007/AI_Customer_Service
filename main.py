import os, sys, time, logging, traceback
from functools import wraps
from dotenv import load_dotenv
sys.path.append("..")
load_dotenv()
import re, json, requests, itertools
from func_timeout import func_timeout
from datetime import datetime
import pandas as pd
import jieba
from jieba import posseg
from opencc import OpenCC
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai
from db import DBhelper
from recommend_api import Search_engine
engine = Search_engine()

DEBUG = False
date = datetime.today().strftime('%Y/%m/%d')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = f'./log/{date.replace("/", "-")}-1.log'
while os.path.exists(log_file_path):
	log_file_path = '-'.join(log_file_path.split('-')[:-1]+[str(int(log_file_path.split('-')[-1][:-4])+1)+'.log'])
with open(log_file_path, 'w+') as f:
	pass
logging.basicConfig(level=logging.INFO,
                    filename=log_file_path,
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] -\t %(levelname)s: %(message)s')

OPEN_AI_KEY_DICT = eval(os.getenv('OPENAI_API_KEY'))
GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
VIP = eval(os.getenv('VIP'))
VVIP = eval(os.getenv('VVIP'))
CHANNEL = eval(os.getenv('CHANNEL'))

app = App(token=SLACK_BOT_TOKEN, name="Bot")

ts_set = set()
actions_ts = set()
now_ts = datetime.timestamp(datetime.now())

def get_openai_key_id(model):
	if 'gpt-4' in model:
		opena_ai_key_id = DBhelper('jupiter_new').ExecuteSelect("SELECT id, counts FROM web_push.AI_service_token_counter x where id < 6 ORDER BY counts limit 1;")
	else:
		opena_ai_key_id = DBhelper('jupiter_new').ExecuteSelect("SELECT id, counts FROM web_push.AI_service_token_counter x ORDER BY counts limit 1;")
	return opena_ai_key_id[0][0]

def print_log(info):
	logger.setLevel(logging.DEBUG)
	print(info)
	logging.info(info)
	logger.setLevel(logging.INFO)

def timing(func):
    @wraps(func)
    def time_count(*args, **kwargs):
        t_start = time.time()
        values = func(*args, **kwargs)
        t_end = time.time()
        print_log(f"{func.__name__} time consuming:  {(t_end - t_start):.3f} seconds")
        return values
    return time_count

def get_config():
	'''
	Returns {web_id: config from jupiter_new -> web_push.AI_service_config}
	'''
	config_dict = {}
	config = DBhelper('jupiter_new').ExecuteSelect("SELECT * FROM web_push.AI_service_config where mode != 0;")
	config_col = [i[0] for i in DBhelper('jupiter_new').ExecuteSelect("SHOW COLUMNS FROM web_push.AI_service_config;")]
	for conf in config:
		config_dict[conf[1]] = {}
		for k, v in zip(config_col, conf):
			config_dict[conf[1]][k] = v
	return config_dict

CONFIG = get_config()

def check_message_length(text, length):
	for url in re.findall(r'https?:\/\/[\w\.\-\/\?\=\+&#$%^;%_]+', text):
		stopSwitch, retry, result = False, 3, None
		while not stopSwitch and retry:
			try:
				response = requests.get(url)
			except Exception as e:
				break
			if response:
				stopSwitch = True
			retry -= 1
		if stopSwitch and response.status_code == 200:
			text = text.replace(url, '')
	return len(text) <= length

@timing
def ask_gpt(message, model="gpt-3.5-turbo"):
	token_id = get_openai_key_id(model)
	DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts + 1 WHERE id = {token_id}')
	openai.api_key = OPEN_AI_KEY_DICT[token_id]
	if type(message) == str:
		message = [{'role': 'user', 'content': message}]
	completion = openai.ChatCompletion.create(model=model, messages=message)
	DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts - 1 WHERE id = {token_id}')
	return completion['choices'][0]['message']['content']

@timing
def question_pos_parser(question, retry=3, web_id='nineyi000360'):
	'''
	:param mode: N => just filter noun
	--------
	It will early return when there's only one word after segmentation.
	It will return one word chosen by chatGPT when there are no words after filtering by chatGPT.
	'''
	question = translation_stw(question).lower()
	for i in [CONFIG[web_id]['web_id'], CONFIG[web_id]['web_name']]+eval(CONFIG[web_id]['other_name']):
		question = question.replace(i, '')
		for j in list(jieba.cut(i)):
			if j in question:
				question = question.replace(j,'')
	stopSwitch, retry, keyword = False, retry, ''
	forbidden_words = {'client_msg_id', '什麼', '有', '我', '你', '妳', '你們', '妳們', '沒有', '怎麼', '怎', '如何', '要'}
	not_noun_list = [w for w, p in list(posseg.cut(question)) if 'n' not in p.lower()]
	while not stopSwitch and retry:
		keyword = ask_gpt(f"""To "{question}", choose the {min((len(question) + 2) // 3, 3)} most important words that are always used as nouns and cannot exceed 3 Chinese characters, and separate by " ".""", model = 'gpt-4').replace('\n','').replace('"','').replace("。",'')
		keyword = [i.strip('.') for i in keyword.split(' ') if not any(re.search(w, i.strip('.')) for w in forbidden_words) and i.strip('.') not in not_noun_list and i.strip('.') in question]
		stopSwitch = len(keyword) > 0
		retry -= 1
	if not keyword:
		keyword = [ask_gpt(f'Choose one important word from "{question}". Just reply the word in 繁體中文.', model='gpt-4').split(' ')[0].replace('\n', '').replace('"', '').replace("。", '')]
	print_log(keyword)
	return list(map(translation_stw, keyword))

def translation_stw(text):
	cc = OpenCC('likr-s2twp')
	return cc.convert(text)

def google_search(keyword_combination, html ,retry):
	for kw in keyword_combination:
		kw = '+'.join(kw)
		print_log(f'Keyword for search:\t {kw}')
		search_html = html + kw
		print_log(f'Search URL:\t {search_html}')
		stopSwitch, cnt, result = False, 1, None
		while not stopSwitch and cnt != retry + 1:
			print_log(f'Search times:\t {cnt}')
			response = requests.get(search_html)
			if response:
				stopSwitch = response.status_code == 200
				result = response.json().get('items')
				result_kw = kw
			cnt += 1
		if result: break
	if not stopSwitch:
		return 'URL ERROR', None
	return result, result_kw

@timing
def likr_search(keyword_list, web_id='nineyi000360', keyword_length=3):
	if len(keyword_list) > keyword_length:
		keyword_list = keyword_list[:keyword_length]
	result = None
	keyword_combination = []
	for i in range(len(keyword_list), 0, -1):
		keyword_combination += list(itertools.combinations(sorted(keyword_list, key=len, reverse=True), i))
	if CONFIG[web_id]['domain_cx']!= '_':
		html = f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={CONFIG[web_id]['domain_cx']}&key={GOOGLE_SEARCH_KEY}&q="
		result, result_kw = google_search(keyword_combination, html, 3)
		if result == 'URL ERROR':
			result, result_kw = google_search(keyword_combination, html[:42] + html[55:], 1)
	if (not result or result == 'URL ERROR') and CONFIG[web_id]['sub_domain_cx']!= '_':
		html = f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={CONFIG[web_id]['sub_domain_cx']}&key={GOOGLE_SEARCH_KEY}&q="
		result, result_kw = google_search(keyword_combination, html, 3)
		if result == 'URL ERROR':
			result, result_kw = google_search(keyword_combination, html[:42] + html[55:], 1)
	if (not result or result == 'URL ERROR') and str(CONFIG[web_id]['mode']) == '3':
		result, result_kw = google_search(keyword_combination, f"https://www.googleapis.com/customsearch/v1?cx=46d551baeb2bc4ead&key={GOOGLE_SEARCH_KEY}&q={CONFIG[web_id]['web_name'].replace(' ', '+')}+", 1)
	if not result:
		print_log(f"No results: {html}, {'+'.join(keyword_list)}")
		result, result_kw = [{'NO RESULTS': True}], '+'.join(keyword_list)
	elif result == 'URL ERROR':
		result = [{'URL ERROR': True}]
		print_log(f"URL ERROR: {html}, {'+'.join(keyword_list)}")
		result_kw = '+'.join(keyword_list)
	return result, result_kw

def reset_result_order(result_search, result_recommend, flags, web_id):
	qa_url = [i.strip('*')for i in CONFIG[web_id]['qa_url'].split(',')]
	result = []
	if type(result_search) == str:
		for v in result_search[:5]:
			if v.get('link') and qa_url in v.get('link'):
				result += v
	if flags.get('is_hot') or type(result_search) == str:
		result += result_recommend
	else:
		result += (result_recommend[:2] + result_search)
	return result

def get_gpt_query(result, query, history, web_id):
	'''
	:param query: result from likr_search
	:param query: question for chatgpt
	-------
	chatgpt_query
		Results:

		[1] "{g[0]['snippet']}"
		URL: "{g[0]['link']}"

		[2] "{g[1]['snippet']}"
		URL: "{g[1]['link']}"

		[3] "{g[2]['snippet']}"
		URL: "{g[2]['link']}"


		Current date: {date}

		Instructions: Using the provided products or Q&A, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:
		Always cite results using [[number](URL)] notation in the sentence's end when using the information from results.
		Write separate answers for each subject.
		"親愛的顧客您好，" in the beginning.
		"祝您愉快！" in the end.

		Query: {query}
	'''
	message = history if history else [{"role": "system", "content": f"我們是{CONFIG[web_id]['web_name']}(代號：{CONFIG[web_id]['web_id']},官方網站：{CONFIG[web_id]['web_url']}),{CONFIG[web_id]['description']}"}]
	if type(result) != str:
		linkSet = set()
		chatgpt_query = """\nResults:"""
		for v in result:
			if not v.get('link') or len(linkSet) == 3:
				continue
			url = v.get('link')
			url = re.search(r'.+detail/[\w\-]+/', url).group(0) if re.search(r'.+detail/[\w\-]+/', url) else url
			print_log(f'Input results:\t {url}')
			if url in linkSet:
				continue
			linkSet.add(url)
			if v.get('htmlTitle'):
				chatgpt_query += f"""\n\n[{len(linkSet)}] "{v.get('htmlTitle')}"""
			if v.get('snippet'):
				chatgpt_query += f""",snippet = "{v.get('snippet')}"""
			if v.get('pagemap') and v.get('pagemap').get('metatags'):
				chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
			if len(url) < 120:
				chatgpt_query += f"""\nURL: "{url}" """
		chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: Using the provided products or Q&A, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\nAlways cite results using [[number](URL)] notation in the sentence's end when using the information from results.\nWrite separate answers for each subject.\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {query}"""
	else:
		chatgpt_query = f"""\n\n\nCurrent date: {date}\n\nInstructions: If you are the brand, "{CONFIG[web_id]['web_name']}"({web_id}) customer service and there is no search result in product list, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {query}"""
	message += [{'role': 'user', 'content': chatgpt_query}]
	return message

def replace_answer(gpt3_ans):
	print_log(f"ChatGPT reply：\t {gpt3_ans}")
	for url_wrong_fmt, url in re.findall(r'(<(https?:\/\/[\w\.\-\?/=+&#$%^;%_]+)\|.*>)', gpt3_ans):
		gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
	for url_wrong_fmt, url in re.findall(r'(\[?\d\]?\(?(https?:\/\/[\w\.\-\/\?\=\+\&\#\$\%\^\;\%\_]+)\)?)', gpt3_ans):
		gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
	gpt3_ans = translation_stw(gpt3_ans)
	gpt3_ans = gpt3_ans.replace('，\n', '，')
	url_set = sorted(list(set(re.findall(r'https?:\/\/[\w\.\-\?/=+&#$%^;%_]+', gpt3_ans))), key=len , reverse=True)
	for url in url_set:
		reurl = url
		for char in '?':
			reurl = reurl.replace(char, '\\' + char)
		gpt3_ans = re.sub(reurl + '(?![\w\.\-\?/=+&#$%^;%_\|])', '<' + url + '|查看更多>', gpt3_ans)
	replace_words = {'此致', '敬禮', '<b>', '</b>', r'\[?\[\d\]?\]?|\[?\[?\d\]\]?', '\w*(抱歉|對不起)\w{0,3}(，|。)'}
	for w in replace_words:
		gpt3_ans = re.sub(w, '', gpt3_ans).strip('\n')
	if '親愛的' in gpt3_ans:
		gpt3_ans = '親愛的' + '親愛的'.join(gpt3_ans.split("親愛的")[1:])
	if '祝您愉快！' in gpt3_ans:
		gpt3_ans = '祝您愉快！'.join(gpt3_ans.split("祝您愉快！")[:-1]) + '祝您愉快！'
	return gpt3_ans

def check_web_id(message):
	for web_id in CONFIG.keys():
		if CONFIG[web_id]['web_id'] in message:
			return web_id
	for web_id in CONFIG.keys():
		if CONFIG[web_id]['web_name'] in message:
			return web_id
	for web_id in CONFIG.keys():
		for name in eval(CONFIG[web_id]['other_name']):
			if name in message:
				return web_id
	return 'nineyi000360'

@timing
def gpt_QA(message, dm_channel, user_id, ts, thread_ts, say):
	web_id = check_web_id(message)
	query = f"""SELECT id, web_id, user_id, counts, question, answer, q_a_history, add_time FROM web_push.AI_service WHERE ts='{thread_ts}';"""
	data = DBhelper('jupiter_new').ExecuteSelect(query)
	QA_report_df = pd.DataFrame(data, columns=['id', 'web_id', 'user_id', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
	if len(QA_report_df) > 0:
		web_id = QA_report_df['web_id'].values[0]
	say(text=f"請稍等為您提供回覆...", channel=dm_channel, thread_ts=ts)

	# Step 1: get keyword from chatGPT (timeout = 3s)
	try:
		keyword_list = func_timeout(10, question_pos_parser, (message, 3, web_id))
		print(keyword_list)
	except Exception as e:
		say(text=f"伺服器忙碌中，請稍後再試。", channel=dm_channel, thread_ts=ts)
		print_log(f'{traceback.format_tb(e.__traceback__)[-1]} ERROR: {e}')
		return
	print_log(f'keywords:\t {keyword_list}')

	# Step 2: get gpt_query with search results from google search engine (timeout = 5s)
	try:
		result, keyword = func_timeout(10, likr_search, (keyword_list, web_id))
		print_log(f'Search_result:\t {[i.get("link") for i in result if i.get("link")], keyword}')
		result_recommend, flags = func_timeout(20, engine.likr_recommend_engine, (message, web_id))
		print_log(f'recommend_result:\t {[i.get("link") for i in result_recommend if i.get("link")], flags}')
	except Exception as e:
		say(text=f"伺服器忙碌中，請稍後再試。", channel=dm_channel, thread_ts=ts)
		print_log(f'{traceback.format_tb(e.__traceback__)[-1]} ERROR: {e}')
		return
	if result[0].get('URL ERROR') and not result_recommend:
		say(text=f"發生錯誤，請再詢問一次！", channel=dm_channel, thread_ts=ts)
		ts_set.add(ts)
		return

	if result[0].get('NO RESULTS') and CONFIG[web_id]['mode'] == 2:
		gpt3_answer = gpt3_answer_slack = f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"
	else:
		# Step 3: response from chatGPT (timeout = 60s)
		timeoutSwitch = False
		history = None
		if len(QA_report_df) > 0:
			history = json.loads(QA_report_df['q_a_history'].iloc[0])
		result = reset_result_order(result, result_recommend, flags, web_id)
		gpt_query = get_gpt_query(result, message, history, web_id)
		while len(str(gpt_query)) > 3000 and len(gpt_query) > 3:
			gpt_query = [gpt_query[0]] + gpt_query[3:]
		print_log(f'ChatGPT input:\t {gpt_query}')
		try:
			gpt3_answer = func_timeout(60, ask_gpt, (gpt_query,))
		except Exception as e:
			print_log(f'{traceback.format_tb(e.__traceback__)[-1]} ERROR: {e}')
			say(text=f"伺服器忙碌中，請稍後再試。", channel=dm_channel, thread_ts=ts)
			timeoutSwitch = True
			gpt3_answer = ask_gpt(gpt_query)
		gpt3_answer_slack = replace_answer(gpt3_answer)
		print_log(f'ChatGPT output:\t {gpt3_answer_slack}')
	if not timeoutSwitch:
		say(text=f"{gpt3_answer_slack}", channel=dm_channel, thread_ts=ts)

	if history:
		gpt_query.append({"role": "assistant", "content": f"{gpt3_answer}"})
		QA_report_df['counts'] += 1
		QA_report_df[['question', 'answer', 'q_a_history']] = [message, gpt3_answer_slack, json.dumps(gpt_query)]
	else:
		gpt3_history = json.dumps(gpt_query + [{"role": "assistant", "content": f"{gpt3_answer}"}])
		QA_report_df = pd.DataFrame([[web_id, user_id, ts if not thread_ts else thread_ts, 1, message, gpt3_answer_slack, gpt3_history, datetime.now()]],
									columns=['web_id', 'user_id', 'ts', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='AI_service', chunk_size=100000, is_ssh=False)
	if 'id' in QA_report_df.columns:
		QA_report_df = QA_report_df.drop(['id'], axis=1)
	QA_report_df = QA_report_df.drop(['q_a_history'], axis=1)
	QA_report_df['keyword'] = keyword
	DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='AI_service_cache', chunk_size=100000, is_ssh=False)


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
		say(text=f"對不起！目前AI客服正在調整中,請稍後再嘗試。", channel=dm_channel, thread_ts=ts)
		print(f'{"@"*20}有人要用！！{"@"*20}')
		return

	if not check_message_length(text, 50):
		print_log('USER ERROR: Input too long!')
		say(text=f"親愛的顧客您好，您的提問長度超過限制，請縮短問題後重新發問。", channel=dm_channel, thread_ts=ts)
		return

	gpt_QA(text, dm_channel, user_id, ts, thread_ts, say)
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