import os, time, re
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from functools import wraps
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from AI_customer_service import QA_api
from utils.log import logger
from db import DBhelper
DEBUG = False

SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
VIP = eval(os.getenv('VIP'))
VVIP = eval(os.getenv('VVIP'))
CHANNEL = eval(os.getenv('CHANNEL'))

app = App(token=SLACK_BOT_TOKEN, name="Bot")
logger = logger()
AI_customer_service = QA_api('slack', logger)
start_app_ts = datetime.timestamp(datetime.now())

def timing(func):
    @wraps(func)
    def time_count(*args, **kwargs):
        t_start = time.time()
        values = func(*args, **kwargs)
        t_end = time.time()
        logger.print(f"{func.__name__} time consuming:  {(t_end - t_start):.3f} seconds")
        return values
    return time_count

def check_web_id(message):
	for web_id in AI_customer_service.CONFIG.keys():
		if AI_customer_service.CONFIG[web_id]['web_id'] in message:
			return web_id
	for web_id in AI_customer_service.CONFIG.keys():
		if AI_customer_service.CONFIG[web_id]['web_name'] in message:
			return web_id
	for web_id in AI_customer_service.CONFIG.keys():
		for name in eval(AI_customer_service.CONFIG[web_id]['other_name']):
			if name in message:
				return web_id
	return 'nineyi000360'

@app.message(re.compile(".*"))  # type: ignore
def show_bert_qa(message, body, say):
	dm_channel = message["channel"]
	user_id = message["user"]
	text = message['text']
	ts = message['ts']
	thread_ts = body.get('event').get('thread_ts') if body.get('event').get('thread_ts') else message['ts']

	# do not reply
	if dm_channel not in CHANNEL or \
	   'bot_profile' in body['event']:
		return
	# float(start_app_ts) > float(thead_ts) or \
	if DEBUG and user_id not in VVIP:
		say(text=f"對不起！目前AI客服正在調整中,請稍後再嘗試。", channel=dm_channel, thread_ts=ts)
		print(f'{"@"*20}有人要用！！{"@"*20}')
		return

	query = f"""SELECT web_id FROM web_push.AI_service WHERE ts='{thread_ts}';"""
	history = DBhelper('jupiter_new').ExecuteSelect(query)
	if history:
		web_id = history[0][0]
	else:
		web_id = check_web_id(text)
	say(text=f"請稍等為您提供回覆...", channel=dm_channel, thread_ts=ts)
	say(text=AI_customer_service.QA(web_id, text, [user_id, thread_ts]),
		channel=dm_channel, thread_ts=ts,
		unfurl_links=False, unfurl_media=False)
	return

@app.event("message")
def handle_message_events(body, logger):
     logger.info(body)

def main():
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
	logger.print('START!!')
	main()