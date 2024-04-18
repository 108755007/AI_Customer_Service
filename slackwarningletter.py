from dotenv import load_dotenv
from slack_sdk import WebClient
import os


class slack_warning:
    def __init__(self):
        load_dotenv()
        self.client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
        #self.client = WebClient(token=os.getenv('SLACK_TOKEN'))

        # self.channel = os.getenv('CHANNEL_TEST')
        self.channel = "C0549DVSE3T"

    def send_letter(self, text):
        self.client.chat_postMessage(channel=self.channel, text=text)


if __name__ == '__main__':
    slack_letter = slack_warning()
    slack_letter.send_letter('檢查log....')
