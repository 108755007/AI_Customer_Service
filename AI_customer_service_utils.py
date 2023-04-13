from opencc import OpenCC
import requests

def translation_stw(text):
    cc = OpenCC('likr-s2twp')
    return cc.convert(text)

def fetch_url_response(url, retry=3):
    while retry:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"URL: {url} 連線失敗，原因: {e}")
        retry -= 1
    return None

