import time

import requests

def clean_html(html):
    return html

try:
    while True:
        try:
            ret = requests.get('http://192.168.0.101', timeout=3)
            print(ret.status_code)
            print(ret.content.decode())
        except Exception as e:
            print(e)
        finally:
            print('nothing found')
            time.sleep(5)
except KeyboardInterrupt:
    print('quitting')
