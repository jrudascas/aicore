import cv2
import requests
import json

file = '/home/jrudascas/PycharmProjects/aicore/00000427_000.png'

im = cv2.imread(file)

url = 'http://127.0.0.1:8000/runprediction/'

# Additional headers.
headers = {'Content-Type': 'application/json',
           'Authorization': 'Token c22a081afe85037c3511cff9d91ed654c0d27e5e'}

# Body
payload = {'model': {'name': 'ChestXNet v1.0'},
           'metadata': {'bytestream': {file: {'stream': im.tolist()}}
                        }}
# convert dict to json by json.dumps() for body data.
resp = requests.patch(url, data=json.dumps(payload, indent=4), headers=headers)

print(resp.content)
