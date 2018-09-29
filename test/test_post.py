# -*- coding: utf-8 -*-
import requests, json

# main
if __name__ == "__main__":
    f_name = 'images_2.jpg'
    files = {'FILE': (f_name, open('test_imgs/%s'%f_name, 'rb'))}

    url = 'http://35.237.167.45:80/uploads'
    response = requests.post(url, files=files)

    print(response.status_code)
    print(json.loads(response.content))
