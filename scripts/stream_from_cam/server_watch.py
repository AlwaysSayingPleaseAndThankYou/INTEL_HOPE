import random
import time
import requests
from requests import exceptions as req_errs
import run_on_single_image as rs
import matplotlib.pyplot as plt


print('looking for server')
batch_group =random.choice(range(300))
img_num = 0
hope = rs.loaded_model()
transforms = rs.applied_transform
fig = plt.figure()
while True:
    try:
        print("dailing")
        data = requests.get('http://192.168.0.101/')
        #block until data recieved
        print(f'connect with status code: {data.status_code}')
        file_path = f'streamed_photos/{batch_group}-{img_num}.jpeg'
        with open(file_path, 'wb+') as f:
            f.write(data.content)
        time.sleep(0.01)
        title=f"{batch_group} - {img_num}"
        out = rs.classify(hope, file_path, transforms)
        ret = rs.plot_from_single_image(out, title=title, fig=fig)
        print(f'drawing image {title}')
        ret.show()
    except req_errs.ConnectionError as conn_err:
        print('failed to connect')
        print(conn_err)
        continue
    except Exception as e:
        print(e)
        break
    finally:
        img_num += 1
