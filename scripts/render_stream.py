import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import run_on_single_image as single
import time
import os
image_folder_path = Path("orig_hands")
#watcher init
watcher = os.listdir(image_folder_path)
#graph init
plt.ion()
fig = plt.figure()
flatplot = fig.add_subplot(2, 1, 1)
threedplot = fig.add_subplot(2, 1, 2)
hope = single.loaded_model()

try:
    while True:
        # check folder
        seeker = os.listdir(image_folder_path)
        if seeker[-1] != watcher[-1]:
            new_image_path = Path(seeker[-1])
            model_output = single.classify(hope, new_image_path, single.applied_transform)

            print('updating canvas')
        # evaluate newest pic in model
        # model_output = single.classify(hope, current_image, single.applied_transform)
        # render
        watcher = seeker
        print(watcher)
        time.sleep(.25)
except KeyboardInterrupt:
        # stop excecution
    print('quitting')
finally:
    print('done')
# This is the redraw step
# current_image = ''
