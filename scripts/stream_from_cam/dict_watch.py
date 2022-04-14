from pathlib import Path
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

target = Path('/Users/aenguscrowley/PycharmProjects/stream_from_esp32_cam/grabbed_photos')
import run_on_single_image as rs

hope = rs.loaded_model()
transforms = rs.applied_transform
import matplotlib.pyplot as plt

fig = plt.figure()


class myeventhandler(FileSystemEventHandler):

    def on_created(self, event):
        print(event)
    def on_modified(self, event):
        title = event.key[1]
        title = title[74:]
        path = Path(event.src_path)
        time.sleep(0.1)
        if path.is_file():
            out = rs.classify(hope, path, transforms)
            ret = rs.plot_from_single_image(out, title=title, fig=fig)
            print(f'drawing image {title}')
            ret.show()
        else:
            print(event)
        time.sleep(1)


    def on_deleted(self, event):
        print(event)

    def on_any_event(self, event):
        print(event)


if __name__ == "__main__":
    meh = myeventhandler()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    event_handler = LoggingEventHandler()
    path = target
    observer = Observer()
    observer.schedule(meh, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except Exception as e:
        print(e)
    finally:
        observer.stop()
        observer.join()
# while True:
#     try:
#         old = os.listdir(target)
#         time.sleep(0.1)
#         new = os.listdir(target)
#         if all([(x in new) for x in old]):
#             print('no new files')
#         else:
#             for new_file in [x for x in new if x not in old]:
#                 print(f"new file is {new_file}")
#         time.sleep(0.01)
#     except KeyboardInterrupt:
#         print('quitting')
#         break
#     except Exception as e:
#         print(e)
#         break
