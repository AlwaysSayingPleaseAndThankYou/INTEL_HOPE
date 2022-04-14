import matplotlib
import numpy as np
from pathlib import Path
from collections import OrderedDict
from models.hopenet import HopeNet
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
plt.ion()

def loaded_model():
    hope = HopeNet()
    p = Path("/Users/aenguscrowley/PycharmProjects/REAL_HOPE/checkpoints/Feb_26.pkl375.pkl")
    bad_dict = torch.load(p, map_location=torch.device('cpu'))

    good_dict = OrderedDict({key[len('module.'):]: value for key, value in bad_dict.items()})
    hope.load_state_dict(good_dict)
    return hope


def classify(model, image: Path, model_transform):
    model = model.eval()
    image = Image.open(image)
    image = model_transform(image).float()
    image = image.unsqueeze(0)
    try:
        return model(image)
    except Exception as e:
        print("#" * 10 + "image" + "#" * 10)
        print(image)
        print("#" * 10 + "error" + "#" * 10)


def plot_from_single_image(classified_result, title, fig):
    """
        note on assigning colors: the final 8 points indicate corners of the object, where the other 21
        points are the vertices describing the hand, currently operating on the guess that point 0 is the wrist point
        and that every collection of 4 points after point 0 within the hand describes each finger, so
        [
            point 0: "wrist"
            points 1-4: first finger (currently presumed to be the thumb)
            points 5-8: second finger
            points 9-12: third finger
            points 13-16: fourth finger
            points 17-20: fifth finger (currently presumed to be the pinky)
            points 21-28: object
        ]
    """
    plt_colors = [ \
        [ \
            0.5 if i == 0 else 1.0 if (i > 0 and i <= 8) or (i > 16 and i <= 20) else 0.0, \
            0.5 if i == 0 else 1.0 if i > 4 and i <= 12 else 0.0, \
            0.5 if i == 0 else 1.0 if i > 12 and i <= 20 else 0.0] \
        for i in range(0, 29)]
    # twoD = fig.add_subplot(2, 1, 1)
    threeD = fig.add_subplot(projection='3d')
    threeD.set_title(label=title)
    # split data
    datatwoD = classified_result[0].detach().numpy()[0]
    dataThreeD = classified_result[2].detach().numpy()[0]

    # 2d plots
    # datatwoDx = [d[0] for d in datatwoD]
    # datatwoDy = [d[1] for d in datatwoD]

    # for i, point in enumerate(zip(datatwoDx, datatwoDy)):
    #     twoD.scatter(point[0], point[1], c=plt_colors[i])
    # 3d plot
    dataThreeDx = [d[0] for d in dataThreeD]
    dataThreeDy = [d[1] for d in dataThreeD]
    dataThreeDz = [d[2] for d in dataThreeD]
    ThreeDPoints = [[dataThreeDx[i], dataThreeDy[i], dataThreeDz[i]] for i in range(0, len(dataThreeDx))]
    print(ThreeDPoints)
    # plot resulting vertices
    for i, point in enumerate(ThreeDPoints):
        threeD.scatter(point[0], point[1], point[2], c=plt_colors[i])
    # plot line segments to properly connect vertices together
    # connect hand segments together
    for i in range(1, 21, 4):
        threeD.plot( \
            [ThreeDPoints[0][0], ThreeDPoints[i][0], ThreeDPoints[i + 1][0], ThreeDPoints[i + 2][0],
             ThreeDPoints[i + 3][0]], \
            [ThreeDPoints[0][1], ThreeDPoints[i][1], ThreeDPoints[i + 1][1], ThreeDPoints[i + 2][1],
             ThreeDPoints[i + 3][1]], \
            [ThreeDPoints[0][2], ThreeDPoints[i][2], ThreeDPoints[i + 1][2], ThreeDPoints[i + 2][2],
             ThreeDPoints[i + 3][2]], \
            c=plt_colors[i])
    # connect object points together
    for i in range(22, 29, 2):
        threeD.plot( \
            [ThreeDPoints[i - 1][0], ThreeDPoints[i][0]], \
            [ThreeDPoints[i - 1][1], ThreeDPoints[i][1]], \
            [ThreeDPoints[i - 1][2], ThreeDPoints[i][2]], \
            c=plt_colors[i])
    for i in list(range(23, 25)) + list(range(27, 29)):
        threeD.plot( \
            [ThreeDPoints[i - 2][0], ThreeDPoints[i][0]], \
            [ThreeDPoints[i - 2][1], ThreeDPoints[i][1]], \
            [ThreeDPoints[i - 2][2], ThreeDPoints[i][2]], \
            c=plt_colors[i])
    for i in range(25, 29):
        threeD.plot( \
            [ThreeDPoints[i - 4][0], ThreeDPoints[i][0]], \
            [ThreeDPoints[i - 4][1], ThreeDPoints[i][1]], \
            [ThreeDPoints[i - 4][2], ThreeDPoints[i][2]], \
            c=plt_colors[i])
    threeD.view_init(0, 0)
    threeD.set_title(label=title)
    return fig
    # plt.draw()
    # plt.pause(0.001)
    # return fig


applied_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class analyze_set:

    def __init__(self, directory, fig, model):
        """

        Parameters
        ----------
        directory : PATH to directory with files
        fig : matplotlib figure
        model: loaded HOPE model
        """
        self.directory = directory
        self.files = os.listdir(self.directory).__iter__()
        self.model = model
        self.fig = fig

    def single_image_data(self, result):
        """
        Unfinished. Attempting to extract the data from run on single image. in a way that preserves daniels lines. Totally not sure how this works.

        Parameters
        ----------
        result

        Returns
        -------

        """

        plt_colors = [[0.5 if i == 0 else 1.0 if (0 < i <= 8) or (16 < i <= 20) else 0.0,
                       0.5 if i == 0 else 1.0 if 4 < i <= 12 else 0.0,
                       0.5 if i == 0 else 1.0 if 12 < i <= 20 else 0.0] for i in range(0, 29)]
        data_three_d = result[2].detach().numpy()[0]
        data_three_dx = [d[0] for d in data_three_d]
        data_three_dy = [d[1] for d in data_three_d]
        data_three_dz = [d[2] for d in data_three_d]
        three_d_points = [[data_three_dx[i], data_three_dy[i], data_three_dz[i]] for i in range(0, len(data_three_dx))]
        data1 = []
        for i in range(1, 21, 4):
            data1.append([[three_d_points[0][0], three_d_points[i][0], three_d_points[i + 1][0], three_d_points[i + 2][0],
                           three_d_points[i + 3][0]],
                          [three_d_points[0][1], three_d_points[i][1], three_d_points[i + 1][1], three_d_points[i + 2][1],
                           three_d_points[i + 3][1]],
                          [three_d_points[0][2], three_d_points[i][2], three_d_points[i + 1][2], three_d_points[i + 2][2],
                           three_d_points[i + 3][2]]])

    def animate(self):
        """
        to be passed to matplotlib.animate
        runs run_on_single_image on single image from stream/directory
        Returns
        -------

        """
        single_image = self.directory / Path(next(self.files))
        classified = classify(self.model, single_image, applied_transform)
        pass


def nope():
    fig = plt.figure()
    hope = loaded_model()

    hand_data = Path('/Users/aenguscrowley/PycharmProjects/stream_from_esp32_cam/grabbed_photos')
    analyse = analyze_set(hand_data, fig, model=hope)
    print(os.listdir(hand_data))

    flatplot = fig.add_subplot(2, 1, 1)
    threedplot = fig.add_subplot(2, 1, 2)
    hands_dict = {'pic_path': Path("orig_hands/test_hand_with_pliers.jpg"),
                  'pic_path_2': Path('orig_hands/dad_hand_with_knife.jpeg'),
                  'ex_hand': Path('orig_hands/example_hand.jpg'), 'vince_hand': Path('orig_hands/vince_hand.jpeg'),
                  'luke_hand': Path('orig_hands/luke_hand.jpeg'), 'image_8': Path('orig_hands/image8.jpeg')}

    num_photos = len(os.listdir(hand_data))
    for i in range(1, num_photos):
        image = f"image{i}.jpeg"
        file = hand_data / Path(image)
        file = hands_dict['ex_hand']
        out = classify(hope, file, applied_transform)
        ret = plot_from_single_image(
            out, title=image, fig=fig)
        plt.pause(0.001)
        ret.canvas.draw()
        # ret.savefig('test.png')

def run(fpath, fig):
    out = classify(hope, fpath, applied_transform)
    ret = plot_from_single_image(hope, )