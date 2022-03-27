import numpy as np
from pathlib import Path
from collections import OrderedDict
from models.hopenet import HopeNet
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def loaded_model():
    hope = HopeNet()
    p = Path("..", "checkpoints", "Feb_26.pkl375.pkl").resolve()
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


def plot_from_single_image(classified_result):
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
    plt_colors = [\
        [\
        0.5 if i == 0 else 1.0 if (i > 0 and i <= 8) or (i > 16 and i <= 20) else 0.0, \
        0.5 if i == 0 else 1.0 if i > 4 and i <= 12 else 0.0, \
        0.5 if i == 0 else 1.0 if i > 12 and i <= 20 else 0.0] \
        for i in range(0,29)]
    # figures
    fig = plt.figure()
    twoD = fig.add_subplot(2, 1, 1)
    threeD = fig.add_subplot(2, 1, 2, projection='3d')
    # split data
    datatwoD = classified_result[0].detach().numpy()[0]
    dataThreeD = classified_result[2].detach().numpy()[0]

    # 2d plots
    datatwoDx = [d[0] for d in datatwoD]
    datatwoDy = [d[1] for d in datatwoD]


    for i, point in enumerate(zip(datatwoDx, datatwoDy)):
        twoD.scatter(point[0], point[1], c=plt_colors[i])
    # 3d plot
    dataThreeDx = [d[0] for d in dataThreeD]
    dataThreeDy = [d[1] for d in dataThreeD]
    dataThreeDz = [d[2] for d in dataThreeD]
    ThreeDPoints = [[dataThreeDx[i], dataThreeDy[i], dataThreeDz[i]] for i in range(0,len(dataThreeDx))]
    print(ThreeDPoints)
    # plot resulting vertices
    for i, point in enumerate(ThreeDPoints):
        threeD.scatter(point[0], point[1], point[2], c=plt_colors[i])
    # plot line segments to properly connect vertices together
    # connect hand segments together
    for i in range(1, 21, 4):
        threeD.plot(\
            [ThreeDPoints[0][0], ThreeDPoints[i][0], ThreeDPoints[i+1][0], ThreeDPoints[i+2][0], ThreeDPoints[i+3][0]],\
            [ThreeDPoints[0][1], ThreeDPoints[i][1], ThreeDPoints[i+1][1], ThreeDPoints[i+2][1], ThreeDPoints[i+3][1]],\
            [ThreeDPoints[0][2], ThreeDPoints[i][2], ThreeDPoints[i+1][2], ThreeDPoints[i+2][2], ThreeDPoints[i+3][2]],\
            c=plt_colors[i])
    # connect object points together
    for i in range(22, 29, 2):
        threeD.plot(\
            [ThreeDPoints[i-1][0], ThreeDPoints[i][0]],\
            [ThreeDPoints[i-1][1], ThreeDPoints[i][1]],\
            [ThreeDPoints[i-1][2], ThreeDPoints[i][2]],\
            c=plt_colors[i])
    for i in list(range(23, 25)) + list(range(27, 29)):
        threeD.plot(\
            [ThreeDPoints[i-2][0], ThreeDPoints[i][0]],\
            [ThreeDPoints[i-2][1], ThreeDPoints[i][1]],\
            [ThreeDPoints[i-2][2], ThreeDPoints[i][2]],\
            c=plt_colors[i])
    for i in range(25, 29):
        threeD.plot(\
            [ThreeDPoints[i-4][0], ThreeDPoints[i][0]],\
            [ThreeDPoints[i-4][1], ThreeDPoints[i][1]],\
            [ThreeDPoints[i-4][2], ThreeDPoints[i][2]],\
            c=plt_colors[i])
    threeD.view_init(45, 60)
    plt.draw()
    plt.pause(0.001)
    return fig


if __name__ == "__main__":
    applied_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])

    pic_path = Path("orig_hands/test_hand_with_pliers.jpg")
    pic_path_2 = Path('orig_hands/dad_hand_with_knife.jpeg')
    ex_hand = Path('orig_hands/example_hand.jpg')
    vince_hand = Path('orig_hands/vince_hand.jpeg')
    luke_hand = Path('orig_hands/luke_hand.jpeg')
    image_8 = Path('orig_hands/image8.jpeg')

    hope = loaded_model()
    out = classify(hope, image_8, applied_transform)
    ret = plot_from_single_image(out)
    ret.show()
    ret.savefig('test.png')
