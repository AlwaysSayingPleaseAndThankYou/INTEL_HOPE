import numpy as np
from pathlib import Path
from collections import OrderedDict
from models.hopenet import HopeNet
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt



def imports():
    import numpy as np
    from pathlib import Path
    from collections import OrderedDict
    from models.hopenet import HopeNet
    from PIL import Image
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt
    return None


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


def plot_from_single_image(classified_result):
    # figures
    fig = plt.figure()
    twoD = fig.add_subplot(3, 1, 1 )
    twoDQ = fig.add_subplot(3, 1, 2)
    threeD = fig.add_subplot(3, 1, 3, projection='3d')
    # split data
    datatwoD = classified_result[0].detach().numpy()[0]
    datatwoDQ = classified_result[1].detach().numpy()[0]
    dataThreeD = classified_result[2].detach().numpy()

    # 2d plots
    datatwoDx = [d[0] for d in datatwoD]
    datatwoDy = [d[1] for d in datatwoD]

    datatwoDqx = [d[0] for d in datatwoD]
    datatwoDqy = [d[1] for d in datatwoD]

    for point in zip(datatwoDx, datatwoDy):
        twoD.scatter(point[0], point[1], c=np.random.rand(3, ))
    for point in zip(datatwoDqx, datatwoDqy):
        twoDQ.scatter(point[0], point[1], c=np.random.rand(3, ))

    return fig

if __name__ == "__main__":
    applied_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
    pic_path = Path( "test_hand_with_pliers.jpg")
    hope = loaded_model()
    out = classify(hope, pic_path, applied_transform)
    ret = plot_from_single_image(out)
    ret.show()