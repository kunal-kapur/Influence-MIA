DATA_DIR='/scratch/scholar/kapur16/data/'
import torchvision.transforms as transforms

TRANSFORM_TRAIN = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

TRANSFORM_TEST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
