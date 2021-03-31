from torchvision.datasets import ImageFolder
from torchvision import transforms

# 加上transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])

dataset_train = ImageFolder('./data/train', transform=transform)
dataset_test = ImageFolder('./data/test', transform=transform)

