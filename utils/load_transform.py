import torchvision.transforms as T

def load_transform(size=512):
    transform_list = []
    if size != 0:
        transform_list.append(T.Resize([size, size]))
    transform_list.append(T.ToTensor())
    transform = T.Compose(transform_list)
    return transform
