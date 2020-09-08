
import torchvision.models as models
import torch.nn as nn
import copy
l1_loss = nn.L1Loss()
def get_model_and_losses(generate_img,target_img):
    layers_default = ['conv_2', 'conv_4', 'conv_6', 'conv_10', 'conv_14']
    vgg = models.vgg19(pretrained=True).features.eval()
    vgg.features[0] = nn.Conv2d(1, 256, kernel_size=3, padding=1)
    model = nn.Sequential()
    vgg = copy.deepcopy(vgg)
    perceptual_loss=[]
    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # 试下把这个去掉的效果
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'maxpool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name in layers_default:
            generate_result = model(generate_img).detach()
            target_result = model(target_img).detach()
            perceptual_loss_i = l1_loss(generate_result,target_result)
            perceptual_loss.append(perceptual_loss_i)
    perceptual_loss_score=0
    for sl in perceptual_loss:
        perceptual_loss_score +=sl


    return  perceptual_loss


