import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from .base_modifier import BaseModifier

class AdversarialPerturbation(BaseModifier):
    def apply(self, input_image_path: str, output_image_path: str) -> None:
        """Applies adversarial perturbation to the input image."""

        # load pre-trained model (e.g., ResNet)
        model = models.resnet50(pretrained=True)
        model.eval()  # Set model to evaluation mode
        
        # load and preprocess the image
        image = cv2.imread(input_image_path)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # convert the image to a tensor
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # aAdd batch dimension

        # make the input tensor require gradients for perturbation
        input_var = torch.autograd.Variable(input_tensor, requires_grad=True)

        # define a loss function (e.g., CrossEntropyLoss)
        criterion = nn.CrossEntropyLoss()

        # create a fake target to "fool" the model (e.g., class index 0)
        target = torch.tensor([0])

        # forward pass through the model
        output = model(input_var)

        # compute the loss with respect to the fake target
        loss = criterion(output, target)
        
        # backpropagate to compute the gradient of the loss w.r.t. the input image
        model.zero_grad()
        loss.backward()

        # FGSM: create a perturbation by adding a small epsilon to the sign of the gradient
        epsilon = 0.01  # small perturbation factor
        perturbation = epsilon * input_var.grad.sign()

        # add the perturbation to the original image
        perturbed_image = input_var + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # clamp the values to be valid pixel values

        # convert the perturbed image back to a NumPy array and save it
        perturbed_image_np = perturbed_image.squeeze().detach().numpy()
        perturbed_image_np = np.transpose(perturbed_image_np, (1, 2, 0))  # reorder channels
        
        # convert the image back from normalized range [0, 1]
        perturbed_image_np = perturbed_image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        perturbed_image_np = np.clip(perturbed_image_np, 0, 1)  # ensure values are within the valid range [0, 1]
        perturbed_image_np = (perturbed_image_np * 255).astype(np.uint8)  # scale to 8-bit (0-255) values

        # save the perturbed image
        cv2.imwrite(output_image_path, perturbed_image_np)
