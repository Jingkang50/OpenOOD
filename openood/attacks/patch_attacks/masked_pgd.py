import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms

class NormalizeWrapper:
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, input):
        normalized_input = self.normalize(input)
        return self.model(normalized_input)

def create_patch(patch_size):
    patch = torch.zeros((3, patch_size, patch_size))
    patch[:, patch_size//4 : patch_size//4*3, patch_size//4 : patch_size//4*3] = 1
    return patch

def masked_pgd_attack(model, input_image, target_class, epsilon, alpha, num_steps, patch_size):
    # Create a copy of the input tensor for perturbation
    perturbed_image = input_image.clone().detach().requires_grad_(True)

    # Generate a random location for the patch
    image_size = input_image.size(-1)
    x = random.randint(0, image_size - patch_size)
    y = random.randint(0, image_size - patch_size)

    # Create the patch
    patch = create_patch(patch_size)

    # PGD attack loop
    for _ in range(num_steps):
        
        perturbed_image.requires_grad = True
        
        # Calculate the loss
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()(output, target_class)

        # Zero out the gradients
        model.model.zero_grad()
        perturbed_image.grad = None

        # Calculate the gradients
        loss.backward()

        # Apply the perturbation with a constraint on epsilon only on the patch
        with torch.no_grad():
            perturbed_image[:, :, x:x+patch_size, y:y+patch_size] += alpha * perturbed_image.grad[:, :, x:x+patch_size, y:y+patch_size].sign()

            perturbed_image[:, :, x:x+patch_size, y:y+patch_size] = torch.clamp(perturbed_image[:, :, x:x+patch_size, y:y+patch_size], input_image[:, :, x:x+patch_size, y:y+patch_size] - epsilon, input_image[:, :, x:x+patch_size, y:y+patch_size] + epsilon)
            # else:
            #     perturbed_image[:, :, x:x+patch_size, y:y+patch_size] = torch.clamp(perturbed_image[:, :, x:x+patch_size, y:y+patch_size], input_image[:, :, x:x+patch_size, y:y+patch_size], input_image[:, :, x:x+patch_size, y:y+patch_size])
            
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()
        
    pred = torch.max(model(perturbed_image), dim=1)[1]
    success = ~(target_class == pred)
    
    # Return the perturbed image
    return  perturbed_image, success 