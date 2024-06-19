import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def generate_mask(matrix_shape, percentage):
    total_elements = matrix_shape[0] * matrix_shape[1]
    num_ones = int(total_elements * percentage)

    # Create a flat array with the required number of ones and zeros
    mask_values = np.array([1] * num_ones + [0] * (total_elements - num_ones))

    # Shuffle the array to randomly distribute ones and zeros
    np.random.shuffle(mask_values)

    # Reshape the array to the desired matrix shape
    mask = mask_values.reshape(matrix_shape)
    return mask

def generate_three_masks(image_shape=(256,256), num_masks=3):
    total_elements = image_shape[0] * image_shape[1]
    num_ones_per_mask = total_elements // num_masks
    remaining_ones = total_elements % num_masks

    masks = []
    for i in range(num_masks):
        # Create a flat array with the required number of ones and zeros
        ones = num_ones_per_mask + (1 if i < remaining_ones else 0)
        mask_values = np.array([1] * ones + [0] * (total_elements - ones))

        # Shuffle the array to randomly distribute ones and zeros
        np.random.shuffle(mask_values)

        # Reshape the array to the desired image shape
        mask = mask_values.reshape(image_shape)
        masks.append(mask)

    return masks

def generate_torch_masks(image_shape, num_masks):
    total_elements = image_shape[0] * image_shape[1]
    num_ones_per_mask = total_elements // num_masks
    remaining_ones = total_elements % num_masks

    masks = []
    for i in range(num_masks):
        # Create a flat tensor with the required number of ones and zeros
        ones = num_ones_per_mask + (1 if i < remaining_ones else 0)
        mask_values = torch.tensor([1] * ones + [0] * (total_elements - ones), dtype=torch.float32).cuda()
        # Shuffle the tensor to randomly distribute ones and zeros
        mask_values = mask_values[torch.randperm(total_elements)]
        # Reshape the tensor to the desired image shape
        mask = mask_values.reshape(1,1,*image_shape)
        masks.append(mask)
    return masks

# Define the MLP architecture
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    # generate_mask
    matrix_shape = (256, 256)
    percentage = 0.33
    mask = generate_mask(matrix_shape, percentage)
    # generate three masks 1/3:
    masks = generate_three_masks(image_shape=(256,256), num_masks=3)
    torch_masks = generate_torch_masks(image_shape=(256,256), num_masks=3)

    # Set the input, hidden, and output sizes
    input_size = 784   # For example, a 28x28 grayscale image has 784 input features
    hidden_size = 128  # A hidden layer with 128 neurons
    output_size = 10   # Assuming 10 output classes (e.g., for the MNIST dataset)
    # Instantiate the MLP model
    net = SimpleMLP(input_size, hidden_size, output_size)



    x =  1