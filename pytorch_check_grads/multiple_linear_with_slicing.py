import torch
import torch.nn as nn
from torchviz import make_dot


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.ln1 = nn.Linear(in_features=3, out_features=3)
        self.ln2 = nn.Linear(in_features=3, out_features=3)
        self.ln3 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        x1, x2 = x
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)

        # do slicing
        x1_1 = torch.unsqueeze(x1[0, 1], 0)
        x2_1 = torch.unsqueeze(x2[0, 1], 0)

        x3 = torch.unsqueeze(torch.cat((x1_1, x2_1)), 0)

        out = self.ln3(x3)

        return out


# function for visualizing compute tree
def print_compute_tree(model_name, node):
    dot = make_dot(node)
    # print(dot)
    dot.render(model_name, format='png')


def main():
    # Instantiate a ThreeLinearNet
    model = LinearNet()

    # Get each linear network's weights and change them
    print('_Defining the initial weights_')
    print('Linear layer 1')
    model.ln1.weight = nn.Parameter(torch.tensor([[0.2, 5, 9], [1, -6, 7], [3, 2, 9]]))
    model.ln1.bias = nn.Parameter(torch.tensor([[-0.1, 6, 7]]))
    print(f'Initial weights: {model.ln1.weight}\n')
    print(f'Initial bias: {model.ln1.bias}\n')

    print('Linear layer 2')
    model.ln2.weight = nn.Parameter(torch.tensor([[0.2, 5, 9], [1, -6, 7], [3, 2, 9]]))
    model.ln2.bias = nn.Parameter(torch.tensor([[-0.1, 6, 7]]))
    print(f'Initial weights: {model.ln2.weight}\n')
    print(f'Initial bias: {model.ln2.bias}\n')

    print('Linear layer 3')
    model.ln3.weight = nn.Parameter(torch.tensor([[0.8, 0.9], [1.5, 7.3]]))
    model.ln3.bias = nn.Parameter(torch.tensor([[-0.1, 0.5]]))
    print(f'Initial weights: {model.ln3.weight}\n')
    print(f'Initial bias: {model.ln3.bias}\n')

    # Register hooks
    print('_Registering hooks_')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            param.register_hook(lambda grad: print(grad))

    # define input
    x_input = (torch.tensor([[3., 1., 5.]]), torch.tensor([[2., 6., 9.]]))

    # do forward pass
    print('\n_Do forward pass_')
    pred = model(x_input)
    print(f'pred: {pred}\n')

    # Visualize the compute tree
    print_compute_tree('linear', pred)

    # backward pass
    print('_Do backward pass_')
    pred.backward()

    # print out parameters manually
    print('\n_Print out parameters manually_')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'name: {name}')
            print(f'data: {param.data}')
            print(f'grad: {param.grad}\n')


if __name__ == '__main__':
    main()
