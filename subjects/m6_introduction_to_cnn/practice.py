import torch
import torch.nn as nn
import gdown
import os
import pandas as pd
import numpy as np
import cv2

torch.manual_seed(42)


def line_break(heading_text):
    print('-' * 8 + heading_text + '-' * 8)

if __name__ == "__main__":
    # Example 1
    line_break(heading_text='Example 1')
    input = torch.randint(5, (1, 6, 6), dtype=torch.float32)

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False
    )

    print(conv_layer.weight)

    init_kernel_weight = torch.randint(
        high=2,
        size=(conv_layer.weight.data.shape),
        dtype=torch.float32
    )

    print(init_kernel_weight)

    conv_layer.weight.data = init_kernel_weight
    print(conv_layer.weight)
    print(conv_layer.bias)

    output = conv_layer(input)
    print(input.shape)
    print(output)

    # Example 2
    line_break('Example 2')
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3
    )

    print(conv_layer.weight)

    conv_layer.weight.data = init_kernel_weight
    print(conv_layer.weight)
    print(conv_layer.bias)

    conv_layer.bias = nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )

    print(conv_layer.bias)

    output = conv_layer(input)
    print(output)

    # Example 3
    line_break(heading_text='Example 3')
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 3)
    )

    init_kernel_weight = torch.randint(
        high=2,
        size=(conv_layer.weight.data.shape),
        dtype=torch.float32
    )

    conv_layer.weight.data = init_kernel_weight

    conv_layer.bias = nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )

    print(conv_layer.weight)
    print(conv_layer.bias)

    output = conv_layer(input)
    print(output)

    # Padding
    line_break(heading_text='Padding')
    line_break(heading_text='Example 1')

    input = torch.randint(5, (1, 4, 4), dtype=torch.float32)
    init_kernel_weight = torch.randint(
        high=2,
        size=(conv_layer.weight.data.shape),
        dtype=torch.float32
    )
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding='same'
    )
    conv_layer.weight.data = init_kernel_weight
    conv_layer.bias = nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )
    print(conv_layer.weight)
    print(conv_layer.bias)
    output = conv_layer(input)
    print(output)

    line_break('Example 2')
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=(2, 1)
    )
    conv_layer.weight.data = init_kernel_weight
    conv_layer.bias = nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )
    print(conv_layer.weight)
    print(conv_layer.bias)
    output = conv_layer(input)
    print(output)

    # Stride
    line_break(heading_text='Stride')
    line_break(heading_text='Example 1')

    input = torch.randint(5, (1, 6, 6), dtype=torch.float32)
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2
    )
    conv_layer.weight.data = init_kernel_weight
    conv_layer.bias =  nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )
    output = conv_layer(input)
    print(output)

    line_break(heading_text='Example 2')

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=(1, 2)
    )

    conv_layer.weight.data = init_kernel_weight
    conv_layer.bias =  nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )
    output = conv_layer(input)
    print(output)

    line_break(heading_text='Example 3')
    input = torch.randint(5, (1, 4, 4), dtype=torch.float32)
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        stride=(2, 2)
    )

    conv_layer.weight.data = init_kernel_weight
    conv_layer.bias =  nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )
    output = conv_layer(input)
    print(output)

    line_break(heading_text='Example 4')
    input = torch.randint(5, (1, 5, 7), dtype=torch.float32)
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        stride=(2, 2)
    )

    output = conv_layer(input)
    print(output)

    # Pooling Layer
    line_break(heading_text='Pooling Layer')
    line_break(heading_text='Max Pooling Layer')
    line_break(heading_text='Example 1')

    input = torch.randint(5, (1, 6, 6), dtype=torch.float32)
    max_pool_layer = nn.MaxPool2d(kernel_size=2)
    output = max_pool_layer(input)
    print(input)
    print(output)

    line_break(heading_text='Example 2')
    max_pool_layer = nn.MaxPool2d(
        kernel_size=2,
        stride=(1, 2)
    )
    output = max_pool_layer(input)
    print(input)
    print(output)

    line_break(heading_text='Example 3')
    max_pool_layer = nn.MaxPool2d(
        kernel_size=(2, 3),
        stride=(2, 2),
        padding=1
    )
    output = max_pool_layer(input)
    print(input)
    print(output)

    line_break(heading_text='Example 4: Max Pooling 1D')

    max_pool_layer = nn.MaxPool1d(
        kernel_size=3,
        stride=3
    )
    output = max_pool_layer(input)
    print(input)
    print(output)

    line_break(heading_text='Average Pooling Layer')
    line_break(heading_text='Example 1')

    avg_pool_layer = nn.AvgPool2d(
        kernel_size=(3, 2),
        stride=(2, 2)
    )

    output = avg_pool_layer(input)
    print(input.shape)
    print(output)

    line_break(heading_text='Example 2')

    avg_pool_layer = nn.AvgPool1d(
        kernel_size=3,
        stride=3
    )

    output = avg_pool_layer(input)
    print(input.shape)
    print(output)

    line_break(heading_text='Flatten')

    input = torch.randint(5, (1, 3, 2), dtype=torch.float32)
    flatten_layer = nn.Flatten()
    output = flatten_layer(input)
    print(output)

    line_break(heading_text='Exercise')
    line_break(heading_text='Ex01')

    input = torch.tensor([[
        [2., 2., 1., 4., 1., 0.],
        [0., 4., 0., 3., 3., 4.],
        [0., 4., 1., 2., 0., 0.],
        [2., 1., 4., 1., 3., 1.]
    ]])

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False
    )

    init_kernel_weight = torch.tensor(
         [[[[1., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]]]]
    )

    conv_layer.weight.data = init_kernel_weight
    output = conv_layer(input)
    print(output)

    conv_layer.bias = nn.Parameter(
        torch.tensor([2], dtype=torch.float32)
    )

    output = conv_layer(input)
    print(output)

    line_break(heading_text='Ex02')

    input = torch.tensor(
          [[[2., 4., 2.],
            [3., 3., 4.],
            [3., 2., 0.],
            [4., 0., 4.],
            [1., 4., 0.]]]
    )
    print(input.shape)

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False,
        padding=1
    )

    print(conv_layer.weight)

    init_kernel_weight = torch.tensor(
         [[[[1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 0.]]]]
    )
    print(init_kernel_weight.shape)

    conv_layer.weight.data = init_kernel_weight

    conv_layer.bias =  nn.Parameter(
        torch.tensor([2], dtype=torch.float32)
    )
    output = conv_layer(input)
    print(output)

    conv_layer.stride = (2, 2)
    output = conv_layer(input)
    print(output)

    line_break(heading_text='Ex03: Conv + Pooling')

    input = torch.tensor(
          [[[2., 4., 2.],
            [1., 3., 2.],
            [3., 2., 1.],
            [0., 0., 1.],
            [0., 0., 1.]]]
    )

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 2),
        bias=False,
    )

    init_kernel_weight = torch.tensor(
         [[[[1., 1.],
            [1., 0.],
            [0., 0.]]]]
    )
    conv_layer.weight.data = init_kernel_weight

    conv_layer.bias =  nn.Parameter(
        torch.tensor([1], dtype=torch.float32)
    )

    max_pool_layer = nn.MaxPool2d(kernel_size=(1, 2))
    avg_pool_layer = nn.AvgPool2d(kernel_size=(1, 2))

    output = conv_layer(input)
    print(f'Output after conv: {output}')

    max_pooling_output = max_pool_layer(output)
    print(f'Output after max pooling: {max_pooling_output}')

    avg_pooling_output = avg_pool_layer(output)
    print(f'Output after avg pooling: {avg_pooling_output}')

    line_break(heading_text='Ex04: Convolutions on Gray Scale Images')

    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 3),
        bias=False,
    )

    base_path = os.path.join('/')

    gdown.download(id='12Q7SpIVm4W2hT6QsiEuARLl5xXKMK8pE',
                   output=base_path)

    file_path = os.path.join(base_path, 'test.csv')
    test_df = pd.read_csv(file_path)
    print(test_df.shape)

    img = np.array(test_df.iloc[1].values).astype(np.uint8).reshape((28, 28))
    print(img.shape)

    resized_img = cv2.resize(img, (7, 7))
    print(resized_img.shape)
    print(resized_img)

    tensor_img = torch.tensor(resized_img).resize(1, 1, 7, 7).float()

    kernel_weight = nn.Parameter(torch.tensor(
         [[[[1., 0. , 1.],
            [1., 0., -1.],
            [1., 0., -1.]]]]
    ))

    conv_layer.weight = kernel_weight

    max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
    output = max_pool_layer(tensor_img)
    print(output)

    output = conv_layer(tensor_img)
    print(output)

    print(max_pool_layer(output))
