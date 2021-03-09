import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
import dnnlib
import os

class DenseNet(nn.Module):
    def __init__(
            self,
            in_channels,
            expansion_rate,
            num_layers,
            kernel_size,
            bottleneck_rate,
            segmentation_channels
    ):
        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        self.expansion_rate = expansion_rate
        self.num_layers = num_layers
        self.bottleneck_rate = bottleneck_rate
        self.kernel_size = kernel_size
        self.segmentation_channels = segmentation_channels

        self.layers = nn.ModuleDict()

        current_channels = in_channels

        self.layers['first_bn'] = nn.BatchNorm2d(num_features=in_channels)

        for layer in range(num_layers):

            if current_channels > 4 * expansion_rate:

                self.layers['bottleneck' + str(layer)] = nn.Conv2d(in_channels=current_channels,
                                                                   out_channels=bottleneck_rate * expansion_rate,
                                                                   kernel_size=1, padding=0)

                self.layers['conv' + str(layer)] = nn.Conv2d(
                    in_channels=bottleneck_rate * expansion_rate, out_channels=expansion_rate,
                    kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

            else:
                self.layers['conv' + str(layer)] = nn.Conv2d(
                    in_channels=current_channels, out_channels=expansion_rate,
                    kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

            self.layers['bn' + str(layer)] = nn.BatchNorm2d(num_features=expansion_rate)

            current_channels += expansion_rate

        self.layers['final'] = nn.Conv2d(
            in_channels=current_channels, out_channels=segmentation_channels,
            kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):

        out = x

        out = self.layers['first_bn'](out)

        for layer in range(self.num_layers):

            if 'bottleneck' + str(layer) in self.layers.keys():
                tmp_out = self.layers['bottleneck' + str(layer)](out)
                tmp_out = self.layers['conv' + str(layer)](tmp_out)
            else:
                tmp_out = self.layers['conv' + str(layer)](out)
            tmp_out = self.layers['bn' + str(layer)](tmp_out)
            tmp_out = torch.relu(tmp_out)

            out = torch.cat((out, tmp_out), dim=1)

        out = self.layers['final'](out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            kernel_size,
            segmentation_channels
    ):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.segmentation_channels = segmentation_channels

        self.layers = nn.ModuleList()

        current_channels = in_channels

        self.layers.append(nn.BatchNorm2d(num_features=in_channels))

        for out_channels in channels:
            self.layers.append(nn.Conv2d(
                in_channels=current_channels, out_channels=out_channels,
                kernel_size=kernel_size, padding=(kernel_size - 1) // 2))

            self.layers.append(nn.BatchNorm2d(num_features=out_channels))

            self.layers.append(nn.ReLU())

            current_channels = out_channels

        self.layers.append(nn.Conv2d(
            in_channels=current_channels, out_channels=segmentation_channels,
            kernel_size=kernel_size, padding=(kernel_size - 1) // 2))

    def forward(self, x):

        out = x

        for layer in self.layers:
            out = layer(out)

        return out


def print_stats(batch, trainLoss, acc, parameters, time):
    s = 0
    for parameter in parameters:
        s += torch.abs(parameter.grad).mean().detach().cpu()

    print('Batch {} in {:.1f}s | Loss {:.4f} | Acc {:.3f} | Grad {:.1f}'.format(
        batch, time, trainLoss, acc, np.log10(s)))


def main():

    dnnlib.util.Logger(should_flush=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    print('Loading data...')
    trainingData = np.load('preprocessed-datasets/celebmask-train.npy')
    testingData = np.load('preprocessed-datasets/celebmask-test.npy')

    criterion = nn.CrossEntropyLoss()
    train_batch_size = 8
    test_batch_size = 16
    num_train_batches = int(np.ceil(trainingData.shape[0] / train_batch_size))
    num_test_batches = int(np.ceil(testingData.shape[0] / test_batch_size))

    architecture = 'dense'

    path = os.getcwd() + '/networks/' + architecture + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    i = 0
    while os.path.exists(path + str(i)):
        i += 1

    os.makedirs(path + str(i))

    write_prefix = path + str(i) + '/'

    # channels = [8, 32, 64, 128, 256, 128, 64, 64, 32]
    expansion_rate = 12
    bottleneck_rate = 4
    num_layers = 14
    in_channels = 3
    segmentation_channels = 3
    kernel_size = 5
    numBatchesPerStep = 32
    lr = 5 * 1e-4
    # model = ResNet(in_channels=in_channels, channels=channels, kernel_size=kernel_size, segmentation_channels=segmentation_channels)
    model = DenseNet(in_channels=in_channels, expansion_rate=expansion_rate, bottleneck_rate=bottleneck_rate, num_layers=num_layers, kernel_size=kernel_size, segmentation_channels=segmentation_channels)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1, threshold=1e-3, eps=1e-6, verbose=True)

    with open(write_prefix + 'train-config.txt', 'w') as file:
        # file.write('channels ' + str(channels) + '\n')
        file.write('expansion_rate ' + str(expansion_rate) + '\n')
        file.write('bottleneck_rate ' + str(bottleneck_rate) + '\n')
        file.write('num_layers ' + str(num_layers) + '\n')
        file.write('kernel_size ' + str(kernel_size) + '\n')
        file.write('batch_size ' + str(train_batch_size) + '\n')
        file.write('numBatchesPerStep ' + str(numBatchesPerStep) + '\n')
        file.write('lr' + str(lr) + '\n')


    testLosses = []
    testAcc = []
    trainLosses = []
    trainAcc = []

    for epoch in range(10):

        print('Starting Epoch {}'.format(epoch))
        epoch_t0 = time.time()

        testLosses.append(0)
        testAcc.append(0)

        model.eval()
        with torch.no_grad():

            for batch in range(num_test_batches):
                x = torch.from_numpy(
                    testingData[batch * test_batch_size: (batch + 1) * test_batch_size, :3]).float().to(device)
                y = torch.from_numpy(testingData[batch * test_batch_size: (batch + 1) * test_batch_size, 3]).long().to(
                    device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                argmax = torch.argmax(y_hat, dim=1)
                acc = (torch.sum(torch.eq(argmax, y)) / y.nelement()).cpu().detach()

                testLosses[-1] += loss.item()
                testAcc[-1] += acc

            testLosses[-1] /= num_test_batches
            testAcc[-1] /= num_test_batches

            print('Test Loss {:.4f} Acc {:.4f}'.format(testLosses[-1], testAcc[-1]))

        permutation = np.random.permutation(trainingData.shape[0])
        trainingData = trainingData[permutation]

        lr_scheduler.step(testLosses[-1])

        model.train()
        for batch in range(num_train_batches):

            t0 = time.time()

            x = torch.from_numpy(trainingData[batch * train_batch_size: (batch + 1) * train_batch_size, :3]).float().to(
                device)
            y = torch.from_numpy(trainingData[batch * train_batch_size: (batch + 1) * train_batch_size, 3]).long().to(
                device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            argmax = torch.argmax(y_hat, dim=1)
            acc = (torch.sum(torch.eq(argmax, y)) / y.nelement()).cpu().detach()

            trainLosses.append(loss.item())
            trainAcc.append(acc)
            (loss / numBatchesPerStep).backward()

            if batch % numBatchesPerStep == (numBatchesPerStep - 1):
                optimizer.step()
                optimizer.zero_grad()

            t_final = time.time() - t0

            if batch % 80 == 0:
                print_stats(batch, loss.item(), acc, model.parameters(), t_final)

        x = None
        y = None
        y_hat = None
        argmax = None
        acc = None
        loss = None
        torch.cuda.empty_cache()

        torch.save(model, write_prefix + str(epoch) + '.pt')
        np.save(write_prefix + str(epoch) + '-testLosses', np.stack(testLosses))
        np.save(write_prefix + str(epoch) + '-trainLosses', np.stack(trainLosses))
        np.save(write_prefix + str(epoch) + '-testAcc', np.stack(testAcc))
        np.save(write_prefix + str(epoch) + '-trainAcc', np.stack(trainAcc))

        print('Epoch {} finished in {:.1f}'.format(epoch, time.time() - epoch_t0))


if __name__ == "__main__":

    main()








