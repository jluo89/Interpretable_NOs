from utils.utilities3 import *

class Wave:
    def __init__(self, training_properties):
        ntrain = training_properties.ntrain
        ntest = training_properties.ntest
        batch_size = training_properties.batch_size
        r = training_properties.r
        s = training_properties.s
        nplot = training_properties.nplot
        t = training_properties.t

        
        TRAIN_PATH = f'./data/wave_64x64_in_{t}.mat'
        TEST_PATH = f'./data/wave_64x64_in_{t}.mat'
        ################################################################
        # load data and data normalization
        ################################################################
        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('x')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('y')[:ntrain,::r,::r][:,:s,:s]

        reader.load_file(TEST_PATH)
        x_test = reader.read_field('x')[-ntest:,::r,::r][:,:s,:s]
        y_test = reader.read_field('y')[-ntest:,::r,::r][:,:s,:s]


        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = self.y_normalizer.encode(y_train)

        x_train = x_train.reshape(ntrain,1,s,s)
        x_test = x_test.reshape(ntest,1,s,s)
        y_train = y_train.reshape(ntrain,1,s,s)
        y_test = y_test.reshape(ntest,1,s,s)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False,drop_last=True)
        self.plot_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test[:nplot], y_test[:nplot]), batch_size=1, shuffle=False, drop_last=True)