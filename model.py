from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.utils.annotations import override
_, nn = try_import_torch()
import torch.nn.functional as F
import torch
import numpy as np

# Defining custom model with CNN and FC-net
class CustomTorchModel(TorchModelV2,nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space,num_outputs,model_config,name) 
        nn.Module.__init__(self)
        self.obs_space=obs_space
        # Conv-net blovk
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        # Calculating output neurons after conv layers
        self.neurons = self.linear_input_neurons()
        # FC-net block
        self.fc1 = torch.nn.Linear(self.neurons, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)

        # Value function branch        
        self.value_function_fc = nn.Linear(256, 1)
        # Advantage function branch
        self.advantage_function_fc = nn.Linear(256, num_outputs)



    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Pre-process inputs
        x = input_dict["obs"].transpose(1,3)
        x = x / np.float32(255)
        # Convolution blocks
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten convnet output
        x = x.view(-1, self.neurons)

        # FC-Net block
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #Value function
        self._cur_value = self.value_function_fc(x).view(-1)
        # Logit output (policy actions)
        logits = self.advantage_function_fc(x)
        return logits, state

    # Calculate size after RELU
    def size_after_relu(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))

        return x.size()


    # after obtaining the size in above method, we call it and multiply all elements of the returned size.
    def linear_input_neurons(self):
        size = self.size_after_relu(torch.rand(1, 16, 72, 96)) # image size: 16 x 72x96
        m = 1
        for i in size:
            m *= i
        return int(m)

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value