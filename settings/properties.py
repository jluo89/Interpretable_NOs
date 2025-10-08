import json
import ipdb

class Testing_properties:
    def __init__(self, path, which_example, which_model,exp_name=None):
        self.which_example = which_example
        self.which_model = which_model

        test_properties_path = path + "/training_properties.json"

        with open(test_properties_path,"r") as f:
            self.test_config = json.load(f)
        
        print('='*20+"Test Config"+'='*20)
        print(self.test_config)

        self._get_setting()
        self.s = int(((self.h - 1)/self.r) + 1)
        self.iterations = self.epochs*(self.ntrain//self.batch_size)

    def _get_setting(self):
        train_attr = list(self.test_config.keys())
        for key in train_attr:
            setattr(self, key, self.test_config[key])


class Training_Properties:
    def __init__(self, which_example, which_model,exp_name=None):
        self.which_example = which_example
        self.which_model = which_model
        train_config_path = f"./config/train_config/{which_example}.json"
        model_config_path = f"./config/model_config/{which_model}/{which_example}.json" if exp_name == None else f"config/model_config/{which_model}/{exp_name}.json"

        with open(train_config_path,"r") as f:
            self.train_config = json.load(f)
        
        print('='*20+"Training Config"+'='*20)
        print(self.train_config)

        with open(model_config_path,"r") as f:
            self.model_config = json.load(f)

        print('='*20+"Model Config"+'='*20)
        print(self.model_config)
        self.t = True

        self._get_setting()
        self.s = int(((self.h - 1)/self.r) + 1)
        self.iterations = self.epochs*(self.ntrain//self.batch_size)

    def _get_setting(self):
        train_attr = list(self.train_config.keys())
        for key in train_attr:
            setattr(self, key, self.train_config[key])

        model_attr = list(self.model_config.keys())
        for key in model_attr:
            setattr(self, key, self.model_config[key])
        

    def save_as_txt(self, file_path):
        with open(file_path, 'w') as fi:
            new_dict = {**self.model_config,**self.train_config}
            json.dump(new_dict,fi,indent=4)