import logging
import os

class AttributeDict():
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, AttributeDict(value))
            else:
                setattr(self, key, value)

class MemoryConfig():
    def __init__(self, arg_config=None):
        self.args = self.__load_arg_config__(arg_config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.global_config.usable_gpu

    def __load_arg_config__(self, arg_config):
        if isinstance(arg_config, ArgConfig):
            return arg_config
        else:
            return ArgConfig(arg_config)

class ArgConfig():
    def __init__(self, obj=None):
        if obj is None:
            # If use default memory, 'FullMemory' will be loaded.
            raise ValueError("No parameter obj provided.")
        elif isinstance(obj, dict):
            # Load from a dict.
            self.add_args(obj)
        elif isinstance(obj, str):
            # Load from a yaml file.
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError('install the optional `pyyaml` dependency for YAML configs') from exc
            with open(obj, 'r', encoding='utf-8') as file:
                try:
                    config = yaml.safe_load(file)
                    logging.info("Parameter configuration loaded successfully")
                    self.add_args(config)
                except yaml.YAMLError as exc:
                    raise ValueError("Fail to load parameter configurations") from exc
        else:
            raise TypeError("Unrecognized parameter configuration type.")

    def add_args(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, AttributeDict(value))
            else:
                setattr(self, key, value)
