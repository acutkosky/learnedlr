

def noop(*args, **kwargs):
    pass

class NoOpModule:
    def __call__(self, *args, **kwargs):
        return self
    def __getattribute__(self, *args, **kwargs):
        return self


def optional_module(module, use_module=True):
    if use_module:
        return module
    else:
        return NoOpModule()