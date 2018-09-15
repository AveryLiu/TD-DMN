"""
Info fields do nothing to the data and
return the passed data untouched.

This is useful if we need to store additional
information of the example but we do not wish
to numericalize it.
"""
from torchtext import data


class InfoField(data.Field):
    def __init__(self, **kwargs):
        super(InfoField, self).__init__(**kwargs)

    def pad(self, mini_batch):
        return mini_batch

    def numericalize(self, arr, device=None):
        return arr

    def process(self, batch, device=None):
        return batch


class NestedInfoField(data.NestedField):
    def __init__(self, *args, **kwargs):
        super(NestedInfoField, self).__init__(*args, **kwargs)

    def process(self, batch, device=None):
        return batch

    def numericalize(self, arrs, device=None):
        return arrs

    def pad(self, mini_batch):
        return mini_batch
