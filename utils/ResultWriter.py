import os
import json


class ResultWriter(object):
    def __init__(self, base_path):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.mkdir(self.base_path)

    def get_file_path(self, identifier, fold_num, pass_num):
        return os.path.join(self.base_path, "{}_fold{}_pass{}.result".format(identifier, fold_num, pass_num))

    def write_result(self, identifier, fold_num, pass_num, result):
        out_file = open(self.get_file_path(identifier, fold_num, pass_num), "a")
        out_file.write(json.dumps(result))
        out_file.write("\n")
