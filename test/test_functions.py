import json
import os
import unittest
import src.utils as util
from src import config
from src.random_shapes import generate


class MyTestCase(unittest.TestCase):
    def test_generate(self):
        path = 'generate_test'
        number_of_masks_to_generate = 5
        os.mkdir(path)
        generate(number_of_masks_to_generate=number_of_masks_to_generate, path=path)
        self.assertEqual(len(os.listdir(path)), number_of_masks_to_generate)

    def test_group_data(self):
        conf = config.Config()
        util.group_data()
        self.assertGreater(len(os.listdir(conf.REAL_TRAIN_MRI)), 0)
        self.assertGreater(len(os.listdir(conf.REAL_TRAIN_MASK)), 0)

    def test_generate_boxes(self):
        conf = config.Config()
        util.generate_boxes()
        file = open(conf.FILE_PATH, 'r')
        mask_dict = dict(json.loads(file.read()))
        self.assertEqual(len(mask_dict.keys()), 2)


if __name__ == '__main__':
    unittest.main()
