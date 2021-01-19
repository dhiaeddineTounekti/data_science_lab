import unittest
import src.utils as util
from src.random_shapes import generate


class MyTestCase(unittest.TestCase):
    def test_generate(self):
        generate(3, )
        self.assertEqual(True, False)

    def test_group_data(self):
        util.group_data()


if __name__ == '__main__':
    unittest.main()
