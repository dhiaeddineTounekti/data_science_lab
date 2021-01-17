import unittest

from src.random_shapes import generate


class MyTestCase(unittest.TestCase):
    def test_generate(self):
        generate(3)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
