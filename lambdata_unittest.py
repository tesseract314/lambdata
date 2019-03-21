import unittest
import lambdata_tesseract.df_utils as lt

X = lt.TEST_DF['A']
y = lt.TEST_DF['B']

X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y)

class HelperFunctionTests(unittest.TestCase):
    """ Testing helper function in lambdata """

    def split_test(self):
        """ Testing if data split into correct lengths """

        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(X))

if __name__ == '__main__':
    unittest.main()
