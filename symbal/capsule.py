
import pickle


class Capsule:

    def __init__(self,
                 dataframe=None,
                 output_name=None,
                 x_units=None,
                 y_units=None,
                 test_df=None,
                 groupby=None):

        self.dataframe = dataframe
        self.output_name = output_name
        self.x_units = x_units
        self.y_units = y_units
        self.test_df = test_df
        self.groupby = groupby

    def store_capsule(self, filename):

        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_capsule(filename):

    with open(filename, 'rb') as handle:
        capsule = pickle.load(handle)

    return capsule

