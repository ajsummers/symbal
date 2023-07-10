
import numpy as np
from symbal.penalties import invquad_penalty


def new_penalty(penalty_array, penalty_function, a, b, by_range, batch_size):
    """
    Selects maximum from given values and penalizes area around selection.

    Assumes first column in penalty_array is penalized value, other columns are
    independent variables.

    Returns: independent variables for selected point & new penalty_array w/ penalized values
    """

    max_index = np.argmax(penalty_array[:, 0])  # index for largest value
    max_pos = penalty_array[max_index, 1:]  # independent variable values for this index
    new_array = np.delete(penalty_array, max_index, axis=0)  # Remove selected value from array

    r_x = np.abs(new_array[:, 1:] - max_pos)  # Distance to selected point for each variable
    if by_range:
        s_x = np.ptp(new_array[:, 1:], axis=0) / batch_size  # Tune width of penalty by range / batch_size
    else:
        s_x = np.std(new_array[:, 1:],
                     axis=0)  # Tune width of penalty by standard deviation of each independent variable
    s_y = np.std(new_array[:, 0], axis=0)  # Standard deviation of penalized value

    penalty = penalty_function(a, b, r_x, s_x, s_y)

    new_array[:, 0] -= penalty  # subtract penalty

    return max_pos, new_array


def batch_selection(uncertainty_array, penalty_function=invquad_penalty, a=1, b=1, by_range=False, batch_size=10,
                    capture_penalties=False):

    captured_penalties = dict()

    if capture_penalties:
        captured_penalties[0] = uncertainty_array

    selected_points = np.empty((batch_size, uncertainty_array.shape[1] - 1))
    penalty_array = uncertainty_array

    for i in range(batch_size):

        selected_point, penalty_array = new_penalty(penalty_array, penalty_function, a, b, by_range, batch_size)

        if capture_penalties:
            captured_penalties[i+1] = penalty_array

        selected_points[i, :] = selected_point

    if capture_penalties:
        return selected_points, captured_penalties
    else:
        return selected_points

