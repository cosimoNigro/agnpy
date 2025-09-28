import numpy as np

from agnpy.time_evolution._time_evolution_utils import group_duplicates, merge_points_preserve_integral

def arr(lst):
    return np.array(lst, dtype=float)

class TestTimeEvolutionUtils:

    def test_grouping_duplicates(self):
        assert group_duplicates([2]) == [[1,2]]
        assert group_duplicates([6,7]) == [[5,6,7]]
        assert group_duplicates([6,7,8]) == [[5,6,7,8]]
        assert group_duplicates([6,7,8,9]) == [[5,6,7,8,9]]
        assert group_duplicates([6,7, 9]) == [[5,6,7],[8,9]]
        assert group_duplicates([6,7, 9,10,11]) == [[5,6,7],[8,9,10,11]]
        assert group_duplicates([6,7, 9,10, 15,16,17]) == [[5,6,7],[8,9,10],[14,15,16,17]]
        assert group_duplicates([5, 10, 15]) == [[4,5],[9,10],[14,15]]

    # this is rather a temporary test, should be reviewed when the merge_points_preserve_integral is refactored and improved
    def test_merging_groups(self):
        x, y = merge_points_preserve_integral(arr([1,2,3,10, 40,49,50,51,60, 90,99,100]), arr([2,2,2,2, 3,3,3,3,3, 4,4,4]),
                                                  [[0,1,2], [5,6,7], [10,11]])
        assert np.alltrue(x == arr([1,10, 40,50,60, 90,100]))
        assert np.alltrue(y == arr([2,2, 3,3,3, 4,4]))

        x, y = merge_points_preserve_integral(np.array([1, 2, 3]), arr([2, 2, 2]),
                                                  [[0, 1, 2]])
        assert np.alltrue(1 <= x <= 3)
        assert np.alltrue(np.isnan(y))

        x, y = merge_points_preserve_integral(arr([1516299.14715, 1523015.93756, 1526979.18482, 1648481.36586]),
                                              arr([3.77632487e-31, 1.15803496e-35, 2.13481715e-34, 2.09818792e-36]),
                                              [[1, 2]])
        assert (1523015.93756 <= x[1] <= 1526979.18482)
        assert y[1] > 0