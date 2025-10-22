from agnpy.time_evolution._time_evolution_utils import group_duplicates

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