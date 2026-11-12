import numpy as np
import pytest

from ev_assignment.methods.inverse_design import preference_rows


def test_robust_preference_uses_competitor_lower_target_upper():
    A, b, labels = preference_rows([[0, 1], [3, 2]], {"D": [[4, 8], [1, 2]]}, {"D": 1})
    assert np.array_equal(A, [[3, 1, 1]])
    assert np.array_equal(b, [2])  # competitor L=4 minus target U=2
    assert labels == [dict(method="D", target=1, competitor=0)]
    with pytest.raises(ValueError, match="ordered"):
        preference_rows([[0], [1]], {"D": [[8, 4], [1, 2]]}, {"D": 1})
