import pytest

# we want to have pytest assert introspection in the helpers
pytest.register_assert_rewrite('sim.src.primitives')
pytest.register_assert_rewrite('sim.test')
