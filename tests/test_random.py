import pytest

from arrayfire import Array
from arrayfire.backend import _clib_wrapper as wrapper
from arrayfire.library import random
from arrayfire.library.random import RandomEngine, RandomEngineType

# Test cases for the Random Engine


def test_random_engine_creation() -> None:
    # Test creating a random engine with default values
    engine = RandomEngine()
    assert engine.get_type() == RandomEngineType.PHILOX
    assert engine.get_seed() == 0
    engine.set_type(RandomEngineType.THREEFRY)
    assert engine.get_type() == RandomEngineType.THREEFRY
    engine.set_seed(42)
    assert engine.get_seed() == 42


def test_random_engine_from_handle() -> None:
    # Test creating a random engine from an existing handle
    handle = wrapper.create_random_engine(RandomEngineType.MERSENNE.value, 1232)
    engine = RandomEngine.from_engine(handle)
    assert engine.get_type() == RandomEngineType.MERSENNE
    assert engine.get_seed() == 1232


def test_random_engine_deletion() -> None:
    # Test engine deletion and resource release
    engine = RandomEngine()
    del engine  # This should release the engine's resources


# Test cases for the randu function


def test_randu_shape_1d() -> None:
    shape = (10,)
    result: Array = random.randu(shape)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_shape_2d() -> None:
    shape = (5, 7)
    result: Array = random.randu(shape)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_shape_3d() -> None:
    shape = (3, 4, 6)
    result: Array = random.randu(shape)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_shape_4d() -> None:
    shape = (2, 2, 3, 5)
    result: Array = random.randu(shape)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_default_engine() -> None:
    shape = (5, 5)
    result: Array = random.randu(shape)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_custom_engine() -> None:
    shape = (3, 3)
    custom_engine = RandomEngine(RandomEngineType.THREEFRY, seed=42)
    result: Array = random.randu(shape, engine=custom_engine)
    assert isinstance(result, Array)
    assert result.shape == shape


def test_randu_invalid_shape() -> None:
    # Test with an invalid shape (empty tuple)
    with pytest.raises(ValueError):
        shape = ()
        random.randu(shape)


def test_randu_invalid_shape_type() -> None:
    # Test with an invalid shape (non-tuple)
    with pytest.raises(ValueError):
        shape = [5, 5]
        random.randu(shape)  # type: ignore[arg-type]
