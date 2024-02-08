import pytest
from arrayfire_wrapper.lib import create_random_engine

import arrayfire as af
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
    handle = create_random_engine(RandomEngineType.MERSENNE.value, 1232)
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
    result: af.Array = af.randu(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_shape_2d() -> None:
    shape = (5, 7)
    result: af.Array = af.randu(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_shape_3d() -> None:
    shape = (3, 4, 6)
    result: af.Array = af.randu(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_shape_4d() -> None:
    shape = (2, 2, 3, 5)
    result: af.Array = af.randu(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_default_engine() -> None:
    shape = (5, 5)
    result: af.Array = af.randu(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_custom_engine() -> None:
    shape = (3, 3)
    custom_engine = RandomEngine(RandomEngineType.THREEFRY, seed=42)
    result: af.Array = af.randu(shape, engine=custom_engine)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randu_invalid_shape() -> None:
    # Test with an invalid shape (empty tuple)
    with pytest.raises(ValueError):
        shape = ()
        af.randu(shape)


def test_randu_invalid_shape_type() -> None:
    # Test with an invalid shape (non-tuple)
    with pytest.raises(ValueError):
        shape = [5, 5]
        af.randu(shape)  # type: ignore[arg-type]


# Test cases for the randn function


def test_randn_shape_1d() -> None:
    shape = (10,)
    result: af.Array = af.randn(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_shape_2d() -> None:
    shape = (5, 7)
    result: af.Array = af.randn(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_shape_3d() -> None:
    shape = (3, 4, 6)
    result: af.Array = af.randn(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_shape_4d() -> None:
    shape = (2, 2, 3, 5)
    result: af.Array = af.randn(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_default_engine() -> None:
    shape = (5, 5)
    result: af.Array = af.randn(shape)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_custom_engine() -> None:
    shape = (3, 3)
    custom_engine = RandomEngine(RandomEngineType.THREEFRY, seed=42)
    result: af.Array = af.randn(shape, engine=custom_engine)
    assert isinstance(result, af.Array)
    assert result.shape == shape


def test_randn_invalid_shape() -> None:
    # Test with an invalid shape (empty tuple)
    with pytest.raises(ValueError):
        shape = ()
        af.randn(shape)


def test_randn_invalid_shape_type() -> None:
    # Test with an invalid shape (non-tuple)
    with pytest.raises(ValueError):
        shape = [5, 5]
        af.randn(shape)  # type: ignore[arg-type]
