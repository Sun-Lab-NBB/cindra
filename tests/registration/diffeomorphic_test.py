"""Contains tests for the diffeomorphic module."""

from __future__ import annotations

import numpy as np

from cindra.registration.diffeomorphic import DiffeomorphicDemonsRegistration


class TestDiffeomorphicDemonsRegistration:
    """Tests DiffeomorphicDemonsRegistration."""

    def test_constructor_stores_images(self) -> None:
        """Verifies that the constructor stores images as float32."""
        images = [np.ones((32, 32), dtype=np.uint8) * 100, np.ones((32, 32), dtype=np.uint8) * 200]
        registration = DiffeomorphicDemonsRegistration(images=images)
        assert len(registration._images) == 2
        assert registration._images[0].dtype == np.float32
        assert registration._images[1].dtype == np.float32

    def test_constructor_preserves_float32(self) -> None:
        """Verifies that float32 images are not re-converted."""
        image = np.ones((32, 32), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image])
        assert registration._images[0] is image

    def test_compute_grid_sampling(self) -> None:
        """Verifies the grid sampling calculation formula."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(
            images=images, final_scale=1.0, final_grid_sampling=16.0, grid_sampling_factor=1.0
        )
        # At final_scale=1.0: grid_sampling = (1.0 - 1.0) * 1.0 * 16.0 + 16.0 = 16.0
        assert registration._compute_grid_sampling(scale=1.0) == 16.0
        # At scale=3.0: grid_sampling = (3.0 - 1.0) * 1.0 * 16.0 + 16.0 = 48.0
        assert registration._compute_grid_sampling(scale=3.0) == 48.0

    def test_cache_operations(self) -> None:
        """Verifies cache set and get operations."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        key = (0, 1, 1.0)
        data = np.zeros((5, 5), dtype=np.float32)
        registration._set_cached(key="test", iteration_key=key, data=data)
        result = registration._get_cached(key="test", iteration_key=key)
        assert result is data

    def test_cache_miss_returns_none(self) -> None:
        """Verifies that a cache miss returns None."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        result = registration._get_cached(key="missing", iteration_key=(0, 0, 0.0))
        assert result is None

    def test_cache_stale_key_returns_none(self) -> None:
        """Verifies that a stale iteration key causes a cache miss."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        data = np.zeros((5, 5), dtype=np.float32)
        registration._set_cached(key="test", iteration_key=(0, 0, 1.0), data=data)
        result = registration._get_cached(key="test", iteration_key=(0, 1, 1.0))
        assert result is None

    def test_register_identical_images(self) -> None:
        """Verifies that registering identical images produces near-identity deformations."""
        image = np.random.default_rng(42).standard_normal((32, 32)).astype(np.float32)
        images = [image.copy(), image.copy()]
        registration = DiffeomorphicDemonsRegistration(
            images=images, scale_sampling=5, final_scale=1.0, final_grid_sampling=8.0
        )
        registration.register(progress=False)
        for i in range(2):
            deformation = registration.get_deformation(image_index=i)
            # Deformations should be near-zero for identical images.
            assert np.max(np.abs(deformation[0])) < 2.0
            assert np.max(np.abs(deformation[1])) < 2.0

    def test_register_produces_deformations(self) -> None:
        """Verifies that registration produces deformations for different images."""
        rng = np.random.default_rng(42)
        image1 = rng.standard_normal((32, 32)).astype(np.float32)
        image2 = rng.standard_normal((32, 32)).astype(np.float32)
        registration = DiffeomorphicDemonsRegistration(
            images=[image1, image2], scale_sampling=5, final_scale=1.0, final_grid_sampling=8.0
        )
        registration.register(progress=False)
        assert 0 in registration._deformations
        assert 1 in registration._deformations

    def test_default_parameters(self) -> None:
        """Verifies that the constructor stores default parameter values."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        assert registration._speed_factor == 3.0
        assert registration._scale_sampling == 30
        assert registration._grid_sampling_factor == 1.0
        assert registration._final_scale == 1.0
        assert registration._final_grid_sampling == 16.0
        assert registration._smooth_scale
        assert registration._injective
        assert registration._freeze_edges
        assert registration._deformation_limit == 1.0
        assert registration._noise_factor == 1.0
