"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from __future__ import annotations

from typing import Any, NamedTuple
from dataclasses import field, dataclass
from collections.abc import Sequence

import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from ataraxis_base_utilities import console


def distance_kernel(radius: int) -> np.ndarray:
    """Returns 2D array containing geometric distance from center, with radius "radius" """
    d = np.arange(-radius, radius + 1)
    dists_2d = norm(np.meshgrid(d, d), axis=0)
    return dists_2d


def median_pix(ypix, xpix):
    ymed, xmed = np.median(ypix), np.median(xpix)
    imin = np.argmin((xpix - xmed) ** 2 + (ypix - ymed) ** 2)
    xmed = xpix[imin]
    ymed = ypix[imin]
    return [ymed, xmed]


class EllipseData(NamedTuple):
    mu: float
    cov: float
    radii: tuple[float, float]
    ellipse: np.ndarray
    dy: int
    dx: int

    @property
    def area(self):
        return (self.radii[0] * self.radii[1]) ** 0.5 * np.pi

    @property
    def radius(self) -> float:
        return self.radii[0] * np.mean((self.dx, self.dy))

    @property
    def aspect_ratio(self) -> float:
        ry, rx = self.radii
        return aspect_ratio(width=ry, height=rx)


@dataclass(frozen=True)
class ROI:
    y_pixels: np.ndarray
    x_pixels: np.ndarray
    pixel_weights: np.ndarray
    centroid: np.ndarray
    do_crop: bool
    rsort: np.ndarray = field(default_factory=lambda: np.sort(distance_kernel(radius=30).flatten()), repr=False)

    def __post_init__(self):
        """Validate inputs."""
        if self.x_pixels.shape != self.y_pixels.shape or self.x_pixels.shape != self.pixel_weights.shape:
            console.error(message="x_pixels, y_pixels, and pixel_weights should all be the same size.", error=TypeError)

    @classmethod
    def from_stat_dict(cls, stat: dict[str, Any], do_crop: bool = True) -> ROI:
        return cls(
            y_pixels=stat["y_pixels"],
            x_pixels=stat["x_pixels"],
            pixel_weights=stat["pixel_weights"],
            centroid=stat["centroid"],
            do_crop=do_crop,
        )

    def to_array(self, Ly: int, Lx: int) -> np.ndarray:
        """Returns a 2D boolean array of shape (Ly x Lx) indicating where the roi is located."""
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[self.y_pixels, self.x_pixels] = 1
        return arr

    @classmethod
    def stats_dicts_to_3d_array(cls, stats: Sequence[dict[str, Any]], Ly: int, Lx: int, label_id: bool = False):
        """Outputs a (roi x Ly x Lx) float array from a sequence of stat dicts.
        Convenience function that repeatedly calls ROI.from_stat_dict() and ROI.to_array() for all rois.

        Parameters
        ----------
        stats : List of dictionary "y_pixels", "x_pixels", "pixel_weights"
        Ly : y size of frame
        Lx : x size of frame
        label_id : whether array should be an integer value indicating ROI id or just 1 (indicating precence of ROI).
        """
        arrays = []
        for i, stat in enumerate(stats):
            array = cls.from_stat_dict(stat=stat).to_array(Ly=Ly, Lx=Lx)
            if label_id:
                array *= i + 1
            arrays.append(array)
        return np.stack(arrays)

    def ravel_indices(self, Ly: int, Lx: int) -> np.ndarray:
        """Returns a 1-dimensional array of indices from the y_pixels and x_pixels coordinates, assuming an image shape Ly x Lx."""
        return np.ravel_multi_index((self.y_pixels, self.x_pixels), (Ly, Lx))

    @classmethod
    def get_overlap_count_image(cls, rois: Sequence[ROI], Ly: int, Lx: int) -> np.ndarray:
        return count_overlaps(
            Ly=Ly, Lx=Lx, y_pixels_list=[roi.y_pixels for roi in rois], x_pixels_list=[roi.x_pixels for roi in rois]
        )

    @classmethod
    def filter_overlappers(cls, rois: Sequence[ROI], overlap_image: np.ndarray, max_overlap: float) -> list[bool]:
        """Returns logical array of rois that remain after removing those that overlap more than fraction max_overlap from overlap_img."""
        return filter_overlappers(
            y_pixels_list=[roi.y_pixels for roi in rois],
            x_pixels_list=[roi.x_pixels for roi in rois],
            overlap_image=overlap_image,
            max_overlap=max_overlap,
        )

    def get_overlap_image(self, overlap_count_image: np.ndarray) -> np.ndarray:
        return overlap_count_image[self.y_pixels, self.x_pixels] > 1

    @property
    def soma_mask(self) -> np.ndarray:
        if self.do_crop and self.y_pixels.size > 10:
            dists = ((self.y_pixels - self.centroid[0]) ** 2 + (self.x_pixels - self.centroid[1]) ** 2) ** 0.5
            radii = np.arange(0, dists.max(), 1)
            area = np.zeros_like(radii)
            for k, radius in enumerate(radii):
                area[k] = self.pixel_weights[dists < radius].sum()
            darea = np.diff(area)
            radius = radii[-1]
            threshold = darea.max() / 3
            if len(np.nonzero(darea > threshold)[0]) > 0:
                ida = np.nonzero(darea > threshold)[0][0]
                if len(np.nonzero(darea[ida:] < threshold)[0]):
                    radius = radii[np.nonzero(darea[ida:] < threshold)[0][0] + ida]
            crop = dists < radius
            if crop.sum() == 0:
                crop = np.ones(self.y_pixels.size, "bool")
            return crop
        return np.ones(self.y_pixels.size, "bool")

    @property
    def mean_r_squared(self) -> float:
        return mean_r_squared(y=self.y_pixels[self.soma_mask], x=self.x_pixels[self.soma_mask])

    @property
    def mean_r_squared_baseline(self) -> float:
        return np.mean(self.rsort[: self.soma_pixel_count])

    @property
    def compactness(self) -> float:
        return self.mean_r_squared / (1e-10 + self.mean_r_squared_baseline)

    @property
    def solidity(self) -> float:
        if self.soma_pixel_count > 10:
            points = np.stack((self.y_pixels[self.soma_mask], self.x_pixels[self.soma_mask]), axis=1)
            try:
                hull = ConvexHull(points)
                volume = hull.volume
            except:
                volume = 10
        else:
            volume = 10
        return self.soma_pixel_count / volume

    @classmethod
    def get_mean_r_squared_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average(
            [roi.mean_r_squared for roi in rois], estimator=np.nanmedian, offset=1e-10, first_n=first_n
        )

    @property
    def soma_pixel_count(self) -> int:
        return self.soma_mask.sum()

    @property
    def pixel_count(self) -> int:
        return self.x_pixels.size

    @classmethod
    def get_pixel_count_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average([roi.pixel_count for roi in rois], first_n=first_n)

    def fit_ellipse(self, dy: float, dx: float) -> EllipseData:
        return fitMVGaus(
            self.y_pixels[self.soma_mask],
            self.x_pixels[self.soma_mask],
            self.pixel_weights[self.soma_mask],
            dy=dy,
            dx=dx,
            thres=2,
        )


def roi_stats(stat, Ly: int, Lx: int, aspect=None, diameter=None, max_overlap=None, do_crop=True):
    """Computes statistics of ROIs.

    Parameters
    ----------
    stat : dictionary
        Dictionary containing "y_pixels", "x_pixels", "pixel_weights".

    FOV size : (Ly, Lx)

    aspect : aspect ratio of recording

    diameter : (dy, dx)

    Returns:
    -------
    stat : dictionary
        Adds "pixel_count", "normalized_pixel_count", "centroid", "footprint", "compactness", "radius", "aspect_ratio".
    """
    if "centroid" not in stat[0]:
        for s in stat:
            s["centroid"] = median_pix(s["y_pixels"], s["x_pixels"])

    # approx size of masks for ROI aspect ratio estimation
    d0 = 10 if diameter is None or (isinstance(diameter, int) and diameter == 0) else diameter
    if aspect is not None:
        diameter = int(d0[0]) if isinstance(d0, (list, np.ndarray)) else int(d0)
        dy, dx = int(aspect * diameter), diameter
    else:
        dy, dx = (int(d0), int(d0)) if not isinstance(d0, (list, np.ndarray)) else (int(d0[0]), int(d0[0]))

    rois = [
        ROI(
            y_pixels=s["y_pixels"],
            x_pixels=s["x_pixels"],
            pixel_weights=s["pixel_weights"],
            centroid=s["centroid"],
            do_crop=do_crop,
        )
        for s in stat
    ]
    n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
    for roi, s in zip(rois, stat, strict=False):
        s["mean_r_squared"] = roi.mean_r_squared
        s["mean_r_squared_baseline"] = roi.mean_r_squared_baseline
        s["compactness"] = roi.compactness
        s["solidity"] = roi.solidity
        s["pixel_count"] = roi.pixel_count
        s["soma_pixel_count"] = roi.soma_pixel_count
        s["soma_mask"] = roi.soma_mask
        s["overlap_mask"] = roi.get_overlap_image(n_overlaps)
        ellipse = roi.fit_ellipse(dy, dx)
        s["radius"] = ellipse.radius
        s["aspect_ratio"] = ellipse.aspect_ratio

    mrs_normeds = norm_by_average(
        values=np.array([s["mean_r_squared"] for s in stat]), estimator=np.nanmedian, offset=1e-10, first_n=100
    )
    npix_normeds = norm_by_average(values=np.array([s["pixel_count"] for s in stat]), first_n=100)
    npix_soma_normeds = norm_by_average(values=np.array([s["soma_pixel_count"] for s in stat]), first_n=100)
    for s, mrs_normed, npix_normed, npix_soma_normed in zip(
        stat, mrs_normeds, npix_normeds, npix_soma_normeds, strict=False
    ):
        s["mean_r_squared"] = mrs_normed
        s["normalized_pixel_count_full"] = npix_normed
        s["normalized_pixel_count"] = npix_soma_normed
        s["footprint"] = 0 if "footprint" not in s else s["footprint"]

    if max_overlap is not None and max_overlap < 1.0:
        keep_rois = ROI.filter_overlappers(rois=rois, overlap_image=n_overlaps, max_overlap=max_overlap)
        stat = stat[keep_rois]
        n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
        rois = [
            ROI(
                y_pixels=s["y_pixels"],
                x_pixels=s["x_pixels"],
                pixel_weights=s["pixel_weights"],
                centroid=s["centroid"],
                do_crop=do_crop,
            )
            for s in stat
        ]
        for roi, s in zip(rois, stat, strict=False):
            s["overlap_mask"] = roi.get_overlap_image(n_overlaps)

    return stat


def mean_r_squared(y: np.ndarray, x: np.ndarray, estimator=np.median) -> float:
    return np.mean(norm(((y - estimator(y)), (x - estimator(x))), axis=0))


def aspect_ratio(width: float, height: float, offset: float = 0.01) -> float:
    return 2 * width / (width + height + offset)


def fitMVGaus(y, x, lam0, dy, dx, thres=2.5, npts: int = 100) -> EllipseData:
    """Computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations.
    Parameters
    ----------
    y : float, array
        pixel locations in y
    x : float, array
        pixel locations in x
    lam0 : float, array
        weights of each pixel
    """
    y = y / dy
    x = x / dx

    # normalize pixel weights
    lam = lam0.copy()
    ix = lam > 0  # lam.max()/5
    y, x, lam = y[ix], x[ix], lam[ix]
    lam /= lam.sum()

    # mean of gaussian
    yx = np.stack((y, x))
    mu = (lam * yx).sum(axis=1)
    yx = (yx - mu[:, np.newaxis]) * lam**0.5
    cov = yx @ yx.T

    # radii of major and minor axes
    radii, evec = np.linalg.eig(cov)
    radii = thres * np.maximum(0, np.real(radii)) ** 0.5

    # compute pts of ellipse
    theta = np.linspace(0, 2 * np.pi, npts)
    p = np.stack((np.cos(theta), np.sin(theta)))
    ellipse = (p.T * radii) @ evec.T + mu
    radii = np.sort(radii)[::-1]
    return EllipseData(mu=mu, cov=cov, radii=radii, ellipse=ellipse, dy=dy, dx=dx)


def count_overlaps(Ly: int, Lx: int, y_pixels_list, x_pixels_list) -> np.ndarray:
    overlap = np.zeros((Ly, Lx))
    for x_pixels, y_pixels in zip(x_pixels_list, y_pixels_list, strict=False):
        overlap[y_pixels, x_pixels] += 1
    return overlap


def filter_overlappers(y_pixels_list, x_pixels_list, overlap_image: np.ndarray, max_overlap: float) -> list[bool]:
    """Returns ROI indices that remain after removing those that overlap more than fraction max_overlap from overlap_img."""
    n_overlaps = overlap_image.copy()
    keep_rois = []
    for y_pixels, x_pixels in reversed(
        list(zip(y_pixels_list, x_pixels_list, strict=False))
    ):  # TODO: is there an ordering effect here that affects which rois will be removed and which will stay?
        keep_roi = np.mean(n_overlaps[y_pixels, x_pixels] > 1) <= max_overlap
        keep_rois.append(keep_roi)
        if not keep_roi:
            n_overlaps[y_pixels, x_pixels] -= 1
    return keep_rois[::-1]


def norm_by_average(values: np.ndarray, estimator=np.mean, first_n: int = 100, offset: float = 0.0) -> np.ndarray:
    """Returns array divided by the (average of the "first_n" values + offset), calculating the average with "estimator"."""
    return np.array(values, dtype="float32") / (estimator(values[:first_n]) + offset)
