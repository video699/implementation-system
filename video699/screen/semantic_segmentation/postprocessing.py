import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.ops import split

from video699.quadrangle.geos import GEOSConvexQuadrangle
from video699.screen.semantic_segmentation.common import is_bigger_than_boundary, get_coordinates, draw_polygon, \
    midpoint
import warnings

warnings.filterwarnings('ignore')


def contour_approximation(contour, lower_bound, factors):
    """
    Approximates single contour into quadrangle.
    Parameters
    ----------
    contour : np.array
        A contour to approximate into quadrangle.
    lower_bound : int
        A lower percentage of whole image area, under which a contour is discarded.
    factors : array-like
        A list of multipliers specifying the maximum Hausdorff distance between new approximated polygon and original
        contour.

    Returns
    -------
    polygon : np.array
        A polygon with specific requirements: not smaller proportion than lower_bound and 4 corner points - quadrangle.
    """
    if is_bigger_than_boundary(cv2.contourArea(contour), lower_bound):
        for factor in factors:
            epsilon = factor * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            if polygon.shape[0] == 4 and cv2.isContourConvex(polygon):
                return polygon


def contours_approximation(contours, lower_bound, factors):
    """
    Multi-contour version of contour_approximation function.
    Parameters
    ----------
    contours : np.array
        The contours to approximate into quadrangles.
    lower_bound : int
        A lower percentage of whole image area, under which a contours are discarded.
    factors : array-like
        A list of multipliers specifying the maximum Hausdorff distance between new approximated polygon and original
        contour.

    Returns
    -------
    polygons : np.array
        The polygons with specific requirements: not smaller proportion than lower_bound and 4 corner points.
    """
    quadrangles = []
    for cnt in contours:
        quadrangle = contour_approximation(contour=cnt, lower_bound=lower_bound, factors=factors)
        if quadrangle is not None:
            quadrangles.append(quadrangle)
    return quadrangles


def approximate(pred, post_processing_params):
    """
    Approximate predictions into quadrangles using methods parametrized by post_processing_params.
    Parameters
    ----------
    pred : np.array
        A semantic segmentation prediction: binary image in open-cv.
    post_processing_params : dict
        A dictionary of post-processing methods with parameters.
    Returns
    -------
    quadrangles : array-like[GEOSConvexQuadrangle]
        The quadrangles estimated by post-processing methods.
    """
    quadrangles = []
    if post_processing_params['base']:
        quadrangles = approximate_baseline(pred, **post_processing_params)

    if post_processing_params['erosion_dilation']:
        erosion_dilation_quadrangles = approximate_erosion_dilation(pred, **post_processing_params)
        quadrangles = erosion_dilation_quadrangles if len(erosion_dilation_quadrangles) > len(
            quadrangles) else quadrangles

    if post_processing_params['ratio_split'] and (
            post_processing_params['base'] or post_processing_params['erosion_dilation']):
        quadrangles = approximate_ratio_split(quadrangles, **post_processing_params)
        return quadrangles

    quadrangles = [GEOSConvexQuadrangle(**get_coordinates(quadrangle)) for quadrangle in
                   quadrangles]
    return quadrangles


def approximate_baseline(pred, base_lower_bound, base_factors, **params):
    """
    Approximate predictions into quadrangles using baseline method.
    Parameters
    ----------
    pred : np.array
        A semantic segmentation prediction: binary image in open-cv.
    base_lower_bound : int
        A lower percentage of whole image area, under which a contours are discarded.
    base_factors : array-like
        A list of multipliers specifying the maximum Hausdorff distance between new approximated polygon and original
            contour.
    params : dict
        A discarded parameters entered into function.

    Returns
    -------
    quadrangles : array-like
        The quadrangle contours estimated by post-processing methods.
    """
    contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangles = contours_approximation(contours, base_lower_bound, base_factors)
    return quadrangles


def approximate_erosion_dilation(pred, erosion_dilation_lower_bound, erosion_dilation_kernel_size,
                                 erosion_dilation_factors,
                                 **params):
    """
    Approximate predictions into quadrangles using morphological operators eroding and dilating.
    Parameters
    ----------
    pred : np.array
        A semantic segmentation prediction: binary image in open-cv.
    erosion_dilation_lower_bound : int
        A lower percentage of whole image area, under which a contours are discarded.
    erosion_dilation_kernel_size : int
        A size of the structuring element in which we perform morphological operator.
    erosion_dilation_factors : array-like
        A list of multipliers specifying the maximum Hausdorff distance between new approximated polygon and original
            contour.
    params : dict
        A discarded parameters entered into function.

    Returns
    -------
    quadrangles : array-like
        The quadrangle contours estimated by post-processing methods.
    """
    kernel = np.ones((erosion_dilation_kernel_size, erosion_dilation_kernel_size), np.uint8)
    erosed = cv2.erode(pred, kernel=kernel)
    contours, _ = cv2.findContours(erosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangles = contours_approximation(contours, erosion_dilation_lower_bound, erosion_dilation_factors)
    erosed_dilated_quadrangles = []
    for quadrangle in quadrangles:
        zeros = np.zeros(pred.shape, dtype='uint8')
        erosed_quadrangle = draw_polygon(quadrangle, zeros)
        dilated_quadrangle = cv2.dilate(erosed_quadrangle, kernel=kernel)
        contours, _ = cv2.findContours(dilated_quadrangle, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        erosed_dilated_quadrangles.extend(
            contours_approximation(contours, erosion_dilation_lower_bound, erosion_dilation_factors))
    return erosed_dilated_quadrangles


def approximate_ratio_split(quadrangles, ratio_split_lower_bound, **params):
    """
    Using quadrangle contours, check their width-height ratio and split if necessary.
    Parameters
    ----------
    quadrangles : array-like
        A quadrangle contour checked by some of the eroding-dilating or base method.
    ratio_split_lower_bound : float
        A ratio under this parameter is split byy a vertical line.
    params : dict,
        A discarded parameters entered into function.

    Returns
    -------
    quadrangles : array-like
        The quadrangle contours estimated by post-processing methods.
    """
    ratio_split_quadrangles = []
    for quadrangle in quadrangles:
        result = []
        geos_quadrangle = GEOSConvexQuadrangle(**get_coordinates(quadrangle))

        if ratio_split_lower_bound < geos_quadrangle.height / geos_quadrangle.width:
            ratio_split_quadrangles.append(geos_quadrangle)
            continue

        if not ratio_split_lower_bound < geos_quadrangle.height / geos_quadrangle.width:
            upper_midpoint = midpoint(geos_quadrangle.top_left, geos_quadrangle.top_right)
            lower_midpoint = midpoint(geos_quadrangle.bottom_left, geos_quadrangle.bottom_right)
            line = LineString([upper_midpoint, lower_midpoint])
            result = split(geos_quadrangle._polygon, line)

        for res in result:
            x, y = res.exterior.coords.xy
            coords = np.array([list(a) for a in zip(x, y)])
            ratio_split_quadrangles.append(GEOSConvexQuadrangle(**get_coordinates(coords)))

    return ratio_split_quadrangles
