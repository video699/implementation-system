import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString
from shapely.ops import split

from video699.configuration import get_configuration
from video699.quadrangle.geos import GEOSConvexQuadrangle

CONFIGURATION = get_configuration()['FastaiVideoScreenDetector']
image_area = CONFIGURATION.getint('image_width') * CONFIGURATION.getint('image_height')


def contour_approx(contours, lower_bound, upper_bound, factors, debug=False):
    quadrangles = []
    for cnt in contours:
        if is_between_percentage(cv2.contourArea(cnt), lower_bound, upper_bound):
            for factor in factors:
                epsilon = factor * cv2.arcLength(cnt, True)
                polygon = cv2.approxPolyDP(cnt, epsilon, True)
                if debug:
                    quadrangles.append(polygon)
                    break
                else:
                    if polygon.shape[0] == 4:
                        quadrangles.append(polygon)
                        break
    return quadrangles


def is_between_percentage(contour_area, lower_area_percentage, upper_area_percentage):
    contour_percentage = contour_area * 100 / image_area
    return lower_area_percentage < contour_percentage < upper_area_percentage


def approximate(pred, methods):
    quadrangles = []
    if methods['base']:
        quadrangles = approximate_baseline(pred, **methods)

    if methods['erode_dilate']:
        erode_dilate_quadrangles = approximate_erose_dilate(pred, **methods)
        quadrangles = erode_dilate_quadrangles if len(erode_dilate_quadrangles) > len(
            quadrangles) else quadrangles

    if methods['ratio_split']:
        quadrangles = approximate_ratio_split(quadrangles, **methods)

    else:
        quadrangles = [GEOSConvexQuadrangle(**get_coordinates(quadrangle)) for quadrangle in
                       quadrangles]
    return quadrangles


def approximate_baseline(pred, base_lower_bound, base_upper_bound, base_factors, **params):
    _, contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangles = contour_approx(contours, base_lower_bound, base_upper_bound, base_factors)
    return quadrangles


def approximate_erose_dilate(pred, erode_dilate_lower_bound, erode_dilate_upper_bound,
                             erode_dilate_iterations, erode_dilate_factors, **params):
    erosed = cv2.erode(pred, None, iterations=erode_dilate_iterations)
    _, contours, _ = cv2.findContours(erosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangles = contour_approx(contours, erode_dilate_lower_bound, erode_dilate_upper_bound, erode_dilate_factors)
    erosed_dilated_quadrangles = []
    for quadrangle in quadrangles:
        zeros = np.zeros(pred.shape, dtype='uint8')
        erosed_quadrangle = draw_polygon(quadrangle, zeros)
        dilated_quadrangle = cv2.dilate(erosed_quadrangle, None, iterations=erode_dilate_iterations)
        _, contours, _ = cv2.findContours(dilated_quadrangle, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        erosed_dilated_quadrangles.extend(
            contour_approx(contours, erode_dilate_lower_bound, erode_dilate_upper_bound, erode_dilate_factors))
    return erosed_dilated_quadrangles


def approximate_ratio_split(quadrangles, ratio_split_lower_bound, ratio_split_upper_bound, **params):
    ratio_split_quadrangles = []
    for quadrangle in quadrangles:
        result = []
        geos_quadrangle = GEOSConvexQuadrangle(**get_coordinates(quadrangle))

        if ratio_split_lower_bound < geos_quadrangle.height / geos_quadrangle.width < ratio_split_upper_bound or \
                geos_quadrangle.area < 80000:
            ratio_split_quadrangles.append(geos_quadrangle)
            continue

        if not ratio_split_lower_bound < geos_quadrangle.height / geos_quadrangle.width:
            upper_midpoint = midpoint(geos_quadrangle.top_left, geos_quadrangle.top_right)
            lower_midpoint = midpoint(geos_quadrangle.bottom_left, geos_quadrangle.bottom_right)
            line = LineString([upper_midpoint, lower_midpoint])
            result = split(geos_quadrangle._polygon, line)

        elif not geos_quadrangle.height / geos_quadrangle.width < ratio_split_upper_bound:
            left_midpoint = midpoint(geos_quadrangle.top_left, geos_quadrangle.bottom_left)
            right_midpoint = midpoint(geos_quadrangle.top_right, geos_quadrangle.bottom_right)
            line = LineString([left_midpoint, right_midpoint])
            result = split(geos_quadrangle._polygon, line)

        for res in result:
            x, y = res.exterior.coords.xy
            coords = np.array([list(a) for a in zip(x, y)])
            ratio_split_quadrangles.append(GEOSConvexQuadrangle(**get_coordinates(coords)))

    return ratio_split_quadrangles


def midpoint(pointA, pointB):
    return (pointA[0] + pointB[0]) / 2, (pointA[1] + pointB[1]) / 2


def get_coordinates(quadrangle):
    squeezed = quadrangle.squeeze()
    x = squeezed[:, 0]
    y = squeezed[:, 1]
    top_left = (x + y).argmin()
    top_right = (max(y) - y + x).argmax()
    bottom_left = (max(x) - x + y).argmax()
    bottom_right = (x + y).argmax()
    return {'top_left': squeezed[top_left],
            'top_right': squeezed[top_right],
            'bottom_right': squeezed[bottom_right],
            'bottom_left': squeezed[bottom_left]}


def draw_polygon(polygon, image):
    copy = image.copy()
    return cv2.fillConvexPoly(copy, polygon, 100)


def draw_polygons(polygons, image, show=True):
    copy = image.copy()
    # Visualization
    if len(polygons) == 0:
        if show:
            plt.imshow(copy)
            plt.show()
    else:
        for polygon in polygons:
            copy = cv2.fillConvexPoly(copy, polygon, 100)
        if show:
            plt.imshow(copy)
            plt.show()
    return copy


def iou(screenA, screenB):
    intersection = screenA.coordinates.intersection_area(screenB.coordinates)
    union = screenA.coordinates.union_area(screenB.coordinates)
    return intersection / union
