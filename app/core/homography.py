#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homography module for Neptune App
Transforms perspective view coordinates to top-down minimap coordinates.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class HomographyTransformer:
    """
    Handles homography transformation from camera perspective to top-down minimap view.

    The 4 source points should correspond to:
    - P1: Top-Left (North-West)
    - P2: Top-Right (North-East)
    - P3: Bottom-Right (South-East)
    - P4: Bottom-Left (South-West)
    """

    def __init__(self, minimap_width: int = 200, minimap_height: int = 150):
        """
        Initialize the homography transformer.

        Args:
            minimap_width: Width of the destination minimap in pixels
            minimap_height: Height of the destination minimap in pixels
        """
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height

        # Source points (from video/camera) - will be set from water zone polygon
        self.src_points: Optional[np.ndarray] = None

        # Destination points (minimap rectangle with margin)
        margin = 10
        self.dst_points = np.array([
            [margin, margin],                                      # P1: Top-Left
            [minimap_width - margin, margin],                      # P2: Top-Right
            [minimap_width - margin, minimap_height - margin],     # P3: Bottom-Right
            [margin, minimap_height - margin]                      # P4: Bottom-Left
        ], dtype=np.float32)

        # Homography matrix
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None

    def set_source_points(self, points: List[Tuple[float, float]]) -> bool:
        """
        Set the 4 source points from the camera view.
        Points should be ordered: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

        Args:
            points: List of 4 (x, y) tuples representing the corners

        Returns:
            True if homography was computed successfully
        """
        if len(points) < 4:
            return False

        self.src_points = np.array(points[:4], dtype=np.float32)
        return self._compute_homography()

    def set_source_from_polygon(self, polygon: List[List[float]]) -> bool:
        """
        Set source points from a water zone polygon.
        Automatically extracts the 4 corner points (extremities).

        Args:
            polygon: List of [x, y] points forming the water zone polygon

        Returns:
            True if homography was computed successfully
        """
        if not polygon or len(polygon) < 4:
            return False

        pts = np.array(polygon, dtype=np.float32)
        corners = self._extract_corners(pts)

        if corners is None:
            return False

        self.src_points = corners
        return self._compute_homography()

    def _extract_corners(self, polygon: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the 4 corner points from a polygon.
        Uses convex hull and finds extreme points.

        Returns points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        """
        if len(polygon) < 4:
            return None

        # Get convex hull
        hull = cv2.convexHull(polygon)
        hull = hull.reshape(-1, 2)

        if len(hull) < 4:
            return None

        # Find centroid
        centroid = np.mean(hull, axis=0)

        # Calculate angles from centroid
        angles = np.arctan2(hull[:, 1] - centroid[1], hull[:, 0] - centroid[0])

        # Sort points by angle (counter-clockwise from right)
        sorted_indices = np.argsort(angles)
        sorted_pts = hull[sorted_indices]

        # Find the 4 extreme corners
        # Top-Left: min(x + y) - closest to origin
        # Top-Right: max(x - y) - far right, close to top
        # Bottom-Right: max(x + y) - farthest from origin
        # Bottom-Left: min(x - y) - far left, close to bottom

        sum_pts = sorted_pts[:, 0] + sorted_pts[:, 1]
        diff_pts = sorted_pts[:, 0] - sorted_pts[:, 1]

        top_left = sorted_pts[np.argmin(sum_pts)]
        bottom_right = sorted_pts[np.argmax(sum_pts)]
        top_right = sorted_pts[np.argmax(diff_pts)]
        bottom_left = sorted_pts[np.argmin(diff_pts)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _compute_homography(self) -> bool:
        """
        Compute the homography matrix from source to destination points.

        Returns:
            True if computation was successful
        """
        if self.src_points is None:
            return False

        try:
            self.H, _ = cv2.findHomography(self.src_points, self.dst_points)
            if self.H is not None:
                self.H_inv = np.linalg.inv(self.H)
                return True
        except Exception as e:
            print(f"[Homography] Error computing matrix: {e}")

        return False

    def transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """
        Transform a point from camera view to minimap coordinates.

        Args:
            point: (x, y) coordinates in the camera view

        Returns:
            (x, y) coordinates in the minimap, or None if transformation fails
        """
        if self.H is None:
            return None

        try:
            # Convert to homogeneous coordinates
            pt = np.array([[point[0], point[1], 1.0]], dtype=np.float32).T

            # Apply homography
            transformed = self.H @ pt

            # Convert back from homogeneous
            if transformed[2, 0] != 0:
                x = transformed[0, 0] / transformed[2, 0]
                y = transformed[1, 0] / transformed[2, 0]
                return (int(x), int(y))
        except Exception:
            pass

        return None

    def transform_points(self, points: List[Tuple[float, float]]) -> List[Optional[Tuple[int, int]]]:
        """
        Transform multiple points from camera view to minimap coordinates.

        Args:
            points: List of (x, y) coordinates in the camera view

        Returns:
            List of transformed (x, y) coordinates (or None for failed transforms)
        """
        return [self.transform_point(p) for p in points]

    def is_valid(self) -> bool:
        """Check if homography matrix has been computed."""
        return self.H is not None

    def get_minimap_bounds(self) -> Tuple[int, int]:
        """Get minimap dimensions."""
        return (self.minimap_width, self.minimap_height)
