import cv2
import numpy as np
import math

class TripleVotingShapeDetector:
    """
    Industrial Part Classifier using Triple-Method Consensus Voting:
    1. Vote 1: Contour Vertex Count & Interior Angle Geometry (Douglas-Peucker)
    2. Vote 2: Circularity & Solidity Geometric Ratio Analysis
    3. Vote 3: Hu Moments Invariant Template Matching (Scale & Rotation Invariant)
    
    Specifically engineered to solve the SVR Robotics problem:
    Reliably differentiating small Hexagons vs Circles even when shadow penumbras
    soften corner edges.
    """

    def __init__(self, min_area=600, max_area_ratio=0.85):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        
        # Pre-compute canonical reference templates for Hu moments matching
        self._canonical_templates = self._generate_canonical_templates()

    def _generate_canonical_templates(self):
        """Generates ideal reference contours for Hu moments comparison."""
        templates = {}
        size = 200
        center = (100, 100)
        radius = 70

        # Canonical Circle
        img_circle = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(img_circle, center, radius, 255, -1)
        cnts, _ = cv2.findContours(img_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        templates["Circle"] = cnts[0]

        # Canonical Hexagon
        img_hex = np.zeros((size, size), dtype=np.uint8)
        hex_pts = np.array([[center[0] + int(radius * np.cos(a)), center[1] + int(radius * np.sin(a))]
                            for a in np.linspace(0, 2*np.pi, 7)[:-1]], dtype=np.int32)
        cv2.fillPoly(img_hex, [hex_pts], 255)
        cnts, _ = cv2.findContours(img_hex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        templates["Hexagon"] = cnts[0]

        # Canonical Square
        img_sq = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(img_sq, (40, 40), (160, 160), 255, -1)
        cnts, _ = cv2.findContours(img_sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        templates["Square"] = cnts[0]

        # Canonical Triangle
        img_tri = np.zeros((size, size), dtype=np.uint8)
        tri_pts = np.array([[center[0] + int(radius * np.cos(a)), center[1] + int(radius * np.sin(a))]
                            for a in np.linspace(-np.pi/2, 3*np.pi/2, 4)[:-1]], dtype=np.int32)
        cv2.fillPoly(img_tri, [tri_pts], 255)
        cnts, _ = cv2.findContours(img_tri, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        templates["Triangle"] = cnts[0]

        return templates

    def _calculate_interior_angles(self, approx_pts):
        """Calculates interior vertex angles in degrees."""
        pts = [p[0] for p in approx_pts]
        n = len(pts)
        if n < 3:
            return []

        angles = []
        for i in range(n):
            p1 = np.array(pts[i - 1], dtype=np.float32)
            p2 = np.array(pts[i], dtype=np.float32)
            p3 = np.array(pts[(i + 1) % n], dtype=np.float32)

            v1, v2 = p1 - p2, p3 - p2
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if norm1 * norm2 > 0:
                cos_val = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angles.append(round(float(np.degrees(np.arccos(cos_val))), 1))
            else:
                angles.append(0.0)
        return angles

    def detect(self, edges, frame):
        """
        Input:
            edges: Binary edge map (from Canny or adaptive gradient)
            frame: Restored BGR frame for visual overlay
        Output:
            annotated_frame: Frame with bounding polygons, coordinates, and labels
            inspected_shapes: List of shape metadata dictionaries for robot arm actuation
        """
        h_frame, w_frame = frame.shape[:2]
        max_area = h_frame * w_frame * self.max_area_ratio

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        inspected_shapes = []

        # Sort contours by area (largest to smallest)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue

            # Check Solidity (Convexity filter - rejects shadows, conveyor seams, cables)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area
            if solidity < 0.85:
                # True mechanical parts have convex boundaries (>0.85). Skip irregular noise.
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

            # --- VOTE 1: Douglas-Peucker Polygon Approximation & Angles ---
            epsilon = 0.026 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            num_vertices = len(approx)
            angles = self._calculate_interior_angles(approx)
            mean_angle = float(np.mean(angles)) if angles else 0.0

            vote_1 = None
            if num_vertices == 3:
                vote_1 = "Triangle"
            elif num_vertices == 4:
                vote_1 = "Square"
            elif num_vertices == 5:
                vote_1 = "Pentagon"
            elif num_vertices == 6 and (100 <= mean_angle <= 140):
                vote_1 = "Hexagon"
            elif num_vertices >= 8:
                vote_1 = "Circle"

            # --- VOTE 2: Circularity Ratio Analysis ---
            vote_2 = None
            if circularity >= 0.895:
                vote_2 = "Circle"
            elif 0.77 <= circularity < 0.895:
                vote_2 = "Hexagon"
            elif 0.70 <= circularity < 0.77:
                vote_2 = "Square"
            elif 0.45 <= circularity < 0.70:
                vote_2 = "Triangle"

            # --- VOTE 3: Hu Moments Invariant Shape Matching ---
            vote_3 = None
            best_match_val = 999.0
            for name, temp_cnt in self._canonical_templates.items():
                # cv2.matchShapes compares the 7 log Hu invariant moments (method 1: I1)
                dist = cv2.matchShapes(cnt, temp_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < best_match_val:
                    best_match_val = dist
                    vote_3 = name

            # --- CONSENSUS VOTING LOGIC (2 out of 3 Majority) ---
            votes = [v for v in [vote_1, vote_2, vote_3] if v is not None]
            
            final_shape = None
            confidence = 0.0

            # Count votes
            vote_counts = {}
            for v in votes:
                vote_counts[v] = vote_counts.get(v, 0) + 1

            for candidate, count in vote_counts.items():
                if count >= 2:
                    final_shape = candidate
                    confidence = 88.0 + (count * 4.0)
                    break

            # Fallback if tie or ambiguous: circularity thresholding
            if final_shape is None and len(votes) > 0:
                final_shape = vote_2 if vote_2 is not None else votes[0]
                confidence = 75.0

            # Majority consensus voting is authoritative
            # If 2 or more methods agree on Circle or Hexagon, respect consensus!

            # Minimum Area Bounding Box for Robot Gripper Coordinates
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), orient_angle = rect
            cx, cy = int(cx), int(cy)
            aspect_ratio = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 1.0

            if final_shape is not None:
                # Draw visual annotations for operator HUD
                color_hud = (0, 255, 128) if final_shape == "Hexagon" else (0, 215, 255)
                
                # Draw polygon perimeter
                cv2.polylines(annotated, [approx], isClosed=True, color=color_hud, thickness=2)

                # Draw vertex landmarks
                for pt in approx:
                    cv2.circle(annotated, tuple(pt[0]), 3, (0, 140, 255), -1)

                # Draw Centroid Crosshair
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

                # Text annotations
                label = f"{final_shape} [{confidence:.0f}%]"
                cv2.putText(annotated, label, (max(8, cx - 40), max(22, cy - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(annotated, label, (max(8, cx - 40), max(22, cy - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_hud, 1)

                coord_str = f"({cx}, {cy}) | {orient_angle:.0f}deg"
                cv2.putText(annotated, coord_str, (max(8, cx - 40), cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)

                inspected_shapes.append({
                    "shape": final_shape,
                    "confidence": round(confidence, 1),
                    "circularity": round(circularity, 3),
                    "solidity": round(solidity, 3),
                    "vertices": num_vertices,
                    "angles": angles,
                    "centroid": [cx, cy],
                    "angle": round(orient_angle, 1),
                    "area": int(area),
                    "votes": {
                        "geometry": vote_1,
                        "circularity": vote_2,
                        "hu_moments": vote_3,
                        "vertex_method": vote_1,
                        "circularity_method": vote_2,
                        "hu_moments_method": vote_3
                    }
                })

        return annotated, inspected_shapes

# Backwards compatibility alias
AdvancedShapeDetector = TripleVotingShapeDetector
