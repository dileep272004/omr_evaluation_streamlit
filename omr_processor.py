import cv2
import numpy as np
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime
from answer_keys import KEYS, SUBJECTS, OPTION_MAP

class OMRProcessor:
    def __init__(self, set_version='A', grid_rows=100, grid_cols=4):
        self.set_version = set_version
        self.key = KEYS[set_version]
        self.db_path = 'omr.db'
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.expected_bubbles = grid_rows * grid_cols
        self.init_db()
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs('uploads', exist_ok=True)

    def init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS results
                         (id INTEGER PRIMARY KEY, student_id TEXT, set_version TEXT,
                          scores TEXT, total INT, image_path TEXT, processed_at TIMESTAMP)''')
            conn.commit()
            conn.close()
            print(f"Database initialized at {self.db_path}")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not load image at {img_path}, attempting grayscale fallback")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: Grayscale fallback failed for {img_path}")
                    return None, None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Resize to increase resolution
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Increase blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            # Adjust adaptive threshold for better contrast
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 8)
            # Morphological operations to refine bubbles
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=4)
            thresh = cv2.erode(thresh, kernel, iterations=2)
            cv2.imwrite(f'{self.results_dir}/thresh_{os.path.basename(img_path)}', thresh)
            print(f"Saved thresholded image at {self.results_dir}/thresh_{os.path.basename(img_path)}")
            # Find corners for perspective correction
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("Warning: No contours found, attempting Hough lines for grid")
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    corners = self.get_corners_from_lines(lines, img.shape)
                    if corners is not None:
                        width, height = 1200, 1500
                        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
                        warped = cv2.warpPerspective(img, matrix, (width, height))
                        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        warped_gray = clahe.apply(warped_gray)
                        thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                      cv2.THRESH_BINARY_INV, 51, 10)
                        thresh = cv2.dilate(thresh, kernel, iterations=4)
                        thresh = cv2.erode(thresh, kernel, iterations=2)
                        cv2.imwrite(f'{self.results_dir}/warped_{os.path.basename(img_path)}', warped)
                        cv2.imwrite(f'{self.results_dir}/thresh_{os.path.basename(img_path)}', thresh)
                        print(f"Saved warped and thresholded images for {img_path}")
                        return thresh, warped
                print("Warning: No valid grid detected, using original thresh")
                return thresh, img
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 4:
                    corners = approx.reshape(4, 2).astype(np.float32)
                    break
            else:
                print("Warning: No quadrilateral found, using original thresh")
                return thresh, img
            sorted_corners = sorted(corners, key=lambda x: x[1])
            top_corners = sorted(sorted_corners[:2], key=lambda x: x[0])
            bottom_corners = sorted(sorted_corners[2:], key=lambda x: x[0])
            corners = np.float32([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]])
            width, height = 1200, 1500
            dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            matrix = cv2.getPerspectiveTransform(corners, dst_pts)
            warped = cv2.warpPerspective(img, matrix, (width, height))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_gray = clahe.apply(warped_gray)
            thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 51, 10)
            thresh = cv2.dilate(thresh, kernel, iterations=4)
            thresh = cv2.erode(thresh, kernel, iterations=2)
            cv2.imwrite(f'{self.results_dir}/warped_{os.path.basename(img_path)}', warped)
            cv2.imwrite(f'{self.results_dir}/thresh_{os.path.basename(img_path)}', thresh)
            print(f"Saved warped and thresholded images for {img_path}")
            return thresh, warped
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            return None, None

    def get_corners_from_lines(self, lines, img_shape):
        if lines is None:
            return None
        h, w = img_shape[:2]
        corners = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            corners.extend([(x1, y1), (x2, y2)])
        if len(corners) < 4:
            return None
        corners = sorted(corners, key=lambda x: x[1])
        top_corners = sorted(corners[:2], key=lambda x: x[0])
        bottom_corners = sorted(corners[-2:], key=lambda x: x[0])
        return np.float32([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]])

    def detect_bubbles(self, thresh, img_path, student_id):
        if thresh is None:
            print("Error: Threshold image is None")
            return None
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("Error: No contours found")
                return None
            
            bubbles = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Tightened filters to reduce over-detection
                if 50 < area < 500 and 5 < w < 50 and 5 < h < 50 and 0.8 < w/h < 1.2:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.6:
                            bubbles.append((x, y, w, h))
            
            print(f"Detected {len(bubbles)} bubbles, expected ~{self.expected_bubbles}")
            print(f"Sample bubble coordinates: {bubbles[:5]}")
            
            debug_img = cv2.imread(img_path)
            if debug_img is None:
                debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            if len(bubbles) < self.expected_bubbles * 0.8 or len(bubbles) > self.expected_bubbles * 1.2:
                print(f"Warning: Bubble count {len(bubbles)} out of range, interpolating grid")
                height, width = thresh.shape
                if bubbles:
                    x_coords = [b[0] + b[2]/2 for b in bubbles]
                    y_coords = [b[1] + b[3]/2 for b in bubbles]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    row_step = (y_max - y_min) / (self.grid_rows - 1) if y_max > y_min else height / self.grid_rows
                    col_step = (x_max - x_min) / (self.grid_cols - 1) if x_max > x_min else width / self.grid_cols
                    x_start, y_start = x_min - col_step / 2, y_min - row_step / 2
                else:
                    row_step = height / self.grid_rows
                    col_step = width / self.grid_cols
                    x_start, y_start = col_step / 2, row_step / 2
                
                grid_bubbles = []
                for i in range(self.grid_rows):
                    row = []
                    for j in range(self.grid_cols):
                        x = x_start + j * col_step
                        y = y_start + i * row_step
                        w, h = 15, 15
                        row.append((int(x), int(y), int(w), int(h)))
                        cv2.rectangle(debug_img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 1)
                    grid_bubbles.append(row)
                bubbles = [b for row in grid_bubbles for b in row]
            else:
                bubbles.sort(key=lambda b: (b[1], b[0]))
                grid_bubbles = []
                for i in range(0, len(bubbles), self.grid_cols):
                    row = sorted(bubbles[i:i+self.grid_cols], key=lambda b: b[0])
                    if len(row) == self.grid_cols:
                        grid_bubbles.append(row)
                    else:
                        print(f"Warning: Row {i//self.grid_cols + 1} has {len(row)} bubbles, expected {self.grid_cols}")
                        return None
                if len(grid_bubbles) != self.grid_rows:
                    print(f"Warning: Detected {len(grid_bubbles)} rows, expected {self.grid_rows}")
                    return None
                for row in grid_bubbles:
                    for x, y, w, h in row:
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            debug_path = f"{self.results_dir}/debug_{student_id}.jpg"
            try:
                cv2.imwrite(debug_path, debug_img)
                print(f"Debug image saved at {debug_path}")
            except Exception as e:
                print(f"Error saving debug image: {str(e)}")
            return np.array(grid_bubbles).reshape(self.grid_rows, self.grid_cols, 4)
        except Exception as e:
            print(f"Error in detect_bubbles: {str(e)}")
            return None

    def extract_responses(self, grid, img):
        if grid is None:
            print("Error: Grid is None")
            return [None] * (self.grid_rows + 1)
        responses = [None] * (self.grid_rows + 1)
        try:
            thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            thresh = clahe.apply(thresh)
            thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 51, 10)
            for q in range(self.grid_rows):
                q_bubbles = []
                for opt_idx, (x, y, w, h) in enumerate(grid[q]):
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size == 0:
                        q_bubbles.append((opt_idx, 0))
                        continue
                    fill_ratio = np.sum(roi == 255) / roi.size
                    q_bubbles.append((opt_idx, fill_ratio))
                    print(f"Q{q+1}, Opt{opt_idx}: Fill ratio = {fill_ratio:.2f}")
                q_bubbles.sort(key=lambda x: x[1], reverse=True)
                # Lower threshold for faint marks
                marked = [opt for opt, ratio in q_bubbles if ratio > 0.02]
                if len(marked) > 1 and q + 1 not in [16, 59]:
                    marked = [q_bubbles[0][0]] if q_bubbles[0][1] > q_bubbles[1][1] * 1.2 else []  # Loosened ratio check for close values
                responses[q+1] = marked if len(marked) > 1 else marked[0] if marked else None
            return responses
        except Exception as e:
            print(f"Error in extract_responses: {str(e)}")
            return [None] * (self.grid_rows + 1)

    def score_responses(self, responses):
        subject_scores = {sub: 0 for sub in SUBJECTS}
        total = 0
        flagged = []
        for q in range(1, self.grid_rows + 1):
            correct = self.key[q]
            student = responses[q]
            print(f"Q{q}: Student={student}, Correct={correct}")
            if student is None:
                flagged.append(q)
                continue
            if isinstance(correct, str) and ' ' in correct:
                correct_set = set(OPTION_MAP[c] for c in correct.split())
                student_set = set(student) if isinstance(student, list) else {student}
                score = 1 if student_set == correct_set else 0.5
            else:
                correct_val = OPTION_MAP[correct]
                student_val = student if isinstance(student, int) else student[0] if isinstance(student, list) and student else None
                score = 1 if student_val == correct_val else 0.5 if student_val is not None else 0
            if score == 0:
                flagged.append(q)
            for sub, (start, end) in SUBJECTS.items():
                if start <= q <= end:
                    subject_scores[sub] += score
                    break
            total += score
        print(f"Flagged {len(flagged)} questions, error rate: {len(flagged)/self.grid_rows:.2%}")
        return subject_scores, total, flagged if len(flagged) / self.grid_rows <= 0.05 else flagged + ['High error']

    def process_sheet(self, img_path, student_id):
        try:
            thresh, img = self.preprocess_image(img_path)
            if thresh is None or img is None:
                print(f"Error: Preprocessing failed for {img_path}")
                return {'error': 'Image preprocessing failed'}
            grid = self.detect_bubbles(thresh, img_path, student_id)
            if grid is None:
                print(f"Error: Grid detection failed for {img_path}")
                return {'error': 'Grid detection failed'}
            responses = self.extract_responses(grid, img)
            scores, total, flagged = self.score_responses(responses)
            processed_path = f'{self.results_dir}/{student_id}_processed.jpg'
            try:
                cv2.imwrite(processed_path, img)
                print(f"Saved processed image at {processed_path}")
            except Exception as e:
                print(f"Error saving processed image: {str(e)}")
            result = {'student_id': student_id, 'set_version': self.set_version, 'scores': scores, 'total': total, 'flagged': flagged}
            json_path = f'{self.results_dir}/{student_id}.json'
            try:
                with open(json_path, 'w') as f:
                    json.dump(result, f)
                print(f"Saved JSON result at {json_path}")
            except Exception as e:
                print(f"Error saving JSON result: {str(e)}")
            csv_path = f'{self.results_dir}/{student_id}.csv'
            try:
                pd.DataFrame([result]).to_csv(csv_path, index=False)
                print(f"Saved CSV result at {csv_path}")
            except Exception as e:
                print(f"Error saving CSV result: {str(e)}")
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute("INSERT INTO results (student_id, set_version, scores, total, image_path, processed_at) VALUES (?, ?, ?, ?, ?, ?)",
                          (student_id, self.set_version, json.dumps(scores), total, img_path, datetime.now()))
                conn.commit()
                conn.close()
                print(f"Saved result to database for {student_id}")
            except Exception as e:
                print(f"Error saving to database: {str(e)}")
            return result
        except Exception as e:
            print(f"Error in process_sheet: {str(e)}")
            return {'error': f'Processing failed: {str(e)}'}