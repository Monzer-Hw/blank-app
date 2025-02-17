import streamlit as st


"""
Streamlit app entry point for the Sudoku Solver.
Uploads an image, processes it, solves the Sudoku puzzle,
and overlays the solution on the original image.
"""

import os
from typing import Optional

import numpy as np
import streamlit as st

from src.core.detector import SudokuDetector
from src.core.solver import SudokuSolver
from src.core.ocr import SudokuTesseract
from src.core.visualizer import SudokuVisualizer
from src.io.ui import SudokuUI



detector = SudokuDetector()
ocr = SudokuTesseract()
solver = SudokuSolver()
visualizer = SudokuVisualizer()
mask_image_path = os.path.join(
            os.path.dirname(__file__), ".", "assets", "mask.png"
        )

def process_image(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Process the provided image to detect the Sudoku grid, recognize digits,
    solve the puzzle, and overlay the solution.

    :param image: Input image as a NumPy array in BGR format.
    :return: Processed image with the Sudoku solution overlay (BGR), or None if processing fails.
    """
    # --- Image Preprocessing ---
    original, _, _, thresh = detector.preprocess_image(image)

    # --- Grid Detection ---
    grid_contour = detector.find_grid_contour(thresh)
    if grid_contour is None:
        st.error("Could not detect the Sudoku grid. Please try another image.")
        return None

    warped, M = detector.perspective_transform(original, grid_contour)
    cells, cell_size = detector.extract_cells(warped)

    # --- Cell Processing and Digit Recognition ---
    processed_cells = ocr.process_cells(cells, mask_image_path)
    sudoku_grid = ocr.recognize(processed_cells)
    positions = np.where(sudoku_grid > 0, 0, 1)

    # --- Sudoku Solving ---
    try:
        solved_grid = solver.solve(sudoku_grid)
    except ValueError as e:
        st.error(str(e))
        return None

    # --- Overlay the Solution ---
    final_image = visualizer.overlay_solution(
        original, warped, solved_grid, positions, grid_contour, cell_size
    )

    return final_image


def main():
    ui = SudokuUI()
    ui.setup_page()
    ui.inject_custom_css()
    ui.display_header()

    col1, col2 = st.columns([2, 2], gap="large")

    with col1:
        uploaded_file = ui.file_uploader()
        if uploaded_file:
            image = ui.display_original_image(uploaded_file)
            if ui.solve_button():
                with ui.processing_status() as status:
                    try:
                        result = process_image(image)
                        if result is not None:
                            status.update(
                                label="✅ Processing Complete!", state="complete"
                            )
                            ui.display_solution(result, col2)
                    except Exception as e:
                        status.update(label="❌ Processing Failed", state="error")
                        st.error(f"Error: {str(e)}")

    if not uploaded_file:
        ui.display_tips(col2)


if __name__ == "__main__":
    main()

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
