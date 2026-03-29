# Stage 1: Floor plan parsing using OpenCV
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from mpl_toolkits.mplot3d import Axes3D

print("\n========== PIPELINE START ==========")
print("Stage 1: Floor Plan Parsing")

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

image = cv2.imread(image_path)

if image is None:
    print("Error loading image")
else:
    scale = image.shape[1] / 15  # px → meters conversion

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # ───────── WALL DETECTION ─────────
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                            minLineLength=50, maxLineGap=10)

    wall_img = image.copy()
    filtered_lines = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]

            if abs(x1-x2) < 10 or abs(y1-y2) < 10:
                filtered_lines.append((x1,y1,x2,y2))
                cv2.line(wall_img, (x1,y1), (x2,y2), (0,255,0), 2)

    print("Filtered Walls:", len(filtered_lines))

    print("\nStage 2: Geometry Reconstruction")
    print("--- GEOMETRY MODEL ---")
    print(f"Walls stored as edges: {len(filtered_lines)}")
    print("Geometry: Walls treated as graph edges")

    # ───────── ROOM DETECTION ─────────
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    room_img = image.copy()
    rooms = []

    total_pixels = image.shape[0] * image.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 0.01 * total_pixels < area < 0.2 * total_pixels:
            x, y, w, h = cv2.boundingRect(cnt)
            rooms.append((x,y,w,h))

    print("Detected Rooms:", len(rooms))

    print("\nStage 4: Material Analysis")

    total_rcc = 0
    total_aac = 0

    for i, (x,y,w,h) in enumerate(rooms):
        area_val = w*h
        span = max(w, h)

        width_m = w / scale
        height_m = h / scale
        area_m2 = (w*h)/(scale*scale)

        # ───── LOAD-BEARING LOGIC ─────
        margin = 0.06 * image.shape[1]

        near_boundary = (x < margin or y < margin or 
                         (x+w) > image.shape[1]-margin or 
                         (y+h) > image.shape[0]-margin)

        is_spine = span > 0.55 * image.shape[1]

        if near_boundary or is_spine:
            wall_type = "Load-bearing"
        else:
            wall_type = "Partition"

        # ───── MATERIAL SCORING ─────
        materials = {
            "RCC": {"cost": 3, "strength": 5, "durability": 5},
            "Red Brick": {"cost": 2, "strength": 4, "durability": 3},
            "AAC Block": {"cost": 1, "strength": 3, "durability": 4}
        }

        if wall_type == "Load-bearing":
            w_cost, w_strength, w_durability = 0.2, 0.5, 0.3
        else:
            w_cost, w_strength, w_durability = 0.5, 0.2, 0.3

        best_score = -1
        material = ""
        ranked = []

        for m, p in materials.items():
            score = (w_strength*p["strength"] +
                     w_durability*p["durability"] -
                     w_cost*p["cost"])

            ranked.append((m, round(score,2)))

            if score > best_score:
                best_score = score
                material = m

        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        options = [r[0] for r in ranked]

        reason = f"{wall_type} wall → evaluated using cost-strength-durability tradeoff → {material} selected"

        if span > 0.4 * image.shape[1]:
            risk = "⚠️ Large span - reinforcement needed"
        else:
            risk = "Safe"

        # Draw
        cv2.rectangle(room_img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(room_img, f"R{i+1}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # Output
        print(f"\nRoom {i+1}")
        print(f"Pixel Size: {w}x{h}")
        print(f"Real Size: {round(width_m,2)}m x {round(height_m,2)}m")
        print(f"Area: {round(area_m2,2)} m²")
        print(f"Wall Type: {wall_type}")
        print(f"Material: {material}")
        print(f"Ranking: {options}")
        print(f"Reason: {reason}")
        print(f"Risk: {risk}")

        if material == "RCC":
            total_rcc += area_val
        else:
            total_aac += area_val

    print("\nStage 5: Explainability")

    print("\n--- STRUCTURAL INSIGHTS ---")
    print("Insight: Large spans may require beam/column support.")
    print("Insight: Load-bearing walls identified using boundary + span logic.")
    print("Tradeoff: Optimized using weighted cost-strength-durability scoring.")

    # ───────── VISUALIZATION ─────────
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(wall_img, cv2.COLOR_BGR2RGB))
    plt.title("Walls")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB))
    plt.title("Rooms")

    plt.show()

    print("\nStage 3: 3D Model Generation")

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for (x1,y1,x2,y2) in filtered_lines:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if length > 0.3 * image.shape[1]:
            height = 60
            color = 'red'
        else:
            height = 30
            color = 'blue'

        dx = x2 - x1 if abs(x2-x1) > 0 else 5
        dy = y2 - y1 if abs(y2-y1) > 0 else 5

        ax.bar3d(x1, y1, 0, dx, dy, height, color=color, alpha=0.7)

    ax.set_title("3D Structural Wall Model")
    plt.show()

print("\n========== PIPELINE END ==========")