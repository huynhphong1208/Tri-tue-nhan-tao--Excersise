def get_manhattan_distance(current_matrix):
    # Định nghĩa vị trí dòng và cột của Goal State (1 đến 8)
    # Định dạng: số -> (dòng, cột)
    goal_positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1)
        # Số 0 là ô trống, không tính khoảng cách Manhattan
    }
    
    total_distance = 0
    
    # Duyệt qua từng ô trong ma trận hiện tại
    for r in range(3):
        for c in range(3):
            val = current_matrix[r][c]
            # Chỉ tính khoảng cách cho các số từ 1 đến 8
            if val in goal_positions:
                goal_r, goal_c = goal_positions[val]
                # Công thức: |r1 - r2| + |c1 - c2|
                total_distance += abs(r - goal_r) + abs(c - goal_c)
                
    return total_distance

def input_matrix():
    print("Nhập ma trận 3x3 (nhập 3 số trên mỗi dòng, cách nhau bằng khoảng trắng):")
    matrix = []
    for i in range(3):
        while True:
            try:
                row = list(map(int, input(f"Dòng {i+1}: ").split()))
                if len(row) != 3:
                    print("Vui lòng nhập đúng 3 số!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu nhập vào phải là số nguyên!")
    return matrix

# --- Chương trình chính ---
if __name__ == "__main__":
    # Nhập dữ liệu từ bàn phím
    matrix = input_matrix()
    
    # Tính toán và in kết quả
    distance = get_manhattan_distance(matrix)
    print("\n--- Kết quả ---")
    print(f"Khoảng cách Manhattan của ma trận là: {distance}")
