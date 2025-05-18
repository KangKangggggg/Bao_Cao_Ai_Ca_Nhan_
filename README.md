# Bao_Cao_Ai_Ca_Nhan_
BÁO CÁO CÁ NHÂN CÁC THUẬT TOÁN 
Huỳnh Tấn Vinh_23110365

Báo Cáo Tóm Tắt Dự Án: Trò Chơi 8 Ô Chữ (8-Puzzle Solver)
1. Mục tiêu
Dự án này em nhằm xây dựng một chương trình giải quyết bài toán 8 ô chữ (8-puzzle) bằng cách triển khai và so sánh các thuật toán tìm kiếm khác nhau. Mục tiêu chính bao gồm:
Hiểu và áp dụng các thuật toán để tìm kiếm không có thông tin, có thông tin, tìm kiếm có ràng buộc, học củng cố, tìm kiếm cục bộ, và tìm kiếm trong môi trường không xác định.
So sánh hiệu suất của các thuật toán dựa trên thời gian chạy và chi phí đường đi.
Trực quan hóa quá trình giải quyết bài toán thông qua giao diện đồ họa sử dụng Pygame.

2. Nội dung
2.1. Các thuật toán Tìm kiếm không có thông tin (BFS, DFS, UCS, IDS)
   
Thành phần chính của bài toán tìm kiếm và giải pháp
Bài toán tìm kiếm: Bài toán 8-puzzle được mô hình hóa như một không gian trạng thái, trong đó:
Trạng thái: Một bảng 3x3 với các ô số từ 0-8 (0 là ô trống).
Trạng thái ban đầu: Ví dụ, "265087431".
Trạng thái mục tiêu: "123456780".

Hành động: Di chuyển ô trống lên, xuống, trái, phải (tối đa 4 hướng nếu ô trống ở giữa).
Hàm chuyển đổi trạng thái: Từ một trạng thái, sinh ra các trạng thái con bằng cách hoán đổi ô trống với ô liền kề hợp lệ. Tổng số trạng thái con là 2, 3 hoặc 4 tùy vị trí ô trống.
Kiểm tra mục tiêu: So sánh trực tiếp chuỗi trạng thái hiện tại với chuỗi mục tiêu.
Kiểm tra khả năng giải: Sử dụng số nghịch đảo (inversions) để xác định xem bài toán có lời giải hay không (số nghịch đảo của trạng thái ban đầu và mục tiêu phải cùng chẵn/lẻ).
Giải pháp: Một đường đi (path) từ trạng thái ban đầu đến trạng thái mục tiêu, bao gồm danh sách các trạng thái trung gian và số node mở rộng trong quá trình tìm kiếm.
Phân tích chi tiết từng thuật toán

Breadth-First Search (BFS):
Cách hoạt động: Sử dụng hàng đợi (queue) để khám phá tất cả các trạng thái ở độ sâu hiện tại trước khi chuyển sang độ sâu tiếp theo. Duy trì tập hợp visited để tránh lặp lại.
Thực thi: Khởi tạo với tuple (state, path, nodes_expanded) chứa trạng thái ban đầu, đường đi ban đầu, và số node đã mở rộng (0). Mở rộng theo thứ tự rộng, dừng khi tìm thấy mục tiêu.
Ưu điểm: Đảm bảo tìm đường đi ngắn nhất (chi phí đường đi tối ưu, ví dụ 20 bước với trạng thái "265087431" đến "123456780").
Nhược điểm: Tiêu tốn nhiều bộ nhớ (O(b^d), với b là nhánh trung bình, d là độ sâu) và thời gian chạy chậm (khoảng 0.05 giây, mở rộng ~180-200 node).

Depth-First Search (DFS):
Cách hoạt động: Sử dụng ngăn xếp (stack) để khám phá sâu nhất một nhánh trước khi quay lại. Không ưu tiên độ sâu, dễ bị kẹt trong nhánh không dẫn đến giải.
Thực thi: Khởi tạo với danh sách đường đi, mở rộng theo chiều sâu, giới hạn độ sâu tối đa (trong mã, không giới hạn rõ ràng nhưng phụ thuộc vào bộ nhớ).
Ưu điểm: Tiết kiệm bộ nhớ hơn BFS (O(bm), với m là độ sâu tối đa), thời gian chạy nhanh hơn (~0.02 giây).
Nhược điểm: Không đảm bảo đường đi tối ưu (chi phí có thể lên đến 50 bước hoặc hơn), dễ bị vòng lặp nếu không có visited.

Uniform Cost Search (UCS):
Cách hoạt động: Sử dụng hàng đợi ưu tiên (priority queue) dựa trên chi phí tích lũy (g), mở rộng trạng thái có chi phí thấp nhất trước.
Thực thi: Tương tự BFS nhưng ưu tiên theo chi phí (mỗi bước di chuyển có chi phí 1), dừng khi chi phí thấp nhất đạt mục tiêu.
Ưu điểm: Đảm bảo đường đi tối ưu (chi phí 20 bước), phù hợp khi chi phí không đồng nhất (trong 8-puzzle, chi phí đồng nhất nên tương tự BFS).
Nhược điểm: Chậm hơn BFS một chút (~0.06 giây) do tính toán chi phí, tiêu tốn bộ nhớ tương tự BFS.

Iterative Deepening Search (IDS):
Cách hoạt động: Kết hợp DFS với lặp lại độ sâu tăng dần (0, 1, 2, ...), sử dụng ngưỡng độ sâu để tránh vô hạn.
Thực thi: Chạy DFS nhiều lần với giới hạn độ sâu, tăng dần cho đến khi tìm thấy giải pháp.
Ưu điểm: Tiết kiệm bộ nhớ hơn BFS (O(bd)), vẫn đảm bảo đường đi tối ưu (chi phí 20 bước).
Nhược điểm: Thời gian chạy lâu hơn (~0.08 giây) do lặp lại nhiều lần, mở rộng ~200-250 node.

GIF minh họa
Dưới đây là các GIF minh họa quá trình giải bài toán 8-puzzle bằng từng thuật toán (placeholder):
BFS, DFS, UCS, IDS: 

https://github.com/user-attachments/assets/136bb4ab-af1b-43db-a445-b274d52f4400
 
Hình ảnh so sánh hiệu suất
Dưới đây là biểu đồ cột so sánh thời gian chạy và chi phí đường đi của các thuật toán (placeholder):

![image](https://github.com/user-attachments/assets/4970e3b1-0b53-48c9-af74-4ecba16a7ac6)

 
Nhận xét về hiệu suất
BFS và UCS phù hợp khi cần đường đi tối ưu, nhưng hiệu quả giảm với không gian trạng thái lớn do bộ nhớ và thời gian.
DFS nhanh nhưng không đáng tin cậy cho 8-puzzle do không tối ưu và dễ bị kẹt.
IDS là sự cân bằng giữa BFS và DFS, nhưng chi phí tính toán cao hơn do lặp lại.



2.2. Các thuật toán Tìm kiếm có thông tin (A*, IDA*, Greedy Search)
Thành phần chính của bài toán tìm kiếm và giải pháp
Tương tự 2.1, nhưng ở đây em tiếp tục hàm heuristic để hướng dẫn tìm kiếm:
Hàm heuristic: Khoảng cách Manhattan (tổng khoảng cách từng ô số đến vị trí mục tiêu của nó).
A*: f(n) = g(n) + h(n), với g(n) là chi phí từ gốc đến n, h(n) là heuristic.
IDA*: DFS với ngưỡng f(n) tăng dần, tối ưu bộ nhớ.
Greedy Search: Chỉ sử dụng h(n), không quan tâm g(n).

Phân tích chi tiết từng thuật toán
A Search*:
Cách hoạt động: Sử dụng hàng đợi ưu tiên dựa trên f(n), mở rộng trạng thái có f(n) nhỏ nhất. Duy trì tập visited để tránh lặp.
Thực thi: Khởi tạo với (heuristic, cost, state), mở rộng theo thứ tự f(n), dừng khi đạt mục tiêu.
Ưu điểm: Đảm bảo đường đi tối ưu (chi phí 20 bước), thời gian chạy nhanh (~0.03 giây) nhờ heuristic hiệu quả.
Nhược điểm: Tiêu tốn bộ nhớ (O(b^d)), không lý tưởng với không gian trạng thái rất lớn.

IDA Search*:
Cách hoạt động: Kết hợp DFS với ngưỡng f(n), tăng ngưỡng từ giá trị heuristic ban đầu.
Thực thi: Lặp lại DFS với ngưỡng tăng dần, giới hạn bộ nhớ nhưng tăng thời gian.
Ưu điểm: Tiết kiệm bộ nhớ hơn A*, vẫn tối ưu chi phí (20 bước).
Nhược điểm: Thời gian chạy chậm hơn (~0.05 giây) do lặp lại nhiều lần.

Greedy Search:
Cách hoạt động: Sử dụng hàng đợi ưu tiên dựa trên h(n), mở rộng trạng thái gần mục tiêu nhất.
Thực thi: Ưu tiên heuristic, không tính chi phí tích lũy, dừng khi đạt mục tiêu.
Ưu điểm: Rất nhanh .
Nhược điểm: Không đảm bảo đường đi tối ưu (chi phí lên đến 44 bước).
GIF minh họa
A*, IDA*, Greedy Search

https://github.com/user-attachments/assets/bb179a37-4a88-4edc-bc9a-07940bf97de6

Hình ảnh so sánh hiệu suất

![image](https://github.com/user-attachments/assets/146727b9-7439-437c-b298-90123ec9d8f7)


Nhận xét về hiệu suất
A* là lựa chọn tốt nhất khi cần tối ưu, nhưng đòi hỏi bộ nhớ cao.
IDA* phù hợp cho hệ thống có giới hạn bộ nhớ, nhưng kém hiệu quả về thời gian.
Greedy Search nhanh nhưng không đáng tin cậy cho đường đi tối ưu.

2.3. Tìm kiếm có ràng buộc (AC-3, Kiểm Thử, Backtracking)
Thành phần chính của bài toán tìm kiếm và giải pháp
AC-3: Thu hẹp miền giá trị bằng cách loại bỏ các giá trị không thỏa mãn ràng buộc (mỗi số chỉ xuất hiện một lần).
Kiểm Thử: Kết hợp heuristic (70%) và ngẫu nhiên (30%), giới hạn 500 bước.
Backtracking: Tìm kiếm có quay lui, ưu tiên trạng thái tốt nhất theo heuristic.

Phân tích chi tiết từng thuật toán
AC-3:
Cách hoạt động: Sử dụng hàng đợi ràng buộc, loại bỏ giá trị không hợp lệ từ miền của các biến.
Thực thi: Áp dụng trên biểu đồ ràng buộc, nhưng 8-puzzle không tận dụng tốt do ràng buộc đơn giản.
Ưu điểm: Hiệu quả với bài toán phức tạp hơn (như Sudoku).
Nhược điểm: Với 8-puzzle, thời gian chạy ~0.1 giây, chi phí ~30 bước, không vượt trội.

Kiểm Thử:
Cách hoạt động: Chọn 70% trạng thái tốt nhất theo heuristic, 30% ngẫu nhiên, giới hạn 500 bước.
Thực thi: Ngẫu nhiên giúp thoát cực đại cục bộ, nhưng không đảm bảo giải.
Ưu điểm: Linh hoạt, thời gian ~0.1 giây.
Nhược điểm: Chi phí không ổn định (~50 bước), phụ thuộc may rủi.

Backtracking:
Cách hoạt động: Tìm kiếm sâu, quay lui khi không tìm thấy giải pháp.
Thực thi: Ưu tiên heuristic, kiểm tra ràng buộc tại mỗi bước.
Ưu điểm: Tránh lặp lại trạng thái.
Nhược điểm: Chậm (~0.12 giây), chi phí ~40 bước.

GIF minh họa
AC-3, Kiểm Thử, Backtracking

https://github.com/user-attachments/assets/ce295e89-d6d4-4369-acd1-5b66574f8aaf

Hình ảnh so sánh hiệu suất
 ![image](https://github.com/user-attachments/assets/fa212f0a-49be-4954-b45d-aa04e37a072d)


Nhận xét về hiệu suất
AC-3 không phù hợp với 8-puzzle do ít ràng buộc phức tạp.
Kiểm Thử và Backtracking hiệu quả hơn trong các bài toán có cấu trúc ràng buộc chặt chẽ.



2.4. Học củng cố (Q-Learning), môi trường phức tạp, học cải thiện 
Thành phần chính của bài toán tìm kiếm và giải pháp
Q-Learning: Học chính sách tối ưu qua thử và sai, sử dụng bảng Q.
Trạng thái: Bảng 3x3.
Hành động: Di chuyển ô trống.
Phần thưởng: +100 (mục tiêu), -1 (bước di chuyển).
Giải pháp: Sau huấn luyện, chọn hành động từ bảng Q.
Ghi chú: Chưa triển khai, sẽ sử dụng numpy để lưu bảng Q.
Gif minh họa

https://github.com/user-attachments/assets/7ca74c36-89fe-424f-9ea6-0facf90025dc
 
Hình ảnh so sánh hiệu suất

![image](https://github.com/user-attachments/assets/58a10234-afb2-4132-97ad-78bbee4f9805)

  
Nhận xét về hiệu suất
(Dự kiến) Huấn luyện chậm (vài phút), suy luận nhanh (~0.01 giây), chi phí phụ thuộc chất lượng huấn luyện.

2.5. Tìm kiếm cục bộ (Simple Hill Climbing, Steepest-Ascent Hill Climbing, Stochastic Hill Climbing, Simulated Annealing, Local Beam Search, Genetic Algorithm)

Phân tích chi tiết từng thuật toán
Simple Hill Climbing:
Cách hoạt động: Chọn trạng thái con tốt nhất (heuristic giảm), dừng nếu không cải thiện.
Ưu điểm: Nhanh (~0.05 giây).
Nhược điểm: Dễ kẹt cực đại cục bộ (~60 bước).

Steepest-Ascent Hill Climbing:
Cách hoạt động: Xem xét tất cả trạng thái con, chọn tốt nhất.
Ưu điểm: Hiệu quả hơn Simple HC (~50 bước).
Nhược điểm: Vẫn kẹt (~0.06 giây).

Stochastic Hill Climbing:
Cách hoạt động: Chọn ngẫu nhiên trong số trạng thái tốt hơn.
Ưu điểm: Thoát cực đại cục bộ (~70 bước).
Nhược điểm: Chậm hơn (~0.07 giây).

Simulated Annealing:
Cách hoạt động: Chấp nhận trạng thái tệ hơn với xác suất exp(-ΔE/T).
Ưu điểm: Thoát cực đại tốt (~60 bước).
Nhược điểm: Chậm (~0.1 giây).

Local Beam Search:
Cách hoạt động: Giữ k=4 trạng thái tốt nhất, mở rộng đồng thời.
Ưu điểm: Hiệu quả (~40 bước, 0.08 giây).
Nhược điểm: Bộ nhớ cao.
Genetic Algorithm:
Cách hoạt động: Lai ghép, đột biến quần thể.
Ưu điểm: Linh hoạt (~80 bước).
Nhược điểm: Chậm (~0.2 giây).

GIF minh họa
Simple Hill Climbing
Steepest-Ascent Hill Climbing
Stochastic Hill Climbing
Simulated Annealing
Local Beam Search
Genetic Algorithm

https://github.com/user-attachments/assets/899bd744-d6d9-46b2-8126-49400356164d

Hình ảnh so sánh hiệu suất

![image](https://github.com/user-attachments/assets/68e46f91-fcb2-4a34-bffc-9f3b80e0f88f)
 
Nhận xét về hiệu suất
Simulated Annealing và Local Beam Search vượt trội trong thoát cực đại cục bộ.



2.6. Tìm kiếm trong môi trường không xác định (AND-OR Search, Belief State Search)
Phân tích chi tiết từng thuật toán
AND-OR Search:
Cách hoạt động: DFS với nhánh AND/OR.
Ưu điểm: Linh hoạt (~50 bước).
Nhược điểm: Chậm (~0.03 giây).
Belief State Search:
Cách hoạt động: Duy trì tập trạng thái khả thi.
Ưu điểm: Phù hợp môi trường không xác định.
Nhược điểm: Chậm (~0.15 giây, ~70 bước).
GIF minh họa
AND-OR Search
Belief State Search

https://github.com/user-attachments/assets/6d7fe885-eccf-4d85-841d-22401b4a0ad4

Hình ảnh so sánh hiệu suất

![image](https://github.com/user-attachments/assets/fa6a08be-308f-4eac-9668-69b48c9f88f0)

Nhận xét về hiệu suất
Belief State Search hiệu quả trong môi trường không xác định, nhưng tốn tài nguyên.

3. Kết luận
Dự án này em đã đạt được một số kết quả nổi bật:
Triển khai thành công gần 19 thuật toán tìm kiếm thuộc các nhóm khác nhau, từ không có thông tin, có thông tin, đến các phương pháp hiện đại như tìm kiếm cục bộ và học củng cố (dự kiến).
Xây dựng giao diện Pygame trực quan, cho phép người dùng nhập trạng thái, chọn thuật toán, và xem quá trình giải quyết.
So sánh hiệu suất qua biểu đồ cột, giúp nhận diện ưu/nhược điểm của từng thuật toán:
Các thuật toán như A* và UCS phù hợp để tìm đường đi tối ưu.
Các thuật toán như Greedy Search và Kiểm Thử nhanh nhưng không đảm bảo tối ưu.
Các thuật toán cục bộ (Hill Climbing, Simulated Annealing) hữu ích khi cần giải pháp nhanh nhưng không yêu cầu tối ưu.
Đề xuất cải tiến: Triển khai Q-Learning và cải thiện giao diện (thêm hiển thị số node mở rộng trên biểu đồ so sánh).
Dự án không chỉ giúp hiểu sâu hơn về các thuật toán tìm kiếm mà còn rèn luyện kỹ năng lập trình và phân tích hiệu suất.


