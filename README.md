# POKÉSURVIVOR SELF-LEARNING AUTOPLAY BOT (v3.1)

Hệ thống bot tự động chơi, né đòn, tự ngắm bắn mục tiêu (Auto-Aim), nâng cấp chiêu thức theo Meta game và tự học hỏi cải thiện thời gian sống sót qua nhiều ván đấu cho tựa game **PokéSurvivor** (`https://pokesurvivor.doofcraft.com/`).

---

## 1. Điểm nổi bật & Cải tiến đột phá v3.1

### 1.1. Triệt tiêu hoàn toàn hiện tượng co giãn / nhấp nháy màn hình (Zero-Jitter Engine)
- **Nguyên nhân cũ**: Playwright gọi `.screenshot()` trên phần tử canvas kích hoạt sự kiện reflow và scroll nội bộ, khiến listener `resize()` của game liên tục co cụm và giật tỷ lệ khung hình.
- **Giải pháp v3.1**: Trích xuất khung hình trực tiếp bằng Base64 qua `toDataURL('image/jpeg', 0.6)` chạy trong context JavaScript với tốc độ cực nhanh (~13ms). Hoàn toàn không tác động vào DOM, giữ màn hình canvas ổn định 100% không rung giật.

### 1.2. Radar quét vật cản đa cự ly & Kiting bình nguyên rộng (Grassland Sweeping)
- **Phát hiện địa hình**: Tích hợp bộ lọc màu HSV phân biệt chính xác hồ nước (Blue/Cyan), vách đá cao và bóng đổ (Dark cliff edges), tảng đá tròn xám (Low saturation boulders), và thân cây rừng.
- **Radar 4 cự ly**: Quét đa bán kính (45px, 80px, 120px, 160px) theo 8 hướng. Khi phát hiện mép nước hoặc đá ở cự ly 160px, bot chủ động uốn góc chạy mượt mà thay vì đợi đâm vào mới né.
- **Chiến thuật bình nguyên rộng**: Bản đồ Shady Forest có hồ nước và vách đá ở góc Tây Bắc (North-West) và cánh đồng cỏ vô tận ở phía Đông và Đông Nam (East / South-East). Bot ưu tiên duy trì đường chạy lướt dài 3.2 giây trên đồng cỏ rộng, kéo đàn quái thành một hàng dài phía sau lưng để chiêu thức Ember/Fire Blast xuyên thẳng tiêu diệt toàn bộ.

### 1.3. Kích hoạt Native Auto-Aim của game
- Tự động kích hoạt cơ chế `AIM: AUTO` (Key T) tích hợp sẵn trong engine PokéSurvivor ở đầu mỗi ván đấu.
- Loại bỏ toàn bộ thao tác di chuột thủ công để tránh tranh chấp quyền điều khiển chiêu thức và loại bỏ các tooltip giao diện che khuất tầm nhìn.

### 1.4. Động cơ học sâu & Rút kinh nghiệm đa tầng (Root-Cause Adaptive Learning)
- Tự động phân loại nguyên nhân dừng ván: `OBSTACLE_TRAP` (Kẹt địa hình), `SWARM_OVERWHELM` (Bị quái bao vây), hay `SLOW_EXP` (Chậm tăng cấp).
- **Minh chứng tiến bộ thực tế (Run #21 $\rightarrow$ Run #22)**:
  * **Ván #21**: Sống sót 69.6s nhưng chỉ đạt Lv.1 do chưa gom đủ ngọc $\rightarrow$ Hệ thống tự động phân tích: *"[RÚT KINH NGHIỆM TĂNG CẤP] Cấp độ còn chậm (Lv.1) -> Tăng trọng số thu thập ngọc EXP lên 0.43"*.
  * **Ván #22 (Áp dụng ngay sau đó)**: Thời gian sống sót tăng lên **78.4 giây**, cấp độ nhảy vọt lên **Level 5** (chọn liên tiếp: Fire Spin, Ember Lv.2, Ember Lv.3, Power Rare), điểm tổng hợp đạt **201.8 pts** (kỷ lục mới)!

---

## 2. Bảng so sánh kết quả thực tế qua các lượt chơi

| Chỉ số đánh giá | Lượt ban đầu (Baseline #1) | Lượt #21 | Lượt #22 (Chính sách v3.1) | Mức độ cải thiện |
| :--- | :---: | :---: | :---: | :---: |
| **Thời gian sinh tồn** | `36.2 giây` | `69.6 giây` | **`78.4 giây`** | **Tăng +116.5%** |
| **Cấp độ đạt được** | `Level 1` | `Level 1` | **`Level 5 (4 thẻ nâng cấp)`** | **Tăng gấp 5 lần** |
| **Số đòn trúng** | Bị áp sát liên tục | Chỉ 12 đòn/70s | Duy trì kiting an toàn | **Tối ưu né tránh** |
| **Hiện tượng nháy màn hình** | Xuất hiện giật | **0% (Đã triệt tiêu)** | **0% (Đã triệt tiêu)** | **Hoàn toàn biến mất** |
| **Va chạm đá / nước** | Dính vật cản | **0 lần (Né từ xa)** | **0 lần (Né từ xa)** | **Điều hướng đồng cỏ** |
| **Điểm số đánh giá** | `5.0 pts` | `134.2 pts` | **`201.8 pts`** | **Kỷ lục mới** |

---

## 3. Kiến thức Meta, Cổng Rift Tím & Bộ Chiêu Kết Hợp (Bond)

### 3.1. Cơ chế Cổng Rift Tím & Săn Boss (Zero Risk, High Reward)
- **Đặc điểm**: Cổng Rift tím là một vòng xoáy không gian xuất hiện trên bản đồ, đi kèm mũi tên tím chỉ đường trên HUD.
- **Quy tắc sinh tử trong Rift**: Khi bước vào cổng, bạn được chuyển vào Đấu trường Boss (Boss Arena). **Nếu chết trong đấu trường Boss, bạn KHÔNG bị Game Over!** Bạn chỉ bị văng trở lại bản đồ chính để tiếp tục sinh tồn. Do đó, bước vào Rift là cơ hội 100% có lợi và hoàn toàn không có rủi ro kết thúc ván đấu!
- **Phần thưởng khi diệt Boss**:
  1. **Item độc nhất (Held Item)**: Mỗi ván chỉ được trang bị đúng 1 Item (từ hòm hoặc giết Boss). Bot tự động giữ Item xịn (Leftovers/Charcoal/Shell Bell) hoặc đổi khi nhặt được Item bậc cao hơn.
  2. **Kết hợp chiêu (Bond / Fusion)**: Kết hợp 2 chiêu thức vào cùng 1 nút để **kích hoạt đồng thời cùng một lúc**!
  3. **Vé Gacha và Vàng**: Gom lượng lớn tài nguyên mở khóa Pokémon vĩnh viễn.

### 3.2. Cấu trúc bộ 4 chiêu kết hợp tối thượng (1 Buff + 3 Tấn công)
Tối đa trong game có thể điều khiển 4 nhóm chiêu kích hoạt cùng lúc. Để đạt hiệu quả hủy diệt tối đa ở các mốc 15–20 phút:
- **1 Chiêu Buff**: `Swords Dance` (tăng mạnh sát thương vật lý), `Nasty Plot` (tăng sát thương đặc biệt), hoặc `Harden` / `Agility`.
- **3 Chiêu Tấn công hỏa lực**: `Fire Blast` (Ember tiến hóa), `Flamethrower` / `Fire Spin` (quét bão lửa diện rộng), và `Take Down` (Tackle tiến hóa).
- **Cơ chế Bond**: Khi nhận thưởng từ Boss, bot ưu tiên ghép chiêu Buff vào chiêu Tấn công chính để mỗi khi xả đạn, nhân vật tự động kích hoạt buff hỏa lực cùng lúc! Khi đã đủ 4 nhóm chiêu chuẩn, bot vẫn tiếp tục vào Rift hạ Boss để tích lũy vé gacha và vàng tối đa.

### 3.3. Công thức tiến hóa chiêu thức (Golden Move Evolution)
1. **Nâng chiêu thức cơ bản lên Level 5**: Dồn điểm vào **Ember** của Scorbunny.
2. **Sở hữu thẻ chỉ số kích hoạt tiến hóa**:
   - `Ember` (Lv.5) + **`Power`** $\rightarrow$ Tiến hóa thành **`Fire Blast`** (Kim thiếp viền vàng - Chiêu thức quét sạch toàn bộ quái vật trên màn hình).
   - `Tackle` + **`Vigor`** $\rightarrow$ Tiến hóa thành **`Take Down`**.
   - `Sprint` $\rightarrow$ Tăng tốc độ di chuyển để không bao giờ bị quái đuổi kịp.

---

## 4. Hướng dẫn sử dụng

### 4.1. Khởi động chế độ sinh tồn không giới hạn
Đã loại bỏ giới hạn thời gian 300s cũ. Bot mặc định chơi không giới hạn (tối đa 60 phút hoặc đến khi kết thúc tự nhiên) để farm tối đa vàng và quái ở các mốc phút cao:
```bash
# Chạy tự động không giới hạn thời gian (tự học hỏi và lưu kinh nghiệm sau mỗi ván)
python main.py --runs 3
```

### 4.2. Phím nóng điều khiển trong lúc chạy
- **Phím F8**: Tạm dừng hoặc tiếp tục gửi thao tác bàn phím (để quan sát bot hoặc can thiệp bằng tay).
- **Phím F9**: Dừng khẩn cấp toàn bộ hệ thống ngay lập tức và lưu lại toàn bộ dữ liệu.

### 4.3. Xem báo cáo thống kê tiến trình học
```bash
python main.py --report
```

```
Game_tool/
├── main.py                     # Entry point chính chạy bot
├── auth_state.json             # Lưu trữ phiên đăng nhập (Cookies / LocalStorage)
├── README.md                   # Tài liệu hướng dẫn sử dụng
│
├── pokesurvivor/               # 11 Module chức năng chuyên biệt
│   ├── browser_controller.py   # Điều khiển Playwright, bàn phím, điều hướng Solo
│   ├── game_observer.py        # Đọc DOM và chụp ảnh canvas cho OpenCV
│   ├── perception.py           # Computer Vision: phát hiện quái, tâm đàn quái, đạn, ngọc EXP
│   ├── danger_estimator.py     # Đánh giá độ an toàn 8 hướng di chuyển
│   ├── movement_policy.py      # Kiting tiếp tuyến, kéo đàn quái, gom EXP
│   ├── upgrade_policy.py       # Meta nâng cấp: Rush Ember Lv.5, Power, Fire Blast, Sprint
│   ├── learning_engine.py      # Tự thích nghi, tăng thời gian sống sót qua từng ván
│   ├── memory_store.py         # Lưu trữ lịch sử dài hạn (JSONL, CSV, JSON)
│   ├── run_manager.py          # Quản lý vòng lặp chơi liên tục đến khi chết tự nhiên
│   ├── state_tracker.py        # Theo dõi HP, số lần trúng đòn, thời gian thực
│   └── metrics_reporter.py     # Báo cáo và bảng so sánh tăng trưởng hiệu suất
│
├── data/                       # Bộ nhớ dài hạn vĩnh viễn
│   ├── run_history.jsonl       # Lịch sử chi tiết từng ván đấu
│   ├── policy.json             # Chiến thuật hiện tại
│   ├── best_policy.json        # Chiến thuật tốt nhất từng lưu
│   └── metrics.csv             # Số liệu thống kê so sánh
│
├── debug_artifacts/            # Chứa ảnh chụp canvas, DOM modal đã khảo sát
└── scratch_scripts/            # Script kiểm thử độc lập
```

---

## 4. Hướng dẫn khởi chạy

### 4.1. Lệnh cơ bản

1. **Chơi tự do đến khi chết và tự học qua từng ván (Mặc định 20 phút/ván):**
```powershell
python main.py --runs 3
```

2. **Chạy ván dài tối đa 20 phút (1200 giây) để leo mốc kỷ lục:**
```powershell
python main.py --runs 5 --max-duration 1200
```

3. **Chạy chế độ trình duyệt ẩn (Headless):**
```powershell
python main.py --runs 3 --headless
```

4. **Xem bảng thống kê tiến bộ các ván đã lưu:**
```powershell
python main.py --report
```

### 4.2. Phím nóng an toàn
- **`F8`**: **Tạm dừng / Tiếp tục (Pause/Resume)** bot. Nhả toàn bộ phím ngay lập tức để người dùng kiểm soát.
- **`F9`**: **Dừng khẩn cấp (Emergency Stop)**. Nhả toàn bộ phím và đóng trình duyệt an toàn.
