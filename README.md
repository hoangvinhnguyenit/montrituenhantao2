# Dự báo doanh số và quản trị tồn kho

Ứng dụng Streamlit hỗ trợ phân tích doanh số, tạo đặc trưng chuỗi thời gian,
huấn luyện LSTM/GRU, lập dự báo và ước tính nhu cầu bổ sung hàng. Các chức năng
được chia thành trang độc lập trong thư mục `app_pages/`.

## Chạy ứng dụng

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Các trang

1. **Dashboard** — tổng quan doanh số, trạng thái mô hình, dự báo và tồn kho.
2. **Dữ liệu** — nhập CSV/Excel/ZIP hoặc tải mẫu theo ERD.
3. **Tiền xử lý & Feature Engineering** — xử lý thiếu, trùng, outlier; tạo đặc
   trưng lịch, lag và rolling không dùng dữ liệu tương lai.
4. **Phân tích EDA** — xu hướng, mùa vụ, nhóm hàng/cửa hàng, khuyến mãi, ngày lễ,
   giá dầu và outlier.
5. **Huấn luyện mô hình** — LSTM/GRU với chia train/validation theo thời gian và
   các chỉ số MAE, RMSE, MAPE, WAPE.
6. **Dự báo doanh số** — dự báo đệ quy từ mô hình đã huấn luyện và xuất CSV.
7. **Quản trị tồn kho** — Safety Stock, ROP, EOQ, mô phỏng tồn kho và ABC.
8. **Báo cáo tổng hợp** — tải dự báo, phân loại ABC và dữ liệu đã xử lý.

## Định dạng dữ liệu

Tải từng bảng CSV hoặc ZIP chứa một phần/toàn bộ các bảng sau. Tên tệp nên dùng
tên bảng để tự động nhận diện:

| Bảng | Cột chính | Khóa / liên kết |
|---|---|---|
| `train.csv` | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` | `id`; `store_nbr` → stores |
| `test.csv` | `id`, `date`, `store_nbr`, `family`, `onpromotion` | `id`; khóa dự báo tương ứng với sample_submission |
| `stores.csv` | `store_nbr`, `city`, `state`, `type`, `cluster` | `store_nbr` |
| `transactions.csv` | `date`, `store_nbr`, `transactions` | `date + store_nbr` |
| `oil.csv` | `date`, `dcoilwtico` | `date` |
| `holidays_events.csv` | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` | ngày có thể có nhiều sự kiện |
| `sample_submission.csv` | `id`, `sales` | `id` |

Các bảng phụ được nối theo khóa ở trên; sự kiện ngày lễ được gộp theo ngày trước
khi nối để tránh nhân bản dòng doanh số. Dữ liệu tổng quát cũng được chấp nhận
nếu có các cột ngày, nhóm hàng/sản phẩm và doanh số/số lượng (ví dụ:
`date, product, quantity`).
Các dòng `train`/`test` trùng `id` và giống hệt nhau được gộp trước khi nối;
nếu cùng `id` có nội dung khác nhau, ứng dụng báo lỗi để tránh tự ý chọn dữ liệu.

Tồn kho thực tế không nằm trong ERD; ứng dụng yêu cầu nhập tồn kho hiện tại và
lead time để tính ROP. Dữ liệu mẫu tạo một bộ nhỏ có cửa hàng, nhóm hàng,
khuyến mãi, giao dịch, giá dầu và ngày lễ.

## Kết nối Supabase (có thể bật khi sẵn sàng)

Trang **Dữ liệu → Supabase** đã hỗ trợ đọc các bảng theo ERD. Để bật kết nối,
đặt thông tin dự án trong môi trường máy chủ Streamlit hoặc secrets; không ghi
khóa vào mã nguồn hoặc Git:

```text
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<key-with-appropriate-RLS-policies>
```

Có thể dùng `.streamlit/secrets.toml` cục bộ (đưa file này vào `.gitignore`):

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "<key-with-appropriate-RLS-policies>"
```

Sau khi cài dependencies từ `requirements.txt`, dùng nút **Nạp từ Supabase**
trong trang Dữ liệu để nạp toàn bộ các dòng có quyền truy cập. Ứng dụng đọc
song song tối đa 4 bảng, yêu cầu tối đa 10.000 dòng mỗi lô (Supabase có thể áp
dụng giới hạn thấp hơn), chỉ lấy các cột cần thiết và hiển thị tiến độ sau mỗi
lô; bảng lớn vẫn có thể cần thời gian và bộ nhớ máy chủ đáng kể. Mỗi bảng dùng
kết nối riêng và ứng dụng tự thử lại một lần nếu kết nối bị ngắt tạm thời.

Bảng `train` là bắt buộc; ứng dụng cũng hỗ trợ tên cũ `train2`. Các bảng
`test`, `stores`, `transactions`, `oil`, `holidays_events` và
`sample_submission` là tùy chọn và sẽ được bỏ qua nếu chưa tồn tại. Cấp quyền
`SELECT` và chính sách RLS phù hợp cho các bảng cần đọc.
Ưu tiên khóa anon/publishable với chính sách RLS phù hợp. Nếu buộc phải dùng
service-role key, chỉ lưu trong môi trường máy chủ tin cậy; tuyệt đối không đưa
khóa đó vào trình duyệt hoặc repository.

## Trợ lý AI trong báo cáo

Cài dependencies bằng `pip install -r requirements.txt`. Để chạy chatbot cục bộ,
tạo `.streamlit/secrets.toml` (tệp đã được bỏ qua bởi Git) với khóa ở cấp cao nhất:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# Tùy chọn: CHATBOT_MODEL = "claude-sonnet-5-5"
```

Trên Streamlit Community Cloud, thêm cùng các giá trị này trong **App settings →
Secrets**. Không khai báo khóa như biến Python trong `app.py`, không gửi khóa lên
GitHub; sau khi thay đổi Secrets, khởi động lại ứng dụng nếu Streamlit chưa tự
tải lại. Nếu giao diện báo chưa đọc được khóa, kiểm tra đúng tên
`ANTHROPIC_API_KEY`, dấu ngoặc kép TOML và rằng secret nằm trong app đang chạy.
