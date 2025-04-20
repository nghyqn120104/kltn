$(document).ready(function () {
    $.ajax({
        url: "http://127.0.0.1:5000/get-news", // Địa chỉ API Flask
        type: "GET",
        dataType: "json",
        success: function (data) {
            if (data.error) {
                $(".table-container").html(`<p class="text-danger">${data.error}</p>`);
                return;
            }

            // Kiểm tra xem dữ liệu có phải là mảng không
            let newsList = Array.isArray(data.news) ? data.news : data;

            if (!Array.isArray(newsList)) {
                $(".table-container").html(`<p class="text-danger">Dữ liệu trả về không hợp lệ.</p>`);
                return;
            }

            let tableHTML = `
                <table class="table table-bordered">
                    <thead>
                        <tr>
                            <th>STT</th>
                            <th>Nội dung</th>
                            <th>Nhãn</th>
                        </tr>
                    </thead>
                    <tbody>`;

            newsList.forEach(news => {
                tableHTML += `
                    <tr>
                        <td class="th-id">${news.id}</td>
                        <td>${news.content}</td>
                        <td>${news.label}</td>
                    </tr>`;
            });

            tableHTML += `</tbody></table>`;

            $(".table-container").html(tableHTML);
        },
        error: function () {
            $(".table-container").html(`<p class="text-danger">Không thể tải dữ liệu từ server.</p>`);
        }
    });
});
