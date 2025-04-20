function sendMessage() {
    let userInput = document.getElementById("userInput").value.trim();
    let errorMessage = document.getElementById("error-message");

    if (!userInput) {
        errorMessage.style.display = "block";  
        return;
    } else {
        errorMessage.style.display = "none";  
    }
    
    appendMessage(userInput, "user-message");
    
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: userInput })
    })
    .then(response => response.json())
    .then(data => {
        let botResponse = data.label;
        appendMessage(botResponse, "bot-message");
    })
    .catch(error => {
        console.error("Lỗi:", error);
        appendMessage("Có lỗi xảy ra, vui lòng thử lại!", "bot-message");
    });

    document.getElementById("userInput").value = "";
}

// function appendMessage(text, className) {
//     let chatBox = document.getElementById("chatBox");
//     let messageDiv = document.createElement("div");
//     messageDiv.className = `message ${className}`;
//     messageDiv.textContent = text;
//     chatBox.appendChild(messageDiv);
//     chatBox.scrollTop = chatBox.scrollHeight;
// }

function appendMessage(text, className) {
    let chatBox = document.getElementById("chatBox");
    let messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    messageDiv.innerHTML = `<strong>${text}</strong>`;

    // Áp dụng màu nền theo phân loại
    if (className === "bot-message") {
        if (text === "Tin thường") {
            messageDiv.style.backgroundColor = "#d4edda"; // xanh nhạt
            messageDiv.style.color = "#155724";
        } else if (text.includes("Tin độc hại")) {
            messageDiv.style.backgroundColor = "#f8d7da"; // đỏ nhạt
            messageDiv.style.color = "#721c24";
        } else {
            messageDiv.style.backgroundColor = "#fff3cd"; // vàng cho không xác định
            messageDiv.style.color = "#856404";
        }
    }

    messageDiv.style.padding = "10px";
    messageDiv.style.margin = "5px";
    messageDiv.style.borderRadius = "5px";

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
