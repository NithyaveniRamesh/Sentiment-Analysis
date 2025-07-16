document.getElementById("sentiment-form").addEventListener("submit", function (e) {
    e.preventDefault();

    const reviewText = document.getElementById("review").value;
    

    fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ review: reviewText }),  
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(res => res.json())
    .then(data => {
        const emoji = getEmoji(data.sentiment);
        const resultBox = document.getElementById("result");
        resultBox.innerHTML = `Sentiment: <strong>${data.sentiment}</strong> ${emoji}`;
        resultBox.style.display = "block";
    });
});

function getEmoji(sentiment) {
    switch (sentiment) {
        case 'positive': return '😊';
        case 'negative': return '😞';
        case 'neutral': return '😐';
        default: return '';
    }
}

document.getElementById("darkModeToggle").addEventListener("change", function () {
    document.body.classList.toggle("dark-mode");
});
