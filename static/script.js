async function askQuestion() {
    const question = document.getElementById("question").value;

    // Show the question
    document.getElementById("answer").innerText = "Thinking...";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })

        });

        const data = await response.json();
        document.getElementById("answer").innerText = data.answer;

    } catch (error) {
        console.error(error);
        document.getElementById("answer").innerText = "⚠️ Server error. Check backend.";
    }
}
