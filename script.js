document.addEventListener("DOMContentLoaded", function () {
    // Sample latest phishing threat message
    const latestThreatMessage = "🚨 Phishing Alert: Suspicious email from 'support@bank-secure.com'. Do not click links!";
    
    // Update the phishing threat paragraph dynamically
    document.getElementById("latestThreat").textContent = latestThreatMessage;

    // Generate a random phishing severity percentage (between 40% - 90%)
    const phishingSeverity = Math.floor(Math.random() * 51) + 40;

    document.getElementById("severityDisplay").innerHTML = `
    <div style="text-align: center;">
        <strong style="font-size: 3rem; color: red; display: block;">${phishingSeverity}% RISK!!</strong>
        <span style="font-size: 15px; color: white; display: block; margin-top: 10px;">
            Based on the scanned content, we have identified a potential risk level. 
            You may proceed with the content at your discretion. Please review the details 
            and take necessary precautions as needed.
        </span>
    </div>
`;


    // Get chart context for event distribution
    const eventCanvas = document.getElementById("eventChart");
    const eventCtx = eventCanvas.getContext("2d");

    // Ensure the chart has a fixed height for proper rendering
    eventCanvas.style.width = "100%";
    eventCanvas.style.height = "300px"; // Adjust as needed
    eventCanvas.height = 300;
    const getRandomData = () => [
    Math.floor(Math.random() * 50) + 10,  // Random value between 10-60
    Math.floor(Math.random() * 50) + 10,
    Math.floor(Math.random() * 50) + 1
    ];

    // Event Distribution Chart
    const eventData = {
    labels: ["CRITICAL", "WARNINGS", "INFOS"],
    datasets: [{
        label: "Threat Events",
        data: getRandomData(),
        backgroundColor: [
            "#eb2222",  // Red for CRITICAL
            "#ffcc00",  // Yellow for WARNINGS
            "#3366ff"   // Blue for INFOS
        ],
        hoverOffset: 12
    }]
};

    new Chart(eventCtx, {
        type: "pie",
        data: eventData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top" },
                title: { display: true, text: "Event Distribution" }
            },
            animation: { animateRotate: true, animateScale: true }
        }
    });
});
