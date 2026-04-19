/**
 * app.js — Drawing Canvas + Prediction Engine
 * Handles mouse/touch drawing, canvas export, and confidence bar rendering.
 */

(function () {
    "use strict";

    // ─── Canvas Setup ───────────────────────────────────────────
    const canvas = document.getElementById("drawingCanvas");
    const ctx = canvas.getContext("2d");
    const predictBtn = document.getElementById("predictBtn");
    const clearBtn = document.getElementById("clearBtn");
    const predictionsContainer = document.getElementById("predictionsContainer");

    let isDrawing = false;

    // Canvas defaults
    function initCanvas() {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
    }

    initCanvas();

    // ─── Drawing Helpers ────────────────────────────────────────
    function getPosition(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches && e.touches.length > 0) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPosition(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getPosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDrawing(e) {
        if (e) e.preventDefault();
        isDrawing = false;
        ctx.beginPath();
    }

    // ─── Mouse Events ───────────────────────────────────────────
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    // ─── Touch Events (Mobile Support) ──────────────────────────
    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDrawing, { passive: false });
    canvas.addEventListener("touchcancel", stopDrawing, { passive: false });

    // ─── Clear Canvas ───────────────────────────────────────────
    clearBtn.addEventListener("click", function () {
        initCanvas();
        predictionsContainer.innerHTML =
            '<p class="placeholder-text">Draw something and click <strong>Predict</strong></p>';
    });

    // ─── Predict ────────────────────────────────────────────────
    predictBtn.addEventListener("click", async function () {
        predictBtn.disabled = true;
        predictBtn.textContent = "Analyzing...";
        predictionsContainer.innerHTML =
            '<p class="placeholder-text">Thinking...</p>';

        try {
            const dataURL = canvas.toDataURL("image/png");

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL }),
            });

            const data = await response.json();

            if (data.error) {
                predictionsContainer.innerHTML =
                    `<p class="error-text">${data.error}</p>`;
                return;
            }

            renderPredictions(data.predictions);
        } catch (err) {
            predictionsContainer.innerHTML =
                `<p class="error-text">Connection error. Is the server running?</p>`;
            console.error("Predict error:", err);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict";
        }
    });

    // ─── Render Confidence Bars ─────────────────────────────────
    function renderPredictions(predictions) {
        const rankLabels = ["1st", "2nd", "3rd"];
        const barClasses = ["bar-first", "bar-second", "bar-third"];

        let html = "";

        predictions.forEach(function (pred, index) {
            const confidence = pred.confidence;
            const label = pred.label;
            const rank = rankLabels[index] || `${index + 1}th`;
            const barClass = barClasses[index] || "bar-third";

            html += `
                <div class="prediction-row">
                    <div class="prediction-info">
                        <span class="rank-badge ${barClass}">${rank}</span>
                        <span class="prediction-label">${label}</span>
                        <span class="prediction-confidence">${confidence.toFixed(1)}%</span>
                    </div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill ${barClass}"
                             style="width: 0%;"
                             data-width="${confidence}">
                        </div>
                    </div>
                </div>
            `;
        });

        predictionsContainer.innerHTML = html;

        // Animate bars after render
        requestAnimationFrame(function () {
            const bars = predictionsContainer.querySelectorAll(".confidence-bar-fill");
            bars.forEach(function (bar) {
                const targetWidth = bar.getAttribute("data-width");
                bar.style.width = targetWidth + "%";
            });
        });
    }
})();
