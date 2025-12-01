document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.querySelector('input[type="file"]');
    const fileLabel = document.getElementById("file-label");

    if (fileInput && fileLabel) {
        fileInput.addEventListener("change", (event) => {
            const files = event.target.files;
            if (files && files.length > 0) {
                fileLabel.textContent = files[0].name;
            } else {
                fileLabel.textContent = "Choose an image";
            }
        });
    }
});
