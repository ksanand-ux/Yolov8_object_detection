async function sendImage() {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image file."); 
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://54.157.28.181:8082/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Request failed with status ${response.status}`);
        }

        const blob = await response.blob();
        const img = document.getElementById('resultImage');
        img.src = URL.createObjectURL(blob); 

    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred during prediction. Please try again."); 
    }
}
