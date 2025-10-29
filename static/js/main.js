// static/js/main.js
const DOMINO_API_BASE = window.location.origin + window.location.pathname.replace(/\/$/, '');
const ORIGINAL_API_BASE = window.DOMINO?.API_BASE || '';
const API_KEY = window.DOMINO?.API_KEY || null;

// Global state
let appState = {
    uploadedFiles: [],
    formData: {}
};

// Helper function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Helper function to make proxy API calls
async function proxyFetch(apiPath, options = {}) {
    const [basePath, queryString] = apiPath.split('?');
    const targetParam = `target=${encodeURIComponent(ORIGINAL_API_BASE)}`;
    const finalQuery = queryString ? `${queryString}&${targetParam}` : targetParam;
    const url = `${DOMINO_API_BASE}/proxy/${basePath.replace(/^\//, '')}?${finalQuery}`;
    
    const defaultHeaders = {
        'X-Domino-Api-Key': API_KEY,
        'accept': 'application/json'
    };
    
    return fetch(url, {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers
        }
    });
}

// Handle file upload
function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    appState.uploadedFiles = files;
    
    displayUploadedFiles();
}

// Display uploaded files
function displayUploadedFiles() {
    const container = document.getElementById('uploaded-files-display');
    
    if (appState.uploadedFiles.length === 0) {
        container.innerHTML = '<p class="no-files">No files uploaded yet</p>';
        return;
    }
    
    const filesHtml = appState.uploadedFiles.map(file => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-name">${file.webkitRelativePath || file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = `
        <div class="files-list">
            <h4>Uploaded Files (${appState.uploadedFiles.length})</h4>
            ${filesHtml}
        </div>
    `;
}

// Validate form
function validateForm() {
    const requiredFields = [
        { id: 'model-name', label: 'Model Name' },
        { id: 'model-owner', label: 'Model Owner' },
        { id: 'model-use-case', label: 'Model Use Case' },
        { id: 'model-usage-pattern', label: 'Model Usage Pattern' },
        { id: 'model-environment-id', label: 'Model Environment ID' }
    ];
    
    const errors = [];
    
    requiredFields.forEach(field => {
        const input = document.getElementById(field.id);
        if (!input.value.trim()) {
            errors.push(`${field.label} is required`);
            input.classList.add('error');
        } else {
            input.classList.remove('error');
        }
    });
    
    if (appState.uploadedFiles.length === 0) {
        errors.push('Please upload model files');
        document.getElementById('model-upload').classList.add('error');
    } else {
        document.getElementById('model-upload').classList.remove('error');
    }
    
    return errors;
}

// Show error messages
function showErrors(errors) {
    const errorContainer = document.getElementById('error-messages');
    
    if (errors.length === 0) {
        errorContainer.innerHTML = '';
        errorContainer.style.display = 'none';
        return;
    }
    
    errorContainer.innerHTML = `
        <div class="error-box">
            <h4>Please fix the following errors:</h4>
            <ul>
                ${errors.map(error => `<li>${error}</li>`).join('')}
            </ul>
        </div>
    `;
    errorContainer.style.display = 'block';
}

// Show loading state
function showLoading(button) {
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span> Registering Model...';
}

// Hide loading state
function hideLoading(button) {
    button.disabled = false;
    button.innerHTML = 'Register External Model with Domino';
}

// Show success message
function showSuccess() {
    const container = document.querySelector('.container');
    const successHtml = `
        <div class="success-box">
            <h2>✓ Model Registered Successfully</h2>
            <p>Your model has been registered with Domino.</p>
            <button class="btn btn-primary" onclick="resetForm()">Register Another Model</button>
        </div>
    `;
    
    container.innerHTML = successHtml;
}

// Reset form
function resetForm() {
    location.reload();
}

// Handle form submission
async function handleSubmit(event) {
    event.preventDefault();
    
    const errors = validateForm();
    if (errors.length > 0) {
        showErrors(errors);
        return;
    }
    
    showErrors([]);
    
    const submitButton = event.target.querySelector('button[type="submit"]');
    showLoading(submitButton);
    
    try {
        // Collect form data
        const formData = new FormData();
        
        formData.append('modelName', document.getElementById('model-name').value.trim());
        formData.append('modelDescription', document.getElementById('model-description').value.trim());
        formData.append('modelOwner', document.getElementById('model-owner').value.trim());
        formData.append('modelUseCase', document.getElementById('model-use-case').value.trim());
        formData.append('modelUsagePattern', document.getElementById('model-usage-pattern').value.trim());
        formData.append('modelEnvironmentId', document.getElementById('model-environment-id').value.trim());
        formData.append('modelExecutionScript', document.getElementById('model-execution-script').value.trim());
        
        // Append files
        appState.uploadedFiles.forEach(file => {
            formData.append('files', file, file.webkitRelativePath || file.name);
        });
        
        // Make API call
        const basePath = window.location.pathname.replace(/\/$/, '');
        const response = await fetch(`${basePath}/register-external-model`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Registration failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('Registration successful:', result);
        
        showSuccess();
        
    } catch (error) {
        console.error('Registration error:', error);
        showErrors([`Failed to register model: ${error.message}`]);
        hideLoading(submitButton);
    }
}

// Initialize form
function initializeForm() {
    const container = document.querySelector('.container');
    
    container.innerHTML = `
        <h1 class="welcome-title">Register External Model</h1>
        
        <div id="error-messages"></div>
        
        <form id="model-upload-form" class="model-form">
            <div class="form-group">
                <label for="model-name">Model Name <span class="required">*</span></label>
                <input type="text" id="model-name" name="modelName" required>
            </div>
            
            <div class="form-group">
                <label for="model-description">Model Description</label>
                <textarea id="model-description" name="modelDescription" rows="4"></textarea>
            </div>
            
            <div class="form-group">
                <label for="model-owner">Model Owner <span class="required">*</span></label>
                <input type="text" id="model-owner" name="modelOwner" required>
            </div>
            
            <div class="form-group">
                <label for="model-use-case">Model Use Case <span class="required">*</span></label>
                <textarea id="model-use-case" name="modelUseCase" rows="4" required></textarea>
            </div>
            
            <div class="form-group">
                <label for="model-usage-pattern">Model Usage Pattern <span class="required">*</span></label>
                <textarea id="model-usage-pattern" name="modelUsagePattern" rows="4" required></textarea>
            </div>
            
            <div class="form-group">
                <label for="model-environment-id">Model Environment ID <span class="required">*</span></label>
                <input type="text" id="model-environment-id" name="modelEnvironmentId" required>
            </div>
            
            <div class="form-group">
                <label for="model-execution-script">Model Execution Script</label>
                <input type="text" id="model-execution-script" name="modelExecutionScript" placeholder="app.sh">
            </div>
            
            <div class="form-group">
                <label for="model-upload">Upload Model Folder <span class="required">*</span></label>
                <input type="file" id="model-upload" webkitdirectory directory multiple required>
                <p class="help-text">Upload a folder containing model.pkl, requirements.txt, metadata.json, and inference.py</p>
            </div>
            
            <div id="uploaded-files-display" class="files-display">
                <p class="no-files">No files uploaded yet</p>
            </div>
            
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">Register External Model with Domino</button>
            </div>
        </form>
    `;
    
    // Attach event listeners
    document.getElementById('model-upload').addEventListener('change', handleFileUpload);
    document.getElementById('model-upload-form').addEventListener('submit', handleSubmit);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initializeForm);

console.log('Model upload form initialized');