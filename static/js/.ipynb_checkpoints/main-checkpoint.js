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
    const files = Array.from(event.target.files)
        .filter(file => {
            const name = file.webkitRelativePath || file.name;
            return !name.split('/').some(part => part.startsWith('.'));
        });
    
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
    
    const filesHtml = appState.uploadedFiles.map(file => {
        const filename = file.webkitRelativePath || file.name;
        return `
            <div class="file-item" data-filename="${filename}">
                <div class="file-info">
                    <span class="file-name">${filename}</span>
                    <span class="file-size">${formatFileSize(file.size)}</span>
                    <span class="file-status-check"></span>
                </div>
            </div>
        `;
    }).join('');
    
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
    
    // Create progress container under the button
    const progressContainer = document.createElement('div');
    progressContainer.id = 'progress-container';
    progressContainer.className = 'progress-container';
    
    progressContainer.innerHTML = `
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: 0%"></div>
        </div>
        <div class="progress-message">Initializing...</div>
    `;
    
    // Insert after the form actions
    const formActions = document.querySelector('.form-actions');
    formActions.after(progressContainer);
}

// Update progress
function updateProgress(data) {
    const progressBar = document.querySelector('.progress-bar');
    const progressMessage = document.querySelector('.progress-message');
    
    if (progressBar && data.progress !== undefined) {
        progressBar.style.width = `${data.progress}%`;
    }
    
    if (progressMessage && data.message) {
        progressMessage.textContent = data.message;
    }
    
    // Update file statuses with checkmarks
    if (data.file_status) {
        Object.entries(data.file_status).forEach(([filename, status]) => {
            const fileItem = document.querySelector(`.file-item[data-filename="${filename}"]`);
            if (fileItem) {
                const statusCheck = fileItem.querySelector('.file-status-check');
                
                if (status === 'uploaded' || status === 'logged') {
                    statusCheck.innerHTML = '✓';
                    statusCheck.classList.add('checked');
                }
            }
        });
    }
}

// Hide loading state
function hideLoading(button) {
    button.disabled = false;
    button.innerHTML = 'Register External Model with Domino';
    
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) {
        setTimeout(() => {
            progressContainer.style.opacity = '0';
            setTimeout(() => progressContainer.remove(), 300);
        }, 2000);
    }
}

// Show success message
function showSuccess(result) {
    const successContainer = document.getElementById('success-message');
    
    if (!result) {
        successContainer.innerHTML = `
            <div class="success-box">
                <h3>✓ Model Registered Successfully</h3>
                <p>Your model has been registered with Domino.</p>
            </div>
        `;
        successContainer.style.display = 'block';
        return;
    }
    
    const isSuccess = result.status === 'success';
    const statusColor = isSuccess ? '#10b981' : '#ef4444';
    const statusText = result.status || 'unknown';
    
    const linksHtml = result.data ? `
        <div class="model-links">
            <h4>Quick Links:</h4>
            <div class="link-buttons-grid">
                ${result.data.security_scan_url ? `
                    <a href="${result.data.security_scan_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-shield-halved"></i>
                        <span>Security Scan</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.bundle_url ? `
                    <a href="${result.data.bundle_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-clipboard-list"></i>
                        <span>Intake Bundle</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.endpoint_url ? `
                    <a href="${result.data.endpoint_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-plug"></i>
                        <span>REST Endpoint</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.model_card_url ? `
                    <a href="${result.data.model_card_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-id-card"></i>
                        <span>Model Card</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.model_artifacts_url ? `
                    <a href="${result.data.model_artifacts_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-cube"></i>
                        <span>Model Artifacts</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.experiment_run_url ? `
                    <a href="${result.data.experiment_run_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-play-circle"></i>
                        <span>Experiment Run</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
                ${result.data.experiment_url ? `
                    <a href="${result.data.experiment_url}" target="_blank" rel="noopener noreferrer" class="link-button">
                        <i class="icon fas fa-flask"></i>
                        <span>Experiment</span>
                        <i class="external-icon fas fa-external-link-alt"></i>
                    </a>
                ` : ''}
            </div>
        </div>
    ` : '';
    
    const infoHtml = result.data ? `
        <div class="model-info">
            <h4>Registration Results:</h4>
            <div class="info-grid">
                ${result.data.model_name ? `
                    <div class="info-item">
                        <span class="info-label">Model Name:</span>
                        <span class="info-value">${result.data.model_name}</span>
                    </div>
                ` : ''}
                ${result.data.model_version !== undefined ? `
                    <div class="info-item">
                        <span class="info-label">Model Version:</span>
                        <span class="info-value">${result.data.model_version}</span>
                    </div>
                ` : ''}
                ${result.data.bundle_name ? `
                    <div class="info-item">
                        <span class="info-label">Bundle Name:</span>
                        <span class="info-value">${result.data.bundle_name}</span>
                    </div>
                ` : ''}
                ${result.data.bundle_id ? `
                    <div class="info-item">
                        <span class="info-label">Bundle ID:</span>
                        <span class="info-value">${result.data.bundle_id}</span>
                    </div>
                ` : ''}
                ${result.data.experiment_name ? `
                    <div class="info-item">
                        <span class="info-label">Experiment Name:</span>
                        <span class="info-value">${result.data.experiment_name}</span>
                    </div>
                ` : ''}
                ${result.data.run_id ? `
                    <div class="info-item">
                        <span class="info-label">Experiment Run ID:</span>
                        <span class="info-value">${result.data.run_id}</span>
                    </div>
                ` : ''}
                ${result.data.policy_name ? `
                    <div class="info-item">
                        <span class="info-label">Policy Name:</span>
                        <span class="info-value">${result.data.policy_name}</span>
                    </div>
                ` : ''}
                ${result.data.policy_id ? `
                    <div class="info-item">
                        <span class="info-label">Policy ID:</span>
                        <span class="info-value">${result.data.policy_id}</span>
                    </div>
                ` : ''}
                ${result.data.project_name ? `
                    <div class="info-item">
                        <span class="info-label">Project Name:</span>
                        <span class="info-value">${result.data.project_name}</span>
                    </div>
                ` : ''}
                ${result.data.project_id ? `
                    <div class="info-item">
                        <span class="info-label">Project ID:</span>
                        <span class="info-value">${result.data.project_id}</span>
                    </div>
                ` : ''}
            </div>
        </div>
    ` : '';
    
    successContainer.innerHTML = `
        <div class="success-box">
            <h3>✓ Model Registration Complete</h3>
            <div class="status-line">
                <span class="status-label">Status:</span>
                <span class="status-value" style="color: ${statusColor}; font-weight: bold;">${statusText}</span>
            </div>
            ${linksHtml}
            ${infoHtml}
        </div>
    `;
    successContainer.style.display = 'block';
}

// Reset form
function resetForm() {
    document.getElementById('model-upload-form').reset();
    appState.uploadedFiles = [];
    displayUploadedFiles();
    showErrors([]);
    document.getElementById('success-message').innerHTML = '';
    document.getElementById('success-message').style.display = 'none';
    
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) {
        progressContainer.remove();
    }
}

// Handle form submission with SSE progress
async function handleSubmit(event) {
    event.preventDefault();
    
    const errors = validateForm();
    if (errors.length > 0) {
        showErrors(errors);
        return;
    }
    
    showErrors([]);
    document.getElementById('success-message').innerHTML = '';
    document.getElementById('success-message').style.display = 'none';
    
    const submitButton = event.target.querySelector('button[type="submit"]');
    showLoading(submitButton);
    
    // Generate unique request ID
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Set up SSE for progress updates
    const basePath = window.location.pathname.replace(/\/$/, '');
    const eventSource = new EventSource(`${basePath}/register-progress/${requestId}`);
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateProgress(data);
        
        if (data.step === 'done') {
            eventSource.close();
        }
    };
    
    eventSource.onerror = () => {
        eventSource.close();
    };
    
    try {
        // Collect form data
        const formData = new FormData();
        
        formData.append('requestId', requestId);
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
        const response = await fetch(`${basePath}/register-external-model`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Registration failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('Registration successful:', result);
        
        showSuccess(result);
        
    } catch (error) {
        console.error('Registration error:', error);
        showErrors([`Failed to register model: ${error.message}`]);
    } finally {
        hideLoading(submitButton);
    }
}

// Initialize form
function initializeForm() {
    const container = document.querySelector('.container');
    
    container.innerHTML = `
        <h1 class="welcome-title">Register External Model</h1>
        
        <div id="error-messages"></div>
        
        <div class="form-layout">
            <form id="model-upload-form" class="model-form">
                <div class="form-columns">
                    <div class="form-column-left">
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
                    </div>
                    
                    <div class="form-column-middle">
                        <div class="form-group">
                            <label for="model-upload">Upload Model Folder <span class="required">*</span></label>
                            <input type="file" id="model-upload" webkitdirectory directory multiple required style="display: none;">
                            <button type="button" class="btn btn-upload" onclick="document.getElementById('model-upload').click()">Choose Files</button>
                            <p class="help-text">Upload a folder containing model.pkl, requirements.txt, metadata.json, and inference.py</p>
                        </div>
                        
                        <div id="uploaded-files-display" class="files-display">
                            <p class="no-files">No files uploaded yet</p>
                        </div>
                        
                        <div class="form-actions">
                            <button type="submit" class="btn btn-primary">Register External Model with Domino</button>
                            <button type="button" class="btn btn-secondary" onclick="resetForm()">Reset</button>
                        </div>
                    </div>
                </div>
            </form>
            
            <div class="results-column">
                <div id="success-message" style="display: none;"></div>
            </div>
        </div>
    `;
    
    // Attach event listeners
    document.getElementById('model-upload').addEventListener('change', handleFileUpload);
    document.getElementById('model-upload-form').addEventListener('submit', handleSubmit);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initializeForm);

console.log('Model upload form initialized');