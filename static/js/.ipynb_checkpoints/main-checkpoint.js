// static/js/main.js
const DOMINO_API_BASE = window.location.origin + window.location.pathname.replace(/\/$/, '');
const ORIGINAL_API_BASE = window.DOMINO?.API_BASE || '';
console.log('window.DOMINO?.API_BASE', window.DOMINO?.API_BASE);
console.log('window.location.origin', window.location.origin);
console.log('window.location.pathname', window.location.pathname);
console.log('using proxy base', DOMINO_API_BASE);
console.log('proxying to', ORIGINAL_API_BASE);
const API_KEY = window.DOMINO?.API_KEY || null;

// Global state - single source of truth
let appState = {
    bundles: null,
    policies: {},
    evidence: {},
    models: {},
    tableData: [],
    securityScans: {} // Store security scan results by experiment ID
};

// Helper function to make proxy API calls
async function proxyFetch(apiPath, options = {}) {
    // Handle query parameters properly
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

// Security scan functions
async function triggerSecurityScan(modelName, modelVersion) {
    try {
        const basePath = window.location.pathname.replace(/\/$/, '');
        const response = await fetch(`${basePath}/security-scan-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                modelName: modelName,
                version: modelVersion,
                fileRegex: ".*",
                excludeRegex: "(^|/)(node_modules|\\.git|\\.venv|venv|env|__pycache__|\\.ipynb_checkpoints)(/|$)",
                semgrepConfig: "auto",
                includeIssues: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`Security scan failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Transform the response to match your display function expectations
        const transformedResult = {
            total_issues: result.scan?.total || 0,
            high_severity: result.scan?.high || 0,
            medium_severity: result.scan?.medium || 0,
            low_severity: result.scan?.low || 0,
            issues: result.issues || [],
            timestamp: Date.now()
        };
        
        return transformedResult;
    } catch (error) {
        console.error('Security scan error:', error);
        throw error;
    }
}

function showSecurityScanSpinner(buttonElement) {
    const originalText = buttonElement.innerHTML;
    buttonElement.innerHTML = '<span class="spinner"></span> Scanning...';
    buttonElement.disabled = true;
    return originalText;
}

function hideSecurityScanSpinner(buttonElement, originalText) {
    buttonElement.innerHTML = originalText;
    buttonElement.disabled = false;
}

function displaySecurityScanResults(results, containerElement) {
    const resultsHtml = `
        <div class="security-scan-results">
            <h4>Security Scan Results</h4>
            <div class="scan-summary">
                <div class="scan-stat">
                    <span class="stat-label">Total Issues:</span>
                    <span class="stat-value ${results.total_issues > 0 ? 'stat-warning' : 'stat-success'}">
                        ${results.total_issues || 0}
                    </span>
                </div>
                <div class="scan-stat">
                    <span class="stat-label">High Severity:</span>
                    <span class="stat-value ${(results.high_severity || 0) > 0 ? 'stat-danger' : 'stat-success'}">
                        ${results.high_severity || 0}
                    </span>
                </div>
                <div class="scan-stat">
                    <span class="stat-label">Medium Severity:</span>
                    <span class="stat-value ${(results.medium_severity || 0) > 0 ? 'stat-warning' : 'stat-success'}">
                        ${results.medium_severity || 0}
                    </span>
                </div>
                <div class="scan-stat">
                    <span class="stat-label">Low Severity:</span>
                    <span class="stat-value">${results.low_severity || 0}</span>
                </div>
            </div>
            ${results.issues && results.issues.length > 0 ? `
                <div class="scan-details">
                    <h5>Issues Found:</h5>
                    <div class="issues-list">
                        ${results.issues.slice(0, 5).map(issue => `
                            <div class="issue-item severity-${issue.severity?.toLowerCase() || 'unknown'}">
                                <div class="issue-title">${issue.test_name || 'Unknown Issue'}</div>
                                <div class="issue-file">${issue.filename || 'Unknown file'}:${issue.line_number || 'N/A'}</div>
                                <div class="issue-message">${issue.issue_text || 'No description available'}</div>
                            </div>
                        `).join('')}
                        ${results.issues.length > 5 ? `
                            <div class="more-issues">
                                ... and ${results.issues.length - 5} more issues
                            </div>
                        ` : ''}
                    </div>
                </div>
            ` : '<div class="no-issues">No security issues found!</div>'}
            <div class="scan-timestamp">
                <small>Scanned: ${new Date(results.timestamp || Date.now()).toLocaleString()}</small>
            </div>
        </div>
    `;
    
    containerElement.innerHTML = resultsHtml;
}

async function handleSecurityScan(modelName, modelVersion, buttonElement) {
    const resultsContainer = buttonElement.parentElement.querySelector('.security-scan-container') || 
                           (() => {
                               const container = document.createElement('div');
                               container.className = 'security-scan-container';
                               buttonElement.parentElement.appendChild(container);
                               return container;
                           })();
    
    
    const originalText = showSecurityScanSpinner(buttonElement);
    
    try {
        const results = await triggerSecurityScan(modelName, modelVersion);
        displaySecurityScanResults(results, resultsContainer);
    } catch (error) {
        resultsContainer.innerHTML = `
            <div class="security-scan-error">
                <h4>Security Scan Failed</h4>
                <p>Error: ${error.message}</p>
                <button onclick="handleSecurityScan('${modelName}', ${modelVersion}, this.parentElement.parentElement.querySelector('.security-scan-btn'))" class="btn btn-secondary">Retry Scan</button>
            </div>
        `;
    } finally {
        hideSecurityScanSpinner(buttonElement, originalText);
    }
}


// API Functions
async function fetchAllData() {
    try {
        // 1. Fetch bundles via proxy
        const bundlesResponse = await proxyFetch('api/governance/v1/bundles');
        
        if (!bundlesResponse.ok) throw new Error(`Bundles API: ${bundlesResponse.status}`);
        const bundlesData = await bundlesResponse.json();
        
        // Filter bundles with fitch policies
        const filteredBundles = bundlesData.data?.filter(bundle => 
            bundle.state !== 'Archived' && 
            bundle.policies?.some(policy => policy.policyName?.toLowerCase().includes('[fitch'))
        ) || [];

        appState.bundles = filteredBundles;

        // 2. Collect all policy IDs
        const policyIds = new Set();
        filteredBundles.forEach(bundle => {
            bundle.policies?.forEach(policy => {
                if (policy.policyId) policyIds.add(policy.policyId);
            });
        });

        // 3. Fetch all policies in parallel via proxy
        const policyPromises = Array.from(policyIds).map(async policyId => {
            try {
                const response = await proxyFetch(`api/governance/v1/policies/${policyId}`);
                if (response.ok) {
                    appState.policies[policyId] = await response.json();
                }
            } catch (err) {
                console.error(`Policy ${policyId} failed:`, err);
            }
        });

        // 4. Fetch all evidence in parallel via proxy
        const evidencePromises = filteredBundles.map(async bundle => {
            try {
                const response = await proxyFetch(`api/governance/v1/drafts/latest?bundleId=${bundle.id}`);
                if (response.ok) {
                    appState.evidence[bundle.id] = await response.json();
                }
            } catch (err) {
                console.error(`Evidence ${bundle.id} failed:`, err);
            }
        });

        // Wait for all API calls
        await Promise.all([...policyPromises, ...evidencePromises]);

        console.log('All data fetched:', appState);
        return true;
        
    } catch (error) {
        console.error('Failed to fetch data:', error);
        return false;
    }
}

// Data Processing - enhanced to capture experiment IDs
async function processData() {
    appState.models = {};
    appState.tableData = [];

    // Process each bundle to extract model data
    for (const bundle of appState.bundles) {
        for (const attachment of bundle.attachments || []) {
            if (attachment.type === 'ModelVersion' && attachment.identifier) {
                const modelKey = `${attachment.identifier.name}_v${attachment.identifier.version}`;
                
                // Initialize model data structure
                if (!appState.models[modelKey]) {
                    appState.models[modelKey] = {
                        modelName: attachment.identifier.name,
                        dominoModelName: attachment.identifier.name,
                        modelVersion: attachment.identifier.version,
                        modelKey: modelKey,
                        bundles: [],
                        evidence: [],
                        policies: [],
                        experimentId: null, // Add experiment ID
                        systemId: null,
                        applicationId: null,
                        applicationType: null,
                        significanceRisk: null,
                        usageRisk: null,
                        complexityRisk: null,
                        userType: null,
                        outputAuthorization: null,
                        expiryDate: null,
                        securityClassification: null,
                        euAIActRisk: null,
                        modelHealth: null
                    };
                }

                const model = appState.models[modelKey];

                // Get experiment ID from the registered model
                try {
                    const modelResponse = await proxyFetch(`api/registeredmodels/v1/${attachment.identifier.name}/versions/${attachment.identifier.version}`);
                    if (modelResponse.ok) {
                        const modelDetails = await modelResponse.json();
                        model.experimentId = modelDetails.experimentRunId;
                    }
                } catch (error) {
                    console.error(`Failed to fetch experiment ID for ${modelKey}:`, error);
                }

                // Add bundle info
                model.bundles.push({
                    bundleId: bundle.id,
                    bundleName: bundle.name,
                    bundleState: bundle.state,
                    createdAt: bundle.createdAt
                });

                // Process evidence for this bundle
                const bundleEvidence = appState.evidence[bundle.id] || [];
                if (Array.isArray(bundleEvidence)) {
                    bundleEvidence.forEach(evidence => {
                        const externalId = getEvidenceExternalId(evidence, bundle.policies);

                        if (externalId === 'system-name') {
                            model.modelName = evidence.artifactContent;
                        }
                        
                        // Find system-id evidence
                        if (externalId === 'system-id') {
                            model.systemId = evidence.artifactContent;
                            model.applicationId = `v${evidence.artifactContent}.${model.modelVersion}`;
                        }
                        
                        // Find application-type evidence
                        if (externalId === 'application-type') {
                            model.applicationType = evidence.artifactContent;
                        }

                        if (externalId === 'service-level') {
                            model.serviceLevel = evidence.artifactContent;
                        }

                        if (externalId === 'significance') {
                            model.significanceRisk = evidence.artifactContent;
                        }

                        if (externalId === 'usage') {
                            model.usageRisk = evidence.artifactContent;
                        }

                        if (externalId === 'complexity') {
                            model.complexityRisk = evidence.artifactContent;
                        }

                        if (externalId === 'user-type') {
                            model.userType = evidence.artifactContent;
                        }

                        if (externalId === 'output-authorization') {
                            model.outputAuthorization = evidence.artifactContent;
                        }

                        if (externalId === 'expiry-date') {
                            model.expiryDate = evidence.artifactContent;
                        }

                        if (externalId === 'Security-Classification-wa39') {
                            model.securityClassification = evidence.artifactContent;
                        }

                        if (externalId === 'eu-ai-act-risk-level') {
                            model.euAIActRisk = evidence.artifactContent;
                        }

                        if (externalId === 'Model-Upload-and-Health-zGg') {
                            model.modelHealth = ((() => {
                              const r = evidence.artifactContent?.[0]?.["model health"];
                              if (typeof r === 'number') return r;
                              if (r && typeof r === 'object') return r.value ?? r.health ?? 0;
                              if (typeof r === 'string') {
                                const m = r.match(/-?\d+(\.\d+)?/);              // handles "health: 0.9679"
                                if (m) return parseFloat(m[0]);
                                try { const o = JSON.parse(r); return o.value ?? o.health ?? 0; } catch { return 0; }
                              }
                              return 0;
                            })() * 100).toFixed(2) + '%';
                        }
                        
                        model.evidence.push({
                            ...evidence,
                            bundleId: bundle.id,
                            bundleName: bundle.name
                        });
                    });
                }

                // Add policies
                bundle.policies?.forEach(policy => {
                    model.policies.push({
                        ...policy,
                        bundleId: bundle.id,
                        fullPolicyData: appState.policies[policy.policyId] || null
                    });
                });
            }
        }
    }

    // Create table data - include experiment ID
    appState.tableData = Object.values(appState.models).map(model => ({
        modelName: model.modelName,
        modelVersion: model.modelVersion,
        dominoModelName: model.dominoModelName,
        applicationId: model.applicationId || `n/a`,
        applicationType: model.applicationType || 'Unknown',
        serviceLevel: model.serviceLevel || 'Unknown',
        significanceRisk: model.significanceRisk || 'n/a',
        usageRisk: model.usageRisk || 'n/a',
        complexityRisk: model.complexityRisk || 'n/a',
        userType: model.userType || 'n/a',
        outputAuthorization: model.outputAuthorization || 'n/a',
        expiryDate: model.expiryDate || 'n/a',
        securityClassification: model.securityClassification || 'n/a',
        euAIActRisk: model.euAIActRisk || 'n/a',
        modelHealth: model.modelHealth || 'n/a',
        bundleName: model.bundles[0]?.bundleName || 'Unknown',
        bundleId: model.bundles[0]?.bundleId || null,
        evidenceStatus: model.evidence[0]?.evidenceId || '-',
        evidenceCreated: model.evidence[0]?.updatedAt || '-',
        owner: model.evidence[0]?.userId || 'Unknown',
        createdAt: model.bundles[0]?.createdAt,
        experimentId: model.experimentId, // Include experiment ID
        findings: [],
        dependencies: [],
    }));

    console.log('Data processed:', { models: appState.models, tableData: appState.tableData });
}

// Helper function to get evidence external ID
function getEvidenceExternalId(evidence, bundlePolicies) {
    for (const policy of bundlePolicies || []) {
        const fullPolicy = appState.policies[policy.policyId];
        if (!fullPolicy?.stages) continue;
        
        for (const stage of fullPolicy.stages) {
            const evidenceDef = stage.evidenceSet?.find(def => def.id === evidence.evidenceId);
            if (evidenceDef) return evidenceDef.externalId;
        }
    }
    return null;
}

// Enhanced rendering function with security scan button
// Enhanced rendering function with security scan button
function renderTable() {
    const tbody = document.querySelector('.table-container tbody');
    if (!tbody) return;

    if (appState.tableData.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="12" style="text-align: center; padding: 40px; color: #888;">
                    No models found
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = appState.tableData.map((model, index) => `
        <tr>
            <td>
                <div class="model-name">${model.modelName}</div>
                <div class="model-type">${model.applicationId}</div>
            </td>
            <td><span class="user-name">${model.applicationType}</span></td>
            <td><span class="status-badge status-${model.serviceLevel?.toLowerCase().replace(/\s+/g, '-')}">${model.serviceLevel}</span></td>
            <td><span class="risk-level" data-risk="${model.significanceRisk}">${model.significanceRisk}</span></td>
            <td><span class="risk-level" data-risk="${model.usageRisk}">${model.usageRisk}</span></td>
            <td><span class="risk-level" data-risk="${model.complexityRisk}">${model.complexityRisk}</span></td>
            <td><span class="user-name">${model.userType}</span></td>
            <td>
              ${Array.isArray(model.outputAuthorization)
                  ? model.outputAuthorization.map(item => `<span class="pill">${item}</span>`).join('')
                  : `<span class="user-name">${model.outputAuthorization}</span>`}
            </td>
            <td><span class="user-name">${model.expiryDate}</span></td>
            <td><span class="user-name">${model.securityClassification}</span></td>
            <td><span class="user-name">${model.euAIActRisk}</span></td>
            <td><span class="user-name">${model.modelHealth}</span></td>
            <td>
                <button class="action-btn" onclick="toggleDetails(this, ${index})">
                    <span>Details</span>
                    <span class="arrow">▼</span>
                </button>
            </td>
        </tr>
        <tr id="details-${index}" class="expandable-row">
            <td colspan="13">
                <div class="expandable-content">
                    <div class="detail-section">
                        <h3>Model Details</h3>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Registered Model Version</div>
                                <div class="detail-value">${model.modelVersion}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Created</div>
                                <div class="detail-value">${formatDate(model.createdAt)}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Bundle</div>
                                <div class="detail-value">${model.bundleName}</div>
                            </div>
                            ${model.experimentId ? `
                                <div class="detail-item">
                                    <div class="detail-label">Experiment ID</div>
                                    <div class="detail-value"><code>${model.experimentId}</code></div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                        <div class="actions-row">
                            <button class="btn btn-primary" disabled>View Live Model Monitoring</button>
                            <button class="btn btn-secondary" disabled>View Governing Bundles</button>
                            
                            ${model.modelName && model.modelVersion
                                ? `
                                    <button class="btn btn-warning security-scan-btn"
                                        onclick="handleSecurityScan('${model.dominoModelName}', '${model.modelVersion}', this)">
                                        Run Security Scan
                                    </button>
                                `
                                : '<button class="btn btn-secondary" disabled>No Model Name / Version</button>'
                            }
                        </div>
                </div>
            </td>
        </tr>
    `).join('');
}

function showLoading() {
    const tbody = document.querySelector('.table-container tbody');
    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="12" style="text-align: center; padding: 40px;">
                    <div style="color: #543FDD; font-size: 18px;">Loading models...</div>
                </td>
            </tr>
        `;
    }
}

// Utility Functions
function getInitials(name) {
    return (name || 'Unknown').split(' ').map(n => n[0]).join('').toUpperCase();
}

function formatDate(date) {
    return date ? new Date(date).toLocaleDateString() : 'Unknown';
}

// Event Handlers
function toggleDetails(button, index) {
    const row = document.getElementById(`details-${index}`);
    const arrow = button.querySelector('.arrow');
    const isCurrentlyOpen = row.classList.contains('show');
    
    // Close all other rows first
    document.querySelectorAll('.expandable-row.show').forEach(r => r.classList.remove('show'));
    document.querySelectorAll('.arrow.rotated').forEach(a => a.classList.remove('rotated'));
    document.querySelectorAll('.action-btn.expanded').forEach(b => {
        b.classList.remove('expanded');
        b.querySelector('span').textContent = 'Details';
    });
    
    // If this row wasn't open, open it
    if (!isCurrentlyOpen) {
        row.classList.add('show');
        arrow.classList.add('rotated');
        button.classList.add('expanded');
        button.querySelector('span').textContent = 'Close';
    }
}

function filterByStatus(status) {
    const rows = document.querySelectorAll('tbody tr:not(.expandable-row)');
    rows.forEach(row => {
        if (status === 'all') {
            row.style.display = '';
        } else {
            const statusCell = row.querySelector('.status-badge');
            const matches = statusCell?.textContent.toLowerCase().includes(status.toLowerCase());
            row.style.display = matches ? '' : 'none';
        }
    });
}

// Main initialization function - simple flow
async function initializeDashboard() {
    console.log('Initializing Dashboard...');
    showLoading();
    
    // 1. Fetch all data
    const success = await fetchAllData();
    
    if (success) {
        // 2. Process data once (now async to fetch experiment IDs)
        await processData();
        
        // 3. Render table
        renderTable();
        
        console.log('Dashboard ready');
    } else {
        const tbody = document.querySelector('.table-container tbody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="12" style="text-align: center; padding: 40px; color: #e74c3c;">
                        Failed to load data
                    </td>
                </tr>
            `;
        }
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', initializeDashboard);

// Tab filtering
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        this.classList.add('active');
        
        const filterValue = this.getAttribute('data-filter');
        filterByStatus(filterValue);
    });
});

// Search functionality
const searchBox = document.querySelector('.search-box');
if (searchBox) {
    searchBox.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const rows = document.querySelectorAll('tbody tr:not(.expandable-row)');
        
        rows.forEach(row => {
            const modelName = row.querySelector('.model-name')?.textContent.toLowerCase() || '';
            const ownerName = row.querySelector('.user-name')?.textContent.toLowerCase() || '';
            const matches = modelName.includes(searchTerm) || ownerName.includes(searchTerm);
            row.style.display = matches ? '' : 'none';
        });
    });
}

console.log('Dashboard initialized');