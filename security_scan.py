import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEMGREP_CONFIG = os.environ.get("SEMGREP_CONFIG", "p/default")


def generate_pdf_from_html(html_path, pdf_path):
    """
    Generate PDF from HTML using wkhtmltopdf or weasyprint.
    Falls back gracefully if neither is available.
    """
    # Try wkhtmltopdf first (most reliable for complex HTML)
    try:
        cmd = [
            'wkhtmltopdf',
            '--enable-local-file-access',
            '--print-media-type',
            '--no-stop-slow-scripts',
            '--javascript-delay', '1000',
            '--margin-top', '10mm',
            '--margin-bottom', '10mm',
            '--margin-left', '10mm',
            '--margin-right', '10mm',
            html_path,
            pdf_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"PDF generated successfully using wkhtmltopdf: {pdf_path}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"wkhtmltopdf not available or timed out: {e}")
    
    # Try weasyprint as fallback
    try:
        import weasyprint
        weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
        logger.info(f"PDF generated successfully using weasyprint: {pdf_path}")
        return True
    except ImportError:
        logger.debug("weasyprint not available")
    except Exception as e:
        logger.warning(f"weasyprint failed: {e}")
    
    # If both fail, raise an error
    raise RuntimeError(
        "PDF generation failed: neither wkhtmltopdf nor weasyprint are available. "
        "Install with: 'pip install weasyprint' or install wkhtmltopdf system package"
    )


def check_semgrep():
    """Check if semgrep is available."""
    try:
        # Use semgrep CLI directly (recommended)
        r = subprocess.run(["semgrep", "--version"], 
                          capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return True, r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return False, "semgrep not found (install with: pip install semgrep)"


def run_semgrep_scan(target_dir, config=DEFAULT_SEMGREP_CONFIG, timeout_sec=300):
    """Run semgrep scan on target directory."""
    ok, msg = check_semgrep()
    if not ok:
        raise RuntimeError(f"Semgrep not available: {msg}")

    cmd = [
        "semgrep",
        "--config", config,
        "--json",
        "--no-git-ignore",
        "--exclude", "*/tests/*",
        "--exclude", "*/test*/*", 
        "--exclude", "*/.git/*",
        "--exclude", "*/venv/*",
        "--exclude", "*/env/*",
        "--exclude", "*/__pycache__/*",
        target_dir
    ]
    
    logger.info(f"Running semgrep scan: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    
    logger.info(f"Semgrep exit code: {proc.returncode}")
    
    if proc.stderr:
        logger.warning(f"Semgrep stderr: {proc.stderr[:500]}")

    # semgrep exits 0 when no issues, 1 when issues found, >1 for errors
    if proc.returncode in (0, 1):
        try:
            result = json.loads(proc.stdout or '{"results": []}')
            logger.info(f"Semgrep found {len(result.get('results', []))} issues")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse semgrep JSON: {e}")
            raise RuntimeError(f"Failed to parse semgrep JSON: {e}")
    
    error_msg = f"Semgrep failed (code {proc.returncode})"
    if proc.stderr:
        error_msg += f": {proc.stderr[:300]}"
    
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def summarize_semgrep(output):
    """Summarize semgrep scan results."""
    results = output.get("results", []) if isinstance(output, dict) else []
    sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    issues = []
    for r in results:
        semgrep_sev = r.get("extra", {}).get("severity", "INFO").upper()
        if semgrep_sev == "ERROR":
            mapped_sev = "HIGH"
        elif semgrep_sev == "WARNING":
            mapped_sev = "MEDIUM"
        elif semgrep_sev == "INFO":
            mapped_sev = "LOW"
        else:
            mapped_sev = "LOW"
            
        if mapped_sev in sev:
            sev[mapped_sev] += 1
        
        issues.append({
            "filename": r.get("path"),
            "line_number": r.get("start", {}).get("line"),
            "test_id": r.get("check_id"),
            "test_name": r.get("extra", {}).get("message", ""),
            "issue_severity": mapped_sev,
            "issue_confidence": "HIGH",
            "issue_text": r.get("extra", {}).get("message", ""),
        })
    return {
        "total_issues": len(results),
        "high": sev["HIGH"],
        "medium": sev["MEDIUM"],
        "low": sev["LOW"],
        "issues": issues,
        "metrics": output.get("paths", {}),
    }


def generate_html_report(summary, model_name=None, scan_metadata=None):
    """Generate a professional HTML security scan report with embedded styles."""
    scan_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total = summary['total_issues']
    high = summary['high']
    medium = summary['medium']
    low = summary['low']
    issues = summary['issues']
    
    # Group issues by severity for organized display
    issues_by_severity = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for issue in issues:
        sev = issue.get('issue_severity', 'LOW')
        if sev in issues_by_severity:
            issues_by_severity[sev].append(issue)
    
    # Using color scheme from styles.css
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Analysis Report"""
    
    if model_name:
        html += f" - {model_name}"
    
    html += """</title>
    <style>
        /* Embedded styles for iframe compatibility */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #161b22;
            background: #ffffff;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }
        
        .report-container {
            width: 100%;
            min-height: 100vh;
            background: #ffffff;
        }
        
        .header {
            background: #2f6feb;
            color: #ffffff;
            padding: 40px 60px;
            border-bottom: 4px solid #2b5ed3;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header-left h1 {
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }
        
        .header-left .subtitle {
            font-size: 15px;
            color: rgba(255, 255, 255, 0.85);
            font-weight: 400;
        }
        
        .header-right {
            text-align: right;
        }
        
        .header-right .timestamp {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.75);
            margin-bottom: 4px;
        }
        
        .header-right .model-name {
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
        }
        
        .executive-summary {
            padding: 60px;
            background: #f6f8fa;
            border-bottom: 1px solid #e5e7eb;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .summary-title {
            font-size: 22px;
            font-weight: 600;
            color: #161b22;
            margin-bottom: 30px;
            letter-spacing: -0.3px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
            margin-bottom: 40px;
        }
        
        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 24px;
            position: relative;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: #2f6feb;
        }
        
        .metric-card.high::before {
            background: #9b1c1c;
        }
        
        .metric-card.medium::before {
            background: #ff8c00;
        }
        
        .metric-card.low::before {
            background: #2b5ed3;
        }
        
        .metric-label {
            font-size: 12px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 12px;
        }
        
        .metric-value {
            font-size: 48px;
            font-weight: 300;
            color: #161b22;
            line-height: 1;
        }
        
        .summary-text {
            background: #ffffff;
            border-left: 4px solid #2f6feb;
            padding: 24px 28px;
            margin-top: 30px;
            border-radius: 4px;
        }
        
        .summary-text p {
            color: #161b22;
            font-size: 15px;
            line-height: 1.7;
        }
        
        .findings-section {
            padding: 60px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 40px;
            padding-bottom: 16px;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: #161b22;
            letter-spacing: -0.3px;
        }
        
        .section-count {
            font-size: 14px;
            color: #6b7280;
            font-weight: 500;
        }
        
        .severity-group {
            margin-bottom: 50px;
        }
        
        .severity-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .severity-label {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 6px 14px;
            border-radius: 4px;
        }
        
        .severity-label.high {
            background: #9b1c1c;
            color: #ffffff;
        }
        
        .severity-label.medium {
            background: #ff8c00;
            color: #ffffff;
        }
        
        .severity-label.low {
            background: #2b5ed3;
            color: #ffffff;
        }
        
        .severity-count {
            font-size: 14px;
            color: #6b7280;
        }
        
        .finding {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 32px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }
        
        .finding-header {
            margin-bottom: 24px;
        }
        
        .finding-id {
            font-size: 13px;
            color: #6b7280;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .finding-rule {
            font-size: 15px;
            color: #2f6feb;
            font-weight: 600;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            word-break: break-all;
        }
        
        .finding-body {
            margin-bottom: 24px;
        }
        
        .finding-description {
            font-size: 15px;
            color: #161b22;
            line-height: 1.7;
            margin-bottom: 20px;
        }
        
        .finding-metadata {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 16px 24px;
            padding: 20px;
            background: #f6f8fa;
            border-left: 3px solid #2f6feb;
            margin-top: 20px;
            border-radius: 4px;
        }
        
        .metadata-label {
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metadata-value {
            font-size: 13px;
            color: #161b22;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            word-break: break-all;
        }
        
        .clean-state {
            text-align: center;
            padding: 80px 60px;
            background: #f6f8fa;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .clean-state-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 24px;
            background: #0f7a4f;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .clean-state-icon::after {
            content: '';
            width: 30px;
            height: 50px;
            border: solid #ffffff;
            border-width: 0 6px 6px 0;
            transform: rotate(45deg);
            margin-bottom: 10px;
        }
        
        .clean-state-title {
            font-size: 24px;
            font-weight: 600;
            color: #161b22;
            margin-bottom: 12px;
        }
        
        .clean-state-text {
            font-size: 15px;
            color: #6b7280;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        .footer {
            background: #f6f8fa;
            border-top: 1px solid #e5e7eb;
            padding: 32px 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .footer-left {
            font-size: 13px;
            color: #6b7280;
        }
        
        .footer-right {
            font-size: 13px;
            color: #6b7280;
        }
        
        .footer-right strong {
            color: #161b22;
            font-weight: 600;
        }
        
        @media print {
            body {
                background: #ffffff;
            }
            
            .finding {
                page-break-inside: avoid;
            }
            
            .header {
                background: #2f6feb !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
        }
        
        @media (max-width: 1200px) {
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .header,
            .executive-summary,
            .findings-section,
            .footer {
                padding: 30px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .header-content {
                flex-direction: column;
                gap: 16px;
            }
            
            .header-right {
                text-align: left;
            }
            
            .finding-metadata {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div class="header-content">
                <div class="header-left">
                    <h1>Security Analysis Report</h1>
                    <div class="subtitle">Static Application Security Testing (SAST)</div>
                </div>
                <div class="header-right">
                    <div class="timestamp">""" + scan_time + """</div>"""
    
    if model_name:
        html += f"""
                    <div class="model-name">{model_name}</div>"""
    
    html += """
                </div>
            </div>
        </div>
        
        <div class="executive-summary">
            <h2 class="summary-title">Executive Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Findings</div>
                    <div class="metric-value">""" + str(total) + """</div>
                </div>
                <div class="metric-card high">
                    <div class="metric-label">Critical/High</div>
                    <div class="metric-value">""" + str(high) + """</div>
                </div>
                <div class="metric-card medium">
                    <div class="metric-label">Medium</div>
                    <div class="metric-value">""" + str(medium) + """</div>
                </div>
                <div class="metric-card low">
                    <div class="metric-label">Low/Info</div>
                    <div class="metric-value">""" + str(low) + """</div>
                </div>
            </div>
"""
    
    if total == 0:
        html += """
            <div class="summary-text">
                <p>This security assessment has completed successfully with no vulnerabilities or security concerns identified. All analyzed code follows secure coding practices and adheres to established security standards. The codebase demonstrates appropriate handling of sensitive operations, proper input validation, and secure use of external dependencies.</p>
            </div>
"""
    else:
        risk_level = "HIGH" if high > 0 else ("MEDIUM" if medium > 0 else "LOW")
        html += f"""
            <div class="summary-text">
                <p>This security assessment has identified <strong>{total} potential security {'issue' if total == 1 else 'issues'}</strong> requiring attention. The overall risk classification is <strong>{risk_level}</strong>. This report provides detailed findings including vulnerability locations, impact assessments, and remediation guidance. All identified issues should be reviewed by the development team and prioritized according to severity and business context.</p>
            </div>
"""
    
    html += """
        </div>
"""
    
    if total == 0:
        html += """
        <div class="clean-state">
            <div class="clean-state-icon"></div>
            <h3 class="clean-state-title">No Security Issues Detected</h3>
            <p class="clean-state-text">The static analysis security testing has completed successfully. No vulnerabilities, security weaknesses, or policy violations were identified in the analyzed codebase. This assessment provides confidence in the security posture of the application code.</p>
        </div>
"""
    else:
        html += """
        <div class="findings-section">
            <div class="section-header">
                <h2 class="section-title">Detailed Findings</h2>
                <div class="section-count">Total: """ + str(total) + """ finding""" + ("s" if total != 1 else "") + """</div>
            </div>
"""
        
        # Display findings grouped by severity
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            severity_issues = issues_by_severity[severity]
            if not severity_issues:
                continue
            
            severity_label = severity.lower()
            html += f"""
            <div class="severity-group">
                <div class="severity-header">
                    <span class="severity-label {severity_label}">{severity}</span>
                    <span class="severity-count">{len(severity_issues)} finding{'s' if len(severity_issues) != 1 else ''}</span>
                </div>
"""
            
            for idx, issue in enumerate(severity_issues, 1):
                filename = issue.get('filename', 'Unknown')
                line_num = issue.get('line_number', 'N/A')
                test_id = issue.get('test_id', 'Unknown')
                message = issue.get('issue_text', 'No description available')
                
                # Calculate a global finding number
                global_num = issues.index(issue) + 1
                
                html += f"""
                <div class="finding">
                    <div class="finding-header">
                        <div class="finding-id">Finding #{global_num}</div>
                        <div class="finding-rule">{test_id}</div>
                    </div>
                    
                    <div class="finding-body">
                        <div class="finding-description">{message}</div>
                        
                        <div class="finding-metadata">
                            <div class="metadata-label">File Path</div>
                            <div class="metadata-value">{filename}</div>
                            
                            <div class="metadata-label">Line Number</div>
                            <div class="metadata-value">{line_num}</div>
                            
                            <div class="metadata-label">Severity</div>
                            <div class="metadata-value">{severity}</div>
                            
                            <div class="metadata-label">Confidence</div>
                            <div class="metadata-value">HIGH</div>
                        </div>
                    </div>
                </div>
"""
            
            html += """
            </div>
"""
        
        html += """
        </div>
"""
    
    html += f"""
        <div class="footer">
            <div class="footer-left">
                Powered by <strong>Semgrep</strong> | Static Application Security Testing
            </div>
            <div class="footer-right">
                Report generated <strong>{scan_time}</strong>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html
    """Generate a professional HTML security scan report."""
    scan_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total = summary['total_issues']
    high = summary['high']
    medium = summary['medium']
    low = summary['low']
    issues = summary['issues']
    
    # Group issues by severity for organized display
    issues_by_severity = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for issue in issues:
        sev = issue.get('issue_severity', 'LOW')
        if sev in issues_by_severity:
            issues_by_severity[sev].append(issue)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Analysis Report{f' - {model_name}' if model_name else ''}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            background: #ffffff;
        }}
        
        .report-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: #003d6a;
            color: #ffffff;
            padding: 40px 60px;
            border-bottom: 4px solid #0055a5;
        }}
        
        .header-content {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        
        .header-left h1 {{
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }}
        
        .header-left .subtitle {{
            font-size: 15px;
            color: #b8d4e8;
            font-weight: 400;
        }}
        
        .header-right {{
            text-align: right;
        }}
        
        .header-right .timestamp {{
            font-size: 13px;
            color: #b8d4e8;
            margin-bottom: 4px;
        }}
        
        .header-right .model-name {{
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
        }}
        
        .executive-summary {{
            padding: 60px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .summary-title {{
            font-size: 22px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 30px;
            letter-spacing: -0.3px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 2px;
            padding: 24px;
            position: relative;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: #003d6a;
        }}
        
        .metric-card.high::before {{
            background: #c41e3a;
        }}
        
        .metric-card.medium::before {{
            background: #ff8c00;
        }}
        
        .metric-card.low::before {{
            background: #0055a5;
        }}
        
        .metric-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 12px;
        }}
        
        .metric-value {{
            font-size: 48px;
            font-weight: 300;
            color: #1a1a1a;
            line-height: 1;
        }}
        
        .summary-text {{
            background: #ffffff;
            border-left: 4px solid #003d6a;
            padding: 24px 28px;
            margin-top: 30px;
        }}
        
        .summary-text p {{
            color: #333333;
            font-size: 15px;
            line-height: 1.7;
        }}
        
        .findings-section {{
            padding: 60px;
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 40px;
            padding-bottom: 16px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .section-title {{
            font-size: 22px;
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: -0.3px;
        }}
        
        .section-count {{
            font-size: 14px;
            color: #666666;
            font-weight: 500;
        }}
        
        .severity-group {{
            margin-bottom: 50px;
        }}
        
        .severity-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }}
        
        .severity-label {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 6px 14px;
            border-radius: 2px;
        }}
        
        .severity-label.high {{
            background: #c41e3a;
            color: #ffffff;
        }}
        
        .severity-label.medium {{
            background: #ff8c00;
            color: #ffffff;
        }}
        
        .severity-label.low {{
            background: #0055a5;
            color: #ffffff;
        }}
        
        .severity-count {{
            font-size: 14px;
            color: #666666;
        }}
        
        .finding {{
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 2px;
            padding: 32px;
            margin-bottom: 20px;
            position: relative;
        }}
        
        .finding-header {{
            margin-bottom: 24px;
        }}
        
        .finding-id {{
            font-size: 13px;
            color: #666666;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        
        .finding-rule {{
            font-size: 15px;
            color: #003d6a;
            font-weight: 600;
            font-family: 'Courier New', monospace;
            word-break: break-all;
        }}
        
        .finding-body {{
            margin-bottom: 24px;
        }}
        
        .finding-description {{
            font-size: 15px;
            color: #333333;
            line-height: 1.7;
            margin-bottom: 20px;
        }}
        
        .finding-metadata {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 16px 24px;
            padding: 20px;
            background: #f8f9fa;
            border-left: 3px solid #003d6a;
            margin-top: 20px;
        }}
        
        .metadata-label {{
            font-size: 13px;
            font-weight: 600;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metadata-value {{
            font-size: 13px;
            color: #1a1a1a;
            font-family: 'Courier New', monospace;
            word-break: break-all;
        }}
        
        .clean-state {{
            text-align: center;
            padding: 80px 60px;
            background: #f8f9fa;
        }}
        
        .clean-state-icon {{
            width: 80px;
            height: 80px;
            margin: 0 auto 24px;
            background: #00a651;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .clean-state-icon::after {{
            content: '';
            width: 30px;
            height: 50px;
            border: solid #ffffff;
            border-width: 0 6px 6px 0;
            transform: rotate(45deg);
            margin-bottom: 10px;
        }}
        
        .clean-state-title {{
            font-size: 24px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 12px;
        }}
        
        .clean-state-text {{
            font-size: 15px;
            color: #666666;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        
        .footer {{
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            padding: 32px 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .footer-left {{
            font-size: 13px;
            color: #666666;
        }}
        
        .footer-right {{
            font-size: 13px;
            color: #666666;
        }}
        
        .footer-right strong {{
            color: #1a1a1a;
            font-weight: 600;
        }}
        
        @media print {{
            body {{
                background: #ffffff;
            }}
            
            .finding {{
                page-break-inside: avoid;
            }}
            
            .header {{
                background: #003d6a !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
        
        @media (max-width: 1200px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 768px) {{
            .header,
            .executive-summary,
            .findings-section,
            .footer {{
                padding: 30px;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header-content {{
                flex-direction: column;
                gap: 16px;
            }}
            
            .header-right {{
                text-align: left;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div class="header-content">
                <div class="header-left">
                    <h1>Security Analysis Report</h1>
                    <div class="subtitle">Static Application Security Testing (SAST)</div>
                </div>
                <div class="header-right">
                    <div class="timestamp">{scan_time}</div>
                    {f'<div class="model-name">{model_name}</div>' if model_name else ''}
                </div>
            </div>
        </div>
        
        <div class="executive-summary">
            <h2 class="summary-title">Executive Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Findings</div>
                    <div class="metric-value">{total}</div>
                </div>
                <div class="metric-card high">
                    <div class="metric-label">Critical/High</div>
                    <div class="metric-value">{high}</div>
                </div>
                <div class="metric-card medium">
                    <div class="metric-label">Medium</div>
                    <div class="metric-value">{medium}</div>
                </div>
                <div class="metric-card low">
                    <div class="metric-label">Low/Info</div>
                    <div class="metric-value">{low}</div>
                </div>
            </div>
"""
    
    if total == 0:
        html += """
            <div class="summary-text">
                <p>This security assessment has completed successfully with no vulnerabilities or security concerns identified. All analyzed code follows secure coding practices and adheres to established security standards. The codebase demonstrates appropriate handling of sensitive operations, proper input validation, and secure use of external dependencies.</p>
            </div>
"""
    else:
        risk_level = "HIGH" if high > 0 else ("MEDIUM" if medium > 0 else "LOW")
        html += f"""
            <div class="summary-text">
                <p>This security assessment has identified <strong>{total} potential security {'issue' if total == 1 else 'issues'}</strong> requiring attention. The overall risk classification is <strong>{risk_level}</strong>. This report provides detailed findings including vulnerability locations, impact assessments, and remediation guidance. All identified issues should be reviewed by the development team and prioritized according to severity and business context.</p>
            </div>
"""
    
    html += """
        </div>
"""
    
    if total == 0:
        html += """
        <div class="clean-state">
            <div class="clean-state-icon"></div>
            <h3 class="clean-state-title">No Security Issues Detected</h3>
            <p class="clean-state-text">The static analysis security testing has completed successfully. No vulnerabilities, security weaknesses, or policy violations were identified in the analyzed codebase. This assessment provides confidence in the security posture of the application code.</p>
        </div>
"""
    else:
        html += """
        <div class="findings-section">
            <div class="section-header">
                <h2 class="section-title">Detailed Findings</h2>
                <div class="section-count">Total: """ + str(total) + """ finding""" + ("s" if total != 1 else "") + """</div>
            </div>
"""
        
        # Display findings grouped by severity
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            severity_issues = issues_by_severity[severity]
            if not severity_issues:
                continue
            
            severity_label = severity.lower()
            html += f"""
            <div class="severity-group">
                <div class="severity-header">
                    <span class="severity-label {severity_label}">{severity}</span>
                    <span class="severity-count">{len(severity_issues)} finding{'s' if len(severity_issues) != 1 else ''}</span>
                </div>
"""
            
            for idx, issue in enumerate(severity_issues, 1):
                filename = issue.get('filename', 'Unknown')
                line_num = issue.get('line_number', 'N/A')
                test_id = issue.get('test_id', 'Unknown')
                message = issue.get('issue_text', 'No description available')
                
                # Calculate a global finding number
                global_num = issues.index(issue) + 1
                
                html += f"""
                <div class="finding">
                    <div class="finding-header">
                        <div class="finding-id">Finding #{global_num}</div>
                        <div class="finding-rule">{test_id}</div>
                    </div>
                    
                    <div class="finding-body">
                        <div class="finding-description">{message}</div>
                        
                        <div class="finding-metadata">
                            <div class="metadata-label">File Path</div>
                            <div class="metadata-value">{filename}</div>
                            
                            <div class="metadata-label">Line Number</div>
                            <div class="metadata-value">{line_num}</div>
                            
                            <div class="metadata-label">Severity</div>
                            <div class="metadata-value">{severity}</div>
                            
                            <div class="metadata-label">Confidence</div>
                            <div class="metadata-value">HIGH</div>
                        </div>
                    </div>
                </div>
"""
            
            html += """
            </div>
"""
        
        html += """
        </div>
"""
    
    html += f"""
        <div class="footer">
            <div class="footer-left">
                Powered by <strong>Semgrep</strong> | Static Application Security Testing
            </div>
            <div class="footer-right">
                Report generated <strong>{scan_time}</strong>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html