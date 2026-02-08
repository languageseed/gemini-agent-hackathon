"""
Security Advisor - Practical recommendations for reducing security tech debt.

Provides actionable, prioritized advice based on scan findings.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from src.models.finding import Finding, Severity, FindingCategory


class EffortLevel(str, Enum):
    """Implementation effort required."""
    QUICK_WIN = "quick_win"      # < 1 hour, no downtime
    LOW = "low"                   # 1-4 hours, minimal impact
    MEDIUM = "medium"             # 1-2 days, some planning
    HIGH = "high"                 # 1+ weeks, significant effort
    STRATEGIC = "strategic"       # Ongoing, architectural change


class ImpactLevel(str, Enum):
    """Security impact of implementing the recommendation."""
    CRITICAL = "critical"         # Prevents major breach
    HIGH = "high"                 # Significantly reduces risk
    MEDIUM = "medium"             # Meaningful improvement
    LOW = "low"                   # Minor improvement


@dataclass
class Recommendation:
    """A specific security recommendation."""
    id: str
    title: str
    description: str
    category: str
    effort: EffortLevel
    impact: ImpactLevel
    priority_score: int  # 1-100, higher = do first
    steps: list[str]
    tools_needed: list[str] = field(default_factory=list)
    related_findings: list[str] = field(default_factory=list)
    estimated_time: str = ""
    downtime_required: bool = False


@dataclass 
class SecurityAdvice:
    """Complete security advice report."""
    quick_wins: list[Recommendation]
    network_architecture: list[Recommendation]
    hardening: list[Recommendation]
    monitoring: list[Recommendation]
    strategic: list[Recommendation]
    summary: dict


class SecurityAdvisor:
    """
    Analyzes findings and generates practical security recommendations.
    
    Focus areas:
    - Quick wins (immediate, low-effort improvements)
    - Network architecture (segmentation, air-gapping, positioning)
    - Hardening (OS, Docker, services)
    - Monitoring & detection
    - Strategic improvements (long-term debt reduction)
    """
    
    def __init__(self, findings: list[Finding], context: Optional[dict] = None):
        self.findings = findings
        self.context = context or {}
        self.finding_categories = self._categorize_findings()
    
    def _categorize_findings(self) -> dict:
        """Group findings by category and severity."""
        cats = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "by_category": {},
        }
        
        for f in self.findings:
            if f.severity == Severity.CRITICAL:
                cats["critical"].append(f)
            elif f.severity == Severity.HIGH:
                cats["high"].append(f)
            elif f.severity == Severity.MEDIUM:
                cats["medium"].append(f)
            else:
                cats["low"].append(f)
            
            cat_name = f.category.value if hasattr(f.category, 'value') else str(f.category)
            if cat_name not in cats["by_category"]:
                cats["by_category"][cat_name] = []
            cats["by_category"][cat_name].append(f)
        
        return cats
    
    def generate_advice(self) -> SecurityAdvice:
        """Generate complete security advice based on findings."""
        return SecurityAdvice(
            quick_wins=self._generate_quick_wins(),
            network_architecture=self._generate_network_advice(),
            hardening=self._generate_hardening_advice(),
            monitoring=self._generate_monitoring_advice(),
            strategic=self._generate_strategic_advice(),
            summary=self._generate_summary(),
        )
    
    def _generate_quick_wins(self) -> list[Recommendation]:
        """Generate quick, high-impact recommendations."""
        recs = []
        
        # Check for world-writable files
        world_writable = [f for f in self.findings if "world-writable" in f.title.lower()]
        if world_writable:
            recs.append(Recommendation(
                id="qw-fix-permissions",
                title="Fix World-Writable System Files",
                description="Remove world-write permissions from system files to prevent unauthorized modifications.",
                category="Quick Wins",
                effort=EffortLevel.QUICK_WIN,
                impact=ImpactLevel.HIGH,
                priority_score=95,
                steps=[
                    "Run: find /etc /usr -type f -perm -0002 -ls",
                    "Review each file to confirm it shouldn't be world-writable",
                    "Fix with: chmod o-w <file>",
                    "For directories: chmod o-w,+t <dir> (sticky bit for /tmp-like dirs)",
                ],
                tools_needed=["chmod", "find"],
                related_findings=[f.id for f in world_writable],
                estimated_time="15-30 minutes",
            ))
        
        # Check for NOPASSWD sudo
        nopasswd = [f for f in self.findings if "nopasswd" in f.title.lower()]
        if nopasswd:
            recs.append(Recommendation(
                id="qw-fix-sudo",
                title="Remove NOPASSWD Sudo Rules",
                description="Require password for sudo commands to prevent lateral movement after initial compromise.",
                category="Quick Wins",
                effort=EffortLevel.QUICK_WIN,
                impact=ImpactLevel.HIGH,
                priority_score=90,
                steps=[
                    "Review /etc/sudoers and /etc/sudoers.d/*",
                    "Remove NOPASSWD from rules where possible",
                    "For automation, use dedicated service accounts with specific command allowlists",
                    "Use: visudo to safely edit sudoers",
                ],
                tools_needed=["visudo"],
                related_findings=[f.id for f in nopasswd],
                estimated_time="15 minutes",
            ))
        
        # SSH password auth
        ssh_password = [f for f in self.findings if "password authentication" in f.title.lower()]
        if ssh_password:
            recs.append(Recommendation(
                id="qw-ssh-keys",
                title="Disable SSH Password Authentication",
                description="Switch to key-based SSH authentication to eliminate brute-force attack risk.",
                category="Quick Wins",
                effort=EffortLevel.LOW,
                impact=ImpactLevel.HIGH,
                priority_score=85,
                steps=[
                    "Ensure all users have SSH keys configured",
                    "Test key-based login before disabling passwords",
                    "Edit /etc/ssh/sshd_config: PasswordAuthentication no",
                    "Reload SSH: systemctl reload sshd",
                ],
                tools_needed=["ssh-keygen", "ssh-copy-id"],
                related_findings=[f.id for f in ssh_password],
                estimated_time="30 minutes",
            ))
        
        # Docker content trust
        no_content_trust = [f for f in self.findings if "content trust" in f.title.lower()]
        if no_content_trust:
            recs.append(Recommendation(
                id="qw-docker-trust",
                title="Enable Docker Content Trust",
                description="Verify image signatures to prevent supply chain attacks via malicious images.",
                category="Quick Wins",
                effort=EffortLevel.QUICK_WIN,
                impact=ImpactLevel.MEDIUM,
                priority_score=70,
                steps=[
                    "Add to /etc/environment: DOCKER_CONTENT_TRUST=1",
                    "Or add to user's .bashrc/.zshrc",
                    "Verify: docker pull will now require signed images",
                    "Note: May break pulls of unsigned images (most community images)",
                ],
                tools_needed=[],
                related_findings=[f.id for f in no_content_trust],
                estimated_time="5 minutes",
            ))
        
        return sorted(recs, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_network_advice(self) -> list[Recommendation]:
        """Generate network architecture and segmentation advice."""
        recs = []
        
        # No firewall detected
        no_firewall = [f for f in self.findings if "firewall" in f.title.lower()]
        open_ports = [f for f in self.findings if "port open" in f.title.lower()]
        
        # Always recommend defense in depth
        recs.append(Recommendation(
            id="net-defense-depth",
            title="Implement Defense in Depth",
            description="Layer security controls so no single point of failure compromises the system.",
            category="Network Architecture",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.HIGH,
            priority_score=90,
            steps=[
                "Layer 1: Perimeter firewall (already in place if behind_network_firewall)",
                "Layer 2: Host-based firewall (ufw/nftables) - even behind perimeter",
                "Layer 3: Application-level controls (auth, rate limiting)",
                "Layer 4: Container isolation (network policies, namespaces)",
                "Document each layer and its purpose",
            ],
            tools_needed=["ufw", "nftables", "docker network"],
            related_findings=[f.id for f in no_firewall],
            estimated_time="2-4 hours",
        ))
        
        # Database/service exposure
        if open_ports:
            db_ports = [f for f in open_ports if any(db in f.title for db in 
                       ["PostgreSQL", "MySQL", "Redis", "MongoDB", "MSSQL"])]
            if db_ports:
                recs.append(Recommendation(
                    id="net-db-isolation",
                    title="Isolate Database Services",
                    description="Move databases to a dedicated network segment accessible only from application tier.",
                    category="Network Architecture",
                    effort=EffortLevel.MEDIUM,
                    impact=ImpactLevel.HIGH,
                    priority_score=85,
                    steps=[
                        "Create dedicated Docker network for databases: docker network create db-internal --internal",
                        "Bind databases to internal IPs only (127.0.0.1 or Docker network)",
                        "Use Docker network aliases for service discovery",
                        "Remove external port mappings from docker-compose",
                        "Access databases only through application containers or SSH tunnels",
                    ],
                    tools_needed=["docker network", "docker-compose"],
                    related_findings=[f.id for f in db_ports],
                    estimated_time="2-3 hours",
                ))
        
        # Micro-segmentation for containers
        recs.append(Recommendation(
            id="net-container-segmentation",
            title="Implement Container Network Segmentation",
            description="Create separate Docker networks for different trust levels and service tiers.",
            category="Network Architecture",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.MEDIUM,
            priority_score=75,
            steps=[
                "Map your services into trust zones (public, app, data, management)",
                "Create Docker networks per zone:",
                "  - docker network create frontend-net",
                "  - docker network create app-net",
                "  - docker network create db-net --internal",
                "Connect containers only to required networks",
                "Use network policies to control inter-container traffic",
            ],
            tools_needed=["docker network", "docker-compose"],
            estimated_time="4 hours",
        ))
        
        # Air-gap sensitive services
        recs.append(Recommendation(
            id="net-airgap-sensitive",
            title="Air-Gap Sensitive Services",
            description="Physically or logically isolate high-value systems from general network access.",
            category="Network Architecture", 
            effort=EffortLevel.HIGH,
            impact=ImpactLevel.CRITICAL,
            priority_score=80,
            steps=[
                "Identify crown jewels: secrets managers, PKI, backup systems, AI model stores",
                "Option A - Physical air-gap: Separate network with no routing to internet",
                "Option B - Logical air-gap: VLAN with strict ACLs, jump box only access",
                "Option C - Docker internal networks: --internal flag prevents external access",
                "Implement break-glass procedures for emergency access",
                "Log all access attempts to air-gapped systems",
            ],
            tools_needed=["VLAN configuration", "jump box", "docker network --internal"],
            estimated_time="1-2 days",
        ))
        
        # Management plane isolation
        recs.append(Recommendation(
            id="net-mgmt-isolation",
            title="Isolate Management Plane",
            description="Separate management traffic (SSH, Docker API, admin panels) from data plane.",
            category="Network Architecture",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.HIGH,
            priority_score=80,
            steps=[
                "Create dedicated management network/VLAN",
                "Bind SSH to management interface only",
                "Never expose Docker socket over TCP; use SSH tunnels",
                "Admin panels (Portainer, etc.) on management network only",
                "Use VPN or jump box for remote management access",
            ],
            tools_needed=["VLAN", "SSH tunneling", "VPN"],
            estimated_time="4-8 hours",
        ))
        
        return sorted(recs, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_hardening_advice(self) -> list[Recommendation]:
        """Generate OS and application hardening recommendations."""
        recs = []
        
        # Docker daemon hardening
        docker_findings = [f for f in self.findings if "docker" in f.title.lower()]
        if docker_findings:
            recs.append(Recommendation(
                id="hard-docker-daemon",
                title="Harden Docker Daemon Configuration",
                description="Apply security best practices to Docker daemon to limit container escape risk.",
                category="Hardening",
                effort=EffortLevel.LOW,
                impact=ImpactLevel.HIGH,
                priority_score=90,
                steps=[
                    "Create/edit /etc/docker/daemon.json:",
                    '{',
                    '  "userns-remap": "default",',
                    '  "no-new-privileges": true,',
                    '  "live-restore": true,',
                    '  "userland-proxy": false,',
                    '  "log-driver": "json-file",',
                    '  "log-opts": {"max-size": "10m", "max-file": "3"}',
                    '}',
                    "Restart Docker: systemctl restart docker",
                    "Note: userns-remap may require container rebuilds",
                ],
                tools_needed=["docker"],
                related_findings=[f.id for f in docker_findings],
                estimated_time="1 hour",
            ))
        
        # Kernel hardening
        kernel_findings = [f for f in self.findings if "kernel" in f.title.lower()]
        if kernel_findings:
            recs.append(Recommendation(
                id="hard-kernel-params",
                title="Apply Kernel Security Parameters",
                description="Harden kernel via sysctl to prevent network attacks and information leakage.",
                category="Hardening",
                effort=EffortLevel.LOW,
                impact=ImpactLevel.MEDIUM,
                priority_score=75,
                steps=[
                    "Create /etc/sysctl.d/99-security.conf:",
                    "# Network security",
                    "net.ipv4.conf.all.send_redirects = 0",
                    "net.ipv4.conf.all.accept_redirects = 0",
                    "net.ipv4.conf.all.accept_source_route = 0",
                    "net.ipv4.conf.all.log_martians = 1",
                    "net.ipv4.tcp_syncookies = 1",
                    "net.ipv4.icmp_echo_ignore_broadcasts = 1",
                    "# Memory protection",
                    "kernel.randomize_va_space = 2",
                    "",
                    "Apply: sysctl --system",
                ],
                tools_needed=["sysctl"],
                related_findings=[f.id for f in kernel_findings],
                estimated_time="30 minutes",
            ))
        
        # Remove legacy services
        legacy_services = [f for f in self.findings if any(svc in f.title.lower() for svc in 
                          ["telnet", "rsh", "rlogin", "tftp", "xinetd"])]
        if legacy_services:
            recs.append(Recommendation(
                id="hard-remove-legacy",
                title="Remove Legacy/Insecure Services",
                description="Disable or remove insecure legacy protocols that have secure alternatives.",
                category="Hardening",
                effort=EffortLevel.LOW,
                impact=ImpactLevel.HIGH,
                priority_score=95,
                steps=[
                    "Identify why legacy services are running (often forgotten)",
                    "Disable: systemctl disable --now telnet xinetd",
                    "Uninstall: apt remove telnetd xinetd rsh-server",
                    "Replace with secure alternatives:",
                    "  - Telnet → SSH",
                    "  - FTP → SFTP/SCP",
                    "  - rsh/rlogin → SSH",
                    "  - TFTP → HTTPS or SCP for file transfer",
                ],
                tools_needed=["systemctl", "apt/yum"],
                related_findings=[f.id for f in legacy_services],
                estimated_time="30 minutes",
            ))
        
        # SUID binaries
        suid_findings = [f for f in self.findings if "suid" in f.title.lower()]
        if suid_findings:
            recs.append(Recommendation(
                id="hard-suid-cleanup",
                title="Audit and Remove Unnecessary SUID Binaries",
                description="SUID binaries in user directories are privilege escalation vectors.",
                category="Hardening",
                effort=EffortLevel.LOW,
                impact=ImpactLevel.HIGH,
                priority_score=90,
                steps=[
                    "List all SUID/SGID: find / -type f \\( -perm -4000 -o -perm -2000 \\) 2>/dev/null",
                    "Review each - compare against known-good list",
                    "Remove SUID from unnecessary binaries: chmod u-s <file>",
                    "Delete SUID binaries in /tmp, /home, /var/tmp",
                    "Mount /tmp and /home with nosuid option in /etc/fstab",
                ],
                tools_needed=["find", "chmod", "mount"],
                related_findings=[f.id for f in suid_findings],
                estimated_time="1 hour",
            ))
        
        return sorted(recs, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_monitoring_advice(self) -> list[Recommendation]:
        """Generate monitoring and detection recommendations."""
        recs = []
        
        # Audit daemon
        audit_findings = [f for f in self.findings if "audit" in f.title.lower()]
        if audit_findings:
            recs.append(Recommendation(
                id="mon-auditd",
                title="Configure Comprehensive Audit Rules",
                description="Use auditd to detect and log security-relevant system events.",
                category="Monitoring",
                effort=EffortLevel.MEDIUM,
                impact=ImpactLevel.HIGH,
                priority_score=85,
                steps=[
                    "Install: apt install auditd audispd-plugins",
                    "Enable: systemctl enable --now auditd",
                    "Add essential rules to /etc/audit/rules.d/security.rules:",
                    "# Identity files",
                    "-w /etc/passwd -p wa -k identity",
                    "-w /etc/shadow -p wa -k identity", 
                    "-w /etc/sudoers -p wa -k sudo",
                    "# Docker",
                    "-w /usr/bin/docker -p x -k docker",
                    "-w /var/run/docker.sock -p rwa -k docker",
                    "# SSH",
                    "-w /etc/ssh/sshd_config -p wa -k ssh",
                    "",
                    "Load rules: augenrules --load",
                ],
                tools_needed=["auditd", "augenrules"],
                related_findings=[f.id for f in audit_findings],
                estimated_time="2 hours",
            ))
        
        # Logging
        logging_findings = [f for f in self.findings if "log" in f.title.lower() or "syslog" in f.title.lower()]
        recs.append(Recommendation(
            id="mon-centralized-logging",
            title="Implement Centralized Logging",
            description="Aggregate logs from all hosts and containers for analysis and retention.",
            category="Monitoring",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.HIGH,
            priority_score=80,
            steps=[
                "Option A - Simple: Configure rsyslog to forward to central server",
                "Option B - Docker: Use Loki + Grafana stack",
                "Option C - Full stack: ELK (Elasticsearch, Logstash, Kibana)",
                "Ensure log retention meets compliance requirements (90+ days)",
                "Set up alerts for critical events (auth failures, sudo, Docker)",
                "Protect log server - if compromised, attackers erase evidence",
            ],
            tools_needed=["rsyslog", "Loki", "Grafana"],
            related_findings=[f.id for f in logging_findings],
            estimated_time="4-8 hours",
        ))
        
        # Container monitoring
        recs.append(Recommendation(
            id="mon-container-security",
            title="Implement Container Runtime Security",
            description="Monitor container behavior for anomalies and policy violations.",
            category="Monitoring",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.HIGH,
            priority_score=75,
            steps=[
                "Option A - Lightweight: Docker events + custom alerting",
                "  docker events --filter 'type=container' --format '{{json .}}'",
                "Option B - Falco: Runtime security monitoring",
                "  Detects shell spawns in containers, sensitive file access",
                "Option C - Sysdig: Full container visibility",
                "Alert on: privileged container starts, exec into containers, mount changes",
            ],
            tools_needed=["docker events", "Falco", "Sysdig"],
            estimated_time="4 hours",
        ))
        
        return sorted(recs, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_strategic_advice(self) -> list[Recommendation]:
        """Generate long-term strategic recommendations."""
        recs = []
        
        recs.append(Recommendation(
            id="strat-zero-trust",
            title="Adopt Zero Trust Architecture",
            description="Assume breach - verify everything, trust nothing, limit blast radius.",
            category="Strategic",
            effort=EffortLevel.STRATEGIC,
            impact=ImpactLevel.CRITICAL,
            priority_score=95,
            steps=[
                "1. Inventory all assets, data flows, and access patterns",
                "2. Implement identity-based access (not network-based)",
                "3. Apply least-privilege everywhere:",
                "   - Container capabilities (drop all, add specific)",
                "   - User permissions (no shared accounts)",
                "   - Service accounts (scoped to specific resources)",
                "4. Encrypt everything in transit (mTLS between services)",
                "5. Log and monitor all access decisions",
                "6. Automate security policy enforcement",
            ],
            tools_needed=["Service mesh (Linkerd/Istio)", "Vault", "OIDC provider"],
            estimated_time="Ongoing - 3-6 month initial implementation",
        ))
        
        recs.append(Recommendation(
            id="strat-immutable-infra",
            title="Adopt Immutable Infrastructure",
            description="Replace patching with rebuilding - reduce configuration drift and attack surface.",
            category="Strategic",
            effort=EffortLevel.STRATEGIC,
            impact=ImpactLevel.HIGH,
            priority_score=80,
            steps=[
                "1. Define infrastructure as code (Terraform, Ansible)",
                "2. Build immutable container images in CI/CD",
                "3. Never SSH to production to make changes",
                "4. Deploy new versions, don't patch in place",
                "5. Use read-only root filesystems in containers",
                "6. Automate security scanning in build pipeline",
                "Benefits: Known-good state, easy rollback, reduced attack surface",
            ],
            tools_needed=["Terraform", "Ansible", "CI/CD pipeline", "Container registry"],
            estimated_time="2-3 months for full adoption",
        ))
        
        recs.append(Recommendation(
            id="strat-secrets-management",
            title="Centralize Secrets Management",
            description="Eliminate secrets in env vars, config files, and code with a secrets manager.",
            category="Strategic",
            effort=EffortLevel.HIGH,
            impact=ImpactLevel.CRITICAL,
            priority_score=90,
            steps=[
                "1. Deploy HashiCorp Vault or similar secrets manager",
                "2. Inventory all secrets (API keys, passwords, certificates)",
                "3. Migrate secrets to Vault, rotate all credentials",
                "4. Update applications to fetch secrets dynamically",
                "5. Enable audit logging on all secret access",
                "6. Implement automatic secret rotation",
                "7. Remove secrets from env vars and config files",
            ],
            tools_needed=["HashiCorp Vault", "Vault agent"],
            estimated_time="2-4 weeks",
        ))
        
        recs.append(Recommendation(
            id="strat-backup-strategy",
            title="Implement 3-2-1 Backup Strategy",
            description="Ensure recoverability from ransomware and disasters with proper backups.",
            category="Strategic",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.CRITICAL,
            priority_score=85,
            steps=[
                "3-2-1 Rule: 3 copies, 2 different media, 1 offsite",
                "1. Identify critical data: databases, configs, secrets, certificates",
                "2. Implement automated daily backups (restic, borg, or similar)",
                "3. Store backups in separate location (different cloud/physical site)",
                "4. One copy should be air-gapped or immutable (ransomware protection)",
                "5. Test restores monthly - untested backups are not backups",
                "6. Document recovery procedures and RTO/RPO targets",
            ],
            tools_needed=["restic", "borg", "rclone", "S3-compatible storage"],
            estimated_time="1-2 weeks",
        ))
        
        recs.append(Recommendation(
            id="strat-security-automation",
            title="Automate Security Scanning in CI/CD",
            description="Shift security left - find issues before they reach production.",
            category="Strategic",
            effort=EffortLevel.MEDIUM,
            impact=ImpactLevel.HIGH,
            priority_score=75,
            steps=[
                "1. Add container image scanning (Trivy) to build pipeline",
                "2. Add dependency scanning (Grype, npm audit) to builds",
                "3. Add secrets detection (Gitleaks) as pre-commit hook",
                "4. Add SAST scanning for code vulnerabilities",
                "5. Set quality gates - fail builds on critical vulnerabilities",
                "6. Generate and store SBOMs for all releases",
                "7. Automate this security auditor as post-deployment check",
            ],
            tools_needed=["Trivy", "Grype", "Gitleaks", "GitHub Actions/GitLab CI"],
            estimated_time="1-2 weeks",
        ))
        
        return sorted(recs, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_summary(self) -> dict:
        """Generate executive summary of security posture and priorities."""
        total = len(self.findings)
        critical = len(self.finding_categories["critical"])
        high = len(self.finding_categories["high"])
        
        # Calculate maturity score (simplified)
        if critical > 0:
            maturity = "Initial"
            maturity_score = 1
        elif high > 5:
            maturity = "Developing"
            maturity_score = 2
        elif high > 0:
            maturity = "Defined"
            maturity_score = 3
        elif total > 10:
            maturity = "Managed"
            maturity_score = 4
        else:
            maturity = "Optimized"
            maturity_score = 5
        
        return {
            "total_findings": total,
            "critical_findings": critical,
            "high_findings": high,
            "security_maturity": maturity,
            "maturity_score": maturity_score,
            "top_priorities": [
                "Address any critical/high severity findings immediately",
                "Implement network segmentation for databases and sensitive services",
                "Enable comprehensive logging and monitoring",
                "Remove legacy services and unnecessary attack surface",
                "Establish secrets management and backup procedures",
            ],
            "estimated_remediation_time": self._estimate_remediation_time(),
        }
    
    def _estimate_remediation_time(self) -> str:
        """Estimate total remediation time based on findings."""
        critical = len(self.finding_categories["critical"])
        high = len(self.finding_categories["high"])
        medium = len(self.finding_categories["medium"])
        
        # Rough estimates
        hours = (critical * 4) + (high * 2) + (medium * 0.5)
        
        if hours < 4:
            return "Less than half a day"
        elif hours < 8:
            return "1 day"
        elif hours < 40:
            return f"{int(hours/8)} days"
        else:
            return f"{int(hours/40)} weeks"


def generate_security_advice(findings: list[Finding], context: dict = None) -> dict:
    """Generate practical security advice from findings."""
    advisor = SecurityAdvisor(findings, context)
    advice = advisor.generate_advice()
    
    return {
        "summary": advice.summary,
        "quick_wins": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "effort": r.effort.value,
                "impact": r.impact.value,
                "priority": r.priority_score,
                "steps": r.steps,
                "time": r.estimated_time,
            }
            for r in advice.quick_wins[:5]
        ],
        "network_architecture": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "effort": r.effort.value,
                "impact": r.impact.value,
                "priority": r.priority_score,
                "steps": r.steps,
                "time": r.estimated_time,
            }
            for r in advice.network_architecture[:5]
        ],
        "hardening": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "effort": r.effort.value,
                "impact": r.impact.value,
                "steps": r.steps,
                "time": r.estimated_time,
            }
            for r in advice.hardening[:5]
        ],
        "monitoring": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "effort": r.effort.value,
                "impact": r.impact.value,
                "steps": r.steps,
                "time": r.estimated_time,
            }
            for r in advice.monitoring[:3]
        ],
        "strategic": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "effort": r.effort.value,
                "impact": r.impact.value,
                "steps": r.steps,
                "time": r.estimated_time,
            }
            for r in advice.strategic[:5]
        ],
    }
